import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from diffusers import DDPMScheduler, StableDiffusionPipeline
from fmri_repr import load_fmri_encoder
from semantic_pipeline import get_clip_model
from perceptual_pipeline import LowLevelManipulationNetwork
from tqdm import tqdm
import numpy as np
import os
import json
from datetime import datetime
import h5py
from PIL import Image
import pickle
import random
from scipy.stats import pearsonr

# Mixed precision training for faster training and lower memory usage
from torch.cuda.amp import autocast, GradScaler

def make_json_serializable(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Configuration class for all hyperparameters and paths
class Config:
    def __init__(self):
        self.data_root = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/NSD_full_brain/"
        self.caption_path = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/COCO_73k_annots_curated.npy"
        self.image_root = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/nsddata_stimuli/stimuli/images"
        
        # Paths to precomputed embeddings for faster data loading
        self.precomputed_fmri_dir = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/precomputed_embeddings/fmri/"
        self.precomputed_vae_dir = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/precomputed_embeddings/vae/"
        
        # Training hyperparameters
        self.image_size = (256, 256)
        self.batch_size = 64
        self.num_epochs = 100
        self.lr = 1e-4
        self.semantic_ratio = 0.5
        self.control_scale = 0.5  # Strength of fMRI influence on generation
        self.output_dir = f"/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/results/{datetime.now().strftime('%Y%m%d_%H%M')}_fixed_training"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

def load_precomputed_data(fmri_dir, vae_dir):
    """Load precomputed fMRI embeddings and VAE latents to speed up training"""
    print("Loading precomputed data...")
    
    # Find latest precomputed files
    fmri_files = [f for f in os.listdir(fmri_dir) if f.startswith('fmri_embeddings_') and f.endswith('.npz')]
    vae_files = [f for f in os.listdir(vae_dir) if f.startswith('vae_latents_') and f.endswith('.npz')]
    
    if not fmri_files or not vae_files:
        raise FileNotFoundError("Precomputed fMRI embeddings or VAE latents not found!")
    
    # Load embeddings and latents
    latest_fmri = max(fmri_files)
    latest_vae = max(vae_files)
    
    print(f"Loading fMRI embeddings from: {latest_fmri}")
    fmri_data = np.load(os.path.join(fmri_dir, latest_fmri))
    fmri_embeddings = fmri_data['embeddings']
    fmri_indices = fmri_data['indices']
    
    print(f"Loading VAE latents from: {latest_vae}")
    vae_data = np.load(os.path.join(vae_dir, latest_vae))
    vae_latents = vae_data['latents']
    vae_indices = vae_data['indices']
    
    # Load metadata if available
    fmri_metadata_files = [f for f in os.listdir(fmri_dir) if f.startswith('fmri_metadata_') and f.endswith('.pkl')]
    vae_metadata_files = [f for f in os.listdir(vae_dir) if f.startswith('vae_metadata_') and f.endswith('.pkl')]
    
    fmri_metadata = None
    vae_metadata = None
    if fmri_metadata_files and vae_metadata_files:
        with open(os.path.join(fmri_dir, max(fmri_metadata_files)), 'rb') as f:
            fmri_metadata = pickle.load(f)
        with open(os.path.join(vae_dir, max(vae_metadata_files)), 'rb') as f:
            vae_metadata = pickle.load(f)
    
    print(f"Loaded {len(fmri_embeddings)} fMRI embeddings and {len(vae_latents)} VAE latents")
    
    return {
        'fmri_embeddings': fmri_embeddings,
        'fmri_indices': fmri_indices,
        'fmri_metadata': fmri_metadata,
        'vae_latents': vae_latents,
        'vae_indices': vae_indices,
        'vae_metadata': vae_metadata
    }

class PrecomputedNSDDataset(Dataset):
    """Dataset that uses precomputed embeddings for faster training"""
    def __init__(self, precomputed_data, h5_paths, caption_data, semantic_targets=None):
        self.precomputed_data = precomputed_data
        self.h5_paths = h5_paths
        self.caption_data = caption_data
        self.semantic_targets = semantic_targets
        
        # Create mapping from global indices to precomputed data
        self.fmri_map = {idx: i for i, idx in enumerate(precomputed_data['fmri_indices'])}
        self.vae_map = {idx: i for i, idx in enumerate(precomputed_data['vae_indices'])}
        
        # Build sample map for samples with both fMRI and VAE data
        self.sample_map = []
        for h5_idx, h5_path in enumerate(h5_paths):
            with h5py.File(h5_path, 'r') as f:
                num_samples = len(f['images'])
                for sample_idx in range(num_samples):
                    coco_id = f['labels'][sample_idx]
                    coco_str = f"image_{coco_id:06d}"
                    global_idx = len(self.sample_map)
                    
                    # Only include samples with both fMRI and VAE data
                    if global_idx in self.fmri_map and global_idx in self.vae_map:
                        self.sample_map.append({
                            'h5_idx': h5_idx,
                            'sample_idx': sample_idx,
                            'coco_id': coco_id,
                            'coco_str': coco_str,
                            'global_idx': global_idx
                        })
        
        print(f"Dataset created with {len(self.sample_map)} samples")
        
    def __len__(self):
        return len(self.sample_map)

    def __getitem__(self, idx):
        sample_info = self.sample_map[idx]
        global_idx = sample_info['global_idx']
        coco_id = sample_info['coco_id']
        
        # Get precomputed embeddings
        fmri_idx = self.fmri_map[global_idx]
        fmri_embedding = torch.FloatTensor(self.precomputed_data['fmri_embeddings'][fmri_idx])
        
        vae_idx = self.vae_map[global_idx]
        vae_latent = torch.FloatTensor(self.precomputed_data['vae_latents'][vae_idx])

        # Load image for reference only
        img_path_jpg = os.path.join(cfg.image_root, f"{sample_info['coco_str']}.jpg")
        img_path_png = os.path.join(cfg.image_root, f"{sample_info['coco_str']}.png")

        if os.path.exists(img_path_jpg):
            image = Image.open(img_path_jpg).convert("RGB")
        elif os.path.exists(img_path_png):
            image = Image.open(img_path_png).convert("RGB")
        else:
            image = Image.new('RGB', cfg.image_size, color=(0, 0, 0))

        image = image.resize(cfg.image_size)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.FloatTensor(image).permute(2,0,1)

        # Get semantic targets
        semantic_target = self.semantic_targets[coco_id] if self.semantic_targets is not None else None

        # Get captions
        captions = self.caption_data.get(str(coco_id), [""] * 5)

        return {
            "fmri_embedding": fmri_embedding,
            "vae_latent": vae_latent,
            "image": image,
            "semantic_target": semantic_target,
            "captions": captions,
            "coco_id": coco_id,
            "global_idx": global_idx
        }

def precomputed_collate_fn(batch):
    """Collate function for batch processing"""
    return {
        "fmri_embedding": torch.stack([item["fmri_embedding"] for item in batch]),
        "vae_latent": torch.stack([item["vae_latent"] for item in batch]),
        "image": torch.stack([item["image"] for item in batch]),
        "semantic_target": torch.stack([item["semantic_target"] for item in batch]) if batch[0]["semantic_target"] is not None else None,
        "captions": [item["captions"] for item in batch],
        "coco_id": [item["coco_id"] for item in batch],
        "global_idx": [item["global_idx"] for item in batch]
    }

def reconstruct_with_proper_timesteps(pred_noise, timesteps, noisy_latents, noise_scheduler):
    """Reconstruct latents using proper timestep handling for each sample in batch"""
    batch_size = pred_noise.shape[0]
    recon_latents = []
    
    for i in range(batch_size):
        single_pred = pred_noise[i:i+1]
        single_noisy = noisy_latents[i:i+1]
        single_timestep = timesteps[i]
        
        # Reconstruct using appropriate timestep
        single_recon = noise_scheduler.step(
            single_pred, single_timestep, single_noisy
        ).pred_original_sample
        
        recon_latents.append(single_recon)
    
    return torch.cat(recon_latents, dim=0)

def setup_partial_unet_unfreezing(llmn):
    """Unfreeze only the last few decoder layers of U-Net for fine-tuning"""
    print("Setting up partial U-Net unfreezing...")
    
    # Freeze all parameters first
    for param in llmn.sd_unet.parameters():
        param.requires_grad = False
    
    # Unfreeze last 2 decoder blocks for adaptation
    num_up_blocks = len(llmn.sd_unet.up_blocks)
    layers_to_unfreeze = 2
    
    unfrozen_params = 0
    total_params = 0
    
    for i, up_block in enumerate(llmn.sd_unet.up_blocks):
        if i >= num_up_blocks - layers_to_unfreeze:
            for param in up_block.parameters():
                param.requires_grad = True
                unfrozen_params += param.numel()
        
        total_params += sum(p.numel() for p in up_block.parameters())
    
    # Unfreeze final output layer
    if hasattr(llmn.sd_unet, 'conv_out'):
        for param in llmn.sd_unet.conv_out.parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
        total_params += sum(p.numel() for p in llmn.sd_unet.conv_out.parameters())
    
    print(f"Unfrozen {unfrozen_params:,} parameters out of {total_params:,} U-Net parameters ({100*unfrozen_params/total_params:.1f}%)")
    return unfrozen_params

def main():
    global cfg
    cfg = Config()

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load precomputed data for faster training
    precomputed_data = load_precomputed_data(cfg.precomputed_fmri_dir, cfg.precomputed_vae_dir)

    # Load captions
    caption_array = np.load(cfg.caption_path, allow_pickle=True)
    
    MAX_CAPTIONS = 5
    caption_data = {}
    for i, row in enumerate(caption_array):
        captions = [cap for cap in row if cap.strip()]
        if len(captions) < MAX_CAPTIONS:
            captions += [""] * (MAX_CAPTIONS - len(captions))
        else:
            captions = captions[:MAX_CAPTIONS]
        caption_data[str(i)] = captions

    # Build dataset with selected subjects
    selected_subjects = ["01", "02", "05", "07"]
    h5_paths = [
        os.path.join(cfg.data_root, f)
        for f in sorted(os.listdir(cfg.data_root))
        if f.endswith('.h5') and f[:2] in selected_subjects
    ]
    
    # Load precomputed semantic targets
    print("Loading precomputed semantic targets...")
    with open("/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/clip_embeddings/semantic_targets.pkl", "rb") as f:
        semantic_targets = pickle.load(f)
    
    # Create dataset
    dataset = PrecomputedNSDDataset(precomputed_data, h5_paths, caption_data, semantic_targets)
    
    # Train/test split
    all_indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(all_indices)
    train_indices = all_indices[:36000]

    train_dataset = Subset(dataset, train_indices)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        collate_fn=precomputed_collate_fn,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # Load models
    print("Loading models...")
    fmri_encoder = load_fmri_encoder().to(cfg.device)
    clip_model, clip_preprocess, clip_tokenizer = get_clip_model(device=cfg.device)
    llmn = LowLevelManipulationNetwork(device=cfg.device)
    sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to(cfg.device)
    
    # Setup partial U-Net unfreezing
    unfrozen_params = setup_partial_unet_unfreezing(llmn)

    # Initialize noise scheduler for diffusion process
    noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder='scheduler')

    # Create optimizer for all trainable parameters
    trainable_params = list(llmn.parameters()) + [p for p in llmn.sd_unet.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.lr)
    
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Initialize mixed precision training
    scaler = GradScaler()

    # Metrics tracking
    all_metrics_history = {"train_loss": []}
    
    # Check for existing checkpoint
    start_epoch = 0
    checkpoint_path = os.path.join(cfg.output_dir, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
        
        llmn.load_state_dict(checkpoint['llmn'])
        if 'fmri_encoder' in checkpoint:
            fmri_encoder.load_state_dict(checkpoint['fmri_encoder'])
        if 'sd_unet' in checkpoint:
            llmn.sd_unet.load_state_dict(checkpoint['sd_unet'])
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        all_metrics_history = checkpoint.get('metrics_history', all_metrics_history)
        
        # Re-setup partial unfreezing after loading
        setup_partial_unet_unfreezing(llmn)
        
        print(f"Resuming training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, cfg.num_epochs):
        print(f"Epoch {epoch}")

        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            # Move data to device
            fmri_embedding = batch["fmri_embedding"].to(cfg.device, non_blocking=True)
            true_latents = batch["vae_latent"].to(cfg.device, non_blocking=True)
            images = batch["image"].to(cfg.device, non_blocking=True)

            # Clean captions
            captions = []
            valid_indices = []
            for i, caps in enumerate(batch["captions"]):
                clean_caps = [cap for cap in caps if cap.strip()]
                if clean_caps:
                    captions.append(clean_caps)
                    valid_indices.append(i)
            
            # Filter by valid indices
            fmri_embedding = fmri_embedding[valid_indices]
            true_latents = true_latents[valid_indices]
            images = images[valid_indices]

            if len(valid_indices) == 0:
                continue

            # Mixed precision forward pass
            with autocast():
                # Get semantic targets
                if batch["semantic_target"] is not None:
                    semantic_target = batch["semantic_target"][valid_indices].to(cfg.device, non_blocking=True)
                else:
                    semantic_target = torch.stack([
                        semantic_targets[int(batch["coco_id"][i])] 
                        for i in valid_indices
                    ]).to(cfg.device, non_blocking=True)

                # Add noise to latents for diffusion training
                noise = torch.randn_like(true_latents)
                timesteps = torch.randint(0, 1000, (len(fmri_embedding),), device=cfg.device)
                noisy_latents = noise_scheduler.add_noise(true_latents, noise, timesteps)

                # Process fMRI embeddings through LLMN
                if fmri_embedding.dim() == 2:
                    batch_size = fmri_embedding.shape[0]
                    fmri_spatial = fmri_embedding.view(batch_size, 1, 256, 256)
                else:
                    fmri_spatial = fmri_embedding

                fmri_features = llmn.transform_fmri(fmri_spatial)
                pred_noise = llmn.manipulate_unet(noisy_latents, fmri_features, timesteps, semantic_target, cfg.control_scale)

                # Compute combined loss
                loss_diff = nn.MSELoss()(pred_noise, noise)  # Diffusion loss
                projected_emb = llmn.project_fmri_to_semantic(fmri_features)
                loss_sem = nn.CosineEmbeddingLoss()(projected_emb, semantic_target, torch.ones(len(fmri_embedding), device=cfg.device))
                loss = loss_diff + loss_sem

            # Mixed precision backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "llmn": llmn.state_dict(),
            "fmri_encoder": fmri_encoder.state_dict(),
            "sd_unet": llmn.sd_unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": vars(cfg),
            "metrics_history": all_metrics_history
        }
        
        torch.save(checkpoint, os.path.join(cfg.output_dir, "latest_checkpoint.pth"))
        if epoch % 10 == 0:
            torch.save(checkpoint, os.path.join(cfg.output_dir, f"checkpoint_epoch_{epoch}.pth"))

        # Track metrics
        avg_loss = float(np.mean(epoch_losses))
        all_metrics_history["train_loss"].append(avg_loss)

        # Save metrics
        with open(os.path.join(cfg.output_dir, "training_metrics.json"), "w") as f:
            json.dump(make_json_serializable(all_metrics_history), f, indent=2)
        

    # Save final checkpoint
    final_checkpoint = {
        "epoch": cfg.num_epochs - 1,
        "llmn": llmn.state_dict(),
        "fmri_encoder": fmri_encoder.state_dict(),
        "sd_unet": llmn.sd_unet.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": vars(cfg),
        "metrics_history": all_metrics_history,
        "training_completed": True
    }
    
    torch.save(final_checkpoint, os.path.join(cfg.output_dir, "final_checkpoint.pth"))
    
    print("Training completed!")
    print(f"Final checkpoint saved to: {os.path.join(cfg.output_dir, 'final_checkpoint.pth')}")
    print(f"Training metrics saved to: {os.path.join(cfg.output_dir, 'training_metrics.json')}")

if __name__ == "__main__":
    main()