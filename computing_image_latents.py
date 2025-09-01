import torch
import h5py
import numpy as np
import os
from tqdm import tqdm
import pickle
from datetime import datetime
import json
from PIL import Image
from diffusers import AutoencoderKL

class ResumableVAELatentPrecomputer:
    def __init__(self, 
                 data_root="/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/NSD_full_brain/",
                 image_root="/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/nsddata_stimuli/stimuli/images",
                 output_dir="/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/precomputed_embeddings/vae/",
                 vae_model_path="stabilityai/sd-vae-ft-mse",
                 batch_size=64,  # Smaller batch size for VAE operations
                 image_size=(256, 256),
                 device="cuda",
                 checkpoint_every=100):  # Save checkpoint every N batches
        
        self.data_root = data_root
        self.image_root = image_root
        self.output_dir = output_dir
        self.vae_model_path = vae_model_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        self.checkpoint_every = checkpoint_every
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Checkpoint file path
        self.checkpoint_path = os.path.join(output_dir, "vae_latent_checkpoint.pkl")
        
        print(f"Initializing VAE latent precomputation...")
        print(f"Data root: {data_root}")
        print(f"Image root: {image_root}")
        print(f"Output directory: {output_dir}")
        print(f"VAE model: {vae_model_path}")
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Image size: {image_size}")
        print(f"Checkpoint every: {checkpoint_every} batches")
        
    def load_vae(self):
        """Load the VAE model"""
        print("Loading VAE model...")
        self.vae = AutoencoderKL.from_pretrained(self.vae_model_path).to(self.device)
        self.vae.eval()
        print("VAE loaded successfully!")
        
    def get_h5_files(self, selected_subjects=["01", "02", "05", "07"]):
        """Get list of H5 files for selected subjects"""
        h5_files = []
        for f in sorted(os.listdir(self.data_root)):
            if f.endswith('.h5') and f[:2] in selected_subjects:
                h5_files.append(os.path.join(self.data_root, f))
        
        print(f"Found {len(h5_files)} H5 files: {[os.path.basename(f) for f in h5_files]}")
        return h5_files
    
    def create_sample_map(self, h5_files):
        """Create a mapping of all samples with their COCO IDs"""
        sample_map = []
        missing_images = []
        
        print("Creating sample map and checking image availability...")
        for h5_idx, h5_path in enumerate(h5_files):
            with h5py.File(h5_path, 'r') as f:
                num_samples = len(f['images'])
                print(f"  {os.path.basename(h5_path)}: {num_samples} samples")
                
                for sample_idx in range(num_samples):
                    coco_id = f['labels'][sample_idx]
                    coco_str = f"image_{coco_id:06d}"
                    
                    # Check if image exists
                    img_path_jpg = os.path.join(self.image_root, f"{coco_str}.jpg")
                    img_path_png = os.path.join(self.image_root, f"{coco_str}.png")
                    
                    if os.path.exists(img_path_jpg):
                        image_path = img_path_jpg
                    elif os.path.exists(img_path_png):
                        image_path = img_path_png
                    else:
                        missing_images.append(coco_str)
                        continue  # Skip missing images
                    
                    global_idx = len(sample_map)
                    sample_map.append({
                        'h5_idx': h5_idx,
                        'h5_path': h5_path,
                        'sample_idx': sample_idx,
                        'coco_id': coco_id,
                        'coco_str': coco_str,
                        'image_path': image_path,
                        'global_idx': global_idx
                    })
        
        if missing_images:
            print(f"Warning: {len(missing_images)} images not found and will be skipped")
            if len(missing_images) <= 10:  # Show first few missing images
                print(f"Missing images: {missing_images}")
        
        print(f"Total valid samples: {len(sample_map)}")
        return sample_map
    
    def load_checkpoint(self):
        """Load existing checkpoint if available"""
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from: {self.checkpoint_path}")
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"Resuming from batch {checkpoint['last_batch_idx']} ({checkpoint['completed_samples']} samples completed)")
            return checkpoint
        else:
            print("No checkpoint found. Starting from beginning.")
            return None
    
    def save_checkpoint(self, batch_idx, all_latents, all_metadata, sample_map):
        """Save checkpoint to disk"""
        checkpoint = {
            'last_batch_idx': batch_idx,
            'completed_samples': len(all_latents),
            'total_samples': len(sample_map),
            'all_latents': all_latents,
            'all_metadata': all_metadata,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'data_root': self.data_root,
                'image_root': self.image_root,
                'vae_model_path': self.vae_model_path,
                'image_size': self.image_size,
                'batch_size': self.batch_size,
                'device': self.device
            }
        }
        
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Checkpoint saved: batch {batch_idx}, {len(all_latents)} samples completed")
    
    def load_image_batch(self, sample_map, start_idx, end_idx):
        """Load a batch of images"""
        batch_images = []
        batch_info = []
        
        for i in range(start_idx, min(end_idx, len(sample_map))):
            sample_info = sample_map[i]
            
            try:
                # Load and process image
                image = Image.open(sample_info['image_path']).convert("RGB")
                image = image.resize(self.image_size)
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.FloatTensor(image).permute(2, 0, 1)  # CHW format
                
                batch_images.append(image)
                batch_info.append(sample_info)
                
            except Exception as e:
                print(f"Error loading image {sample_info['image_path']}: {e}")
                continue
        
        if batch_images:
            batch_images = torch.stack(batch_images, dim=0)  # Stack into batch
        
        return batch_images, batch_info
    
    def encode_image_batch(self, image_batch):
        """Encode a batch of images to latent space"""
        with torch.no_grad():
            latents = self.vae.encode(image_batch).latent_dist.sample()
            latents = latents * 0.18215  # Scaling factor for SD VAE
        return latents
    
    def precompute_all_latents(self):
        """Pre-compute all VAE latents and save to disk with checkpointing"""
        print("\n" + "="*50)
        print("STARTING VAE LATENT PRECOMPUTATION")
        print("="*50)
        
        # Load VAE
        self.load_vae()
        
        # Get H5 files and create sample map
        h5_files = self.get_h5_files()
        sample_map = self.create_sample_map(h5_files)
        
        # Try to load checkpoint
        checkpoint = self.load_checkpoint()
        
        if checkpoint is not None:
            # Resume from checkpoint
            all_latents = checkpoint['all_latents']
            all_metadata = checkpoint['all_metadata']
            start_batch_idx = checkpoint['last_batch_idx'] + 1
            print(f"Resuming: {len(all_latents)} latents already computed")
        else:
            # Start fresh
            all_latents = {}
            all_metadata = {}
            start_batch_idx = 0
        
        # Process in batches
        total_samples = len(sample_map)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        print(f"\nProcessing {total_samples} samples in {num_batches} batches...")
        print(f"Starting from batch {start_batch_idx}")
        
        try:
            with torch.no_grad():
                for batch_idx in tqdm(range(start_batch_idx, num_batches), 
                                     desc="Processing batches", 
                                     initial=start_batch_idx, 
                                     total=num_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = start_idx + self.batch_size
                    
                    try:
                        # Load batch
                        batch_images, batch_info = self.load_image_batch(sample_map, start_idx, end_idx)
                        
                        if len(batch_images) == 0:
                            continue
                        
                        # Move to device
                        batch_images = batch_images.to(self.device)
                        
                        # Encode to latents
                        latents = self.encode_image_batch(batch_images)
                        
                        # Store latents and metadata
                        for i, (latent, info) in enumerate(zip(latents, batch_info)):
                            global_idx = info['global_idx']
                            coco_id = info['coco_id']
                            
                            # Store latent (move to CPU for storage)
                            all_latents[global_idx] = latent.cpu().numpy()
                            
                            # Store metadata
                            all_metadata[global_idx] = {
                                'coco_id': int(coco_id),
                                'coco_str': info['coco_str'],
                                'h5_file': os.path.basename(info['h5_path']),
                                'sample_idx': info['sample_idx'],
                                'image_path': info['image_path'],
                                'latent_shape': latent.shape
                            }
                        
                        # Save checkpoint periodically
                        if (batch_idx + 1) % self.checkpoint_every == 0:
                            self.save_checkpoint(batch_idx, all_latents, all_metadata, sample_map)
                        
                        # Clear GPU cache periodically
                        if batch_idx % 10 == 0:
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}: {e}")
                        # Save checkpoint on error
                        self.save_checkpoint(batch_idx - 1, all_latents, all_metadata, sample_map)
                        raise e
        
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving checkpoint...")
            self.save_checkpoint(batch_idx - 1, all_latents, all_metadata, sample_map)
            print("Checkpoint saved. You can resume later.")
            return all_latents, all_metadata
        
        print(f"\nSuccessfully computed {len(all_latents)} VAE latents!")
        
        # Save final results
        self.save_final_latents(all_latents, all_metadata)
        
        # Clean up checkpoint file
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            print("Checkpoint file removed (processing completed)")
        
        return all_latents, all_metadata
    
    def save_final_latents(self, latents, metadata):
        """Save final latents and metadata to disk"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save latents as compressed numpy
        latents_path = os.path.join(self.output_dir, f"vae_latents_{timestamp}.npz")
        print(f"Saving final latents to: {latents_path}")
        
        # Convert to numpy arrays for efficient storage
        latents_array = np.array([latents[i] for i in sorted(latents.keys())])
        indices_array = np.array(sorted(latents.keys()))
        
        np.savez_compressed(latents_path,
                           latents=latents_array,
                           indices=indices_array)
        
        # Save metadata as pickle
        metadata_path = os.path.join(self.output_dir, f"vae_metadata_{timestamp}.pkl")
        print(f"Saving final metadata to: {metadata_path}")
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save configuration
        config_path = os.path.join(self.output_dir, f"vae_config_{timestamp}.json")
        print(f"Saving final configuration to: {config_path}")
        
        config = {
            'data_root': self.data_root,
            'image_root': self.image_root,
            'vae_model_path': self.vae_model_path,
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'device': self.device,
            'total_samples': len(latents),
            'latent_shape': list(latents_array.shape),
            'timestamp': timestamp
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"All final files saved successfully!")
        print(f"Latent shape: {latents_array.shape}")
        print(f"Total size: {latents_array.nbytes / (1024**3):.2f} GB")

    def verify_latents(self, timestamp=None):
        """Verify the saved latents by loading and checking a few samples"""
        if timestamp is None:
            # Find the most recent file
            files = [f for f in os.listdir(self.output_dir) if f.startswith('vae_latents_')]
            if not files:
                print("No latent files found to verify!")
                return
            timestamp = max(files).split('_')[2].split('.')[0]
        
        print(f"\nVerifying latents with timestamp: {timestamp}")
        
        # Load files
        latents_path = os.path.join(self.output_dir, f"vae_latents_{timestamp}.npz")
        metadata_path = os.path.join(self.output_dir, f"vae_metadata_{timestamp}.pkl")
        
        if not os.path.exists(latents_path) or not os.path.exists(metadata_path):
            print("Latent files not found!")
            return
        
        # Load data
        latent_data = np.load(latents_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        latents = latent_data['latents']
        indices = latent_data['indices']
        
        print(f"Loaded {len(latents)} latents")
        print(f"Latent shape: {latents.shape}")
        print(f"Data type: {latents.dtype}")
        print(f"Memory usage: {latents.nbytes / (1024**3):.2f} GB")
        
        # Check a few samples
        print("\nSample verification:")
        for i in range(min(5, len(indices))):
            idx = indices[i]
            latent = latents[i]
            meta = metadata[idx]
            print(f"  Index {idx}: COCO ID {meta['coco_id']}, shape {latent.shape}, "
                  f"min/max: {latent.min():.3f}/{latent.max():.3f}")

def main():
    """Main function to run VAE latent precomputation with resumability"""
    
    # Configuration
    config = {
        'data_root': "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/NSD_full_brain/",
        'image_root': "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/nsddata_stimuli/stimuli/images",
        'output_dir': "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/precomputed_embeddings/vae/",
        'vae_model_path': "stabilityai/sd-vae-ft-mse",
        'batch_size': 64,  # Adjust based on your GPU memory
        'image_size': (256, 256),
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'checkpoint_every': 50  # Save checkpoint every 50 batches
    }
    
    print("Resumable VAE Latent Precomputation")
    print("="*40)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("="*40)
    
    # Create precomputer
    precomputer = ResumableVAELatentPrecomputer(**config)
    
    # Run precomputation (will resume if checkpoint exists)
    latents, metadata = precomputer.precompute_all_latents()
    
    # Verify results
    precomputer.verify_latents()
    
    print("\n" + "="*50)
    print("VAE LATENT PRECOMPUTATION COMPLETED!")
    print("="*50)
    print(f"Computed latents for {len(latents)} samples")
    print(f"Saved to: {config['output_dir']}")
    print("\nYou can now use these precomputed latents in your training script.")

if __name__ == "__main__":
    main()