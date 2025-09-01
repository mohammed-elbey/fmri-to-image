import torch
import h5py
import numpy as np
import os
from tqdm import tqdm
import pickle
from datetime import datetime
import json

# Import your modules
from fmri_repr import load_fmri_encoder, load_fmri_sample, load_transform

class ResumableFMRIEmbeddingPrecomputer:
    def __init__(self, 
                 data_root="/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/NSD_full_brain/",
                 vc_mask_path="/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/vc_roi.npz",
                 output_dir="/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/precomputed_embeddings/fmri/",
                 batch_size=64,
                 device="cuda",
                 checkpoint_every=100):  # Save checkpoint every N batches
        
        self.data_root = data_root
        self.vc_mask_path = vc_mask_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.device = device
        self.image_size = (256, 256)
        self.checkpoint_every = checkpoint_every
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Checkpoint file path
        self.checkpoint_path = os.path.join(output_dir, "fmri_embedding_checkpoint.pkl")
        
        print(f"Initializing fMRI embedding precomputation...")
        print(f"Data root: {data_root}")
        print(f"Output directory: {output_dir}")
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"Checkpoint every: {checkpoint_every} batches")
        
    def load_models(self):
        """Load the fMRI encoder and transform"""
        print("Loading fMRI encoder...")
        self.fmri_encoder = load_fmri_encoder().to(self.device)
        self.fmri_encoder.eval()
        
        print("Loading fMRI transform...")
        self.transform = load_transform(self.vc_mask_path, self.image_size)
        
        print("Models loaded successfully!")
        
    def get_h5_files(self, selected_subjects=["01", "02", "05", "07"]):
        """Get list of H5 files for selected subjects"""
        h5_files = []
        for f in sorted(os.listdir(self.data_root)):
            if f.endswith('.h5') and f[:2] in selected_subjects:
                h5_files.append(os.path.join(self.data_root, f))
        
        print(f"Found {len(h5_files)} H5 files: {[os.path.basename(f) for f in h5_files]}")
        return h5_files
    
    def create_sample_map(self, h5_files):
        """Create a mapping of all samples across files"""
        sample_map = []
        
        print("Creating sample map...")
        for h5_idx, h5_path in enumerate(h5_files):
            with h5py.File(h5_path, 'r') as f:
                num_samples = len(f['images'])
                print(f"  {os.path.basename(h5_path)}: {num_samples} samples")
                
                for sample_idx in range(num_samples):
                    coco_id = f['labels'][sample_idx]
                    global_idx = len(sample_map)
                    sample_map.append({
                        'h5_idx': h5_idx,
                        'h5_path': h5_path,
                        'sample_idx': sample_idx,
                        'coco_id': coco_id,
                        'global_idx': global_idx
                    })
        
        print(f"Total samples: {len(sample_map)}")
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
    
    def save_checkpoint(self, batch_idx, all_embeddings, all_metadata, sample_map):
        """Save checkpoint to disk"""
        checkpoint = {
            'last_batch_idx': batch_idx,
            'completed_samples': len(all_embeddings),
            'total_samples': len(sample_map),
            'all_embeddings': all_embeddings,
            'all_metadata': all_metadata,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'data_root': self.data_root,
                'vc_mask_path': self.vc_mask_path,
                'image_size': self.image_size,
                'batch_size': self.batch_size,
                'device': self.device
            }
        }
        
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Checkpoint saved: batch {batch_idx}, {len(all_embeddings)} samples completed")
    
    def load_fmri_batch(self, sample_map, start_idx, end_idx):
        """Load a batch of fMRI data"""
        batch_fmri = []
        batch_info = []

        for i in range(start_idx, min(end_idx, len(sample_map))):
            sample_info = sample_map[i]

            # Load fMRI data
            fmri, _ = load_fmri_sample(sample_info['h5_path'], sample_info['sample_idx'])
            fmri = self.transform(fmri)

            # fMRI comes as (256, 256) - need to add channel dimension to make it (1, 256, 256)
            # then add batch dimension to make it (1, 1, 256, 256)
            if fmri.ndim == 2:  # Shape is (H, W) = (256, 256)
                fmri = torch.FloatTensor(fmri).unsqueeze(0).unsqueeze(0)  # [1, 1, 256, 256]
            else:
                # Handle unexpected shapes
                fmri = torch.FloatTensor(fmri)
                while fmri.ndim < 4:
                    fmri = fmri.unsqueeze(0)

            batch_fmri.append(fmri)
            batch_info.append(sample_info)

        if batch_fmri:
            batch_fmri = torch.cat(batch_fmri, dim=0)  # Stack into batch: [B, 1, 256, 256]

        return batch_fmri, batch_info
    
    def precompute_all_embeddings(self):
        """Pre-compute all fMRI embeddings and save to disk with checkpointing"""
        print("\n" + "="*50)
        print("STARTING FMRI EMBEDDING PRECOMPUTATION")
        print("="*50)
        
        # Load models
        self.load_models()
        
        # Get H5 files and create sample map
        h5_files = self.get_h5_files()
        sample_map = self.create_sample_map(h5_files)
        
        # Try to load checkpoint
        checkpoint = self.load_checkpoint()
        
        if checkpoint is not None:
            # Resume from checkpoint
            all_embeddings = checkpoint['all_embeddings']
            all_metadata = checkpoint['all_metadata']
            start_batch_idx = checkpoint['last_batch_idx'] + 1
            print(f"Resuming: {len(all_embeddings)} embeddings already computed")
        else:
            # Start fresh
            all_embeddings = {}
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
                        batch_fmri, batch_info = self.load_fmri_batch(sample_map, start_idx, end_idx)
                        
                        if len(batch_fmri) == 0:
                            continue
                        
                        # Move to device
                        batch_fmri = batch_fmri.to(self.device)
                        
                        # Compute embeddings
                        embeddings = self.fmri_encoder(batch_fmri)[0]  # Take first output
                        
                        # Store embeddings and metadata
                        for i, (embedding, info) in enumerate(zip(embeddings, batch_info)):
                            global_idx = info['global_idx']
                            coco_id = info['coco_id']
                            
                            # Store embedding (move to CPU for storage)
                            all_embeddings[global_idx] = embedding.cpu().numpy()
                            
                            # Store metadata
                            all_metadata[global_idx] = {
                                'coco_id': int(coco_id),
                                'h5_file': os.path.basename(info['h5_path']),
                                'sample_idx': info['sample_idx'],
                                'embedding_shape': embedding.shape
                            }
                        
                        # Save checkpoint periodically
                        if (batch_idx + 1) % self.checkpoint_every == 0:
                            self.save_checkpoint(batch_idx, all_embeddings, all_metadata, sample_map)
                        
                        # Clear GPU cache periodically
                        if batch_idx % 10 == 0:
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}: {e}")
                        # Save checkpoint on error
                        self.save_checkpoint(batch_idx - 1, all_embeddings, all_metadata, sample_map)
                        raise e
        
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving checkpoint...")
            self.save_checkpoint(batch_idx - 1, all_embeddings, all_metadata, sample_map)
            print("Checkpoint saved. You can resume later.")
            return all_embeddings, all_metadata
        
        print(f"\nSuccessfully computed {len(all_embeddings)} fMRI embeddings!")
        
        # Save final results
        self.save_final_embeddings(all_embeddings, all_metadata)
        
        # Clean up checkpoint file
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            print("Checkpoint file removed (processing completed)")
        
        return all_embeddings, all_metadata
    
    def save_final_embeddings(self, embeddings, metadata):
        """Save final embeddings and metadata to disk"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save embeddings as compressed numpy
        embeddings_path = os.path.join(self.output_dir, f"fmri_embeddings_{timestamp}.npz")
        print(f"Saving final embeddings to: {embeddings_path}")
        
        # Convert to numpy arrays for efficient storage
        embedding_array = np.array([embeddings[i] for i in sorted(embeddings.keys())])
        indices_array = np.array(sorted(embeddings.keys()))
        
        np.savez_compressed(embeddings_path,
                           embeddings=embedding_array,
                           indices=indices_array)
        
        # Save metadata as pickle
        metadata_path = os.path.join(self.output_dir, f"fmri_metadata_{timestamp}.pkl")
        print(f"Saving final metadata to: {metadata_path}")
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save configuration
        config_path = os.path.join(self.output_dir, f"fmri_config_{timestamp}.json")
        print(f"Saving final configuration to: {config_path}")
        
        config = {
            'data_root': self.data_root,
            'vc_mask_path': self.vc_mask_path,
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'device': self.device,
            'total_samples': len(embeddings),
            'embedding_shape': list(embedding_array.shape),
            'timestamp': timestamp
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"All final files saved successfully!")
        print(f"Embedding shape: {embedding_array.shape}")
        print(f"Total size: {embedding_array.nbytes / (1024**3):.2f} GB")

def main():
    """Main function to run fMRI embedding precomputation with resumability"""
    
    # Configuration
    config = {
        'data_root': "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/NSD_full_brain/",
        'vc_mask_path': "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/vc_roi.npz",
        'output_dir': "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/precomputed_embeddings/fmri/",
        'batch_size': 64,  # Adjust based on your GPU memory
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'checkpoint_every': 50  # Save checkpoint every 50 batches
    }
    
    print("Resumable fMRI Embedding Precomputation")
    print("="*40)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("="*40)
    
    # Create precomputer
    precomputer = ResumableFMRIEmbeddingPrecomputer(**config)
    
    # Run precomputation (will resume if checkpoint exists)
    embeddings, metadata = precomputer.precompute_all_embeddings()
    
    print("\n" + "="*50)
    print("FMRI EMBEDDING PRECOMPUTATION COMPLETED!")
    print("="*50)
    print(f"Computed embeddings for {len(embeddings)} samples")
    print(f"Saved to: {config['output_dir']}")
    print("\nYou can now use these precomputed embeddings in your training script.")

if __name__ == "__main__":
    main()