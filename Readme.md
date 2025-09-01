# fMRI-to-Image Neural Decoding

## Files Overview

### Core Training
- **`train.py`** - Main training script with mixed precision and checkpointing. Uses precomputed embeddings for faster training.

### Data Preprocessing
- **`computing_fmri_embs.py`** - Precomputes fMRI embeddings using the Neural_fMRI2fMRI encoder with resumable checkpointing.
- **`computing_image_latents.py`** - Precomputes VAE latents from images using Stable Diffusion's VAE encoder.
- **`CLIPemb.py`** - Precomputes CLIP embeddings for images and text captions, creates semantic targets.

### Model Components
- **`fmri_repr.py`** - Loads fMRI encoder model and handles fMRI data transformation using visual cortex masks.
- **`semantic_pipeline.py`** - CLIP model utilities for encoding text and images.
- **`perceptual_pipeline.py`** - Low-level manipulation network that injects fMRI features into Stable Diffusion's U-Net.

## Data Sources (from neuropictor HuggingFace repo)

### Required Files
You can find them in the neuropictor huggingface : [text](https://huggingface.co/Fudan-fMRI/neuropictor/tree/main)
- **NSD fMRI data**: `.h5` files containing brain activity data for subjects 01, 02, 05, 07
- **Visual cortex mask**: `vc_roi.npz` - defines visual cortex regions of interest
- **Image stimuli**: COCO images in `/nsddata_stimuli/stimuli/images/` (it is in the form of a zip file inside the neuropictor/NSD/nsddata_stimuli.zip)
- **Captions**: `COCO_73k_annots_curated.npy` - curated COCO captions
- **fMRI encoder checkpoint**: `checkpoint_120000.pth` - pretrained fMRI autoencoder
- **fMRI encoder config**: `fMRI_AutoEncoder.yaml` - model configuration

### Models Used
- **Stable Diffusion 2.1**: For image generation and VAE encoding
- **CLIP ViT-B/32**: For semantic embeddings
- **Neural_fMRI2fMRI**: Pretrained fMRI encoder from neuropictor

## Usage

1. Run preprocessing scripts to generate embeddings:
   
   python computing_fmri_embs.py
   python computing_image_latents.py  
   python CLIPemb.py

2. Run the training:

    python train.py

# fMRI Visual Grounding Pipeline

## Files Overview

### Visual Grounding Generation
- **`grounding.py`** - Uses GroundingDINO to generate spatial grounding annotations for image-caption pairs. Processes COCO images with their captions to create bounding box annotations.

### Data Processing
- **`preprocess_grounding.py`** - Filters and preprocesses grounding data by confidence and bounding box size. Converts coordinates to DETR format (normalized center coordinates).

### Model Training
- **`bbox_MLP.py`** - Complete training pipeline for fMRI-to-bounding-box prediction using an MLP architecture with Hungarian matching loss. Includes data loading, model definition, and evaluation metrics.

## Data Sources (from neuropictor HuggingFace repo)

### Required Files
- **fMRI embeddings**: `fmri_embeddings_XXXXX.npz` - precomputed fMRI representations
- **fMRI metadata**: `fmri_metadata_XXXXX.pkl` - mapping between fMRI data and COCO IDs
- **Image stimuli**: COCO images in `/nsddata_stimuli/stimuli/images/`
- **Captions**: `COCO_73k_annots_curated.npy` - curated COCO captions

### Models Used
- **GroundingDINO**: For generating spatial grounding from text descriptions
- **CLIP ViT-B/32**: For text encoding and similarity matching
- **Hungarian Algorithm**: For optimal assignment between predictions and ground truth

## Usage

1. Generate grounding annotations:

   python grounding.py

2. Preprocess the grounding data:

   python preprocess_grounding.py

3. Train the grounding model:

   python bbox_MLP.py