import torch
# from omegaconf import OmegaConf
from fmrienc_src.transformer_models import Neural_fMRI2fMRI  # Adjust import to your actual model location
from utils.utils import Config
import numpy as np
import cv2
import h5py
# from omegaconf import OmegaConf
import torch

def compute_crop_coords(vc_mask):
#     Compute bounding box for the VC mask
    H, W = vc_mask.shape
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx, yy], axis=0)

    masked_grid = grid * vc_mask[np.newaxis]
    x1 = min(int(masked_grid[0].max()) + 1, W)
    y1 = min(int(masked_grid[1].max()) + 10, H)
    masked_grid[masked_grid == 0] = 1e6
    x0 = max(int(masked_grid[0].min() - 1), 0)
    y0 = max(int(masked_grid[1].min() - 10), 0)
    return [x0, x1, y0, y1]

def load_transform(vc_mask_path, image_size=(256, 256)):
#     Load VC mask and return the fMRI transform function
    data = np.load(vc_mask_path)
    mask = data['images']
    if mask.ndim == 3:
        mask = mask[0]  # safely get the 2D mask

    vc_mask = mask == 1
    coords = compute_crop_coords(vc_mask)
    crop_mask = vc_mask[coords[2]:coords[3]+1, coords[0]:coords[1]+1]
    cmask = cv2.resize(crop_mask.astype(np.float32), image_size, interpolation=cv2.INTER_NEAREST)

    def transform(image):
        crop = image[coords[2]:coords[3]+1, coords[0]:coords[1]+1]
        resized = cv2.resize(crop, image_size)
        resized[cmask == 0] = 0
        return resized

    return transform

def load_fmri_sample(h5_path, index=0):
#     Load one sample from an fMRI .h5 file
    with h5py.File(h5_path, 'r') as f:
        images = f['images']
        labels = f['labels']
        raw_image = np.array(images[index])
        label = labels[index]
    return raw_image, label



def load_fmri_encoder():
    ckpt_encoder = '/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/trying/fMRI2fMRI_UKB/checkpoint_120000.pth'                                                                                                          
    cfg_file = '/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/trying/fMRI2fMRI_UKB/fMRI_AutoEncoder.yaml'
    config = Config(cfg_file)

    model_encoder = Neural_fMRI2fMRI(config)

    # load without module
    pretrain_metafile = torch.load(ckpt_encoder, map_location='cpu')
    model_keys = set(model_encoder.state_dict().keys())
    load_keys = set(pretrain_metafile['model'].keys())
    state_dict = pretrain_metafile['model']
    if model_keys != load_keys:
        print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
        if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model_encoder.load_state_dict(state_dict)
    print('-----------Loaded FMRI Encoder-----------')

    # del model_encoder.transformer.decoder_pos_embed
    # del model_encoder.transformer.decoder_blocks
    # del model_encoder.transformer.decoder_pred
    # del model_encoder.transformer.decoder_embed
    # del model_encoder.transformer.decoder_norm

    return model_encoder


