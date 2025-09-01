import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import pickle
from semantic_pipeline import get_clip_model, get_text_embedding

# --- Configuration ---
data_root = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/nsddata_stimuli/stimuli/images"
caption_path = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/COCO_73k_annots_curated.npy"
out_dir = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/clip_embeddings/"
os.makedirs(out_dir, exist_ok=True)

# Semantic target configuration (should match your training config)
SEMANTIC_RATIO = 0.5  # This should match cfg.semantic_ratio in your training

# --- Load Models ---
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess, clip_tokenizer = get_clip_model(device=device)

# --- Load Captions ---
caption_array = np.load(caption_path, allow_pickle=True)
MAX_CAPTIONS = 5
caption_data = {}
for i, row in enumerate(caption_array):
    captions = [cap for cap in row if cap.strip()]
    if len(captions) < MAX_CAPTIONS:
        captions += [""] * (MAX_CAPTIONS - len(captions))
    else:
        captions = captions[:MAX_CAPTIONS]
    caption_data[str(i)] = captions

# --- Precompute Embeddings ---
image_embs = {}
text_embs = {}
semantic_targets = {}  # NEW: Combined semantic targets

all_ids = sorted([int(fname[6:12]) for fname in os.listdir(data_root)
                  if (fname.endswith(".jpg") or fname.endswith(".png"))])

for coco_id in tqdm(all_ids, desc="Encoding CLIP embeddings"):
    str_id = str(coco_id)
    img_path_jpg = os.path.join(data_root, f"image_{coco_id:06d}.jpg")
    img_path_png = os.path.join(data_root, f"image_{coco_id:06d}.png")
    
    if os.path.exists(img_path_jpg):
        img = Image.open(img_path_jpg).convert("RGB")
    elif os.path.exists(img_path_png):
        img = Image.open(img_path_png).convert("RGB")
    else:
        continue  # skip if image missing

    img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_emb = clip_model.encode_image(img_tensor).squeeze(0).cpu()
        text_emb = get_text_embedding(clip_model, clip_tokenizer, caption_data.get(str_id, ["a photo"]), device).cpu()
        
        # NEW: Precompute the combined semantic target
        # Make sure text_emb has the right shape
        if text_emb.dim() > 1:
            text_emb_flat = text_emb.squeeze(0)
        else:
            text_emb_flat = text_emb
            
        semantic_target = SEMANTIC_RATIO * image_emb + (1 - SEMANTIC_RATIO) * text_emb_flat
    
    image_embs[coco_id] = image_emb
    text_embs[coco_id] = text_emb
    semantic_targets[coco_id] = semantic_target  # NEW: Store combined target

# --- Save to Disk ---
with open(os.path.join(out_dir, "clip_image_embs.pkl"), "wb") as f:
    pickle.dump(image_embs, f)

with open(os.path.join(out_dir, "clip_text_embs.pkl"), "wb") as f:
    pickle.dump(text_embs, f)

# NEW: Save the precomputed semantic targets
with open(os.path.join(out_dir, "semantic_targets.pkl"), "wb") as f:
    pickle.dump(semantic_targets, f)

print("CLIP embeddings and semantic targets saved to disk.")
print(f"Processed {len(semantic_targets)} images with semantic targets")

# Print some stats for verification
if semantic_targets:
    sample_target = next(iter(semantic_targets.values()))
    print(f"Semantic target shape: {sample_target.shape}")
    print(f"Semantic target dtype: {sample_target.dtype}")