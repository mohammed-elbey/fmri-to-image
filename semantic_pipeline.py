import torch
import open_clip
from PIL import Image

def get_clip_model(model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', device='cuda'):
    
# Loads the CLIP model, preprocess function, and tokenizer
    
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model.eval().to(device), preprocess, tokenizer

def get_image_embedding(model, preprocess, image_path, device='cuda'):
    
#     Preprocesses an image and returns its CLIP embedding.
    img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(img)

def get_text_embedding(model, tokenizer, captions, device='cuda'):
    
#     Takes a list of captions and returns the averaged CLIP embedding.
#     If captions is empty, uses a fallback prompt.
    
    if not captions:
        captions = ["a photo"]  # Fallback

    tokens = tokenizer(captions).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(tokens)  # shape: [N, D]
    
    # Average over all non-empty captions
    return text_feats.mean(dim=0, keepdim=True)
