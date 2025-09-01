import os
import sys
import torch
import numpy as np
import pickle
import json
from PIL import Image
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add GroundingDINO to path - adjust this path based on your setup
sys.path.append('/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/trying/GroundingDINO')

try:
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    from groundingdino.util import box_ops
    import groundingdino.datasets.transforms as T
except ImportError as e:
    print(f"Error importing GroundingDINO: {e}")
    print("Make sure GroundingDINO is properly installed and the path is correct")
    sys.exit(1)


class GroundingDINOProcessor:
    def __init__(self, config_path, checkpoint_path, device='cuda'):
        self.device = device
        self.model = self.load_model(config_path, checkpoint_path)
        self.transform = self.get_transform()
        
    def load_model(self, config_path, checkpoint_path):
        """Load GroundingDINO model with proper error handling"""
        try:
            # Load config
            args = SLConfig.fromfile(config_path)
            args.device = self.device
            
            # Build model
            model = build_model(args)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            print(f"Model loading result: {load_res}")
            
            model.eval()
            model = model.to(self.device)
            
            return model
            
        except Exception as e:
            print(f"Error loading GroundingDINO model: {e}")
            raise e
    
    def get_transform(self):
        """Get image transformation pipeline"""
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
        
        return transform
    
    def preprocess_image(self, image_path):
        """Load and preprocess image with proper error handling"""
        try:
            # Load image
            image_pil = Image.open(image_path).convert("RGB")
            
            # Convert to numpy for OpenCV operations if needed
            image_np = np.array(image_pil)
            
            # Apply transforms
            transformed_image, _ = self.transform(image_pil, None)
            
            return image_pil, image_np, transformed_image
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None, None, None
    
    def process_caption(self, caption):
        """Clean and process caption text"""
        if not caption or not isinstance(caption, str):
            return ""
        
        # Clean caption
        caption = caption.strip()
        if caption.endswith('...'):
            caption = caption[:-3]
        if caption.endswith('.'):
            caption = caption[:-1]
            
        # Make sure caption ends with period for GroundingDINO
        if not caption.endswith('.'):
            caption += '.'
            
        return caption
    
    def generate_grounding(self, image_path, caption, box_threshold=0.35, text_threshold=0.25):
        """Generate grounding information for image-caption pair"""
        try:
            # Preprocess image
            image_pil, image_np, transformed_image = self.preprocess_image(image_path)
            if transformed_image is None:
                return None
            
            # Process caption
            processed_caption = self.process_caption(caption)
            if not processed_caption:
                return None
            
            # Ensure transformed_image is a tensor and move to device
            if isinstance(transformed_image, np.ndarray):
                transformed_image = torch.from_numpy(transformed_image).float()
            
            # Make sure it's on the correct device
            transformed_image = transformed_image.to(self.device)
            
            # Add batch dimension if needed
            if transformed_image.dim() == 3:
                transformed_image = transformed_image.unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(transformed_image, captions=[processed_caption])
            
            # Extract predictions
            prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
            prediction_boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
            
            # Filter predictions
            logits_filt = prediction_logits.clone()
            boxes_filt = prediction_boxes.clone()
            filt_mask = logits_filt.max(dim=1)[0] > box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
            
            # Get phrase information
            tokenlizer = self.model.tokenizer
            tokenized = tokenlizer(processed_caption)
            
            # Get phrases from position map
            pred_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(
                    logit > text_threshold, 
                    tokenized, 
                    tokenlizer
                )
                pred_phrases.append(pred_phrase)
            
            # Convert boxes to proper format (x1, y1, x2, y2)
            H, W = image_pil.size[1], image_pil.size[0]  # PIL returns (W, H)
            
            # Convert from center format to corner format and scale
            boxes_scaled = []
            for box in boxes_filt:
                # box is in format [center_x, center_y, width, height] normalized
                center_x, center_y, width, height = box.tolist()
                
                # Convert to corner format and scale
                x1 = (center_x - width / 2) * W
                y1 = (center_y - height / 2) * H
                x2 = (center_x + width / 2) * W
                y2 = (center_y + height / 2) * H
                
                # Clamp to image bounds
                x1 = max(0, min(W, x1))
                y1 = max(0, min(H, y1))
                x2 = max(0, min(W, x2))
                y2 = max(0, min(H, y2))
                
                boxes_scaled.append([x1, y1, x2, y2])
            
            # Create grounding data structure
            grounding_data = {
                'image_path': image_path,
                'original_caption': caption,
                'processed_caption': processed_caption,
                'image_size': {'width': W, 'height': H},
                'detections': []
            }
            
            # Add detections
            for i, (phrase, box, logit) in enumerate(zip(pred_phrases, boxes_scaled, logits_filt)):
                detection = {
                    'phrase': phrase,
                    'bbox': box,  # [x1, y1, x2, y2]
                    'confidence': float(logit.max().item()),
                    'logits': logit.cpu().numpy().tolist()
                }
                grounding_data['detections'].append(detection)
            
            return grounding_data
            
        except Exception as e:
            print(f"Error in generate_grounding for {image_path}: {e}")
            return None


def load_caption_data(caption_path):
    """Load caption data from numpy file"""
    print(f"Loading captions from {caption_path}")
    caption_array = np.load(caption_path, allow_pickle=True)
    
    MAX_CAPTIONS = 5
    caption_data = {}
    
    for i, row in enumerate(caption_array):
        captions = [cap for cap in row if cap and cap.strip()]
        if len(captions) < MAX_CAPTIONS:
            captions += [""] * (MAX_CAPTIONS - len(captions))
        else:
            captions = captions[:MAX_CAPTIONS]
        caption_data[i] = captions
    
    print(f"Loaded captions for {len(caption_data)} images")
    return caption_data


def main():
    # Configuration
    config = {
        'grounding_dino_config': '/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/trying/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
        'grounding_dino_checkpoint': '/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/trying/groundingdino_swint_ogc.pth',
        'caption_path': '/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/COCO_73k_annots_curated.npy',
        'image_root': '/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/nsddata_stimuli/stimuli/images',
        'output_dir': '/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/grounding_data',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'box_threshold': 0.35,
        'text_threshold': 0.25,
        'max_images': None,  # Set to a number to limit processing, None for all
        'save_frequency': 7300  # Save intermediate results every N images
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize GroundingDINO processor
    print("Initializing GroundingDINO...")
    try:
        processor = GroundingDINOProcessor(
            config['grounding_dino_config'],
            config['grounding_dino_checkpoint'],
            config['device']
        )
        print(f"GroundingDINO initialized successfully on {config['device']}")
    except Exception as e:
        print(f"Failed to initialize GroundingDINO: {e}")
        return
    
    # Load captions
    caption_data = load_caption_data(config['caption_path'])
    
    # Get list of available images
    image_files = [f for f in os.listdir(config['image_root']) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} image files")
    
    # Process images
    grounding_results = {}
    processed_count = 0
    error_count = 0
    
    # Determine which images to process
    images_to_process = []
    for image_file in image_files:
        # Extract COCO ID from filename (assuming format like 'image_000001.jpg')
        if image_file.startswith('image_'):
            try:
                coco_id = int(image_file.split('_')[1].split('.')[0])
                if coco_id in caption_data:
                    images_to_process.append((image_file, coco_id))
                    if config['max_images'] and len(images_to_process) >= config['max_images']:
                        break
            except (ValueError, IndexError):
                continue
    
    print(f"Will process {len(images_to_process)} images with captions")
    
    # Process each image with all its captions
    pbar = tqdm(images_to_process, desc="Generating grounding data")
    
    for image_file, coco_id in pbar:
        try:
            image_path = os.path.join(config['image_root'], image_file)
            captions = caption_data[coco_id]
            
            # Get all non-empty captions
            valid_captions = [cap for cap in captions if cap and cap.strip()]
            if not valid_captions:
                continue
            
            # Store grounding data for all captions of this image
            image_grounding_data = {
                'image_path': image_path,
                'image_size': None,  # Will be set from first successful processing
                'coco_id': coco_id,
                'caption_groundings': []  # List of grounding data for each caption
            }
            
            caption_success_count = 0
            
            # Process each valid caption
            for caption_idx, caption in enumerate(valid_captions):
                try:
                    # Generate grounding for this specific caption
                    grounding_data = processor.generate_grounding(
                        image_path, 
                        caption,
                        config['box_threshold'],
                        config['text_threshold']
                    )
                    
                    if grounding_data is not None:
                        # Store image size from first successful processing
                        if image_grounding_data['image_size'] is None:
                            image_grounding_data['image_size'] = grounding_data['image_size']
                        
                        # Create caption-specific grounding entry
                        caption_grounding = {
                            'caption_index': caption_idx,
                            'original_caption': caption,
                            'processed_caption': grounding_data['processed_caption'],
                            'detections': grounding_data['detections'],
                            'detection_count': len(grounding_data['detections'])
                        }
                        
                        image_grounding_data['caption_groundings'].append(caption_grounding)
                        caption_success_count += 1
                        
                except Exception as e:
                    print(f"Error processing caption {caption_idx} for {image_file}: {e}")
                    continue
            
            # Only save if we have at least one successful caption processing
            if caption_success_count > 0:
                grounding_results[coco_id] = image_grounding_data
                processed_count += 1
            else:
                error_count += 1
                
            # Update progress bar
            pbar.set_postfix({
                'processed': processed_count, 
                'errors': error_count,
                'captions': caption_success_count,
                'current': f'COCO_{coco_id}'
            })
            
            # Save intermediate results
            if processed_count % config['save_frequency'] == 0 and processed_count > 0:
                intermediate_path = os.path.join(config['output_dir'], f'grounding_data_intermediate_{processed_count}.pkl')
                with open(intermediate_path, 'wb') as f:
                    pickle.dump(grounding_results, f)
                print(f"\nSaved intermediate results: {processed_count} images processed")
                
        except Exception as e:
            error_count += 1
            print(f"Error processing {image_file}: {e}")
            continue
    
    # Save final results
    print(f"\nProcessing complete! Processed: {processed_count}, Errors: {error_count}")
    
    # Save as pickle
    output_pickle = os.path.join(config['output_dir'], 'grounding_data_complete.pkl')
    with open(output_pickle, 'wb') as f:
        pickle.dump(grounding_results, f)
    print(f"Saved grounding data to: {output_pickle}")
    
    # Save as JSON (for human readability, but will be large)
    output_json = os.path.join(config['output_dir'], 'grounding_data_sample.json')
    # Save only first 10 results as JSON sample
    sample_results = dict(list(grounding_results.items())[:10])
    with open(output_json, 'w') as f:
        json.dump(sample_results, f, indent=2)
    print(f"Saved sample grounding data to: {output_json}")
    
    # Save configuration
    config_output = os.path.join(config['output_dir'], 'generation_config.json')
    with open(config_output, 'w') as f:
        # Remove non-serializable items
        config_to_save = {k: v for k, v in config.items() if k != 'device'}
        config_to_save['device_used'] = str(config['device'])
        json.dump(config_to_save, f, indent=2)
    
    # Print summary statistics
    if grounding_results:
        total_images = len(grounding_results)
        total_captions = sum(len(data['caption_groundings']) for data in grounding_results.values())
        total_detections = sum(
            sum(caption['detection_count'] for caption in data['caption_groundings']) 
            for data in grounding_results.values()
        )
        avg_captions_per_image = total_captions / total_images
        avg_detections_per_caption = total_detections / total_captions if total_captions > 0 else 0
        avg_detections_per_image = total_detections / total_images
        
        print(f"\nSummary Statistics:")
        print(f"- Total images processed: {total_images}")
        print(f"- Total captions processed: {total_captions}")
        print(f"- Average captions per image: {avg_captions_per_image:.2f}")
        print(f"- Total detections: {total_detections}")
        print(f"- Average detections per caption: {avg_detections_per_caption:.2f}")
        print(f"- Average detections per image: {avg_detections_per_image:.2f}")
        
        # Print distribution of captions per image
        caption_counts = [len(data['caption_groundings']) for data in grounding_results.values()]
        from collections import Counter
        caption_distribution = Counter(caption_counts)
        print(f"\nCaption Distribution:")
        for num_captions, count in sorted(caption_distribution.items()):
            print(f"- {num_captions} caption(s): {count} images")
        
        # Save statistics
        stats = {
            'total_images': total_images,
            'total_captions': total_captions,
            'average_captions_per_image': avg_captions_per_image,
            'total_detections': total_detections,
            'average_detections_per_caption': avg_detections_per_caption,
            'average_detections_per_image': avg_detections_per_image,
            'processed_count': processed_count,
            'error_count': error_count,
            'caption_distribution': dict(caption_distribution)
        }
        
        stats_output = os.path.join(config['output_dir'], 'statistics.json')
        with open(stats_output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to: {stats_output}")


if __name__ == "__main__":
    main()