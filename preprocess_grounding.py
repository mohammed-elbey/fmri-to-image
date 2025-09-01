import pickle
import numpy as np
from collections import defaultdict
import json

def preprocess_grounding_data(input_path, output_path, min_confidence=0.3, min_area_ratio=0.01):
    
    
    # Load the data
    print("Loading grounding data...")
    with open(input_path, 'rb') as f:
        grounding_data = pickle.load(f)
    
    processed_data = {}
    stats = {
        'total_images_original': 0,
        'total_images_processed': 0,
        'total_captions_original': 0,
        'total_captions_processed': 0,
        'total_detections_original': 0,
        'total_detections_processed': 0,
        'images_with_no_valid_detections': 0,
        'filtered_by_confidence': 0,
        'filtered_by_size': 0,
        'caption_distribution_original': defaultdict(int),
        'caption_distribution_processed': defaultdict(int),
        'detection_distribution_original': defaultdict(int),
        'detection_distribution_processed': defaultdict(int),
    }
    
    for image_id, image_data in grounding_data.items():
        stats['total_images_original'] += 1
        
        image_width = image_data['image_size']['width']
        image_height = image_data['image_size']['height']
        total_image_area = image_width * image_height
        min_area = total_image_area * min_area_ratio
        
        processed_image_data = {
            'image_path': image_data['image_path'],
            'image_size': image_data['image_size'],
            'coco_id': image_data['coco_id'],
            'caption_groundings': []
        }
        
        stats['total_captions_original'] += len(image_data['caption_groundings'])
        stats['caption_distribution_original'][len(image_data['caption_groundings'])] += 1
        
        has_valid_detections = False
        
        for caption_grounding in image_data['caption_groundings']:
            stats['total_detections_original'] += len(caption_grounding['detections'])
            stats['detection_distribution_original'][len(caption_grounding['detections'])] += 1
            
            filtered_detections = []
            
            for detection in caption_grounding['detections']:
                bbox = detection['bbox']
                confidence = detection['confidence']
                
                # Calculate bounding box area
                x1, y1, x2, y2 = bbox
                bbox_width = abs(x2 - x1)
                bbox_height = abs(y2 - y1)
                bbox_area = bbox_width * bbox_height
                
                # Filter by confidence
                if confidence < min_confidence:
                    stats['filtered_by_confidence'] += 1
                    continue
                
                # Filter by area
                if bbox_area < min_area:
                    stats['filtered_by_size'] += 1
                    continue
                
                # Convert bbox to DETR format (normalized coordinates)
                # DETR expects [center_x, center_y, width, height] normalized to [0, 1]
                center_x = (x1 + x2) / 2.0 / image_width
                center_y = (y1 + y2) / 2.0 / image_height
                norm_width = bbox_width / image_width
                norm_height = bbox_height / image_height
                
                # Ensure coordinates are within bounds
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))
                
                detr_detection = {
                    'phrase': detection['phrase'],
                    'bbox': [center_x, center_y, norm_width, norm_height],  # DETR format
                    'confidence': confidence,
                    'original_bbox': bbox  # Keep original for reference
                }
                
                filtered_detections.append(detr_detection)
                stats['total_detections_processed'] += 1
                has_valid_detections = True
            
            # Only keep captions that have valid detections
            if filtered_detections:
                processed_caption = {
                    'caption_index': caption_grounding['caption_index'],
                    'original_caption': caption_grounding['original_caption'],
                    'processed_caption': caption_grounding['processed_caption'],
                    'detections': filtered_detections,
                    'detection_count': len(filtered_detections)
                }
                processed_image_data['caption_groundings'].append(processed_caption)
                stats['total_captions_processed'] += 1
        
        # Only keep images that have at least one valid detection
        if has_valid_detections:
            processed_data[image_id] = processed_image_data
            stats['total_images_processed'] += 1
            stats['caption_distribution_processed'][len(processed_image_data['caption_groundings'])] += 1
            
            # Count detections per image
            total_detections_in_image = sum(
                len(cg['detections']) for cg in processed_image_data['caption_groundings']
            )
            stats['detection_distribution_processed'][total_detections_in_image] += 1
        else:
            stats['images_with_no_valid_detections'] += 1
    
    # Calculate averages
    if stats['total_images_processed'] > 0:
        stats['average_captions_per_image'] = stats['total_captions_processed'] / stats['total_images_processed']
        stats['average_detections_per_image'] = stats['total_detections_processed'] / stats['total_images_processed']
    
    if stats['total_captions_processed'] > 0:
        stats['average_detections_per_caption'] = stats['total_detections_processed'] / stats['total_captions_processed']
    
    # Save processed data
    print(f"Saving processed data to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    # Print statistics
    print("\n" + "="*50)
    print("PREPROCESSING STATISTICS")
    print("="*50)
    
    print(f"\nIMAGE STATISTICS:")
    print(f"Original images: {stats['total_images_original']:,}")
    print(f"Processed images: {stats['total_images_processed']:,}")
    print(f"Images filtered out (no valid detections): {stats['images_with_no_valid_detections']:,}")
    print(f"Retention rate: {stats['total_images_processed']/stats['total_images_original']*100:.2f}%")
    
    print(f"\nCAPTION STATISTICS:")
    print(f"Original captions: {stats['total_captions_original']:,}")
    print(f"Processed captions: {stats['total_captions_processed']:,}")
    print(f"Average captions per image (processed): {stats['average_captions_per_image']:.4f}")
    
    print(f"\nDETECTION STATISTICS:")
    print(f"Original detections: {stats['total_detections_original']:,}")
    print(f"Processed detections: {stats['total_detections_processed']:,}")
    print(f"Filtered by confidence (<{min_confidence}): {stats['filtered_by_confidence']:,}")
    print(f"Filtered by size (<{min_area_ratio*100}% of image): {stats['filtered_by_size']:,}")
    print(f"Detection retention rate: {stats['total_detections_processed']/stats['total_detections_original']*100:.2f}%")
    print(f"Average detections per image (processed): {stats['average_detections_per_image']:.4f}")
    print(f"Average detections per caption (processed): {stats['average_detections_per_caption']:.4f}")
    
    print(f"\nCAPTION DISTRIBUTION (Processed):")
    for num_captions in sorted(stats['caption_distribution_processed'].keys()):
        count = stats['caption_distribution_processed'][num_captions]
        print(f"  {num_captions} caption(s): {count:,} images")
    
    print(f"\nDETECTION DISTRIBUTION (Processed - top 10):")
    sorted_detection_dist = sorted(stats['detection_distribution_processed'].items(), key=lambda x: x[1], reverse=True)
    for num_detections, count in sorted_detection_dist[:10]:
        print(f"  {num_detections} detection(s): {count:,} images")
    
    # Save statistics
    stats_path = output_path.replace('.pkl', '_stats.json')
    with open(stats_path, 'w') as f:
        # Convert defaultdict to dict for JSON serialization
        stats_json = {k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in stats.items()}
        json.dump(stats_json, f, indent=2)
    
    print(f"\nStatistics saved to: {stats_path}")
    print(f"Processed data saved to: {output_path}")
    
    return processed_data, stats

# Run the preprocessing
if __name__ == "__main__":
    input_path = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/grounding_data/grounding_data_complete.pkl"
    output_path = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/grounding_data/grounding_data_processed.pkl"
    
    processed_data, stats = preprocess_grounding_data(
        input_path=input_path,
        output_path=output_path,
        min_confidence=0.3,
        min_area_ratio=0.01
    )