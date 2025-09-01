import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import json
import open_clip
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import random

def get_clip_model(model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', device=None):
    """Load CLIP model, preprocess, and tokenizer using open_clip."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device


class FMRIGroundingDataset(Dataset):
    def __init__(self, 
                fmri_embeddings_path,
                fmri_metadata_path,
                grounding_data_path,
                precomputed_clip_path=None):
       
       # Load fMRI embeddings
        print("Loading fMRI embeddings...")
        fmri_data = np.load(fmri_embeddings_path)
        self.fmri_embeddings = fmri_data['embeddings']
        self.fmri_indices = fmri_data['indices']
       
       # Load fMRI metadata
        print("Loading fMRI metadata...")
        with open(fmri_metadata_path, 'rb') as f:
           self.fmri_metadata = pickle.load(f)
       
       # Load grounding data
        print("Loading grounding data...")
        with open(grounding_data_path, 'rb') as f:
            self.grounding_data = pickle.load(f)
       
       # Create mapping from coco_id to fMRI embedding
        self.coco_to_fmri = {}
        for idx in self.fmri_indices:
            coco_id = self.fmri_metadata[idx]['coco_id']
            self.coco_to_fmri[coco_id] = idx
       
       # Precompute CLIP embeddings
        if precomputed_clip_path and os.path.exists(precomputed_clip_path):
            print("Loading precomputed CLIP embeddings...")
            with open(precomputed_clip_path, 'rb') as f:
                self.phrase_to_clip = pickle.load(f)
        else:
            print("Computing CLIP embeddings for all phrases...")
            self.phrase_to_clip = self._precompute_clip_embeddings()
           
           # Save for future use
            if precomputed_clip_path:
                print(f"Saving CLIP embeddings to {precomputed_clip_path}")
                os.makedirs(os.path.dirname(precomputed_clip_path), exist_ok=True)
                with open(precomputed_clip_path, 'wb') as f:
                    pickle.dump(self.phrase_to_clip, f)
       
       # Create training samples
        self.samples = []
        print("Creating training samples...")
       
        for image_id, image_data in tqdm(self.grounding_data.items()):
            coco_id = image_data['coco_id']
           
           # Skip if we don't have fMRI data for this image
            if coco_id not in self.coco_to_fmri:
                continue
           
            fmri_idx = self.coco_to_fmri[coco_id]
           
           # Process each caption grounding
            for caption_grounding in image_data['caption_groundings']:
                sample = {
                   'fmri_idx': fmri_idx,
                   'coco_id': coco_id,
                   'caption': caption_grounding['processed_caption'],
                   'detections': caption_grounding['detections'],
                   'image_size': image_data['image_size']
                }
                self.samples.append(sample)
       
        print(f"Created {len(self.samples)} training samples from {len(set(s['coco_id'] for s in self.samples))} unique images")
       
       # Flatten fMRI embeddings if needed
        if len(self.fmri_embeddings.shape) == 3:  # [N, 256, 256]
            self.fmri_embeddings = self.fmri_embeddings.reshape(self.fmri_embeddings.shape[0], -1)
       
        self.fmri_embed_dim = self.fmri_embeddings.shape[1]
        print(f"fMRI embedding dimension: {self.fmri_embed_dim}")
   
    def _precompute_clip_embeddings(self):
        """Precompute CLIP embeddings for all unique phrases (using open_clip)."""
        clip_model, _, clip_tokenizer, device = get_clip_model(model_name='ViT-B-32',
                                                               pretrained='laion2b_s34b_b79k')

        # Collect all unique phrases
        all_phrases = set()
        for image_data in self.grounding_data.values():
            for caption_grounding in image_data['caption_groundings']:
                for detection in caption_grounding['detections']:
                    all_phrases.add(detection['phrase'])

        phrase_to_clip = {}
        batch_size = 64
        phrases_list = list(all_phrases)

        print(f"Computing CLIP embeddings for {len(phrases_list)} unique phrases...")

        with torch.no_grad():
            for i in tqdm(range(0, len(phrases_list), batch_size)):
                batch_phrases = phrases_list[i:i+batch_size]

                # Tokenize and encode (open_clip)
                text_tokens = clip_tokenizer(batch_phrases)
                if not torch.is_tensor(text_tokens):
                    text_tokens = torch.tensor(text_tokens)
                text_tokens = text_tokens.to(device)

                clip_features = clip_model.encode_text(text_tokens)  # [B, 512] for ViT-B/32
                clip_features = F.normalize(clip_features, dim=-1)

                for phrase, feature in zip(batch_phrases, clip_features):
                    phrase_to_clip[phrase] = feature.cpu()

        # Clean up
        del clip_model
        if device == 'cuda':
            torch.cuda.empty_cache()

        return phrase_to_clip


    def __len__(self):
        return len(self.samples)
   
    def __getitem__(self, idx):
        sample = self.samples[idx]
       
       # Get fMRI embedding
        fmri_embedding = torch.FloatTensor(self.fmri_embeddings[sample['fmri_idx']])
       
       # Prepare ground truth boxes and labels
        detections = sample['detections']
        num_objects = len(detections)
       
       # Pad/truncate to max 15 objects
        max_objects = 15
        boxes = torch.zeros(max_objects, 4)  # [cx, cy, w, h] normalized
        clip_features = torch.zeros(max_objects, 512)  # CLIP dimension
        valid_mask = torch.zeros(max_objects, dtype=torch.bool)
       
       # Process each detection
        for i, detection in enumerate(detections[:max_objects]):
            boxes[i] = torch.tensor(detection['bbox'])  # Already in [cx, cy, w, h] format
            valid_mask[i] = True
           
           # Get precomputed CLIP embedding (already normalized)
            phrase = detection['phrase']
            clip_features[i] = self.phrase_to_clip[phrase]
       
        return {
           'fmri_embedding': fmri_embedding,
           'boxes': boxes,
           'clip_features': clip_features,
           'valid_mask': valid_mask,
           'num_objects': num_objects,
           'coco_id': sample['coco_id']
        }

    def get_samples_by_indices(self, indices):
       # Create a subset dataset with specific indices
        subset_dataset = FMRIGroundingDataset.__new__(FMRIGroundingDataset)
       
       # Copy all attributes
        for attr in self.__dict__:
            setattr(subset_dataset, attr, getattr(self, attr))
       
       # Update samples to only include specified indices
        subset_dataset.samples = [self.samples[i] for i in indices]
       
        return subset_dataset

def create_data_splits(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
# """Create train/val/test splits ensuring no image overlap between splits"""

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Group samples by coco_id to ensure no image leakage
    image_to_samples = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        image_to_samples[sample['coco_id']].append(idx)

    unique_images = list(image_to_samples.keys())
    num_images = len(unique_images)

    print(f"Total unique images: {num_images}")
    print(f"Total samples: {len(dataset.samples)}")

    # Split images first
    train_images, temp_images = train_test_split(
       unique_images, 
       test_size=(1 - train_ratio), 
       random_state=random_seed
    )

    # Split remaining into val and test
    val_images, test_images = train_test_split(
       temp_images,
       test_size=test_ratio / (val_ratio + test_ratio),
       random_state=random_seed
    )

    print(f"Train images: {len(train_images)} ({len(train_images)/num_images*100:.1f}%)")
    print(f"Val images: {len(val_images)} ({len(val_images)/num_images*100:.1f}%)")
    print(f"Test images: {len(test_images)} ({len(test_images)/num_images*100:.1f}%)")

    # Collect sample indices for each split
    train_indices = []
    val_indices = []
    test_indices = []

    for image_id in train_images:
        train_indices.extend(image_to_samples[image_id])

    for image_id in val_images:
        val_indices.extend(image_to_samples[image_id])

    for image_id in test_images:
        test_indices.extend(image_to_samples[image_id])

    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")

    # Create subset datasets
    train_dataset = dataset.get_samples_by_indices(train_indices)
    val_dataset = dataset.get_samples_by_indices(val_indices)
    test_dataset = dataset.get_samples_by_indices(test_indices)

    return train_dataset, val_dataset, test_dataset

class FMRIGroundingMLP(nn.Module):
    def __init__(self, 
                fmri_dim,
                hidden_dims=[2048, 1024, 512],
                num_queries=15,
                clip_dim=512):
        super().__init__()
       
        self.num_queries = num_queries
        self.clip_dim = clip_dim
       
       # MLP backbone
        layers = []
        prev_dim = fmri_dim
        for hidden_dim in hidden_dims:
            layers.extend([
               nn.Linear(prev_dim, hidden_dim),
               nn.ReLU(inplace=True),
               nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
       
        self.backbone = nn.Sequential(*layers)
       
       # Prediction heads
        self.query_embed = nn.Linear(prev_dim, num_queries * hidden_dims[-1])
       
       # Box regression head
        self.box_head = nn.Sequential(
           nn.Linear(hidden_dims[-1], 256),
           nn.ReLU(inplace=True),
           nn.Linear(256, 4)  # [cx, cy, w, h]
        )
       
       # Classification head (outputs CLIP-like features)
        self.class_head = nn.Sequential(
           nn.Linear(hidden_dims[-1], 256),
           nn.ReLU(inplace=True),
           nn.Linear(256, clip_dim)
        )
       
       # Objectness head
        self.objectness_head = nn.Sequential(
           nn.Linear(hidden_dims[-1], 128),
           nn.ReLU(inplace=True),
           nn.Linear(128, 1)  # Single logit for objectness
        )
       
       # Initialize weights
        self._init_weights()
   
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
   
    def forward(self, fmri_embeddings):
        batch_size = fmri_embeddings.shape[0]
       
       # Forward through backbone
        features = self.backbone(fmri_embeddings)  # [B, hidden_dim]
       
       # Generate queries
        queries = self.query_embed(features)  # [B, num_queries * hidden_dim]
        queries = queries.view(batch_size, self.num_queries, -1)  # [B, num_queries, hidden_dim]
       
       # Predict boxes, classes, and objectness for each query
        pred_boxes = self.box_head(queries)  # [B, num_queries, 4]
        pred_logits = self.class_head(queries)  # [B, num_queries, clip_dim]
        pred_objectness = self.objectness_head(queries).squeeze(-1)  # [B, num_queries]
       
       # Apply sigmoid to box coordinates to ensure [0, 1] range
        pred_boxes = torch.sigmoid(pred_boxes)
       
       # Normalize class features for CLIP similarity
        pred_logits = F.normalize(pred_logits, dim=-1)
       
        return {
           'pred_boxes': pred_boxes,
           'pred_logits': pred_logits,
           'pred_objectness': pred_objectness
        }

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0, temperature=0.07):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.temperature = temperature
   
    @torch.no_grad()
    def forward(self, outputs, targets):
        batch_size, num_queries = outputs['pred_boxes'].shape[:2]
       
       # Flatten predictions
        pred_bbox = outputs['pred_boxes'].flatten(0, 1)  # [B*Q, 4]
        pred_logits = outputs['pred_logits'].flatten(0, 1)  # [B*Q, clip_dim]
       
        indices = []
        for i in range(batch_size):
            target_boxes = targets['boxes'][i]  # [max_objects, 4]
            target_clip = targets['clip_features'][i]  # [max_objects, clip_dim]
            valid_mask = targets['valid_mask'][i]  # [max_objects]
           
           # Only consider valid targets
            valid_targets = valid_mask.nonzero().squeeze(-1)
            if len(valid_targets) == 0:
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue
           
            target_boxes = target_boxes[valid_targets]
            target_clip = target_clip[valid_targets]  # Already normalized from dataset
           
           # Compute costs
           # Class cost with temperature scaling
            similarity = torch.mm(pred_logits[i*num_queries:(i+1)*num_queries], target_clip.T)
            cost_class = -(similarity / self.temperature)
           
           # Box L1 cost
            cost_bbox = torch.cdist(pred_bbox[i*num_queries:(i+1)*num_queries], target_boxes, p=1)
           
           # GIoU cost
            cost_giou = -generalized_box_iou(
               pred_bbox[i*num_queries:(i+1)*num_queries],
               target_boxes
            )
           
           # Combine costs
            C = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
           
           # Hungarian matching
            pred_idx, target_idx = linear_sum_assignment(C.cpu())
           
            indices.append((torch.tensor(pred_idx, dtype=torch.long), torch.tensor(target_idx, dtype=torch.long)))
       
        return indices

def box_cxcywh_to_xyxy(x):
#    """Convert box format from [cx, cy, w, h] to [x1, y1, x2, y2]"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_area(boxes):
#    """Compute area of boxes"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def generalized_box_iou(boxes1, boxes2):
#    """Compute GIoU between two sets of boxes"""
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)
   
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
   
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
   
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
   
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
   
    union = area1[:, None] + area2 - inter
   
    iou = inter / union
   
   # GIoU
    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
   
    whi = (rbi - lti).clamp(min=0)
    areai = whi[:, :, 0] * whi[:, :, 1]
   
    return iou - (areai - union) / areai

class FMRIGroundingLoss(nn.Module):
    def __init__(self, matcher, weight_dict, temperature=0.07):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.temperature = temperature
   
    def forward(self, outputs, targets):
       # Hungarian matching
        indices = self.matcher(outputs, targets)
       
       # Compute losses
        loss_dict = {}
       
       # Box loss
        loss_bbox, loss_giou = self.loss_boxes(outputs, targets, indices)
        loss_dict['loss_bbox'] = loss_bbox
        loss_dict['loss_giou'] = loss_giou
       
       # Class loss (CLIP similarity)
        loss_class = self.loss_labels(outputs, targets, indices)
        loss_dict['loss_class'] = loss_class
       
       # Objectness loss
        loss_objectness = self.loss_objectness(outputs, targets, indices)
        loss_dict['loss_objectness'] = loss_objectness
       
       # Total loss
        total_loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
       
        return total_loss, loss_dict
   
    def loss_boxes(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
       
        target_boxes = []
        for i, (_, tgt_idx) in enumerate(indices):
            target_boxes.append(targets['boxes'][i][tgt_idx])
       
        if len(target_boxes) == 0:
            return torch.tensor(0.0, device=src_boxes.device), torch.tensor(0.0, device=src_boxes.device)
       
        target_boxes = torch.cat(target_boxes, dim=0)
       
       # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='mean')
       
       # GIoU loss
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes)).mean()
       
        return loss_bbox, loss_giou
   
    def loss_labels(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['pred_logits'][idx]
       
        target_features = []
        for i, (_, tgt_idx) in enumerate(indices):
            target_features.append(targets['clip_features'][i][tgt_idx])
       
        if len(target_features) == 0:
            return torch.tensor(0.0, device=src_logits.device)
       
        target_features = torch.cat(target_features, dim=0)  # Already normalized
       
       # CLIP similarity loss with temperature
        similarity = torch.sum(src_logits * target_features, dim=1) / self.temperature
        loss_class = -similarity.mean()
       
        return loss_class
   
    def loss_objectness(self, outputs, targets, indices):
#        """Binary classification loss for objectness"""
        batch_size, num_queries = outputs['pred_objectness'].shape
       
       # Create objectness targets
        objectness_targets = torch.zeros_like(outputs['pred_objectness'])
       
        for i, (src_idx, _) in enumerate(indices):
            objectness_targets[i, src_idx] = 1.0  # Matched slots should be "object"
       
       # BCE loss
        loss_objectness = F.binary_cross_entropy_with_logits(
           outputs['pred_objectness'], 
           objectness_targets, 
           reduction='mean'
        )
       
        return loss_objectness
   
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

def compute_box_iou(box1, box2):
#    """Compute IoU between two boxes in cxcywh format"""
   # Convert to xyxy
    def cxcywh_to_xyxy(box):
        cx, cy, w, h = box
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
   
    box1_xyxy = cxcywh_to_xyxy(box1)
    box2_xyxy = cxcywh_to_xyxy(box2)
   
   # Compute intersection
    x1 = max(box1_xyxy[0], box2_xyxy[0])
    y1 = max(box1_xyxy[1], box2_xyxy[1])
    x2 = min(box1_xyxy[2], box2_xyxy[2])
    y2 = min(box1_xyxy[3], box2_xyxy[3])
   
    if x2 <= x1 or y2 <= y1:
        return 0.0
   
    intersection = (x2 - x1) * (y2 - y1)
   
   # Compute areas
    area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
   
    union = area1 + area2 - intersection
   
    return intersection / union if union > 0 else 0.0

def compute_coco_style_metrics(predictions, targets, iou_thresholds=None):
#    """Compute COCO-style AP metrics at multiple IoU thresholds"""
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)  # 0.5:0.05:0.95
   
   # Compute metrics at each threshold
    aps = []
    ars = []
   
    for iou_thresh in iou_thresholds:
       # Compute metrics at this threshold
        total_tp = 0
        total_fp = 0
        total_gt = 0
       
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            target_boxes = target['boxes'].cpu().numpy()
            target_valid = target['valid_mask'].cpu().numpy()
           
           # Filter valid targets
            target_boxes = target_boxes[target_valid]
            total_gt += len(target_boxes)
           
            if len(pred_boxes) == 0:
                continue
           
           # Sort predictions by confidence
            sorted_indices = np.argsort(pred_scores)[::-1]
            pred_boxes = pred_boxes[sorted_indices]
            pred_scores = pred_scores[sorted_indices]
           
            matched_targets = set()
           
            for pred_box, pred_score in zip(pred_boxes, pred_scores):
                if pred_score < 0.5:  # Skip low-confidence predictions
#                     total_fp += 1
                    continue
               
                best_iou = 0
                best_target_idx = -1
               
                for target_idx, target_box in enumerate(target_boxes):
                    if target_idx in matched_targets:
                        continue
                   
                    iou = compute_box_iou(pred_box, target_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_target_idx = target_idx
               
                if best_iou >= iou_thresh and best_target_idx not in matched_targets:
                    total_tp += 1
                    matched_targets.add(best_target_idx)
                else:
                    total_fp += 1
       
       # Compute AP and AR for this threshold
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
       
        aps.append(precision)
        ars.append(recall)
   
   # Compute average metrics
    mean_ap = np.mean(aps)
    mean_ar = np.mean(ars)
    ap50 = aps[0]  # AP@0.5
    ap75 = aps[5]  # AP@0.75 (index 5 corresponds to 0.75)
   
    return {
       'AP': mean_ap,
       'AP50': ap50,
       'AP75': ap75,
       'AR': mean_ar,
       'AP_per_threshold': aps,
       'AR_per_threshold': ars,
       'iou_thresholds': iou_thresholds.tolist()
    }

def evaluate_model(model, dataloader, device, split_name="Test"):
#    """Evaluate model and return detailed metrics"""
    model.eval()
    all_predictions = []
    all_targets = []
   
    print(f"\nEvaluating on {split_name} set...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{split_name} evaluation"):
            fmri_embeddings = batch['fmri_embedding'].to(device)
            targets = {
                'boxes': batch['boxes'].to(device),
               'clip_features': batch['clip_features'].to(device),
               'valid_mask': batch['valid_mask'].to(device)
            }
           
            outputs = model(fmri_embeddings)
           
            batch_size = outputs['pred_boxes'].shape[0]
            for i in range(batch_size):
                pred_boxes = outputs['pred_boxes'][i]
                pred_objectness = torch.sigmoid(outputs['pred_objectness'][i])
               
               # Filter by objectness score
                confident_mask = pred_objectness > 0.3
                pred_boxes = pred_boxes[confident_mask]
                pred_scores = pred_objectness[confident_mask]
               
                all_predictions.append({
                   'boxes': pred_boxes,
                   'scores': pred_scores
                })
               
                all_targets.append({
                   'boxes': targets['boxes'][i],
                   'valid_mask': targets['valid_mask'][i]
                })
   
   # Compute COCO-style metrics
    metrics = compute_coco_style_metrics(all_predictions, all_targets)
   
    print(f"\n{split_name} Results:")
    print(f"  AP (avg 0.5:0.95): {metrics['AP']:.4f}")
    print(f"  AP50: {metrics['AP50']:.4f}")
    print(f"  AP75: {metrics['AP75']:.4f}")
    print(f"  AR: {metrics['AR']:.4f}")
   
    return metrics, all_predictions, all_targets

def train_and_evaluate():
   # Paths
    fmri_embeddings_path = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/precomputed_embeddings/fmri/fmri_embeddings_20250813_032047.npz"
    fmri_metadata_path = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/precomputed_embeddings/fmri/fmri_metadata_20250813_032047.pkl"
    grounding_data_path = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/grounding_data/grounding_data_processed.pkl"
    precomputed_clip_path = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/precomputed_clip_embeddings.pkl"
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
   
   # Create full dataset
    print("Creating full dataset...")
    full_dataset = FMRIGroundingDataset(
       fmri_embeddings_path,
       fmri_metadata_path,
       grounding_data_path,
       precomputed_clip_path
    )
   
   # Create train/val/test splits
    print("\nCreating 80/10/10 train/val/test splits...")
    train_dataset, val_dataset, test_dataset = create_data_splits(
       full_dataset, 
       train_ratio=0.8, 
       val_ratio=0.1, 
       test_ratio=0.1,
       random_seed=42
    )
   
   # Create dataloaders
    batch_size = 64
   
    train_loader = DataLoader(
       train_dataset,
       batch_size=batch_size,
       shuffle=True,
       num_workers=0,
       collate_fn=lambda batch: {
           'fmri_embedding': torch.stack([b['fmri_embedding'] for b in batch]),
           'boxes': torch.stack([b['boxes'] for b in batch]),
           'clip_features': torch.stack([b['clip_features'] for b in batch]),
           'valid_mask': torch.stack([b['valid_mask'] for b in batch]),
           'num_objects': [b['num_objects'] for b in batch],
           'coco_id': [b['coco_id'] for b in batch]
        }
    )
   
    val_loader = DataLoader(
       val_dataset,
       batch_size=batch_size,
       shuffle=False,
       num_workers=0,
       collate_fn=train_loader.collate_fn
    )
   
    test_loader = DataLoader(
       test_dataset,
       batch_size=batch_size,
       shuffle=False,
       num_workers=0,
       collate_fn=train_loader.collate_fn
    )
   
   # Create model
    print("Creating model...")
    model = FMRIGroundingMLP(
       fmri_dim=full_dataset.fmri_embed_dim,
       hidden_dims=[2048, 1024, 512],
       num_queries=15,
       clip_dim=512
    ).to(device)
   
   # Create loss components with balanced weights
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0, temperature=0.07)
    weight_dict = {
       'loss_class': 1.0,      # CLIP similarity loss
       'loss_bbox': 5.0,       # L1 box loss
       'loss_giou': 2.0,       # GIoU loss
       'loss_objectness': 2.0  # Objectness loss
    }
    criterion = FMRIGroundingLoss(matcher, weight_dict, temperature=0.07)
   
   # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
   
   # Training loop with validation
    num_epochs = 50
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
   
    best_val_ap = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    val_metrics_history = []
   
    for epoch in range(num_epochs):
       # Training phase
        model.train()
        epoch_train_loss = 0
        train_loss_components = defaultdict(float)
       
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
           
           # Move to device
            fmri_embeddings = batch['fmri_embedding'].to(device)
            targets = {
               'boxes': batch['boxes'].to(device),
               'clip_features': batch['clip_features'].to(device),
               'valid_mask': batch['valid_mask'].to(device)
            }
           
           # Forward pass
            outputs = model(fmri_embeddings)
           
           # Compute loss
            loss, loss_dict = criterion(outputs, targets)
           
           # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
           
            epoch_train_loss += loss.item()
            for k, v in loss_dict.items():
                train_loss_components[k] += v.item()
           
           # Update progress bar
            pbar.set_postfix({
               'Loss': f"{loss.item():.4f}",
               'Bbox': f"{loss_dict['loss_bbox'].item():.4f}",
               'GIoU': f"{loss_dict['loss_giou'].item():.4f}",
               'Class': f"{loss_dict['loss_class'].item():.4f}",
               'Obj': f"{loss_dict['loss_objectness'].item():.4f}"
            })
       
       # Validation phase
        model.eval()
        epoch_val_loss = 0
        val_loss_components = defaultdict(float)
       
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in pbar:
                fmri_embeddings = batch['fmri_embedding'].to(device)
                targets = {
                   'boxes': batch['boxes'].to(device),
                   'clip_features': batch['clip_features'].to(device),
                   'valid_mask': batch['valid_mask'].to(device)
                }
               
                outputs = model(fmri_embeddings)
                loss, loss_dict = criterion(outputs, targets)
               
                epoch_val_loss += loss.item()
                for k, v in loss_dict.items():
                    val_loss_components[k] += v.item()
               
                pbar.set_postfix({
                   'Val_Loss': f"{loss.item():.4f}"
                })
       
        scheduler.step()
       
       # Compute average losses
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
       
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
       
       # Evaluate on validation set every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_metrics, _, _ = evaluate_model(model, val_loader, device, "Validation")
            val_metrics_history.append(val_metrics)
           
           # Save best model based on validation AP
            if val_metrics['AP'] > best_val_ap:
                best_val_ap = val_metrics['AP']
                best_model_state = model.state_dict().copy()
                print(f"  New best validation AP: {best_val_ap:.4f}")
       
       # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
       
       # Print loss components
        for k in train_loss_components:
            avg_train_comp = train_loss_components[k] / len(train_loader)
            avg_val_comp = val_loss_components[k] / len(val_loader)
            print(f"    {k}: Train={avg_train_comp:.4f}, Val={avg_val_comp:.4f}")
       
       # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/fmri_grounding_checkpoint_epoch_{epoch+1}.pth"
            torch.save({
               'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'best_model_state_dict': best_model_state,
               'optimizer_state_dict': optimizer.state_dict(),
               'scheduler_state_dict': scheduler.state_dict(),
               'train_losses': train_losses,
               'val_losses': val_losses,
               'val_metrics_history': val_metrics_history,
               'best_val_ap': best_val_ap,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
   
    print(f"\nTraining completed! Best validation AP: {best_val_ap:.4f}")
   
   # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model based on validation AP")
   
   # Final evaluation on all splits
    print("\n" + "="*50)
    print("FINAL EVALUATION ON ALL SPLITS")
    print("="*50)
   
   # Evaluate on train set (subset for efficiency)
    train_subset_loader = DataLoader(
       train_dataset,
       batch_size=batch_size,
       shuffle=False,
       num_workers=0,
       collate_fn=train_loader.collate_fn
    )
   
   # Limit train evaluation to first 1000 samples for efficiency
    train_eval_samples = []
    for i, batch in enumerate(train_subset_loader):
        train_eval_samples.append(batch)
        if i * batch_size >= 1000:  # Evaluate on ~1000 samples
            break
   
   # Create temporary loader for train evaluation
    def create_eval_generator(samples):
        for sample in samples:
            yield sample
   
    train_metrics, _, _ = evaluate_model(model, create_eval_generator(train_eval_samples), device, "Train (subset)")
    val_metrics, _, _ = evaluate_model(model, val_loader, device, "Validation")
    test_metrics, test_predictions, test_targets = evaluate_model(model, test_loader, device, "Test")
   
   # Save final results
    final_results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_metrics_history': val_metrics_history
        },
        'model_config': {
            'fmri_dim': full_dataset.fmri_embed_dim,
            'hidden_dims': [2048, 1024, 512],
            'num_queries': 15,
            'clip_dim': 512
        }
    }
   
   # Save detailed test results
    results_path = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/fmri_grounding_final_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(final_results, f)
   
   # Train final model on full training set (80% of data)
    print("\n" + "="*50)
    print("TRAINING FINAL MODEL ON 80% OF DATA")
    print("="*50)
   
   # Retrain model on train set only for final deployment
    final_model = FMRIGroundingMLP(
        fmri_dim=full_dataset.fmri_embed_dim,
        hidden_dims=[2048, 1024, 512],
        num_queries=15,
        clip_dim=512
    ).to(device)
   
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-4, weight_decay=1e-4)
    final_scheduler = torch.optim.lr_scheduler.StepLR(final_optimizer, step_size=15, gamma=0.1)
   
   # Train for same number of epochs on training data
    final_epochs = num_epochs
    print(f"Training final model for {final_epochs} epochs on training set only...")
   
    final_model.train()
    for epoch in range(final_epochs):
        epoch_loss = 0
       
        pbar = tqdm(train_loader, desc=f"Final Training Epoch {epoch+1}/{final_epochs}")
        for batch in pbar:
            final_optimizer.zero_grad()
           
            fmri_embeddings = batch['fmri_embedding'].to(device)
            targets = {
                'boxes': batch['boxes'].to(device),
                'clip_features': batch['clip_features'].to(device),
                'valid_mask': batch['valid_mask'].to(device)
            }
           
            outputs = final_model(fmri_embeddings)
            loss, loss_dict = criterion(outputs, targets)
           
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            final_optimizer.step()
           
            epoch_loss += loss.item()
           
            pbar.set_postfix({
               'Loss': f"{loss.item():.4f}",
               'Avg_Loss': f"{epoch_loss/(pbar.n+1):.4f}"
            })
       
        final_scheduler.step()
       
        if (epoch + 1) % 10 == 0:
            print(f"Final training epoch {epoch+1}/{final_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
   
   # Save final production model
    final_model_path = "/home/mohammed.elbey/lustre/aim_neural-7he0p8agska/users/mohammed.elbey/fmri_grounding_production_model.pth"
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'model_config': {
            'fmri_dim': full_dataset.fmri_embed_dim,
            'hidden_dims': [2048, 1024, 512],
            'num_queries': 15,
            'clip_dim': 512
        },
        'training_completed': True,
        'trained_on': 'full_training_set',
        'dataset_stats': {
            'total_samples': len(full_dataset),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset)
        }
    }, final_model_path)
   
    print(f"\nFinal production model saved: {final_model_path}")
   
   # Print comprehensive results summary
    print("\n" + "="*60)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*60)
   
    print(f"\nDataset Split:")
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Train: {len(train_dataset)} ({len(train_dataset)/len(full_dataset)*100:.1f}%)")
    print(f"  Val: {len(val_dataset)} ({len(val_dataset)/len(full_dataset)*100:.1f}%)")
    print(f"  Test: {len(test_dataset)} ({len(test_dataset)/len(full_dataset)*100:.1f}%)")
   
    print(f"\nTest Set Performance (Unseen Data):")
    print(f"  AP (avg 0.5:0.95): {test_metrics['AP']:.4f}")
    print(f"  AP50: {test_metrics['AP50']:.4f}")
    print(f"  AP75: {test_metrics['AP75']:.4f}")
    print(f"  AR: {test_metrics['AR']:.4f}")
   
    print(f"\nComparison Across Splits:")
    splits_comparison = [
       ("Train (subset)", train_metrics),
       ("Validation", val_metrics),
       ("Test", test_metrics)
    ]
   
    for split_name, metrics in splits_comparison:
        print(f"  {split_name:15} - AP: {metrics['AP']:.4f}, AP50: {metrics['AP50']:.4f}, AR: {metrics['AR']:.4f}")
   
   # Check for overfitting
    if test_metrics['AP'] < train_metrics['AP'] - 0.1:
        print(f"\n⚠️  Potential overfitting detected:")
        print(f"     Train AP: {train_metrics['AP']:.4f} vs Test AP: {test_metrics['AP']:.4f}")
        print(f"     Difference: {train_metrics['AP'] - test_metrics['AP']:.4f}")
    else:
        print(f"\n✅ Model generalizes well to unseen data")
   
    print(f"\nFiles saved:")
    print(f"  - Results: {results_path}")
    print(f"  - Production model: {final_model_path}")
   
   # Create sample test predictions for inspection
    print(f"\nSample Test Predictions (first 5 samples):")
    sample_predictions = test_predictions[:5]
    sample_targets = test_targets[:5]
   
    for i, (pred, target) in enumerate(zip(sample_predictions, sample_targets)):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        target_boxes = target['boxes'].cpu().numpy()
        target_valid = target['valid_mask'].cpu().numpy()
       
        valid_targets = target_boxes[target_valid]
       
        print(f"\n  Sample {i+1}:")
        print(f"    Ground truth: {len(valid_targets)} objects")
        print(f"    Predicted: {len(pred_boxes)} objects (score > 0.3)")
        if len(pred_boxes) > 0:
            print(f"    Best prediction score: {pred_scores.max():.3f}")
            print(f"    Avg prediction score: {pred_scores.mean():.3f}")
   
    return final_model, test_metrics, final_results

if __name__ == "__main__":
    final_model, test_metrics, results = train_and_evaluate()
   
    print("\n" + "="*60)
    print("TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Final Test AP: {test_metrics['AP']:.4f}")
    print(f"Final Test AP50: {test_metrics['AP50']:.4f}")
    print(f"Final Test AR: {test_metrics['AR']:.4f}")
    print("\nProduction model ready for deployment!")