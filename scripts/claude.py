import argparse
import json
import os
import random
from dataclasses import dataclass
import math

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    RetinaNet_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from tqdm import tqdm
import sys
if 'ipykernel' in sys.modules:
    sys.argv = [sys.argv[0]]


def default_num_workers():
    cpu_count = os.cpu_count() or 0
    return max(cpu_count - 1, 0)


@dataclass
class TrainConfig:
    batch_size: int = 16
    num_epochs: int = 70
    learning_rate: float = 0.001
    num_classes: int = 2
    conf_threshold: float = 0.05
    model_name: str = 'fasterrcnn'
    backbone: str = 'resnet50'
    num_workers: int = default_num_workers()
    train_img_dir: str = os.path.join('dataset-2', 'images', 'train')
    train_label_dir: str = os.path.join('dataset-2', 'labels', 'train')
    val_img_dir: str = os.path.join('dataset-2', 'images', 'val')
    val_label_dir: str = os.path.join('dataset-2', 'labels', 'val')
    output_dir: str = 'results'
    seed: int = 42
    use_amp: bool = True
    pretrained: bool = True
    warmup_epochs: int = 5


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_augmentation_pipeline(train=True):
    """
    Albumentations pipeline'ını oluşturur.
    ToTensorV2 ve Normalize dahildir.
    """
    if train:
        transform = A.Compose([
            # Augmentations
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                brightness_by_max=True,
                p=0.5),
            A.GaussNoise(std_range= [0.01,0.07], p=0.2),
            A.Blur(blur_limit=[1,4], p=0.2),         
          
            A.ShiftScaleRotate(shift_limit= [-0.0625, 0.0625], scale_limit=[-0.1,0.1], 
                              rotate_limit=[-10,10], p=0.3),
            # Normalization ve ToTensorV2
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            A.ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo', 
            label_fields=['class_labels'],
            min_area=0,
            min_visibility=0.3,
            clip=True
        ))
    else:
        # Validation için sadece normalization ve ToTensorV2
        transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            A.ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo', 
            label_fields=['class_labels'],
            min_area=0,
            min_visibility=0.3,
            clip=True
        ))
    
    return transform


class PotholeDataset(Dataset):
    """YOLO formatındaki çukur verilerini PyTorch object detection formatına dönüştürür"""
    
    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        
        self.images = [f for f in os.listdir(image_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    def __len__(self):
        return len(self.images)
    
    def yolo_to_pascal_voc(self, yolo_boxes, img_width, img_height):
        """YOLO formatını Pascal VOC formatına dönüştür"""
        pascal_boxes = []
        for box in yolo_boxes:
            x_center, y_center, width, height = box
            
            x_min = (x_center - width / 2) * img_width
            y_min = (y_center - height / 2) * img_height
            x_max = (x_center + width / 2) * img_width
            y_max = (y_center + height / 2) * img_height
            
            pascal_boxes.append([x_min, y_min, x_max, y_max])
        
        return pascal_boxes
    
    def clip_pascal_boxes(self, boxes, img_width, img_height):
        """Pascal VOC kutularını görüntü boyutuna sınırlar ve geçersizleri atar"""
        clipped = []
        for x_min, y_min, x_max, y_max in boxes:
            x_min = np.clip(x_min, 0, img_width)
            y_min = np.clip(y_min, 0, img_height)
            x_max = np.clip(x_max, 0, img_width)
            y_max = np.clip(y_max, 0, img_height)
            if x_max > x_min and y_max > y_min:
                clipped.append([x_min, y_min, x_max, y_max])
        return clipped
    
    def clip_boxes(self, boxes):
        """Bbox koordinatlarını [0, 1] aralığına sınırla"""
        clipped_boxes = []
        for box in boxes:
            x_center = np.clip(box[0], 0.0, 1.0)
            y_center = np.clip(box[1], 0.0, 1.0)
            width = np.clip(box[2], 0.0, 1.0)
            height = np.clip(box[3], 0.0, 1.0)
            
            x_center = np.clip(x_center, width/2, 1.0 - width/2)
            y_center = np.clip(y_center, height/2, 1.0 - height/2)
            
            clipped_boxes.append([x_center, y_center, width, height])
        return clipped_boxes
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        if class_id == 0:
                            x_center, y_center, width, height = map(float, parts[1:])
                            x_center = np.clip(x_center, 0.0, 1.0)
                            y_center = np.clip(y_center, 0.0, 1.0)
                            width = np.clip(width, 0.0, 1.0)
                            height = np.clip(height, 0.0, 1.0)
                            boxes.append([x_center, y_center, width, height])
                            labels.append(1)
        
        # Albumentations transformunu uygula
        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed['image']  # ToTensorV2 ile tensor'e dönüştü
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
            h, w = image.shape[1], image.shape[2]
        
        if len(boxes) > 0:
            boxes = self.yolo_to_pascal_voc(boxes, w, h)
            boxes = self.clip_pascal_boxes(boxes, w, h)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        return image, target


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def debug_sample(dataset, idx=0, save_path=None):
    """Bir örneği görselleştir"""
    image, target = dataset[idx]
    
    # Tensor'ı numpy'a çevir (ToTensorV2 sonrası)
    if torch.is_tensor(image):
        # Inverse normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        image_np = image.numpy().transpose(1, 2, 0)
        image_np = image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        image_np = np.clip(image_np, 0, 1)
    else:
        image_np = image
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_np)
    
    boxes = target['boxes'].cpu().numpy() if torch.is_tensor(target['boxes']) else target['boxes']
    labels = target['labels'].cpu().numpy() if torch.is_tensor(target['labels']) else target['labels']
    
    print(f"Image shape: {image_np.shape}")
    print(f"Number of boxes: {len(boxes)}")
    print(f"Labels: {labels}")
    
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), w, h, 
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.text(x1, y1-5, f'Pothole ({label})', 
                color='white', fontsize=10, 
                bbox=dict(facecolor='red', alpha=0.8))
    
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def debug_batch_samples(dataset, output_dir='debug_samples', num_positive=10, num_negative=10):
    """Pozitif ve negatif örnekleri kaydet"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    positive_samples = []
    negative_samples = []
    
    for idx in range(len(dataset)):
        image, target = dataset[idx]
        img_name = dataset.images[idx]
        num_boxes = len(target['boxes'])
        
        if num_boxes > 0 and len(positive_samples) < num_positive:
            positive_samples.append((idx, img_name, image, target))
        elif num_boxes == 0 and len(negative_samples) < num_negative:
            negative_samples.append((idx, img_name, image, target))
        
        if len(positive_samples) >= num_positive and len(negative_samples) >= num_negative:
            break
    
    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)
    
    for idx, img_name, image, target in all_samples:
        if torch.is_tensor(image):
            image_np = image.numpy().transpose(1, 2, 0)
            image_np = image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            image_np = np.clip(image_np, 0, 1)
        else:
            image_np = image
        
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_np)
        
        boxes = target['boxes'].cpu().numpy() if torch.is_tensor(target['boxes']) else target['boxes']
        
        for box in boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        sample_type = "POS" if len(boxes) > 0 else "NEG"
        ax.set_title(f'{sample_type} - {img_name} - Boxes: {len(boxes)}')
        ax.axis('off')
        
        save_name = f"{sample_type}_{os.path.splitext(img_name)[0]}.png"
        plt.savefig(os.path.join(output_dir, save_name), bbox_inches='tight', dpi=100)
        plt.close()
        print(f"Saved: {save_name}")
    
    print(f"\n✓ {output_dir}/ klasörüne kaydedildi")

# Çalıştır:

def collate_fn(batch):
    """DataLoader için custom collate function"""
    return tuple(zip(*batch))


def clean_batch(images, targets, device):
    """Kutuları doğrula, ters/NaN kutuları at ve tensörleri cihaza taşı"""
    processed_images = []
    processed_targets = []
    
    for img, tgt in zip(images, targets):
        img = img.to(device)
        tgt = {k: v.to(device) if hasattr(v, 'to') else v for k, v in tgt.items()}
        boxes = tgt['boxes']
        
        if boxes.numel() > 0:
            finite_mask = torch.isfinite(boxes).all(dim=1)
            size_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            keep = finite_mask & size_mask
            
            if keep.any():
                tgt = tgt.copy()
                tgt['boxes'] = boxes[keep]
                if len(tgt['labels']) == len(boxes):
                    tgt['labels'] = tgt['labels'][keep]
                tgt['area'] = (tgt['boxes'][:, 3] - tgt['boxes'][:, 1]) * (tgt['boxes'][:, 2] - tgt['boxes'][:, 0])
            else:
                tgt = tgt.copy()
                tgt['boxes'] = boxes.new_zeros((0, 4))
                tgt['labels'] = torch.zeros((0,), dtype=torch.int64, device=boxes.device)
                tgt['area'] = boxes.new_zeros((0,))
        else:
            tgt = tgt.copy()
            tgt['boxes'] = boxes
            tgt['area'] = boxes.new_zeros((0,)) if hasattr(boxes, 'new_zeros') else torch.zeros((0,), device=device)
        
        processed_images.append(img)
        processed_targets.append(tgt)
    
    return processed_images, processed_targets


def create_faster_rcnn_optimized(num_classes=2, pretrained=True, img_size=640):
    """
    Çukur tespiti için optimize edilmiş Faster R-CNN
    """
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn
    )
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.rpn import RPNHead

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    anchor_sizes = (
        (8, 12, 16),
        (24, 32, 48),
        (64, 96, 128),
        (192, 256, 320),
        (384, 512, 768),
    )
    aspect_ratios = ((0.5, 1.0, 2.0, 3.0),) * len(anchor_sizes)

    model.rpn.anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )

    in_channels = model.backbone.out_channels
    num_anchors_per_location = model.rpn.anchor_generator.num_anchors_per_location()
    num_anchors = num_anchors_per_location[0]

    model.rpn.head = RPNHead(
        in_channels=in_channels,
        num_anchors=num_anchors
    )

    model.rpn.fg_iou_thresh = 0.5
    model.rpn.bg_iou_thresh = 0.3
    model.rpn.batch_size_per_image = 512
    model.rpn.positive_fraction = 0.5

    model.rpn.nms_thresh = 0.7
    model.rpn.score_thresh = 0.0
    model.rpn.pre_nms_top_n_train = 2000
    model.rpn.post_nms_top_n_train = 2000
    model.rpn.pre_nms_top_n_test = 1000
    model.rpn.post_nms_top_n_test = 1000

    model.roi_heads.fg_iou_thresh = 0.5
    model.roi_heads.bg_iou_thresh = 0.5
    model.roi_heads.batch_size_per_image = 512
    model.roi_heads.positive_fraction = 0.5
    model.roi_heads.nms_thresh = 0.4
    model.roi_heads.score_thresh = 0.05
    model.roi_heads.detections_per_img = 300

    print("\n" + "=" * 70)
    print("✓ FASTER R-CNN ÇUKUR-SPESİFİK KONFİGÜRASYON")
    print("=" * 70)
    print(f"Anchor Sizes: {anchor_sizes}")
    print(f"Aspect Ratios: {aspect_ratios[0]}")
    print(f"RPN Input Channels: {in_channels}")
    print(f"Anchors per Location: {num_anchors}")
    print("=" * 70 + "\n")

    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, warmup_epochs=5, base_lr = 0.001):
    """Warmup ve gradient clipping ile geliştirilmiş eğitim"""
    model.train()
    total_loss = 0
    num_steps = 0
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        if epoch <= warmup_epochs:
            warmup_progress = ((epoch - 1) * len(data_loader) + batch_idx) / (warmup_epochs * len(data_loader))
            lr = base_lr * warmup_progress
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        use_amp = scaler is not None and scaler.is_enabled()
        images, targets = clean_batch(images, targets, device)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            loss_dict = model(images, targets)
            
            if 'loss_classifier' in loss_dict:
                losses = loss_dict['loss_classifier'] * 2.0 + \
                        loss_dict['loss_box_reg'] * 1.0 + \
                        loss_dict['loss_objectness'] * 1.0 + \
                        loss_dict['loss_rpn_box_reg'] * 1.0
            else:
                losses = loss_dict.get('classification', 0) + loss_dict.get('bbox_regression', 0)
        
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += losses.item()
        num_steps += 1
        pbar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
    
    return total_loss / max(num_steps, 1)


def calculate_iou(box1, box2):
    """IoU (Intersection over Union) hesapla"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def evaluate_model(model, data_loader, device, iou_threshold=0.5, conf_threshold=0.25):
    """Model performansını değerlendir ve bilimsel metrikler hesapla"""
    model.eval()
    
    tp, fp, fn = 0, 0, 0
    total_iou = 0
    iou_count = 0
    
    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc='Evaluating')):
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for i, (output, target) in enumerate(zip(outputs, targets)):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()

                if len(pred_boxes) > 0:
                    import torchvision.ops as ops
                    nms_indices = ops.nms(
                        torch.tensor(pred_boxes),
                        torch.tensor(pred_scores),
                        iou_threshold=0.3
                    )
                    pred_boxes = pred_boxes[nms_indices]
                    pred_scores = pred_scores[nms_indices]
                
                mask = pred_scores >= conf_threshold
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]
                
                gt_boxes = target['boxes'].cpu().numpy()
                
                if len(gt_boxes) == 0:
                    fp += len(pred_boxes)
                    continue
                
                if len(pred_boxes) == 0:
                    fn += len(gt_boxes)
                    continue
                
                matched_pred = set()
                for gt_box in gt_boxes:
                    best_iou = 0
                    best_pred_idx = -1
                    
                    for idx, pred_box in enumerate(pred_boxes):
                        if idx in matched_pred:
                            continue
                        
                        iou = calculate_iou(gt_box, pred_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_idx = idx
                    
                    if best_iou >= iou_threshold:
                        tp += 1
                        matched_pred.add(best_pred_idx)
                        total_iou += best_iou
                        iou_count += 1
                    else:
                        fn += 1
                
                fp += len([i for i in range(len(pred_boxes)) if i not in matched_pred])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = total_iou / iou_count if iou_count > 0 else 0
    
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'mAP@0.5': precision,
        'Mean IoU': mean_iou,
        'True Positives': tp,
        'False Positives': fp,
        'False Negatives': fn,
        'Confidence Threshold': conf_threshold
    }
    
    return metrics


def print_metrics_for_paper(metrics):
    """Bilimsel makale için metrik çıktısı"""
    print("\n" + "="*60)
    print("BİLİMSEL MAKALE İÇİN PERFORMANS METRİKLERİ")
    print("="*60)
    print(f"\nPrecision (Kesinlik):        {metrics['Precision']:.4f}")
    print(f"Recall (Duyarlılık):         {metrics['Recall']:.4f}")
    print(f"F1-Score:                    {metrics['F1-Score']:.4f}")
    print(f"mAP@0.5:                     {metrics['mAP@0.5']:.4f}")
    print(f"Mean IoU:                    {metrics['Mean IoU']:.4f}")
    print(f"Confidence Threshold:        {metrics['Confidence Threshold']:.2f}")
    print(f"\nTrue Positives (TP):         {metrics['True Positives']}")
    print(f"False Positives (FP):        {metrics['False Positives']}")
    print(f"False Negatives (FN):        {metrics['False Negatives']}")
    print("="*60)


def parse_args():
    parser = argparse.ArgumentParser(description='Pothole tespiti için Faster R-CNN eğitimi')
    parser.add_argument('--train-img-dir', type=str, default=None)
    parser.add_argument('--train-label-dir', type=str, default=None)
    parser.add_argument('--val-img-dir', type=str, default=None)
    parser.add_argument('--val-label-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--conf-threshold', type=float, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--model', type=str, default=None, choices=['retinanet', 'fasterrcnn'])
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--img-size', type=int, default=640, help='Görüntü boyutu (anchor hesaplaması için)')
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig()
    
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.conf_threshold is not None:
        config.conf_threshold = args.conf_threshold
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.train_img_dir:
        config.train_img_dir = args.train_img_dir
    if args.train_label_dir:
        config.train_label_dir = args.train_label_dir
    if args.val_img_dir:
        config.val_img_dir = args.val_img_dir
    if args.val_label_dir:
        config.val_label_dir = args.val_label_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.no_amp:
        config.use_amp = False
    if args.no_pretrained:
        config.pretrained = False
    if args.model:
        config.model_name = args.model
    
    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    pin_memory = device.type == 'cuda'
    
    os.makedirs(config.output_dir, exist_ok=True)
    seed_everything(config.seed)
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('medium')
    
    amp_active = config.use_amp and device.type == "cuda"
    print(f'Using device: {device}')
    print(f'Pin memory: {pin_memory} | Workers: {config.num_workers} | AMP: {amp_active}')
    
    print('Loading datasets...')
    train_dataset = PotholeDataset(
        config.train_img_dir, 
        config.train_label_dir,
        transforms=get_augmentation_pipeline(train=True)
    )
    
    val_dataset = PotholeDataset(
        config.val_img_dir, 
        config.val_label_dir,
        transforms=get_augmentation_pipeline(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_fn
    )

    # Debug için
    debug_batch_samples(train_dataset, output_dir='debug_samples', num_positive=10, num_negative=10)
    debug_sample(train_dataset, idx=0, save_path='debug_train.png')
    debug_sample(val_dataset, idx=10, save_path='debug_val.png')
    
   
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    
    print('\nCreating model...')
    img_size = args.img_size if args else 640
    model = create_faster_rcnn_optimized(
        num_classes=config.num_classes,
        pretrained=config.pretrained,
        img_size=img_size
    )
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=config.learning_rate,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = param_group['lr']
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[35, 55],
        gamma=0.1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == 'cuda')
    
    print('\nStarting training...')
    print(f'Initial Confidence Threshold: {config.conf_threshold}')
    best_f1 = -float('inf')
    best_model_path = os.path.join(config.output_dir, 'best_pothole_model.pth')
    metrics_path = os.path.join(config.output_dir, 'metrics.json')
    
    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, 
            epoch, scaler=scaler, warmup_epochs=config.warmup_epochs, base_lr = config.learning_rate
        )
        
        if epoch > config.warmup_epochs:
            lr_scheduler.step()
        
        print(f'Epoch {epoch}/{config.num_epochs} - train loss: {train_loss:.4f} - LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if epoch % 3 == 0 or epoch == config.num_epochs:
            print(f'\nValidating epoch {epoch}...')
            metrics = evaluate_model(model, val_loader, device, conf_threshold=config.conf_threshold)
            print_metrics_for_paper(metrics)
            
            if metrics['F1-Score'] > best_f1:
                best_f1 = metrics['F1-Score']
                torch.save(model.state_dict(), best_model_path)
                print(f'✓ Best model saved! F1-Score: {best_f1:.4f}')
    
    print('\n\nFINAL EVALUATION:')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        final_metrics = evaluate_model(model, val_loader, device, conf_threshold=config.conf_threshold)
        print_metrics_for_paper(final_metrics)
        
        final_metrics_json = {}
        for k, v in final_metrics.items():
            if isinstance(v, (np.floating,)):
                final_metrics_json[k] = float(v)
            elif isinstance(v, (np.integer,)):
                final_metrics_json[k] = int(v)
            elif hasattr(v, 'item'):
                final_metrics_json[k] = v.item()
            else:
                final_metrics_json[k] = v
        
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics_json, f, indent=4)
        
        print('\n✓ Training completed!')
        print(f'✓ Model saved: {best_model_path}')
        print(f'✓ Metrics saved: {metrics_path}')
    else:
        print('Best model bulunamadı!')


if __name__ == '__main__':
    main()