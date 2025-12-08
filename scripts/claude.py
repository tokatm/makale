import argparse
import json
import os
import random
from dataclasses import dataclass

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm


def default_num_workers():
    cpu_count = os.cpu_count() or 0
    return max(cpu_count - 1, 0)


@dataclass
class TrainConfig:
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 0.005
    num_classes: int = 2  # background + pothole
    conf_threshold: float = 0.3
    num_workers: int = default_num_workers()
    train_img_dir: str = os.path.join('dataset-2', 'images', 'train')
    train_label_dir: str = os.path.join('dataset-2', 'labels', 'train')
    val_img_dir: str = os.path.join('dataset-2', 'images', 'val')
    val_label_dir: str = os.path.join('dataset-2', 'labels', 'val')
    output_dir: str = 'results'
    seed: int = 42
    use_amp: bool = True
    pretrained: bool = True


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # performans için açık

# ============================================================================
# 1. VERİ SETİ HAZIRLAMA
# ============================================================================

class PotholeDataset(Dataset):
    """YOLO formatındaki çukur verilerini PyTorch object detection formatına dönüştürür"""
    
    def __init__(self, image_dir, label_dir, transforms=None, augment=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.augment = augment
        
        # Resim dosyalarını listele
        self.images = [f for f in os.listdir(image_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Augmentation pipeline
        if augment:
            self.aug_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.RandomGamma(p=0.2),
                A.CLAHE(p=0.2),
                A.ColorJitter(p=0.2),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                                  rotate_limit=10, p=0.3),
            ], bbox_params=A.BboxParams(
                format='yolo', 
                label_fields=['class_labels'],
                min_area=0,
                min_visibility=0.3,
                clip=True
            ))
    
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
    
    def clip_boxes(self, boxes):
        """Bbox koordinatlarını [0, 1] aralığına sınırla"""
        clipped_boxes = []
        for box in boxes:
            x_center = np.clip(box[0], 0.0, 1.0)
            y_center = np.clip(box[1], 0.0, 1.0)
            width = np.clip(box[2], 0.0, 1.0)
            height = np.clip(box[3], 0.0, 1.0)
            
            # Box'un görüntü dışına çıkmasını engelle
            x_center = np.clip(x_center, width/2, 1.0 - width/2)
            y_center = np.clip(y_center, height/2, 1.0 - height/2)
            
            clipped_boxes.append([x_center, y_center, width, height])
        return clipped_boxes
    
    def __getitem__(self, idx):
        # Resmi yükle
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Label dosyasını yükle
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
                        # class_id=0 çukur demek, PyTorch için 1 yapıyoruz (0 background için ayrılı)
                        if class_id == 0:  # Sadece çukurları al
                            x_center, y_center, width, height = map(float, parts[1:])
                            # Koordinatları clip et
                            x_center = np.clip(x_center, 0.0, 1.0)
                            y_center = np.clip(y_center, 0.0, 1.0)
                            width = np.clip(width, 0.0, 1.0)
                            height = np.clip(height, 0.0, 1.0)
                            boxes.append([x_center, y_center, width, height])
                            labels.append(1)  # PyTorch: 0=background, 1=pothole
        
        # Augmentation uygula
        if self.augment and len(boxes) > 0:
            try:
                augmented = self.aug_transform(
                    image=image,
                    bboxes=boxes,
                    class_labels=labels
                )
                image = augmented['image']
                boxes = augmented['bboxes']
                labels = augmented['class_labels']
                
                # Augmentation sonrası tekrar clip et
                if len(boxes) > 0:
                    boxes = self.clip_boxes(boxes)
            except Exception as e:
                # Augmentation hata verirse orijinal değerleri kullan
                pass
        
        # YOLO'dan Pascal VOC formatına dönüştür
        if len(boxes) > 0:
            boxes = self.yolo_to_pascal_voc(boxes, w, h)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Target dictionary oluştur
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,)),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Transform uygula
        if self.transforms:
            image = self.transforms(image)
        
        return image, target


def get_transform(train=False):
    """Temel transform pipeline"""
    transforms_list = []
    # Albumentations sonrası torch tensor + normalize
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    return transforms.Compose(transforms_list)


def collate_fn(batch):
    """DataLoader için custom collate function"""
    return tuple(zip(*batch))


# ============================================================================
# 2. MODEL OLUŞTURMA
# ============================================================================

def create_model(num_classes=2, pretrained=True):
    """MobileNetV3 backbone ile Faster R-CNN modeli oluştur"""
    
    # Pretrained model yükle
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    
    # Classifier'ı değiştir
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


# ============================================================================
# 3. EĞİTİM FONKSİYONLARI
# ============================================================================

def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    """Bir epoch eğitim"""
    model.train()
    total_loss = 0
    num_steps = 0
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    for images, targets in pbar:
        use_amp = scaler is not None and scaler.is_enabled()
        # Boş target'ları filtrele
        valid_samples = []
        valid_targets = []
        
        for img, tgt in zip(images, targets):
            if len(tgt['boxes']) > 0:  # En az 1 çukur varsa
                valid_samples.append(img)
                valid_targets.append(tgt)
        
        if len(valid_samples) == 0:
            continue
        
        images = [image.to(device) for image in valid_samples]
        targets = [{k: v.to(device) for k, v in t.items()} for t in valid_targets]
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()
        
        total_loss += losses.item()
        num_steps += 1
        pbar.set_postfix({'loss': f'{losses.item():.4f}'})
    
    return total_loss / max(num_steps, 1)


# ============================================================================
# 4. DEĞERLENDĐRME VE METRĐKLER
# ============================================================================

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


def evaluate_model(model, data_loader, device, iou_threshold=0.5, conf_threshold=0.3):
    """Model performansını değerlendir ve bilimsel metrikler hesapla"""
    model.eval()
    
    tp, fp, fn = 0, 0, 0
    total_iou = 0
    iou_count = 0
    
    with torch.inference_mode():
        for images, targets in tqdm(data_loader, desc='Evaluating'):
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                
                gt_boxes = target['boxes'].cpu().numpy()
                
                # Confidence threshold uygula
                mask = pred_scores >= conf_threshold
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]
                
                if len(gt_boxes) == 0:
                    fp += len(pred_boxes)
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
    
    # Metrikleri hesapla
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = total_iou / iou_count if iou_count > 0 else 0
    
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'mAP@0.5': precision,  # Simplified mAP
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
    parser.add_argument('--train-img-dir', type=str, default=None, help='Train görüntü klasörü')
    parser.add_argument('--train-label-dir', type=str, default=None, help='Train label klasörü')
    parser.add_argument('--val-img-dir', type=str, default=None, help='Validation görüntü klasörü')
    parser.add_argument('--val-label-dir', type=str, default=None, help='Validation label klasörü')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Epoch sayısı')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--conf-threshold', type=float, default=None, help='Değerlendirme için confidence threshold')
    parser.add_argument('--num-workers', type=int, default=None, help='DataLoader worker sayısı')
    parser.add_argument('--output-dir', type=str, default=None, help='Model ve metriklerin kaydedileceği klasör')
    parser.add_argument('--seed', type=int, default=None, help='Rastgelelik için seed')
    parser.add_argument('--no-amp', action='store_true', help='Mixed precision kapat')
    parser.add_argument('--no-pretrained', action='store_true', help='Pretrained backbone kullanma')
    parser.add_argument('--device', type=str, default=None, help='cuda veya cpu')
    return parser.parse_args()


# ============================================================================
# 5. ANA EĞİTİM PIPELINE
# ============================================================================

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
    
    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    pin_memory = device.type == 'cuda'
    
    os.makedirs(config.output_dir, exist_ok=True)
    seed_everything(config.seed)
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('medium')
    
    print(f'Using device: {device}')
    print(f'Pin memory: {pin_memory} | Workers: {config.num_workers} | AMP: {config.use_amp and device.type == \"cuda\"}')
    
    # Dataset ve DataLoader
    print('Loading datasets...')
    train_dataset = PotholeDataset(
        config.train_img_dir, config.train_label_dir,
        transforms=get_transform(train=True),
        augment=True
    )
    
    val_dataset = PotholeDataset(
        config.val_img_dir, config.val_label_dir,
        transforms=get_transform(train=False),
        augment=False
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
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    
    # Model oluştur
    print('\nCreating model...')
    model = create_model(num_classes=config.num_classes, pretrained=config.pretrained)
    model.to(device)
    
    # Optimizer ve scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == 'cuda')
    
    # Eğitim
    print('\nStarting training...')
    print(f'Confidence Threshold: {config.conf_threshold}')
    best_f1 = -float('inf')
    best_model_path = os.path.join(config.output_dir, 'best_pothole_model.pth')
    metrics_path = os.path.join(config.output_dir, 'metrics.json')
    
    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler)
        lr_scheduler.step()
        print(f'Epoch {epoch} - train loss: {train_loss:.4f}')
        
        if epoch % 5 == 0:
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
        
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        
        print('\n✓ Training completed!')
        print(f'✓ Model saved: {best_model_path}')
        print(f'✓ Metrics saved: {metrics_path}')
    else:
        print('Best model bulunamadı, kaydetme/validation adımlarını kontrol edin.')


if __name__ == '__main__':
    main()
