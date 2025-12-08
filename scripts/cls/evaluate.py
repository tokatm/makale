"""
evaluate.py - Model Performance Evaluation Script
Eğitilmiş modeli val/test setinde değerlendirir ve detaylı metrikler üretir.
"""

import os
import yaml
import argparse
import glob
import csv
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Metrik hesaplama
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# Görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns


class PotholeDataset(Dataset):
    """Train.py ile aynı dataset sınıfı"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # Label belirleme (prefix'e göre)
        if img_name.startswith('positive'):
            label = 1
        elif img_name.startswith('negative'):
            label = 0
        else:
            raise ValueError(f"Etiket adı sınıflandırılamadı: {img_name}")
        
        img = Image.open(img_path).convert('RGB')
        img = np.array(img).astype(np.float32)
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        return img, label


def load_config(config_path):
    """YAML config dosyasını yükle"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def find_best_checkpoint(search_dirs=['checkpoints', 'saved_models']):
    """En iyi checkpoint dosyasını otomatik bul"""
    best_ckpt = None
    best_loss = float('inf')
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        
        # *best* içeren .pth dosyalarını ara
        pattern = os.path.join(search_dir, '*best*.pth')
        checkpoints = glob.glob(pattern)
        
        for ckpt_path in checkpoints:
            # Dosya adından loss bilgisi çıkar (örn: loss0.1234)
            try:
                basename = os.path.basename(ckpt_path)
                if 'loss' in basename.lower():
                    # loss0.1234 veya valLoss_0.1234 formatlarını destekle
                    loss_str = basename.lower().split('loss')[1].split('_')[0].split('.pth')[0]
                    loss_str = loss_str.replace('_', '.')
                    loss = float(loss_str)
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_ckpt = ckpt_path
            except:
                continue
    
    return best_ckpt


def load_model(cfg, checkpoint_path, device):
    """Model yükle ve checkpoint'i restore et"""
    model_name = cfg['model']['name']
    num_classes = cfg['model']['num_classes']
    
    # Model import (dinamik)
    if model_name.lower() == 'mobilenetv3':
        from models.mobilenetv3 import MobileNetV3
        model = MobileNetV3(num_classes=num_classes)
    elif model_name.lower() == 'convnext':
        from models.convnext import ConvNeXt
        model = ConvNeXt(num_classes=num_classes)
    elif model_name.lower() in ['swin', 'swin_tiny']:
        from models.swin_tiny import SwinTiny
        model = SwinTiny(num_classes=num_classes)
    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}")
    
    # Checkpoint yükle
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # State dict formatını kontrol et
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"✅ Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        print(f"✅ Val Acc: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    else:
        # Sadece model weights
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def create_dataloader(cfg, split='val', batch_size=None):
    """DataLoader oluştur"""
    img_size = tuple(cfg['data']['img_size'])
    dataset_root = cfg['data']['root']
    
    # Split dizinini belirle
    split_dir = os.path.join(dataset_root, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split dizini bulunamadı: {split_dir}")
    
    # Transform (sadece inference - augmentation yok)
    transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # Dataset ve DataLoader
    dataset = PotholeDataset(root_dir=split_dir, transform=transform)
    
    if batch_size is None:
        batch_size = cfg['data']['batch_size']
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg['data'].get('num_workers', 2),
        pin_memory=False
    )
    
    print(f"✅ {split.upper()} set: {len(dataset)} samples, {len(loader)} batches")
    
    return loader


def evaluate_model(model, dataloader, device):
    """Model değerlendirme - predictions ve ground truth topla"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Softmax probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Predictions
            _, preds = torch.max(outputs, dim=1)
            
            # Listeye ekle
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Positive class prob
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def calculate_metrics(y_true, y_pred, y_prob):
    """Tüm metrikleri hesapla"""
    metrics = {}
    
    # Temel metrikler
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC-AUC (sadece binary classification için)
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    except:
        metrics['roc_auc'] = 0.0
    
    # Confusion Matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def print_metrics(metrics):
    """Metrikleri terminale yazdır"""
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("="*60 + "\n")


def save_metrics_csv(metrics, output_path, model_name, split):
    """Metrikleri CSV'ye kaydet"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # CSV header ve data
    fieldnames = [
        'timestamp', 'model', 'split', 'accuracy', 'precision', 
        'recall', 'f1_score', 'roc_auc'
    ]
    
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_name,
        'split': split,
        'accuracy': f"{metrics['accuracy']:.4f}",
        'precision': f"{metrics['precision']:.4f}",
        'recall': f"{metrics['recall']:.4f}",
        'f1_score': f"{metrics['f1_score']:.4f}",
        'roc_auc': f"{metrics['roc_auc']:.4f}"
    }
    
    # CSV'ye yaz (append mode)
    file_exists = os.path.exists(output_path)
    
    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)
    
    print(f"✅ Metrics saved to: {output_path}")


def plot_confusion_matrix(cm, output_path, class_names=['Negative', 'Positive']):
    """Confusion Matrix görselleştir"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix saved to: {output_path}")


def plot_roc_curve(y_true, y_prob, output_path, roc_auc):
    """ROC Curve çiz"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ ROC curve saved to: {output_path}")


def main():
    # CLI Argümanları
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    parser.add_argument('--config', type=str, default='configs/mobilenetv3.yaml',
                        help='Config YAML dosya yolu')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint yolu (opsiyonel - otomatik bulunur)')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='Değerlendirme split (val veya test)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size override (opsiyonel)')
    
    args = parser.parse_args()
    
    # Config yükle
    print(f"📋 Loading config: {args.config}")
    cfg = load_config(args.config)
    model_name = cfg['model']['name']
    
    # Device belirleme
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"🚀 Using device: Apple MPS")
    else:
        device = torch.device('cpu')
        print(f"🚀 Using device: CPU")
    
    # Checkpoint bul veya yükle
    if args.checkpoint is None:
        print("\n🔍 Searching for best checkpoint...")
        checkpoint_path = find_best_checkpoint()
        
        if checkpoint_path is None:
            print("❌ Checkpoint bulunamadı! Manuel olarak --checkpoint ile belirtin.")
            return
        
        print(f"✅ Found checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint dosyası bulunamadı: {checkpoint_path}")
            return
    
    # Model yükle
    print(f"\n🤖 Loading model: {model_name}")
    model = load_model(cfg, checkpoint_path, device)
    
    # DataLoader oluştur
    print(f"\n📦 Loading {args.split} dataset...")
    dataloader = create_dataloader(cfg, split=args.split, batch_size=args.batch_size)
    
    # Değerlendirme
    print(f"\n🔬 Evaluating on {args.split} set...")
    y_true, y_pred, y_prob = evaluate_model(model, dataloader, device)
    
    # Metrik hesaplama
    print("📊 Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Sonuçları yazdır
    print_metrics(metrics)
    
    # Sonuçları kaydet
    results_dir = 'results'
    
    # 1. CSV kaydet
    csv_path = os.path.join(results_dir, 'metrics.csv')
    save_metrics_csv(metrics, csv_path, model_name, args.split)
    
    # 2. Confusion Matrix kaydet
    cm_path = os.path.join(results_dir, f'confusion_matrix_{args.split}.png')
    plot_confusion_matrix(metrics['confusion_matrix'], cm_path)
    
    # 3. ROC Curve kaydet
    roc_path = os.path.join(results_dir, f'roc_curve_{args.split}.png')
    plot_roc_curve(y_true, y_prob, roc_path, metrics['roc_auc'])
    
    print("\n✅ Evaluation completed successfully!")
    print(f"📁 Results saved to: {results_dir}/")


if __name__ == '__main__':
    main()