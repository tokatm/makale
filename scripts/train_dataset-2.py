import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import YOLODataset
from transforms import get_train_transforms, get_val_transforms
from metrics import box_iou

# MODELS
from models.mobilenet_detector import MobileNetDetector
from models.convnext_detector import ConvNeXtTinyDetector
from models.swin_detector import SwinTinyDetector


def collate_fn(batch):
    imgs = []
    targets = []
    for img, tgt in batch:
        imgs.append(img)
        targets.append(tgt)
    imgs = torch.stack(imgs, dim=0)
    return imgs, targets


# -----------------------------------------------------------
# Loss Functions
# -----------------------------------------------------------
def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary classification
    inputs: [B*N] - raw logits
    targets: [B*N] - binary labels (0 or 1)
    """
    prob = torch.sigmoid(inputs)
    pt = torch.where(targets == 1, prob, 1 - prob)
    pt = torch.clamp(pt, min=1e-6, max=1.0)
    
    focal_weight = (1 - pt) ** gamma
    alpha_weight = torch.where(targets == 1, alpha, 1 - alpha)
    
    loss = -alpha_weight * focal_weight * torch.log(pt)
    return loss.mean()


def giou_loss(pred_boxes, gt_boxes):
    """
    Generalized IoU Loss
    pred_boxes, gt_boxes: [N, 4] in format [xmin, ymin, xmax, ymax]
    """
    if pred_boxes.size(0) == 0:
        return torch.tensor(0.0, device=pred_boxes.device)

    # Ensure boxes are valid
    pred_boxes = pred_boxes.clamp(min=0)
    gt_boxes = gt_boxes.clamp(min=0)

    # IoU calculation
    inter_xmin = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    inter_ymin = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    inter_xmax = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    inter_ymax = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])

    inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_area = inter_w * inter_h

    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_pred + area_gt - inter_area + 1e-6

    iou = inter_area / union

    # Enclosing box
    enc_xmin = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
    enc_ymin = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
    enc_xmax = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
    enc_ymax = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])
    
    enc_area = (enc_xmax - enc_xmin) * (enc_ymax - enc_ymin) + 1e-6
    giou = iou - (enc_area - union) / enc_area
    loss = 1 - giou
    
    return loss.mean()


# -----------------------------------------------------------
# Simple IoU-based Matching (FIXED)
# -----------------------------------------------------------
def match_predictions_simple(cls_logits, bbox_reg, targets, iou_threshold=0.5):
    """
    Simple IoU-based matching between predictions and ground truths
    
    Args:
        cls_logits: [B, N, 1] - classification logits
        bbox_reg: [B, N, 4] - predicted boxes [xmin, ymin, xmax, ymax]
        targets: list of dicts with 'boxes' [M, 4] and 'labels' [M]
        iou_threshold: IoU threshold for positive assignment
    
    Returns:
        cls_targets: [B, N] - binary classification targets
        reg_targets: [B, N, 4] - regression targets
        pos_mask: [B, N] - mask for positive samples
        neg_mask: [B, N] - mask for negative samples
    """
    B, N, _ = cls_logits.shape
    device = cls_logits.device
    
    cls_targets = torch.zeros((B, N), dtype=torch.float32, device=device)
    reg_targets = torch.zeros((B, N, 4), dtype=torch.float32, device=device)
    pos_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    neg_mask = torch.ones((B, N), dtype=torch.bool, device=device)
    
    for b in range(B):
        pred_boxes = bbox_reg[b]  # [N, 4]
        gt_boxes = targets[b]["boxes"]  # [M, 4]
        
        if gt_boxes.size(0) == 0:
            # No ground truth - all negative
            continue
        
        # Calculate IoU matrix [N, M]
        iou_matrix = box_iou(pred_boxes, gt_boxes)
        
        # For each prediction, find best matching GT
        max_iou, gt_idx = torch.max(iou_matrix, dim=1)  # [N]
        
        # Positive samples: IoU >= threshold
        pos_indices = max_iou >= iou_threshold
        
        if pos_indices.sum() > 0:
            # Assign positive samples
            cls_targets[b, pos_indices] = 1.0
            reg_targets[b, pos_indices] = gt_boxes[gt_idx[pos_indices]]
            pos_mask[b, pos_indices] = True
            neg_mask[b, pos_indices] = False
        
        # Also ensure each GT has at least one positive prediction
        for gt_i in range(gt_boxes.size(0)):
            # Find prediction with highest IoU for this GT
            iou_for_gt = iou_matrix[:, gt_i]
            best_pred_idx = torch.argmax(iou_for_gt)
            
            if iou_for_gt[best_pred_idx] > 0.1:  # At least some overlap
                cls_targets[b, best_pred_idx] = 1.0
                reg_targets[b, best_pred_idx] = gt_boxes[gt_i]
                pos_mask[b, best_pred_idx] = True
                neg_mask[b, best_pred_idx] = False
    
    return cls_targets, reg_targets, pos_mask, neg_mask


# -----------------------------------------------------------
# Model Selector
# -----------------------------------------------------------
def load_model(name):
    if name == "mobilenet":
        return MobileNetDetector()
    elif name == "convnext":
        return ConvNeXtTinyDetector()
    elif name == "swin_tiny":
        return SwinTinyDetector()
    else:
        raise ValueError(f"Unknown model {name}")


# -----------------------------------------------------------
# Train One Epoch
# -----------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, cfg):
    model.train()
    total_losses = []
    cls_losses = []
    reg_losses = []
    num_pos_samples = []
    
    cls_weight = cfg["train"].get("cls_weight", 1.0)
    reg_weight = cfg["train"].get("reg_weight", 2.0)

    pbar = tqdm(loader, desc="Training")
    for imgs, targets in pbar:
        imgs = imgs.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(imgs)
        cls_logits = outputs["cls_logits"]  # [B, N, 1]
        bbox_reg = outputs["bbox_reg"]      # [B, N, 4]

        # Match predictions to ground truths
        cls_targets, reg_targets, pos_mask, neg_mask = match_predictions_simple(
            cls_logits, bbox_reg, targets, iou_threshold=0.5
        )

        # Classification loss (all samples)
        B, N = cls_targets.shape
        cls_logits_flat = cls_logits.view(B * N)
        cls_targets_flat = cls_targets.view(B * N)
        
        loss_cls = sigmoid_focal_loss(cls_logits_flat, cls_targets_flat)
        
        # Regression loss (only positive samples)
        pos_mask_flat = pos_mask.view(B * N)
        num_pos = pos_mask_flat.sum().item()
        
        if num_pos > 0:
            pred_boxes_pos = bbox_reg.view(B * N, 4)[pos_mask_flat]
            target_boxes_pos = reg_targets.view(B * N, 4)[pos_mask_flat]
            loss_reg = giou_loss(pred_boxes_pos, target_boxes_pos)
        else:
            loss_reg = torch.tensor(0.0, device=device)

        # Total loss
        loss = cls_weight * loss_cls + reg_weight * loss_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_losses.append(loss.item())
        cls_losses.append(loss_cls.item())
        reg_losses.append(loss_reg.item() if isinstance(loss_reg, torch.Tensor) else 0.0)
        num_pos_samples.append(num_pos)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{loss_cls.item():.4f}',
            'reg': f'{loss_reg.item() if isinstance(loss_reg, torch.Tensor) else 0:.4f}',
            'pos': num_pos
        })

    avg_pos = sum(num_pos_samples) / len(num_pos_samples)
    print(f"  Avg positive samples per batch: {avg_pos:.1f}")
    
    return {
        'total': sum(total_losses) / len(total_losses),
        'cls': sum(cls_losses) / len(cls_losses),
        'reg': sum(reg_losses) / len(reg_losses)
    }


# -----------------------------------------------------------
# Validation Step
# -----------------------------------------------------------
def validate(model, loader, device, cfg):
    model.eval()
    total_losses = []
    cls_losses = []
    reg_losses = []
    
    cls_weight = cfg["train"].get("cls_weight", 1.0)
    reg_weight = cfg["train"].get("reg_weight", 2.0)

    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Validating"):
            imgs = imgs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(imgs)
            cls_logits = outputs["cls_logits"]
            bbox_reg = outputs["bbox_reg"]

            cls_targets, reg_targets, pos_mask, neg_mask = match_predictions_simple(
                cls_logits, bbox_reg, targets, iou_threshold=0.5
            )

            # Classification loss
            B, N = cls_targets.shape
            cls_logits_flat = cls_logits.view(B * N)
            cls_targets_flat = cls_targets.view(B * N)
            loss_cls = sigmoid_focal_loss(cls_logits_flat, cls_targets_flat)
            
            # Regression loss
            pos_mask_flat = pos_mask.view(B * N)
            num_pos = pos_mask_flat.sum().item()
            
            if num_pos > 0:
                pred_boxes_pos = bbox_reg.view(B * N, 4)[pos_mask_flat]
                target_boxes_pos = reg_targets.view(B * N, 4)[pos_mask_flat]
                loss_reg = giou_loss(pred_boxes_pos, target_boxes_pos)
            else:
                loss_reg = torch.tensor(0.0, device=device)

            loss = cls_weight * loss_cls + reg_weight * loss_reg

            total_losses.append(loss.item())
            cls_losses.append(loss_cls.item())
            reg_losses.append(loss_reg.item() if isinstance(loss_reg, torch.Tensor) else 0.0)

    return {
        'total': sum(total_losses) / len(total_losses),
        'cls': sum(cls_losses) / len(cls_losses),
        'reg': sum(reg_losses) / len(reg_losses)
    }


# -----------------------------------------------------------
# Main Training
# -----------------------------------------------------------
def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TRANSFORMS
    train_tf = get_train_transforms(cfg)
    val_tf = get_val_transforms(cfg)

    # PATH SETUP
    root = cfg["data"]["root"]
    train_img_dir = os.path.join(root, cfg["data"]["train_images"])
    train_lbl_dir = os.path.join(root, cfg["data"]["train_labels"])
    val_img_dir = os.path.join(root, cfg["data"]["val_images"])
    val_lbl_dir = os.path.join(root, cfg["data"]["val_labels"])

    # DATASETS
    train_dataset = YOLODataset(train_img_dir, train_lbl_dir, cfg["data"]["img_size"], train_tf)
    val_dataset = YOLODataset(val_img_dir, val_lbl_dir, cfg["data"]["img_size"], val_tf)

    print(f"\nDataset Statistics:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg["data"]["batch_size"], 
        shuffle=True,
        num_workers=4, 
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg["data"]["batch_size"], 
        shuffle=False,
        num_workers=4, 
        collate_fn=collate_fn,
        pin_memory=True
    )

    # MODEL
    model = load_model(cfg["model"]["name"]).to(device)
    print(f"\nModel: {cfg['model']['name']}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # OPTIMIZER & SCHEDULER
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg["train"]["lr"], 
        weight_decay=cfg["train"]["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=cfg["train"]["epochs"],
        eta_min=cfg["train"]["lr"] * 0.01
    )

    # EXPERIMENT DIRECTORY
    exp_dir = os.path.join("experiments", cfg["save"]["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)
    
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    best_loss = float("inf")
    train_losses = []
    val_losses = []
    train_cls_losses = []
    train_reg_losses = []

    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")

    # TRAINING LOOP
    for epoch in range(cfg["train"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['train']['epochs']}")
        print("-" * 60)

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg)
        
        # Validate
        val_metrics = validate(model, val_loader, device, cfg)

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Store metrics
        train_losses.append(train_metrics['total'])
        val_losses.append(val_metrics['total'])
        train_cls_losses.append(train_metrics['cls'])
        train_reg_losses.append(train_metrics['reg'])

        # Print metrics
        print(f"\nResults:")
        print(f"  Train - Total: {train_metrics['total']:.4f} | Cls: {train_metrics['cls']:.4f} | Reg: {train_metrics['reg']:.4f}")
        print(f"  Val   - Total: {val_metrics['total']:.4f} | Cls: {val_metrics['cls']:.4f} | Reg: {val_metrics['reg']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_metrics['total'] < best_loss:
            best_loss = val_metrics['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_metrics['total'],
            }, os.path.join(exp_dir, "best_model.pth"))
            print("  ✓ Best model saved!")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(exp_dir, f"checkpoint_epoch_{epoch+1}.pth"))

    # PLOT TRAINING CURVES
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Total loss
    axes[0].plot(train_losses, label="Train Loss", linewidth=2)
    axes[0].plot(val_losses, label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Total Loss", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Component losses
    axes[1].plot(train_cls_losses, label="Classification", linewidth=2)
    axes[1].plot(train_reg_losses, label="Regression", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].set_title("Training Loss Components", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "training_curves.png"), dpi=150)
    plt.close()

    print(f"\n{'='*60}")
    print("Training Completed!")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    args = parser.parse_args()
    main(args.config)