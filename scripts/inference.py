import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import yaml
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from transforms import get_inference_transforms, get_train_transforms
from metrics import apply_nms
from models.mobilenet_detector import MobileNetDetector
"""from models.convnext_detector import ConvNeXtTinyDetector
from models.swin_detector import SwinTinyDetector"""


def load_model(config_path, checkpoint_path, device):
    """Load model from checkpoint"""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Initialize model
    model_name = cfg["model"]["name"]
    if model_name == "mobilenet":
        model = MobileNetDetector()
    elif model_name == "convnext":
        model = ConvNeXtTinyDetector()
    elif model_name == "swin_tiny":
        model = SwinTinyDetector()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, cfg


def predict_image(model, image, transforms, device, conf_threshold=0.5, nms_threshold=0.5):
    """
    Run inference on a single image
    
    Args:
        model: trained detection model
        image: numpy array (H, W, 3) in RGB
        transforms: albumentations transforms
        device: torch device
        conf_threshold: confidence threshold for detections
        nms_threshold: IoU threshold for NMS
    
    Returns:
        boxes: [N, 4] detected boxes in original image coordinates
        scores: [N] confidence scores
    """
    orig_h, orig_w = image.shape[:2]
    
    # Apply transforms
    transformed = transforms(image=image)
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        cls_logits = outputs["cls_logits"]  # [1, N, 1]
        bbox_reg = outputs["bbox_reg"]      # [1, N, 4]
    
    # Convert to probabilities
    scores = torch.sigmoid(cls_logits).squeeze()  # [N]
    boxes = bbox_reg.squeeze(0)  # [N, 4]
    
    # Filter by confidence threshold
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    
    if boxes.size(0) == 0:
        return np.array([]), np.array([])
    
    # Apply NMS
    keep_indices = apply_nms(boxes, scores, iou_threshold=nms_threshold)
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    
    # Boxes are already in pixel coordinates relative to resized image
    # Scale to original image size
    img_h, img_w = img_tensor.shape[2:]
    scale_x = orig_w / img_w
    scale_y = orig_h / img_h
    
    boxes[:, 0] *= scale_x  # xmin
    boxes[:, 1] *= scale_y  # ymin
    boxes[:, 2] *= scale_x  # xmax
    boxes[:, 3] *= scale_y  # ymax
    
    # Clamp to image boundaries
    boxes[:, 0] = torch.clamp(boxes[:, 0], 0, orig_w)
    boxes[:, 1] = torch.clamp(boxes[:, 1], 0, orig_h)
    boxes[:, 2] = torch.clamp(boxes[:, 2], 0, orig_w)
    boxes[:, 3] = torch.clamp(boxes[:, 3], 0, orig_h)
    
    return boxes.cpu().numpy(), scores.cpu().numpy()


def draw_predictions(image, boxes, scores, class_names=['pothole'], color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on image
    
    Args:
        image: numpy array (H, W, 3)
        boxes: [N, 4] array of boxes
        scores: [N] array of confidence scores
        class_names: list of class names
        color: BGR color tuple
        thickness: line thickness
    
    Returns:
        image with drawn boxes
    """
    img_draw = image.copy()
    
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw rectangle
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"{class_names[0]}: {score:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background
        cv2.rectangle(
            img_draw,
            (x1, y1 - label_size[1] - 4),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img_draw,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return img_draw


def process_image(image_path, model, transforms, cfg, device, output_dir, 
                 conf_threshold=0.5, nms_threshold=0.5, save_viz=True):
    """Process a single image"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  ❌ Failed to read image: {image_path}")
        return np.array([]), np.array([])
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    # Run prediction
    boxes, scores = predict_image(
        model, image_rgb, transforms, device,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold
    )
    
    if len(boxes) > 0:
        print(f"  ✓ Detected {len(boxes)} pothole(s)")
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            print(f"    {i+1}. Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] (w={w:.0f}, h={h:.0f}) | Conf: {score:.3f}")
    else:
        print(f"  ⚠ No detections")
    
    # Save visualization
    if save_viz:
        if len(boxes) > 0:
            img_viz = draw_predictions(image, boxes, scores)
        else:
            img_viz = image
        
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), img_viz)
        if len(boxes) > 0:
            print(f"  💾 Saved to: {output_path}")
    
    return boxes, scores


def process_directory(input_dir, model, transforms, cfg, device, output_dir,
                     conf_threshold=0.5, nms_threshold=0.5):
    """Process all images in a directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(f'*{ext}')))
        image_files.extend(list(input_dir.glob(f'*{ext.upper()}')))
    
    print(f"\n{'='*60}")
    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"{'='*60}\n")
    
    if len(image_files) == 0:
        print("No images found! Check the directory path.")
        return
    
    total_detections = 0
    images_with_detections = 0
    
    for i, img_path in enumerate(sorted(image_files), 1):
        print(f"[{i}/{len(image_files)}] Processing: {img_path.name}")
        boxes, scores = process_image(
            img_path, model, transforms, cfg, device, output_dir,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            save_viz=True
        )
        total_detections += len(boxes)
        if len(boxes) > 0:
            images_with_detections += 1
        print()
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/len(image_files):.2f}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Run inference on pothole detection model')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--input', required=True, help='Path to input image or directory')
    parser.add_argument('--output', default='results', help='Output directory for visualizations')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, cfg = load_model(args.config, args.checkpoint, device)
    print("Model loaded successfully!")
    
    # Get transforms
    transforms = get_inference_transforms(cfg)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        print(f"\nProcessing single image: {input_path}")
        process_image(
            input_path, model, transforms, cfg, device, output_dir,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold
        )
    elif input_path.is_dir():
        # Directory of images
        print(f"\nProcessing directory: {input_path}")
        process_directory(
            input_path, model, transforms, cfg, device, output_dir,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold
        )
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return
    
    print(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()