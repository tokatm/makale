import torch
import numpy as np


def box_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes
    Args:
        boxes1: [N, 4] in format [xmin, ymin, xmax, ymax]
        boxes2: [M, 4] in format [xmin, ymin, xmax, ymax]
    Returns:
        iou: [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Union
    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou


def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    Args:
        boxes: [N, 4] predicted boxes
        scores: [N] confidence scores
        iou_threshold: IoU threshold for suppression
    Returns:
        keep: indices of boxes to keep
    """
    if boxes.size(0) == 0:
        return torch.tensor([], dtype=torch.long)

    # Sort by score
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    while sorted_indices.numel() > 0:
        # Take highest scoring box
        idx = sorted_indices[0]
        keep.append(idx.item())
        
        if sorted_indices.numel() == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[idx].unsqueeze(0)
        remaining_boxes = boxes[sorted_indices[1:]]
        
        ious = box_iou(current_box, remaining_boxes).squeeze(0)
        
        # Keep boxes with IoU less than threshold
        mask = ious < iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long)


def calculate_ap(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate Average Precision for single class
    Args:
        predictions: list of dicts with 'boxes' [N, 4] and 'scores' [N]
        ground_truths: list of dicts with 'boxes' [M, 4]
        iou_threshold: IoU threshold for considering detection as correct
    Returns:
        ap: Average Precision score
    """
    # Collect all predictions and ground truths
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_image_ids = []
    
    all_gt_boxes = []
    all_gt_image_ids = []
    
    for img_id, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        if pred['boxes'].size(0) > 0:
            all_pred_boxes.append(pred['boxes'])
            all_pred_scores.append(pred['scores'])
            all_pred_image_ids.extend([img_id] * pred['boxes'].size(0))
        
        if gt['boxes'].size(0) > 0:
            all_gt_boxes.append(gt['boxes'])
            all_gt_image_ids.extend([img_id] * gt['boxes'].size(0))
    
    if len(all_pred_boxes) == 0:
        return 0.0
    
    # Concatenate all predictions
    all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
    all_pred_scores = torch.cat(all_pred_scores, dim=0)
    all_pred_image_ids = torch.tensor(all_pred_image_ids)
    
    # Sort predictions by score (descending)
    sorted_indices = torch.argsort(all_pred_scores, descending=True)
    all_pred_boxes = all_pred_boxes[sorted_indices]
    all_pred_scores = all_pred_scores[sorted_indices]
    all_pred_image_ids = all_pred_image_ids[sorted_indices]
    
    # Track which ground truths have been matched
    num_gt = len(all_gt_image_ids)
    gt_matched = [False] * num_gt
    
    # Calculate TP and FP for each prediction
    tp = []
    fp = []
    
    for pred_idx in range(len(all_pred_boxes)):
        pred_box = all_pred_boxes[pred_idx]
        pred_img_id = all_pred_image_ids[pred_idx].item()
        
        # Find ground truths for this image
        gt_indices = [i for i, img_id in enumerate(all_gt_image_ids) if img_id == pred_img_id]
        
        if len(gt_indices) == 0:
            # No ground truth for this image -> False Positive
            tp.append(0)
            fp.append(1)
            continue
        
        # Get ground truth boxes for this image
        gt_boxes_img = []
        for idx in gt_indices:
            gt_box_idx = 0
            for i, (gt_img_id, gt_box_list) in enumerate(zip(all_gt_image_ids, 
                                                              [gt['boxes'] for gt in ground_truths])):
                if gt_img_id == pred_img_id:
                    if gt_box_idx < gt_box_list.size(0):
                        gt_boxes_img.append(gt_box_list[gt_box_idx])
                        gt_box_idx += 1
        
        if len(gt_boxes_img) == 0:
            tp.append(0)
            fp.append(1)
            continue
        
        gt_boxes_img = torch.stack(gt_boxes_img)
        
        # Calculate IoU with all ground truths in this image
        ious = box_iou(pred_box.unsqueeze(0), gt_boxes_img).squeeze(0)
        max_iou, max_idx = torch.max(ious, dim=0)
        
        # Check if this ground truth has already been matched
        global_gt_idx = gt_indices[max_idx.item()]
        
        if max_iou >= iou_threshold and not gt_matched[global_gt_idx]:
            # True Positive
            tp.append(1)
            fp.append(0)
            gt_matched[global_gt_idx] = True
        else:
            # False Positive
            tp.append(0)
            fp.append(1)
    
    # Calculate precision and recall
    tp = np.array(tp)
    fp = np.array(fp)
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / (num_gt + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # Add sentinel values at the end
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[1], precisions, [0]])
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate mean Average Precision
    For single class detection, mAP = AP
    """
    return calculate_ap(predictions, ground_truths, iou_threshold)


def calculate_precision_recall(predictions, ground_truths, iou_threshold=0.5, conf_threshold=0.5):
    """
    Calculate precision and recall at given thresholds
    """
    tp = 0
    fp = 0
    fn = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        gt_boxes = gt['boxes']
        
        # Filter by confidence
        mask = pred_scores >= conf_threshold
        pred_boxes = pred_boxes[mask]
        
        # Track matched GTs
        gt_matched = torch.zeros(gt_boxes.size(0), dtype=torch.bool)
        
        for pred_box in pred_boxes:
            if gt_boxes.size(0) == 0:
                fp += 1
                continue
            
            # Calculate IoU with all GT boxes
            ious = box_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
            max_iou, max_idx = torch.max(ious, dim=0)
            
            if max_iou >= iou_threshold and not gt_matched[max_idx]:
                tp += 1
                gt_matched[max_idx] = True
            else:
                fp += 1
        
        # Count unmatched ground truths
        fn += (~gt_matched).sum().item()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }