"""
Downstream Task Metrics

Metrics for evaluating performance on downstream tasks like segmentation,
object detection, etc.
"""

import numpy as np
from typing import Optional
from .registry import register_metric

# Import the existing downstream task evaluator
from pkl_dg.evaluation.tasks import DownstreamTasks


@register_metric(
    name="cellpose_f1",
    category="downstream",
    description="F1 score using Cellpose segmentation",
    requires_reference=False,
    requires_input=False
)
def cellpose_f1_metric(pred: np.ndarray, gt_masks: Optional[np.ndarray] = None, **kwargs) -> float:
    """Compute Cellpose F1 score."""
    if gt_masks is None:
        return 0.0
    
    try:
        return DownstreamTasks.cellpose_f1(pred, gt_masks)
    except Exception:
        return 0.0


@register_metric(
    name="hausdorff_distance", 
    category="downstream",
    description="Hausdorff distance between segmentation masks",
    requires_reference=False,
    requires_input=False
)
def hausdorff_distance_metric(pred: np.ndarray, gt_masks: Optional[np.ndarray] = None, **kwargs) -> float:
    """Compute Hausdorff distance between predicted and ground truth masks."""
    if gt_masks is None:
        return float('inf')
    
    try:
        # Run Cellpose to get predicted masks
        from cellpose import models
        model = models.Cellpose(model_type='cyto')
        pred_masks, _, _, _ = model.eval([pred], diameter=None, channels=[0, 0])
        
        return DownstreamTasks.hausdorff_distance(pred_masks[0], gt_masks)
    except Exception:
        return float('inf')


@register_metric(
    name="dice_coefficient",
    category="downstream", 
    description="Dice coefficient for segmentation overlap",
    requires_reference=False,
    requires_input=False
)
def dice_coefficient_metric(pred: np.ndarray, gt_masks: Optional[np.ndarray] = None, **kwargs) -> float:
    """Compute Dice coefficient between predicted and ground truth masks."""
    if gt_masks is None:
        return 0.0
    
    try:
        # Run Cellpose to get predicted masks
        from cellpose import models
        model = models.Cellpose(model_type='cyto')
        pred_masks, _, _, _ = model.eval([pred], diameter=None, channels=[0, 0])
        
        pred_binary = pred_masks[0] > 0
        gt_binary = gt_masks > 0
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        total = pred_binary.sum() + gt_binary.sum()
        
        if total == 0:
            return 1.0  # Both masks are empty
        
        dice = 2.0 * intersection / total
        return float(dice)
        
    except Exception:
        return 0.0


@register_metric(
    name="iou",
    category="downstream",
    description="Intersection over Union for segmentation",
    requires_reference=False,
    requires_input=False
)
def iou_metric(pred: np.ndarray, gt_masks: Optional[np.ndarray] = None, **kwargs) -> float:
    """Compute IoU between predicted and ground truth masks."""
    if gt_masks is None:
        return 0.0
    
    try:
        # Run Cellpose to get predicted masks
        from cellpose import models
        model = models.Cellpose(model_type='cyto')
        pred_masks, _, _, _ = model.eval([pred], diameter=None, channels=[0, 0])
        
        pred_binary = pred_masks[0] > 0
        gt_binary = gt_masks > 0
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        if union == 0:
            return 1.0  # Both masks are empty
        
        iou = intersection / union
        return float(iou)
        
    except Exception:
        return 0.0


def compute_downstream_metrics(
    pred: np.ndarray, 
    gt_masks: Optional[np.ndarray] = None
) -> dict:
    """
    Compute all downstream task metrics.
    
    This is a convenience function that maintains compatibility with 
    the existing evaluation scripts.
    """
    if gt_masks is None:
        return {
            "cellpose_f1": 0.0,
            "hausdorff_distance": float('inf'),
            "dice_coefficient": 0.0,
            "iou": 0.0
        }
    
    return {
        "cellpose_f1": cellpose_f1_metric(pred, gt_masks=gt_masks),
        "hausdorff_distance": hausdorff_distance_metric(pred, gt_masks=gt_masks),
        "dice_coefficient": dice_coefficient_metric(pred, gt_masks=gt_masks),
        "iou": iou_metric(pred, gt_masks=gt_masks)
    }
