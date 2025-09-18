from typing import Dict

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

try:
    from cellpose import models
    from cellpose.metrics import average_precision
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False


def _safe_hausdorff(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Hausdorff distance, returning inf if a mask is empty."""
    if not np.any(mask1) or not np.any(mask2):
        return np.inf
    
    coords1 = np.argwhere(mask1)
    coords2 = np.argwhere(mask2)

    # Compute directed Hausdorff distances
    d1 = directed_hausdorff(coords1, coords2)[0]
    d2 = directed_hausdorff(coords2, coords1)[0]
    
    return max(d1, d2)


class DownstreamTasks:
    """Wrapper for downstream scientific task evaluations."""

    @staticmethod
    def cellpose_f1(pred_img: np.ndarray, gt_masks: np.ndarray) -> float:
        """
        Run Cellpose segmentation on a reconstructed image and compute F1 score.

        Args:
            pred_img: Reconstructed image to be segmented.
            gt_masks: Ground truth segmentation masks.

        Returns:
            F1 score (average precision at 0.5 IoU threshold).
        """
        if not CELLPOSE_AVAILABLE:
            raise ImportError("Cellpose is not installed. Please install it to run this evaluation.")

        # Use a pre-trained Cellpose model
        model = models.Cellpose(model_type='cyto')
        pred_masks, _, _, _ = model.eval([pred_img], diameter=None, channels=[0, 0])
        
        # Compute average precision (F1-score at IoU 0.5)
        ap, _, _, _ = average_precision(gt_masks, pred_masks[0])
        
        return float(ap[0, 5])  # IoU threshold 0.5

    @staticmethod
    def hausdorff_distance(pred_masks: np.ndarray, gt_masks: np.ndarray) -> float:
        """
        Compute the Hausdorff distance between predicted and ground truth masks.

        Args:
            pred_masks: Predicted segmentation masks.
            gt_masks: Ground truth segmentation masks.

        Returns:
            Mean Hausdorff distance over all corresponding mask pairs.
        """
        distances = []
        
        # Find matched pairs of masks
        pred_labels = np.unique(pred_masks)
        gt_labels = np.unique(gt_masks)

        # Iterate over ground truth masks and find best matching predicted mask
        for gt_label in tqdm(gt_labels[1:], desc="Computing Hausdorff distances", leave=False):  # Skip background
            gt_mask = (gt_masks == gt_label)
            
            best_match_dist = np.inf
            
            for pred_label in pred_labels[1:]:
                pred_mask = (pred_masks == pred_label)
                
                # Check for overlap
                if np.any(gt_mask & pred_mask):
                    dist = _safe_hausdorff(pred_mask, gt_mask)
                    if dist < best_match_dist:
                        best_match_dist = dist
            
            if best_match_dist != np.inf:
                distances.append(best_match_dist)

        return np.mean(distances) if distances else np.inf
