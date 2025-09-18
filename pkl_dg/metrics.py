"""
Unified Metrics Module

This module consolidates all metric computation functions to avoid duplication.
Provides a single interface for computing image quality metrics (PSNR, SSIM, FRC)
with consistent data range handling.
"""

import numpy as np
from typing import Dict, List, Optional, Union

# Import the main Metrics class from evaluation
try:
    from .evaluation import Metrics
except ImportError:
    # Fallback in case of import issues
    from pkl_dg.evaluation import Metrics


def compute_standard_metrics(
    pred: np.ndarray, 
    target: np.ndarray, 
    data_range: Optional[float] = None
) -> Dict[str, float]:
    """Compute standard image quality metrics (PSNR, SSIM, FRC).
    
    This is the main function that should be used across the codebase
    instead of the various compute_*_metrics functions.
    
    Args:
        pred: Predicted/reconstructed image
        target: Ground truth target image
        data_range: Dynamic range of the images. If None, computed from target.
        
    Returns:
        Dictionary with metric names and values
    """
    try:
        results = {
            "psnr": Metrics.psnr(pred, target, data_range=data_range),
            "ssim": Metrics.ssim(pred, target, data_range=data_range),
            "frc": Metrics.frc(pred, target, threshold=0.143)
        }
        return results
    except Exception as e:
        print(f"⚠️ Error computing metrics: {e}")
        return {
            "psnr": float('nan'),
            "ssim": float('nan'),
            "frc": float('nan')
        }


def compute_evaluation_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics between prediction and ground truth.
    
    Legacy wrapper for backward compatibility with run_microscopy.py
    """
    return compute_standard_metrics(pred, gt, data_range=1.0)


def compute_metrics(
    pred: np.ndarray, 
    target: np.ndarray, 
    metric_names: Optional[List[str]] = None,
    data_range: Optional[float] = None
) -> Dict[str, float]:
    """Compute specified metrics between predicted and target images.
    
    Legacy wrapper for backward compatibility with evaluation.py
    
    Args:
        pred: Predicted image
        target: Target image  
        metric_names: List of metric names to compute. If None, computes all.
        data_range: Dynamic range of the images
        
    Returns:
        Dictionary with requested metrics
    """
    all_metrics = compute_standard_metrics(pred, target, data_range=data_range)
    
    if metric_names is None:
        return all_metrics
    
    return {name: all_metrics[name] for name in metric_names if name in all_metrics}


def compute_baseline_metrics(
    pred: np.ndarray, 
    target: np.ndarray,
    data_range: Optional[float] = None
) -> Dict[str, float]:
    """Compute metrics for baseline methods.
    
    Legacy wrapper for backward compatibility with baseline.py
    Uses automatic data range calculation like the original baseline implementation.
    """
    if data_range is None:
        data_range = float(target.max() - target.min()) if target.size > 0 else 1.0
    
    return compute_standard_metrics(pred, target, data_range=data_range)


def evaluate_model_performance(
    pred_images: List[np.ndarray], 
    target_images: List[np.ndarray]
) -> Dict[str, float]:
    """Evaluate model performance on a set of images.
    
    Computes metrics for each image pair and returns averaged results.
    
    Args:
        pred_images: List of predicted images
        target_images: List of target images
        
    Returns:
        Dictionary with averaged metrics
    """
    all_results = []
    for pred, target in zip(pred_images, target_images):
        metrics = compute_standard_metrics(pred, target)
        all_results.append(metrics)
    
    if not all_results:
        return {}
    
    # Average all metrics
    averaged_metrics = {}
    metric_names = all_results[0].keys()
    
    for metric_name in metric_names:
        values = [result[metric_name] for result in all_results if not np.isnan(result[metric_name])]
        if values:
            averaged_metrics[metric_name] = float(np.mean(values))
        else:
            averaged_metrics[metric_name] = float('nan')
    
    return averaged_metrics


# Export all functions for backward compatibility
__all__ = [
    'compute_standard_metrics',
    'compute_evaluation_metrics', 
    'compute_metrics',
    'compute_baseline_metrics',
    'evaluate_model_performance',
    'Metrics'  # Re-export the main Metrics class
]
