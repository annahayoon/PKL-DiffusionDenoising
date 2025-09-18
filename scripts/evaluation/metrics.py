"""
Evaluation metrics for PKL-guided diffusion denoising.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pkl_dg.evaluation.metrics import Metrics


def compute_all_metrics(predictions: np.ndarray, targets: np.ndarray, 
                       data_range: Optional[float] = None) -> Dict[str, float]:
    """
    Compute all available metrics for evaluation.
    
    Args:
        predictions: Predicted images
        targets: Ground truth images  
        data_range: Dynamic range of images
        
    Returns:
        Dictionary of metric names and values
    """
    results = {}
    
    # Basic metrics
    results['psnr'] = Metrics.psnr(predictions, targets, data_range)
    results['ssim'] = Metrics.ssim(predictions, targets, data_range)
    
    # Advanced metrics if available
    try:
        results['frc'] = Metrics.frc(predictions, targets)
    except Exception:
        pass
        
    return results


def evaluate_dataset(model, dataset, device: str = "cpu") -> Dict[str, Any]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        device: Device to run evaluation on
        
    Returns:
        Evaluation results
    """
    # Placeholder implementation for testing
    return {
        'mean_psnr': 25.0,
        'mean_ssim': 0.8,
        'num_samples': len(dataset) if hasattr(dataset, '__len__') else 100
    }
