"""
Evaluation Metrics Package

This package provides a comprehensive set of metrics for evaluating image
reconstruction and denoising methods, organized by category and using a
registry pattern for easy extension.

Categories:
- image_quality: Standard metrics like PSNR, SSIM, FRC
- perceptual: Advanced metrics like SAR, sharpness, contrast
- downstream: Task-specific metrics like segmentation F1, Hausdorff distance  
- robustness: Robustness and hallucination detection metrics

Usage:
    from scripts.evaluation.metrics import compute_metrics, list_metrics
    
    # Compute specific metrics
    results = compute_metrics(pred, target, metric_names=['psnr', 'ssim'])
    
    # Compute all applicable metrics
    results = compute_metrics(pred, target)
    
    # List available metrics
    available = list_metrics()
"""

# Import all metric modules to register them - DISABLED due to circular imports
# from . import image_quality
# from . import perceptual  
# from . import downstream
# from . import robustness  # Commented out to avoid circular import

# Import registry functions for easy access
from .registry import (
    register_metric,
    get_metric,
    get_metric_info,
    list_metrics,
    list_categories,
    compute_metrics,
    print_registry_info,
    METRIC_REGISTRY
)

# Import convenience functions - DISABLED due to circular imports  
from .downstream import compute_downstream_metrics
# from .robustness import test_alignment_error_with_sampler, add_out_of_focus_artifact  # Commented out to avoid circular import

# Import the original Metrics class for backward compatibility (direct path import)
import importlib.util
import os
metrics_file_path = os.path.join(os.path.dirname(__file__), '..', 'metrics.py')
spec = importlib.util.spec_from_file_location("metrics_module", metrics_file_path)
metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_module)
Metrics = metrics_module.Metrics

__all__ = [
    # Registry functions
    'register_metric',
    'get_metric', 
    'get_metric_info',
    'list_metrics',
    'list_categories',
    'compute_metrics',
    'print_registry_info',
    'METRIC_REGISTRY',
    
    # Convenience functions
    'compute_downstream_metrics',
    'test_alignment_error_with_sampler',
    'add_out_of_focus_artifact',
    
    # Module references
    'image_quality',
    'perceptual',
    'downstream', 
    'robustness',
    
    # Backward compatibility
    'Metrics'
]


def get_default_metrics() -> list:
    """Get the default set of metrics for evaluation."""
    return [
        'psnr',
        'ssim', 
        'frc',
        'sar',
        'sharpness',
        'contrast'
    ]


def get_comprehensive_metrics() -> list:
    """Get a comprehensive set of metrics for thorough evaluation."""
    return [
        # Image quality
        'psnr',
        'ssim',
        'frc', 
        'mse',
        'mae',
        'snr',
        
        # Perceptual
        'sar',
        'sharpness',
        'contrast',
        'entropy',
        'local_variance',
        'spectral_angle',
        
        # Robustness
        'noise_robustness',
        'hallucination_score'
    ]


def print_metrics_help():
    """Print help information about available metrics."""
    print(__doc__)
    print_registry_info()
