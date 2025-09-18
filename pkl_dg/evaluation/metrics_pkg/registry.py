"""
Metric Registry System for Dynamic Metric Registration and Discovery

This module provides a registry pattern for metrics, allowing easy extension
and configuration-driven metric selection.
"""

from typing import Dict, Callable, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Global registry for metrics
METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_metric(
    name: str, 
    category: str = "general",
    description: str = "",
    requires_reference: bool = True,
    requires_input: bool = False
):
    """
    Decorator to register metrics in the global registry.
    
    Args:
        name: Unique name for the metric
        category: Category of metric (image_quality, perceptual, downstream, etc.)
        description: Human-readable description
        requires_reference: Whether metric needs a reference/target image
        requires_input: Whether metric needs the original input image
    """
    def decorator(metric_fn: Callable) -> Callable:
        if name in METRIC_REGISTRY:
            logger.warning(f"Metric '{name}' already registered, overwriting")
        
        METRIC_REGISTRY[name] = {
            'function': metric_fn,
            'category': category,
            'description': description,
            'requires_reference': requires_reference,
            'requires_input': requires_input
        }
        
        logger.debug(f"Registered metric: {name} ({category})")
        return metric_fn
    
    return decorator


def get_metric(name: str) -> Callable:
    """Get registered metric function by name."""
    if name not in METRIC_REGISTRY:
        available = list_metrics()
        raise ValueError(f"Unknown metric: {name}. Available: {available}")
    return METRIC_REGISTRY[name]['function']


def get_metric_info(name: str) -> Dict[str, Any]:
    """Get full metric information by name."""
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}")
    return METRIC_REGISTRY[name]


def list_metrics(category: Optional[str] = None) -> List[str]:
    """
    List all registered metrics, optionally filtered by category.
    
    Args:
        category: Optional category filter
        
    Returns:
        List of metric names
    """
    if category is None:
        return list(METRIC_REGISTRY.keys())
    
    return [
        name for name, info in METRIC_REGISTRY.items() 
        if info['category'] == category
    ]


def list_categories() -> List[str]:
    """List all available metric categories."""
    return list(set(info['category'] for info in METRIC_REGISTRY.values()))


def compute_metrics(
    pred: 'np.ndarray', 
    target: 'np.ndarray' = None,
    input_img: 'np.ndarray' = None,
    metric_names: List[str] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Compute multiple metrics using the registry.
    
    Args:
        pred: Predicted/reconstructed image
        target: Ground truth reference image
        input_img: Original input image (for metrics that need it)
        metric_names: List of metrics to compute. If None, computes all applicable
        **kwargs: Additional arguments passed to metric functions
        
    Returns:
        Dictionary mapping metric names to computed values
    """
    import numpy as np
    
    if metric_names is None:
        # Determine which metrics we can compute based on available inputs
        metric_names = []
        for name, info in METRIC_REGISTRY.items():
            if info['requires_reference'] and target is None:
                continue
            if info['requires_input'] and input_img is None:
                continue
            metric_names.append(name)
    
    results = {}
    for name in metric_names:
        try:
            metric_info = get_metric_info(name)
            metric_fn = metric_info['function']
            
            # Prepare arguments based on metric requirements
            args = [pred]
            if metric_info['requires_reference']:
                if target is None:
                    logger.warning(f"Skipping {name}: requires reference image")
                    continue
                args.append(target)
            
            if metric_info['requires_input']:
                if input_img is None:
                    logger.warning(f"Skipping {name}: requires input image")
                    continue
                args.append(input_img)
            
            # Compute metric
            result = metric_fn(*args, **kwargs)
            
            # Handle potential NaN/inf values
            if isinstance(result, (int, float)):
                if not np.isfinite(result):
                    logger.warning(f"Metric {name} returned non-finite value: {result}")
                    result = 0.0
            
            results[name] = result
            
        except Exception as e:
            logger.error(f"Error computing metric {name}: {e}")
            results[name] = 0.0
    
    return results


def print_registry_info():
    """Print information about all registered metrics."""
    categories = list_categories()
    
    print("=" * 60)
    print("REGISTERED METRICS")
    print("=" * 60)
    
    for category in sorted(categories):
        print(f"\n{category.upper()}:")
        metrics = list_metrics(category)
        for metric in sorted(metrics):
            info = get_metric_info(metric)
            req_ref = "✓" if info['requires_reference'] else "✗"
            req_inp = "✓" if info['requires_input'] else "✗"
            desc = info['description'] or "No description"
            print(f"  {metric:20} | Ref:{req_ref} Inp:{req_inp} | {desc}")
    
    print(f"\nTotal: {len(METRIC_REGISTRY)} metrics across {len(categories)} categories")
