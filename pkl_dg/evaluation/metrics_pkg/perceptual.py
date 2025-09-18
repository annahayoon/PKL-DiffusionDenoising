"""
Perceptual and Advanced Metrics

Metrics that measure perceptual quality, artifacts, and advanced image properties.
"""

import numpy as np
from typing import Optional
from .registry import register_metric

# Import the existing metrics directly to avoid circular import
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from metrics import Metrics


@register_metric(
    name="sar",
    category="perceptual",
    description="Signal-to-Artifact Ratio",
    requires_reference=True,
    requires_input=True
)
def sar_metric(pred: np.ndarray, target: np.ndarray, input_img: np.ndarray, **kwargs) -> float:
    """Compute SAR between prediction and target, using input for artifact detection."""
    try:
        return Metrics.sar(pred, target, input_img)
    except Exception:
        return 0.0


@register_metric(
    name="sharpness",
    category="perceptual", 
    description="Image sharpness using gradient magnitude",
    requires_reference=False,
    requires_input=False
)
def sharpness_metric(pred: np.ndarray, **kwargs) -> float:
    """Compute image sharpness using gradient magnitude."""
    # Compute gradients
    grad_x = np.gradient(pred, axis=1)
    grad_y = np.gradient(pred, axis=0)
    
    # Compute gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Return mean gradient magnitude as sharpness measure
    return float(np.mean(grad_mag))


@register_metric(
    name="contrast",
    category="perceptual",
    description="RMS contrast measure",
    requires_reference=False,
    requires_input=False
)
def contrast_metric(pred: np.ndarray, **kwargs) -> float:
    """Compute RMS contrast of image."""
    mean_intensity = np.mean(pred)
    contrast = np.sqrt(np.mean((pred - mean_intensity) ** 2))
    return float(contrast)


@register_metric(
    name="entropy",
    category="perceptual", 
    description="Image entropy (information content)",
    requires_reference=False,
    requires_input=False
)
def entropy_metric(pred: np.ndarray, **kwargs) -> float:
    """Compute image entropy."""
    # Normalize to 0-255 and compute histogram
    pred_norm = ((pred - pred.min()) / (pred.max() - pred.min() + 1e-8) * 255).astype(np.uint8)
    hist, _ = np.histogram(pred_norm, bins=256, range=(0, 256))
    
    # Compute probabilities
    hist = hist / np.sum(hist)
    
    # Remove zero probabilities
    hist = hist[hist > 0]
    
    # Compute entropy
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)


@register_metric(
    name="local_variance",
    category="perceptual",
    description="Local variance (texture measure)",
    requires_reference=False,
    requires_input=False
)
def local_variance_metric(pred: np.ndarray, **kwargs) -> float:
    """Compute local variance as a texture measure."""
    from scipy.ndimage import uniform_filter
    
    window_size = kwargs.get('window_size', 9)
    
    # Compute local mean and local mean of squares
    local_mean = uniform_filter(pred, size=window_size)
    local_mean_sq = uniform_filter(pred**2, size=window_size)
    
    # Compute local variance
    local_var = local_mean_sq - local_mean**2
    
    return float(np.mean(local_var))


@register_metric(
    name="spectral_angle",
    category="perceptual",
    description="Spectral Angle Mapper between images",
    requires_reference=True,
    requires_input=False
)
def spectral_angle_metric(pred: np.ndarray, target: np.ndarray, **kwargs) -> float:
    """Compute spectral angle between prediction and target."""
    # Flatten images to vectors
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Compute dot product and norms
    dot_product = np.dot(pred_flat, target_flat)
    norm_pred = np.linalg.norm(pred_flat)
    norm_target = np.linalg.norm(target_flat)
    
    # Avoid division by zero
    if norm_pred == 0 or norm_target == 0:
        return 90.0  # Maximum angle in degrees
    
    # Compute cosine of angle
    cos_angle = dot_product / (norm_pred * norm_target)
    cos_angle = np.clip(cos_angle, -1, 1)  # Ensure valid range
    
    # Convert to degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return float(angle_deg)
