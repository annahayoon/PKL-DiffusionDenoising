"""
16-bit Image Processing Utilities for Microscopy

This module contains core image processing functions for handling
16-bit grayscale microscopy images, including normalization,
format conversion, and validation utilities.
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional, List
from PIL import Image


# Constants for 16-bit image processing
UINT16_MAX = 65535
UINT16_MIN = 0
MODEL_RANGE_MIN = -1.0
MODEL_RANGE_MAX = 1.0


# All 16-bit processing functions moved to utils_16bit.py
# Import them here for backward compatibility and to maintain the API
from .utils_16bit import (
    normalize_16bit_to_model_input,
    denormalize_model_output_to_16bit,
    load_16bit_image,
    resize_16bit_image,
    create_16bit_test_image,
    to_uint16_grayscale,
    save_16bit_grayscale,
    save_16bit_comparison,
    validate_16bit_range,
    validate_model_range
)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction to image.
    
    Args:
        image: Input image array
        gamma: Gamma value (< 1 brightens, > 1 darkens)
        
    Returns:
        Gamma corrected image
    """
    if gamma <= 0:
        raise ValueError("Gamma must be positive")
    
    # Normalize to [0, 1]
    normalized = image / UINT16_MAX
    
    # Apply gamma correction
    corrected = np.power(normalized, gamma)
    
    # Scale back to 16-bit
    return (corrected * UINT16_MAX).astype(np.float32)


def bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> np.ndarray:
    """Apply bilateral filter for noise reduction while preserving edges.
    
    Args:
        image: Input image array
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
        
    Returns:
        Filtered image
    """
    try:
        import cv2
        
        # Convert to uint16 for OpenCV
        img_uint16 = np.clip(image, 0, UINT16_MAX).astype(np.uint16)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(img_uint16, d, sigma_color, sigma_space)
        
        return filtered.astype(np.float32)
        
    except ImportError:
        print("⚠️ OpenCV not available, using Gaussian filter as fallback")
        return gaussian_filter(image, sigma=1.0)


def gaussian_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian filter using scipy if available.
    
    Args:
        image: Input image array
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Filtered image
    """
    try:
        from scipy.ndimage import gaussian_filter as scipy_gaussian
        return scipy_gaussian(image, sigma=sigma).astype(np.float32)
    except ImportError:
        print("⚠️ SciPy not available, returning original image")
        return image


def unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.0,
    strength: float = 1.0
) -> np.ndarray:
    """Apply unsharp masking for edge enhancement.
    
    Args:
        image: Input image array
        sigma: Standard deviation for Gaussian blur
        strength: Strength of sharpening
        
    Returns:
        Sharpened image
    """
    # Create blurred version
    blurred = gaussian_filter(image, sigma=sigma)
    
    # Create mask
    mask = image - blurred
    
    # Apply sharpening
    sharpened = image + strength * mask
    
    return np.clip(sharpened, 0, UINT16_MAX).astype(np.float32)


# resize_16bit_image moved to utils_16bit.py


def crop_center(
    image: np.ndarray,
    crop_size: Tuple[int, int]
) -> np.ndarray:
    """Crop image from center.
    
    Args:
        image: Input image array
        crop_size: Target (height, width)
        
    Returns:
        Cropped image
    """
    h, w = image.shape[-2:]
    crop_h, crop_w = crop_size
    
    if crop_h > h or crop_w > w:
        raise ValueError(f"Crop size {crop_size} larger than image size {(h, w)}")
    
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    return image[..., start_h:start_h + crop_h, start_w:start_w + crop_w]


def pad_to_size(
    image: np.ndarray,
    target_size: Tuple[int, int],
    mode: str = "constant",
    constant_value: float = 0
) -> np.ndarray:
    """Pad image to target size.
    
    Args:
        image: Input image array
        target_size: Target (height, width)
        mode: Padding mode ("constant", "reflect", "edge")
        constant_value: Value for constant padding
        
    Returns:
        Padded image
    """
    h, w = image.shape[-2:]
    target_h, target_w = target_size
    
    if target_h < h or target_w < w:
        raise ValueError(f"Target size {target_size} smaller than image size {(h, w)}")
    
    pad_h = target_h - h
    pad_w = target_w - w
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    if image.ndim == 2:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
    elif image.ndim == 3:
        pad_width = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")
    
    return np.pad(image, pad_width, mode=mode, constant_values=constant_value)


# create_16bit_test_image moved to utils_16bit.py


# to_uint16_grayscale moved to utils_16bit.py


# save_16bit_grayscale moved to utils_16bit.py


# save_16bit_comparison moved to utils_16bit.py
