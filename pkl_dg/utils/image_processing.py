"""
16-bit Image Processing Utilities for Microscopy

This module contains core image processing functions for handling
16-bit grayscale microscopy images, including normalization,
format conversion, and validation utilities.
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional
from PIL import Image


# Constants for 16-bit image processing
UINT16_MAX = 65535
UINT16_MIN = 0
MODEL_RANGE_MIN = -1.0
MODEL_RANGE_MAX = 1.0


def normalize_16bit_to_model_input(x: torch.Tensor) -> torch.Tensor:
    """Normalize 16-bit images (0-65535) to model input range [-1, 1].
    
    Args:
        x: Input tensor with values in 16-bit range [0, 65535]
        
    Returns:
        Normalized tensor with values in [-1, 1]
    """
    return (x / 32767.5) - 1.0


def denormalize_model_output_to_16bit(x: torch.Tensor) -> torch.Tensor:
    """Denormalize model output [-1, 1] back to 16-bit range [0, 65535].
    
    Args:
        x: Model output tensor with values in [-1, 1]
        
    Returns:
        Denormalized tensor with values in [0, 65535]
    """
    return torch.clamp((x + 1.0) * 32767.5, UINT16_MIN, UINT16_MAX)


def load_16bit_image(image_path: str, ensure_16bit: bool = True) -> np.ndarray:
    """Load a 16-bit microscopy image from file.
    
    Args:
        image_path: Path to the image file
        ensure_16bit: Whether to ensure the image is in 16-bit range
        
    Returns:
        Image array as float32 in 16-bit range [0, 65535]
    """
    img = Image.open(image_path)
    
    # Handle different image modes
    if ensure_16bit:
        if img.mode != 'I;16':
            if img.mode == 'I':
                # Already 32-bit int, convert to 16-bit range
                img = img.point(lambda x: min(x, UINT16_MAX))
            img = img.convert('I;16')
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Ensure we're in the correct range
    if ensure_16bit:
        img_array = np.clip(img_array, UINT16_MIN, UINT16_MAX)
    
    return img_array


def robust_normalize_16bit(
    image: np.ndarray, 
    percentile_low: float = 1.0, 
    percentile_high: float = 99.0
) -> np.ndarray:
    """Robust normalization for 16-bit images using percentiles.
    
    Args:
        image: Input 16-bit image array
        percentile_low: Lower percentile for normalization
        percentile_high: Upper percentile for normalization
        
    Returns:
        Normalized image in 16-bit range [0, 65535]
    """
    lo = np.percentile(image, percentile_low)
    hi = np.percentile(image, percentile_high)
    
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    
    # Normalize to [0, 1]
    normalized = (image - lo) / (hi - lo)
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # Scale to 16-bit range
    return (normalized * UINT16_MAX).astype(np.float32)


def convert_8bit_to_16bit_equivalent(image: np.ndarray) -> np.ndarray:
    """Convert 8-bit image to 16-bit equivalent range.
    
    Args:
        image: 8-bit image array [0, 255]
        
    Returns:
        Image scaled to 16-bit equivalent range [0, 65535]
    """
    return (image.astype(np.float32) * 257.0).astype(np.float32)  # 65535/255 ≈ 257


def adaptive_histogram_equalization(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image: Input image array
        clip_limit: Clipping limit for contrast enhancement
        tile_grid_size: Size of the grid for adaptive equalization
        
    Returns:
        Enhanced image
    """
    try:
        import cv2
        
        # Convert to uint16 for OpenCV
        img_uint16 = np.clip(image, 0, UINT16_MAX).astype(np.uint16)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Apply CLAHE
        enhanced = clahe.apply(img_uint16)
        
        return enhanced.astype(np.float32)
        
    except ImportError:
        print("⚠️ OpenCV not available, using simple histogram equalization")
        return simple_histogram_equalization(image)


def simple_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Simple histogram equalization without OpenCV dependency.
    
    Args:
        image: Input image array
        
    Returns:
        Equalized image
    """
    # Flatten image
    flat = image.flatten()
    
    # Get histogram
    hist, bins = np.histogram(flat, bins=256, range=(0, UINT16_MAX))
    
    # Compute cumulative distribution function
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    
    # Map pixel values
    equalized = np.interp(flat, bins[:-1], cdf_normalized * UINT16_MAX)
    
    return equalized.reshape(image.shape).astype(np.float32)


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


def resize_16bit_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    method: str = "bilinear"
) -> np.ndarray:
    """Resize 16-bit image while preserving dynamic range.
    
    Args:
        image: Input image array
        target_size: Target (height, width)
        method: Resize method ("bilinear", "nearest", "bicubic")
        
    Returns:
        Resized image
    """
    try:
        import cv2
        
        # Map method names
        method_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4
        }
        
        if method not in method_map:
            method = "bilinear"
        
        # Convert to uint16 for OpenCV
        img_uint16 = np.clip(image, 0, UINT16_MAX).astype(np.uint16)
        
        # Resize (note: OpenCV uses (width, height) order)
        resized = cv2.resize(img_uint16, (target_size[1], target_size[0]), 
                           interpolation=method_map[method])
        
        return resized.astype(np.float32)
        
    except ImportError:
        # Fallback using PIL
        img_pil = Image.fromarray(np.clip(image, 0, UINT16_MAX).astype(np.uint16))
        
        method_map = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS
        }
        
        pil_method = method_map.get(method, Image.BILINEAR)
        resized_pil = img_pil.resize((target_size[1], target_size[0]), pil_method)
        
        return np.array(resized_pil, dtype=np.float32)


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


def create_16bit_test_image(
    height: int = 256, 
    width: int = 256, 
    pattern: str = "gradient"
) -> np.ndarray:
    """Create a test 16-bit image for debugging and testing.
    
    Args:
        height: Image height
        width: Image width
        pattern: Type of test pattern ("gradient", "checkerboard", "random", "spots")
        
    Returns:
        Test image in 16-bit range
    """
    if pattern == "gradient":
        # Linear gradient from 0 to 65535
        gradient = np.linspace(0, UINT16_MAX, width)
        image = np.tile(gradient, (height, 1))
    
    elif pattern == "checkerboard":
        # Checkerboard pattern
        check_size = min(height, width) // 8
        image = np.zeros((height, width))
        for i in range(0, height, check_size):
            for j in range(0, width, check_size):
                if (i // check_size + j // check_size) % 2 == 0:
                    image[i:i+check_size, j:j+check_size] = UINT16_MAX
    
    elif pattern == "random":
        # Random noise
        image = np.random.randint(0, UINT16_MAX + 1, (height, width))
    
    elif pattern == "spots":
        # Bright spots on dark background (simulating fluorescence)
        image = np.random.randint(0, 1000, (height, width))  # Dark background
        # Add bright spots
        num_spots = 20
        for _ in range(num_spots):
            y = np.random.randint(20, height - 20)
            x = np.random.randint(20, width - 20)
            # Gaussian spot
            yy, xx = np.ogrid[:height, :width]
            spot = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 10**2))
            image += spot * np.random.randint(30000, UINT16_MAX)
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return np.clip(image, UINT16_MIN, UINT16_MAX).astype(np.float32)
