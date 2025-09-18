"""
Utilities for 16-bit microscopy image processing.

This module contains functions and classes specifically designed for handling
16-bit grayscale microscopy images commonly used in biological imaging,
including wide-field (WF), two-photon (2P), and PSF images.
"""

import torch
import numpy as np
from typing import Union, Tuple
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


def get_16bit_image_stats(image: Union[np.ndarray, torch.Tensor]) -> dict:
    """Get statistics for a 16-bit image.
    
    Args:
        image: 16-bit image array or tensor
        
    Returns:
        Dictionary with image statistics
    """
    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().numpy()
    else:
        image_np = image
    
    return {
        "min": float(image_np.min()),
        "max": float(image_np.max()),
        "mean": float(image_np.mean()),
        "std": float(image_np.std()),
        "shape": image_np.shape,
        "dtype": str(image_np.dtype),
        "dynamic_range": float(image_np.max() - image_np.min()),
        "is_16bit_range": image_np.min() >= 0 and image_np.max() <= UINT16_MAX
    }


def validate_16bit_image(image: Union[np.ndarray, torch.Tensor], name: str = "image") -> bool:
    """Validate that an image is in proper 16-bit range.
    
    Args:
        image: Image to validate
        name: Name for error messages
        
    Returns:
        True if valid, raises ValueError if not
    """
    if isinstance(image, torch.Tensor):
        min_val = image.min().item()
        max_val = image.max().item()
    else:
        min_val = image.min()
        max_val = image.max()
    
    if min_val < UINT16_MIN:
        raise ValueError(f"{name} has values below 16-bit range: min={min_val}")
    
    if max_val > UINT16_MAX:
        raise ValueError(f"{name} has values above 16-bit range: max={max_val}")
    
    return True


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


class Microscopy16BitProcessor:
    """Processor class for 16-bit microscopy images."""
    
    def __init__(self, normalize_on_load: bool = True, percentile_range: Tuple[float, float] = (1.0, 99.0)):
        """Initialize the processor.
        
        Args:
            normalize_on_load: Whether to normalize images when loading
            percentile_range: Percentile range for robust normalization
        """
        self.normalize_on_load = normalize_on_load
        self.percentile_low, self.percentile_high = percentile_range
    
    def load_image(self, path: str) -> np.ndarray:
        """Load and optionally normalize a 16-bit image."""
        image = load_16bit_image(path, ensure_16bit=True)
        
        if self.normalize_on_load:
            image = robust_normalize_16bit(image, self.percentile_low, self.percentile_high)
        
        return image
    
    def to_model_input(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert 16-bit image to model input format."""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        return normalize_16bit_to_model_input(image)
    
    def from_model_output(self, output: torch.Tensor) -> torch.Tensor:
        """Convert model output back to 16-bit format."""
        return denormalize_model_output_to_16bit(output)
    
    def get_stats(self, image: Union[np.ndarray, torch.Tensor]) -> dict:
        """Get image statistics."""
        return get_16bit_image_stats(image)
    
    def validate(self, image: Union[np.ndarray, torch.Tensor], name: str = "image") -> bool:
        """Validate image is in 16-bit range."""
        return validate_16bit_image(image, name)
