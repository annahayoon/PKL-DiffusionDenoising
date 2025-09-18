"""
16-bit Image Processing Utilities with Adaptive Normalization

This module consolidates all 16-bit image processing functions with consistent
implementations. Uses tifffile for loading and PyTorch-style normalization math.

Key functions:
- load_16bit_image(): Load 16-bit images using tifffile
- normalize_16bit_to_model_input(): Convert [0, 65535] to [-1, 1] (legacy)
- denormalize_model_output_to_16bit(): Convert [-1, 1] to [0, 65535] (legacy)
- AdaptiveNormalizer: Data-driven normalization for better DDPM training
- NormalizationParams: Parameters for adaptive normalization with exact inverse transforms
"""

import json
import torch
import numpy as np
from typing import Union, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict


# Constants for 16-bit image processing
UINT16_MAX = 65535
UINT16_MIN = 0
MODEL_RANGE_MIN = -1.0
MODEL_RANGE_MAX = 1.0


# =============================================================================
# Adaptive Normalization Classes
# =============================================================================

@dataclass
class NormalizationParams:
    """Parameters for normalization and inverse transformation."""
    
    # Modality-specific parameters
    wf_min: float
    wf_max: float
    tp_min: float  
    tp_max: float
    wf_percentile_min: float = 0.1
    wf_percentile_max: float = 99.9
    tp_percentile_min: float = 0.1
    tp_percentile_max: float = 99.9
    
    # Global parameters
    target_min: float = -1.0
    target_max: float = 1.0
    
    def save(self, path: Union[str, Path]) -> None:
        """Save normalization parameters to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'NormalizationParams':
        """Load normalization parameters from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class AdaptiveNormalizer:
    """Data-driven normalizer that adapts to actual intensity distributions."""
    
    def __init__(self, params: Optional[NormalizationParams] = None):
        self.params = params
        
    def compute_normalization_params(
        self,
        wf_data: np.ndarray,
        tp_data: np.ndarray,
        wf_percentiles: Tuple[float, float] = (0.1, 99.9),
        tp_percentiles: Tuple[float, float] = (0.1, 99.9)
    ) -> NormalizationParams:
        """
        Compute normalization parameters from actual data distributions.
        
        Args:
            wf_data: Wide-field image data
            tp_data: Two-photon image data  
            wf_percentiles: (min_percentile, max_percentile) for WF
            tp_percentiles: (min_percentile, max_percentile) for 2P
            
        Returns:
            NormalizationParams object with computed scaling parameters
        """
        
        # Compute percentile-based ranges
        wf_min = np.percentile(wf_data, wf_percentiles[0])
        wf_max = np.percentile(wf_data, wf_percentiles[1])
        
        tp_min = np.percentile(tp_data, tp_percentiles[0]) 
        tp_max = np.percentile(tp_data, tp_percentiles[1])
        
        # Ensure we don't have zero range
        if wf_max - wf_min < 1e-6:
            print(f"Warning: WF data has very small range [{wf_min}, {wf_max}]")
            wf_max = wf_min + 1.0
            
        if tp_max - tp_min < 1e-6:
            print(f"Warning: 2P data has very small range [{tp_min}, {tp_max}]")
            tp_max = tp_min + 1.0
        
        params = NormalizationParams(
            wf_min=float(wf_min),
            wf_max=float(wf_max),
            wf_percentile_min=wf_percentiles[0],
            wf_percentile_max=wf_percentiles[1],
            tp_min=float(tp_min),
            tp_max=float(tp_max), 
            tp_percentile_min=tp_percentiles[0],
            tp_percentile_max=tp_percentiles[1]
        )
        
        self.params = params
        
        print(f"Computed normalization parameters:")
        print(f"  WF: [{wf_min:.1f}, {wf_max:.1f}] -> [-1, 1]")
        print(f"  2P: [{tp_min:.1f}, {tp_max:.1f}] -> [-1, 1]")
        
        return params
    
    def normalize_wf(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Normalize WF data to [-1, 1] using computed parameters."""
        if self.params is None:
            raise ValueError("Must compute or load normalization parameters first")
            
        # Clip to computed range
        x_clipped = np.clip(x, self.params.wf_min, self.params.wf_max) if isinstance(x, np.ndarray) else torch.clamp(x, self.params.wf_min, self.params.wf_max)
        
        # Scale to [0, 1]
        x_scaled = (x_clipped - self.params.wf_min) / (self.params.wf_max - self.params.wf_min)
        
        # Scale to [-1, 1]
        return 2 * x_scaled - 1
    
    def normalize_tp(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Normalize 2P data to [-1, 1] using computed parameters."""
        if self.params is None:
            raise ValueError("Must compute or load normalization parameters first")
            
        # Clip to computed range  
        x_clipped = np.clip(x, self.params.tp_min, self.params.tp_max) if isinstance(x, np.ndarray) else torch.clamp(x, self.params.tp_min, self.params.tp_max)
        
        # Scale to [0, 1]
        x_scaled = (x_clipped - self.params.tp_min) / (self.params.tp_max - self.params.tp_min)
        
        # Scale to [-1, 1]
        return 2 * x_scaled - 1
    
    def denormalize_wf(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Denormalize WF data from [-1, 1] back to original intensity scale."""
        if self.params is None:
            raise ValueError("Must load normalization parameters first")
            
        # From [-1, 1] to [0, 1]
        x_scaled = (x + 1) / 2
        
        # From [0, 1] to original intensity range
        x_original = x_scaled * (self.params.wf_max - self.params.wf_min) + self.params.wf_min
        
        # Clamp to ensure valid range (non-negative)
        return np.clip(x_original, 0, None) if isinstance(x_original, np.ndarray) else torch.clamp(x_original, min=0)
    
    def denormalize_tp(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Denormalize 2P data from [-1, 1] back to original intensity scale."""
        if self.params is None:
            raise ValueError("Must load normalization parameters first")
            
        # From [-1, 1] to [0, 1]
        x_scaled = (x + 1) / 2
        
        # From [0, 1] to original intensity range
        x_original = x_scaled * (self.params.tp_max - self.params.tp_min) + self.params.tp_min
        
        # Clamp to ensure valid range (non-negative)
        return np.clip(x_original, 0, None) if isinstance(x_original, np.ndarray) else torch.clamp(x_original, min=0)


# =============================================================================
# Legacy 16-bit Normalization Functions
# =============================================================================

def load_16bit_image(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Load 16-bit image from file using tifffile.
    
    Args:
        path: Path to the image file
        
    Returns:
        Image array as float32 in 16-bit range [0, 65535], or None if loading fails
    """
    try:
        import tifffile
        
        img = tifffile.imread(str(path))
        
        if img.dtype == np.uint16:
            return img.astype(np.float32)
        elif img.dtype == np.uint8:
            raise ValueError(f"8-bit images are not supported. Image {path} should be 16-bit TIFF format.")
        else:
            # Handle other dtypes by converting to float32
            return img.astype(np.float32)
            
    except ImportError:
        raise ImportError("tifffile is required for loading 16-bit images. Install with: pip install tifffile")
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def normalize_16bit_to_model_input(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Normalize 16-bit images from [0, 65535] to model input range [-1, 1].
    
    Uses PyTorch-style normalization: (x / 32767.5) - 1.0
    This is mathematically equivalent to: (2*x / 65535) - 1.0
    
    Args:
        x: Input tensor/array with values in 16-bit range [0, 65535]
        
    Returns:
        Normalized tensor/array with values in [-1, 1]
    """
    if x is None:
        return None
        
    if isinstance(x, torch.Tensor):
        return (x / 32767.5) - 1.0
    else:
        # NumPy array
        return ((x / 32767.5) - 1.0).astype(np.float32)


def denormalize_model_output_to_16bit(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Denormalize model output from [-1, 1] back to 16-bit range [0, 65535].
    
    Uses PyTorch-style denormalization with clamping: (x + 1.0) * 32767.5
    
    Args:
        x: Model output tensor/array with values in [-1, 1]
        
    Returns:
        Denormalized tensor/array with values clamped to [0, 65535]
    """
    if x is None:
        return None
        
    if isinstance(x, torch.Tensor):
        return torch.clamp((x + 1.0) * 32767.5, UINT16_MIN, UINT16_MAX)
    else:
        # NumPy array
        denorm = (x + 1.0) * 32767.5
        return np.clip(denorm, UINT16_MIN, UINT16_MAX).astype(np.float32)


def validate_16bit_range(x: Union[torch.Tensor, np.ndarray], name: str = "image") -> bool:
    """Validate that values are in valid 16-bit range [0, 65535].
    
    Args:
        x: Input tensor/array to validate
        name: Name for error messages
        
    Returns:
        True if valid, False otherwise
    """
    if x is None:
        return False
        
    if isinstance(x, torch.Tensor):
        min_val, max_val = x.min().item(), x.max().item()
    else:
        min_val, max_val = float(x.min()), float(x.max())
    
    if min_val < UINT16_MIN or max_val > UINT16_MAX:
        print(f"Warning: {name} values [{min_val:.1f}, {max_val:.1f}] outside 16-bit range [{UINT16_MIN}, {UINT16_MAX}]")
        return False
        
    return True


def validate_model_range(x: Union[torch.Tensor, np.ndarray], name: str = "tensor") -> bool:
    """Validate that values are in valid model input range [-1, 1].
    
    Args:
        x: Input tensor/array to validate  
        name: Name for error messages
        
    Returns:
        True if valid, False otherwise
    """
    if x is None:
        return False
        
    if isinstance(x, torch.Tensor):
        min_val, max_val = x.min().item(), x.max().item()
    else:
        min_val, max_val = float(x.min()), float(x.max())
    
    if min_val < MODEL_RANGE_MIN or max_val > MODEL_RANGE_MAX:
        print(f"Warning: {name} values [{min_val:.3f}, {max_val:.3f}] outside model range [{MODEL_RANGE_MIN}, {MODEL_RANGE_MAX}]")
        return False
        
    return True


def save_16bit_patch(
    patch: np.ndarray, 
    output_path: Union[str, Path],
    preserve_range: bool = True,
    photometric: str = 'minisblack'
) -> None:
    """Save a microscopy patch as 16-bit TIFF with preserved intensity ranges.
    
    Args:
        patch: Input patch array
        output_path: Output file path (should end with .tif or .tiff)
        preserve_range: Whether to preserve the original intensity range
        photometric: TIFF photometric interpretation
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile is required for saving 16-bit images. Install with: pip install tifffile")
    
    # Ensure output path has correct extension
    output_path = str(output_path)
    if not (output_path.endswith('.tif') or output_path.endswith('.tiff')):
        output_path = output_path + '.tif'
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Handle different input types
    if isinstance(patch, torch.Tensor):
        patch_np = patch.detach().cpu().numpy()
    else:
        patch_np = patch.copy()
    
    # Ensure 2D
    if patch_np.ndim > 2:
        patch_np = patch_np.squeeze()
    
    if preserve_range:
        # Keep original intensity range - just ensure proper dtype
        if patch_np.dtype == np.uint16:
            # Already 16-bit, save directly
            tifffile.imwrite(output_path, patch_np, photometric=photometric)
        else:
            # Convert to float32 and clamp to 16-bit range
            patch_np = patch_np.astype(np.float32)
            patch_np = np.clip(patch_np, UINT16_MIN, UINT16_MAX)
            tifffile.imwrite(output_path, patch_np.astype(np.uint16), photometric=photometric)
    else:
        # Normalize to full 16-bit range
        patch_min = patch_np.min()
        patch_max = patch_np.max()
        
        if patch_max > patch_min:
            patch_normalized = (patch_np - patch_min) / (patch_max - patch_min)
        else:
            patch_normalized = np.zeros_like(patch_np)
        
        patch_16bit = (patch_normalized * UINT16_MAX).astype(np.uint16)
        tifffile.imwrite(output_path, patch_16bit, photometric=photometric)


def extract_and_save_patches_16bit(
    image: np.ndarray,
    output_dir: Union[str, Path],
    patch_size: int = 256,
    stride: Optional[int] = None,
    prefix: str = "patch",
    preserve_range: bool = True
) -> list:
    """Extract patches from a large image and save as 16-bit TIFF files.
    
    Args:
        image: Input image array
        output_dir: Directory to save patches
        patch_size: Size of patches to extract
        stride: Stride between patches (defaults to patch_size for non-overlapping)
        prefix: Filename prefix for patches
        preserve_range: Whether to preserve original intensity ranges
        
    Returns:
        List of saved patch file paths
    """
    if stride is None:
        stride = patch_size
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    h, w = image.shape[:2]
    patch_paths = []
    patch_id = 0
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patch
            patch = image[y:y+patch_size, x:x+patch_size]
            
            # Skip if patch is too small
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue
            
            # Save patch
            patch_filename = f"{prefix}_{patch_id:06d}.tif"
            patch_path = output_dir / patch_filename
            
            save_16bit_patch(patch, str(patch_path), preserve_range=preserve_range)
            patch_paths.append(str(patch_path))
            patch_id += 1
    
    return patch_paths


def resize_16bit_image(
    image: np.ndarray,
    target_size: tuple,
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
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(np.clip(image, 0, UINT16_MAX).astype(np.uint16))
        
        method_map = {
            "nearest": PILImage.NEAREST,
            "bilinear": PILImage.BILINEAR,
            "bicubic": PILImage.BICUBIC,
            "lanczos": PILImage.LANCZOS
        }
        
        pil_method = method_map.get(method, PILImage.BILINEAR)
        resized_pil = img_pil.resize((target_size[1], target_size[0]), pil_method)
        
        return np.array(resized_pil, dtype=np.float32)


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


def to_uint16_grayscale(
    tensor: Union[torch.Tensor, np.ndarray], 
    transform: Optional[any] = None,
    percentile_clip: float = 99.5,
    preserve_range: bool = True
) -> np.ndarray:
    """
    Convert model output tensor to 16-bit grayscale numpy array.
    
    Args:
        tensor: Model output tensor in model domain [-1, 1] or intensity domain
        transform: Transform object to convert from model domain to intensity domain
        percentile_clip: Percentile for clipping outliers (default: 99.5)
        preserve_range: If True, preserve original intensity range; if False, normalize to full 16-bit
        
    Returns:
        16-bit grayscale numpy array (0-65535)
    """
    # Convert to numpy and squeeze to 2D if needed
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            array = tensor.detach().cpu().numpy()
        else:
            array = tensor.detach().numpy()
    else:
        array = np.array(tensor)
    
    # Remove batch and channel dimensions if present
    while array.ndim > 2:
        array = array.squeeze()
    
    # Convert from model domain to intensity domain if transform provided
    if transform is not None:
        # Convert back to tensor for transform
        tensor_2d = torch.from_numpy(array).float()
        if tensor_2d.ndim == 2:
            tensor_2d = tensor_2d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Apply inverse transform
        intensity_tensor = transform.inverse(tensor_2d)
        array = intensity_tensor.squeeze().detach().cpu().numpy()
    
    # Ensure non-negative
    array = np.clip(array, 0, None)
    
    if preserve_range:
        # Preserve the original intensity range, just convert to uint16
        # Clip outliers based on percentile
        max_val = np.percentile(array, percentile_clip)
        array = np.clip(array, 0, max_val)
        
        # Scale to 16-bit range while preserving relative intensities
        if max_val > 0:
            array = (array / max_val * 65535.0)
    else:
        # Normalize to full 16-bit range
        min_val = np.percentile(array, 100 - percentile_clip)
        max_val = np.percentile(array, percentile_clip)
        
        if max_val > min_val:
            array = (array - min_val) / (max_val - min_val)
        else:
            array = np.zeros_like(array)
        
        array = array * 65535.0
    
    # Convert to uint16
    return np.clip(array, 0, 65535).astype(np.uint16)


def save_16bit_grayscale(
    tensor: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
    transform: Optional[any] = None,
    percentile_clip: float = 99.5,
    preserve_range: bool = True
) -> None:
    """
    Save model output as 16-bit grayscale TIFF image.
    
    Args:
        tensor: Model output tensor
        path: Output file path (should end with .tif or .tiff)
        transform: Transform object to convert from model domain
        percentile_clip: Percentile for clipping outliers
        preserve_range: Whether to preserve original intensity range
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile is required for saving 16-bit images. Install with: pip install tifffile")
    
    # Convert to 16-bit array
    array_16bit = to_uint16_grayscale(
        tensor, 
        transform=transform, 
        percentile_clip=percentile_clip,
        preserve_range=preserve_range
    )
    
    # Save as 16-bit TIFF
    tifffile.imwrite(str(path), array_16bit, photometric='minisblack')


def save_16bit_comparison(
    wf_input: Union[torch.Tensor, np.ndarray],
    prediction: Union[torch.Tensor, np.ndarray], 
    gt_target: Optional[Union[torch.Tensor, np.ndarray]],
    path: Union[str, Path],
    transform: Optional[any] = None,
    percentile_clip: float = 99.5,
    preserve_range: bool = True
) -> None:
    """
    Save side-by-side comparison as 16-bit TIFF: [WF | Prediction | GT].
    
    Args:
        wf_input: WF input tensor
        prediction: Model prediction tensor
        gt_target: Ground truth tensor (optional)
        path: Output file path
        transform: Transform object
        percentile_clip: Percentile for clipping
        preserve_range: Whether to preserve intensity range
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile is required for saving 16-bit images. Install with: pip install tifffile")
    
    # Convert all to 16-bit arrays
    wf_16bit = to_uint16_grayscale(wf_input, transform, percentile_clip, preserve_range)
    pred_16bit = to_uint16_grayscale(prediction, transform, percentile_clip, preserve_range)
    
    # Create comparison
    if gt_target is not None:
        gt_16bit = to_uint16_grayscale(gt_target, transform, percentile_clip, preserve_range)
        comparison = np.concatenate([wf_16bit, pred_16bit, gt_16bit], axis=1)
    else:
        comparison = np.concatenate([wf_16bit, pred_16bit], axis=1)
    
    # Save as 16-bit TIFF
    tifffile.imwrite(str(path), comparison, photometric='minisblack')


# =============================================================================
# Adaptive Normalization Utility Functions
# =============================================================================

def create_normalization_params_from_metadata(metadata_path: Union[str, Path]) -> NormalizationParams:
    """
    Create normalization parameters from dataset metadata file.
    
    This is a convenience function for when you have already processed data
    and want to extract normalization parameters from the metadata.
    """
    import yaml
    
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    wf_meta = metadata['wf_metadata']
    tp_meta = metadata['tp_metadata']
    
    # Use a conservative percentile approach based on the metadata
    # We'll use the actual min/max but could adjust for percentiles
    params = NormalizationParams(
        wf_min=float(wf_meta['min_intensity']),
        wf_max=float(wf_meta['max_intensity']),
        wf_percentile_min=0.0,  # Using actual min
        wf_percentile_max=100.0,  # Using actual max
        tp_min=float(tp_meta['min_intensity']),
        tp_max=float(tp_meta['max_intensity']),
        tp_percentile_min=0.0,  # Using actual min
        tp_percentile_max=100.0,  # Using actual max
    )
    
    print(f"Created normalization parameters from metadata:")
    print(f"  WF range: [{params.wf_min}, {params.wf_max}]")
    print(f"  2P range: [{params.tp_min}, {params.tp_max}]")
    
    return params


def analyze_current_normalization_issues(metadata_path: Union[str, Path]) -> None:
    """Analyze and report current normalization issues."""
    
    params = create_normalization_params_from_metadata(metadata_path)
    
    print("\n=== NORMALIZATION ANALYSIS ===")
    print(f"Raw WF range: [{params.wf_min}, {params.wf_max}]")
    print(f"Raw 2P range: [{params.tp_min}, {params.tp_max}]")
    
    # Show what current 16-bit normalization produces
    current_wf_min = (params.wf_min / 32767.5) - 1.0
    current_wf_max = (params.wf_max / 32767.5) - 1.0
    current_tp_min = (params.tp_min / 32767.5) - 1.0  
    current_tp_max = (params.tp_max / 32767.5) - 1.0
    
    print(f"\nCurrent 16-bit normalization results:")
    print(f"  WF: [{current_wf_min:.3f}, {current_wf_max:.3f}] (range: {current_wf_max - current_wf_min:.3f})")
    print(f"  2P: [{current_tp_min:.3f}, {current_tp_max:.3f}] (range: {current_tp_max - current_tp_min:.3f})")
    
    # Show what adaptive normalization would produce
    print(f"\nAdaptive normalization would produce:")
    print(f"  WF: [-1.000, 1.000] (range: 2.000)")
    print(f"  2P: [-1.000, 1.000] (range: 2.000)")
    
    # Calculate improvement
    wf_improvement = 2.0 / (current_wf_max - current_wf_min)
    tp_improvement = 2.0 / (current_tp_max - current_tp_min)
    
    print(f"\nDynamic range improvement:")
    print(f"  WF: {wf_improvement:.1f}x better")
    print(f"  2P: {tp_improvement:.1f}x better")


# =============================================================================
# Module Exports
# =============================================================================

# Export the constants and functions
__all__ = [
    # Constants
    'UINT16_MAX',
    'UINT16_MIN', 
    'MODEL_RANGE_MIN',
    'MODEL_RANGE_MAX',
    
    # Adaptive normalization classes
    'NormalizationParams',
    'AdaptiveNormalizer',
    
    # Adaptive normalization utility functions
    'create_normalization_params_from_metadata',
    'analyze_current_normalization_issues',
    
    # Legacy 16-bit functions
    'load_16bit_image',
    'normalize_16bit_to_model_input',
    'denormalize_model_output_to_16bit',
    'validate_16bit_range',
    'validate_model_range',
    'save_16bit_patch',
    'extract_and_save_patches_16bit',
    'resize_16bit_image',
    'create_16bit_test_image',
    'to_uint16_grayscale',
    'save_16bit_grayscale',
    'save_16bit_comparison'
]
