import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tifffile
from PIL import Image
import yaml

from ..utils import convert_8bit_to_16bit_equivalent, UINT16_MAX


class PSF:
    """Point Spread Function handler for microscopy."""

    def __init__(self, psf_path: str = None, psf_array: np.ndarray = None, 
                 pixel_size_xy_nm: float = None, pixel_size_z_nm: float = None):
        """
        Initialize PSF from file or array.

        Args:
            psf_path: Path to PSF TIFF file
            psf_array: Direct PSF array
            pixel_size_xy_nm: XY pixel size in nanometers
            pixel_size_z_nm: Z pixel size in nanometers
        """
        # Store pixel size information
        self.pixel_size_xy_nm = pixel_size_xy_nm
        self.pixel_size_z_nm = pixel_size_z_nm
        
        if psf_path is not None:
            self.psf = self._load_psf(psf_path)
        elif psf_array is not None:
            arr = psf_array
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            if arr.ndim == 3:
                arr = arr[arr.shape[0] // 2]
            self.psf = arr.astype(np.float32)
        else:
            # Default Gaussian PSF
            self.psf = self._create_gaussian_psf()

        # Normalize PSF to sum to 1
        s = float(self.psf.sum())
        if s == 0.0:
            # Fallback to default Gaussian if provided PSF is degenerate
            self.psf = self._create_gaussian_psf()
        else:
            self.psf = self.psf / s

    def _load_psf(self, path: str) -> np.ndarray:
        """Load PSF from TIFF file - handles 16-bit microscopy images."""
        psf = tifffile.imread(path)
        if psf.ndim == 3:  # If 3D, take central slice
            psf = psf[psf.shape[0] // 2]
        
        # Handle 16-bit images properly
        if psf.dtype == np.uint16:
            psf = psf.astype(np.float32)
        elif psf.dtype == np.uint8:
            # Convert 8-bit to 16-bit equivalent range
            psf = convert_8bit_to_16bit_equivalent(psf)
        else:
            psf = psf.astype(np.float32)
        
        # Ensure PSF is in proper range for 16-bit processing
        if psf.max() <= 1.0:
            # Normalized PSF, scale to 16-bit range
            psf = psf * UINT16_MAX
        
        return psf

    def _create_gaussian_psf(self, size: int = 15, sigma: float = 2.0) -> np.ndarray:
        """Create default Gaussian PSF."""
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        xx, yy = np.meshgrid(x, y)
        psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        return psf.astype(np.float32)

    def to_torch(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        """Convert PSF to torch tensor."""
        return torch.from_numpy(self.psf).to(device).to(dtype)

    def broaden(self, factor: float = 1.1):
        """Broaden PSF for robustness testing."""
        from scipy.ndimage import gaussian_filter

        sigma = (factor - 1.0) * 2.0  # Heuristic scaling
        broadened = gaussian_filter(self.psf, sigma)
        return PSF(psf_array=broadened, pixel_size_xy_nm=self.pixel_size_xy_nm, 
                  pixel_size_z_nm=self.pixel_size_z_nm)
    
    def scale_for_pixel_size(self, target_pixel_size_xy_nm: float, 
                           target_pixel_size_z_nm: float = None) -> 'PSF':
        """
        Scale PSF to match different pixel sizes.
        
        Args:
            target_pixel_size_xy_nm: Target XY pixel size in nanometers
            target_pixel_size_z_nm: Target Z pixel size in nanometers (optional)
            
        Returns:
            New PSF object scaled for the target pixel sizes
        """
        if self.pixel_size_xy_nm is None:
            print("Warning: PSF has no calibrated pixel size information. Cannot scale.")
            return PSF(psf_array=self.psf.copy(), 
                      pixel_size_xy_nm=target_pixel_size_xy_nm,
                      pixel_size_z_nm=target_pixel_size_z_nm)
        
        # Calculate scaling factor
        scale_factor = self.pixel_size_xy_nm / target_pixel_size_xy_nm
        
        if abs(scale_factor - 1.0) < 0.01:  # Less than 1% difference
            print(f"PSF pixel size ({self.pixel_size_xy_nm} nm) is close to target "
                  f"({target_pixel_size_xy_nm} nm). No scaling needed.")
            return PSF(psf_array=self.psf.copy(), 
                      pixel_size_xy_nm=target_pixel_size_xy_nm,
                      pixel_size_z_nm=target_pixel_size_z_nm)
        
        print(f"Scaling PSF from {self.pixel_size_xy_nm} nm to {target_pixel_size_xy_nm} nm "
              f"(scale factor: {scale_factor:.3f})")
        
        return self._resize_psf(scale_factor, target_pixel_size_xy_nm, target_pixel_size_z_nm)
    
    def _resize_psf(self, scale_factor: float, target_xy_nm: float, target_z_nm: float) -> 'PSF':
        """Resize PSF by the given scale factor."""
        try:
            from scipy.ndimage import zoom
            
            # If scale_factor > 1, PSF needs to be larger (finer pixels)
            # If scale_factor < 1, PSF needs to be smaller (coarser pixels)
            scaled_psf = zoom(self.psf, scale_factor, order=1)  # Linear interpolation
            
            # Normalize after scaling
            scaled_psf = scaled_psf / (scaled_psf.sum() + 1e-12)
            
            return PSF(psf_array=scaled_psf, 
                      pixel_size_xy_nm=target_xy_nm,
                      pixel_size_z_nm=target_z_nm)
            
        except ImportError:
            # Fallback: use the existing PSF with a warning
            print("Warning: scipy not available for PSF scaling. Using original PSF.")
            return PSF(psf_array=self.psf.copy(),
                      pixel_size_xy_nm=target_xy_nm,
                      pixel_size_z_nm=target_z_nm)


# PSF Estimation Functions (formerly in psf_estimator.py)

def _load_grayscale_images(dir_path: Path) -> List[np.ndarray]:
    """Load grayscale images from a directory."""
    images: List[np.ndarray] = []
    if not dir_path.exists():
        return images
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        for p in dir_path.glob(ext):
            arr = np.array(Image.open(p))
            if arr.ndim == 3:
                arr = arr.mean(axis=-1)
            images.append(arr.astype(np.float32))
    return images


def _center_crop(img: np.ndarray, size: int = 33) -> np.ndarray:
    """Center crop image around brightest point."""
    h, w = img.shape[:2]
    cy, cx = np.unravel_index(np.argmax(img), img.shape)
    half = size // 2
    y0 = max(cy - half, 0)
    x0 = max(cx - half, 0)
    y1 = min(y0 + size, h)
    x1 = min(x0 + size, w)
    crop = img[y0:y1, x0:x1]
    # pad if near boundaries
    if crop.shape[0] != size or crop.shape[1] != size:
        pad_y = size - crop.shape[0]
        pad_x = size - crop.shape[1]
        crop = np.pad(crop, ((0, pad_y), (0, pad_x)), mode="constant")
    return crop


def _normalize_unit_sum(psf: np.ndarray) -> np.ndarray:
    """Normalize PSF to sum to 1."""
    s = float(psf.sum())
    if s <= 0:
        return psf
    return psf / s


def estimate_psf_from_beads(bead_images: List[np.ndarray], crop_size: int = 33) -> np.ndarray:
    """Estimate 2D PSF by centering, cropping, and averaging bead images."""
    if not bead_images:
        raise ValueError("No bead images provided for PSF estimation")
    crops = []
    for img in bead_images:
        img = img.astype(np.float32)
        # background subtract using median
        img = img - float(np.median(img))
        img = np.clip(img, 0, None)
        crop = _center_crop(img, size=crop_size)
        crop = crop / (crop.max() + 1e-8)
        crops.append(crop)
    psf = np.mean(np.stack(crops, axis=0), axis=0)
    psf = np.clip(psf, 0, None)
    psf = _normalize_unit_sum(psf)
    return psf.astype(np.float32)


def fit_second_moments_sigma(psf: np.ndarray) -> Tuple[float, float]:
    """Estimate Gaussian sigma_x, sigma_y from second moments of PSF."""
    psf = np.clip(psf, 0, None)
    psf = _normalize_unit_sum(psf)
    h, w = psf.shape
    yy, xx = np.mgrid[0:h, 0:w]
    y0 = (psf * yy).sum()
    x0 = (psf * xx).sum()
    var_y = (psf * (yy - y0) ** 2).sum()
    var_x = (psf * (xx - x0) ** 2).sum()
    sigma_y = float(np.sqrt(max(var_y, 1e-8)))
    sigma_x = float(np.sqrt(max(var_x, 1e-8)))
    return sigma_x, sigma_y


def _load_bead_metadata(bead_root: Path) -> Dict:
    """Load bead metadata from YAML file."""
    metadata_path = bead_root / "bead_metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def build_psf_bank(bead_root: str | Path) -> Dict[str, torch.Tensor]:
    """Build a PSF bank from bead data. Expects subdirs like 'with_AO' and 'no_AO'."""
    root = Path(bead_root)
    bank: Dict[str, torch.Tensor] = {}
    
    # Load metadata if available
    metadata = _load_bead_metadata(root)
    pixel_info = {}
    
    # Extract pixel size information from metadata
    if metadata and "bead_data" in metadata and "with_AO" in metadata["bead_data"]:
        imaging_params = metadata["bead_data"]["with_AO"].get("imaging_parameters", {})
        pixel_size = imaging_params.get("pixel_size", {})
        if pixel_size:
            pixel_info = {
                "xy_nm": pixel_size.get("xy_nm"),
                "z_nm": pixel_size.get("z_nm"),
                "xy_um": pixel_size.get("xy_um"),
                "z_um": pixel_size.get("z_um")
            }
    
    modes = {"with_AO": ["with_AO", "with-ao", "withao"], "no_AO": ["no_AO", "no-ao", "noao"]}
    for key, aliases in modes.items():
        for alias in aliases:
            dir_path = root / alias
            imgs = _load_grayscale_images(dir_path)
            if imgs:
                psf_np = estimate_psf_from_beads(imgs)
                psf_tensor = torch.from_numpy(psf_np)
                
                # Store pixel size as tensor attributes if available
                if pixel_info.get("xy_nm") is not None:
                    psf_tensor.pixel_size_xy_nm = float(pixel_info["xy_nm"])
                if pixel_info.get("z_nm") is not None:
                    psf_tensor.pixel_size_z_nm = float(pixel_info["z_nm"])
                if pixel_info.get("xy_um") is not None:
                    psf_tensor.pixel_size_xy_um = float(pixel_info["xy_um"])
                if pixel_info.get("z_um") is not None:
                    psf_tensor.pixel_size_z_um = float(pixel_info["z_um"])
                
                bank[key] = psf_tensor
                break
    
    # Fallback: if only one found, clone to the other key
    if "with_AO" in bank and "no_AO" not in bank:
        bank["no_AO"] = bank["with_AO"].clone()
        # Copy pixel size attributes
        if hasattr(bank["with_AO"], "pixel_size_xy_nm"):
            bank["no_AO"].pixel_size_xy_nm = bank["with_AO"].pixel_size_xy_nm
        if hasattr(bank["with_AO"], "pixel_size_z_nm"):
            bank["no_AO"].pixel_size_z_nm = bank["with_AO"].pixel_size_z_nm
        if hasattr(bank["with_AO"], "pixel_size_xy_um"):
            bank["no_AO"].pixel_size_xy_um = bank["with_AO"].pixel_size_xy_um
        if hasattr(bank["with_AO"], "pixel_size_z_um"):
            bank["no_AO"].pixel_size_z_um = bank["with_AO"].pixel_size_z_um
            
    if "no_AO" in bank and "with_AO" not in bank:
        bank["with_AO"] = bank["no_AO"].clone()
        # Copy pixel size attributes
        if hasattr(bank["no_AO"], "pixel_size_xy_nm"):
            bank["with_AO"].pixel_size_xy_nm = bank["no_AO"].pixel_size_xy_nm
        if hasattr(bank["no_AO"], "pixel_size_z_nm"):
            bank["with_AO"].pixel_size_z_nm = bank["no_AO"].pixel_size_z_nm
        if hasattr(bank["no_AO"], "pixel_size_xy_um"):
            bank["with_AO"].pixel_size_xy_um = bank["no_AO"].pixel_size_xy_um
        if hasattr(bank["no_AO"], "pixel_size_z_um"):
            bank["with_AO"].pixel_size_z_um = bank["no_AO"].pixel_size_z_um
    
    if not bank:
        raise ValueError(f"No bead images found under {root}")
    
    # Print pixel size information if available
    for mode, psf_tensor in bank.items():
        if hasattr(psf_tensor, "pixel_size_xy_nm"):
            print(f"PSF {mode}: XY pixel size = {psf_tensor.pixel_size_xy_nm} nm, "
                  f"Z pixel size = {getattr(psf_tensor, 'pixel_size_z_nm', 'N/A')} nm")
    
    return bank


def psf_params_from_tensor(psf: torch.Tensor) -> Tuple[float, float]:
    """Compute (sigma_x, sigma_y) from PSF tensor via second moments."""
    arr = psf.detach().cpu().numpy().astype(np.float32)
    sx, sy = fit_second_moments_sigma(arr)
    return float(sx), float(sy)

