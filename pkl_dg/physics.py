"""Physical modeling components: PSF, forward operator, and noise models."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import tifffile
from PIL import Image
import yaml
from tqdm import tqdm

UINT16_MAX = 65535


try:
    import kornia.filters as K_filters
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False


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
            self.psf = self._create_gaussian_psf()

        
        s = float(self.psf.sum())
        if s == 0.0:
            self.psf = self._create_gaussian_psf()
        else:
            self.psf = self.psf / s

    def _load_psf(self, path: str) -> np.ndarray:
        """Load PSF from TIFF file - handles 16-bit microscopy images."""
        psf = tifffile.imread(path)
        if psf.ndim == 3:
            psf = psf[psf.shape[0] // 2]
        
        
        if psf.dtype == np.uint16:
            psf = psf.astype(np.float32)
        elif psf.dtype == np.uint8:
            raise ValueError(f"8-bit PSF images are not supported. PSF file {path} should be 16-bit TIFF format.")
        else:
            psf = psf.astype(np.float32)
        
        if psf.max() <= 1.0:
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


def _load_grayscale_images(dir_path: Path) -> List[np.ndarray]:
    """Load grayscale images from a directory - only supports 16-bit TIFF files."""
    images: List[np.ndarray] = []
    if not dir_path.exists():
        return images
    for ext in ("*.tif", "*.tiff"):
        for p in dir_path.glob(ext):
            try:
                import tifffile
                arr = tifffile.imread(p)
                if arr.ndim == 3:
                    arr = arr.mean(axis=-1)
                if arr.dtype == np.uint8:
                    print(f"Warning: Skipping 8-bit image {p}. Only 16-bit TIFF files are supported.")
                    continue
                images.append(arr.astype(np.float32))
            except ImportError:
                print(f"Warning: tifffile not available, skipping {p}")
                continue
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
    
    modes = {"with_AO": ["with_AO", "with-ao", "withao"]}
    
    # First try to find subdirectories
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
    
    # If no subdirectories found, look for direct TIFF files in the beads directory
    if not bank:
        # Map specific filenames to modes
        file_mappings = {
            "with_AO": [
                "*after_AO*.tif", "*after_AO*.tiff", 
                "*with_AO*.tif", "*with_AO*.tiff",
                "*withAO*.tif", "*withAO*.tiff"
            ],
            "no_AO": [
                "*no_AO*.tif", "*no_AO*.tiff",
                "*noAO*.tif", "*noAO*.tiff",
                "*without_AO*.tif", "*without_AO*.tiff"
            ]
        }
        
        for key, patterns in file_mappings.items():
            for pattern in patterns:
                tiff_files = list(root.glob(pattern))
                if tiff_files:
                    # Load the TIFF file directly (it should be a 3D stack)
                    try:
                        import tifffile
                        bead_stack = tifffile.imread(str(tiff_files[0]))
                        
                        # Take middle slice for 2D PSF
                        if bead_stack.ndim == 3:
                            middle_slice = bead_stack[bead_stack.shape[0] // 2]
                        else:
                            middle_slice = bead_stack
                        
                        # Process as single bead image
                        psf_np = estimate_psf_from_beads([middle_slice.astype(np.float32)])
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
                        print(f"Loaded {key} PSF from {tiff_files[0].name}")
                        break
                        
                    except ImportError:
                        print("Warning: tifffile not available for direct TIFF loading")
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to load {tiff_files[0]}: {e}")
                        continue
    
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
    
    # Pixel size information is stored in PSF tensors for internal use
    
    return bank


def psf_params_from_tensor(psf: torch.Tensor) -> Tuple[float, float]:
    """Compute (sigma_x, sigma_y) from PSF tensor via second moments."""
    arr = psf.detach().cpu().numpy().astype(np.float32)
    sx, sy = fit_second_moments_sigma(arr)
    return float(sx), float(sy)


def psf_from_config(physics_config: dict | object) -> Tuple[PSF, Optional[float]]:
    """Create a PSF from a config-like object and return (PSF, target_pixel_size_xy_nm).

    The function handles these cases:
    - physics.use_psf and physics.use_bead_psf: build PSF from bead images via build_psf_bank
    - physics.use_psf with physics.psf_path: load PSF from file
    - otherwise: optional Gaussian fallback using cfg.psf fields or default PSF

    The returned PSF may carry pixel size metadata if available. The second return
    value is the configured target pixel size in nm if present, otherwise None.
    """
    # Access helpers for nested attributes/dicts
    def _get(obj, key, default=None):
        try:
            # OmegaConf-style or dict
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        except Exception:
            return default

    use_psf = bool(_get(physics_config, "use_psf", False))
    use_bead_psf = bool(_get(physics_config, "use_bead_psf", False))

    # Default result
    psf_obj: Optional[PSF] = None

    if use_psf and use_bead_psf:
        beads_dir = _get(physics_config, "beads_dir", "data/beads")
        try:
            bank = build_psf_bank(beads_dir)
            bead_mode = _get(physics_config, "bead_mode", None)
            if bead_mode and bead_mode in bank:
                psf_tensor = bank[bead_mode]
            elif "with_AO" in bank:
                psf_tensor = bank["with_AO"]
            elif "no_AO" in bank:
                psf_tensor = bank["no_AO"]
            else:
                psf_tensor = next(iter(bank.values()))

            psf_array = psf_tensor.detach().cpu().numpy().astype(np.float32)
            pixel_size_xy_nm = getattr(psf_tensor, "pixel_size_xy_nm", None)
            psf_obj = PSF(psf_array=psf_array, pixel_size_xy_nm=pixel_size_xy_nm)
        except Exception:
            psf_obj = None

    if psf_obj is None and use_psf:
        psf_path = _get(physics_config, "psf_path", None)
        if psf_path:
            psf_obj = PSF(psf_path=psf_path)
        else:
            psf_obj = PSF()

    if psf_obj is None:
        psf_cfg = _get(physics_config, "psf", {})
        try:
            # Expect a dict-like with type/size/sigma_x/sigma_y
            psf_type = _get(psf_cfg, "type", "gaussian")
            if psf_type == "gaussian":
                size = int(_get(psf_cfg, "size", 21))
                sigma_x = float(_get(psf_cfg, "sigma_x", 2.0))
                sigma_y = float(_get(psf_cfg, "sigma_y", 2.0))
                x = np.arange(size) - size // 2
                y = np.arange(size) - size // 2
                xx, yy = np.meshgrid(x, y)
                psf_array = np.exp(-(xx ** 2 / (2 * sigma_x ** 2) + yy ** 2 / (2 * sigma_y ** 2)))
                psf_obj = PSF(psf_array=psf_array.astype(np.float32))
            else:
                psf_obj = PSF()
        except Exception:
            psf_obj = PSF()

    target_pixel_size_xy_nm = _get(physics_config, "target_pixel_size_xy_nm", None)
    if target_pixel_size_xy_nm is not None:
        try:
            target_pixel_size_xy_nm = float(target_pixel_size_xy_nm)
        except Exception:
            target_pixel_size_xy_nm = None

    return psf_obj, target_pixel_size_xy_nm


class ForwardModel:
    """WF to 2P forward model with PSF convolution and noise."""

    def __init__(self, psf: torch.Tensor, background: float = 0.0, device: str = "cuda", 
                 common_sizes: Optional[list] = None, read_noise_sigma: float = 0.0,
                 psf_pixel_size_xy_nm: float = None, target_pixel_size_xy_nm: float = None):
        """
        Initialize forward model.

        Args:
            psf: Point spread function tensor
            background: Background intensity level
            device: Computation device
            common_sizes: List of (height, width) tuples for pre-computing FFTs
            read_noise_sigma: Read noise standard deviation
            psf_pixel_size_xy_nm: Original PSF pixel size in nm (if known)
            target_pixel_size_xy_nm: Target image pixel size in nm (if different from PSF)
        """
        self.device = device
        self.background = background
        self.read_noise_sigma = float(read_noise_sigma)
        
        
        self.psf_pixel_size_xy_nm = psf_pixel_size_xy_nm
        self.target_pixel_size_xy_nm = target_pixel_size_xy_nm

        processed_psf = self._process_psf_for_pixel_size(psf)
        
        
        self.psf = processed_psf.to(device)
        if self.psf.ndim == 2:
            self.psf = self.psf.unsqueeze(0).unsqueeze(0)
        self._psf_fft_cache = {}
        self._psf_fft_cache_base = {}
        
        
        if common_sizes is None:
            common_sizes = [(256, 256), (512, 512), (128, 128), (64, 64), (1024, 1024)]
        
        self._precompute_common_ffts(common_sizes)
    
    def _process_psf_for_pixel_size(self, psf: torch.Tensor) -> torch.Tensor:
        """Process PSF to handle pixel size scaling if needed."""
        if hasattr(psf, 'pixel_size_xy_nm'):
            self.psf_pixel_size_xy_nm = psf.pixel_size_xy_nm
        
        
        if (self.psf_pixel_size_xy_nm is not None and 
            self.target_pixel_size_xy_nm is not None and
            abs(self.psf_pixel_size_xy_nm - self.target_pixel_size_xy_nm) > 0.01):
            
            print(f"ForwardModel: Scaling PSF from {self.psf_pixel_size_xy_nm} nm "
                  f"to {self.target_pixel_size_xy_nm} nm")
            
            
            psf_np = psf.detach().cpu().numpy()
            psf_obj = PSF(psf_array=psf_np, pixel_size_xy_nm=self.psf_pixel_size_xy_nm)
            scaled_psf_obj = psf_obj.scale_for_pixel_size(self.target_pixel_size_xy_nm)
            return torch.from_numpy(scaled_psf_obj.psf)
        
        return psf

    def _precompute_common_ffts(self, common_sizes: list) -> None:
        """Pre-compute PSF FFTs for common image sizes to improve runtime performance."""
        # Only precompute base FFTs (device-agnostic, high precision)
        # Device-specific FFTs will be materialized on-demand
        for height, width in tqdm(common_sizes, desc="Pre-computing PSF FFTs", leave=False):
            try:
                # Pre-compute base FFT (CPU, float32) for this size
                base_key = (height, width)
                if base_key not in self._psf_fft_cache_base:
                    # Force computation of base FFT
                    self._get_psf_fft(height, width, torch.float32, torch.device('cpu'))
            except Exception:
                # Skip if there are memory or compatibility issues
                continue

    def set_psf(self, psf: torch.Tensor, common_sizes: Optional[list] = None) -> None:
        """Update PSF and clear cached FFTs to maintain correctness.

        Args:
            psf: New PSF tensor, will be moved to model device and shaped to [1,1,H,W]
            common_sizes: Optional list of sizes to pre-compute FFTs for
        """
        self.psf = psf.to(self.device)
        if self.psf.ndim == 2:
            self.psf = self.psf.unsqueeze(0).unsqueeze(0)
        # Invalidate all caches since kernel changed
        self._psf_fft_cache.clear()
        self._psf_fft_cache_base.clear()
        
        # Re-precompute common FFTs for new PSF
        if common_sizes is None:
            common_sizes = [(256, 256), (512, 512), (128, 128), (64, 64), (1024, 1024)]
        self._precompute_common_ffts(common_sizes)

    def _get_psf_fft(self, height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return cached PSF FFT for given size/dtype or compute and store it.

        Optimizations:
        - Maintain a high-precision CPU base cache per (H, W)
        - Materialize device/dtype-specific tensors from the base cache to avoid recomputation
        """
        device_key = (height, width, str(dtype), str(device))
        cached_dev = self._psf_fft_cache.get(device_key)
        if cached_dev is not None:
            return cached_dev

        # Try base cache first (CPU, high precision)
        base_key = (height, width)
        base_fft = self._psf_fft_cache_base.get(base_key)

        if base_fft is None:
            # Build base FFT on CPU in float32 for stability
            psf_h, psf_w = self.psf.shape[-2:]
            psf_padded = torch.zeros((1, 1, height, width), device="cpu", dtype=torch.float32)
            psf_src = self.psf.to(device="cpu", dtype=torch.float32)

            # If image smaller than PSF, center-crop PSF; otherwise place PSF then roll to center
            if height < psf_h or width < psf_w:
                start_h = max((psf_h - height) // 2, 0)
                start_w = max((psf_w - width) // 2, 0)
                end_h = start_h + min(psf_h, height)
                end_w = start_w + min(psf_w, width)
                cropped = psf_src[..., start_h:end_h, start_w:end_w]
                ph, pw = cropped.shape[-2:]
                psf_padded[..., :ph, :pw] = cropped
                psf_padded = torch.roll(psf_padded, shifts=(-(ph // 2), -(pw // 2)), dims=(-2, -1))
            else:
                psf_padded[..., :psf_h, :psf_w] = psf_src
                psf_padded = torch.roll(psf_padded, shifts=(-(psf_h // 2), -(psf_w // 2)), dims=(-2, -1))

            base_fft = torch.fft.rfft2(psf_padded)
            self._psf_fft_cache_base[base_key] = base_fft

        # Materialize for target device (cast at multiplication time if needed)
        dev_fft = base_fft.to(device)
        self._psf_fft_cache[device_key] = dev_fft
        return dev_fft

    def _fft_convolve(self, x: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        """
        Circular convolution via FFT with kernel centered at origin.
        Ensures discrete adjointness with correlation for apply_psf_adjoint.
        """
        _, _, img_h, img_w = x.shape
        x_fft = torch.fft.rfft2(x)
        psf_fft = self._get_psf_fft(img_h, img_w, x.dtype, x.device)
        y_fft = x_fft * psf_fft
        y = torch.fft.irfft2(y_fft, s=(img_h, img_w))
        return y

    def apply_psf(self, x: torch.Tensor, use_kornia: bool = False) -> torch.Tensor:
        """Apply PSF convolution using FFT with padding or Kornia filters."""
        if use_kornia and KORNIA_AVAILABLE:
            return self._kornia_convolve(x, self.psf)
        return self._fft_convolve(x, self.psf)

    def _kornia_convolve(self, x: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        """Alternative convolution using Kornia for differentiable operations."""
        if not KORNIA_AVAILABLE:
            raise ImportError("Kornia not available. Use FFT convolution instead.")
        
        # Kornia expects kernel in shape [1, 1, H, W] 
        kernel = psf if psf.ndim == 4 else psf.unsqueeze(0).unsqueeze(0)
        
        # Use kornia's filter2d which handles padding automatically
        return K_filters.filter2d(x, kernel, border_type='reflect')

    def apply_psf_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint (correlation) using conjugate multiplication in Fourier domain."""
        _, _, img_h, img_w = y.shape
        y_fft = torch.fft.rfft2(y)
        psf_fft = self._get_psf_fft(img_h, img_w, y.dtype, y.device)
        at_fft = y_fft * torch.conj(psf_fft)
        at = torch.fft.irfft2(at_fft, s=(img_h, img_w))
        return at

    def forward(self, x: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        """
        Full forward model: A(x) + B [+ Poisson noise].
        """
        y = self.apply_psf(x)
        y = y + self.background
        if add_noise:
            y = torch.poisson(torch.clamp(y, min=0))
        return y

    def clear_cache(self) -> None:
        """Clear all FFT caches to free memory."""
        self._psf_fft_cache.clear()
        self._psf_fft_cache_base.clear()
    
    def get_cache_stats(self) -> dict:
        """Return cache statistics for testing and diagnostics."""
        # Calculate memory usage
        base_memory = sum(
            fft.numel() * fft.element_size() 
            for fft in self._psf_fft_cache_base.values()
        ) / 1e6  # MB
        
        device_memory = sum(
            fft.numel() * fft.element_size() 
            for fft in self._psf_fft_cache.values()
        ) / 1e6  # MB
        
        return {
            "base_entries": len(self._psf_fft_cache_base),
            "device_entries": len(self._psf_fft_cache),
            "base_memory_mb": base_memory,
            "device_memory_mb": device_memory,
            "total_memory_mb": base_memory + device_memory,
        }
    
    def optimize_cache_memory(self, max_memory_mb: float = 512) -> None:
        """Optimize cache memory usage by removing least recently used entries."""
        current_stats = self.get_cache_stats()
        
        if current_stats["total_memory_mb"] > max_memory_mb:
            # Simple strategy: clear device cache (can be regenerated from base)
            num_cleared = len(self._psf_fft_cache)
            self._psf_fft_cache.clear()
            print(f"Cleared {num_cleared} device FFT cache entries to save memory")
    
    def batch_apply_psf(self, x_batch: torch.Tensor, use_kornia: bool = False, 
                       chunk_size: int = None) -> torch.Tensor:
        """Apply PSF to a batch of images efficiently with optional chunking.
        
        Args:
            x_batch: Input batch [B, C, H, W]
            use_kornia: Whether to use Kornia for convolution
            chunk_size: Process in chunks to manage memory (None = auto-detect)
            
        Returns:
            PSF-convolved batch [B, C, H, W]
        """
        if x_batch.dim() != 4:
            raise ValueError(f"Expected 4D batch tensor, got {x_batch.dim()}D")
        
        batch_size = x_batch.shape[0]
        
        if batch_size == 1:
            # Single image - use regular method
            return self.apply_psf(x_batch, use_kornia)
        
        # Auto-detect chunk size based on available memory
        if chunk_size is None:
            if torch.cuda.is_available():
                # Estimate memory per image and available memory
                img_memory = x_batch[0].numel() * x_batch.element_size() / 1e9  # GB
                available_memory = torch.cuda.get_device_properties(0).total_memory * 0.5 / 1e9
                chunk_size = max(1, int(available_memory / (img_memory * 4)))  # 4x overhead
                chunk_size = min(chunk_size, batch_size)
            else:
                chunk_size = min(8, batch_size)  # Conservative CPU default
        
        # Process in chunks if needed
        if chunk_size < batch_size:
            results = []
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk = x_batch[i:end_idx]
                result_chunk = self._batch_apply_psf_chunk(chunk, use_kornia)
                results.append(result_chunk)
            return torch.cat(results, dim=0)
        else:
            return self._batch_apply_psf_chunk(x_batch, use_kornia)
    
    def _batch_apply_psf_chunk(self, x_batch: torch.Tensor, use_kornia: bool = False) -> torch.Tensor:
        """Apply PSF to a chunk of images."""
        # For batch processing, we can use the same FFT for all images in the batch
        if use_kornia and KORNIA_AVAILABLE:
            return self._kornia_convolve(x_batch, self.psf)
        
        # Batch FFT convolution - vectorized for efficiency
        _, _, img_h, img_w = x_batch.shape
        
        # Use torch.fft.rfft2 which is optimized for batch operations
        x_fft = torch.fft.rfft2(x_batch, dim=(-2, -1))
        psf_fft = self._get_psf_fft(img_h, img_w, x_batch.dtype, x_batch.device)
        
        # Broadcast PSF FFT to batch dimension efficiently
        y_fft = x_fft * psf_fft.unsqueeze(0)
        y = torch.fft.irfft2(y_fft, s=(img_h, img_w), dim=(-2, -1))
        return y
    
    def parallel_apply_psf_cpu(self, x_list: list, num_workers: int = None) -> list:
        """Apply PSF to multiple images in parallel using CPU multiprocessing.
        
        Args:
            x_list: List of input tensors
            num_workers: Number of worker processes
            
        Returns:
            List of PSF-convolved tensors
        """
        if num_workers is None:
            import os
            num_workers = min(8, os.cpu_count() or 1)
        
        if len(x_list) <= 1 or num_workers <= 1:
            return [self.apply_psf(x) for x in x_list]
        
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        
        # Create a worker function that can be pickled
        def worker_apply_psf(args):
            x, psf_data, background, device_str = args
            # Recreate forward model in worker process
            worker_fm = ForwardModel(psf_data, background, device_str)
            return worker_fm.apply_psf(x)
        
        # Prepare arguments for workers
        psf_data = self.psf.cpu()  # Move PSF to CPU for serialization
        args_list = [(x.cpu(), psf_data, self.background, "cpu") for x in x_list]
        
        # Process in parallel
        try:
            with ProcessPoolExecutor(max_workers=num_workers, 
                                   mp_context=mp.get_context('spawn')) as executor:
                results = list(executor.map(worker_apply_psf, args_list))
            
            # Move results back to original device if needed
            if x_list[0].device != torch.device('cpu'):
                results = [r.to(x_list[0].device) for r in results]
            
            return results
        except Exception as e:
            # Fallback to sequential processing
            print(f"Parallel processing failed: {e}. Falling back to sequential.")
            return [self.apply_psf(x) for x in x_list]


class PoissonNoise:
    """Poisson noise model for photon-limited imaging."""

    @staticmethod
    def add_noise(signal: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        signal = torch.clamp(signal, min=0)
        signal_scaled = signal * gain
        noisy = torch.poisson(signal_scaled) / gain
        return noisy


class GaussianBackground:
    """Gaussian background noise model."""

    @staticmethod
    def add_background(signal: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        noise = torch.randn_like(signal) * std + mean
        return signal + noise


__all__ = [
    "PSF",
    "ForwardModel",
    "PoissonNoise",
    "GaussianBackground",
    "_load_grayscale_images",
    "estimate_psf_from_beads",
    "build_psf_bank",
    "psf_params_from_tensor",
    "fit_second_moments_sigma",
]
