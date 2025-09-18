# Phase 2: Physics and Forward Model (Week 1-2)

### Step 2.1: Implement PSF Handling
```python
# pkl_dg/physics/psf.py
import torch
import numpy as np
from pathlib import Path
import tifffile

class PSF:
    """Point Spread Function handler for microscopy."""
    
    def __init__(self, psf_path: str = None, psf_array: np.ndarray = None):
        """
        Initialize PSF from file or array.
        
        Args:
            psf_path: Path to PSF TIFF file
            psf_array: Direct PSF array
        """
        if psf_path is not None:
            self.psf = self._load_psf(psf_path)
        elif psf_array is not None:
            self.psf = psf_array
        else:
            # Default Gaussian PSF
            self.psf = self._create_gaussian_psf()
        
        # Normalize PSF to sum to 1
        self.psf = self.psf / self.psf.sum()
    
    def _load_psf(self, path: str) -> np.ndarray:
        """Load PSF from TIFF file."""
        psf = tifffile.imread(path)
        if psf.ndim == 3:  # If 3D, take central slice
            psf = psf[psf.shape[0]//2]
        return psf.astype(np.float32)
    
    def _create_gaussian_psf(self, size: int = 15, sigma: float = 2.0) -> np.ndarray:
        """Create default Gaussian PSF."""
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        xx, yy = np.meshgrid(x, y)
        psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return psf.astype(np.float32)
    
    def to_torch(self, device: str = 'cpu', dtype: torch.dtype = torch.float32):
        """Convert PSF to torch tensor."""
        return torch.from_numpy(self.psf).to(device).to(dtype)
    
    def broaden(self, factor: float = 1.1):
        """Broaden PSF for robustness testing."""
        from scipy.ndimage import gaussian_filter
        sigma = (factor - 1.0) * 2.0  # Heuristic scaling
        broadened = gaussian_filter(self.psf, sigma)
        return PSF(psf_array=broadened)
```

### Step 2.2: Implement Forward Model
```python
# pkl_dg/physics/forward_model.py
import torch
import torch.nn.functional as F

class ForwardModel:
    """WF to 2P forward model with PSF convolution and noise."""
    
    def __init__(
        self,
        psf: torch.Tensor,
        background: float = 0.0,
        device: str = 'cuda'
    ):
        """
        Initialize forward model.
        
        Args:
            psf: Point spread function tensor
            background: Background intensity level
            device: Computation device
        """
        self.device = device
        self.background = background
        
        # Ensure PSF is shape [1, 1, H, W]
        self.psf = psf.to(device)
        if self.psf.ndim == 2:
            self.psf = self.psf.unsqueeze(0).unsqueeze(0)
    
    def _fft_convolve(self, x: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        """
        FFT-based convolution with proper padding of both image and PSF.
        """
        # Reflective padding to reduce boundary artifacts
        pad_h = psf.shape[-2] // 2
        pad_w = psf.shape[-1] // 2
        x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        
        # Pad PSF to padded image spatial size
        img_h, img_w = x_padded.shape[-2:]
        psf_padded = F.pad(
            psf,
            (
                0, img_w - psf.shape[-1],
                0, img_h - psf.shape[-2]
            )
        )
        
        # Frequency domain multiplication
        x_fft = torch.fft.rfft2(x_padded)
        psf_fft = torch.fft.rfft2(psf_padded)
        y_fft = x_fft * psf_fft
        y = torch.fft.irfft2(y_fft, s=(img_h, img_w))
        
        # Crop back to original size
        y = y[..., pad_h:-pad_h, pad_w:-pad_w]
        return y
    
    def apply_psf(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PSF convolution using FFT with padding."""
        return self._fft_convolve(x, self.psf)
    
    def apply_psf_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint (correlation) using flipped PSF and FFT."""
        psf_flipped = torch.flip(self.psf, dims=[-2, -1])
        return self._fft_convolve(y, psf_flipped)
    
    def forward(self, x: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        """
        Full forward model: A(x) + B [+ Poisson noise].
        """
        y = self.apply_psf(x)
        y = y + self.background
        if add_noise:
            y = torch.poisson(torch.clamp(y, min=0))
        return y
```

### Step 2.3: Implement Noise Models
```python
# pkl_dg/physics/noise.py
import torch

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
```
