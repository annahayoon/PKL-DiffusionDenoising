import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class SignalRecovery:
    """
    Signal Recovery forward model for SID (See in the Dark) scenarios.
    
    This class adapts the PKL framework for signal recovery tasks where the main
    degradation is severe underexposure (low gain) rather than optical blur.
    
    Forward model: y = gain * x + readout_noise
    where gain << 1 represents severe underexposure.
    """

    def __init__(
        self, 
        gain: float = 0.1, 
        readout_noise_sigma: float = 0.01,
        background: float = 0.0,
        device: str = "cuda"
    ):
        """
        Initialize signal recovery forward model.

        Args:
            gain: Multiplicative gain factor (< 1 for underexposure)
            readout_noise_sigma: Standard deviation of readout noise
            background: Background level (usually minimal in SID)
            device: Computation device
        """
        self.device = device
        self.gain = float(gain)
        self.background = float(background)
        self.readout_noise_sigma = float(readout_noise_sigma)
        
        # For compatibility with existing PKL guidance interface
        self.read_noise_sigma = readout_noise_sigma

    def apply_gain(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gain (multiplicative degradation)."""
        return self.gain * x

    def apply_gain_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint of gain operation (same as forward for scalar gain)."""
        return self.gain * y

    def forward(self, x: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        """
        Full forward model for signal recovery: y = gain * x + background + noise.
        
        Args:
            x: Clean image tensor [B, C, H, W]
            add_noise: Whether to add readout noise
            
        Returns:
            Degraded image tensor
        """
        # Apply gain (underexposure)
        y = self.apply_gain(x)
        
        # Add background
        y = y + self.background
        
        # Add readout noise if requested
        if add_noise and self.readout_noise_sigma > 0:
            noise = torch.randn_like(y) * self.readout_noise_sigma
            y = y + noise
            
        # Clamp to valid range
        y = torch.clamp(y, min=0.0)
        
        return y

    def condition_on_gain_params(self) -> torch.Tensor:
        """Return gain as conditioning parameter (for compatibility)."""
        return torch.tensor([self.gain], device=self.device)

    def get_cache_stats(self) -> dict:
        """Return statistics (for compatibility with ForwardModel interface)."""
        return {
            "gain": self.gain,
            "readout_noise_sigma": self.readout_noise_sigma,
            "background": self.background
        }

    # Compatibility methods to work with existing PKL guidance
    def apply_psf(self, x: torch.Tensor, use_kornia: bool = False) -> torch.Tensor:
        """Compatibility method: apply_psf -> apply_gain for signal recovery."""
        return self.apply_gain(x)

    def apply_psf_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Compatibility method: apply_psf_adjoint -> apply_gain_adjoint."""
        return self.apply_gain_adjoint(y)


class SIDForwardModel(SignalRecovery):
    """
    Specialized SignalRecovery class for SID (See in the Dark) datasets.
    
    Incorporates typical SID characteristics:
    - Very low gain (severe underexposure)
    - Camera-specific readout noise patterns
    - Optional color channel handling
    """

    def __init__(
        self,
        camera_type: str = "Sony",
        iso: int = 1600,
        exposure_ratio: float = 100.0,  # How much brighter the target should be
        device: str = "cuda"
    ):
        """
        Initialize SID-specific forward model.
        
        Args:
            camera_type: Camera type (Sony, Fuji, etc.)
            iso: ISO setting
            exposure_ratio: Ratio between target and input exposure
            device: Computation device
        """
        # Compute gain from exposure ratio
        gain = 1.0 / exposure_ratio
        
        # Camera-specific noise parameters
        noise_params = self._get_camera_noise_params(camera_type, iso)
        
        super().__init__(
            gain=gain,
            readout_noise_sigma=noise_params["readout_sigma"],
            background=noise_params["background"],
            device=device
        )
        
        self.camera_type = camera_type
        self.iso = iso
        self.exposure_ratio = exposure_ratio

    def _get_camera_noise_params(self, camera_type: str, iso: int) -> dict:
        """Get camera-specific noise parameters."""
        # Empirical parameters based on SID dataset characteristics
        params = {
            "Sony": {
                "readout_sigma": 0.01 * (iso / 1600) ** 0.5,
                "background": 0.001
            },
            "Fuji": {
                "readout_sigma": 0.012 * (iso / 1600) ** 0.5,
                "background": 0.0015
            }
        }
        
        return params.get(camera_type, params["Sony"])

    def simulate_sid_degradation(
        self, 
        clean_image: torch.Tensor,
        add_noise: bool = True
    ) -> torch.Tensor:
        """
        Simulate complete SID degradation pipeline.
        
        Args:
            clean_image: Clean, well-exposed image [B, C, H, W]
            add_noise: Whether to add camera noise
            
        Returns:
            Severely underexposed image simulating SID input
        """
        return self.forward(clean_image, add_noise=add_noise)

    def get_sid_metadata(self) -> dict:
        """Return SID-specific metadata."""
        return {
            "camera_type": self.camera_type,
            "iso": self.iso,
            "exposure_ratio": self.exposure_ratio,
            "effective_gain": self.gain,
            "readout_noise_sigma": self.readout_noise_sigma
        }


def create_sid_forward_model(
    camera_type: str = "Sony",
    iso: int = 1600,
    exposure_ratio: float = 100.0,
    device: str = "cuda"
) -> SIDForwardModel:
    """
    Factory function to create SID forward model.
    
    Args:
        camera_type: Camera type (Sony, Fuji)
        iso: ISO setting
        exposure_ratio: Target/input exposure ratio
        device: Computation device
        
    Returns:
        Configured SIDForwardModel instance
    """
    return SIDForwardModel(
        camera_type=camera_type,
        iso=iso,
        exposure_ratio=exposure_ratio,
        device=device
    )


# Utility functions for SID-specific operations
def normalize_sid_input(image: torch.Tensor, percentile: float = 99.5) -> torch.Tensor:
    """
    Normalize SID input images using percentile normalization.
    
    Args:
        image: Input image tensor
        percentile: Percentile for normalization
        
    Returns:
        Normalized image tensor
    """
    # Flatten spatial dimensions for percentile computation
    flat = image.view(image.shape[0], image.shape[1], -1)
    
    # Compute percentile per channel
    percentile_vals = torch.quantile(flat, percentile / 100.0, dim=2, keepdim=True)
    percentile_vals = percentile_vals.unsqueeze(-1)  # [B, C, 1, 1]
    
    # Normalize
    normalized = image / (percentile_vals + 1e-8)
    return torch.clamp(normalized, 0.0, 1.0)


def sid_to_rgb(raw_image: torch.Tensor, camera_type: str = "Sony") -> torch.Tensor:
    """
    Convert SID RAW format to RGB (simplified version).
    
    Args:
        raw_image: RAW image tensor [B, 1, H, W] or [B, 4, H, W]
        camera_type: Camera type for color matrix
        
    Returns:
        RGB image tensor [B, 3, H, W]
    """
    if raw_image.shape[1] == 1:
        # Grayscale case - replicate to 3 channels
        return raw_image.repeat(1, 3, 1, 1)
    elif raw_image.shape[1] == 4:
        # RGGB Bayer pattern - simple demosaicing
        r = raw_image[:, 0:1, :, :]  # Red
        g = (raw_image[:, 1:2, :, :] + raw_image[:, 2:3, :, :]) / 2  # Average greens
        b = raw_image[:, 3:4, :, :]  # Blue
        
        return torch.cat([r, g, b], dim=1)
    else:
        # Already RGB or other format
        return raw_image[:, :3, :, :] if raw_image.shape[1] > 3 else raw_image
