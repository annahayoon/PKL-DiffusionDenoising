"""
Loss functions for diffusion models.

This module consolidates various loss functions used in diffusion training,
including frequency domain losses and consistency losses.

Features:
- Basic losses (MSE, L1, Huber)
- Frequency domain losses (Fourier, Wavelet, Spectral)
- Multi-scale frequency consistency
- High-frequency detail preservation
- Composite loss combinations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
import numpy as np
import math

try:
    import torchvision.models as models
    from torchvision.transforms import Normalize
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


class MSELoss(nn.Module):
    """Standard MSE loss for noise prediction."""
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predicted, target, reduction=self.reduction)


class L1Loss(nn.Module):
    """L1 loss for noise prediction."""
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(predicted, target, reduction=self.reduction)


class HuberLoss(nn.Module):
    """Huber loss for robust noise prediction."""
    
    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(predicted, target, delta=self.delta, reduction=self.reduction)


class FourierLoss(nn.Module):
    """Loss in Fourier domain for preserving frequency characteristics."""
    
    def __init__(self, 
                 loss_type: str = "l1",
                 log_scale: bool = True,
                 high_freq_weight: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.log_scale = log_scale
        self.high_freq_weight = high_freq_weight
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss in Fourier domain."""
        # Compute FFT
        pred_fft = torch.fft.fft2(predicted)
        target_fft = torch.fft.fft2(target)
        
        # Convert to magnitude and phase
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        if self.log_scale:
            pred_mag = torch.log(pred_mag + 1e-8)
            target_mag = torch.log(target_mag + 1e-8)
            
        # Compute loss
        if self.loss_type == "l1":
            loss = F.l1_loss(pred_mag, target_mag)
        elif self.loss_type == "l2":
            loss = F.mse_loss(pred_mag, target_mag)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        # Weight high frequencies more if requested
        if self.high_freq_weight != 1.0:
            h, w = predicted.shape[-2:]
            center_h, center_w = h // 2, w // 2
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            distance = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
            weight = 1.0 + (self.high_freq_weight - 1.0) * (distance / distance.max())
            weight = weight.to(predicted.device)
            loss = loss * weight.mean()
            
        return loss


class WaveletLoss(nn.Module):
    """Loss in wavelet domain for multi-scale analysis."""
    
    def __init__(self, 
                 wavelet: str = "db4",
                 levels: int = 3,
                 loss_type: str = "l1"):
        super().__init__()
        
        if not PYWT_AVAILABLE:
            raise ImportError("PyWavelets is required for WaveletLoss")
            
        self.wavelet = wavelet
        self.levels = levels
        self.loss_type = loss_type
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss in wavelet domain."""
        batch_size = predicted.shape[0]
        total_loss = 0.0
        
        for i in range(batch_size):
            pred_img = predicted[i, 0].cpu().numpy()
            target_img = target[i, 0].cpu().numpy()
            
            # Wavelet decomposition
            pred_coeffs = pywt.wavedec2(pred_img, self.wavelet, level=self.levels)
            target_coeffs = pywt.wavedec2(target_img, self.wavelet, level=self.levels)
            
            # Compute loss for each level
            for pred_c, target_c in zip(pred_coeffs, target_coeffs):
                if isinstance(pred_c, tuple):  # Detail coefficients
                    for pred_detail, target_detail in zip(pred_c, target_c):
                        pred_tensor = torch.from_numpy(pred_detail).to(predicted.device)
                        target_tensor = torch.from_numpy(target_detail).to(predicted.device)
                        
                        if self.loss_type == "l1":
                            total_loss += F.l1_loss(pred_tensor, target_tensor)
                        elif self.loss_type == "l2":
                            total_loss += F.mse_loss(pred_tensor, target_tensor)
                else:  # Approximation coefficients
                    pred_tensor = torch.from_numpy(pred_c).to(predicted.device)
                    target_tensor = torch.from_numpy(target_c).to(predicted.device)
                    
                    if self.loss_type == "l1":
                        total_loss += F.l1_loss(pred_tensor, target_tensor)
                    elif self.loss_type == "l2":
                        total_loss += F.mse_loss(pred_tensor, target_tensor)
                        
        return total_loss / batch_size


class CycleConsistencyLoss(nn.Module):
    """Cycle consistency loss for paired/unpaired training."""
    
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, 
                original: torch.Tensor, 
                reconstructed: torch.Tensor) -> torch.Tensor:
        """Compute cycle consistency loss."""
        if self.loss_type == "l1":
            return F.l1_loss(reconstructed, original)
        elif self.loss_type == "l2":
            return F.mse_loss(reconstructed, original)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class GradientLoss(nn.Module):
    """Gradient-based loss for edge preservation."""
    
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient loss."""
        # Compute gradients
        pred_grad_x = predicted[:, :, :, 1:] - predicted[:, :, :, :-1]
        pred_grad_y = predicted[:, :, 1:, :] - predicted[:, :, :-1, :]
        
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # Compute loss
        if self.loss_type == "l1":
            loss_x = F.l1_loss(pred_grad_x, target_grad_x)
            loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        elif self.loss_type == "l2":
            loss_x = F.mse_loss(pred_grad_x, target_grad_x)
            loss_y = F.mse_loss(pred_grad_y, target_grad_y)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        return loss_x + loss_y


class SpectralLoss(nn.Module):
    """Spectral loss for preserving power spectral density characteristics."""
    
    def __init__(
        self,
        loss_type: str = "l1",
        log_power: bool = True,
        radial_average: bool = True,
        frequency_bands: Optional[List[Tuple[float, float]]] = None,
        band_weights: Optional[List[float]] = None,
    ):
        """Initialize spectral loss."""
        super().__init__()
        
        self.loss_type = loss_type.lower()
        self.log_power = log_power
        self.radial_average = radial_average
        self.frequency_bands = frequency_bands
        self.band_weights = band_weights or []
        
        # Loss function
        if self.loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif self.loss_type == "l2":
            self.loss_fn = F.mse_loss
        elif self.loss_type == "smooth_l1":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute spectral loss."""
        pred = predicted  # Support both interfaces
        
        # Compute power spectral density
        pred_psd = self._compute_power_spectral_density(pred)
        target_psd = self._compute_power_spectral_density(target)
        
        if self.radial_average:
            pred_radial = self._radial_average(pred_psd)
            target_radial = self._radial_average(target_psd)
            return self.loss_fn(pred_radial, target_radial, reduction='mean')
        elif self.frequency_bands:
            return self._compute_band_loss(pred_psd, target_psd)
        else:
            return self.loss_fn(pred_psd, target_psd, reduction='mean')
    
    def _compute_power_spectral_density(self, x: torch.Tensor) -> torch.Tensor:
        """Compute power spectral density."""
        x_fft = torch.fft.fft2(x)
        psd = torch.abs(x_fft) ** 2
        if self.log_power:
            psd = torch.log(psd + 1e-8)
        return psd
    
    def _radial_average(self, psd: torch.Tensor) -> torch.Tensor:
        """Compute radially averaged power spectrum."""
        batch_size, channels, h, w = psd.shape
        center_y, center_x = h // 2, w // 2
        y, x = torch.meshgrid(
            torch.arange(h, device=psd.device) - center_y,
            torch.arange(w, device=psd.device) - center_x,
            indexing='ij'
        )
        radius = torch.sqrt(x**2 + y**2)
        max_radius = min(center_y, center_x)
        radial_bins = torch.arange(0, max_radius, device=psd.device)
        
        radial_averages = []
        for b in range(batch_size):
            for c in range(channels):
                psd_2d = psd[b, c]
                radial_avg = []
                for r in radial_bins:
                    mask = (radius >= r) & (radius < r + 1)
                    if mask.sum() > 0:
                        avg_value = psd_2d[mask].mean()
                    else:
                        avg_value = torch.tensor(0.0, device=psd.device)
                    radial_avg.append(avg_value)
                radial_averages.append(torch.stack(radial_avg))
        
        return torch.stack(radial_averages).view(batch_size, channels, -1)
    
    def _compute_band_loss(self, pred_psd: torch.Tensor, target_psd: torch.Tensor) -> torch.Tensor:
        """Compute loss for specific frequency bands."""
        total_loss = 0.0
        h, w = pred_psd.shape[-2:]
        
        freq_y = torch.fft.fftfreq(h, device=pred_psd.device)
        freq_x = torch.fft.fftfreq(w, device=pred_psd.device)
        freq_y_grid, freq_x_grid = torch.meshgrid(freq_y, freq_x, indexing='ij')
        freq_magnitude = torch.sqrt(freq_y_grid**2 + freq_x_grid**2)
        freq_magnitude = freq_magnitude / (freq_magnitude.max() + 1e-8)
        
        for i, (low_freq, high_freq) in enumerate(self.frequency_bands):
            band_mask = (freq_magnitude >= low_freq) & (freq_magnitude < high_freq)
            if band_mask.sum() > 0:
                pred_band = pred_psd * band_mask.unsqueeze(0).unsqueeze(0)
                target_band = target_psd * band_mask.unsqueeze(0).unsqueeze(0)
                band_loss = self.loss_fn(pred_band, target_band, reduction='mean')
                band_weight = self.band_weights[i] if i < len(self.band_weights) else 1.0
                total_loss += band_weight * band_loss
        
        return total_loss / len(self.frequency_bands)


class MultiScaleFrequencyLoss(nn.Module):
    """Multi-scale frequency loss combining multiple frequency domain losses."""
    
    def __init__(
        self,
        scales: List[int] = [1, 2, 4],
        use_fourier: bool = True,
        use_spectral: bool = True,
        use_wavelet: bool = False,
        fourier_weight: float = 1.0,
        spectral_weight: float = 0.5,
        wavelet_weight: float = 0.3,
    ):
        """Initialize multi-scale frequency loss."""
        super().__init__()
        
        self.scales = scales
        self.use_fourier = use_fourier
        self.use_spectral = use_spectral
        self.use_wavelet = use_wavelet and PYWT_AVAILABLE
        self.fourier_weight = fourier_weight
        self.spectral_weight = spectral_weight
        self.wavelet_weight = wavelet_weight
        
        # Initialize loss functions
        if self.use_fourier:
            self.fourier_loss = FourierLoss()
        if self.use_spectral:
            self.spectral_loss = SpectralLoss()
        if self.use_wavelet:
            self.wavelet_loss = WaveletLoss()
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale frequency loss."""
        pred = predicted  # Support both interfaces
        total_loss = 0.0
        
        for scale in self.scales:
            if scale == 1:
                pred_scaled = pred
                target_scaled = target
            else:
                pred_scaled = F.avg_pool2d(pred, scale)
                target_scaled = F.avg_pool2d(target, scale)
            
            scale_loss = 0.0
            if self.use_fourier:
                fourier_loss = self.fourier_loss(pred_scaled, target_scaled)
                scale_loss += self.fourier_weight * fourier_loss
            if self.use_spectral:
                spectral_loss = self.spectral_loss(pred_scaled, target_scaled)
                scale_loss += self.spectral_weight * spectral_loss
            if self.use_wavelet:
                wavelet_loss = self.wavelet_loss(pred_scaled, target_scaled)
                scale_loss += self.wavelet_weight * wavelet_loss
            
            scale_weight = 1.0 / scale
            total_loss += scale_weight * scale_loss
        
        return total_loss / len(self.scales)


class HighFrequencyPreservationLoss(nn.Module):
    """Loss specifically designed to preserve high-frequency details."""
    
    def __init__(
        self,
        high_freq_threshold: float = 0.5,
        emphasis_factor: float = 2.0,
        use_gradient: bool = True,
        use_laplacian: bool = True,
        gradient_weight: float = 1.0,
        laplacian_weight: float = 0.5,
    ):
        """Initialize high-frequency preservation loss."""
        super().__init__()
        
        self.high_freq_threshold = high_freq_threshold
        self.emphasis_factor = emphasis_factor
        self.use_gradient = use_gradient
        self.use_laplacian = use_laplacian
        self.gradient_weight = gradient_weight
        self.laplacian_weight = laplacian_weight
        
        # Laplacian kernel
        self.register_buffer('laplacian_kernel', torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute high-frequency preservation loss."""
        pred = predicted  # Support both interfaces
        total_loss = 0.0
        
        # Frequency domain high-frequency emphasis
        freq_loss = self._compute_frequency_emphasis_loss(pred, target)
        total_loss += freq_loss
        
        if self.use_gradient:
            gradient_loss = self._compute_gradient_loss(pred, target)
            total_loss += self.gradient_weight * gradient_loss
        
        if self.use_laplacian:
            laplacian_loss = self._compute_laplacian_loss(pred, target)
            total_loss += self.laplacian_weight * laplacian_loss
        
        return total_loss
    
    def _compute_frequency_emphasis_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute frequency domain high-frequency emphasis loss."""
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        h, w = pred.shape[-2:]
        freq_y = torch.fft.fftfreq(h, device=pred.device)
        freq_x = torch.fft.fftfreq(w, device=pred.device)
        freq_y_grid, freq_x_grid = torch.meshgrid(freq_y, freq_x, indexing='ij')
        freq_magnitude = torch.sqrt(freq_y_grid**2 + freq_x_grid**2)
        
        freq_magnitude = freq_magnitude / (freq_magnitude.max() + 1e-8)
        high_freq_mask = (freq_magnitude > self.high_freq_threshold).float()
        high_freq_mask = high_freq_mask * self.emphasis_factor + (1 - high_freq_mask)
        
        pred_magnitude = torch.abs(pred_fft) * high_freq_mask
        target_magnitude = torch.abs(target_fft) * high_freq_mask
        
        return F.l1_loss(pred_magnitude, target_magnitude)
    
    def _compute_gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient-based high-frequency loss."""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
        
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        
        pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
        target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        return F.l1_loss(pred_grad_mag, target_grad_mag)
    
    def _compute_laplacian_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian-based high-frequency loss."""
        pred_laplacian = F.conv2d(pred, self.laplacian_kernel, padding=1)
        target_laplacian = F.conv2d(target, self.laplacian_kernel, padding=1)
        return F.l1_loss(pred_laplacian, target_laplacian)


class CompositeLoss(nn.Module):
    """Composite loss combining multiple loss functions."""
    
    def __init__(self, 
                 losses: Dict[str, nn.Module],
                 weights: Dict[str, float]):
        super().__init__()
        
        self.losses = nn.ModuleDict(losses)
        self.weights = weights
        
        # Validate that all losses have corresponding weights
        for loss_name in losses.keys():
            if loss_name not in weights:
                raise ValueError(f"No weight specified for loss: {loss_name}")
                
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute composite loss and return individual components."""
        loss_dict = {}
        total_loss = 0.0
        
        for loss_name, loss_fn in self.losses.items():
            loss_value = loss_fn(predicted, target)
            weighted_loss = self.weights[loss_name] * loss_value
            
            loss_dict[loss_name] = loss_value
            loss_dict[f"{loss_name}_weighted"] = weighted_loss
            total_loss += weighted_loss
            
        loss_dict["total"] = total_loss
        return loss_dict


# Factory functions for creating loss functions
def create_loss_function(loss_config: Dict[str, Any]) -> nn.Module:
    """Create a loss function from configuration."""
    loss_type = loss_config.get("type", "mse")
    
    if loss_type == "mse":
        return MSELoss(**loss_config.get("params", {}))
    elif loss_type == "l1":
        return L1Loss(**loss_config.get("params", {}))
    elif loss_type == "huber":
        return HuberLoss(**loss_config.get("params", {}))
    elif loss_type == "fourier":
        return FourierLoss(**loss_config.get("params", {}))
    elif loss_type == "wavelet":
        return WaveletLoss(**loss_config.get("params", {}))
    elif loss_type == "spectral":
        return SpectralLoss(**loss_config.get("params", {}))
    elif loss_type == "multi_scale":
        return MultiScaleFrequencyLoss(**loss_config.get("params", {}))
    elif loss_type == "high_frequency":
        return HighFrequencyPreservationLoss(**loss_config.get("params", {}))
    elif loss_type == "cycle_consistency":
        return CycleConsistencyLoss(**loss_config.get("params", {}))
    elif loss_type == "gradient":
        return GradientLoss(**loss_config.get("params", {}))
    elif loss_type == "composite":
        # For composite loss, recursively create component losses
        component_losses = {}
        weights = loss_config.get("weights", {})
        
        for component_name, component_config in loss_config.get("components", {}).items():
            component_losses[component_name] = create_loss_function(component_config)
            
        return CompositeLoss(component_losses, weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def create_frequency_loss(
    loss_type: str = "fourier",
    **kwargs
) -> nn.Module:
    """Create frequency domain loss function.
    
    Args:
        loss_type: Type of frequency loss
        **kwargs: Additional arguments for specific loss types
        
    Returns:
        Configured frequency loss function
    """
    if loss_type == "fourier":
        return FourierLoss(**kwargs)
    elif loss_type == "wavelet":
        return WaveletLoss(**kwargs)
    elif loss_type == "spectral":
        return SpectralLoss(**kwargs)
    elif loss_type == "multi_scale":
        return MultiScaleFrequencyLoss(**kwargs)
    elif loss_type == "high_frequency":
        return HighFrequencyPreservationLoss(**kwargs)
    else:
        raise ValueError(f"Unknown frequency loss type: {loss_type}")


def get_frequency_loss_config() -> Dict[str, Any]:
    """Get default configuration for frequency losses."""
    return {
        "fourier": {
            "loss_type": "l1",
            "log_magnitude": True,
            "phase_weight": 0.1,
            "magnitude_weight": 1.0,
            "high_freq_emphasis": 1.5,
            "low_freq_emphasis": 1.0,
        },
        "spectral": {
            "loss_type": "l1",
            "log_power": True,
            "radial_average": True,
        },
        "wavelet": {
            "wavelet": "db4",
            "levels": 3,
            "loss_type": "l1",
            "approximation_weight": 1.0,
        },
        "multi_scale": {
            "scales": [1, 2, 4],
            "fourier_weight": 1.0,
            "spectral_weight": 0.5,
            "wavelet_weight": 0.3,
        },
        "high_frequency": {
            "high_freq_threshold": 0.5,
            "emphasis_factor": 2.0,
            "gradient_weight": 1.0,
            "laplacian_weight": 0.5,
        }
    }


__all__ = [
    # Basic losses
    "MSELoss",
    "L1Loss", 
    "HuberLoss",
    
    
    # Frequency domain losses
    "FourierLoss",
    "WaveletLoss",
    "SpectralLoss",
    "MultiScaleFrequencyLoss",
    "HighFrequencyPreservationLoss",
    
    # Other losses
    "CycleConsistencyLoss",
    "GradientLoss",
    "CompositeLoss",
    
    # Factory functions
    "create_loss_function",
    "create_frequency_loss",
    "get_frequency_loss_config",
]
