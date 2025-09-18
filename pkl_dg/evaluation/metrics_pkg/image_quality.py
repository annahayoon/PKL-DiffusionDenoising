"""
Image Quality Metrics

Standard image quality metrics like PSNR, SSIM, FRC, etc.
"""

import numpy as np
from typing import Optional
from .registry import register_metric

# Direct implementation to avoid circular imports
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


@register_metric(
    name="psnr",
    category="image_quality", 
    description="Peak Signal-to-Noise Ratio",
    requires_reference=True,
    requires_input=False
)
def psnr_metric(pred: np.ndarray, target: np.ndarray, **kwargs) -> float:
    """Compute PSNR between prediction and target."""
    data_range = kwargs.get('data_range', None)
    if data_range is None:
        data_range = float(target.max() - target.min()) if target.size > 0 else 1.0
    
    # Direct implementation to avoid circular import
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    if data_range == 0.0:
        data_range = 1.0
    err = float(np.mean((pred - target) ** 2))
    if err <= 1e-12:
        return 100.0
    return float(10.0 * np.log10((data_range ** 2) / err))


@register_metric(
    name="ssim", 
    category="image_quality",
    description="Structural Similarity Index Measure",
    requires_reference=True,
    requires_input=False
)
def ssim_metric(pred: np.ndarray, target: np.ndarray, **kwargs) -> float:
    """Compute SSIM between prediction and target."""
    data_range = kwargs.get('data_range', None)
    if data_range is None:
        data_range = float(target.max() - target.min()) if target.size > 0 else 1.0
    
    return structural_similarity(target, pred, data_range=data_range)


@register_metric(
    name="frc",
    category="image_quality", 
    description="Fourier Ring Correlation",
    requires_reference=True,
    requires_input=False
)
def frc_metric(pred: np.ndarray, target: np.ndarray, **kwargs) -> float:
    """Compute FRC between prediction and target."""
    threshold = kwargs.get('threshold', 0.143)
    
    # Direct FRC implementation to avoid circular import
    # FFTs
    fft_pred = np.fft.fft2(pred)
    fft_target = np.fft.fft2(target)

    # Cross-correlation numerator and power terms
    correlation = np.real(fft_pred * np.conj(fft_target))
    power_pred = np.abs(fft_pred) ** 2
    power_target = np.abs(fft_target) ** 2

    # Radial bins
    h, w = pred.shape
    y, x = np.ogrid[:h, :w]
    center = (h // 2, w // 2)
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(int)

    # Compute FRC curve via radial averaging
    max_r = min(center)
    frc_curve = []
    for radius in range(1, max_r):
        mask = r == radius
        if mask.sum() > 0:
            corr = correlation[mask].mean()
            power = np.sqrt(power_pred[mask].mean() * power_target[mask].mean())
            frc_val = corr / (power + 1e-10)
            frc_curve.append(frc_val)
        else:
            frc_curve.append(0.0)

    # Find first crossing below threshold
    frc_curve = np.array(frc_curve)
    below_threshold = np.where(frc_curve < threshold)[0]
    if len(below_threshold) > 0:
        return float(below_threshold[0] + 1)  # +1 because we started from radius 1
    else:
        return float(len(frc_curve))


@register_metric(
    name="mse",
    category="image_quality",
    description="Mean Squared Error", 
    requires_reference=True,
    requires_input=False
)
def mse_metric(pred: np.ndarray, target: np.ndarray, **kwargs) -> float:
    """Compute MSE between prediction and target."""
    return float(np.mean((pred - target) ** 2))


@register_metric(
    name="mae",
    category="image_quality",
    description="Mean Absolute Error",
    requires_reference=True, 
    requires_input=False
)
def mae_metric(pred: np.ndarray, target: np.ndarray, **kwargs) -> float:
    """Compute MAE between prediction and target."""
    return float(np.mean(np.abs(pred - target)))


@register_metric(
    name="snr",
    category="image_quality",
    description="Signal-to-Noise Ratio",
    requires_reference=True,
    requires_input=False  
)
def snr_metric(pred: np.ndarray, target: np.ndarray, **kwargs) -> float:
    """Compute SNR between prediction and target."""
    signal_power = np.mean(target ** 2)
    noise_power = np.mean((pred - target) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    return float(snr_db)
