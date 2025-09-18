"""
Robustness and Hallucination Metrics

Metrics for evaluating model robustness and detecting hallucinations.
"""

import numpy as np
import torch
from typing import Optional, Any
from .registry import register_metric

# Import the existing robustness and hallucination tests directly
from ..robustness import RobustnessTests
from ..hallucination import HallucinationTests


@register_metric(
    name="alignment_error_robustness",
    category="robustness",
    description="Robustness to alignment errors",
    requires_reference=True,
    requires_input=False
)
def alignment_error_robustness_metric(pred: np.ndarray, target: np.ndarray, **kwargs) -> float:
    """Test robustness to alignment errors."""
    try:
        return RobustnessTests.alignment_error_robustness(pred, target)
    except Exception:
        return 0.0


@register_metric(
    name="psf_mismatch_robustness", 
    category="robustness",
    description="Robustness to PSF mismatch",
    requires_reference=True,
    requires_input=False
)
def psf_mismatch_robustness_metric(pred: np.ndarray, target: np.ndarray, **kwargs) -> float:
    """Test robustness to PSF mismatch."""
    try:
        psf = kwargs.get('psf', None)
        if psf is None:
            return 0.0
        return RobustnessTests.psf_mismatch_robustness(pred, target, psf)
    except Exception:
        return 0.0


@register_metric(
    name="noise_robustness",
    category="robustness", 
    description="Robustness to different noise levels",
    requires_reference=True,
    requires_input=False
)
def noise_robustness_metric(pred: np.ndarray, target: np.ndarray, **kwargs) -> float:
    """Test robustness to noise variations."""
    try:
        # Add different levels of noise and measure performance degradation
        noise_levels = [0.01, 0.02, 0.05, 0.1]
        robustness_scores = []
        
        # Original PSNR
        original_psnr = 20 * np.log10(target.max() / np.sqrt(np.mean((pred - target) ** 2)))
        
        for noise_level in noise_levels:
            # Add noise to prediction
            noise = np.random.normal(0, noise_level * target.max(), pred.shape)
            noisy_pred = pred + noise
            
            # Compute PSNR with noise
            noisy_psnr = 20 * np.log10(target.max() / np.sqrt(np.mean((noisy_pred - target) ** 2)))
            
            # Compute robustness as ratio
            if original_psnr > 0:
                robustness = noisy_psnr / original_psnr
            else:
                robustness = 0.0
            
            robustness_scores.append(robustness)
        
        return float(np.mean(robustness_scores))
    except Exception:
        return 0.0


@register_metric(
    name="commission_sar",
    category="robustness",
    description="Commission error SAR (hallucination detection)",
    requires_reference=False,
    requires_input=False
)
def commission_sar_metric(pred: np.ndarray, artifact_mask: Optional[np.ndarray] = None, **kwargs) -> float:
    """Compute commission error SAR for hallucination detection."""
    if artifact_mask is None:
        return 0.0
    
    try:
        return HallucinationTests.commission_sar(pred, artifact_mask)
    except Exception:
        return 0.0


@register_metric(
    name="hallucination_score",
    category="robustness",
    description="General hallucination detection score", 
    requires_reference=True,
    requires_input=False
)
def hallucination_score_metric(pred: np.ndarray, target: np.ndarray, **kwargs) -> float:
    """Compute general hallucination score."""
    try:
        # Simple hallucination detection based on high-frequency content
        # that's not present in the target
        
        # Apply high-pass filter to both images
        from scipy.ndimage import gaussian_filter
        
        sigma = kwargs.get('sigma', 1.0)
        
        pred_smooth = gaussian_filter(pred, sigma=sigma)
        target_smooth = gaussian_filter(target, sigma=sigma)
        
        pred_highfreq = pred - pred_smooth
        target_highfreq = target - target_smooth
        
        # Measure excess high-frequency content in prediction
        pred_hf_energy = np.sum(pred_highfreq ** 2)
        target_hf_energy = np.sum(target_highfreq ** 2)
        
        if target_hf_energy == 0:
            return 1.0 if pred_hf_energy > 0 else 0.0
        
        hallucination_ratio = pred_hf_energy / target_hf_energy
        
        # Convert to a score between 0 and 1 (lower is better)
        score = np.tanh(hallucination_ratio - 1.0)
        return float(max(0.0, score))
        
    except Exception:
        return 0.0


def test_alignment_error_with_sampler(sampler: Any, input_tensor: torch.Tensor, **kwargs) -> float:
    """
    Test alignment error robustness with a sampler object.
    
    This maintains compatibility with the existing evaluation script.
    """
    try:
        shift_pixels = kwargs.get('shift_pixels', 0.5)
        result = RobustnessTests.alignment_error_test(sampler, input_tensor, shift_pixels=shift_pixels)
        return result.squeeze().detach().cpu().numpy().astype(np.float32)
    except Exception:
        return np.zeros_like(input_tensor.squeeze().detach().cpu().numpy(), dtype=np.float32)


def add_out_of_focus_artifact(image: np.ndarray, center: tuple, **kwargs) -> tuple:
    """
    Add out-of-focus artifact for hallucination testing.
    
    This maintains compatibility with the existing evaluation script.
    """
    try:
        return HallucinationTests.add_out_of_focus_artifact(image, center=center)
    except Exception:
        return image, np.zeros_like(image, dtype=bool)
