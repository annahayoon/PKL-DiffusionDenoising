# Phase 7: Evaluation Suite (Week 5)

### Step 7.1: Implement Metrics
```python
# pkl_dg/evaluation/metrics.py
import torch
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.spatial.distance import directed_hausdorff
from typing import Optional, Tuple

class Metrics:
    """Image quality metrics."""
    
    @staticmethod
    def psnr(pred: np.ndarray, target: np.ndarray, data_range: Optional[float] = None) -> float:
        """Compute PSNR."""
        if data_range is None:
            data_range = target.max() - target.min()
        return peak_signal_noise_ratio(target, pred, data_range=data_range)
    
    @staticmethod
    def ssim(pred: np.ndarray, target: np.ndarray, data_range: Optional[float] = None) -> float:
        """Compute SSIM."""
        if data_range is None:
            data_range = target.max() - target.min()
        return structural_similarity(target, pred, data_range=data_range)
    
    @staticmethod
    def frc(pred: np.ndarray, target: np.ndarray, threshold: float = 0.143) -> float:
        """
        Compute Fourier Ring Correlation.
        
        Args:
            pred: Predicted image
            target: Target image
            threshold: Resolution threshold (1/7 for standard)
        
        Returns:
            Resolution in pixels
        """
        # Compute FFT
        fft_pred = np.fft.fft2(pred)
        fft_target = np.fft.fft2(target)
        
        # Compute correlation
        correlation = np.real(fft_pred * np.conj(fft_target))
        power_pred = np.abs(fft_pred) ** 2
        power_target = np.abs(fft_target) ** 2
        
        # Radial averaging
        h, w = pred.shape
        y, x = np.ogrid[:h, :w]
        center = (h//2, w//2)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        
        # Compute FRC curve
        max_r = min(center)
        frc_curve = []
        
        for radius in range(1, max_r):
            mask = r == radius
            if mask.sum() > 0:
                corr = correlation[mask].mean()
                power = np.sqrt(power_pred[mask].mean() * power_target[mask].mean())
                frc = corr / (power + 1e-10)
                frc_curve.append(frc)
        
        # Find resolution at threshold
        frc_curve = np.array(frc_curve)
        indices = np.where(frc_curve < threshold)[0]
        
        if len(indices) > 0:
            resolution = indices[0]
        else:
            resolution = len(frc_curve)
        
        # Convert to physical units if needed
        return resolution
    
    @staticmethod
    def sar(pred: np.ndarray, artifact_mask: np.ndarray) -> float:
        """
        Compute Signal-to-Artifact Ratio.
        
        Args:
            pred: Predicted image
            artifact_mask: Binary mask of artifact region
        
        Returns:
            SAR in dB
        """
        signal_region = ~artifact_mask
        
        signal_power = np.mean(pred[signal_region] ** 2)
        artifact_power = np.mean(pred[artifact_mask] ** 2)
        
        sar = 10 * np.log10(signal_power / (artifact_power + 1e-10))
        
        return sar
    
    @staticmethod
    def hausdorff_distance(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
        """
        Compute Hausdorff distance between masks.
        
        Args:
            pred_mask: Predicted segmentation mask
            target_mask: Target segmentation mask
        
        Returns:
            Hausdorff distance
        """
        # Get contour points
        pred_points = np.argwhere(pred_mask)
        target_points = np.argwhere(target_mask)
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')
        
        # Compute directed Hausdorff distances
        d_forward = directed_hausdorff(pred_points, target_points)[0]
        d_backward = directed_hausdorff(target_points, pred_points)[0]
        
        # Return maximum (symmetric Hausdorff)
        return max(d_forward, d_backward)
```

### Step 7.2: Implement Robustness Tests
```python
# pkl_dg/evaluation/robustness.py
import torch
import numpy as np
from typing import Dict, Any

class RobustnessTests:
    """Robustness evaluation tests."""
    
    @staticmethod
    def psf_mismatch_test(
        sampler: 'DDIMSampler',
        y: torch.Tensor,
        psf_true: 'PSF',
        mismatch_factor: float = 1.1
    ) -> torch.Tensor:
        """
        Test robustness to PSF mismatch.
        
        Args:
            sampler: DDIM sampler
            y: Measurement with true PSF
            psf_true: True PSF
            mismatch_factor: PSF broadening factor
        
        Returns:
            Reconstruction with mismatched PSF
        """
        # Create mismatched PSF
        psf_mismatched = psf_true.broaden(mismatch_factor)
        
        # Update forward model with mismatched PSF
        original_psf = sampler.forward_model.psf
        sampler.forward_model.psf = psf_mismatched.to_torch(
            device=sampler.forward_model.device
        )
        
        # Run reconstruction
        shape = (1, 1, y.shape[-2], y.shape[-1])
        reconstruction = sampler.sample(y, shape, verbose=False)
        
        # Restore original PSF
        sampler.forward_model.psf = original_psf
        
        return reconstruction
    
    @staticmethod
    def alignment_error_test(
        sampler: 'DDIMSampler',
        y: torch.Tensor,
        shift_pixels: float = 0.5
    ) -> torch.Tensor:
        """
        Test robustness to alignment errors.
        
        Args:
            sampler: DDIM sampler
            y: Original measurement
            shift_pixels: Subpixel shift amount
        
        Returns:
            Reconstruction with shifted input
        """
        import kornia
        
        # Create random shift
        theta = torch.tensor([
            [1, 0, shift_pixels / y.shape[-1]],
            [0, 1, shift_pixels / y.shape[-2]]
        ], dtype=torch.float32).unsqueeze(0)
        
        # Apply shift
        grid = kornia.utils.create_meshgrid(
            y.shape[-2], y.shape[-1], normalized_coordinates=True
        )
        y_shifted = kornia.geometry.transform.warp_affine(
            y.unsqueeze(0).unsqueeze(0),
            theta,
            dsize=(y.shape[-2], y.shape[-1])
        ).squeeze()
        
        # Run reconstruction
        shape = (1, 1, y.shape[-2], y.shape[-1])
        reconstruction = sampler.sample(y_shifted, shape, verbose=False)
        
        return reconstruction
```
