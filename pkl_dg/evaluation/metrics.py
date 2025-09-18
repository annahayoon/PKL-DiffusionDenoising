import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.spatial.distance import directed_hausdorff
from typing import Optional
from tqdm import tqdm


class Metrics:
    """Image quality metrics."""

    @staticmethod
    def psnr(pred: np.ndarray, target: np.ndarray, data_range: Optional[float] = None) -> float:
        """Compute PSNR with stable handling for identical images and zero data range."""
        pred = pred.astype(np.float32)
        target = target.astype(np.float32)
        if data_range is None:
            data_range = float(target.max() - target.min())
            if data_range == 0.0:
                data_range = 1.0
        err = float(np.mean((pred - target) ** 2))
        if err <= 1e-12:
            return 100.0
        return float(10.0 * np.log10((data_range ** 2) / err))

    @staticmethod
    def ssim(pred: np.ndarray, target: np.ndarray, data_range: Optional[float] = None) -> float:
        """Compute SSIM.

        Args:
            pred: Predicted image as numpy array
            target: Target image as numpy array
            data_range: Dynamic range of the target image values. If None, uses target.max() - target.min().

        Returns:
            Structural Similarity Index value.
        """
        if data_range is None:
            data_range = target.max() - target.min()
        return structural_similarity(target, pred, data_range=data_range)

    @staticmethod
    def frc(pred: np.ndarray, target: np.ndarray, threshold: float = 0.143) -> float:
        """
        Compute Fourier Ring Correlation resolution threshold index.

        Args:
            pred: Predicted image (2D numpy array)
            target: Target image (2D numpy array)
            threshold: Resolution threshold (e.g., 0.143 or 1/7)

        Returns:
            Resolution radius in pixels where FRC first falls below threshold.
        """
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
        for radius in tqdm(range(1, max_r), desc="Computing FRC", leave=False):
            mask = r == radius
            if mask.sum() > 0:
                corr = correlation[mask].mean()
                power = np.sqrt(power_pred[mask].mean() * power_target[mask].mean())
                frc_val = corr / (power + 1e-10)
                frc_curve.append(frc_val)

        frc_curve = np.array(frc_curve) if len(frc_curve) > 0 else np.array([0.0])
        indices = np.where(frc_curve < threshold)[0]
        if len(indices) > 0:
            resolution = float(indices[0])
        else:
            resolution = float(len(frc_curve))
        return resolution

    @staticmethod
    def sar(pred: np.ndarray, artifact_mask: np.ndarray) -> float:
        """
        Compute Signal-to-Artifact Ratio (SAR) in dB.

        Args:
            pred: Predicted image (2D numpy array)
            artifact_mask: Boolean mask where artifacts are True

        Returns:
            SAR value in dB.
        """
        signal_region = ~artifact_mask
        signal_power = float(np.mean(pred[signal_region] ** 2))
        artifact_power = float(np.mean(pred[artifact_mask] ** 2))
        sar = 10.0 * np.log10(signal_power / (artifact_power + 1e-10))
        return float(sar)

    @staticmethod
    def hausdorff_distance(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
        """
        Compute symmetric Hausdorff distance between binary masks.

        Args:
            pred_mask: Predicted segmentation mask (boolean array)
            target_mask: Target segmentation mask (boolean array)

        Returns:
            Hausdorff distance as a float. Returns inf if any mask has no positive pixels.
        """
        pred_points = np.argwhere(pred_mask)
        target_points = np.argwhere(target_mask)
        if len(pred_points) == 0 or len(target_points) == 0:
            return float("inf")
        d_forward = directed_hausdorff(pred_points, target_points)[0]
        d_backward = directed_hausdorff(target_points, pred_points)[0]
        return float(max(d_forward, d_backward))


