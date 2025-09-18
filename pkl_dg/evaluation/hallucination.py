from typing import Tuple

import numpy as np
import torch

# Import Metrics directly to avoid circular imports
from pkl_dg.evaluation.metrics import Metrics


class HallucinationTests:
    """Adversarial hallucination protocols: commission and omission tests."""

    @staticmethod
    def add_out_of_focus_artifact(
        image: np.ndarray,
        center: Tuple[int, int],
        radius: int = 8,
        intensity: float = 2.0,
        sigma: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add a blurred bright disk artifact and return modified image and mask."""
        h, w = image.shape
        yy, xx = np.ogrid[:h, :w]
        mask = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2
        artifact = np.zeros_like(image, dtype=np.float32)
        artifact[mask] = intensity
        # Gaussian blur via FFT-based convolution for speed
        from scipy.ndimage import gaussian_filter  # type: ignore

        artifact = gaussian_filter(artifact, sigma=sigma).astype(np.float32)
        out = image.astype(np.float32) + artifact
        return out, (artifact > 1e-6)

    @staticmethod
    def commission_sar(
        reconstructed: np.ndarray,
        artifact_mask: np.ndarray,
    ) -> float:
        """Compute SAR in dB: higher is better (less hallucinated artifact)."""
        return Metrics.sar(reconstructed.astype(np.float32), artifact_mask.astype(bool))

    @staticmethod
    def insert_faint_structure(
        clean: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        width: int = 1,
        amplitude: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Insert a faint line segment into a clean image, returning new image and mask."""
        img = clean.astype(np.float32).copy()
        mask = np.zeros_like(img, dtype=bool)
        # Bresenham-like rasterization for a thin line
        x0, y0 = start[1], start[0]
        x1, y1 = end[1], end[0]
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for wx in range(-width, width + 1):
                for wy in range(-width, width + 1):
                    yy = min(max(y0 + wy, 0), img.shape[0] - 1)
                    xx = min(max(x0 + wx, 0), img.shape[1] - 1)
                    img[yy, xx] += amplitude
                    mask[yy, xx] = True
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return img, mask

    @staticmethod
    def structure_fidelity_psnr(
        reconstructed: np.ndarray,
        target_with_structure: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Compute PSNR restricted to a mask region to assess faint structure fidelity."""
        m = mask.astype(bool)
        if not np.any(m):
            return 100.0
        pred = reconstructed[m].astype(np.float32)
        tgt = target_with_structure[m].astype(np.float32)
        data_range = float(tgt.max() - tgt.min()) if np.any(tgt) else 1.0
        mse = float(np.mean((pred - tgt) ** 2))
        if mse <= 1e-12:
            return 100.0
        return float(10.0 * np.log10((data_range ** 2) / mse))


