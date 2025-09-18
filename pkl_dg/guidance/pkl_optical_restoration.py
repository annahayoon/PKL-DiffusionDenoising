import torch
from .base import GuidanceStrategy


class PKLGuidance(GuidanceStrategy):
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def compute_gradient(
        self,
        x0_hat: torch.Tensor,
        y: torch.Tensor,
        forward_model: "ForwardModel",
        t: int,
    ) -> torch.Tensor:
        """Batch-aware PKL gradient computation.

        Supports both single image [1,1,H,W] and batched tensors [B,1,H,W].
        All ops are vectorized; no per-sample loops.
        """
        # Ensure shapes are broadcastable and on the same device
        # forward_model.apply_psf is already batch-aware
        Ax = forward_model.apply_psf(x0_hat)
        Ax_plus_B = Ax + forward_model.background

        # Read-noise-aware denominator: A(x)+B + sigma^2
        sigma2 = 0.0
        try:
            sigma = float(getattr(forward_model, "read_noise_sigma", 0.0))
            sigma2 = sigma * sigma
        except Exception:
            sigma2 = 0.0

        # Avoid division by zero via epsilon; supports batch tensors natively
        denom = Ax_plus_B + sigma2 + self.epsilon
        ratio = y / denom
        residual = 1.0 - ratio

        # Adjoint is also batch-aware
        gradient = forward_model.apply_psf_adjoint(residual)
        return gradient

