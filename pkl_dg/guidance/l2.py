import torch
from .base import GuidanceStrategy


class L2Guidance(GuidanceStrategy):
    def compute_gradient(
        self,
        x0_hat: torch.Tensor,
        y: torch.Tensor,
        forward_model: "ForwardModel",
        t: int,
    ) -> torch.Tensor:
        # Gradient of 1/2 * ||y - (A(x)+B)||^2 wrt x is A^T(A(x)+B - y)
        # But for guidance, we want the negative gradient: A^T(y - (A(x)+B))
        Ax = forward_model.apply_psf(x0_hat)
        Ax_plus_B = Ax + forward_model.background
        residual = y - Ax_plus_B  # Fixed sign to match paper equation (80)
        gradient = forward_model.apply_psf_adjoint(residual)
        return gradient

