import torch
from typing import TYPE_CHECKING
from .base import GuidanceStrategy

if TYPE_CHECKING:
    from pkl_dg.physics.forward_model import ForwardModel


class AnscombeGuidance(GuidanceStrategy):
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def anscombe_transform(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sqrt(x + 3.0 / 8.0)

    def anscombe_derivative(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (torch.sqrt(x + 3.0 / 8.0) + self.epsilon)

    def compute_gradient(
        self,
        x0_hat: torch.Tensor,
        y: torch.Tensor,
        forward_model: "ForwardModel",
        t: int,
    ) -> torch.Tensor:
        Ax = forward_model.apply_psf(x0_hat)
        Ax_plus_B = Ax + forward_model.background
        y_a = self.anscombe_transform(y)
        Ax_a = self.anscombe_transform(Ax_plus_B)
        # Residual in transformed space (A(x) side derivative applied)
        residual_a = Ax_a - y_a
        chain = residual_a * self.anscombe_derivative(Ax_plus_B)
        gradient = forward_model.apply_psf_adjoint(chain)
        return gradient

