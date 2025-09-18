"""
Guidance strategies for physics-guided diffusion models.

This module provides various guidance strategies for incorporating physical
constraints and measurements into the diffusion sampling process.
"""

import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pkl_dg.physics import ForwardModel

try:
    from einops import rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False


class GuidanceStrategy(ABC):
    """Abstract base class for guidance strategies.

    Implementations should provide a gradient in the intensity domain that has
    the same shape as the current estimate x0_hat.
    """

    @abstractmethod
    def compute_gradient(
        self,
        x0_hat: torch.Tensor,
        y: torch.Tensor,
        forward_model: "ForwardModel",
        t: int,
    ) -> torch.Tensor:
        """Compute guidance gradient in intensity domain.

        Args:
            x0_hat: Current estimate of the clean image at time step t
            y: Observed measurement in the sensor domain
            forward_model: Forward model that provides PSF ops and background
            t: Current diffusion time step

        Returns:
            Tensor with the same shape as x0_hat representing the gradient.
        """
        raise NotImplementedError

    def apply_guidance(
        self,
        x0_hat: torch.Tensor,
        gradient: torch.Tensor,
        lambda_t: float,
    ) -> torch.Tensor:
        """Apply guidance step to the current estimate.

        x_{t+1} = x_t - lambda_t * grad
        """
        return x0_hat - lambda_t * gradient


class PKLGuidance(GuidanceStrategy):
    """Poisson-Kullback-Leibler guidance strategy.
    
    This is the main guidance strategy that incorporates Poisson noise
    characteristics for microscopy image restoration.
    """
    
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
        Ax = forward_model.apply_psf(x0_hat)
        Ax_plus_B = Ax + forward_model.background

        
        sigma2 = 0.0
        try:
            sigma = float(getattr(forward_model, "read_noise_sigma", 0.0))
            sigma2 = sigma * sigma
        except Exception:
            sigma2 = 0.0

        
        denom = Ax_plus_B + sigma2 + self.epsilon
        ratio = y / denom
        residual = 1.0 - ratio

        
        gradient = forward_model.apply_psf_adjoint(residual)
        return gradient


class L2Guidance(GuidanceStrategy):
    """L2 (least squares) guidance strategy.
    
    Standard L2 guidance for comparison with PKL guidance.
    """
    
    def compute_gradient(
        self,
        x0_hat: torch.Tensor,
        y: torch.Tensor,
        forward_model: "ForwardModel",
        t: int,
    ) -> torch.Tensor:
        Ax = forward_model.apply_psf(x0_hat)
        Ax_plus_B = Ax + forward_model.background
        residual = y - Ax_plus_B
        gradient = forward_model.apply_psf_adjoint(residual)
        return gradient


class AnscombeGuidance(GuidanceStrategy):
    """Anscombe transform-based guidance strategy.
    
    Uses Anscombe transform to stabilize Poisson noise variance
    before applying L2-like guidance.
    """
    
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def anscombe_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Anscombe transform to stabilize Poisson noise variance."""
        return 2.0 * torch.sqrt(x + 3.0 / 8.0)

    def anscombe_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute derivative of Anscombe transform."""
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
        residual_a = Ax_a - y_a
        chain = residual_a * self.anscombe_derivative(Ax_plus_B)
        gradient = forward_model.apply_psf_adjoint(chain)
        return gradient


class AdaptiveSchedule:
    """Adaptive scheduling for guidance strength lambda_t.
    
    Automatically adjusts guidance strength based on gradient magnitude
    and diffusion timestep.
    """
    
    def __init__(
        self,
        lambda_base: float = 0.1,
        T_threshold: int = 800,
        epsilon_lambda: float = 1e-3,
        T_total: int = 1000,
    ):
        self.lambda_base = lambda_base
        self.T_threshold = T_threshold
        self.epsilon_lambda = epsilon_lambda
        self.T_total = T_total

    def get_lambda_t(self, gradient: torch.Tensor, t: int) -> float:
        """Compute adaptive lambda_t based on gradient and timestep."""
        if EINOPS_AVAILABLE:
            grad_flat = rearrange(gradient, '... -> (...)')
        else:
            grad_flat = gradient.flatten()
        
        
        grad_norm = torch.linalg.vector_norm(grad_flat) + self.epsilon_lambda
        step_size = self.lambda_base / grad_norm
        warmup = min((self.T_total - t) / (self.T_total - self.T_threshold), 1.0)
        lambda_t = step_size * warmup
        return lambda_t.item() if isinstance(lambda_t, torch.Tensor) else lambda_t


def create_pkl_guidance(epsilon: float = 1e-6) -> PKLGuidance:
    """Create PKL guidance strategy."""
    return PKLGuidance(epsilon=epsilon)


def create_l2_guidance() -> L2Guidance:
    """Create L2 guidance strategy."""
    return L2Guidance()


def create_anscombe_guidance(epsilon: float = 1e-6) -> AnscombeGuidance:
    """Create Anscombe guidance strategy."""
    return AnscombeGuidance(epsilon=epsilon)


def create_adaptive_schedule(
    lambda_base: float = 0.1,
    T_threshold: int = 800,
    epsilon_lambda: float = 1e-3,
    T_total: int = 1000,
) -> AdaptiveSchedule:
    """Create adaptive schedule for guidance strength."""
    return AdaptiveSchedule(
        lambda_base=lambda_base,
        T_threshold=T_threshold,
        epsilon_lambda=epsilon_lambda,
        T_total=T_total,
    )


__all__ = [
    "GuidanceStrategy",
    "PKLGuidance",
    "L2Guidance",
    "AnscombeGuidance",
    "AdaptiveSchedule",
    "create_pkl_guidance",
    "create_l2_guidance",
    "create_anscombe_guidance",
    "create_adaptive_schedule",
]
