from abc import ABC, abstractmethod
from typing import Optional

import torch


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


