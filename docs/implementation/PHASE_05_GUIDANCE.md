# Phase 5: Guidance Mechanisms (Week 4)

### Step 5.1: Implement Base Guidance Class
```python
# pkl_dg/guidance/base.py
from abc import ABC, abstractmethod
import torch
from typing import Optional

class GuidanceStrategy(ABC):
    """Abstract base class for guidance strategies."""
    
    @abstractmethod
    def compute_gradient(
        self,
        x0_hat: torch.Tensor,
        y: torch.Tensor,
        forward_model: 'ForwardModel',
        t: int
    ) -> torch.Tensor:
        """
        Compute guidance gradient in intensity domain.
        Returns tensor with same shape as x0_hat.
        """
        pass
    
    def apply_guidance(
        self,
        x0_hat: torch.Tensor,
        gradient: torch.Tensor,
        lambda_t: float
    ) -> torch.Tensor:
        return x0_hat - lambda_t * gradient
```

### Step 5.2: Implement PKL Guidance
```python
# pkl_dg/guidance/pkl.py
import torch
from .base import GuidanceStrategy

class PKLGuidance(GuidanceStrategy):
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
    
    def compute_gradient(
        self,
        x0_hat: torch.Tensor,
        y: torch.Tensor,
        forward_model: 'ForwardModel',
        t: int
    ) -> torch.Tensor:
        Ax = forward_model.apply_psf(x0_hat)
        Ax_plus_B = Ax + forward_model.background
        ratio = y / (Ax_plus_B + self.epsilon)
        residual = 1.0 - ratio
        gradient = forward_model.apply_psf_adjoint(residual)
        return gradient
```

### Step 5.3: Implement L2 and Anscombe Guidance
```python
# pkl_dg/guidance/l2.py
import torch
from .base import GuidanceStrategy

class L2Guidance(GuidanceStrategy):
    def compute_gradient(
        self,
        x0_hat: torch.Tensor,
        y: torch.Tensor,
        forward_model: 'ForwardModel',
        t: int
    ) -> torch.Tensor:
        # Gradient of 1/2 * ||y - (A(x)+B)||^2 wrt x is A^T(A(x)+B - y)
        Ax = forward_model.apply_psf(x0_hat)
        Ax_plus_B = Ax + forward_model.background
        residual = Ax_plus_B - y
        gradient = forward_model.apply_psf_adjoint(residual)
        return gradient

# pkl_dg/guidance/anscombe.py
import torch
from .base import GuidanceStrategy

class AnscombeGuidance(GuidanceStrategy):
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
    
    def anscombe_transform(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sqrt(x + 3.0/8.0)
    
    def anscombe_derivative(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (torch.sqrt(x + 3.0/8.0) + self.epsilon)
    
    def compute_gradient(
        self,
        x0_hat: torch.Tensor,
        y: torch.Tensor,
        forward_model: 'ForwardModel',
        t: int
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
```

### Step 5.4: Implement Adaptive Schedule
```python
# pkl_dg/guidance/schedules.py
import torch

class AdaptiveSchedule:
    def __init__(
        self,
        lambda_base: float = 0.1,
        T_threshold: int = 800,
        epsilon_lambda: float = 1e-3,
        T_total: int = 1000
    ):
        self.lambda_base = lambda_base
        self.T_threshold = T_threshold
        self.epsilon_lambda = epsilon_lambda
        self.T_total = T_total
    
    def get_lambda_t(self, gradient: torch.Tensor, t: int) -> float:
        grad_norm = torch.norm(gradient, p=2) + self.epsilon_lambda
        step_size = self.lambda_base / grad_norm
        warmup = min((self.T_total - t) / (self.T_total - self.T_threshold), 1.0)
        lambda_t = step_size * warmup
        return lambda_t.item() if isinstance(lambda_t, torch.Tensor) else lambda_t
```
