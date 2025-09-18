import torch

try:
    from einops import rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False


class AdaptiveSchedule:
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
        # Use einops for cleaner tensor flattening if available
        if EINOPS_AVAILABLE:
            grad_flat = rearrange(gradient, '... -> (...)')
        else:
            grad_flat = gradient.flatten()
        
        # Use more efficient vector norm
        grad_norm = torch.linalg.vector_norm(grad_flat) + self.epsilon_lambda
        step_size = self.lambda_base / grad_norm
        warmup = min((self.T_total - t) / (self.T_total - self.T_threshold), 1.0)
        lambda_t = step_size * warmup
        return lambda_t.item() if isinstance(lambda_t, torch.Tensor) else lambda_t

