import torch
from .base import GuidanceStrategy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pkl_dg.physics.signal_recovery import SignalRecovery


class PKLSignalRecoveryGuidance(GuidanceStrategy):
    """
    PKL guidance specifically adapted for signal recovery tasks.
    
    This guidance strategy adapts the Poisson-Kullback-Leibler divergence
    for signal recovery scenarios where the forward model is multiplicative
    (y = gain * x + noise) rather than convolutional.
    
    The key insight is that PKL can still work with multiplicative operators,
    but the gradient computation needs to account for the different forward
    model structure.
    """

    def __init__(self, epsilon: float = 1e-6, adaptive_epsilon: bool = True):
        """
        Initialize PKL guidance for signal recovery.
        
        Args:
            epsilon: Regularization parameter to avoid division by zero
            adaptive_epsilon: Whether to adapt epsilon based on signal level
        """
        self.epsilon = epsilon
        self.adaptive_epsilon = adaptive_epsilon

    def compute_gradient(
        self,
        x0_hat: torch.Tensor,
        y: torch.Tensor,
        forward_model: "SignalRecovery",
        t: int,
    ) -> torch.Tensor:
        """
        Compute PKL gradient for signal recovery.

        For signal recovery with forward model y = gain * x + background + noise,
        the PKL gradient becomes:
        
        grad = gain * (1 - y / (gain * x + background + sigma^2 + epsilon))
        
        Args:
            x0_hat: Current estimate of the clean image [B, C, H, W]
            y: Observed underexposed measurement [B, C, H, W]
            forward_model: SignalRecovery forward model
            t: Current diffusion time step
            
        Returns:
            Gradient tensor with same shape as x0_hat
        """
        # Apply forward model (gain multiplication)
        Ax = forward_model.apply_gain(x0_hat)  # gain * x
        Ax_plus_B = Ax + forward_model.background
        
        # Get readout noise variance
        sigma2 = forward_model.readout_noise_sigma ** 2
        
        # Adaptive epsilon based on signal level
        if self.adaptive_epsilon:
            epsilon = self.epsilon * torch.mean(Ax_plus_B).item()
        else:
            epsilon = self.epsilon
        
        # PKL denominator with readout noise
        denom = Ax_plus_B + sigma2 + epsilon
        
        # PKL ratio computation
        ratio = y / denom
        residual = 1.0 - ratio
        
        # Gradient: gain * residual (adjoint of gain operation)
        gradient = forward_model.apply_gain_adjoint(residual)
        
        return gradient

    def compute_data_consistency_loss(
        self,
        x0_hat: torch.Tensor,
        y: torch.Tensor,
        forward_model: "SignalRecovery",
    ) -> torch.Tensor:
        """
        Compute data consistency loss for signal recovery.
        
        Returns the PKL divergence: D_KL(y || gain*x + background)
        """
        Ax = forward_model.apply_gain(x0_hat)
        Ax_plus_B = Ax + forward_model.background
        sigma2 = forward_model.readout_noise_sigma ** 2
        
        # Avoid numerical issues
        epsilon = self.epsilon
        if self.adaptive_epsilon:
            epsilon = self.epsilon * torch.mean(Ax_plus_B).item()
            
        denom = Ax_plus_B + sigma2 + epsilon
        
        # PKL divergence components
        log_ratio = torch.log(y / denom + 1e-10)
        kl_div = y * log_ratio - y + denom
        
        return torch.mean(kl_div)


class AdaptivePKLSignalRecoveryGuidance(PKLSignalRecoveryGuidance):
    """
    Adaptive PKL guidance that adjusts parameters based on signal characteristics.
    
    This version adapts the guidance strength and regularization based on:
    - Signal-to-noise ratio
    - Exposure level
    - Diffusion timestep
    """

    def __init__(
        self, 
        epsilon: float = 1e-6,
        min_epsilon: float = 1e-8,
        max_epsilon: float = 1e-4,
        snr_threshold: float = 10.0
    ):
        super().__init__(epsilon=epsilon, adaptive_epsilon=True)
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.snr_threshold = snr_threshold

    def _estimate_snr(self, y: torch.Tensor, forward_model: "SignalRecovery") -> float:
        """Estimate signal-to-noise ratio of the measurement."""
        signal_power = torch.mean(y ** 2).item()
        noise_power = forward_model.readout_noise_sigma ** 2
        
        if noise_power < 1e-10:
            return float('inf')
        
        snr = signal_power / noise_power
        return float(snr)

    def _adapt_epsilon(
        self, 
        y: torch.Tensor, 
        forward_model: "SignalRecovery",
        t: int
    ) -> float:
        """Adapt epsilon based on signal characteristics and timestep."""
        # Base epsilon from signal level
        signal_level = torch.mean(y).item()
        base_epsilon = self.epsilon * max(signal_level, 0.01)
        
        # SNR-based adaptation
        snr = self._estimate_snr(y, forward_model)
        if snr < self.snr_threshold:
            # Lower SNR -> higher regularization
            snr_factor = self.snr_threshold / max(snr, 1.0)
        else:
            # Higher SNR -> lower regularization
            snr_factor = 1.0
        
        # Timestep-based adaptation (higher regularization early in diffusion)
        timestep_factor = 1.0 + 0.1 * (t / 1000.0)  # Assuming 1000 timesteps
        
        # Combine factors
        adapted_epsilon = base_epsilon * snr_factor * timestep_factor
        
        # Clamp to reasonable range
        return float(torch.clamp(
            torch.tensor(adapted_epsilon), 
            self.min_epsilon, 
            self.max_epsilon
        ).item())

    def compute_gradient(
        self,
        x0_hat: torch.Tensor,
        y: torch.Tensor,
        forward_model: "SignalRecovery",
        t: int,
    ) -> torch.Tensor:
        """Compute adaptive PKL gradient."""
        # Apply forward model
        Ax = forward_model.apply_gain(x0_hat)
        Ax_plus_B = Ax + forward_model.background
        
        # Get noise variance
        sigma2 = forward_model.readout_noise_sigma ** 2
        
        # Adaptive epsilon
        epsilon = self._adapt_epsilon(y, forward_model, t)
        
        # PKL computation
        denom = Ax_plus_B + sigma2 + epsilon
        ratio = y / denom
        residual = 1.0 - ratio
        
        # Gradient with adaptive scaling
        gradient = forward_model.apply_gain_adjoint(residual)
        
        return gradient


def create_pkl_signal_recovery_guidance(
    adaptive: bool = True,
    epsilon: float = 1e-6,
    **kwargs
) -> PKLSignalRecoveryGuidance:
    """
    Factory function to create PKL guidance for signal recovery.
    
    Args:
        adaptive: Whether to use adaptive version
        epsilon: Base regularization parameter
        **kwargs: Additional parameters for adaptive version
        
    Returns:
        PKL guidance instance
    """
    if adaptive:
        return AdaptivePKLSignalRecoveryGuidance(epsilon=epsilon, **kwargs)
    else:
        return PKLSignalRecoveryGuidance(epsilon=epsilon)
