import torch

class PoissonNoise:
    """Poisson noise model for photon-limited imaging."""

    @staticmethod
    def add_noise(signal: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        signal = torch.clamp(signal, min=0)
        signal_scaled = signal * gain
        noisy = torch.poisson(signal_scaled) / gain
        return noisy


class GaussianBackground:
    """Gaussian background noise model."""

    @staticmethod
    def add_background(signal: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        noise = torch.randn_like(signal) * std + mean
        return signal + noise


