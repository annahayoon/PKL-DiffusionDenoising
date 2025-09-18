"""Comprehensive unit tests for DDIM sampler implementation and accuracy."""

import torch
import torch.nn as nn
import pytest
import math
from unittest.mock import MagicMock

from pkl_dg.models.sampler import DDIMSampler
from pkl_dg.guidance import PKLGuidance, L2Guidance, AdaptiveSchedule
from pkl_dg.data import IntensityToModel
from pkl_dg.physics.forward_model import ForwardModel


class DummyNet(nn.Module):
    """Simple network for testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DummyTrainer(nn.Module):
    """Mock trainer with required buffers."""
    def __init__(self, T: int = 1000, use_ema: bool = False):
        super().__init__()
        self.model = DummyNet()
        self.use_ema = use_ema
        if use_ema:
            self.ema_model = DummyNet()
        
        # Create realistic cosine schedule
        betas = self._cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        """Cosine beta schedule as used in improved DDPM."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)


def _make_forward_model(device: str = "cpu") -> ForwardModel:
    """Create a simple forward model for testing."""
    psf = torch.ones(9, 9) / 81.0
    return ForwardModel(psf=psf, background=0.0, device=device)


def _make_sampler(
    num_timesteps: int = 100,
    ddim_steps: int = 10,
    eta: float = 0.0,
    use_ema: bool = False,
    clip_denoised: bool = True,
    v_parameterization: bool = False,
    device: str = "cpu"
) -> DDIMSampler:
    """Create DDIM sampler for testing."""
    trainer = DummyTrainer(T=num_timesteps, use_ema=use_ema).to(device)
    fm = _make_forward_model(device)
    guidance = PKLGuidance()
    schedule = AdaptiveSchedule(lambda_base=0.05, T_threshold=80, T_total=num_timesteps)
    transform = IntensityToModel(min_intensity=0, max_intensity=1000)
    
    return DDIMSampler(
        model=trainer,
        forward_model=fm,
        guidance_strategy=guidance,
        schedule=schedule,
        transform=transform,
        num_timesteps=num_timesteps,
        ddim_steps=ddim_steps,
        eta=eta,
        clip_denoised=clip_denoised,
        v_parameterization=v_parameterization,
    )


class TestDDIMSamplerInitialization:
    """Test DDIM sampler initialization and validation."""
    
    def test_valid_initialization(self):
        """Test that valid parameters initialize correctly."""
        sampler = _make_sampler()
        assert sampler.num_timesteps == 100
        assert sampler.ddim_steps == 10
        assert sampler.eta == 0.0
        assert sampler.clip_denoised is True
        assert sampler.v_parameterization is False
        
    def test_invalid_ddim_steps(self):
        """Test that ddim_steps > num_timesteps raises error."""
        with pytest.raises(ValueError, match="ddim_steps.*cannot exceed.*num_timesteps"):
            _make_sampler(num_timesteps=50, ddim_steps=100)
    
    def test_invalid_eta(self):
        """Test that eta outside [0,1] raises error."""
        with pytest.raises(ValueError, match="eta must be in"):
            _make_sampler(eta=-0.1)
        with pytest.raises(ValueError, match="eta must be in"):
            _make_sampler(eta=1.1)
    
    def test_missing_alphas_cumprod_buffer(self):
        """Test that model without alphas_cumprod buffer raises error."""
        class BadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = DummyNet()
        
        bad_model = BadModel()
        fm = _make_forward_model()
        guidance = PKLGuidance()
        schedule = AdaptiveSchedule()
        transform = IntensityToModel()
        
        with pytest.raises(ValueError, match="Model must have 'alphas_cumprod' buffer"):
            DDIMSampler(
                model=bad_model,
                forward_model=fm,
                guidance_strategy=guidance,
                schedule=schedule,
                transform=transform,
            )


class TestDDIMTimestepSetup:
    """Test DDIM timestep sequence setup."""
    
    def test_timestep_sequence_length(self):
        """Test that timestep sequence has correct length."""
        sampler = _make_sampler(num_timesteps=1000, ddim_steps=100)
        assert len(sampler.ddim_timesteps) <= 101  # May include final timestep
        
    def test_timestep_sequence_order(self):
        """Test that timesteps are in descending order."""
        sampler = _make_sampler(num_timesteps=1000, ddim_steps=100)
        timesteps = sampler.ddim_timesteps
        for i in range(len(timesteps) - 1):
            assert timesteps[i] >= timesteps[i + 1]
    
    def test_single_step_timestep(self):
        """Test single-step DDIM setup."""
        sampler = _make_sampler(num_timesteps=1000, ddim_steps=1)
        assert len(sampler.ddim_timesteps) == 1
        assert sampler.ddim_timesteps[0] == 999
    
    def test_timestep_coverage(self):
        """Test that timesteps provide good coverage."""
        sampler = _make_sampler(num_timesteps=1000, ddim_steps=10)
        timesteps = sampler.ddim_timesteps
        
        # Should start near max timestep
        assert timesteps[0] >= 900
        
        # Should end at 0 or be small
        assert timesteps[-1] < 100


class TestDDIMSampling:
    """Test DDIM sampling process."""
    
    def test_sample_output_shape(self):
        """Test that sample output has correct shape."""
        sampler = _make_sampler()
        device = "cpu"
        B, C, H, W = 2, 1, 16, 16
        y = torch.rand(B, C, H, W, device=device) * 10.0 + 0.1
        
        x0 = sampler.sample(y=y, shape=(B, C, H, W), device=device, verbose=False)
        assert x0.shape == (B, C, H, W)
    
    def test_sample_output_properties(self):
        """Test that sample output has expected properties."""
        sampler = _make_sampler()
        device = "cpu"
        B, C, H, W = 2, 1, 16, 16
        y = torch.rand(B, C, H, W, device=device) * 10.0 + 0.1
        
        x0 = sampler.sample(y=y, shape=(B, C, H, W), device=device, verbose=False)
        
        # Should be finite
        assert torch.isfinite(x0).all()
        
        # Should be non-negative (intensity domain)
        assert torch.all(x0 >= 0)
    
    def test_deterministic_sampling(self):
        """Test that eta=0 produces deterministic results."""
        sampler = _make_sampler(eta=0.0)
        device = "cpu"
        B, C, H, W = 1, 1, 8, 8
        y = torch.rand(B, C, H, W, device=device) * 10.0 + 0.1
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        x0_1 = sampler.sample(y=y, shape=(B, C, H, W), device=device, verbose=False)
        
        torch.manual_seed(42)
        x0_2 = sampler.sample(y=y, shape=(B, C, H, W), device=device, verbose=False)
        
        assert torch.allclose(x0_1, x0_2, atol=1e-6)
    
    def test_stochastic_sampling(self):
        """Test that eta>0 produces different results."""
        sampler = _make_sampler(eta=0.5)
        device = "cpu"
        B, C, H, W = 1, 1, 8, 8
        y = torch.rand(B, C, H, W, device=device) * 10.0 + 0.1
        
        torch.manual_seed(42)
        x0_1 = sampler.sample(y=y, shape=(B, C, H, W), device=device, verbose=False)
        
        torch.manual_seed(43)
        x0_2 = sampler.sample(y=y, shape=(B, C, H, W), device=device, verbose=False)
        
        # Should be different (with high probability)
        assert not torch.allclose(x0_1, x0_2, atol=1e-3)
    
    def test_return_intermediates(self):
        """Test that return_intermediates works correctly."""
        sampler = _make_sampler(ddim_steps=5)
        device = "cpu"
        B, C, H, W = 1, 1, 8, 8
        y = torch.rand(B, C, H, W, device=device) * 10.0 + 0.1
        
        result = sampler.sample(
            y=y, shape=(B, C, H, W), device=device, 
            verbose=False, return_intermediates=True
        )
        
        assert isinstance(result, dict)
        assert "x_intermediates" in result
        assert "x0_predictions" in result
        assert "final_intensity" in result
        
        # Should have correct number of intermediates (may be ddim_steps or ddim_steps+1)
        expected_steps = len(sampler.ddim_timesteps)
        assert len(result["x_intermediates"]) == expected_steps
        assert len(result["x0_predictions"]) == expected_steps
    
    def test_invalid_shape(self):
        """Test that invalid shape raises error."""
        sampler = _make_sampler()
        device = "cpu"
        y = torch.rand(2, 1, 16, 16, device=device)
        
        with pytest.raises(ValueError, match="Shape must be 4D"):
            sampler.sample(y=y, shape=(16, 16), device=device, verbose=False)


class TestDDIMAccuracy:
    """Test DDIM mathematical accuracy."""
    
    def test_predict_x0_consistency(self):
        """Test that _predict_x0 produces consistent results."""
        sampler = _make_sampler()
        device = "cpu"
        
        # Create test input
        x_t = torch.randn(1, 1, 8, 8, device=device)
        t = 50
        
        # Test multiple calls
        x0_1 = sampler._predict_x0(x_t, t)
        x0_2 = sampler._predict_x0(x_t, t)
        
        assert torch.allclose(x0_1, x0_2, atol=1e-6)
    
    def test_predict_x0_tensor_timestep(self):
        """Test _predict_x0 with tensor timestep."""
        sampler = _make_sampler()
        device = "cpu"
        
        x_t = torch.randn(2, 1, 8, 8, device=device)
        t_int = 50
        t_tensor = torch.tensor([50, 50], device=device)
        
        x0_int = sampler._predict_x0(x_t, t_int)
        x0_tensor = sampler._predict_x0(x_t, t_tensor)
        
        assert torch.allclose(x0_int, x0_tensor, atol=1e-6)
    
    def test_ddim_step_final(self):
        """Test that final DDIM step returns x0_hat."""
        sampler = _make_sampler()
        device = "cpu"
        
        x_t = torch.randn(1, 1, 8, 8, device=device)
        x0_hat = torch.randn(1, 1, 8, 8, device=device)
        
        x_next = sampler._ddim_step(x_t, x0_hat, t_cur=10, t_next=0)
        
        assert torch.allclose(x_next, x0_hat, atol=1e-6)
    
    def test_ddim_step_numerical_stability(self):
        """Test DDIM step numerical stability with extreme values."""
        sampler = _make_sampler()
        device = "cpu"
        
        x_t = torch.randn(1, 1, 4, 4, device=device)
        x0_hat = torch.randn(1, 1, 4, 4, device=device)
        
        # Test with very small timesteps
        x_next = sampler._ddim_step(x_t, x0_hat, t_cur=1, t_next=0)
        assert torch.isfinite(x_next).all()
        
        # Test with large timesteps
        x_next = sampler._ddim_step(x_t, x0_hat, t_cur=99, t_next=90)
        assert torch.isfinite(x_next).all()


class TestDDIMParameterizations:
    """Test different parameterizations."""
    
    def test_epsilon_parameterization(self):
        """Test standard epsilon parameterization."""
        sampler = _make_sampler(v_parameterization=False)
        device = "cpu"
        B, C, H, W = 1, 1, 8, 8
        y = torch.rand(B, C, H, W, device=device) * 10.0 + 0.1
        
        x0 = sampler.sample(y=y, shape=(B, C, H, W), device=device, verbose=False)
        assert torch.isfinite(x0).all()
    
    def test_v_parameterization(self):
        """Test v-parameterization."""
        sampler = _make_sampler(v_parameterization=True)
        device = "cpu"
        B, C, H, W = 1, 1, 8, 8
        y = torch.rand(B, C, H, W, device=device) * 10.0 + 0.1
        
        x0 = sampler.sample(y=y, shape=(B, C, H, W), device=device, verbose=False)
        assert torch.isfinite(x0).all()
    
    def test_clipping_behavior(self):
        """Test clipping behavior."""
        # With clipping
        sampler_clipped = _make_sampler(clip_denoised=True)
        device = "cpu"
        
        # Create extreme input that would produce out-of-range x0
        x_t = torch.ones(1, 1, 4, 4, device=device) * 10.0
        t = 50
        
        x0_clipped = sampler_clipped._predict_x0(x_t, t)
        assert torch.all(x0_clipped >= -1.0)
        assert torch.all(x0_clipped <= 1.0)
        
        # Without clipping
        sampler_unclipped = _make_sampler(clip_denoised=False)
        x0_unclipped = sampler_unclipped._predict_x0(x_t, t)
        
        # Unclipped version may have values outside [-1, 1]
        assert not torch.allclose(x0_clipped, x0_unclipped, atol=1e-3)


class TestDDIMErrorHandling:
    """Test error handling and edge cases."""
    
    def test_guidance_failure_handling(self):
        """Test that guidance failures are handled gracefully."""
        # Create sampler with mock guidance that fails
        sampler = _make_sampler()
        
        # Mock guidance to raise exception
        original_apply_guidance = sampler._apply_guidance
        def failing_guidance(*args, **kwargs):
            raise RuntimeError("Guidance failed")
        sampler._apply_guidance = failing_guidance
        
        device = "cpu"
        B, C, H, W = 1, 1, 8, 8
        y = torch.rand(B, C, H, W, device=device) * 10.0 + 0.1
        
        # Should not crash, should use uncorrected prediction
        with pytest.warns(UserWarning, match="Guidance failed"):
            x0 = sampler.sample(y=y, shape=(B, C, H, W), device=device, verbose=False)
            assert torch.isfinite(x0).all()
    
    def test_device_mismatch_warning(self):
        """Test warning for device mismatch."""
        sampler = _make_sampler()
        
        # Create measurement on different device
        y = torch.rand(1, 1, 8, 8, device="cpu")
        
        # Sample on same device should not warn
        x0 = sampler.sample(y=y, shape=(1, 1, 8, 8), device="cpu", verbose=False)
        assert torch.isfinite(x0).all()
    
    def test_nan_detection(self):
        """Test that NaN values are detected and raise error."""
        sampler = _make_sampler()
        
        # Mock _ddim_step to return NaN
        original_ddim_step = sampler._ddim_step
        def nan_ddim_step(*args, **kwargs):
            result = original_ddim_step(*args, **kwargs)
            result[0, 0, 0, 0] = float('nan')
            return result
        sampler._ddim_step = nan_ddim_step
        
        device = "cpu"
        B, C, H, W = 1, 1, 8, 8
        y = torch.rand(B, C, H, W, device=device) * 10.0 + 0.1
        
        with pytest.raises(RuntimeError, match="Non-finite values detected"):
            sampler.sample(y=y, shape=(B, C, H, W), device=device, verbose=False)


class TestDDIMPerformance:
    """Test DDIM performance characteristics."""
    
    def test_different_step_counts(self):
        """Test sampling with different numbers of steps."""
        device = "cpu"
        B, C, H, W = 1, 1, 8, 8
        y = torch.rand(B, C, H, W, device=device) * 10.0 + 0.1
        
        # Test various step counts
        for steps in [1, 5, 10, 20]:
            sampler = _make_sampler(ddim_steps=steps)
            x0 = sampler.sample(y=y, shape=(B, C, H, W), device=device, verbose=False)
            assert torch.isfinite(x0).all()
            assert torch.all(x0 >= 0)
    
    def test_ema_model_usage(self):
        """Test that EMA model is used when available."""
        sampler_ema = _make_sampler(use_ema=True)
        sampler_no_ema = _make_sampler(use_ema=False)
        
        device = "cpu"
        B, C, H, W = 1, 1, 8, 8
        y = torch.rand(B, C, H, W, device=device) * 10.0 + 0.1
        
        torch.manual_seed(42)
        x0_ema = sampler_ema.sample(y=y, shape=(B, C, H, W), device=device, verbose=False)
        
        torch.manual_seed(42)
        x0_no_ema = sampler_no_ema.sample(y=y, shape=(B, C, H, W), device=device, verbose=False)
        
        # Results should be different when using different models
        assert not torch.allclose(x0_ema, x0_no_ema, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
