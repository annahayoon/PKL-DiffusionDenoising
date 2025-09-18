import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from pkl_dg.models.unet import UNet
from pkl_dg.models.diffusion import DDPMTrainer


@pytest.mark.cpu
def test_dpm_solver_fast_guided_sampling_runs_cpu():
    # Minimal model and training config
    model_cfg = {
        "sample_size": 16,
        "in_channels": 1,
        "out_channels": 1,
        "layers_per_block": 1,
        "block_out_channels": [8, 16],
        "down_block_types": ["DownBlock2D", "DownBlock2D"],
        "up_block_types": ["UpBlock2D", "UpBlock2D"],
    }
    training_cfg = {
        "num_timesteps": 10,
        "beta_schedule": "cosine",
        "use_ema": False,
        "learning_rate": 1e-4,
        "use_scheduler": False,
        "max_epochs": 1,
        # enable diffusers schedulers if available
        "use_diffusers_scheduler": True,
        "scheduler_type": "dpm_solver",
        "use_karras_sigmas": True,
    }

    device = "cpu"
    unet = UNet(model_cfg)
    ddpm = DDPMTrainer(model=unet, config=training_cfg)
    ddpm.to(device)
    ddpm.eval()

    # Dummy measurement y
    y = torch.rand(1, 1, 16, 16, device=device)

    # Dummy forward model and simple guidance strategy that returns zeros
    class _Forward:
        background = 0.0
        def apply_psf(self, x):
            return x
        def apply_psf_adjoint(self, x):
            return x
    forward = _Forward()

    class _Guidance:
        def compute_gradient(self, x0_int, y, forward_model, t):
            return torch.zeros_like(x0_int)
    guidance = _Guidance()

    # Simple linear transform between intensity and model domain
    class _Transform:
        def __call__(self, x):
            return (x - 0.5) * 2.0
        def inverse(self, x):
            return torch.clamp(x * 0.5 + 0.5, min=0.0)
    transform = _Transform()

    class _Schedule:
        def get_lambda_t(self, grad, t):
            return 0.0
    schedule = _Schedule()

    # Run fast guided sampling with a few steps
    out = ddpm.sample_with_scheduler_and_guidance(
        y=y,
        forward_model=forward,
        guidance_strategy=guidance,
        schedule=schedule,
        transform=transform,
        num_inference_steps=4,
        device=device,
        use_ema=False,
        use_karras_sigmas=True,
    )

    # Verify shape and finiteness (model domain output here)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 1, 16, 16)
    assert torch.isfinite(out).all()
