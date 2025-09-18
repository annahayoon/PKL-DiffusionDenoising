import os

import pytest
import torch

from pkl_dg.models import UNet, DDPMTrainer


def minimal_unet_config():
    # Keep tiny to speed up tests
    return {
        "sample_size": 16,
        "in_channels": 1,
        "out_channels": 1,
        "layers_per_block": 1,
        "block_out_channels": [16, 32],
        "down_block_types": ["DownBlock2D", "AttnDownBlock2D"],
        "up_block_types": ["AttnUpBlock2D", "UpBlock2D"],
    }


@pytest.mark.parametrize("batch,channels,height,width", [(2, 1, 16, 16), (1, 1, 16, 16)])
def test_unet_forward_shape(batch, channels, height, width):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(minimal_unet_config()).to(device)
    x = torch.randn(batch, channels, height, width, device=device)
    t = torch.randint(0, 1000, (batch,), device=device)
    y = model(x, t)
    assert y.shape == x.shape


def test_ddpm_noise_schedule_shapes():
    model = UNet(minimal_unet_config())
    trainer = DDPMTrainer(model, {"num_timesteps": 1000, "beta_schedule": "cosine"})
    assert trainer.betas.shape[0] == 1000
    assert torch.all(trainer.alphas > 0)
    assert torch.all(trainer.alphas < 1)
    assert trainer.sqrt_alphas_cumprod.shape == torch.Size([1000])
    assert trainer.sqrt_one_minus_alphas_cumprod.shape == torch.Size([1000])


def _fake_batch(batch=2, channels=1, size=16, device="cpu"):
    x = torch.randn(batch, channels, size, size, device=device)
    y = torch.zeros(batch, dtype=torch.long, device=device)
    return x, y


def test_training_step_runs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(minimal_unet_config()).to(device)
    trainer = DDPMTrainer(
        model,
        {"num_timesteps": 1000, "beta_schedule": "cosine", "use_ema": False},
    ).to(device)
    batch = _fake_batch(device=device)
    loss = trainer.training_step(batch, 0)
    assert torch.is_tensor(loss) and loss.ndim == 0


def test_validation_step_runs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(minimal_unet_config()).to(device)
    trainer = DDPMTrainer(
        model,
        {"num_timesteps": 1000, "beta_schedule": "cosine", "use_ema": False},
    ).to(device)
    batch = _fake_batch(device=device)
    val = trainer.validation_step(batch, 0)
    assert torch.is_tensor(val) and val.ndim == 0


def test_configure_optimizers_returns_optimizer_and_scheduler():
    model = UNet(minimal_unet_config())
    trainer = DDPMTrainer(
        model,
        {"learning_rate": 1e-4, "weight_decay": 1e-6, "use_scheduler": True},
    )
    optimizers, schedulers = trainer.configure_optimizers()
    assert isinstance(optimizers, list) and len(optimizers) == 1
    assert isinstance(schedulers, list) and len(schedulers) == 1



