import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from omegaconf import OmegaConf

# Import from the actual codebase
from pkl_dg.models.unet import UNet
from pkl_dg.models.diffusion import DDPMTrainer

def run_training(config):
    """Test utility function for training."""
    model = UNet(config.model)
    trainer = DDPMTrainer(model, config.training)
    return trainer


# Import shared test utilities
from tests.test_utils import make_tiny_dataset as _make_tiny_dataset


@pytest.mark.cpu
def test_multi_gpu_config_falls_back_on_cpu(tmp_path: Path):
    # Create tiny dataset
    data_dir = tmp_path / "data"
    _make_tiny_dataset(data_dir, num_images=4, size=16)

    cfg = OmegaConf.create(
        {
            "experiment": {"name": "mgpu_fallback", "seed": 0, "mixed_precision": False, "device": "cpu"},
            "paths": {"data": str(data_dir), "checkpoints": str(tmp_path / "ckpt"), "outputs": str(tmp_path / "outs"), "logs": str(tmp_path / "logs")},
            "wandb": {"project": "test", "entity": None, "mode": "disabled"},
            "physics": {"psf_path": None, "background": 0.0},
            "data": {"image_size": 16, "min_intensity": 0, "max_intensity": 1000},
            "training": {
                "batch_size": 2,
                "num_workers": 0,
                "max_epochs": 1,
                "num_timesteps": 10,
                "beta_schedule": "cosine",
                "use_ema": False,
                # Request multi-GPU but we are on CPU → should fallback
                "devices": 2,
                "accelerator": "gpu",
                "precision": "16-mixed",
                "distributed": {"enabled": True},
            },
            "model": {
                "sample_size": 16,
                "in_channels": 1,
                "out_channels": 1,
                "layers_per_block": 1,
                "block_out_channels": [8, 8],
                "down_block_types": ["DownBlock2D", "DownBlock2D"],
                "up_block_types": ["UpBlock2D", "UpBlock2D"],
            },
        }
    )

    trainer = run_training(cfg)

    # Verify fallback training saved a final checkpoint
    final_model = Path(cfg.paths.checkpoints) / "final_model.pt"
    assert final_model.exists(), "Expected final model to be saved in fallback mode"

    # Basic sanity: noise schedule should exist
    assert hasattr(trainer, "alphas_cumprod")
