import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
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


def _make_tiny_dataset(root: Path, num_images: int = 4, size: int = 32):
    for split in ["train", "val"]:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_images):
            arr = (np.random.rand(size, size) * 255).astype(np.uint8)
            Image.fromarray(arr).save(split_dir / f"img_{i}.png")


@pytest.mark.cpu
def test_run_training_minimal(tmp_path: Path, monkeypatch):
    # Create tiny dataset
    data_dir = tmp_path / "data"
    _make_tiny_dataset(data_dir, num_images=4, size=32)

    # Minimal config using the Hydra structure expected by the script
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "test", "seed": 0, "mixed_precision": False},
            "paths": {
                "data": str(data_dir),
                "checkpoints": str(tmp_path / "ckpt"),
            },
            "wandb": {"project": "test", "entity": None, "mode": "disabled"},
            "physics": {"psf_path": None, "background": 0.0},
            "data": {"image_size": 32, "min_intensity": 0, "max_intensity": 1000},
            "training": {
                "batch_size": 2,
                "num_workers": 0,
                "max_epochs": 1,
                "num_gpus": 1,
                "gradient_clip": 1.0,
                "accumulate_grad_batches": 1,
                "val_check_interval": 1.0,
                "early_stopping_patience": 1,
                "learning_rate": 1e-4,
                "weight_decay": 1e-6,
                "use_scheduler": False,
                "num_timesteps": 10,
                "beta_schedule": "cosine",
                "use_ema": False,
            },
            "model": {
                "sample_size": 32,
                "in_channels": 1,
                "out_channels": 1,
                "layers_per_block": 1,
                "block_out_channels": [8, 16],
                "down_block_types": ["DownBlock2D", "DownBlock2D"],
                "up_block_types": ["UpBlock2D", "UpBlock2D"],
            },
        }
    )

    trainer = run_training(cfg)

    # Verify model saved
    final_model = Path(cfg.paths.checkpoints) / "final_model.pt"
    assert final_model.exists(), "Expected final model to be saved"

    # Basic sanity on buffers (noise schedule registered)
    assert hasattr(trainer, "alphas_cumprod"), "Noise schedule buffers not set"


