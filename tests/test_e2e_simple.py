#!/usr/bin/env python3
"""
Simple End-to-End SOTA DDPM Pipeline Test
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.models.progressive import ProgressiveTrainer, ProgressiveUNet


class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 1, 3, padding=1)
        self.time_embed = nn.Embedding(1000, 32)
        
    def forward(self, x, timesteps, cond=None):
        t_emb = self.time_embed(timesteps).view(x.shape[0], -1, 1, 1)
        h = torch.relu(self.conv1(x))
        h = h + t_emb.expand_as(h)
        return self.conv2(h)


def test_e2e_pipeline():
    """Test complete E2E pipeline."""
    print("🚀 Testing E2E SOTA DDPM Pipeline")
    print("=" * 50)
    
    # Setup - force CPU to avoid device issues
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA
    
    device = 'cpu'
    model = SimpleUNet().to(device)
    progressive_model = ProgressiveUNet(model, max_resolution=128)
    
    config = {
        "num_timesteps": 50,
        "beta_schedule": "cosine",
        "use_diffusers_scheduler": True,
        "mixed_precision": False,
        "use_ema": False,
        "progressive": {
            "enabled": True,
            "max_resolution": 128,
            "start_resolution": 64,
            "epochs_per_resolution": [2, 3],
            "cross_resolution_consistency": True,
            "consistency_weight": 0.1,
        }
    }
    
    # Test progressive trainer - keep on CPU
    trainer = ProgressiveTrainer(progressive_model, config)
    trainer.to(device)  # Ensure trainer stays on CPU
    
    # Create test data on the same device
    batch_size = 2
    test_batch = (
        torch.randn(batch_size, 1, 64, 64).to(device),  # target
        torch.randn(batch_size, 1, 64, 64).to(device)   # source
    )
    
    # Test training
    trainer.train()
    loss = trainer.training_step(test_batch, 0)
    
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    print(f"✅ Training loss: {loss.item():.4f}")
    
    # Test sampling
    trainer.eval()
    with torch.no_grad():
        samples = trainer.ddpm_sample(
            num_images=1,
            image_shape=(1, 64, 64)
        )
        
    assert samples.shape == (1, 1, 64, 64)
    assert not torch.isnan(samples).any()
    print(f"✅ Sampling successful: {samples.shape}")
    
    # Test progressive functionality
    config = trainer.get_current_resolution_config()
    assert config is not None
    print(f"✅ Progressive config: resolution={config.get('resolution', 'N/A')}")
    
    # Test phase advancement
    initial_phase = trainer.current_phase
    trainer.advance_progressive_phase()
    print(f"✅ Phase advancement: {initial_phase} → {trainer.current_phase}")
    
    print("\n🎉 E2E Pipeline Test Completed Successfully!")
    print("✅ All core components working correctly")
    print("✅ Progressive training functional")
    print("✅ Sampling pipeline operational")
    
    return True


if __name__ == "__main__":
    success = test_e2e_pipeline()
    sys.exit(0 if success else 1)
