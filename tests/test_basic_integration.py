#!/usr/bin/env python3
"""
Basic Integration Test for SOTA DDPM Components
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pkl_dg.models.diffusion import DDPMTrainer


class BasicUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        
    def forward(self, x, timesteps, cond=None):
        return self.conv(x)


def test_basic_integration():
    """Test basic integration without complex features."""
    print("🧪 Testing Basic DDPM Integration")
    print("=" * 40)
    
    # Force CPU usage
    torch.cuda.is_available = lambda: False
    
    # Simple model and config
    model = BasicUNet()
    config = {
        "num_timesteps": 50,
        "beta_schedule": "linear",  # Use simple linear schedule
        "use_diffusers_scheduler": False,  # Use manual schedule
        "mixed_precision": False,
        "use_ema": False,
        "learning_rate": 1e-4,
    }
    
    # Create trainer
    trainer = DDPMTrainer(model, config)
    
    # Test data
    batch = (
        torch.randn(2, 1, 32, 32),  # Small size
        torch.randn(2, 1, 32, 32)
    )
    
    # Test training step
    trainer.train()
    loss = trainer.training_step(batch, 0)
    
    print(f"✅ Training loss: {loss.item():.4f}")
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    
    # Test sampling
    trainer.eval()
    with torch.no_grad():
        samples = trainer.ddpm_sample(
            num_images=1,
            image_shape=(1, 32, 32)
        )
    
    print(f"✅ Generated samples shape: {samples.shape}")
    assert samples.shape == (1, 1, 32, 32)
    assert not torch.isnan(samples).any()
    
    print("\n🎉 Basic Integration Test Passed!")
    # Use assertion instead of return for pytest compatibility
    assert True  # Test completed successfully


if __name__ == "__main__":
    try:
        test_basic_integration()
        print(f"\n✅ SUCCESS")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        sys.exit(1)
