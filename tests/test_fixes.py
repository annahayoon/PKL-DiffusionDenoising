#!/usr/bin/env python3
"""
Test the specific fixes for cascaded sampling and hierarchical strategy.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class SimpleTestUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        
    def forward(self, x, timesteps, cond=None):
        return self.conv(x)


def test_cascaded_sampling_api_fix():
    """Test the cascaded sampling API fix."""
    print("🧪 Testing Cascaded Sampling API Fix")
    print("-" * 40)
    
    try:
        from pkl_dg.models.cascaded_sampling import CascadedSampler
        from pkl_dg.models.diffusion import DDPMTrainer
        
        # Create test models using legacy API
        models = {}
        resolutions = [64, 128]
        
        for res in resolutions:
            base_unet = SimpleTestUNet()
            config = {
                'num_timesteps': 50,
                'beta_schedule': 'linear',
                'use_diffusers_scheduler': False,
                'mixed_precision': False,
                'use_ema': False,
            }
            trainer = DDPMTrainer(base_unet, config)
            models[res] = trainer
        
        # Test legacy API
        sampler = CascadedSampler(
            models=models,
            resolutions=resolutions,
            num_inference_steps=10,
            device='cpu'
        )
        
        print("✅ CascadedSampler legacy API initialization successful")
        
        # Test sampling
        with torch.no_grad():
            samples = sampler.sample(batch_size=1, verbose=False)
            print(f"✅ Sample generation successful: {samples.shape}")
            assert samples.shape == (1, 1, 128, 128)
            assert not torch.isnan(samples).any()
        
        print("🎉 Cascaded Sampling API Fix: SUCCESS")
        return True
        
    except Exception as e:
        print(f"❌ Cascaded Sampling API Fix: FAILED - {e}")
        return False


def test_hierarchical_device_fix():
    """Test the hierarchical device handling fix."""
    print("\n🧪 Testing Hierarchical Device Fix")
    print("-" * 40)
    
    try:
        from pkl_dg.models.hierarchical_strategy import HierarchicalTrainer
        from pkl_dg.models.progressive import ProgressiveUNet
        
        # Force CPU usage
        torch.cuda.is_available = lambda: False
        
        base_unet = SimpleTestUNet()
        progressive_unet = ProgressiveUNet(base_unet, max_resolution=128)
        
        config = {
            "num_timesteps": 50,
            "beta_schedule": "linear",
            "use_diffusers_scheduler": False,
            "mixed_precision": False,
            "use_ema": False,
            "progressive": {
                "enabled": True,
                "max_resolution": 128,
                "start_resolution": 64,
                "epochs_per_resolution": [2, 3],
                "cross_resolution_consistency": False,  # Disable to avoid complex consistency checks
                "consistency_weight": 0.0,
            },
            "hierarchical": {
                "enabled": True,
                "cross_scale_consistency_weight": 0.0,  # Disable to avoid feature extractor
            }
        }
        
        trainer = HierarchicalTrainer(progressive_unet, config)
        print("✅ HierarchicalTrainer initialization successful")
        
        # Test training step with CPU tensors
        batch = (
            torch.randn(2, 1, 64, 64),  # target
            torch.randn(2, 1, 64, 64)   # source
        )
        
        trainer.train()
        loss = trainer.training_step(batch, 0)
        
        print(f"✅ Training step successful: loss = {loss.item():.4f}")
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        
        print("🎉 Hierarchical Device Fix: SUCCESS")
        return True
        
    except Exception as e:
        print(f"❌ Hierarchical Device Fix: FAILED - {e}")
        return False


def main():
    """Run fix tests."""
    print("🔧 Testing SOTA Component Fixes")
    print("=" * 50)
    
    tests = [
        test_cascaded_sampling_api_fix,
        test_hierarchical_device_fix,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} CRASHED: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Fix Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All fixes working correctly!")
        print("✅ Cascaded sampling API fixed")
        print("✅ Hierarchical device handling fixed")
    else:
        print(f"⚠️ {failed} fixes still need work")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
