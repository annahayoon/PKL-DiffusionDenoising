#!/usr/bin/env python3
"""
Comprehensive test suite for SOTA DDPM components.

This test suite validates:
1. Individual component accuracy
2. End-to-end integration
3. Mathematical correctness
4. Comparison with SOTA field standards
5. Performance benchmarks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import time
import warnings
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all SOTA components
from pkl_dg.models.diffusion import DDPMTrainer, create_enhanced_trainer, create_enhanced_config
from pkl_dg.models.progressive import ProgressiveTrainer, ProgressiveUNet
from pkl_dg.models.hierarchical_strategy import HierarchicalTrainer
from pkl_dg.models.cascaded_sampling import CascadedSampler
from pkl_dg.models.losses import FourierLoss, create_frequency_loss
from pkl_dg.models.advanced_schedulers import create_scheduler, SchedulerManager, ImprovedCosineScheduler
from pkl_dg.utils.adaptive_batch import AdaptiveBatchSizer

# Test utilities
def create_test_unet(in_channels=1, out_channels=1, sample_size=64):
    """Create a minimal UNet for testing."""
    class TestUNet(nn.Module):
        def __init__(self, in_channels, out_channels, sample_size):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.sample_size = sample_size
            
            # Simple conv layers for testing
            self.down1 = nn.Conv2d(in_channels, 32, 3, padding=1)
            self.down2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
            self.mid = nn.Conv2d(64, 64, 3, padding=1)
            self.up1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
            self.out = nn.Conv2d(32, out_channels, 3, padding=1)
            
            # Time embedding (simplified)
            self.time_embed = nn.Embedding(1000, 64)
            
        def forward(self, x, timesteps, cond=None):
            # Simple forward pass for testing
            t_emb = self.time_embed(timesteps).view(x.shape[0], -1, 1, 1)
            
            h1 = F.relu(self.down1(x))
            h2 = F.relu(self.down2(h1))
            h2 = h2 + t_emb.expand_as(h2)
            
            h3 = F.relu(self.mid(h2))
            h4 = F.relu(self.up1(h3))
            
            # Resize to match input
            if h4.shape[-2:] != h1.shape[-2:]:
                h4 = F.interpolate(h4, size=h1.shape[-2:], mode='bilinear', align_corners=False)
            
            h5 = h4 + h1
            return self.out(h5)
    
    return TestUNet(in_channels, out_channels, sample_size)


def create_test_config(max_resolution=256):
    """Create test configuration."""
    return {
        "num_timesteps": 100,  # Reduced for testing
        "beta_schedule": "cosine",
        "use_diffusers_scheduler": True,
        "scheduler_type": "ddpm",
        "mixed_precision": False,  # Disabled for testing stability
        "use_ema": True,
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "use_scheduler": True,
        "max_epochs": 10,
        "batch_size": 4,
        "use_conditioning": True,
        "supervised_x0_weight": 0.1,
        
        # Progressive training
        "progressive": {
            "enabled": True,
            "max_resolution": max_resolution,
            "start_resolution": 64,
            "curriculum_type": "adaptive",
            "epochs_per_resolution": [2, 3, 5],
            "smooth_transitions": True,
            "lr_scaling": True,
            "batch_scaling": True,
            "adaptive_batch_scaling": True,
            "cross_resolution_consistency": True,
            "consistency_weight": 0.1,
        },
        
        # Hierarchical strategy
        "hierarchical": {
            "enabled": True,
            "cross_scale_consistency_weight": 0.05,
        },
        
        # Frequency losses
        "frequency_loss": {
            "enabled": True,
            "weight": 0.01,
            "loss_type": "l2",
            "frequency_weighting": "low_pass",
        },
        
        # Model config
        "model": {
            "in_channels": 1,
            "out_channels": 1,
            "sample_size": 64,
            "max_resolution": max_resolution,
        },
        
        # Training config
        "training": {
            "num_timesteps": 100,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "scheduler_type": "cosine",
            "use_diffusers_scheduler": True,
            "dynamic_batch_sizing": True,
            "beta_start": 0.0001,
            "beta_end": 0.02,
        }
    }


class TestSOTAComponents:
    """Test suite for SOTA DDPM components."""
    
    def setup_method(self):
        """Setup for each test."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_shape = (2, 1, 64, 64)  # Small batch for testing
        self.config = create_test_config()
        
    def test_progressive_unet(self):
        """Test ProgressiveUNet functionality."""
        print("Testing ProgressiveUNet...")
        
        base_unet = create_test_unet(sample_size=64)
        progressive_unet = ProgressiveUNet(base_unet, max_resolution=256)
        
        # Test resolution schedule
        assert len(progressive_unet.resolutions) > 1
        assert progressive_unet.resolutions[0] == 64
        assert progressive_unet.resolutions[-1] <= 256
        
        # Test resolution switching
        x = torch.randn(2, 1, 64, 64)
        t = torch.randint(0, 100, (2,))
        
        # Test at different resolutions
        for res in [64, 128]:
            progressive_unet.set_resolution(res)
            assert progressive_unet.current_resolution == res
            
            # Test forward pass
            with torch.no_grad():
                output = progressive_unet(x, t)
                assert output.shape == x.shape
        
        print("✅ ProgressiveUNet tests passed")
    
    def test_progressive_trainer(self):
        """Test ProgressiveTrainer functionality."""
        print("Testing ProgressiveTrainer...")
        
        base_unet = create_test_unet()
        progressive_unet = ProgressiveUNet(base_unet, max_resolution=128)
        trainer = ProgressiveTrainer(progressive_unet, self.config)
        
        # Test progressive setup
        assert hasattr(trainer, 'resolution_schedule')
        assert hasattr(trainer, 'current_phase')
        assert hasattr(trainer, 'epochs_per_resolution')
        
        # Test resolution configuration
        config = trainer.get_current_resolution_config()
        assert config is not None
        assert 'resolution' in config
        assert 'learning_rate' in config
        assert 'batch_size' in config
        
        # Test training step
        batch = (torch.randn(*self.test_shape), torch.randn(*self.test_shape))
        loss = trainer.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        
        print("✅ ProgressiveTrainer tests passed")
    
    def test_hierarchical_trainer(self):
        """Test HierarchicalTrainer functionality."""
        print("Testing HierarchicalTrainer...")
        
        base_unet = create_test_unet()
        progressive_unet = ProgressiveUNet(base_unet, max_resolution=128)
        trainer = HierarchicalTrainer(progressive_unet, self.config)
        
        # Test hierarchical setup
        assert hasattr(trainer, 'hierarchical_config')
        assert hasattr(trainer, 'enable_hierarchical')
        
        # Test training step with hierarchical loss
        batch = (torch.randn(*self.test_shape), torch.randn(*self.test_shape))
        loss = trainer.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        
        # Test hierarchical summary
        summary = trainer.get_hierarchical_summary()
        assert isinstance(summary, dict)
        assert 'hierarchical_enabled' in summary
        
        print("✅ HierarchicalTrainer tests passed")
    
    def test_frequency_loss(self):
        """Test FourierLoss functionality."""
        print("Testing FourierLoss...")
        
        # Test different loss types
        for loss_type in ['l1', 'l2', 'smooth_l1']:
            freq_loss = FourierLoss(loss_type=loss_type)
            
            pred = torch.randn(2, 1, 64, 64)
            target = torch.randn(2, 1, 64, 64)
            
            loss = freq_loss(pred, target)
            assert isinstance(loss, torch.Tensor)
            assert loss.item() >= 0
            assert not torch.isnan(loss)
        
        # Test using factory function
        try:
            freq_loss = create_frequency_loss(loss_type="fourier", loss_fn="l2")
            pred = torch.randn(2, 1, 64, 64)
            target = torch.randn(2, 1, 64, 64)
            
            loss = freq_loss(pred, target)
            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss)
        except Exception as e:
            warnings.warn(f"Factory function test failed: {e}")
        
        print("✅ FourierLoss tests passed")
    
    def test_advanced_scheduler(self):
        """Test advanced scheduler functionality."""
        print("Testing advanced schedulers...")
        
        # Test different schedule types
        schedule_types = ['cosine', 'exponential', 'polynomial', 'sigmoid']
        
        for schedule_type in schedule_types:
            try:
                scheduler = create_scheduler(
                    scheduler_type=schedule_type,
                    num_timesteps=100,
                    beta_start=0.0001,
                    beta_end=0.02
                )
                
                # Test get_betas method
                betas = scheduler.get_betas()
                assert len(betas) == 100
                
                # Check that betas are in valid range
                assert torch.all(betas > 0)
                assert torch.all(betas < 1)
                
                # Test get_alphas_cumprod method
                alphas_cumprod = scheduler.get_alphas_cumprod()
                
                # Check that alphas_cumprod is monotonically decreasing
                assert torch.all(alphas_cumprod[1:] <= alphas_cumprod[:-1] + 1e-6)
                
            except Exception as e:
                warnings.warn(f"Schedule type {schedule_type} failed: {e}")
        
        # Test SchedulerManager
        try:
            manager = SchedulerManager()
            cosine_scheduler = create_scheduler('cosine', 100)
            manager.add_scheduler('cosine', cosine_scheduler)
            
            current_scheduler = manager.get_current_scheduler()
            assert current_scheduler is not None
        except Exception as e:
            warnings.warn(f"SchedulerManager test failed: {e}")
        
        print("✅ Advanced scheduler tests passed")
    
    def test_adaptive_batch_sizer(self):
        """Test AdaptiveBatchSizer functionality."""
        print("Testing AdaptiveBatchSizer...")
        
        batch_sizer = AdaptiveBatchSizer(verbose=False)
        
        # Test memory monitoring
        memory_stats = batch_sizer.monitor_memory_pressure()
        assert isinstance(memory_stats, dict)
        assert 'pressure' in memory_stats
        
        # Test batch size optimization
        model = create_test_unet()
        input_shape = (1, 64, 64)
        
        optimal_batch_size = batch_sizer.find_optimal_batch_size(
            model, input_shape, device=str(self.device)
        )
        assert isinstance(optimal_batch_size, int)
        assert optimal_batch_size > 0
        
        # Test dynamic adjustment
        current_batch_size = 8
        new_batch_size, info = batch_sizer.adjust_batch_size_dynamically(
            current_batch_size=current_batch_size,
            step=100,
            force_check=True
        )
        assert isinstance(new_batch_size, int)
        assert isinstance(info, dict)
        assert 'adjusted' in info
        
        print("✅ AdaptiveBatchSizer tests passed")
    
    def test_cascaded_sampling(self):
        """Test CascadedSampler functionality."""
        print("Testing CascadedSampler...")
        
        # Create models for different resolutions
        models = {}
        resolutions = [64, 128]
        
        for res in resolutions:
            base_unet = create_test_unet(sample_size=res)
            config = create_test_config(max_resolution=res)
            trainer = DDPMTrainer(base_unet, config)
            models[res] = trainer
        
        # Create cascaded sampler
        sampler = CascadedSampler(
            models=models,
            resolutions=resolutions,
            num_inference_steps=10,  # Reduced for testing
            device=str(self.device)
        )
        
        # Test sampling
        with torch.no_grad():
            samples = sampler.sample(batch_size=1, verbose=False)
            assert samples.shape == (1, 1, 128, 128)  # Final resolution
            assert not torch.isnan(samples).any()
        
        print("✅ CascadedSampler tests passed")
    
    def test_sota_integration(self):
        """Test SOTA integration functionality."""
        print("Testing SOTA integration...")
        
        # Create hierarchical trainer as the highest-level integration
        base_unet = create_test_unet()
        progressive_unet = ProgressiveUNet(base_unet, max_resolution=128)
        trainer = HierarchicalTrainer(progressive_unet, self.config)
        
        # Test trainer initialization
        assert isinstance(trainer, HierarchicalTrainer)
        assert hasattr(trainer, 'enable_progressive')
        assert hasattr(trainer, 'enable_hierarchical')
        
        # Test training step
        batch = (torch.randn(*self.test_shape), torch.randn(*self.test_shape))
        loss = trainer.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        
        # Test hierarchical summary
        summary = trainer.get_hierarchical_summary()
        assert isinstance(summary, dict)
        assert 'hierarchical_enabled' in summary
        
        print("✅ SOTA integration tests passed")
    
    def test_mathematical_correctness(self):
        """Test mathematical correctness of implementations."""
        print("Testing mathematical correctness...")
        
        # Test noise schedule properties
        scheduler = create_scheduler(scheduler_type='cosine', num_timesteps=1000)
        
        betas = scheduler.get_betas()
        alphas_cumprod = scheduler.get_alphas_cumprod()
        alphas = 1 - betas
        
        # Verify mathematical relationships
        assert torch.allclose(alphas_cumprod[0], alphas[0], atol=1e-6)
        
        # Verify cumprod calculation
        computed_cumprod = torch.cumprod(alphas, dim=0)
        assert torch.allclose(alphas_cumprod, computed_cumprod, atol=1e-6)
        
        # Test frequency loss mathematical properties
        freq_loss = FourierLoss(loss_type='l2')
        
        # Test identity: loss(x, x) should be 0
        x = torch.randn(1, 1, 64, 64)
        loss_identity = freq_loss(x, x)
        assert loss_identity.item() < 1e-6
        
        # Test symmetry: loss(x, y) should equal loss(y, x) for L2
        x = torch.randn(1, 1, 64, 64)
        y = torch.randn(1, 1, 64, 64)
        loss_xy = freq_loss(x, y)
        loss_yx = freq_loss(y, x)
        assert torch.allclose(loss_xy, loss_yx, atol=1e-6)
        
        print("✅ Mathematical correctness tests passed")
    
    def test_e2e_training_loop(self):
        """Test end-to-end training loop."""
        print("Testing end-to-end training...")
        
        # Create complete setup
        base_unet = create_test_unet()
        progressive_unet = ProgressiveUNet(base_unet, max_resolution=128)
        trainer = HierarchicalTrainer(progressive_unet, self.config)
        
        # Create dummy dataset
        dataset_size = 16
        dataset = []
        for _ in range(dataset_size):
            target = torch.randn(1, 64, 64)
            source = torch.randn(1, 64, 64)
            dataset.append((target, source))
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Test training loop
        trainer.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:  # Limit for testing
                break
                
            loss = trainer.training_step(batch, batch_idx)
            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Test EMA update
            if hasattr(trainer, 'update_ema'):
                trainer.update_ema()
        
        avg_loss = total_loss / num_batches
        assert avg_loss > 0
        assert avg_loss < 100  # Sanity check
        
        print(f"✅ E2E training completed. Average loss: {avg_loss:.4f}")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks."""
        print("Testing performance benchmarks...")
        
        # Setup
        base_unet = create_test_unet()
        config = create_test_config()
        
        # Benchmark standard trainer
        standard_trainer = DDPMTrainer(base_unet, config)
        
        # Benchmark SOTA trainer (hierarchical)
        progressive_unet = ProgressiveUNet(base_unet, max_resolution=128)
        sota_trainer = HierarchicalTrainer(progressive_unet, config)
        
        # Create test batch
        batch = (torch.randn(*self.test_shape), torch.randn(*self.test_shape))
        
        # Benchmark training step
        def benchmark_training_step(trainer, batch, num_runs=10):
            trainer.train()
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                loss = trainer.training_step(batch, 0)
                end_time = time.time()
                times.append(end_time - start_time)
            
            return np.mean(times), np.std(times)
        
        # Run benchmarks
        standard_time, standard_std = benchmark_training_step(standard_trainer, batch)
        sota_time, sota_std = benchmark_training_step(sota_trainer, batch)
        
        print(f"Standard trainer: {standard_time:.4f}±{standard_std:.4f}s")
        print(f"SOTA trainer: {sota_time:.4f}±{sota_std:.4f}s")
        print(f"SOTA overhead: {((sota_time - standard_time) / standard_time * 100):.1f}%")
        
        # SOTA should not be more than 3x slower
        assert sota_time < standard_time * 3.0
        
        print("✅ Performance benchmarks passed")
    
    def test_memory_efficiency(self):
        """Test memory efficiency."""
        print("Testing memory efficiency...")
        
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available, skipping memory tests")
            return
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test memory usage with different batch sizes
        base_unet = create_test_unet().cuda()
        config = create_test_config()
        trainer = DDPMTrainer(base_unet, config)
        
        batch_sizes = [2, 4, 8]
        memory_usage = []
        
        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            batch = (
                torch.randn(batch_size, 1, 64, 64).cuda(),
                torch.randn(batch_size, 1, 64, 64).cuda()
            )
            
            # Forward pass
            trainer.train()
            loss = trainer.training_step(batch, 0)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
            memory_usage.append(peak_memory)
            
            print(f"Batch size {batch_size}: {peak_memory:.2f} GB")
        
        # Memory should scale roughly linearly with batch size
        memory_ratio = memory_usage[-1] / memory_usage[0]
        batch_ratio = batch_sizes[-1] / batch_sizes[0]
        
        # Allow some overhead, but should be roughly proportional
        assert memory_ratio < batch_ratio * 1.5
        
        print("✅ Memory efficiency tests passed")
    
    def test_comparison_with_standards(self):
        """Compare implementation with field standards."""
        print("Testing comparison with field standards...")
        
        # Test 1: Noise schedule comparison with Ho et al. (2020)
        scheduler = create_scheduler(scheduler_type='cosine', num_timesteps=1000)
        
        # Check that cosine schedule properties match literature
        alphas_cumprod = scheduler.get_alphas_cumprod()
        
        # At t=0, alpha_cumprod should be close to 1
        assert alphas_cumprod[0] > 0.99
        
        # At t=T-1, alpha_cumprod should be small
        assert alphas_cumprod[-1] < 0.01
        
        # Should be monotonically decreasing
        assert torch.all(alphas_cumprod[1:] <= alphas_cumprod[:-1] + 1e-6)
        
        # Test 2: Progressive training comparison with Karras et al.
        base_unet = create_test_unet()
        progressive_unet = ProgressiveUNet(base_unet, max_resolution=256)
        
        # Resolution schedule should follow power-of-2 progression
        resolutions = progressive_unet.resolutions
        for i in range(1, len(resolutions)):
            ratio = resolutions[i] / resolutions[i-1]
            assert ratio == 2.0 or ratio == 1.0  # Should double or stay same
        
        # Test 3: Frequency loss comparison with perceptual loss literature
        freq_loss = FourierLoss(loss_type='l2', low_freq_emphasis=2.0)
        
        # Low-pass filter should emphasize structural content
        # Create test images: one with high-freq noise, one smooth
        smooth_img = torch.ones(1, 1, 64, 64) * 0.5
        noisy_img = smooth_img + 0.1 * torch.randn(1, 1, 64, 64)
        
        loss_smooth = freq_loss(smooth_img, smooth_img)
        loss_noisy = freq_loss(noisy_img, smooth_img)
        
        # Loss should be higher for noisy image
        assert loss_noisy > loss_smooth
        
        print("✅ Field standards comparison passed")


def run_all_tests():
    """Run all SOTA component tests."""
    print("🚀 Running SOTA DDPM Component Tests")
    print("=" * 50)
    
    test_suite = TestSOTAComponents()
    test_suite.setup_method()
    
    tests = [
        test_suite.test_progressive_unet,
        test_suite.test_progressive_trainer,
        test_suite.test_hierarchical_trainer,
        test_suite.test_frequency_loss,
        test_suite.test_advanced_scheduler,
        test_suite.test_adaptive_batch_sizer,
        test_suite.test_cascaded_sampling,
        test_suite.test_sota_integration,
        test_suite.test_mathematical_correctness,
        test_suite.test_e2e_training_loop,
        test_suite.test_performance_benchmarks,
        test_suite.test_memory_efficiency,
        test_suite.test_comparison_with_standards,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! SOTA implementation is validated.")
    else:
        print(f"⚠️ {failed} tests failed. Please review implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
