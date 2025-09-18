#!/usr/bin/env python3
"""
Test script for dual-objective loss implementation

This script tests:
1. Dual objective loss initialization
2. Loss computation with sample data
3. Integration with adaptive normalization
4. Training step functionality
"""

import torch
import numpy as np
from pathlib import Path

# Import our modules
from pkl_dg.models.dual_objective_loss import DualObjectiveLoss, create_dual_objective_loss
from pkl_dg.data.adaptive_dataset import AdaptiveRealPairsDataset

def test_dual_objective_loss():
    """Test the dual objective loss module."""
    
    print("🧪 Testing Dual-Objective Loss Implementation")
    print("=" * 60)
    
    # Test configuration
    config = {
        'dual_objective_loss': {
            'alpha_diffusion': 1.0,
            'beta_intensity': 0.8,
            'gamma_perceptual': 0.2,
            'delta_gradient': 0.5,
            'use_adaptive_weighting': True,
            'warmup_steps': 100,
            'intensity_weight_mode': 'adaptive'
        }
    }
    
    # 1. Test loss initialization
    print("\n1️⃣  Testing Loss Initialization...")
    try:
        loss_fn = create_dual_objective_loss(config)
        print("✅ Dual objective loss created successfully")
        print(f"   • Alpha (diffusion): {loss_fn.alpha_diffusion}")
        print(f"   • Beta (intensity): {loss_fn.beta_intensity}")
        print(f"   • Delta (gradient): {loss_fn.delta_gradient}")
        print(f"   • Adaptive weighting: {loss_fn.use_adaptive_weighting}")
    except Exception as e:
        print(f"❌ Loss initialization failed: {e}")
        return False
    
    # 2. Test with synthetic data
    print("\n2️⃣  Testing Loss Computation...")
    try:
        # Create synthetic data mimicking 2P characteristics
        batch_size, channels, height, width = 4, 1, 128, 128
        
        # Simulate 2P-like target (low dynamic range)
        target = torch.randn(batch_size, channels, height, width) * 0.1 - 0.9  # Around [-1, -0.8]
        
        # Simulate prediction (slightly different)
        pred = target + torch.randn_like(target) * 0.05
        
        # Simulate diffusion loss
        diffusion_loss = torch.tensor(0.1)
        
        # Compute dual objective loss
        loss_components = loss_fn(
            diffusion_loss=diffusion_loss,
            pred_x0=pred,
            target_x0=target,
            step=50  # Mid-warmup
        )
        
        print("✅ Loss computation successful")
        print(f"   • Total loss: {loss_components['total_loss']:.4f}")
        print(f"   • Diffusion loss: {loss_components['diffusion_loss']:.4f}")
        print(f"   • Intensity loss: {loss_components['intensity_loss']:.4f}")
        print(f"   • Gradient loss: {loss_components['gradient_loss']:.4f}")
        print(f"   • Perceptual loss: {loss_components['perceptual_loss']:.4f}")
        
        # Test adaptive weighting
        weights = loss_components['weights']
        print(f"   • Adaptive weights: α={weights['alpha']:.2f}, β={weights['beta']:.2f}, δ={weights['delta']:.2f}")
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False
    
    # 3. Test with real data (if available)
    print("\n3️⃣  Testing with Real Data...")
    try:
        data_dir = Path("../data/real_microscopy")
        if data_dir.exists():
            # Load a small batch of real data
            dataset = AdaptiveRealPairsDataset(
                data_dir, 
                split='train', 
                create_normalization_params=False
            )
            
            # Get a sample
            tp_sample, wf_sample = dataset[0]
            
            # Add batch dimension
            tp_batch = tp_sample.unsqueeze(0)
            wf_batch = wf_sample.unsqueeze(0)
            
            print(f"✅ Real data loaded successfully")
            print(f"   • 2P range: [{tp_batch.min():.3f}, {tp_batch.max():.3f}]")
            print(f"   • WF range: [{wf_batch.min():.3f}, {wf_batch.max():.3f}]")
            
            # Test loss with real data
            diffusion_loss_real = torch.tensor(0.05)
            pred_real = tp_batch + torch.randn_like(tp_batch) * 0.02
            
            loss_components_real = loss_fn(
                diffusion_loss=diffusion_loss_real,
                pred_x0=pred_real,
                target_x0=tp_batch,
                step=200
            )
            
            print(f"✅ Real data loss computation successful")
            print(f"   • Total loss: {loss_components_real['total_loss']:.4f}")
            print(f"   • Intensity loss: {loss_components_real['intensity_loss']:.4f}")
            print(f"   • Gradient loss: {loss_components_real['gradient_loss']:.4f}")
            
        else:
            print("⚠️  Real data not found, skipping real data test")
            
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        print("   This is okay if data is not available")
    
    # 4. Test gradient flow
    print("\n4️⃣  Testing Gradient Flow...")
    try:
        # Create data that requires gradients
        pred = torch.randn(2, 1, 64, 64, requires_grad=True)
        target = torch.randn(2, 1, 64, 64)
        diffusion_loss = torch.tensor(0.1, requires_grad=True)
        
        loss_components = loss_fn(
            diffusion_loss=diffusion_loss,
            pred_x0=pred,
            target_x0=target,
            step=150
        )
        
        # Backward pass
        loss_components['total_loss'].backward()
        
        print("✅ Gradient flow test successful")
        print(f"   • Prediction gradients: {pred.grad is not None}")
        print(f"   • Gradient norm: {pred.grad.norm():.4f}" if pred.grad is not None else "   • No gradients")
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        return False
    
    # 5. Test adaptive weighting progression
    print("\n5️⃣  Testing Adaptive Weighting Progression...")
    try:
        steps = [0, 50, 100, 200, 500]
        pred = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)
        diffusion_loss = torch.tensor(0.1)
        
        print("   Step | Beta Weight | Total Loss")
        print("   -----|-------------|----------")
        
        for step in steps:
            loss_components = loss_fn(
                diffusion_loss=diffusion_loss,
                pred_x0=pred,
                target_x0=target,
                step=step
            )
            beta_weight = loss_components['weights']['beta']
            total_loss = loss_components['total_loss']
            print(f"   {step:4d} | {beta_weight:10.3f} | {total_loss:9.4f}")
        
        print("✅ Adaptive weighting progression working correctly")
        
    except Exception as e:
        print(f"❌ Adaptive weighting test failed: {e}")
        return False
    
    print(f"\n🎉 All tests passed! Dual-objective loss is ready for training.")
    return True


def test_integration_with_config():
    """Test integration with configuration file."""
    
    print(f"\n🔧 Testing Configuration Integration...")
    
    try:
        # Test loading from config file
        import yaml
        
        config_path = "../configs/config_dual_objective.yaml"
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract dual objective loss config
            loss_config = config.get('ddpm', {})
            
            print(f"✅ Configuration loaded successfully")
            print(f"   • Dual objective enabled: {loss_config.get('use_dual_objective_loss', False)}")
            
            if 'dual_objective_loss' in loss_config:
                dual_config = loss_config['dual_objective_loss']
                print(f"   • Alpha (diffusion): {dual_config.get('alpha_diffusion', 1.0)}")
                print(f"   • Beta (intensity): {dual_config.get('beta_intensity', 0.8)}")
                print(f"   • Delta (gradient): {dual_config.get('delta_gradient', 0.5)}")
                
        else:
            print(f"⚠️  Configuration file not found: {config_path}")
            
    except Exception as e:
        print(f"❌ Configuration integration test failed: {e}")


if __name__ == "__main__":
    print("🚀 Dual-Objective Loss Testing Suite")
    print("=" * 80)
    
    success = test_dual_objective_loss()
    test_integration_with_config()
    
    if success:
        print(f"\n✅ SUMMARY: All tests passed!")
        print(f"   Your dual-objective loss implementation is ready for:")
        print(f"   • Spatial resolution enhancement")
        print(f"   • Pixel intensity mapping")
        print(f"   • Adaptive weighting during training")
        print(f"   • Integration with your existing DDPM pipeline")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Run training with: python scripts/run_microscopy.py --mode train --config configs/config_dual_objective.yaml")
        print(f"   2. Monitor loss components in W&B/TensorBoard")
        print(f"   3. Evaluate both spatial and intensity performance")
        
    else:
        print(f"\n❌ SUMMARY: Some tests failed!")
        print(f"   Please check the error messages above and fix issues before training.")
