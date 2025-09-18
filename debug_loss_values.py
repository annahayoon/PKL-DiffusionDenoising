#!/usr/bin/env python3
"""
Debug script to analyze loss component values and identify the source of high training loss.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pkl_dg.models.dual_objective_loss import DualObjectiveLoss
from pkl_dg.utils.utils_16bit import normalize_16bit_to_model_input

def analyze_loss_components():
    """Analyze individual loss components to identify the source of high loss."""
    
    print("🔍 Debugging Loss Components")
    print("=" * 50)
    
    # Create sample data similar to your training
    batch_size = 8
    height, width = 128, 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulate properly normalized data [-1, 1]
    print("1. Testing with properly normalized data [-1, 1]:")
    pred_x0_norm = torch.randn(batch_size, 1, height, width, device=device) * 0.5  # Range ~[-1.5, 1.5]
    target_x0_norm = torch.randn(batch_size, 1, height, width, device=device) * 0.5
    diffusion_loss_norm = torch.randn(1, device=device).abs()  # Typical diffusion loss ~0.5-2.0
    
    # Test dual objective loss with normalized data
    dual_loss = DualObjectiveLoss(
        alpha_diffusion=1.0,
        beta_intensity=0.6,
        delta_gradient=0.4,
        use_adaptive_weighting=True,
        warmup_steps=1000
    ).to(device)
    
    loss_components_norm = dual_loss(
        diffusion_loss=diffusion_loss_norm,
        pred_x0=pred_x0_norm,
        target_x0=target_x0_norm,
        step=0  # Initial step
    )
    
    print(f"  Diffusion Loss: {loss_components_norm['diffusion_loss'].item():.4f}")
    print(f"  Intensity Loss: {loss_components_norm['intensity_loss'].item():.4f}")
    print(f"  Gradient Loss: {loss_components_norm['gradient_loss'].item():.4f}")
    print(f"  Total Loss: {loss_components_norm['total_loss'].item():.4f}")
    print(f"  Adaptive Weights: {loss_components_norm['weights']}")
    print()
    
    # Simulate improperly normalized data [0, 65535] (16-bit range)
    print("2. Testing with 16-bit range data [0, 65535] (WRONG):")
    pred_x0_16bit = torch.randint(0, 65535, (batch_size, 1, height, width), device=device, dtype=torch.float32)
    target_x0_16bit = torch.randint(0, 65535, (batch_size, 1, height, width), device=device, dtype=torch.float32)
    
    loss_components_16bit = dual_loss(
        diffusion_loss=diffusion_loss_norm,
        pred_x0=pred_x0_16bit,
        target_x0=target_x0_16bit,
        step=0
    )
    
    print(f"  Diffusion Loss: {loss_components_16bit['diffusion_loss'].item():.4f}")
    print(f"  Intensity Loss: {loss_components_16bit['intensity_loss'].item():.4f}")
    print(f"  Gradient Loss: {loss_components_16bit['gradient_loss'].item():.4f}")
    print(f"  Total Loss: {loss_components_16bit['total_loss'].item():.4f}")
    print()
    
    # Test normalization function
    print("3. Testing 16-bit normalization function:")
    sample_16bit = torch.tensor([0, 32767, 65535], dtype=torch.float32)
    sample_normalized = normalize_16bit_to_model_input(sample_16bit)
    print(f"  16-bit values: {sample_16bit.tolist()}")
    print(f"  Normalized values: {sample_normalized.tolist()}")
    print(f"  Expected: [-1.0, 0.0, 1.0]")
    print()
    
    # Calculate expected loss ranges
    print("4. Expected loss ranges for your configuration:")
    print("  At step 0 (adaptive weighting):")
    print("    α=1.0 (diffusion), β=0.0→0.6 (intensity), γ=0.0→0.2 (perceptual), δ=0.0→0.4 (gradient)")
    print("  Expected total loss: 1.0-2.5 (with proper normalization)")
    print("  Your actual loss: 41.00 (16-34x too high!)")
    print()
    
    print("🎯 DIAGNOSIS:")
    if loss_components_16bit['total_loss'].item() > 10 * loss_components_norm['total_loss'].item():
        print("  ❌ LIKELY ISSUE: Data not properly normalized!")
        print("  ❌ Images may be in [0, 65535] range instead of [-1, 1]")
        print("  ✅ SOLUTION: Verify use_16bit_normalization is working in dataset")
    else:
        print("  ✅ Normalization seems OK, investigating other causes...")
    
    return loss_components_norm, loss_components_16bit

def check_data_loading():
    """Check if data loading is applying normalization correctly."""
    print("\n🔍 Checking Data Loading")
    print("=" * 50)
    
    try:
        from pkl_dg.data.dataset import MicroscopyDataset
        from omegaconf import OmegaConf
        
        # Load your config
        config_path = project_root / "configs" / "config_microscopy.yaml"
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            
            print(f"Config use_16bit_normalization: {cfg.data.get('use_16bit_normalization', 'NOT SET')}")
            print(f"Config max_intensity: {cfg.data.get('max_intensity', 'NOT SET')}")
            print(f"Config min_intensity: {cfg.data.get('min_intensity', 'NOT SET')}")
            
            # Try to create dataset
            data_dir = cfg.paths.data
            if Path(data_dir).exists():
                print(f"Data directory exists: {data_dir}")
                
                # Create small test dataset
                dataset = MicroscopyDataset(
                    data_dir=data_dir,
                    image_size=cfg.data.image_size,
                    use_16bit_normalization=cfg.data.use_16bit_normalization,
                    max_samples=1  # Just one sample for testing
                )
                
                if len(dataset) > 0:
                    sample = dataset[0]
                    tp_image, wf_image = sample
                    
                    print(f"Sample shape: {tp_image.shape}")
                    print(f"TP image range: [{tp_image.min():.4f}, {tp_image.max():.4f}]")
                    print(f"WF image range: [{wf_image.min():.4f}, {wf_image.max():.4f}]")
                    
                    if tp_image.min() >= -1.1 and tp_image.max() <= 1.1:
                        print("  ✅ Data appears properly normalized to [-1, 1]")
                    else:
                        print("  ❌ Data NOT properly normalized!")
                        print("  ❌ This is likely causing your high loss values")
                else:
                    print("  ⚠️ Dataset is empty")
            else:
                print(f"  ❌ Data directory not found: {data_dir}")
        else:
            print(f"  ❌ Config file not found: {config_path}")
            
    except Exception as e:
        print(f"  ❌ Error checking data loading: {e}")

if __name__ == "__main__":
    print("🚀 PKL Diffusion Loss Debugging Tool")
    print("=" * 60)
    
    # Analyze loss components
    norm_losses, bad_losses = analyze_loss_components()
    
    # Check data loading
    check_data_loading()
    
    print("\n📋 SUMMARY:")
    print("=" * 50)
    print("1. Check if your data is properly normalized to [-1, 1]")
    print("2. Verify use_16bit_normalization=True is working in dataset")
    print("3. If data is normalized but loss is still high, check individual components")
    print("4. VGG perceptual loss has been removed from the repository")
    print("\n💡 Next steps:")
    print("   Run: python debug_loss_values.py")
    print("   Then check individual loss component values in your training logs")
