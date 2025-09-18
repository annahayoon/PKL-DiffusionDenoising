"""
Dual-Objective DDPM Training Strategy
=====================================

Optimized approach for:
1. Spatial Resolution (sharpness) enhancement
2. Pixel Intensity (signal mapping) prediction

Given data constraints:
- WF: 18.8% dynamic range utilization (good for spatial)
- 2P: 5.4% dynamic range utilization (challenging for intensity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from pkl_dg.data.adaptive_dataset import create_adaptive_datasets


class DualObjectiveLoss(nn.Module):
    """
    Combined loss function for spatial resolution and intensity mapping.
    
    Components:
    1. Diffusion Loss: Standard DDPM loss for spatial structure
    2. Intensity Loss: MSE loss for pixel-wise intensity mapping
    3. Perceptual Loss: VGG-based loss for spatial quality
    4. Gradient Loss: Preserves edge sharpness
    """
    
    def __init__(
        self,
        alpha_diffusion: float = 1.0,
        beta_intensity: float = 0.5,
        gamma_perceptual: float = 0.3,
        delta_gradient: float = 0.2,
        intensity_weight_schedule: str = "adaptive"
    ):
        super().__init__()
        self.alpha_diffusion = alpha_diffusion
        self.beta_intensity = beta_intensity
        self.gamma_perceptual = gamma_perceptual
        self.delta_gradient = delta_gradient
        self.intensity_weight_schedule = intensity_weight_schedule
        
        # Initialize perceptual loss (simplified - would use actual VGG in practice)
        self.perceptual_loss = nn.MSELoss()
        self.intensity_loss = nn.MSELoss()
        self.gradient_loss = GradientLoss()
        
    def forward(
        self, 
        predicted: torch.Tensor,
        target: torch.Tensor,
        diffusion_loss: torch.Tensor,
        step: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for dual objectives.
        
        Args:
            predicted: Model prediction
            target: Ground truth 2P image
            diffusion_loss: Standard DDPM loss
            step: Training step (for adaptive weighting)
        """
        
        # 1. Diffusion loss (spatial structure)
        loss_diffusion = diffusion_loss
        
        # 2. Intensity mapping loss (pixel-wise accuracy)
        loss_intensity = self.intensity_loss(predicted, target)
        
        # 3. Perceptual loss (spatial quality)
        loss_perceptual = self.perceptual_loss(predicted, target)
        
        # 4. Gradient loss (edge preservation)
        loss_gradient = self.gradient_loss(predicted, target)
        
        # Adaptive weighting for intensity loss
        if self.intensity_weight_schedule == "adaptive":
            # Increase intensity weight as training progresses
            intensity_weight = self.beta_intensity * min(1.0, step / 10000)
        else:
            intensity_weight = self.beta_intensity
        
        # Combined loss
        total_loss = (
            self.alpha_diffusion * loss_diffusion +
            intensity_weight * loss_intensity +
            self.gamma_perceptual * loss_perceptual +
            self.delta_gradient * loss_gradient
        )
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': loss_diffusion,
            'intensity_loss': loss_intensity,
            'perceptual_loss': loss_perceptual,
            'gradient_loss': loss_gradient,
            'intensity_weight': intensity_weight
        }


# Import GradientLoss from the main losses module to avoid duplication
from pkl_dg.models.losses import GradientLoss


class IntensityAugmentation:
    """
    Data augmentation specifically designed to improve intensity mapping
    with limited 2P dynamic range.
    """
    
    def __init__(
        self,
        intensity_scale_range: Tuple[float, float] = (0.95, 1.05),
        noise_std: float = 0.01,
        contrast_range: Tuple[float, float] = (0.98, 1.02)
    ):
        self.intensity_scale_range = intensity_scale_range
        self.noise_std = noise_std
        self.contrast_range = contrast_range
        
    def __call__(self, wf: torch.Tensor, tp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply intensity-focused augmentations."""
        
        # 1. Intensity scaling (subtle to preserve signal mapping)
        scale = torch.uniform(*self.intensity_scale_range)
        tp_aug = tp * scale
        
        # 2. Add small amount of noise to increase variation
        if self.noise_std > 0:
            noise = torch.randn_like(tp) * self.noise_std
            tp_aug = tp_aug + noise
        
        # 3. Contrast adjustment (very subtle)
        contrast = torch.uniform(*self.contrast_range)
        tp_mean = tp_aug.mean()
        tp_aug = (tp_aug - tp_mean) * contrast + tp_mean
        
        # 4. Clamp to valid range
        tp_aug = torch.clamp(tp_aug, -1, 1)
        
        return wf, tp_aug


def create_dual_objective_training_config():
    """Create optimized training configuration for dual objectives."""
    
    config = {
        # Model configuration
        'model': {
            'unet_channels': [64, 128, 256, 512],  # Sufficient capacity
            'attention_resolutions': [16, 8],      # Focus on fine details
            'num_res_blocks': 2,
            'dropout': 0.1
        },
        
        # Training configuration
        'training': {
            'batch_size': 8,                       # Stable for limited range data
            'learning_rate': 1e-4,                 # Conservative for intensity mapping
            'lr_schedule': 'cosine_with_restarts', # Helps escape local minima
            'max_epochs': 200,                     # Longer training for limited data
            'warmup_steps': 1000,                  # Gradual warmup
            'gradient_clip_val': 1.0,              # Stability with limited range
        },
        
        # Loss configuration
        'loss': {
            'alpha_diffusion': 1.0,    # Standard diffusion loss
            'beta_intensity': 0.8,     # High weight for intensity mapping
            'gamma_perceptual': 0.3,   # Spatial quality
            'delta_gradient': 0.4,     # Edge preservation (important for sharpness)
            'intensity_weight_schedule': 'adaptive'
        },
        
        # Data augmentation
        'augmentation': {
            'use_intensity_aug': True,
            'intensity_scale_range': (0.98, 1.02),  # Subtle for signal preservation
            'noise_std': 0.005,                     # Small noise injection
            'contrast_range': (0.99, 1.01)         # Very subtle contrast changes
        },
        
        # Optimization strategies
        'optimization': {
            'use_ema': True,                       # Exponential moving average
            'ema_decay': 0.9999,
            'use_progressive_training': True,      # Start low-res, increase gradually
            'progressive_schedule': [64, 96, 128], # Resolution progression
            'curriculum_learning': True,           # Easy samples first
        },
        
        # Monitoring and validation
        'validation': {
            'val_every_n_epochs': 5,
            'metrics': [
                'mse_loss',           # Intensity accuracy
                'ssim',               # Spatial quality
                'psnr',               # Overall quality
                'gradient_magnitude', # Sharpness measure
                'intensity_histogram_distance'  # Signal mapping accuracy
            ]
        }
    }
    
    return config


def progressive_training_strategy():
    """
    Progressive training strategy optimized for your dual objectives.
    """
    
    strategy = {
        'phase_1': {
            'name': 'Spatial Structure Learning',
            'resolution': 64,
            'epochs': 50,
            'focus': 'Learn basic spatial structures with high diffusion weight',
            'loss_weights': {
                'alpha_diffusion': 1.0,
                'beta_intensity': 0.2,
                'gamma_perceptual': 0.5,
                'delta_gradient': 0.1
            }
        },
        
        'phase_2': {
            'name': 'Intensity Mapping Integration',
            'resolution': 96,
            'epochs': 75,
            'focus': 'Balance spatial and intensity learning',
            'loss_weights': {
                'alpha_diffusion': 1.0,
                'beta_intensity': 0.6,
                'gamma_perceptual': 0.4,
                'delta_gradient': 0.3
            }
        },
        
        'phase_3': {
            'name': 'Fine-tuning and Sharpness',
            'resolution': 128,
            'epochs': 75,
            'focus': 'Maximize sharpness and intensity accuracy',
            'loss_weights': {
                'alpha_diffusion': 1.0,
                'beta_intensity': 0.8,
                'gamma_perceptual': 0.3,
                'delta_gradient': 0.5
            }
        }
    }
    
    return strategy


def main():
    """Example usage of the dual-objective training approach."""
    
    print("🚀 Dual-Objective DDPM Training Strategy")
    print("=" * 50)
    
    # Get training configuration
    config = create_dual_objective_training_config()
    print("✅ Training configuration created")
    
    # Get progressive training strategy
    strategy = progressive_training_strategy()
    print("✅ Progressive training strategy defined")
    
    # Create datasets with optimal normalization
    datasets = create_adaptive_datasets(
        data_dir="data/real_microscopy",
        batch_size=config['training']['batch_size'],
        percentiles=(0.0, 100.0)  # Maximum range utilization
    )
    print("✅ Datasets created with adaptive normalization")
    
    # Print strategy summary
    print("\n📊 TRAINING STRATEGY SUMMARY:")
    print(f"  • Dual-objective loss with 4 components")
    print(f"  • Progressive training: 64→96→128 resolution")
    print(f"  • Intensity-aware augmentation")
    print(f"  • Adaptive loss weighting")
    print(f"  • Focus on both spatial sharpness AND intensity mapping")
    
    print("\n🎯 EXPECTED OUTCOMES:")
    print("  ✅ Spatial Resolution: Excellent (DDPM strength)")
    print("  ✅ Intensity Mapping: Good (with specialized losses)")
    print("  ✅ Convergence: Stable with progressive approach")
    print("  ✅ Signal Preservation: Enhanced with dual-loss design")
    
    return config, strategy, datasets


if __name__ == "__main__":
    config, strategy, datasets = main()
