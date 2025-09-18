"""
Dual-Objective Loss for Spatial Resolution and Pixel Intensity Prediction

This module implements a pure self-supervised dual-objective training system optimized for:
1. Spatial resolution (sharpness) enhancement
2. Pixel intensity (signal mapping) prediction

Components:
- DualObjectiveLoss: Multi-component loss function combining diffusion, intensity, and gradient losses
- IntensityAugmentation: Specialized data augmentation for intensity mapping with limited 2P dynamic range
- Configuration utilities: Functions to create optimized training configurations and progressive training strategies

The loss combines (all self-supervised):
- Diffusion Loss: Standard DDPM loss for spatial structure learning
- Intensity Loss: MSE loss for pixel-wise intensity accuracy
- Gradient Loss: Edge preservation for sharpness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union, Any


class GradientLoss(nn.Module):
    """Gradient loss for preserving edge sharpness and spatial details."""
    
    def __init__(self, loss_type: str = "l1"):
        """
        Args:
            loss_type: Type of loss to use ("l1", "l2", "smooth_l1")
        """
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient loss between prediction and target.
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            
        Returns:
            Gradient loss value
        """
        # Compute gradients using Sobel-like operators
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # Compute loss on gradients
        if self.loss_type == "l1":
            loss_x = F.l1_loss(pred_grad_x, target_grad_x)
            loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        elif self.loss_type == "l2":
            loss_x = F.mse_loss(pred_grad_x, target_grad_x)
            loss_y = F.mse_loss(pred_grad_y, target_grad_y)
        elif self.loss_type == "smooth_l1":
            loss_x = F.smooth_l1_loss(pred_grad_x, target_grad_x)
            loss_y = F.smooth_l1_loss(pred_grad_y, target_grad_y)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss_x + loss_y


class IntensityMappingLoss(nn.Module):
    """Specialized loss for pixel intensity mapping with 2P characteristics."""
    
    def __init__(
        self, 
        loss_type: str = "mse",
        intensity_weight_mode: str = "uniform",
        focus_range: Optional[Tuple[float, float]] = None,
        nll_sigma: Optional[float] = None,
        nll_scale: Optional[float] = None,
    ):
        """
        Args:
            loss_type: Base loss type ("mse", "l1", "smooth_l1")
            intensity_weight_mode: How to weight different intensities
                - "uniform": Equal weight for all intensities
                - "adaptive": Higher weight for rare intensities
                - "focus": Focus on specific intensity range
            focus_range: Intensity range to focus on if using "focus" mode
        """
        super().__init__()
        self.loss_type = loss_type
        self.intensity_weight_mode = intensity_weight_mode
        self.focus_range = focus_range
        # Parameters for NLL variants (fixed variance/scale)
        # gaussian_nll uses sigma (std); laplace_nll uses scale b
        self.nll_sigma = nll_sigma if (nll_sigma is not None) else 0.1
        self.nll_scale = nll_scale if (nll_scale is not None) else 0.1
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute intensity mapping loss.
        
        Args:
            pred: Predicted intensities [B, C, H, W]
            target: Target 2P intensities [B, C, H, W]
            
        Returns:
            Intensity mapping loss
        """
        if self.intensity_weight_mode == "uniform":
            # Standard uniform loss
            if self.loss_type == "mse":
                return F.mse_loss(pred, target)
            elif self.loss_type == "l1":
                return F.l1_loss(pred, target)
            elif self.loss_type == "smooth_l1":
                return F.smooth_l1_loss(pred, target)
            elif self.loss_type == "gaussian_nll":
                # 0.5[(x-μ)^2/σ^2 + log σ^2]
                sigma2 = max(self.nll_sigma ** 2, 1e-6)
                point = 0.5 * ((pred - target) ** 2 / sigma2 + float(torch.log(torch.tensor(sigma2))))
                return point.mean()
            elif self.loss_type == "laplace_nll":
                # |x-μ|/b + log(2b)
                b = max(self.nll_scale, 1e-6)
                point = torch.abs(pred - target) / b + float(torch.log(torch.tensor(2.0 * b)))
                return point.mean()
            
        elif self.intensity_weight_mode == "adaptive":
            # Weight inversely proportional to intensity frequency
            # Higher weight for rare intensity values
            diff = pred - target
            if self.loss_type == "mse":
                pointwise_loss = diff ** 2
            elif self.loss_type == "l1":
                pointwise_loss = torch.abs(diff)
            elif self.loss_type == "smooth_l1":
                pointwise_loss = F.smooth_l1_loss(pred, target, reduction='none')
            elif self.loss_type == "gaussian_nll":
                sigma2 = max(self.nll_sigma ** 2, 1e-6)
                pointwise_loss = 0.5 * ((diff ** 2) / sigma2 + float(torch.log(torch.tensor(sigma2))))
            elif self.loss_type == "laplace_nll":
                b = max(self.nll_scale, 1e-6)
                pointwise_loss = torch.abs(diff) / b + float(torch.log(torch.tensor(2.0 * b)))
            else:
                pointwise_loss = diff ** 2
            
            # Adaptive weighting based on target intensity rarity
            # For 2P data, lower intensities are more common, so weight higher intensities more
            intensity_weights = torch.clamp(target + 1.0, 0.1, 2.0)  # Assuming [-1,1] normalized
            weighted_loss = pointwise_loss * intensity_weights
            return weighted_loss.mean()
            
        elif self.intensity_weight_mode == "focus" and self.focus_range is not None:
            # Focus loss on specific intensity range
            min_val, max_val = self.focus_range
            mask = (target >= min_val) & (target <= max_val)
            
            if self.loss_type in ("mse", "l1", "smooth_l1"):
                if self.loss_type == "mse":
                    loss_focused = F.mse_loss(pred[mask], target[mask]) if mask.any() else 0.0
                    loss_general = F.mse_loss(pred[~mask], target[~mask]) if (~mask).any() else 0.0
                elif self.loss_type == "l1":
                    loss_focused = F.l1_loss(pred[mask], target[mask]) if mask.any() else 0.0
                    loss_general = F.l1_loss(pred[~mask], target[~mask]) if (~mask).any() else 0.0
                else:  # smooth_l1
                    loss_focused = F.smooth_l1_loss(pred[mask], target[mask]) if mask.any() else 0.0
                    loss_general = F.smooth_l1_loss(pred[~mask], target[~mask]) if (~mask).any() else 0.0
            elif self.loss_type == "gaussian_nll":
                sigma2 = max(self.nll_sigma ** 2, 1e-6)
                def g_nll(a,b):
                    return 0.5 * (((a-b) ** 2).mean() / sigma2 + float(torch.log(torch.tensor(sigma2))))
                loss_focused = g_nll(pred[mask], target[mask]) if mask.any() else 0.0
                loss_general = g_nll(pred[~mask], target[~mask]) if (~mask).any() else 0.0
            elif self.loss_type == "laplace_nll":
                b = max(self.nll_scale, 1e-6)
                def l_nll(a,b_t):
                    diff = torch.abs(a-b_t)
                    return (diff.mean() / b) + float(torch.log(torch.tensor(2.0 * b)))
                loss_focused = l_nll(pred[mask], target[mask]) if mask.any() else 0.0
                loss_general = l_nll(pred[~mask], target[~mask]) if (~mask).any() else 0.0
            
            # Weight focused region more heavily
            return 2.0 * loss_focused + 0.5 * loss_general
        
        else:
            # Fallback to uniform
            return F.mse_loss(pred, target)


class DualObjectiveLoss(nn.Module):
    """
    Pure self-supervised multi-component loss function for dual objectives:
    1. Spatial resolution (sharpness)
    2. Pixel intensity (signal mapping)
    
    NO supervised components (VGG, ImageNet features, etc.)
    """
    
    def __init__(
        self,
        # Loss component weights (self-supervised only)
        alpha_diffusion: float = 1.0,
        beta_intensity: float = 0.6,
        delta_gradient: float = 0.4,
        
        # Loss component configurations
        intensity_loss_type: str = "mse",
        gradient_loss_type: str = "l1",
        intensity_weight_mode: str = "adaptive",
        
        # NLL parameters (fixed-variance/scale)
        nll_sigma: Optional[float] = None,
        nll_scale: Optional[float] = None,
        
        # Adaptive weighting
        use_adaptive_weighting: bool = True,
        warmup_steps: int = 1000
    ):
        """
        Args:
            alpha_diffusion: Weight for diffusion loss (spatial structure)
            beta_intensity: Weight for intensity mapping loss
            delta_gradient: Weight for gradient loss (sharpness)
            intensity_loss_type: Type of intensity loss ("mse", "l1", "smooth_l1")
            gradient_loss_type: Type of gradient loss ("l1", "l2", "smooth_l1")
            intensity_weight_mode: Intensity weighting strategy
            use_adaptive_weighting: Whether to adapt weights during training
            warmup_steps: Steps to gradually increase intensity loss weight
        """
        super().__init__()
        
        # Store weights (self-supervised only)
        self.alpha_diffusion = alpha_diffusion
        self.beta_intensity = beta_intensity
        self.delta_gradient = delta_gradient
        
        # Adaptive weighting parameters
        self.use_adaptive_weighting = use_adaptive_weighting
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # Initialize loss components
        self.intensity_loss = IntensityMappingLoss(
            loss_type=intensity_loss_type,
            intensity_weight_mode=intensity_weight_mode,
            nll_sigma=nll_sigma,
            nll_scale=nll_scale,
        )
        
        self.gradient_loss = GradientLoss(loss_type=gradient_loss_type)
    
    def _get_adaptive_weights(self, step: int) -> Dict[str, float]:
        """Get adaptive weights based on training progress."""
        if not self.use_adaptive_weighting:
            return {
                'alpha': self.alpha_diffusion,
                'beta': self.beta_intensity,
                'delta': self.delta_gradient
            }
        
        # Gradually increase intensity loss weight during warmup
        warmup_factor = min(1.0, step / self.warmup_steps)
        
        # Start with more focus on diffusion, gradually balance
        alpha = self.alpha_diffusion
        beta = self.beta_intensity * warmup_factor
        delta = self.delta_gradient * (0.5 + 0.5 * warmup_factor)  # Gradual gradient loss
        
        return {'alpha': alpha, 'beta': beta, 'delta': delta}
    
    def forward(
        self,
        diffusion_loss: torch.Tensor,
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        step: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined dual-objective loss.
        
        Args:
            diffusion_loss: Standard DDPM loss (noise prediction)
            pred_x0: Predicted clean image (x0 reconstruction)
            target_x0: Target clean 2P image
            step: Current training step (for adaptive weighting)
            
        Returns:
            Dictionary containing all loss components and total loss
        """
        if step is not None:
            self.current_step = step
        
        # Get adaptive weights
        weights = self._get_adaptive_weights(self.current_step)
        
        # 1. Diffusion loss (spatial structure learning)
        loss_diffusion = diffusion_loss
        
        # 2. Intensity mapping loss (pixel-wise accuracy)
        # Clamp inputs to valid model range to prevent blowups from rare out-of-range values
        pred_x0_clamped = torch.clamp(pred_x0, -1.0, 1.0)
        target_x0_clamped = torch.clamp(target_x0, -1.0, 1.0)
        loss_intensity = self.intensity_loss(pred_x0_clamped, target_x0_clamped)
        
        # 3. Gradient loss (edge preservation for sharpness)
        loss_gradient = self.gradient_loss(pred_x0_clamped, target_x0_clamped)
        
        # Combine losses with adaptive weights (self-supervised only)
        total_loss = (
            weights['alpha'] * loss_diffusion +
            weights['beta'] * loss_intensity +
            weights['delta'] * loss_gradient
        )
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': loss_diffusion,
            'intensity_loss': loss_intensity,
            'gradient_loss': loss_gradient,
            'weights': weights
        }
    
    def update_step(self, step: int):
        """Update current step for adaptive weighting."""
        self.current_step = step


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
        scale_min, scale_max = self.intensity_scale_range
        scale = torch.rand(1).item() * (scale_max - scale_min) + scale_min
        tp_aug = tp * scale
        
        # 2. Add small amount of noise to increase variation
        if self.noise_std > 0:
            noise = torch.randn_like(tp) * self.noise_std
            tp_aug = tp_aug + noise
        
        # 3. Contrast adjustment (very subtle)
        contrast_min, contrast_max = self.contrast_range
        contrast = torch.rand(1).item() * (contrast_max - contrast_min) + contrast_min
        tp_mean = tp_aug.mean()
        tp_aug = (tp_aug - tp_mean) * contrast + tp_mean
        
        # 4. Clamp to valid range
        tp_aug = torch.clamp(tp_aug, -1, 1)
        
        return wf, tp_aug


def create_dual_objective_loss(config: Dict) -> DualObjectiveLoss:
    """
    Factory function to create dual objective loss from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured DualObjectiveLoss instance
    """
    loss_config = config.get('dual_objective_loss', {})
    
    return DualObjectiveLoss(
        alpha_diffusion=loss_config.get('alpha_diffusion', 1.0),
        beta_intensity=loss_config.get('beta_intensity', 0.6),
        delta_gradient=loss_config.get('delta_gradient', 0.4),
        intensity_loss_type=loss_config.get('intensity_loss_type', 'mse'),
        gradient_loss_type=loss_config.get('gradient_loss_type', 'l1'),
        intensity_weight_mode=loss_config.get('intensity_weight_mode', 'adaptive'),
        nll_sigma=loss_config.get('nll_sigma', 0.1),
        nll_scale=loss_config.get('nll_scale', 0.1),
        use_adaptive_weighting=loss_config.get('use_adaptive_weighting', True),
        warmup_steps=loss_config.get('warmup_steps', 1000)
    )


def create_dual_objective_training_config() -> Dict[str, Any]:
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
        
        # Loss configuration (self-supervised only)
        'loss': {
            'alpha_diffusion': 1.0,    # Standard diffusion loss
            'beta_intensity': 0.8,     # High weight for intensity mapping
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


def create_progressive_training_strategy() -> Dict[str, Any]:
    """
    Progressive training strategy optimized for dual objectives.
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
                'delta_gradient': 0.5
            }
        }
    }
    
    return strategy
