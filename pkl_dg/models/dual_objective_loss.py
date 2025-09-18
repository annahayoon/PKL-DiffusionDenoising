"""
Dual-Objective Loss for Spatial Resolution and Pixel Intensity Prediction

This module implements a multi-component loss function optimized for:
1. Spatial resolution (sharpness) enhancement
2. Pixel intensity (signal mapping) prediction

The loss combines:
- Diffusion Loss: Standard DDPM loss for spatial structure learning
- Intensity Loss: MSE loss for pixel-wise intensity accuracy
- Gradient Loss: Edge preservation for sharpness
- Perceptual Loss: Spatial quality assessment (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union


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
        focus_range: Optional[Tuple[float, float]] = None
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
            
        elif self.intensity_weight_mode == "adaptive":
            # Weight inversely proportional to intensity frequency
            # Higher weight for rare intensity values
            diff = pred - target
            if self.loss_type == "mse":
                pointwise_loss = diff ** 2
            elif self.loss_type == "l1":
                pointwise_loss = torch.abs(diff)
            else:  # smooth_l1
                pointwise_loss = F.smooth_l1_loss(pred, target, reduction='none')
            
            # Adaptive weighting based on target intensity rarity
            # For 2P data, lower intensities are more common, so weight higher intensities more
            intensity_weights = torch.clamp(target + 1.0, 0.1, 2.0)  # Assuming [-1,1] normalized
            weighted_loss = pointwise_loss * intensity_weights
            return weighted_loss.mean()
            
        elif self.intensity_weight_mode == "focus" and self.focus_range is not None:
            # Focus loss on specific intensity range
            min_val, max_val = self.focus_range
            mask = (target >= min_val) & (target <= max_val)
            
            if self.loss_type == "mse":
                loss_focused = F.mse_loss(pred[mask], target[mask]) if mask.any() else 0.0
                loss_general = F.mse_loss(pred[~mask], target[~mask]) if (~mask).any() else 0.0
            elif self.loss_type == "l1":
                loss_focused = F.l1_loss(pred[mask], target[mask]) if mask.any() else 0.0
                loss_general = F.l1_loss(pred[~mask], target[~mask]) if (~mask).any() else 0.0
            else:  # smooth_l1
                loss_focused = F.smooth_l1_loss(pred[mask], target[mask]) if mask.any() else 0.0
                loss_general = F.smooth_l1_loss(pred[~mask], target[~mask]) if (~mask).any() else 0.0
            
            # Weight focused region more heavily
            return 2.0 * loss_focused + 0.5 * loss_general
        
        else:
            # Fallback to uniform
            return F.mse_loss(pred, target)


class DualObjectiveLoss(nn.Module):
    """
    Multi-component loss function for dual objectives:
    1. Spatial resolution (sharpness)
    2. Pixel intensity (signal mapping)
    """
    
    def __init__(
        self,
        # Loss component weights
        alpha_diffusion: float = 1.0,
        beta_intensity: float = 0.6,
        gamma_perceptual: float = 0.2,
        delta_gradient: float = 0.4,
        
        # Loss component configurations
        intensity_loss_type: str = "mse",
        gradient_loss_type: str = "l1",
        intensity_weight_mode: str = "adaptive",
        
        # Adaptive weighting
        use_adaptive_weighting: bool = True,
        warmup_steps: int = 1000,
        
        # Optional components
        use_perceptual_loss: bool = False,
        perceptual_layers: Optional[list] = None
    ):
        """
        Args:
            alpha_diffusion: Weight for diffusion loss (spatial structure)
            beta_intensity: Weight for intensity mapping loss
            gamma_perceptual: Weight for perceptual loss (if enabled)
            delta_gradient: Weight for gradient loss (sharpness)
            intensity_loss_type: Type of intensity loss ("mse", "l1", "smooth_l1")
            gradient_loss_type: Type of gradient loss ("l1", "l2", "smooth_l1")
            intensity_weight_mode: Intensity weighting strategy
            use_adaptive_weighting: Whether to adapt weights during training
            warmup_steps: Steps to gradually increase intensity loss weight
            use_perceptual_loss: Whether to include perceptual loss
            perceptual_layers: Which layers to use for perceptual loss
        """
        super().__init__()
        
        # Store weights
        self.alpha_diffusion = alpha_diffusion
        self.beta_intensity = beta_intensity
        self.gamma_perceptual = gamma_perceptual
        self.delta_gradient = delta_gradient
        
        # Adaptive weighting parameters
        self.use_adaptive_weighting = use_adaptive_weighting
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # Initialize loss components
        self.intensity_loss = IntensityMappingLoss(
            loss_type=intensity_loss_type,
            intensity_weight_mode=intensity_weight_mode
        )
        
        self.gradient_loss = GradientLoss(loss_type=gradient_loss_type)
        
        # Optional perceptual loss
        self.use_perceptual_loss = use_perceptual_loss
        if use_perceptual_loss:
            try:
                # Try to use torchvision VGG for perceptual loss
                import torchvision.models as models
                self.vgg = models.vgg16(pretrained=True).features[:16]  # Up to relu3_3
                for param in self.vgg.parameters():
                    param.requires_grad = False
                self.vgg.eval()
            except ImportError:
                print("Warning: torchvision not available, disabling perceptual loss")
                self.use_perceptual_loss = False
                self.vgg = None
        else:
            self.vgg = None
    
    def _compute_perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss using VGG features."""
        if not self.use_perceptual_loss or self.vgg is None:
            return torch.tensor(0.0, device=pred.device)
        
        # Convert single channel to RGB if needed
        if pred.shape[1] == 1:
            pred_rgb = pred.repeat(1, 3, 1, 1)
            target_rgb = target.repeat(1, 3, 1, 1)
        else:
            pred_rgb = pred
            target_rgb = target
        
        # Extract VGG features
        pred_features = self.vgg(pred_rgb)
        target_features = self.vgg(target_rgb)
        
        return F.mse_loss(pred_features, target_features)
    
    def _get_adaptive_weights(self, step: int) -> Dict[str, float]:
        """Get adaptive weights based on training progress."""
        if not self.use_adaptive_weighting:
            return {
                'alpha': self.alpha_diffusion,
                'beta': self.beta_intensity,
                'gamma': self.gamma_perceptual,
                'delta': self.delta_gradient
            }
        
        # Gradually increase intensity loss weight during warmup
        warmup_factor = min(1.0, step / self.warmup_steps)
        
        # Start with more focus on diffusion, gradually balance
        alpha = self.alpha_diffusion
        beta = self.beta_intensity * warmup_factor
        gamma = self.gamma_perceptual * warmup_factor
        delta = self.delta_gradient * (0.5 + 0.5 * warmup_factor)  # Gradual gradient loss
        
        return {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta}
    
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
        loss_intensity = self.intensity_loss(pred_x0, target_x0)
        
        # 3. Gradient loss (edge preservation for sharpness)
        loss_gradient = self.gradient_loss(pred_x0, target_x0)
        
        # 4. Perceptual loss (spatial quality)
        loss_perceptual = self._compute_perceptual_loss(pred_x0, target_x0)
        
        # Combine losses with adaptive weights
        total_loss = (
            weights['alpha'] * loss_diffusion +
            weights['beta'] * loss_intensity +
            weights['gamma'] * loss_perceptual +
            weights['delta'] * loss_gradient
        )
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': loss_diffusion,
            'intensity_loss': loss_intensity,
            'gradient_loss': loss_gradient,
            'perceptual_loss': loss_perceptual,
            'weights': weights
        }
    
    def update_step(self, step: int):
        """Update current step for adaptive weighting."""
        self.current_step = step


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
        gamma_perceptual=loss_config.get('gamma_perceptual', 0.2),
        delta_gradient=loss_config.get('delta_gradient', 0.4),
        intensity_loss_type=loss_config.get('intensity_loss_type', 'mse'),
        gradient_loss_type=loss_config.get('gradient_loss_type', 'l1'),
        intensity_weight_mode=loss_config.get('intensity_weight_mode', 'adaptive'),
        use_adaptive_weighting=loss_config.get('use_adaptive_weighting', True),
        warmup_steps=loss_config.get('warmup_steps', 1000),
        use_perceptual_loss=loss_config.get('use_perceptual_loss', False)
    )
