"""
Hierarchical Training and Sampling Strategies for Diffusion Models

This module implements hierarchical approaches to diffusion model training and sampling,
including multi-scale feature consistency, hierarchical noise schedules, and
pyramid-based generation strategies.

Features:
- Hierarchical feature consistency across scales
- Multi-level noise scheduling
- Pyramid-based training and sampling
- Cross-scale attention mechanisms
- Coarse-to-fine generation strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import math
import numpy as np
from collections import defaultdict

from .diffusion import DDPMTrainer
from .progressive import ProgressiveTrainer


class HierarchicalFeatureExtractor(nn.Module):
    """Feature extractor for hierarchical consistency across scales."""
    
    def __init__(
        self,
        base_channels: int = 64,
        num_levels: int = 4,
        feature_dims: Optional[List[int]] = None,
        use_attention: bool = True,
    ):
        """Initialize hierarchical feature extractor.
        
        Args:
            base_channels: Base number of channels
            num_levels: Number of hierarchical levels
            feature_dims: Feature dimensions for each level
            use_attention: Use attention for feature extraction
        """
        super().__init__()
        
        self.num_levels = num_levels
        # Create adaptive feature dimensions based on base_channels
        if feature_dims is None:
            self.feature_dims = [base_channels * (2 ** i) for i in range(num_levels)]
            # Cap at reasonable max
            self.feature_dims = [min(dim, 512) for dim in self.feature_dims]
        else:
            self.feature_dims = feature_dims
        self.use_attention = use_attention
        
        # Feature extraction layers for each level
        self.feature_layers = nn.ModuleList()
        
        for i in range(num_levels):
            in_dim = base_channels if i == 0 else feature_dims[i-1]
            out_dim = feature_dims[i]
            
            layers = [
                nn.Conv2d(in_dim, out_dim, 3, padding=1),
                nn.GroupNorm(8, out_dim),
                nn.SiLU(),
                nn.Conv2d(out_dim, out_dim, 3, padding=1),
                nn.GroupNorm(8, out_dim),
                nn.SiLU(),
            ]
            
            if use_attention and i >= 2:  # Add attention to higher levels
                from .nn import SelfAttention2D
                layers.append(SelfAttention2D(out_dim, num_heads=8))
            
            self.feature_layers.append(nn.Sequential(*layers))
        
        # Cross-scale feature fusion
        self.fusion_layers = nn.ModuleList()
        for i in range(1, num_levels):
            fusion = nn.Conv2d(feature_dims[i] + feature_dims[i-1], feature_dims[i], 1)
            self.fusion_layers.append(fusion)
    
    def forward(self, x: torch.Tensor, return_all_levels: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Extract hierarchical features.
        
        Args:
            x: Input tensor [B, C, H, W]
            return_all_levels: Return features from all levels
            
        Returns:
            Features from all levels or just the final level
        """
        features = []
        current = x
        
        for i, layer in enumerate(self.feature_layers):
            # Apply feature extraction
            feat = layer(current)
            features.append(feat)
            
            # Prepare input for next level (downsample)
            if i < len(self.feature_layers) - 1:
                current = F.avg_pool2d(feat, 2)
                
                # Fuse with previous level if not first
                if i > 0:
                    prev_feat_upsampled = F.interpolate(
                        features[i-1], size=current.shape[-2:], 
                        mode='bilinear', align_corners=False
                    )
                    current = self.fusion_layers[i-1](
                        torch.cat([current, prev_feat_upsampled], dim=1)
                    )
        
        if return_all_levels:
            return features
        else:
            return features[-1]


class HierarchicalNoiseScheduler:
    """Hierarchical noise scheduler with multi-scale noise injection."""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "hierarchical_cosine",
        num_levels: int = 4,
        level_weights: Optional[List[float]] = None,
        frequency_emphasis: bool = True,
    ):
        """Initialize hierarchical noise scheduler.
        
        Args:
            num_timesteps: Total number of timesteps
            schedule_type: Type of noise schedule
            num_levels: Number of hierarchical levels
            level_weights: Weights for each level
            frequency_emphasis: Emphasize different frequencies at different levels
        """
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        self.num_levels = num_levels
        self.frequency_emphasis = frequency_emphasis
        
        # Default level weights (coarse to fine)
        if level_weights is None:
            self.level_weights = [1.0 / (i + 1) for i in range(num_levels)]
        else:
            self.level_weights = level_weights
        
        # Create hierarchical schedules
        self.schedules = self._create_hierarchical_schedules()
        
    def _create_hierarchical_schedules(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Create noise schedules for each hierarchical level."""
        schedules = {}
        
        for level in range(self.num_levels):
            if self.schedule_type == "hierarchical_cosine":
                schedule = self._create_cosine_schedule(level)
            elif self.schedule_type == "hierarchical_linear":
                schedule = self._create_linear_schedule(level)
            elif self.schedule_type == "frequency_adaptive":
                schedule = self._create_frequency_adaptive_schedule(level)
            else:
                schedule = self._create_cosine_schedule(level)
            
            schedules[level] = schedule
        
        return schedules
    
    def _create_cosine_schedule(self, level: int) -> Dict[str, torch.Tensor]:
        """Create cosine schedule for specific level."""
        # Adjust schedule based on level (coarser levels get more noise earlier)
        level_factor = (level + 1) / self.num_levels
        
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        
        # Cosine schedule with level-specific offset
        offset = 0.008 * level_factor
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + offset) / (1.0 + offset) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        return {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        }
    
    def _create_linear_schedule(self, level: int) -> Dict[str, torch.Tensor]:
        """Create linear schedule for specific level."""
        # Level-specific beta range
        beta_start = 0.0001 * (level + 1)
        beta_end = 0.02 * (level + 1) / self.num_levels
        
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        return {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        }
    
    def _create_frequency_adaptive_schedule(self, level: int) -> Dict[str, torch.Tensor]:
        """Create frequency-adaptive schedule for specific level."""
        # Higher levels (finer details) get different noise characteristics
        if level < self.num_levels // 2:
            # Coarse levels: emphasize low frequencies
            return self._create_cosine_schedule(level)
        else:
            # Fine levels: emphasize high frequencies
            schedule = self._create_cosine_schedule(level)
            
            # Modify schedule to emphasize high-frequency noise
            schedule["betas"] = schedule["betas"] * (1.5 - 0.5 * level / self.num_levels)
            schedule["alphas"] = 1.0 - schedule["betas"]
            schedule["alphas_cumprod"] = torch.cumprod(schedule["alphas"], dim=0)
            schedule["sqrt_alphas_cumprod"] = torch.sqrt(schedule["alphas_cumprod"])
            schedule["sqrt_one_minus_alphas_cumprod"] = torch.sqrt(1.0 - schedule["alphas_cumprod"])
            
            return schedule
    
    def get_schedule(self, level: int) -> Dict[str, torch.Tensor]:
        """Get noise schedule for specific level."""
        return self.schedules.get(level, self.schedules[0])
    
    def sample_noise_hierarchical(
        self, 
        x_0: torch.Tensor, 
        t: torch.Tensor, 
        level: int,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sample noise using hierarchical schedule.
        
        Args:
            x_0: Clean input
            t: Timestep
            level: Hierarchical level
            noise: Optional noise tensor
            
        Returns:
            Noisy sample
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        schedule = self.get_schedule(level)
        
        sqrt_alpha_t = schedule["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = schedule["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
        
        # Apply level-specific weighting
        level_weight = self.level_weights[level]
        
        return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise * level_weight


class HierarchicalTrainer(ProgressiveTrainer):
    """Hierarchical trainer with multi-scale consistency and pyramid training."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        transform: Optional[Any] = None,
        forward_model: Optional[Any] = None
    ):
        """Initialize hierarchical trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            transform: Data transform
            forward_model: Forward model for guidance
        """
        super().__init__(model, config, transform, forward_model)
        
        # Hierarchical configuration
        self.hierarchical_config = config.get("hierarchical", {})
        self.enable_hierarchical = self.hierarchical_config.get("enabled", False)
        
        if self.enable_hierarchical:
            self._setup_hierarchical_components()
    
    def _setup_hierarchical_components(self):
        """Setup hierarchical training components."""
        # Number of hierarchical levels
        self.num_levels = self.hierarchical_config.get("num_levels", 4)
        
        # Feature consistency
        self.feature_consistency_weight = self.hierarchical_config.get("feature_consistency_weight", 0.1)
        self.pyramid_consistency_weight = self.hierarchical_config.get("pyramid_consistency_weight", 0.05)
        
        # Hierarchical noise scheduler
        self.hierarchical_scheduler = HierarchicalNoiseScheduler(
            num_timesteps=self.num_timesteps,
            schedule_type=self.hierarchical_config.get("schedule_type", "hierarchical_cosine"),
            num_levels=self.num_levels,
            frequency_emphasis=self.hierarchical_config.get("frequency_emphasis", True)
        )
        
        # Feature extractor for consistency
        if self.feature_consistency_weight > 0:
            # Determine input channels from model
            input_channels = getattr(self.model, 'in_channels', 1)
            if hasattr(self.model, 'base_unet'):
                input_channels = getattr(self.model.base_unet, 'in_channels', 1)
            
            self.feature_extractor = HierarchicalFeatureExtractor(
                base_channels=input_channels,
                num_levels=self.num_levels,
                use_attention=self.hierarchical_config.get("use_feature_attention", True)
            )
            self.feature_extractor.to(next(self.model.parameters()).device)
        
        # Pyramid loss components
        self.pyramid_levels = self.hierarchical_config.get("pyramid_levels", [1, 2, 4, 8])  # Downsampling factors
        
        # Frequency domain components
        if self.hierarchical_config.get("use_frequency_loss", False):
            self._setup_frequency_components()
        
        print(f"✅ Hierarchical training enabled:")
        print(f"   Levels: {self.num_levels}")
        print(f"   Feature consistency weight: {self.feature_consistency_weight}")
        print(f"   Pyramid consistency weight: {self.pyramid_consistency_weight}")
        print(f"   Pyramid levels: {self.pyramid_levels}")
    
    def _setup_frequency_components(self):
        """Setup frequency domain components."""
        # Frequency weights for different levels
        self.frequency_weights = self.hierarchical_config.get(
            "frequency_weights", [1.0, 0.8, 0.6, 0.4]
        )
        
        # Frequency bands for analysis
        self.frequency_bands = self.hierarchical_config.get(
            "frequency_bands", [(0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]
        )
    
    def training_step(self, batch, batch_idx):
        """Hierarchical training step with multi-scale consistency."""
        if not self.enable_hierarchical:
            return super().training_step(batch, batch_idx)
        
        # Preprocess batch
        if isinstance(batch, (list, tuple)):
            x_0, c_wf = batch
            x_0 = self.preprocess_batch_progressive(x_0)
            if c_wf is not None:
                c_wf = self.preprocess_batch_progressive(c_wf)
        else:
            x_0 = self.preprocess_batch_progressive(batch)
            c_wf = None
        
        b = x_0.shape[0]
        device = x_0.device
        
        # Get current resolution config
        config = self.get_current_resolution_config()
        current_resolution = config.get("resolution", self.resolution_schedule[0])
        
        # Sample timesteps and hierarchical levels
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        levels = torch.randint(0, self.num_levels, (b,), device=device)
        
        # Compute hierarchical losses
        total_loss = 0.0
        loss_components = {}
        
        # Main DDPM loss with hierarchical noise
        ddpm_loss = self._compute_hierarchical_ddpm_loss(x_0, c_wf, t, levels)
        total_loss += ddpm_loss
        loss_components["ddpm_loss"] = ddpm_loss
        
        # Feature consistency loss
        if self.feature_consistency_weight > 0:
            feature_loss = self._compute_feature_consistency_loss(x_0, current_resolution)
            total_loss += self.feature_consistency_weight * feature_loss
            loss_components["feature_loss"] = feature_loss
        
        # Pyramid consistency loss
        if self.pyramid_consistency_weight > 0:
            pyramid_loss = self._compute_pyramid_consistency_loss(x_0, t)
            total_loss += self.pyramid_consistency_weight * pyramid_loss
            loss_components["pyramid_loss"] = pyramid_loss
        
        # Frequency domain loss
        if hasattr(self, "frequency_weights"):
            frequency_loss = self._compute_frequency_consistency_loss(x_0, levels)
            total_loss += 0.1 * frequency_loss  # Weight from config
            loss_components["frequency_loss"] = frequency_loss
        
        # Cross-resolution consistency (from parent class)
        if self.cross_resolution_consistency and self.current_phase > 0:
            # Use a dummy noise prediction for consistency calculation
            with torch.no_grad():
                noise = torch.randn_like(x_0)
                x_t = self.q_sample(x_0, t, noise)
                if self.mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
                        noise_pred = self.model(x_t, t, cond=c_wf) if c_wf is not None else self.model(x_t, t)
                else:
                    noise_pred = self.model(x_t, t, cond=c_wf) if c_wf is not None else self.model(x_t, t)
                
                consistency_loss = self._compute_cross_resolution_consistency(
                    x_0, x_t, t, noise_pred, current_resolution
                )
                total_loss += config["consistency_weight"] * consistency_loss
                loss_components["consistency_loss"] = consistency_loss
        
        # Log all loss components
        for name, loss_val in loss_components.items():
            self._log_if_trainer(f"train/{name}", loss_val)
        
        # Log hierarchical info
        self._log_if_trainer("hierarchical/num_levels", self.num_levels)
        self._log_if_trainer("hierarchical/avg_level", levels.float().mean())
        
        # Track statistics
        self.phase_stats["losses"].append(total_loss.item())
        
        # Update EMA
        if self.use_ema and self.global_step % 10 == 0:
            self._update_ema()
        
        # Advance step counter
        if not hasattr(self, "_global_step"):
            self._global_step = 0
        self._global_step += 1
        
        return total_loss
    
    def _compute_hierarchical_ddpm_loss(
        self, 
        x_0: torch.Tensor, 
        c_wf: Optional[torch.Tensor], 
        t: torch.Tensor, 
        levels: torch.Tensor
    ) -> torch.Tensor:
        """Compute DDPM loss with hierarchical noise scheduling."""
        total_loss = 0.0
        
        # Group samples by level for efficient computation
        unique_levels = torch.unique(levels)
        
        for level in unique_levels:
            level_mask = (levels == level)
            level_indices = torch.where(level_mask)[0]
            
            if len(level_indices) == 0:
                continue
            
            # Ensure indices are on the same device as the tensors
            level_indices = level_indices.to(x_0.device)
            
            # Get samples for this level
            x_0_level = x_0[level_indices]
            t_level = t[level_indices]
            c_wf_level = c_wf[level_indices] if c_wf is not None else None
            
            # Sample noise using hierarchical scheduler
            noise = torch.randn_like(x_0_level)
            x_t_level = self.hierarchical_scheduler.sample_noise_hierarchical(
                x_0_level, t_level, level.item(), noise
            )
            
            # Forward pass
            use_conditioning = bool(self.config.get("use_conditioning", True))
            
            if self.mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
                    if use_conditioning and c_wf_level is not None:
                        try:
                            noise_pred = self.model(x_t_level, t_level, cond=c_wf_level)
                        except TypeError:
                            noise_pred = self.model(x_t_level, t_level)
                    else:
                        noise_pred = self.model(x_t_level, t_level)
            else:
                if use_conditioning and c_wf_level is not None:
                    try:
                        noise_pred = self.model(x_t_level, t_level, cond=c_wf_level)
                    except TypeError:
                        noise_pred = self.model(x_t_level, t_level)
                else:
                    noise_pred = self.model(x_t_level, t_level)
            
            # Compute loss for this level
            level_loss = F.mse_loss(noise_pred, noise)
            
            # Weight by number of samples at this level
            level_weight = len(level_indices) / len(levels)
            total_loss += level_weight * level_loss
        
        return total_loss
    
    def _compute_feature_consistency_loss(
        self, 
        x_0: torch.Tensor, 
        current_resolution: int
    ) -> torch.Tensor:
        """Compute feature consistency loss across scales."""
        if not hasattr(self, 'feature_extractor'):
            return torch.tensor(0.0, device=x_0.device)
        
        # Extract features at current resolution
        features_current = self.feature_extractor(x_0, return_all_levels=True)
        
        # Create downsampled versions
        consistency_loss = 0.0
        num_comparisons = 0
        
        for i, downsample_factor in enumerate([2, 4]):
            if current_resolution // downsample_factor < 32:  # Skip if too small
                continue
            
            # Downsample input
            x_0_down = F.avg_pool2d(x_0, downsample_factor)
            
            # Extract features from downsampled version
            features_down = self.feature_extractor(x_0_down, return_all_levels=True)
            
            # Compare features at corresponding levels
            for level in range(min(len(features_current), len(features_down))):
                feat_current = features_current[level]
                feat_down = features_down[level]
                
                # Resize to same spatial dimensions
                if feat_current.shape[-2:] != feat_down.shape[-2:]:
                    feat_down_resized = F.interpolate(
                        feat_down, size=feat_current.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                else:
                    feat_down_resized = feat_down
                
                # Compute consistency loss
                consistency_loss += F.mse_loss(feat_current, feat_down_resized)
                num_comparisons += 1
        
        return consistency_loss / max(num_comparisons, 1)
    
    def _compute_pyramid_consistency_loss(
        self, 
        x_0: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute pyramid consistency loss across multiple scales."""
        pyramid_loss = 0.0
        num_levels = len(self.pyramid_levels)
        
        # Generate predictions at different scales
        predictions = []
        
        for scale_factor in self.pyramid_levels:
            if scale_factor == 1:
                # Original scale
                x_scaled = x_0
            else:
                # Downsampled scale
                target_size = (x_0.shape[-2] // scale_factor, x_0.shape[-1] // scale_factor)
                if target_size[0] < 16 or target_size[1] < 16:  # Skip if too small
                    continue
                
                x_scaled = F.avg_pool2d(x_0, scale_factor)
            
            # Add noise and predict
            noise = torch.randn_like(x_scaled)
            x_t_scaled = self.q_sample(x_scaled, t, noise)
            
            # Get prediction
            if self.mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
                    noise_pred = self.model(x_t_scaled, t)
            else:
                noise_pred = self.model(x_t_scaled, t)
            
            # Predict x0
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)
            x0_pred = (x_t_scaled - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            
            predictions.append((x0_pred, scale_factor))
        
        # Compute consistency between scales
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                pred_i, scale_i = predictions[i]
                pred_j, scale_j = predictions[j]
                
                # Resize to same scale for comparison
                if scale_i < scale_j:
                    # Upsample pred_i to match pred_j
                    pred_i_resized = F.interpolate(
                        pred_i, size=pred_j.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                    pyramid_loss += F.mse_loss(pred_i_resized, pred_j)
                else:
                    # Downsample pred_j to match pred_i
                    pred_j_resized = F.avg_pool2d(pred_j, scale_j // scale_i)
                    if pred_j_resized.shape[-2:] == pred_i.shape[-2:]:
                        pyramid_loss += F.mse_loss(pred_i, pred_j_resized)
        
        return pyramid_loss / max(len(predictions) * (len(predictions) - 1) // 2, 1)
    
    def _compute_frequency_consistency_loss(
        self, 
        x_0: torch.Tensor, 
        levels: torch.Tensor
    ) -> torch.Tensor:
        """Compute frequency domain consistency loss."""
        if not hasattr(self, "frequency_bands"):
            return torch.tensor(0.0, device=x_0.device)
        
        # Compute FFT
        x_fft = torch.fft.fft2(x_0)
        x_magnitude = torch.abs(x_fft)
        
        frequency_loss = 0.0
        
        # Analyze frequency content for each sample based on its level
        for level in range(self.num_levels):
            level_mask = (levels == level)
            if not level_mask.any():
                continue
            
            level_magnitude = x_magnitude[level_mask]
            
            # Get frequency band for this level
            if level < len(self.frequency_bands):
                low_freq, high_freq = self.frequency_bands[level]
                
                # Create frequency mask
                h, w = level_magnitude.shape[-2:]
                freq_mask = self._create_frequency_mask(h, w, low_freq, high_freq, level_magnitude.device)
                
                # Apply frequency emphasis
                emphasized_magnitude = level_magnitude * freq_mask
                
                # Compute loss (encourage appropriate frequency content)
                target_magnitude = level_magnitude.mean(dim=[2, 3], keepdim=True) * freq_mask
                frequency_loss += F.mse_loss(emphasized_magnitude, target_magnitude)
        
        return frequency_loss / self.num_levels
    
    def _create_frequency_mask(
        self, 
        h: int, 
        w: int, 
        low_freq: float, 
        high_freq: float, 
        device: str
    ) -> torch.Tensor:
        """Create frequency mask for specific band."""
        # Create frequency coordinates
        freq_y = torch.fft.fftfreq(h, device=device)
        freq_x = torch.fft.fftfreq(w, device=device)
        
        freq_y_grid, freq_x_grid = torch.meshgrid(freq_y, freq_x, indexing='ij')
        freq_magnitude = torch.sqrt(freq_y_grid**2 + freq_x_grid**2)
        
        # Normalize to [0, 1]
        freq_magnitude = freq_magnitude / (freq_magnitude.max() + 1e-8)
        
        # Create band mask
        mask = ((freq_magnitude >= low_freq) & (freq_magnitude < high_freq)).float()
        
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    def get_hierarchical_summary(self) -> Dict[str, Any]:
        """Get summary of hierarchical training state."""
        if not self.enable_hierarchical:
            return {"hierarchical_enabled": False}
        
        summary = {
            "hierarchical_enabled": True,
            "num_levels": self.num_levels,
            "feature_consistency_weight": self.feature_consistency_weight,
            "pyramid_consistency_weight": self.pyramid_consistency_weight,
            "pyramid_levels": self.pyramid_levels,
            "schedule_type": self.hierarchical_scheduler.schedule_type,
        }
        
        # Add frequency info if available
        if hasattr(self, "frequency_bands"):
            summary.update({
                "frequency_bands": self.frequency_bands,
                "frequency_weights": self.frequency_weights,
            })
        
        # Add progressive info if available
        if hasattr(self, "get_progressive_summary"):
            summary.update(self.get_progressive_summary())
        
        return summary


class HierarchicalSampler:
    """Hierarchical sampler with coarse-to-fine generation."""
    
    def __init__(
        self,
        model: nn.Module,
        hierarchical_scheduler: HierarchicalNoiseScheduler,
        num_levels: int = 4,
        coarse_to_fine: bool = True,
    ):
        """Initialize hierarchical sampler.
        
        Args:
            model: Trained diffusion model
            hierarchical_scheduler: Hierarchical noise scheduler
            num_levels: Number of hierarchical levels
            coarse_to_fine: Use coarse-to-fine sampling strategy
        """
        self.model = model
        self.hierarchical_scheduler = hierarchical_scheduler
        self.num_levels = num_levels
        self.coarse_to_fine = coarse_to_fine
    
    @torch.no_grad()
    def sample_hierarchical(
        self,
        shape: Tuple[int, int, int, int],
        num_inference_steps: int = 50,
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Perform hierarchical sampling.
        
        Args:
            shape: Output shape (B, C, H, W)
            num_inference_steps: Number of inference steps
            device: Device to run on
            verbose: Show progress
            
        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        B, C, H, W = shape
        
        if self.coarse_to_fine:
            return self._sample_coarse_to_fine(shape, num_inference_steps, device, verbose)
        else:
            return self._sample_parallel_levels(shape, num_inference_steps, device, verbose)
    
    def _sample_coarse_to_fine(
        self,
        shape: Tuple[int, int, int, int],
        num_inference_steps: int,
        device: str,
        verbose: bool
    ) -> torch.Tensor:
        """Sample using coarse-to-fine strategy."""
        B, C, H, W = shape
        
        # Start with coarsest level
        current_sample = torch.randn(B, C, H // 4, W // 4, device=device)
        
        for level in range(self.num_levels):
            if verbose:
                print(f"   Hierarchical level {level + 1}/{self.num_levels}")
            
            # Get schedule for this level
            schedule = self.hierarchical_scheduler.get_schedule(level)
            
            # Determine target size for this level
            scale_factor = 2 ** (self.num_levels - level - 1)
            target_h = max(H // scale_factor, H // 4)
            target_w = max(W // scale_factor, W // 4)
            
            # Upsample if needed
            if current_sample.shape[-2:] != (target_h, target_w):
                current_sample = F.interpolate(
                    current_sample, size=(target_h, target_w),
                    mode='bilinear', align_corners=False
                )
            
            # Refine at this level
            current_sample = self._refine_at_level(
                current_sample, level, schedule, num_inference_steps // self.num_levels, device
            )
        
        # Final upsample to target resolution
        if current_sample.shape[-2:] != (H, W):
            current_sample = F.interpolate(
                current_sample, size=(H, W),
                mode='bilinear', align_corners=False
            )
        
        return current_sample
    
    def _sample_parallel_levels(
        self,
        shape: Tuple[int, int, int, int],
        num_inference_steps: int,
        device: str,
        verbose: bool
    ) -> torch.Tensor:
        """Sample all levels in parallel then combine."""
        B, C, H, W = shape
        
        # Generate samples at each level
        level_samples = []
        
        for level in range(self.num_levels):
            if verbose:
                print(f"   Generating level {level + 1}/{self.num_levels}")
            
            # Start from noise
            sample = torch.randn(shape, device=device)
            
            # Get schedule for this level
            schedule = self.hierarchical_scheduler.get_schedule(level)
            
            # Refine at this level
            refined_sample = self._refine_at_level(
                sample, level, schedule, num_inference_steps, device
            )
            
            level_samples.append(refined_sample)
        
        # Combine samples (simple average for now)
        combined_sample = torch.stack(level_samples).mean(dim=0)
        
        return combined_sample
    
    def _refine_at_level(
        self,
        sample: torch.Tensor,
        level: int,
        schedule: Dict[str, torch.Tensor],
        num_steps: int,
        device: str
    ) -> torch.Tensor:
        """Refine sample at specific hierarchical level."""
        # Use level-specific noise schedule
        timesteps = torch.linspace(num_steps - 1, 0, num_steps, dtype=torch.long, device=device)
        
        current_sample = sample
        
        for t in timesteps:
            t_batch = t.expand(sample.shape[0])
            
            # Predict noise
            if hasattr(self.model, 'model'):
                noise_pred = self.model.model(current_sample, t_batch)
            else:
                noise_pred = self.model(current_sample, t_batch)
            
            # Use level-specific schedule for denoising step
            alpha_t = schedule["alphas_cumprod"][t].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)
            
            # Predict x0
            x0_pred = (current_sample - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            
            # DDIM step
            if t > 0:
                t_next = timesteps[timesteps.tolist().index(t) + 1] if t != timesteps[-1] else torch.tensor(0, device=device)
                alpha_next = schedule["alphas_cumprod"][t_next] if t_next > 0 else torch.tensor(1.0, device=device)
                
                sqrt_alpha_next = torch.sqrt(alpha_next)
                sqrt_one_minus_alpha_next = torch.sqrt(1 - alpha_next)
                
                current_sample = sqrt_alpha_next * x0_pred + sqrt_one_minus_alpha_next * noise_pred
            else:
                current_sample = x0_pred
        
        return current_sample


# Factory functions
def create_hierarchical_trainer(
    model: nn.Module,
    config: Dict[str, Any],
    **kwargs
) -> HierarchicalTrainer:
    """Create hierarchical trainer with configuration."""
    return HierarchicalTrainer(model, config, **kwargs)


def create_hierarchical_sampler(
    model: nn.Module,
    config: Dict[str, Any]
) -> HierarchicalSampler:
    """Create hierarchical sampler with configuration."""
    hierarchical_config = config.get("hierarchical", {})
    
    # Create hierarchical scheduler
    scheduler = HierarchicalNoiseScheduler(
        num_timesteps=config.get("num_timesteps", 1000),
        schedule_type=hierarchical_config.get("schedule_type", "hierarchical_cosine"),
        num_levels=hierarchical_config.get("num_levels", 4),
        frequency_emphasis=hierarchical_config.get("frequency_emphasis", True)
    )
    
    return HierarchicalSampler(
        model=model,
        hierarchical_scheduler=scheduler,
        num_levels=hierarchical_config.get("num_levels", 4),
        coarse_to_fine=hierarchical_config.get("coarse_to_fine", True)
    )
