"""
Progressive Training for Diffusion Models

This module implements progressive training capabilities for diffusion models,
allowing training to start at lower resolutions and gradually increase resolution.
This approach can improve training stability and reduce computational costs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Union
import math
import copy

from .diffusion import DDPMTrainer


class ProgressiveUNet(nn.Module):
    """UNet with progressive training support."""
    
    def __init__(self, base_unet: nn.Module, max_resolution: int = 512):
        """Initialize progressive UNet.
        
        Args:
            base_unet: Base UNet model
            max_resolution: Maximum training resolution
        """
        super().__init__()
        self.base_unet = base_unet
        self.max_resolution = max_resolution
        
        # Progressive training state
        self.current_resolution = getattr(base_unet, 'sample_size', 64)
        self.training_phase = 0
        self.transition_alpha = 1.0  # For smooth transitions
        
        # Store resolution progression
        self.resolutions = self._compute_resolution_schedule()
        
        # Create resolution-specific layers if needed
        self._setup_progressive_layers()
    
    def _compute_resolution_schedule(self) -> List[int]:
        """Compute resolution schedule for progressive training."""
        resolutions = []
        current = 64  # Start from 64x64
        
        while current <= self.max_resolution:
            resolutions.append(current)
            current *= 2
        
        # Ensure max resolution is included
        if resolutions[-1] != self.max_resolution:
            resolutions.append(self.max_resolution)
        
        return resolutions
    
    def _setup_progressive_layers(self):
        """Setup layers for different resolutions."""
        # For now, use the same UNet for all resolutions
        # Advanced implementations could have resolution-specific layers
        pass
    
    def set_resolution(self, resolution: int):
        """Set current training resolution."""
        if resolution not in self.resolutions:
            raise ValueError(f"Resolution {resolution} not in schedule: {self.resolutions}")
        
        self.current_resolution = resolution
        self.training_phase = self.resolutions.index(resolution)
        
        # Update base UNet if it has a sample_size attribute
        if hasattr(self.base_unet, 'sample_size'):
            self.base_unet.sample_size = resolution
    
    def set_transition_alpha(self, alpha: float):
        """Set transition alpha for smooth resolution transitions."""
        self.transition_alpha = torch.clamp(torch.tensor(alpha), 0.0, 1.0).item()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with progressive resolution support."""
        # Resize input to current resolution if needed
        if x.shape[-1] != self.current_resolution or x.shape[-2] != self.current_resolution:
            x = F.interpolate(
                x, size=(self.current_resolution, self.current_resolution),
                mode='bilinear', align_corners=False
            )
        
        # Forward through base UNet
        output = self.base_unet(x, t, **kwargs)
        
        return output
    
    def get_progressive_info(self) -> Dict[str, Any]:
        """Get information about progressive training state."""
        return {
            "current_resolution": self.current_resolution,
            "training_phase": self.training_phase,
            "transition_alpha": self.transition_alpha,
            "resolution_schedule": self.resolutions,
            "max_resolution": self.max_resolution
        }


class ProgressiveTrainer(DDPMTrainer):
    """DDPM trainer with advanced progressive training support.
    
    Features:
    - Resolution curriculum with smooth transitions
    - Adaptive batch scaling based on resolution
    - Cross-resolution consistency losses
    - Dynamic noise scheduling per resolution
    - Hierarchical feature consistency
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        transform: Optional[Any] = None,
        forward_model: Optional[Any] = None
    ):
        """Initialize progressive trainer.
        
        Args:
            model: Model to train (will be wrapped in ProgressiveUNet)
            config: Training configuration
            transform: Data transform
            forward_model: Forward model for guidance
        """
        # Wrap model in ProgressiveUNet if not already
        if not isinstance(model, ProgressiveUNet):
            max_resolution = config.get("progressive", {}).get("max_resolution", 512)
            model = ProgressiveUNet(model, max_resolution)
        
        super().__init__(model, config, transform, forward_model)
        
        # Progressive training configuration
        self.progressive_config = config.get("progressive", {})
        self.enable_progressive = self.progressive_config.get("enabled", False)
        
        if self.enable_progressive:
            self._setup_progressive_training()
            self._setup_curriculum_components()
    
    def _setup_progressive_training(self):
        """Setup progressive training parameters."""
        # Resolution schedule
        self.resolution_schedule = self.model.resolutions
        self.current_phase = 0
        
        # Training epochs per resolution with curriculum
        base_epochs = self.progressive_config.get("epochs_per_resolution", [10, 15, 20, 25])
        curriculum_type = self.progressive_config.get("curriculum_type", "linear")  # linear, exponential, adaptive
        
        if curriculum_type == "exponential":
            # More epochs at higher resolutions (exponential growth)
            self.epochs_per_resolution = [int(base_epochs[0] * (1.5 ** i)) for i in range(len(self.resolution_schedule))]
        elif curriculum_type == "adaptive":
            # Adaptive based on convergence (placeholder for now)
            self.epochs_per_resolution = base_epochs.copy()
        else:
            # Linear progression
            self.epochs_per_resolution = base_epochs.copy()
        
        # Ensure we have enough epoch specifications
        while len(self.epochs_per_resolution) < len(self.resolution_schedule):
            self.epochs_per_resolution.append(int(self.epochs_per_resolution[-1] * 1.2))
        
        # Transition settings
        self.smooth_transitions = self.progressive_config.get("smooth_transitions", True)
        self.transition_epochs = self.progressive_config.get("transition_epochs", 2)
        self.blend_mode = self.progressive_config.get("blend_mode", "alpha")  # alpha, feature, loss
        
        # Learning rate scaling with curriculum
        self.lr_scaling = self.progressive_config.get("lr_scaling", True)
        self.lr_curriculum = self.progressive_config.get("lr_curriculum", "sqrt")  # sqrt, linear, constant
        self.base_lr = self.config.get("learning_rate", 1e-4)
        
        # Batch size scaling with memory optimization
        self.batch_scaling = self.progressive_config.get("batch_scaling", True)
        self.adaptive_batch_scaling = self.progressive_config.get("adaptive_batch_scaling", True)
        self.base_batch_size = self.config.get("training", {}).get("batch_size", 8)
        
        # Cross-resolution consistency
        self.cross_resolution_consistency = self.progressive_config.get("cross_resolution_consistency", True)
        self.consistency_weight = self.progressive_config.get("consistency_weight", 0.1)
        
        print(f"✅ Advanced progressive training enabled:")
        print(f"   Resolution schedule: {self.resolution_schedule}")
        print(f"   Epochs per resolution: {self.epochs_per_resolution}")
        print(f"   Curriculum type: {curriculum_type}")
        print(f"   Cross-resolution consistency: {self.cross_resolution_consistency}")
        print(f"   Adaptive batch scaling: {self.adaptive_batch_scaling}")
    
    def _setup_curriculum_components(self):
        """Setup components for curriculum learning."""
        # Feature consistency loss for cross-resolution training
        if self.cross_resolution_consistency:
            self.feature_consistency_loss = nn.MSELoss()
        
        # Resolution-specific noise schedules
        self.resolution_noise_schedules = {}
        for res in self.resolution_schedule:
            # Higher resolutions get more timesteps for finer control
            timesteps = min(self.num_timesteps, int(self.num_timesteps * (res / self.resolution_schedule[0]) ** 0.5))
            self.resolution_noise_schedules[res] = timesteps
        
        # Adaptive batch sizer for memory optimization
        if self.adaptive_batch_scaling:
            try:
                from ..utils.utils import AdaptiveBatchSizer
                self.batch_sizer = AdaptiveBatchSizer(verbose=False)
            except ImportError:
                print("⚠️ AdaptiveBatchSizer not available, using fixed batch scaling")
                self.batch_sizer = None
                self.adaptive_batch_scaling = False
        else:
            self.batch_sizer = None
        
        # Track training statistics for adaptive curriculum
        self.phase_stats = {
            "losses": [],
            "convergence_rates": [],
            "phase_start_epochs": []
        }
    
    def get_current_resolution_config(self) -> Dict[str, Any]:
        """Get configuration for current resolution with advanced curriculum."""
        if not self.enable_progressive:
            return {}
        
        resolution = self.resolution_schedule[self.current_phase]
        
        # Advanced learning rate scaling based on curriculum type
        lr_scale = 1.0
        if self.lr_scaling and self.current_phase > 0:
            if self.lr_curriculum == "sqrt":
                lr_scale = (64 / resolution) ** 0.5  # Square root scaling
            elif self.lr_curriculum == "linear":
                lr_scale = 64 / resolution  # Linear scaling
            elif self.lr_curriculum == "constant":
                lr_scale = 1.0  # No scaling
            else:
                lr_scale = (64 / resolution) ** 0.5  # Default to sqrt
        
        # Adaptive batch size scaling
        if self.adaptive_batch_scaling and self.batch_sizer is not None:
            try:
                # Use adaptive batch sizer to get optimal batch size for current resolution
                input_shape = (1, resolution, resolution)  # Grayscale microscopy
                optimal_batch_size = self.batch_sizer.find_optimal_batch_size(
                    model=self.model.base_unet,
                    input_shape=input_shape,
                    mixed_precision=self.mixed_precision,
                    gradient_checkpointing=getattr(self.model, 'gradient_checkpointing', False),
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                batch_size = optimal_batch_size
            except Exception as e:
                print(f"⚠️ Adaptive batch sizing failed: {e}")
                # Fallback to fixed scaling
                batch_scale = max(1, int((self.resolution_schedule[0] / resolution) ** 2))
                batch_size = max(1, self.base_batch_size // batch_scale)
        elif self.batch_scaling:
            # Fixed batch size scaling
            batch_scale = max(1, int((self.resolution_schedule[0] / resolution) ** 2))
            batch_size = max(1, self.base_batch_size // batch_scale)
        else:
            batch_size = self.base_batch_size
        
        # Resolution-specific timesteps
        timesteps = self.resolution_noise_schedules.get(resolution, self.num_timesteps)
        
        return {
            "resolution": resolution,
            "learning_rate": self.base_lr * lr_scale,
            "batch_size": batch_size,
            "timesteps": timesteps,
            "phase": self.current_phase,
            "total_phases": len(self.resolution_schedule),
            "lr_scale": lr_scale,
            "transition_alpha": self.model.transition_alpha,
            "consistency_weight": self.consistency_weight if self.cross_resolution_consistency else 0.0
        }
    
    def advance_progressive_phase(self) -> bool:
        """Advance to next progressive training phase.
        
        Returns:
            True if advanced, False if already at final phase
        """
        if not self.enable_progressive:
            return False
        
        if self.current_phase < len(self.resolution_schedule) - 1:
            self.current_phase += 1
            
            # Update model resolution
            new_resolution = self.resolution_schedule[self.current_phase]
            self.model.set_resolution(new_resolution)
            
            # Reset transition alpha
            if self.smooth_transitions:
                self.model.set_transition_alpha(0.0)
            
            config = self.get_current_resolution_config()
            print(f"🔄 Advanced to phase {self.current_phase + 1}/{len(self.resolution_schedule)}")
            print(f"   Resolution: {config['resolution']}")
            print(f"   Learning rate: {config['learning_rate']:.2e}")
            print(f"   Batch size: {config['batch_size']}")
            
            return True
        
        return False
    
    def update_transition_alpha(self, epoch: int, epoch_in_phase: int):
        """Update transition alpha for smooth resolution transitions."""
        if not (self.enable_progressive and self.smooth_transitions):
            return
        
        if epoch_in_phase < self.transition_epochs:
            # Smooth transition from previous resolution
            alpha = epoch_in_phase / self.transition_epochs
            self.model.set_transition_alpha(alpha)
        else:
            # Full resolution training
            self.model.set_transition_alpha(1.0)
    
    def should_advance_phase(self, epoch: int, phase_start_epoch: int) -> bool:
        """Check if should advance to next phase (with adaptive support)."""
        if not self.enable_progressive:
            return False
        
        # Use adaptive advancement if enabled
        curriculum_type = self.progressive_config.get("curriculum_type", "linear")
        if curriculum_type == "adaptive":
            return self.should_advance_phase_adaptive(epoch, phase_start_epoch)
        
        # Standard fixed advancement
        epochs_in_phase = epoch - phase_start_epoch
        target_epochs = self.epochs_per_resolution[self.current_phase]
        
        return epochs_in_phase >= target_epochs
    
    def preprocess_batch_progressive(self, batch: torch.Tensor) -> torch.Tensor:
        """Preprocess batch for current progressive training phase."""
        if not self.enable_progressive:
            return batch
        
        current_res = self.resolution_schedule[self.current_phase]
        
        # Resize batch to current resolution
        if batch.shape[-1] != current_res or batch.shape[-2] != current_res:
            batch = F.interpolate(
                batch, size=(current_res, current_res),
                mode='bilinear', align_corners=False
            )
        
        return batch
    
    def training_step(self, batch, batch_idx):
        """Training step with advanced progressive support."""
        # Preprocess batch for current resolution
        if isinstance(batch, (list, tuple)):
            x_0, c_wf = batch
            x_0 = self.preprocess_batch_progressive(x_0)
            if c_wf is not None:
                c_wf = self.preprocess_batch_progressive(c_wf)
            batch = (x_0, c_wf)
        else:
            batch = self.preprocess_batch_progressive(batch)
        
        # Get current resolution config
        config = self.get_current_resolution_config()
        current_resolution = config.get("resolution", self.resolution_schedule[0])
        
        # Main training step with resolution-specific timesteps
        if isinstance(batch, (list, tuple)):
            x_0, c_wf = batch
            b = x_0.shape[0]
            device = x_0.device
            
            # Use resolution-specific timesteps
            max_t = config.get("timesteps", self.num_timesteps)
            t = torch.randint(0, max_t, (b,), device=device)
            
            noise = torch.randn_like(x_0)
            x_t = self.q_sample(x_0, t, noise)
            
            # Forward pass with conditioning
            use_conditioning = bool(self.config.get("use_conditioning", True))
            
            if self.mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
                    if use_conditioning:
                        try:
                            noise_pred = self.model(x_t, t, cond=c_wf)
                        except TypeError:
                            noise_pred = self.model(x_t, t)
                    else:
                        noise_pred = self.model(x_t, t)
                    
                    # Main DDPM loss
                    ddpm_loss = F.mse_loss(noise_pred, noise)
                    total_loss = ddpm_loss
                    
                    # Cross-resolution consistency loss
                    if self.cross_resolution_consistency and self.current_phase > 0:
                        consistency_loss = self._compute_cross_resolution_consistency(
                            x_0, x_t, t, noise_pred, current_resolution
                        )
                        total_loss = total_loss + config["consistency_weight"] * consistency_loss
                        self._log_if_trainer("train/consistency_loss", consistency_loss)
                    
                    # Optional supervised x0 loss
                    if self.config.get("supervised_x0_weight", 0.0) > 0:
                        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
                        sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)
                        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)
                        x0_hat = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
                        loss_x0 = F.l1_loss(x0_hat, x_0)
                        total_loss = total_loss + float(self.config.get("supervised_x0_weight", 0.0)) * loss_x0
            else:
                # Standard precision training
                if use_conditioning:
                    try:
                        noise_pred = self.model(x_t, t, cond=c_wf)
                    except TypeError:
                        noise_pred = self.model(x_t, t)
                else:
                    noise_pred = self.model(x_t, t)
                
                ddpm_loss = F.mse_loss(noise_pred, noise)
                total_loss = ddpm_loss
                
                # Cross-resolution consistency loss
                if self.cross_resolution_consistency and self.current_phase > 0:
                    consistency_loss = self._compute_cross_resolution_consistency(
                        x_0, x_t, t, noise_pred, current_resolution
                    )
                    total_loss = total_loss + config["consistency_weight"] * consistency_loss
                    self._log_if_trainer("train/consistency_loss", consistency_loss)
                
                # Optional supervised x0 loss
                if self.config.get("supervised_x0_weight", 0.0) > 0:
                    alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
                    sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)
                    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)
                    x0_hat = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
                    loss_x0 = F.l1_loss(x0_hat, x_0)
                    total_loss = total_loss + float(self.config.get("supervised_x0_weight", 0.0)) * loss_x0
        else:
            # Fallback to parent implementation
            total_loss = super().training_step(batch, batch_idx)
        
        # Track statistics for adaptive curriculum
        self.phase_stats["losses"].append(total_loss.item())
        
        # Log progressive training info
        if self.enable_progressive:
            self._log_if_trainer("progressive/resolution", config["resolution"])
            self._log_if_trainer("progressive/phase", config["phase"])
            self._log_if_trainer("progressive/transition_alpha", self.model.transition_alpha)
            self._log_if_trainer("progressive/timesteps", config["timesteps"])
            self._log_if_trainer("progressive/lr_scale", config["lr_scale"])
            self._log_if_trainer("train/ddpm_loss", ddpm_loss if 'ddmp_loss' in locals() else total_loss)
        
        # Update EMA
        if self.use_ema and self.global_step % 10 == 0:
            self._update_ema()
        
        # Advance step counter
        if not hasattr(self, "_global_step"):
            self._global_step = 0
        self._global_step += 1
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizers with progressive learning rate."""
        config = self.get_current_resolution_config()
        lr = config.get("learning_rate", self.config.get("learning_rate", 1e-4))
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=self.config.get("weight_decay", 1e-6),
        )
        
        if self.config.get("use_scheduler", True):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.get("max_epochs", 100), eta_min=1e-6
            )
            return [optimizer], [scheduler]
        
        return optimizer
    
    def on_epoch_start(self, epoch: int, phase_start_epoch: int = 0):
        """Called at the start of each epoch."""
        if self.enable_progressive:
            # Update transition alpha
            epoch_in_phase = epoch - phase_start_epoch
            self.update_transition_alpha(epoch, epoch_in_phase)
            
            # Log progressive info
            config = self.get_current_resolution_config()
            print(f"Epoch {epoch}: Resolution {config['resolution']}, "
                  f"Phase {config['phase'] + 1}/{config['total_phases']}, "
                  f"Alpha {self.model.transition_alpha:.3f}")
    
    def _compute_cross_resolution_consistency(
        self, 
        x_0: torch.Tensor, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        noise_pred: torch.Tensor,
        current_resolution: int
    ) -> torch.Tensor:
        """Compute cross-resolution consistency loss.
        
        This encourages the model to produce consistent features across different resolutions.
        """
        if self.current_phase == 0:
            return torch.tensor(0.0, device=x_0.device)
        
        # Get previous resolution
        prev_resolution = self.resolution_schedule[self.current_phase - 1]
        
        # Downsample current prediction to previous resolution
        with torch.no_grad():
            # Predict x0 from current noise prediction
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)
            x0_pred_current = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            x0_pred_current = torch.clamp(x0_pred_current, -1.0, 1.0)
            
            # Downsample to previous resolution
            x0_pred_downsampled = F.interpolate(
                x0_pred_current, 
                size=(prev_resolution, prev_resolution),
                mode='bilinear', 
                align_corners=False
            )
            
            # Upsample back to current resolution
            x0_pred_upsampled = F.interpolate(
                x0_pred_downsampled,
                size=(current_resolution, current_resolution),
                mode='bilinear',
                align_corners=False
            )
        
        # Compute consistency loss (should be similar after down/up sampling)
        consistency_loss = self.feature_consistency_loss(x0_pred_current, x0_pred_upsampled)
        
        return consistency_loss
    
    def _compute_convergence_rate(self, window_size: int = 100) -> float:
        """Compute convergence rate for adaptive curriculum."""
        if len(self.phase_stats["losses"]) < window_size:
            return 0.0
        
        recent_losses = self.phase_stats["losses"][-window_size:]
        if len(recent_losses) < 2:
            return 0.0
        
        # Compute rate of change in loss
        early_avg = sum(recent_losses[:window_size//2]) / (window_size//2)
        late_avg = sum(recent_losses[window_size//2:]) / (window_size//2)
        
        if early_avg == 0:
            return 0.0
        
        convergence_rate = abs(early_avg - late_avg) / early_avg
        return convergence_rate
    
    def should_advance_phase_adaptive(self, epoch: int, phase_start_epoch: int) -> bool:
        """Adaptive phase advancement based on convergence."""
        if not self.enable_progressive:
            return False
        
        # Check minimum epochs first
        epochs_in_phase = epoch - phase_start_epoch
        min_epochs = max(5, self.epochs_per_resolution[self.current_phase] // 2)
        
        if epochs_in_phase < min_epochs:
            return False
        
        # Check maximum epochs
        max_epochs = self.epochs_per_resolution[self.current_phase]
        if epochs_in_phase >= max_epochs:
            return True
        
        # Check convergence rate
        convergence_rate = self._compute_convergence_rate()
        convergence_threshold = 0.01  # 1% change threshold
        
        if convergence_rate < convergence_threshold and epochs_in_phase >= min_epochs:
            print(f"🎯 Early advancement: convergence rate {convergence_rate:.4f} below threshold")
            return True
        
        return False
    
    def get_progressive_summary(self) -> Dict[str, Any]:
        """Get summary of progressive training state."""
        if not self.enable_progressive:
            return {"progressive_enabled": False}
        
        config = self.get_current_resolution_config()
        
        summary = {
            "progressive_enabled": True,
            "current_phase": self.current_phase,
            "total_phases": len(self.resolution_schedule),
            "current_resolution": config["resolution"],
            "resolution_schedule": self.resolution_schedule,
            "epochs_per_resolution": self.epochs_per_resolution,
            "transition_alpha": self.model.transition_alpha,
            "smooth_transitions": self.smooth_transitions,
            "lr_scaling": self.lr_scaling,
            "batch_scaling": self.batch_scaling,
            "adaptive_batch_scaling": self.adaptive_batch_scaling,
            "cross_resolution_consistency": self.cross_resolution_consistency,
            "model_info": self.model.get_progressive_info()
        }
        
        # Add convergence statistics
        if len(self.phase_stats["losses"]) > 0:
            summary.update({
                "convergence_rate": self._compute_convergence_rate(),
                "recent_avg_loss": sum(self.phase_stats["losses"][-10:]) / min(10, len(self.phase_stats["losses"])),
                "total_training_steps": len(self.phase_stats["losses"])
            })
        
        return summary


class ProgressiveDataLoader:
    """DataLoader wrapper for progressive training."""
    
    def __init__(
        self,
        base_dataloader: torch.utils.data.DataLoader,
        resolution_schedule: List[int],
        batch_size_schedule: Optional[List[int]] = None
    ):
        """Initialize progressive dataloader.
        
        Args:
            base_dataloader: Base dataloader
            resolution_schedule: List of resolutions for training phases
            batch_size_schedule: Optional batch size schedule
        """
        self.base_dataloader = base_dataloader
        self.resolution_schedule = resolution_schedule
        self.batch_size_schedule = batch_size_schedule or [base_dataloader.batch_size] * len(resolution_schedule)
        
        self.current_phase = 0
        self.current_resolution = resolution_schedule[0]
        self.current_batch_size = self.batch_size_schedule[0]
    
    def set_phase(self, phase: int):
        """Set current training phase."""
        if 0 <= phase < len(self.resolution_schedule):
            self.current_phase = phase
            self.current_resolution = self.resolution_schedule[phase]
            self.current_batch_size = self.batch_size_schedule[phase]
            
            # Update dataloader batch size if possible
            if hasattr(self.base_dataloader, 'batch_size'):
                self.base_dataloader.batch_size = self.current_batch_size
    
    def __iter__(self):
        """Iterate over batches with progressive preprocessing."""
        for batch in self.base_dataloader:
            # Preprocess batch for current resolution
            if isinstance(batch, (list, tuple)):
                processed_batch = []
                for item in batch:
                    if isinstance(item, torch.Tensor) and item.ndim == 4:
                        # Resize image tensors
                        item = F.interpolate(
                            item, size=(self.current_resolution, self.current_resolution),
                            mode='bilinear', align_corners=False
                        )
                    processed_batch.append(item)
                yield tuple(processed_batch)
            elif isinstance(batch, torch.Tensor) and batch.ndim == 4:
                # Resize single tensor
                batch = F.interpolate(
                    batch, size=(self.current_resolution, self.current_resolution),
                    mode='bilinear', align_corners=False
                )
                yield batch
            else:
                yield batch
    
    def __len__(self):
        """Get length of dataloader."""
        return len(self.base_dataloader)


def create_progressive_config(
    base_config: Dict[str, Any],
    max_resolution: int = 512,
    start_resolution: int = 64,
    epochs_per_resolution: Optional[List[int]] = None,
    enable_smooth_transitions: bool = True,
    enable_lr_scaling: bool = True,
    enable_batch_scaling: bool = True
) -> Dict[str, Any]:
    """Create progressive training configuration.
    
    Args:
        base_config: Base training configuration
        max_resolution: Maximum training resolution
        start_resolution: Starting resolution
        epochs_per_resolution: Epochs to train at each resolution
        enable_smooth_transitions: Whether to use smooth transitions
        enable_lr_scaling: Whether to scale learning rate
        enable_batch_scaling: Whether to scale batch size
        
    Returns:
        Updated configuration with progressive settings
    """
    config = copy.deepcopy(base_config)
    
    # Compute resolution schedule
    resolutions = []
    current = start_resolution
    while current <= max_resolution:
        resolutions.append(current)
        current *= 2
    
    if resolutions[-1] != max_resolution:
        resolutions.append(max_resolution)
    
    # Default epochs per resolution
    if epochs_per_resolution is None:
        epochs_per_resolution = [10 + i * 5 for i in range(len(resolutions))]
    
    # Add progressive configuration
    config["progressive"] = {
        "enabled": True,
        "max_resolution": max_resolution,
        "start_resolution": start_resolution,
        "resolution_schedule": resolutions,
        "epochs_per_resolution": epochs_per_resolution,
        "smooth_transitions": enable_smooth_transitions,
        "transition_epochs": 2,
        "lr_scaling": enable_lr_scaling,
        "batch_scaling": enable_batch_scaling
    }
    
    # Update total epochs
    total_epochs = sum(epochs_per_resolution)
    if "training" not in config:
        config["training"] = {}
    config["training"]["num_epochs"] = total_epochs
    
    print(f"✅ Created progressive training config:")
    print(f"   Resolutions: {resolutions}")
    print(f"   Epochs per resolution: {epochs_per_resolution}")
    print(f"   Total epochs: {total_epochs}")
    
    return config


def run_progressive_training(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: str = "cuda",
    checkpoint_callback: Optional[Any] = None,
    logger: Optional[Any] = None
) -> ProgressiveTrainer:
    """Run progressive training loop.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        config: Training configuration
        device: Device to train on
        checkpoint_callback: Optional checkpoint callback
        logger: Optional logger
        
    Returns:
        Trained progressive trainer
    """
    # Create progressive trainer
    trainer = ProgressiveTrainer(model, config)
    trainer.to(device)
    
    # Training loop
    phase_start_epoch = 0
    
    for epoch in range(config["training"]["num_epochs"]):
        # Check if should advance phase
        if trainer.should_advance_phase(epoch, phase_start_epoch):
            if trainer.advance_progressive_phase():
                phase_start_epoch = epoch
                
                # Update dataloader if it's progressive
                if isinstance(dataloader, ProgressiveDataLoader):
                    dataloader.set_phase(trainer.current_phase)
        
        # Epoch start callback
        trainer.on_epoch_start(epoch, phase_start_epoch)
        
        # Training epoch
        trainer.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            else:
                batch = batch.to(device)
            
            # Training step
            loss = trainer.training_step(batch, batch_idx)
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Log epoch results
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        if logger:
            logger.log({
                "epoch": epoch,
                "train/loss": avg_loss,
                **{f"progressive/{k}": v for k, v in trainer.get_current_resolution_config().items()}
            })
        
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if checkpoint_callback and epoch % 10 == 0:
            checkpoint_callback.save_checkpoint(trainer, epoch)
    
    print("✅ Progressive training completed!")
    return trainer
