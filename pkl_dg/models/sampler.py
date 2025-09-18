import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from tqdm import tqdm

try:
    from einops import rearrange, repeat
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
import warnings


class DDIMSampler:
    """DDIM sampler with guidance injection.
    
    Implements the DDIM sampling algorithm with physics-guided corrections.
    Supports both deterministic (eta=0) and stochastic (eta>0) sampling.
    Compatible with both supervised and self-supervised DDPMTrainer modes.
    
    Args:
        model: Trained diffusion model (DDPMTrainer) with noise schedule buffers
        forward_model: Physics forward model for guidance computation (optional)
        guidance_strategy: Strategy for computing guidance gradients (optional)
        schedule: Adaptive schedule for guidance strength (optional)
        transform: Transform between intensity and model domains (optional)
        num_timesteps: Number of timesteps used during training
        ddim_steps: Number of inference steps (must be <= num_timesteps)
        eta: Stochasticity parameter (0=deterministic, 1=DDPM-like)
        use_autocast: Whether to use mixed precision for model inference
        clip_denoised: Whether to clip predicted x0 to [-1,1]
        v_parameterization: Whether model uses v-parameterization
    """

    def __init__(
        self,
        model: nn.Module,  # DDPMTrainer (LightningModule) exposing buffers
        forward_model: Optional['ForwardModel'] = None,
        guidance_strategy: Optional['GuidanceStrategy'] = None,
        schedule: Optional['AdaptiveSchedule'] = None,
        transform: Optional['IntensityToModel'] = None,
        num_timesteps: int = 1000,
        ddim_steps: int = 100,
        eta: float = 0.0,
        use_autocast: bool = False,
        clip_denoised: bool = True,
        v_parameterization: bool = False,
    ):
        # Validate inputs
        if ddim_steps > num_timesteps:
            raise ValueError(f"ddim_steps ({ddim_steps}) cannot exceed num_timesteps ({num_timesteps})")
        if not 0 <= eta <= 1:
            raise ValueError(f"eta must be in [0, 1], got {eta}")
        
        self.model = model
        self.forward_model = forward_model
        self.guidance = guidance_strategy
        self.schedule = schedule
        self.transform = transform
        self.num_timesteps = num_timesteps
        self.ddim_steps = ddim_steps
        self.eta = eta
        self.use_autocast = use_autocast
        self.clip_denoised = clip_denoised
        self.v_parameterization = v_parameterization
        self.ddim_timesteps = self._setup_ddim_timesteps()
        
        # Validate model has required buffers
        if not hasattr(model, 'alphas_cumprod'):
            raise ValueError("Model must have 'alphas_cumprod' buffer for DDIM sampling")
        
        # Check if model is in self-supervised mode
        self.self_supervised = getattr(model, 'self_supervised', False)
        
        # Auto-detect mixed precision settings from model
        if hasattr(model, 'mixed_precision') and hasattr(model, 'autocast_dtype'):
            self.model_mixed_precision = model.mixed_precision
            self.model_autocast_dtype = model.autocast_dtype
        else:
            self.model_mixed_precision = False
            self.model_autocast_dtype = torch.float32

    def _setup_ddim_timesteps(self):
        """Setup DDIM timestep sequence with proper spacing."""
        if self.ddim_steps == 1:
            return torch.tensor([self.num_timesteps - 1])
        
        # Use uniform spacing for better coverage
        step_size = self.num_timesteps // self.ddim_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size)
        
        # Ensure we include the last timestep if not already present
        if timesteps[-1] != self.num_timesteps - 1:
            timesteps = torch.cat([timesteps, torch.tensor([self.num_timesteps - 1])])
        
        # Reverse for sampling (high to low)
        return timesteps.flip(0)

    @torch.no_grad()
    def sample(
        self, 
        y: torch.Tensor, 
        shape: tuple, 
        device: Optional[str] = None, 
        verbose: bool = True,
        return_intermediates: bool = False,
        conditioner: Optional[torch.Tensor] = None,
        return_16bit: bool = False,
        save_path: Optional[str] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run guided DDIM sampling.
        
        Args:
            y: Measurement tensor for guidance
            shape: Shape of samples to generate (B, C, H, W)
            device: Device to run sampling on (defaults to y.device)
            verbose: Whether to show progress bar
            return_intermediates: Whether to return intermediate states
            conditioner: Optional conditioning tensor
            return_16bit: If True, return 16-bit numpy array instead of tensor
            save_path: Optional path to save 16-bit TIFF output
            
        Returns:
            Generated samples in intensity domain, or dict with intermediates
            If return_16bit=True, returns 16-bit numpy array
        """
        if device is None:
            device = y.device
            
        # Validate inputs
        if len(shape) != 4:
            raise ValueError(f"Shape must be 4D (B,C,H,W), got {shape}")
        if y.device != torch.device(device):
            warnings.warn(f"Moving measurement from {y.device} to {device}")
            
        # Initialize noise
        x_t = torch.randn(shape, device=device, dtype=y.dtype)
        y = y.to(device)
        
        # Storage for intermediates
        intermediates = {"x_intermediates": [], "x0_predictions": []} if return_intermediates else None
        
        iterator = tqdm(self.ddim_timesteps, desc="DDIM Sampling") if verbose else self.ddim_timesteps
        
        for i, t in enumerate(iterator):
            t_cur = t.item() if isinstance(t, torch.Tensor) else t
            t_next = self.ddim_timesteps[i + 1].item() if i < len(self.ddim_timesteps) - 1 else 0
            
            # If we already reached t=0, x_t should already be x0 from previous step
            if t_cur == 0:
                if return_intermediates:
                    intermediates["x_intermediates"].append(x_t.clone())
                    intermediates["x0_predictions"].append(x_t.clone())
                break
            
            # Predict clean image with optional conditioner
            x0_hat = self._predict_x0(x_t, t_cur, conditioner)
            
            # Apply guidance correction (except at final step)
            if t_cur > 0 and self._can_apply_guidance():
                try:
                    x0_hat_corrected = self._apply_guidance(x0_hat, y, t_cur)
                except Exception as e:
                    warnings.warn(f"Guidance failed at step {i}: {e}. Using uncorrected prediction.")
                    x0_hat_corrected = x0_hat
            else:
                x0_hat_corrected = x0_hat
            
            # Store intermediates
            if return_intermediates:
                intermediates["x_intermediates"].append(x_t.clone())
                intermediates["x0_predictions"].append(x0_hat_corrected.clone())
            
            # DDIM step
            x_t = self._ddim_step(x_t, x0_hat_corrected, t_cur, t_next)
            
            # Check for NaN/Inf
            if not torch.isfinite(x_t).all():
                raise RuntimeError(f"Non-finite values detected at step {i}")
        
        # Final clean estimate is x_t after the last DDIM step
        # At the terminal step we already produce x0_hat in model domain,
        # so avoid an extra (and incorrect) model forward at t=0.
        if self.transform is not None:
            x0_intensity = self.transform.inverse(x_t)
        else:
            # If no transform, assume x_t is already in the desired domain
            x0_intensity = x_t
        
        # Handle 16-bit output if requested
        if return_16bit or save_path:
            from .diffusion import to_uint16_grayscale, save_16bit_grayscale
            
            # Convert to 16-bit using the transform
            output_16bit = to_uint16_grayscale(x_t, transform=self.transform)
            
            if save_path:
                # Save as 16-bit TIFF
                save_16bit_grayscale(x_t, save_path, transform=self.transform)
            
            if return_16bit:
                if return_intermediates:
                    intermediates["final_intensity"] = x0_intensity
                    intermediates["final_16bit"] = output_16bit
                    return intermediates
                return output_16bit
        
        if return_intermediates:
            intermediates["final_intensity"] = x0_intensity
            return intermediates
        
        return x0_intensity

    def _predict_x0(self, x_t: torch.Tensor, t: Union[int, torch.Tensor], conditioner: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict clean image x0 from noisy input x_t at timestep t."""
        # Convert timestep to tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x_t.device, dtype=torch.long)
        if t.dim() == 0:
            t = t.repeat(x_t.shape[0])
        
        # Get noise schedule values
        alpha_t = self.model.alphas_cumprod[t]
        if EINOPS_AVAILABLE:
            alpha_t = rearrange(alpha_t, 'b -> b 1 1 1')
        else:
            alpha_t = alpha_t.view(-1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)  # Add epsilon for stability
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)
        
        # Use EMA model if available - updated for unified DDPMTrainer
        if hasattr(self.model, 'ema_model') and getattr(self.model, 'use_ema', False):
            net = self.model.ema_model
        else:
            # In unified DDPMTrainer, the UNet is stored in self.model.model
            net = getattr(self.model, 'model', self.model)
        
        # Model prediction with unified mixed precision handling
        if (self.use_autocast or self.model_mixed_precision) and x_t.is_cuda:
            # Use model's autocast dtype if available, otherwise default to float16
            autocast_dtype = getattr(self, 'model_autocast_dtype', torch.float16)
            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                model_output = self._forward_model(net, x_t, t, conditioner)
        else:
            model_output = self._forward_model(net, x_t, t, conditioner)
        
        # Handle different parameterizations
        if self.v_parameterization:
            # v-parameterization: x0 = sqrt(alpha_t) * x_t - sqrt(1-alpha_t) * v
            x0_hat = sqrt_alpha_t * x_t - sqrt_one_minus_alpha_t * model_output
        else:
            # epsilon-parameterization: x0 = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
            x0_hat = (x_t - sqrt_one_minus_alpha_t * model_output) / sqrt_alpha_t
        
        # Optional clipping
        if self.clip_denoised:
            x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
        
        return x0_hat

    def _can_apply_guidance(self) -> bool:
        """Check if all components needed for guidance are available."""
        return all([
            self.forward_model is not None,
            self.guidance is not None,
            self.schedule is not None,
            self.transform is not None
        ])
    
    def _apply_guidance(self, x0_hat: torch.Tensor, y: torch.Tensor, t: int) -> torch.Tensor:
        """Apply physics-guided correction with proper error handling."""
        # Batch domain conversions to reduce overhead
        with torch.no_grad():
            # Single conversion to intensity domain
            x0_intensity = self.transform.inverse(x0_hat)
            x0_intensity = torch.clamp(x0_intensity, min=0)
            
            # Compute guidance
            gradient = self.guidance.compute_gradient(x0_intensity, y, self.forward_model, t)
            lambda_t = self.schedule.get_lambda_t(gradient, t)
            x0_corrected = self.guidance.apply_guidance(x0_intensity, gradient, lambda_t)
            x0_corrected = torch.clamp(x0_corrected, min=0)
            
            # Single conversion back to model domain
            x0_corrected_model = self.transform(x0_corrected)
        return x0_corrected_model

    def _ddim_step(self, x_t: torch.Tensor, x0_hat: torch.Tensor, t_cur: int, t_next: int) -> torch.Tensor:
        """Perform single DDIM step from t_cur to t_next."""
        # Handle final step
        if t_next == 0:
            return x0_hat
        
        # Get noise schedule values with numerical stability
        alpha_cur = self.model.alphas_cumprod[t_cur]
        alpha_next = self.model.alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=x_t.device)
        
        # Ensure alphas are valid
        alpha_cur = torch.clamp(alpha_cur, min=1e-8, max=1.0)
        alpha_next = torch.clamp(alpha_next, min=1e-8, max=1.0)
        
        # DDIM variance parameter
        sigma_t = self.eta * torch.sqrt((1 - alpha_next) / (1 - alpha_cur)) * torch.sqrt(1 - alpha_cur / alpha_next)
        
        # Predicted noise (epsilon parameterization) with numerically stable denominator
        sqrt_alpha_cur = torch.sqrt(alpha_cur)
        denom = torch.sqrt(1 - alpha_cur).clamp_min(1e-8)
        pred_noise = (x_t - sqrt_alpha_cur * x0_hat) / denom
        
        # DDIM update equation
        sqrt_alpha_next = torch.sqrt(alpha_next)
        dir_x_t = torch.sqrt(torch.clamp(1 - alpha_next - sigma_t**2, min=0)) * pred_noise
        
        x_next = sqrt_alpha_next * x0_hat + dir_x_t
        
        # Add stochastic component if eta > 0
        if self.eta > 0 and t_next > 0:
            noise = torch.randn_like(x_t)
            x_next += sigma_t * noise
        
        return x_next
    
    @classmethod
    def from_ddpm_trainer(
        cls,
        trainer: 'DDPMTrainer',
        forward_model: Optional['ForwardModel'] = None,
        guidance_strategy: Optional['GuidanceStrategy'] = None,
        schedule: Optional['AdaptiveSchedule'] = None,
        ddim_steps: int = 100,
        eta: float = 0.0,
        clip_denoised: bool = True,
        v_parameterization: bool = False,
    ) -> 'DDIMSampler':
        """Create DDIMSampler from a trained DDPMTrainer.
        
        Args:
            trainer: Trained DDPMTrainer instance
            forward_model: Optional physics forward model
            guidance_strategy: Optional guidance strategy
            schedule: Optional adaptive schedule
            ddim_steps: Number of DDIM steps
            eta: Stochasticity parameter
            clip_denoised: Whether to clip predictions
            v_parameterization: Whether model uses v-parameterization
            
        Returns:
            Configured DDIMSampler instance
        """
        # Extract configuration from trainer
        num_timesteps = trainer.num_timesteps
        transform = trainer.transform
        use_autocast = getattr(trainer, 'mixed_precision', False)
        
        # Use trainer's forward model if not provided
        if forward_model is None:
            forward_model = getattr(trainer, 'forward_model', None)
        
        return cls(
            model=trainer,
            forward_model=forward_model,
            guidance_strategy=guidance_strategy,
            schedule=schedule,
            transform=transform,
            num_timesteps=num_timesteps,
            ddim_steps=ddim_steps,
            eta=eta,
            use_autocast=use_autocast,
            clip_denoised=clip_denoised,
            v_parameterization=v_parameterization,
        )
    
    def _forward_model(self, net: nn.Module, x_t: torch.Tensor, t: torch.Tensor, conditioner: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model with proper conditioning support."""
        try:
            if conditioner is not None:
                return net(x_t, t, cond=conditioner)
            else:
                return net(x_t, t)
        except TypeError:
            # Fallback for models that don't support conditioning
            return net(x_t, t)


