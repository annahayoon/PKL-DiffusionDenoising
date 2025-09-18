from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from .nn import NNDefaults, get_activation
from .dual_objective_loss import DualObjectiveLoss, create_dual_objective_loss

# Import dataset from data module
try:
    from ..data import UnpairedDataset
    UNPAIRED_DATASET_AVAILABLE = True
except ImportError:
    UnpairedDataset = None
    UNPAIRED_DATASET_AVAILABLE = False

# Import advanced components for SOTA integration
try:
    from .progressive import ProgressiveTrainer, ProgressiveUNet
    from .hierarchical_strategy import HierarchicalTrainer, HierarchicalSampler
    from .cascaded_sampling import CascadedSampler, create_cascaded_sampler
    from .losses import create_frequency_loss, get_frequency_loss_config
    from .advanced_schedulers import create_scheduler, get_scheduler_config, SchedulerManager
    from ..utils.utils import AdaptiveBatchSizer, create_adaptive_dataloader
    SOTA_FEATURES_AVAILABLE = True
except ImportError:
    # Fallback when SOTA features are not available
    SOTA_FEATURES_AVAILABLE = False
    ProgressiveTrainer = None
    HierarchicalTrainer = None

# Import diffusers schedulers with fallback
try:
    from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
    DIFFUSERS_SCHEDULERS_AVAILABLE = True
except (ImportError, RuntimeError, ModuleNotFoundError):
    DIFFUSERS_SCHEDULERS_AVAILABLE = False
    DDPMScheduler = None
    DDIMScheduler = None
    DPMSolverMultistepScheduler = None

try:
    import pytorch_lightning as pl  # type: ignore

    LightningModuleBase = pl.LightningModule  # type: ignore
except Exception:  # pragma: no cover - fallback when lightning is unavailable
    class LightningModuleBase(nn.Module):
        def __init__(self):
            super().__init__()
            self._global_step = 0

        def log(self, *args, **kwargs):
            # no-op in fallback
            return None

        @property
        def global_step(self) -> int:
            return getattr(self, "_global_step", 0)


# Import loss functions from losses module
try:
    from .losses import CycleConsistencyLoss, PerceptualLoss, create_loss_function
    LOSSES_AVAILABLE = True
except ImportError:
    LOSSES_AVAILABLE = False
    CycleConsistencyLoss = None
    PerceptualLoss = None


class DDPMTrainer(LightningModuleBase):
    """DDPM training with PyTorch Lightning."""

    def __init__(
        self, 
        model: nn.Module, 
        config: Dict[str, Any], 
        transform: Optional[Any] = None,
        forward_model: Optional[Any] = None
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.transform = transform
        self.forward_model = forward_model
        self.num_timesteps = config.get("num_timesteps", 1000)
        self.beta_schedule = config.get("beta_schedule", "cosine")
        self.use_diffusers_scheduler = config.get("use_diffusers_scheduler", True) and DIFFUSERS_SCHEDULERS_AVAILABLE
        
        # Self-supervised training mode
        self.self_supervised = config.get("self_supervised", False)
        
        # Advanced features configuration
        self.advanced_config = config.get("advanced_features", {})
        self.enable_frequency_losses = self.advanced_config.get("enable_frequency_losses", False)
        self.enable_advanced_schedulers = self.advanced_config.get("enable_advanced_schedulers", False)
        self.enable_memory_optimization = self.advanced_config.get("enable_memory_optimization", False)
        self.enable_cascaded_sampling = self.advanced_config.get("enable_cascaded_sampling", False)
        
        # Initialize self-supervised components if enabled
        if self.self_supervised:
            self._init_self_supervised_components()
        
        # Initialize advanced components if available and enabled
        if SOTA_FEATURES_AVAILABLE:
            self._init_advanced_components()
        
        # Setup noise schedule (diffusers or manual)
        if self.use_diffusers_scheduler:
            self._setup_diffusers_scheduler()
        else:
            self._setup_manual_noise_schedule()
            
        self.use_ema = config.get("use_ema", True)
        if self.use_ema:
            self.ema_model = self._create_ema_model()
            
        # Mixed precision training setup
        self.mixed_precision = config.get("mixed_precision", False)
        self.autocast_dtype = self._get_optimal_dtype()
        
        # Enable memory-efficient attention on newer GPUs
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability()
            self.enable_optimized_attention = device_capability[0] >= 8  # A100/H100/RTX 40 series
            
            # Initialize CUDA streams for async operations
            self.enable_cuda_streams = config.get("enable_cuda_streams", True)
            if self.enable_cuda_streams:
                self.compute_stream = torch.cuda.Stream()
                self.data_stream = torch.cuda.Stream()
            else:
                self.compute_stream = None
                self.data_stream = None
        else:
            self.enable_optimized_attention = False
            self.enable_cuda_streams = False
            self.compute_stream = None
            self.data_stream = None
        
        # Initialize gradient scaler for mixed precision
        if self.mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Initialize dual objective loss if enabled
        self.use_dual_objective_loss = config.get("use_dual_objective_loss", False)
        if self.use_dual_objective_loss:
            self.dual_objective_loss = create_dual_objective_loss(config)
            print("✅ Dual-objective loss enabled for spatial resolution + intensity mapping")
        else:
            self.dual_objective_loss = None
    
    def _init_self_supervised_components(self):
        """Initialize components for self-supervised training."""
        # Loss components
        self.cycle_loss = CycleConsistencyLoss(
            loss_type=self.config.get("cycle_loss_type", "l1")
        )
        
        # Perceptual loss - configurable pretrained features
        use_pretrained_perceptual = self.config.get("use_pretrained_perceptual", False)
        self.perceptual_loss = PerceptualLoss(
            use_pretrained=use_pretrained_perceptual
        ) if self.config.get("use_perceptual_loss", False) else None
        
        # Loss weights
        self.ddpm_weight = self.config.get("ddpm_loss_weight", 1.0)
        self.cycle_weight = self.config.get("cycle_loss_weight", 0.1)
        self.perceptual_weight = self.config.get("perceptual_loss_weight", 0.01)
        
        # Forward model type
        self.forward_model_type = self.config.get("forward_model_type", "physics")  # 'physics', 'learned', 'identity'
    
    def _init_advanced_components(self):
        """Initialize advanced feature components if available."""
        # Advanced frequency losses
        if self.enable_frequency_losses:
            self._setup_frequency_losses()
        
        # Advanced schedulers (beyond diffusers)
        if self.enable_advanced_schedulers:
            self._setup_advanced_schedulers()
        
        # Memory optimization
        if self.enable_memory_optimization:
            self._setup_memory_optimization()
        
        # Cascaded sampling support
        if self.enable_cascaded_sampling:
            self._setup_cascaded_sampling()
        
        # Performance monitoring
        self.performance_stats = {
            "memory_usage": [],
            "training_speed": [],
            "loss_improvements": [],
            "batch_adjustments": 0,
            "scheduler_transitions": 0,
        }
        
        if any([self.enable_frequency_losses, self.enable_advanced_schedulers, 
                self.enable_memory_optimization, self.enable_cascaded_sampling]):
            print("🚀 Advanced features enabled:")
            if self.enable_frequency_losses:
                print("   ✅ Frequency domain losses")
            if self.enable_advanced_schedulers:
                print("   ✅ Advanced schedulers")
            if self.enable_memory_optimization:
                print("   ✅ Memory optimization")
            if self.enable_cascaded_sampling:
                print("   ✅ Cascaded sampling")
    
    def _setup_frequency_losses(self):
        """Setup advanced frequency domain losses."""
        freq_config = self.advanced_config.get("frequency_losses", {})
        
        # Multi-scale frequency loss
        self.frequency_loss = create_frequency_loss(
            loss_type="multi_scale",
            scales=[1, 2, 4],
            use_fourier=freq_config.get("use_fourier", True),
            use_spectral=freq_config.get("use_spectral", True),
            use_wavelet=freq_config.get("use_wavelet", False),
            fourier_weight=freq_config.get("fourier_weight", 1.0),
            spectral_weight=freq_config.get("spectral_weight", 0.5),
            wavelet_weight=freq_config.get("wavelet_weight", 0.3),
        )
        
        # High-frequency preservation loss
        self.high_freq_loss = create_frequency_loss(
            loss_type="high_frequency",
            high_freq_threshold=freq_config.get("high_freq_threshold", 0.5),
            emphasis_factor=freq_config.get("emphasis_factor", 2.0),
        )
        
        # Loss weights
        self.frequency_loss_weight = self.advanced_config.get("frequency_loss_weight", 0.1)
        self.high_freq_loss_weight = freq_config.get("high_freq_loss_weight", 0.05)
    
    def _setup_advanced_schedulers(self):
        """Setup advanced noise schedulers beyond diffusers."""
        scheduler_config = self.advanced_config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "improved_cosine")
        
        # Create advanced scheduler
        self.advanced_scheduler = create_scheduler(
            scheduler_type=scheduler_type,
            num_timesteps=self.num_timesteps,
            **scheduler_config.get("params", {})
        )
        
        # Scheduler manager for dynamic switching
        self.scheduler_manager = SchedulerManager()
        self.scheduler_manager.add_scheduler("base", self.advanced_scheduler)
        
        # Add alternative schedulers for dynamic switching
        alt_schedulers = scheduler_config.get("alternatives", [])
        for alt_name, alt_config in alt_schedulers:
            alt_scheduler = create_scheduler(
                scheduler_type=alt_config["type"],
                num_timesteps=self.num_timesteps,
                **alt_config.get("params", {})
            )
            self.scheduler_manager.add_scheduler(alt_name, alt_scheduler)
    
    def _setup_memory_optimization(self):
        """Setup memory optimization components."""
        memory_config = self.advanced_config.get("memory_optimization", {})
        
        # Adaptive batch sizer
        self.batch_sizer = AdaptiveBatchSizer(
            safety_factor=memory_config.get("safety_factor", 0.8),
            enable_dynamic_scaling=memory_config.get("enable_dynamic_scaling", True),
            memory_pressure_threshold=memory_config.get("memory_pressure_threshold", 0.85),
            verbose=memory_config.get("verbose", False),
        )
        
        # Memory monitoring
        self.memory_monitoring_enabled = memory_config.get("enable_monitoring", True)
    
    def _setup_cascaded_sampling(self):
        """Setup cascaded sampling components."""
        cascaded_config = self.advanced_config.get("cascaded_sampling", {})
        
        if cascaded_config.get("enabled", True):
            # Create cascaded sampler
            self.cascaded_sampler = create_cascaded_sampler(
                model=self.model,
                sampler_type=cascaded_config.get("type", "basic"),
                resolution_schedule=cascaded_config.get("resolution_schedule", [64, 128, 256, 512]),
                **cascaded_config.get("params", {})
            )
        else:
            self.cascaded_sampler = None

    def _log_if_trainer(self, *args, **kwargs) -> None:
        """Log only if a Lightning Trainer is attached to avoid warnings in tests.

        Accessing `self.trainer` on a LightningModule that is not attached raises.
        Wrap in try/except to keep standalone loops working.
        """
        try:
            trainer_ref = getattr(self, "trainer")  # Lightning property may raise
        except Exception:
            trainer_ref = None
        if trainer_ref is not None:
            try:
                # type: ignore[attr-defined]
                self.log(*args, **kwargs)  # pragma: no cover - passthrough when trainer present
            except Exception:
                pass

    def _setup_diffusers_scheduler(self):
        """Setup noise schedule using diffusers schedulers for better performance."""
        # Choose scheduler based on beta_schedule config
        scheduler_type = self.config.get("scheduler_type", "ddpm")  # ddpm, ddim, dpm_solver
        
        # Common scheduler config optimized for microscopy
        scheduler_config = {
            "num_train_timesteps": self.num_timesteps,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "squaredcos_cap_v2" if self.beta_schedule == "cosine" else self.beta_schedule,
            "prediction_type": "epsilon",
        }
        
        # Create the appropriate scheduler
        if scheduler_type == "ddim":
            self.scheduler = DDIMScheduler(**scheduler_config)
        elif scheduler_type == "dpm_solver":
            # DPM-Solver specific config
            dpm_config = {
                "num_train_timesteps": self.num_timesteps,
                "beta_start": 0.0001,
                "beta_end": 0.02,
                "beta_schedule": "squaredcos_cap_v2" if self.beta_schedule == "cosine" else self.beta_schedule,
                "prediction_type": "epsilon",
                "algorithm_type": "dpmsolver++",
                "solver_order": 2,
                "use_karras_sigmas": bool(self.config.get("use_karras_sigmas", True)),
            }
            self.scheduler = DPMSolverMultistepScheduler(**dpm_config)
        else:  # Default to DDPM
            self.scheduler = DDPMScheduler(**scheduler_config)
        
        # Extract noise schedule parameters for compatibility
        self._extract_scheduler_parameters()
        
        print(f"✅ Using {type(self.scheduler).__name__} scheduler")
    
    def _extract_scheduler_parameters(self):
        """Extract parameters from diffusers scheduler for backward compatibility."""
        # Get the scheduler's noise schedule
        betas = self.scheduler.betas
        alphas = self.scheduler.alphas
        alphas_cumprod = self.scheduler.alphas_cumprod
        
        # Register as buffers for compatibility with existing code
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
        # Compute additional parameters for DDPM sampling
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Posterior parameters
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def _setup_manual_noise_schedule(self):
        if self.beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, self.num_timesteps)
        elif self.beta_schedule == "cosine":
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        # Precompute posterior coefficients as in ddpm.py for DDPM reverse sampling
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def _create_ema_model(self):
        from copy import deepcopy

        ema_model = deepcopy(self.model)
        ema_model.requires_grad_(False)
        return ema_model
    
    def _get_optimal_dtype(self):
        """Determine optimal dtype for mixed precision based on GPU capability."""
        if not torch.cuda.is_available():
            return torch.float32
            
        # Check GPU compute capability
        device_capability = torch.cuda.get_device_capability()
        major, minor = device_capability
        
        # Use bfloat16 on Ampere (8.x) and newer, float16 on older GPUs
        if major >= 8:
            return torch.bfloat16
        elif major >= 7:  # Volta and Turing
            return torch.float16
        else:
            return torch.float32  # Older GPUs don't support mixed precision well

    # ===== Helper conversions between model and intensity domains =====
    @staticmethod
    def _model_to_intensity(x_model: torch.Tensor, transform: Optional[Any]) -> torch.Tensor:
        """Map model domain [-1,1] to non-negative intensity domain via transform.inverse.

        If no transform is provided, assume inputs are already intensities and clamp to >= 0.
        """
        if transform is None:
            return torch.clamp(x_model, min=0)
        # Transform modules in this repo expose .inverse() for model->intensity
        return torch.clamp(transform.inverse(x_model), min=0)

    @staticmethod
    def _intensity_to_model(x_intensity: torch.Tensor, transform: Optional[Any]) -> torch.Tensor:
        """Map non-negative intensity domain to model domain [-1,1].

        If no transform is provided, pass through (assumes model trained in intensity domain).
        """
        if transform is None:
            return x_intensity
        return torch.clamp(transform(x_intensity), -1.0, 1.0)

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1, 1
        )
        return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise

    # ===== DDPM reverse-process utilities (mirroring ddpm.py) =====
    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Extract a[t] then reshape to broadcast over x - CORRECTED."""
        # Use direct indexing instead of gather for clarity and consistency
        out = a[t]
        return out.view(-1, *([1] * (len(x_shape) - 1)))

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)
        return (
            self._extract(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        model_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return model_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise: torch.Tensor, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True):
        # Recover x0 then compute posterior q(x_{t-1} | x_t, x0)
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, pred_noise: torch.Tensor, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True) -> torch.Tensor:
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    # ===== Guided DDPM sampling per ICLR_2025.tex (PKL, L2, Anscombe) =====
    @torch.no_grad()
    def ddpm_guided_sample(
        self,
        y: torch.Tensor,
        forward_model: Any,
        guidance_strategy: Any,
        schedule: Optional[Any] = None,
        transform: Optional[Any] = None,
        num_steps: Optional[int] = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Guided ancestral sampling using a physics-informed gradient.

        Args:
            y: Observed measurement tensor in intensity counts, shape [B, C, H, W]
            forward_model: Instance providing apply_psf(.), apply_psf_adjoint(.), and .background
            guidance_strategy: Instance with compute_gradient(x0_hat, y, forward_model, t)
            schedule: Instance with get_lambda_t(gradient, t) implementing the adaptive schedule
            transform: Transform object mapping intensity<->model domains (e.g., IntensityToModel)
            num_steps: Optional number of diffusion steps; defaults to configured self.num_timesteps
            use_ema: Whether to use EMA weights for prediction

        Returns:
            Samples in model domain [-1, 1] with shape [B, C, H, W]
        """
        device = next(self.model.parameters()).device
        B, C, H, W = y.shape
        steps = int(num_steps) if num_steps is not None else int(self.num_timesteps)

        # Initialize x_T ~ N(0, I) in model domain
        x_t = torch.randn((B, C, H, W), device=device)
        net = self.ema_model if (use_ema and getattr(self, 'use_ema', False)) else self.model

        for t_scalar in reversed(range(0, steps)):
            tt = torch.full((B,), t_scalar, device=device, dtype=torch.long)
            # Predict noise epsilon_theta(x_t, t) with mixed precision
            if self.mixed_precision and torch.cuda.is_available():
                with autocast(dtype=self.autocast_dtype):
                    pred_noise = net(x_t, tt)
            else:
                pred_noise = net(x_t, tt)

            # Recover current estimate x0_hat in model domain
            x0_hat_model = self.predict_start_from_noise(x_t, t=tt, noise=pred_noise)
            x0_hat_model = torch.clamp(x0_hat_model, -1.0, 1.0)

            # Convert to intensity domain for physics guidance
            x0_hat_int = self._model_to_intensity(x0_hat_model, transform)

            # Compute guidance gradient (in intensity domain) if components are provided
            if (forward_model is not None) and (guidance_strategy is not None) and (schedule is not None):
                grad = guidance_strategy.compute_gradient(x0_hat_int, y, forward_model, t_scalar)
                lambda_t = schedule.get_lambda_t(grad, t_scalar)
                x0_hat_int_guided = torch.clamp(x0_hat_int - float(lambda_t) * grad, min=0)
                x0_hat_model_guided = self._intensity_to_model(x0_hat_int_guided, transform)
            else:
                # No guidance; use uncorrected estimate
                x0_hat_model_guided = x0_hat_model

            # Use guided x0_hat in posterior q(x_{t-1} | x_t, x0)
            model_mean, _, model_log_variance = self.q_posterior(x_start=x0_hat_model_guided, x_t=x_t, t=tt)

            # DDPM noise injection (no noise when t == 0)
            noise = torch.randn_like(x_t)
            nonzero_mask = (tt != 0).float().view(B, 1, 1, 1)
            x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

        return x_t

    @torch.no_grad()
    def ddpm_sample(self, num_images: int, image_shape: tuple, use_ema: bool = True) -> torch.Tensor:
        """Generate samples by full DDPM ancestral sampling (slow).

        Args:
            num_images: Batch size
            image_shape: (C, H, W)
            use_ema: Whether to use EMA weights for prediction
        Returns:
            Samples in model domain [-1, 1]
        """
        device = next(self.model.parameters()).device
        samples = torch.randn((num_images, *image_shape), device=device)
        net = self.ema_model if (use_ema and getattr(self, 'use_ema', False)) else self.model
        for t in reversed(range(0, self.num_timesteps)):
            tt = torch.full((num_images,), t, device=device, dtype=torch.long)
            # Use mixed precision for sampling
            if self.mixed_precision and torch.cuda.is_available():
                with autocast(dtype=self.autocast_dtype):
                    pred_noise = net(samples, tt)
            else:
                pred_noise = net(samples, tt)
            samples = self.p_sample(pred_noise, samples, tt, clip_denoised=True)
        return samples

    def training_step(self, batch, batch_idx):
        # Expect (target_2p, conditioner_wf)
        x_0, c_wf = batch
        b = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        
        # Use mixed precision if enabled
        use_conditioning = bool(self.config.get("use_conditioning", True))
        
        if self.mixed_precision and torch.cuda.is_available():
            with autocast(dtype=self.autocast_dtype):
                if use_conditioning:
                    try:
                        noise_pred = self.model(x_t, t, cond=c_wf)
                    except TypeError:
                        noise_pred = self.model(x_t, t)
                else:
                    noise_pred = self.model(x_t, t)
                
                # Standard diffusion loss (noise prediction)
                diffusion_loss = F.mse_loss(noise_pred, noise)
                
                # Reconstruct x0 for dual objective loss
                alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)
                x0_hat = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
                
                # Apply dual objective loss if enabled
                if self.use_dual_objective_loss:
                    loss_components = self.dual_objective_loss(
                        diffusion_loss=diffusion_loss,
                        pred_x0=x0_hat,
                        target_x0=x_0,
                        step=self.global_step
                    )
                    loss = loss_components['total_loss']
                    
                    # Log individual loss components
                    self._log_if_trainer("train/diffusion_loss", loss_components['diffusion_loss'])
                    self._log_if_trainer("train/intensity_loss", loss_components['intensity_loss'])
                    self._log_if_trainer("train/gradient_loss", loss_components['gradient_loss'])
                    self._log_if_trainer("train/perceptual_loss", loss_components['perceptual_loss'])
                    
                else:
                    loss = diffusion_loss
                    
                    # Optional supervised x0 loss to encourage paired mapping
                    if self.config.get("supervised_x0_weight", 0.0) > 0:
                        loss_x0 = F.l1_loss(x0_hat, x_0)
                        loss = loss + float(self.config.get("supervised_x0_weight", 0.0)) * loss_x0
                        
        else:
            # Standard precision training
            if use_conditioning:
                try:
                    noise_pred = self.model(x_t, t, cond=c_wf)
                except TypeError:
                    noise_pred = self.model(x_t, t)
            else:
                noise_pred = self.model(x_t, t)
            
            # Standard diffusion loss (noise prediction)
            diffusion_loss = F.mse_loss(noise_pred, noise)
            
            # Reconstruct x0 for dual objective loss
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)
            x0_hat = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            
            # Apply dual objective loss if enabled
            if self.use_dual_objective_loss:
                loss_components = self.dual_objective_loss(
                    diffusion_loss=diffusion_loss,
                    pred_x0=x0_hat,
                    target_x0=x_0,
                    step=self.global_step
                )
                loss = loss_components['total_loss']
                
                # Log individual loss components
                self._log_if_trainer("train/diffusion_loss", loss_components['diffusion_loss'])
                self._log_if_trainer("train/intensity_loss", loss_components['intensity_loss'])
                self._log_if_trainer("train/gradient_loss", loss_components['gradient_loss'])
                self._log_if_trainer("train/perceptual_loss", loss_components['perceptual_loss'])
                
            else:
                loss = diffusion_loss
                
                # Optional supervised x0 loss to encourage paired mapping
                if self.config.get("supervised_x0_weight", 0.0) > 0:
                    loss_x0 = F.l1_loss(x0_hat, x_0)
                    loss = loss + float(self.config.get("supervised_x0_weight", 0.0)) * loss_x0
        
        # Add frequency domain losses if enabled
        if SOTA_FEATURES_AVAILABLE and self.enable_frequency_losses:
            freq_loss = self._compute_frequency_losses(batch)
            loss = loss + freq_loss
        
        # Advanced features monitoring and optimization
        if SOTA_FEATURES_AVAILABLE and hasattr(self, 'performance_stats'):
            # Monitor memory if enabled
            if self.enable_memory_optimization and batch_idx % 50 == 0:
                self._monitor_memory_usage()
            
            # Update performance stats
            self.performance_stats["loss_improvements"].append(loss.item())
            
            # Dynamic batch size adjustment
            if self.enable_memory_optimization and batch_idx % 100 == 0:
                self._check_batch_size_adjustment(batch_idx)
            
            # Log advanced feature metrics
            self._log_advanced_metrics(batch_idx, loss)
        
        self._log_if_trainer("train/loss", loss, prog_bar=True)
        if self.use_ema and self.global_step % 10 == 0:
            self._update_ema()
        # advance step counter in fallback environments
        if not hasattr(self, "_global_step"):
            self._global_step = 0
        self._global_step += 1
        return loss

    def _update_ema(self, decay: float = 0.999):
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
    
    def update_ema(self, decay: float = 0.999):
        """Public method to update EMA model parameters."""
        if self.use_ema and self.ema_model is not None:
            self._update_ema(decay)

    def validation_step(self, batch, batch_idx):
        x_0, c_wf = batch
        device = x_0.device
        losses = []
        # Select evaluation timesteps based on configured num_timesteps to avoid OOB indices
        if self.num_timesteps <= 3:
            t_candidates = list(range(max(int(self.num_timesteps) - 1, 1)))
            if not t_candidates:
                t_candidates = [0]
        else:
            # Use approximately 10%, 50%, 90% of the range
            t_candidates = [
                max(0, min(self.num_timesteps - 1, int(0.1 * (self.num_timesteps - 1)))),
                max(0, min(self.num_timesteps - 1, int(0.5 * (self.num_timesteps - 1)))),
                max(0, min(self.num_timesteps - 1, int(0.9 * (self.num_timesteps - 1)))),
            ]
            t_candidates = sorted(set(t_candidates))
        
        use_conditioning = bool(self.config.get("use_conditioning", True))
        
        for t_val in t_candidates:
            t = torch.full((x_0.shape[0],), int(t_val), device=device, dtype=torch.long)
            noise = torch.randn_like(x_0)
            x_t = self.q_sample(x_0, t, noise)
            
            # Use mixed precision for validation too
            if self.mixed_precision and torch.cuda.is_available():
                with autocast(dtype=self.autocast_dtype):
                    if use_conditioning:
                        try:
                            noise_pred = self.model(x_t, t, cond=c_wf)
                        except TypeError:
                            noise_pred = self.model(x_t, t)
                    else:
                        noise_pred = self.model(x_t, t)
                    losses.append(F.mse_loss(noise_pred, noise))
            else:
                if use_conditioning:
                    try:
                        noise_pred = self.model(x_t, t, cond=c_wf)
                    except TypeError:
                        noise_pred = self.model(x_t, t)
                else:
                    noise_pred = self.model(x_t, t)
                losses.append(F.mse_loss(noise_pred, noise))
                
        avg_loss = torch.stack(losses).mean()
        self._log_if_trainer("val/loss", avg_loss, prog_bar=True)
        return avg_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-4),
            betas=(0.9, 0.999),
            weight_decay=self.config.get("weight_decay", 1e-6),
        )
        if self.config.get("use_scheduler", True):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.get("max_epochs", 100), eta_min=1e-6
            )
            return [optimizer], [scheduler]
        return optimizer
    
    # ===== Enhanced sampling methods using diffusers schedulers =====
    @torch.no_grad()
    def sample_with_scheduler(
        self,
        shape: tuple,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        device: Optional[str] = None,
        use_ema: bool = True,
        conditioner: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Enhanced sampling using diffusers scheduler for faster inference.
        
        Args:
            shape: Shape of samples to generate (B, C, H, W)
            num_inference_steps: Number of denoising steps (fewer = faster)
            guidance_scale: Guidance scale (1.0 = no guidance)
            generator: Random generator for reproducibility
            device: Device to run on
            use_ema: Whether to use EMA model
            conditioner: Optional conditioning tensor to pass to the model
            
        Returns:
            Generated samples in model domain [-1, 1]
        """
        if not self.use_diffusers_scheduler:
            # Fallback to manual sampling
            return self.ddpm_sample(shape[0], shape[1:], use_ema=use_ema)
        
        if device is None:
            device = next(self.model.parameters()).device
        
        # Initialize noise
        if generator is not None:
            sample = torch.randn(shape, generator=generator, device=device)
        else:
            sample = torch.randn(shape, device=device)
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Select model
        net = self.ema_model if (use_ema and getattr(self, 'use_ema', False)) else self.model
        net.eval()
        
        # Denoising loop
        for t in timesteps:
            # Expand timestep for batch
            t_batch = t.expand(shape[0])
            
            # Model prediction with mixed precision
            if self.mixed_precision and torch.cuda.is_available():
                with autocast(dtype=self.autocast_dtype):
                    try:
                        model_output = net(sample, t_batch, cond=conditioner)
                    except TypeError:
                        model_output = net(sample, t_batch)
            else:
                try:
                    model_output = net(sample, t_batch, cond=conditioner)
                except TypeError:
                    model_output = net(sample, t_batch)
            
            # Scheduler step
            sample = self.scheduler.step(model_output, t, sample).prev_sample
        
        return sample
    
    @torch.no_grad()
    def fast_sample(
        self,
        shape: tuple,
        num_inference_steps: int = 25,
        device: Optional[str] = None,
        use_ema: bool = True,
        conditioner: Optional[torch.Tensor] = None,
        use_karras_sigmas: Optional[bool] = None,
    ) -> torch.Tensor:
        """Ultra-fast sampling using DPM-Solver++ for quick inference.
        
        Args:
            shape: Shape of samples to generate (B, C, H, W)
            num_inference_steps: Number of steps (25 is often sufficient)
            device: Device to run on
            use_ema: Whether to use EMA model
            conditioner: Optional conditioning tensor to pass to the model
            use_karras_sigmas: Whether to use Karras sigmas (defaults to config)
            
        Returns:
            Generated samples in model domain [-1, 1]
        """
        if not DIFFUSERS_SCHEDULERS_AVAILABLE:
            print("⚠️ Diffusers not available, using standard sampling")
            return self.ddpm_sample(shape[0], shape[1:], use_ema=use_ema)
        
        if device is None:
            device = next(self.model.parameters()).device
        
        # Create fast scheduler
        if use_karras_sigmas is None:
            use_karras_sigmas = bool(self.config.get("use_karras_sigmas", True))
        scheduler_config = {
            "num_train_timesteps": self.num_timesteps,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "epsilon",
            "algorithm_type": "dpmsolver++",
            "solver_order": 2,
            "use_karras_sigmas": use_karras_sigmas,
        }
        
        fast_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        
        # Initialize noise
        sample = torch.randn(shape, device=device)
        
        # Set timesteps
        fast_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = fast_scheduler.timesteps
        
        # Select model
        net = self.ema_model if (use_ema and getattr(self, 'use_ema', False)) else self.model
        net.eval()
        
        # Fast denoising loop
        for t in timesteps:
            t_batch = t.expand(shape[0])
            
            # Model prediction with mixed precision
            if self.mixed_precision and torch.cuda.is_available():
                with autocast(dtype=self.autocast_dtype):
                    try:
                        model_output = net(sample, t_batch, cond=conditioner)
                    except TypeError:
                        model_output = net(sample, t_batch)
            else:
                try:
                    model_output = net(sample, t_batch, cond=conditioner)
                except TypeError:
                    model_output = net(sample, t_batch)
            
            # Fast scheduler step
            sample = fast_scheduler.step(model_output, t, sample).prev_sample
        
        return sample
    
    def get_scheduler_info(self) -> Dict[str, Any]:
        """Get information about the current scheduler setup."""
        info = {
            "using_diffusers_scheduler": self.use_diffusers_scheduler,
            "num_timesteps": self.num_timesteps,
            "beta_schedule": self.beta_schedule,
        }
        
        if self.use_diffusers_scheduler and hasattr(self, 'scheduler'):
            info.update({
                "scheduler_type": type(self.scheduler).__name__,
                "scheduler_config": self.scheduler.config if hasattr(self.scheduler, 'config') else None,
            })
        
        return info

    @torch.no_grad()
    def sample_with_scheduler_and_guidance(
        self,
        y: torch.Tensor,
        forward_model: Any,
        guidance_strategy: Any,
        schedule: Any,
        transform: Any,
        num_inference_steps: int = 25,
        device: Optional[str] = None,
        use_ema: bool = True,
        use_karras_sigmas: Optional[bool] = None,
    ) -> torch.Tensor:
        """Fast physics-guided sampling using a diffusers scheduler (DPM-Solver++).

        Applies physics guidance in the x0 domain each step, then maps back to epsilon
        to drive the scheduler update.
        """
        if not DIFFUSERS_SCHEDULERS_AVAILABLE:
            # Fallback to standard guided DDPM if diffusers not available
            return self.ddpm_guided_sample(
                y=y,
                forward_model=forward_model,
                guidance_strategy=guidance_strategy,
                schedule=schedule,
                transform=transform,
                num_steps=self.num_timesteps,
                use_ema=use_ema,
            )
        if device is None:
            device = next(self.model.parameters()).device
        y = y.to(device)

        # Build scheduler (DPMSolver++)
        if use_karras_sigmas is None:
            use_karras_sigmas = bool(self.config.get("use_karras_sigmas", True))
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=self.num_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2" if self.beta_schedule == "cosine" else self.beta_schedule,
            prediction_type="epsilon",
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=use_karras_sigmas,
        )
        scheduler.set_timesteps(num_inference_steps, device=device)
        # Initialize begin index defensively to avoid OOB in final step
        try:
            scheduler.set_begin_index(0)
        except Exception:
            pass
        timesteps = scheduler.timesteps

        # Initialize x_T ~ N(0, I)
        B, C, H, W = y.shape
        x_t = torch.randn((B, C, H, W), device=device)

        # Choose network
        net = self.ema_model if (use_ema and getattr(self, 'use_ema', False)) else self.model
        net.eval()

        # Use a safe view of timesteps to avoid step_index overflow at the end
        safe_timesteps = timesteps[:-1] if len(timesteps) > 1 else timesteps
        for t in safe_timesteps:
            t_batch = t.expand(B)

            # Predict epsilon
            if self.mixed_precision and torch.cuda.is_available():
                with autocast(dtype=self.autocast_dtype):
                    eps_pred = net(x_t, t_batch)
            else:
                eps_pred = net(x_t, t_batch)

            # Convert to x0 estimate
            alpha_t = self._extract(self.alphas_cumprod, t_batch, x_t.shape)
            sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)
            x0_hat_model = (x_t - sqrt_one_minus_alpha_t * eps_pred) / sqrt_alpha_t
            x0_hat_model = torch.clamp(x0_hat_model, -1.0, 1.0)

            # Guidance in intensity domain
            try:
                x0_int = self._model_to_intensity(x0_hat_model, transform)
                grad = guidance_strategy.compute_gradient(x0_int, y, forward_model, int(t.item())) if (forward_model is not None and guidance_strategy is not None) else None
                if grad is not None:
                    lambda_t = schedule.get_lambda_t(grad, int(t.item())) if schedule is not None else 0.0
                    x0_int_guided = torch.clamp(x0_int - float(lambda_t) * grad, min=0)
                    x0_hat_model = self._intensity_to_model(x0_int_guided, transform)
            except Exception:
                # If guidance fails, proceed without modification
                pass

            # Map guided x0 back to epsilon for scheduler step
            eps_guided = (x_t - sqrt_alpha_t * x0_hat_model) / sqrt_one_minus_alpha_t

            # Scheduler update
            x_t = scheduler.step(eps_guided, t, x_t).prev_sample

        return x_t
    
    @torch.no_grad()
    def inference(
        self, 
        x_source: torch.Tensor, 
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        use_ema: bool = True,
        return_16bit: bool = False,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Generate target-like image from source input (unified inference method).
        
        Args:
            x_source: Input source image [B, C, H, W] (e.g., WF, low-quality)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for conditioning
            use_ema: Whether to use EMA model
            return_16bit: If True, return 16-bit numpy array instead of tensor
            save_path: Optional path to save 16-bit TIFF output
            
        Returns:
            Generated target-like image [B, C, H, W] (e.g., 2P, high-quality)
            If return_16bit=True, returns 16-bit numpy array
        """
        model = self.ema_model if (use_ema and hasattr(self, 'ema_model')) else self.model
        model.eval()
        
        batch_size = x_source.shape[0]
        device = x_source.device
        
        # Start from random noise
        x_t = torch.randn_like(x_source)
        
        # Use diffusers scheduler if available for better sampling
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
        else:
            # Fallback to linear schedule
            timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = t.expand(batch_size) if t.dim() == 0 else t
            
            # Predict noise
            try:
                noise_pred = model(x_t, t_batch, cond=x_source)
            except TypeError:
                noise_pred = model(x_t, t_batch)
            
            # Apply guidance if specified
            if guidance_scale != 1.0:
                noise_pred_uncond = model(x_t, t_batch)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            
            # Denoising step
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                x_t = self.scheduler.step(noise_pred, t, x_t).prev_sample
            else:
                # Manual denoising step
                alpha_t = self.alphas_cumprod[t_batch].view(-1, 1, 1, 1)
                alpha_t_prev = self.alphas_cumprod[torch.clamp(t_batch - self.num_timesteps // num_inference_steps, 0)].view(-1, 1, 1, 1)
                
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                
                x_0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
                
                if i < len(timesteps) - 1:
                    sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
                    sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)
                    x_t = sqrt_alpha_t_prev * x_0_pred + sqrt_one_minus_alpha_t_prev * torch.randn_like(x_t)
                else:
                    x_t = x_0_pred
        
        # Ensure output is in valid range
        x_t = torch.clamp(x_t, -1, 1)
        
        # Handle 16-bit output if requested
        if return_16bit or save_path:
            # Convert to 16-bit using the transform
            from ..utils.image_processing import to_uint16_grayscale, save_16bit_grayscale
            output_16bit = to_uint16_grayscale(x_t, transform=self.transform)
            
            if save_path:
                # Save as 16-bit TIFF
                save_16bit_grayscale(x_t, save_path, transform=self.transform)
            
            if return_16bit:
                return output_16bit
        
        return x_t
    
    def get_adaptive_batch_config(
        self, 
        input_shape: tuple, 
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """Get recommended adaptive batch configuration for this model.
        
        Args:
            input_shape: Input shape (C, H, W) without batch dimension
            device: Device to test on
            
        Returns:
            Dictionary with recommended batch size and settings
        """
        try:
            from ..utils.utils import AdaptiveBatchSizer
            
            batch_sizer = AdaptiveBatchSizer(verbose=True)
            config = batch_sizer.get_recommended_config(
                self.model, input_shape, device
            )
            
            # Add current model settings
            config.update({
                "current_mixed_precision": self.mixed_precision,
                "current_autocast_dtype": str(self.autocast_dtype),
                "using_diffusers_scheduler": self.use_diffusers_scheduler,
            })
            
            return config
            
        except ImportError:
            # Fallback if adaptive batch module not available
            return {
                "batch_size": 4,
                "mixed_precision": self.mixed_precision,
                "gradient_checkpointing": False,
                "reason": "Adaptive batch sizing not available",
            }
    
    def backward_with_scaling(self, loss, optimizer):
        """Perform backward pass with gradient scaling for mixed precision."""
        if self.mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def optimizer_step_with_scaling(self, optimizer):
        """Perform optimizer step with gradient scaling and unscaling."""
        if self.mixed_precision and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def get_mixed_precision_info(self) -> Dict[str, Any]:
        """Get information about mixed precision settings."""
        return {
            "mixed_precision_enabled": self.mixed_precision,
            "autocast_dtype": str(self.autocast_dtype),
            "scaler_enabled": self.scaler is not None,
            "optimized_attention_enabled": getattr(self, 'enable_optimized_attention', False),
            "cuda_streams_enabled": getattr(self, 'enable_cuda_streams', False),
            "gpu_capability": torch.cuda.get_device_capability() if torch.cuda.is_available() else None,
        }
    
    def async_training_step(self, batch, batch_idx):
        """Async training step using CUDA streams for better GPU utilization."""
        if not self.enable_cuda_streams or not torch.cuda.is_available():
            # Fallback to regular training step
            return self.training_step(batch, batch_idx)
        
        # Expect (target_2p, conditioner_wf)
        x_0, c_wf = batch
        b = x_0.shape[0]
        device = x_0.device
        
        # Use data stream for data preparation
        with torch.cuda.stream(self.data_stream):
            t = torch.randint(0, self.num_timesteps, (b,), device=device)
            noise = torch.randn_like(x_0)
            x_t = self.q_sample(x_0, t, noise)
        
        # Synchronize streams before computation
        self.compute_stream.wait_stream(self.data_stream)
        
        # Use compute stream for model forward pass
        with torch.cuda.stream(self.compute_stream):
            use_conditioning = bool(self.config.get("use_conditioning", True))
            
            if self.mixed_precision:
                with autocast(dtype=self.autocast_dtype):
                    if use_conditioning:
                        try:
                            noise_pred = self.model(x_t, t, cond=c_wf)
                        except TypeError:
                            noise_pred = self.model(x_t, t)
                    else:
                        noise_pred = self.model(x_t, t)
                    loss = F.mse_loss(noise_pred, noise)
                    
                    # Optional supervised x0 loss
                    if self.config.get("supervised_x0_weight", 0.0) > 0:
                        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
                        sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)
                        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)
                        x0_hat = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
                        loss_x0 = F.l1_loss(x0_hat, x_0)
                        loss = loss + float(self.config.get("supervised_x0_weight", 0.0)) * loss_x0
            else:
                # Standard precision training
                if use_conditioning:
                    try:
                        noise_pred = self.model(x_t, t, cond=c_wf)
                    except TypeError:
                        noise_pred = self.model(x_t, t)
                else:
                    noise_pred = self.model(x_t, t)
                loss = F.mse_loss(noise_pred, noise)
        
        # Synchronize compute stream before returning
        torch.cuda.current_stream().wait_stream(self.compute_stream)
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True)
        
        return loss
    
    def parallel_inference(self, batch_list: list, num_parallel: int = None) -> list:
        """Run inference on multiple batches in parallel using CUDA streams."""
        if not torch.cuda.is_available() or not self.enable_cuda_streams:
            # Fallback to sequential processing
            return [self._single_inference(batch) for batch in batch_list]
        
        if num_parallel is None:
            # Auto-detect based on GPU memory
            num_parallel = min(4, len(batch_list))
        
        # Create multiple streams for parallel processing
        streams = [torch.cuda.Stream() for _ in range(num_parallel)]
        results = [None] * len(batch_list)
        
        # Process batches in parallel
        for i, batch in enumerate(batch_list):
            stream_idx = i % num_parallel
            with torch.cuda.stream(streams[stream_idx]):
                results[i] = self._single_inference(batch)
        
        # Synchronize all streams
        for stream in streams:
            stream.synchronize()
        
        return results
    
    def _single_inference(self, batch):
        """Single inference forward pass."""
        with torch.no_grad():
            if self.mixed_precision:
                with autocast(dtype=self.autocast_dtype):
                    return self.model(batch)
            else:
                return self.model(batch)
    
    def get_training_mode_info(self) -> Dict[str, Any]:
        """Get information about the current training mode and configuration."""
        info = {
            "self_supervised": self.self_supervised,
            "forward_model_type": getattr(self, 'forward_model_type', None),
            "has_forward_model": self.forward_model is not None,
        }
        
        if self.self_supervised:
            info.update({
                "ddpm_weight": getattr(self, 'ddpm_weight', 1.0),
                "cycle_weight": getattr(self, 'cycle_weight', 0.1),
                "perceptual_weight": getattr(self, 'perceptual_weight', 0.01),
                "has_perceptual_loss": hasattr(self, 'perceptual_loss') and self.perceptual_loss is not None,
                "cycle_loss_type": getattr(self.cycle_loss, 'loss_type', 'l1') if hasattr(self, 'cycle_loss') else None,
            })
        
        return info
    
    def _compute_frequency_losses(self, batch) -> torch.Tensor:
        """Compute frequency domain losses."""
        if isinstance(batch, (list, tuple)):
            x_0, c_wf = batch
        else:
            x_0 = batch
            c_wf = None
        
        total_freq_loss = 0.0
        
        # Generate prediction for frequency analysis
        with torch.no_grad():
            # Sample timestep and noise
            t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device)
            noise = torch.randn_like(x_0)
            x_t = self.q_sample(x_0, t, noise)
            
            # Get prediction
            if self.mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
                    noise_pred = self.model(x_t, t, cond=c_wf) if c_wf is not None else self.model(x_t, t)
            else:
                noise_pred = self.model(x_t, t, cond=c_wf) if c_wf is not None else self.model(x_t, t)
            
            # Predict x0
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)
            x0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        # Multi-scale frequency loss
        if hasattr(self, 'frequency_loss'):
            freq_loss = self.frequency_loss(x0_pred, x_0)
            total_freq_loss += self.frequency_loss_weight * freq_loss
            self._log_if_trainer("train/frequency_loss", freq_loss)
        
        # High-frequency preservation loss
        if hasattr(self, 'high_freq_loss'):
            high_freq_loss = self.high_freq_loss(x0_pred, x_0)
            total_freq_loss += self.high_freq_loss_weight * high_freq_loss
            self._log_if_trainer("train/high_freq_loss", high_freq_loss)
        
        return total_freq_loss
    
    def _monitor_memory_usage(self):
        """Monitor memory usage and update stats."""
        if not torch.cuda.is_available():
            return
        
        memory_info = {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        
        if hasattr(self, 'performance_stats'):
            self.performance_stats["memory_usage"].append(memory_info)
            
            # Keep only recent history
            if len(self.performance_stats["memory_usage"]) > 1000:
                self.performance_stats["memory_usage"] = self.performance_stats["memory_usage"][-1000:]
        
        # Log memory stats
        self._log_if_trainer("memory/allocated_gb", memory_info["allocated_gb"])
        self._log_if_trainer("memory/reserved_gb", memory_info["reserved_gb"])
    
    def _check_batch_size_adjustment(self, step: int):
        """Check if batch size needs adjustment."""
        if not hasattr(self, 'batch_sizer'):
            return
        
        # Get current batch size (from dataloader or config)
        current_batch_size = self.config.get("training", {}).get("batch_size", 8)
        
        # Check for adjustment
        loss_val = self.performance_stats["loss_improvements"][-1] if self.performance_stats["loss_improvements"] else None
        new_batch_size, adjustment_info = self.batch_sizer.adjust_batch_size_dynamically(
            current_batch_size=current_batch_size,
            step=step,
            loss=loss_val,
        )
        
        if adjustment_info["adjusted"]:
            self.performance_stats["batch_adjustments"] += 1
            print(f"🔄 Batch size adjusted at step {step}: {adjustment_info['old_batch_size']} → {adjustment_info['new_batch_size']}")
            
            # Update config (for logging)
            if "training" not in self.config:
                self.config["training"] = {}
            self.config["training"]["batch_size"] = new_batch_size
    
    def _log_advanced_metrics(self, batch_idx: int, loss: torch.Tensor):
        """Log advanced feature metrics."""
        if not hasattr(self, 'performance_stats'):
            return
        
        # Memory metrics
        if self.enable_memory_optimization and batch_idx % 100 == 0:
            if hasattr(self, 'batch_sizer'):
                summary = self.batch_sizer.get_performance_summary()
                self._log_if_trainer("advanced/current_batch_size", summary.get("current_batch_size", 0))
                self._log_if_trainer("advanced/memory_pressure", summary.get("current_memory_pressure", 0))
        
        # Performance metrics
        self._log_if_trainer("advanced/total_batch_adjustments", self.performance_stats["batch_adjustments"])
        self._log_if_trainer("advanced/scheduler_transitions", self.performance_stats["scheduler_transitions"])
    
    def get_advanced_features_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of advanced features and performance."""
        if not SOTA_FEATURES_AVAILABLE or not hasattr(self, 'performance_stats'):
            return {"advanced_features_available": False}
        
        summary = {
            "advanced_features_available": True,
            "advanced_features_enabled": {
                "frequency_losses": self.enable_frequency_losses,
                "advanced_schedulers": self.enable_advanced_schedulers,
                "memory_optimization": self.enable_memory_optimization,
                "cascaded_sampling": self.enable_cascaded_sampling,
            },
            "performance_stats": self.performance_stats.copy() if hasattr(self, 'performance_stats') else {},
        }
        
        # Add memory optimization summary
        if hasattr(self, 'batch_sizer'):
            summary["memory_optimization"] = self.batch_sizer.get_performance_summary()
        
        return summary
    
    @torch.no_grad()
    def enhanced_inference(
        self,
        x_source: torch.Tensor,
        inference_type: str = "cascaded",
        num_inference_steps: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """Enhanced inference with multiple advanced sampling strategies.
        
        Args:
            x_source: Source input
            inference_type: Type of inference ('cascaded', 'standard')
            num_inference_steps: Number of inference steps
            **kwargs: Additional arguments
            
        Returns:
            Generated output
        """
        if inference_type == "cascaded" and hasattr(self, 'cascaded_sampler') and self.cascaded_sampler is not None:
            # Use cascaded sampling
            shape = x_source.shape
            return self.cascaded_sampler.sample_cascaded(
                shape=shape,
                num_inference_steps=num_inference_steps,
                **kwargs
            )
        else:
            # Standard inference with current scheduler
            return self.inference(
                x_source=x_source,
                num_inference_steps=num_inference_steps,
                **kwargs
            )


# Legacy compatibility - keep SelfSupervisedDDPMTrainer as alias
SelfSupervisedDDPMTrainer = DDPMTrainer


# ===== 16-bit Output Utilities (moved to utils.image_processing) =====


# ===== Enhanced Factory Functions =====

def create_enhanced_trainer(
    model: nn.Module,
    config: Dict[str, Any],
    transform: Optional[Any] = None,
    forward_model: Optional[Any] = None,
    enable_all_features: bool = True,
) -> DDPMTrainer:
    """Factory function to create DDPM trainer with advanced features.
    
    Args:
        model: Model to train
        config: Training configuration
        transform: Data transform
        forward_model: Forward model
        enable_all_features: Enable all SOTA features
        
    Returns:
        Configured DDPM trainer with advanced features
    """
    if enable_all_features and SOTA_FEATURES_AVAILABLE:
        config = create_enhanced_config(config)
    
    return DDPMTrainer(
        model=model,
        config=config,
        transform=transform,
        forward_model=forward_model,
    )


def create_enhanced_config(
    base_config: Optional[Dict[str, Any]] = None,
    max_resolution: int = 512,
    enable_all_features: bool = True,
) -> Dict[str, Any]:
    """Create comprehensive enhanced configuration.
    
    Args:
        base_config: Base configuration to extend
        max_resolution: Maximum training resolution
        enable_all_features: Enable all SOTA features
        
    Returns:
        Complete enhanced configuration
    """
    if base_config is None:
        base_config = {}
    
    enhanced_config = base_config.copy()
    
    if enable_all_features and SOTA_FEATURES_AVAILABLE:
        # Advanced features
        enhanced_config["advanced_features"] = {
            "enable_frequency_losses": True,
            "enable_advanced_schedulers": True,
            "enable_memory_optimization": True,
            "enable_cascaded_sampling": True,
            "frequency_loss_weight": 0.1,
            
            # Frequency losses config
            "frequency_losses": {
                "use_fourier": True,
                "use_spectral": True,
                "use_wavelet": False,
                "fourier_weight": 1.0,
                "spectral_weight": 0.5,
                "high_freq_loss_weight": 0.05,
            },
            
            # Scheduler config
            "scheduler": {
                "type": "improved_cosine",
                "params": {
                    "s": 0.008,
                    "cosine_power": 1.0,
                    "max_beta": 0.999,
                },
            },
            
            # Memory optimization
            "memory_optimization": {
                "enable_dynamic_scaling": True,
                "memory_pressure_threshold": 0.85,
                "safety_factor": 0.8,
                "enable_monitoring": True,
            },
            
            # Cascaded sampling
            "cascaded_sampling": {
                "enabled": True,
                "type": "basic",
                "resolution_schedule": [64, 128, 256, max_resolution],
                "params": {
                    "consistency_weight": 0.1,
                    "upsampling_method": "bilinear",
                    "memory_efficient": True,
                },
            },
        }
    
    return enhanced_config


def get_advanced_feature_status() -> Dict[str, bool]:
    """Get status of all advanced features implementation.
    
    Returns:
        Dictionary with feature implementation status
    """
    return {
        "advanced_features_available": SOTA_FEATURES_AVAILABLE,
        "frequency_domain_loss": SOTA_FEATURES_AVAILABLE,
        "advanced_schedulers": SOTA_FEATURES_AVAILABLE,
        "memory_optimization": SOTA_FEATURES_AVAILABLE,
        "cascaded_sampling": SOTA_FEATURES_AVAILABLE,
    }


def print_advanced_features_summary():
    """Print comprehensive summary of advanced features implementation."""
    print("🚀 Advanced DDPM Features Summary")
    print("=" * 50)
    
    features = get_advanced_feature_status()
    for feature, implemented in features.items():
        status = "✅" if implemented else "❌"
        print(f"{status} {feature.replace('_', ' ').title()}")
    
    if SOTA_FEATURES_AVAILABLE:
        print("\n🔥 All advanced features successfully integrated into DDPMTrainer!")
        print("\nKey Components:")
        print("• Advanced frequency domain losses")
        print("• Advanced noise schedulers (beyond diffusers)")
        print("• Adaptive batch sizing with memory optimization")
        print("• Multi-resolution cascaded sampling")
        print("• Comprehensive performance monitoring")
        print("• Memory-efficient training and inference")
        
        print("\n💡 Usage:")
        print("```python")
        print("from pkl_dg.models.diffusion import create_enhanced_trainer, create_enhanced_config")
        print("")
        print("# Create enhanced configuration")
        print("config = create_enhanced_config(max_resolution=512)")
        print("")
        print("# Create enhanced trainer")
        print("trainer = create_enhanced_trainer(model, config)")
        print("")
        print("# Train with all advanced features")
        print("trainer.fit(dataloader)")
        print("```")
    else:
        print("\n⚠️ Advanced features not available - missing dependencies")
        print("Install with: pip install -e .")


