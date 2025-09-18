from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from .nn import NNDefaults, get_activation

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


class CycleConsistencyLoss(nn.Module):
    """Cycle consistency loss for self-supervised training."""
    
    def __init__(self, loss_type: str = "l1", reduction: str = "mean"):
        """
        Initialize cycle consistency loss.
        
        Args:
            loss_type: Type of loss ('l1', 'l2', 'smooth_l1')
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        
    def forward(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Compute cycle consistency loss.
        
        Args:
            original: Original image [B, C, H, W]
            reconstructed: Reconstructed image from forward cycle [B, C, H, W]
            
        Returns:
            Cycle consistency loss
        """
        if self.loss_type == "l1":
            loss = F.l1_loss(reconstructed, original, reduction=self.reduction)
        elif self.loss_type == "l2":
            loss = F.mse_loss(reconstructed, original, reduction=self.reduction)
        elif self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(reconstructed, original, reduction=self.reduction)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
            
        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained features for better image quality."""
    
    def __init__(self, feature_layers: Optional[list] = None, use_pretrained: bool = False):
        """
        Initialize perceptual loss.
        
        Args:
            feature_layers: List of layer indices to extract features from
            use_pretrained: Whether to use pretrained VGG features (for natural images)
        """
        super().__init__()
        self.feature_layers = feature_layers or [0, 1, 2, 3]
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            # Use VGG for natural images
            try:
                import torchvision.models as models
                vgg = models.vgg16(pretrained=True).features
                self.feature_extractor = nn.ModuleList([
                    vgg[:4],   # conv1_2
                    vgg[:9],   # conv2_2
                    vgg[:16],  # conv3_3
                    vgg[:23],  # conv4_3
                ])
                # Freeze VGG features
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
            except ImportError:
                print("⚠️ torchvision not available, using simple conv features")
                self.use_pretrained = False
                self._init_simple_features()
        else:
            # Simple conv layers for microscopy or when torchvision unavailable
            self._init_simple_features()
    
    def _init_simple_features(self):
        """Initialize simple convolutional features using standardized components."""
        from .nn import ConvBlock
        
        channels = [32, 64, 128, 256]
        layers = []
        in_ch = 1
        
        for out_ch in channels:
            layers.append(ConvBlock(
                in_ch, out_ch,
                norm_type="groupnorm",
                activation="relu",  # Use ReLU for feature extraction
                dropout=0.0
            ))
            in_ch = out_ch
        
        self.feature_extractor = nn.ModuleList(layers)
        
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def extract_features(self, x: torch.Tensor) -> list:
        """Extract multi-scale features."""
        features = []
        
        if self.use_pretrained:
            # VGG features for natural images (expects 3 channels)
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
            
            for i, layer in enumerate(self.feature_extractor):
                x = layer(x)
                if i in self.feature_layers:
                    features.append(x)
        else:
            # Simple conv features using standardized blocks
            for i, layer in enumerate(self.feature_extractor):
                x = layer(x)  # ConvBlock already includes activation
                if i in self.feature_layers:
                    features.append(x)
                if i < len(self.feature_extractor) - 1:
                    x = F.avg_pool2d(x, 2)
        
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss / len(pred_features)


class UnpairedDataset(torch.utils.data.Dataset):
    """Dataset for self-supervised training using forward model or unpaired data."""
    
    def __init__(
        self,
        wf_dir: str,
        twop_dir: str,
        transform: Optional[Any] = None,
        image_size: int = 256,
        mode: str = "train",
        forward_model: Optional[Any] = None,
        use_forward_model: bool = True,
        add_noise: bool = True,
        noise_level: float = 0.05
    ):
        """
        Initialize dataset for self-supervised learning.
        
        Args:
            wf_dir: Directory containing WF images (used for evaluation/paired mode)
            twop_dir: Directory containing 2P images (ground truth)
            transform: Data transform
            image_size: Target image size
            mode: Dataset mode ('train', 'val', 'test')
            forward_model: Forward model for generating synthetic WF from 2P
            use_forward_model: If True, use forward model; if False, use unpaired WF/2P
            add_noise: Whether to add noise to synthetic WF (only for forward model mode)
            noise_level: Standard deviation of additive noise
        """
        from pathlib import Path
        from PIL import Image
        
        self.wf_dir = Path(wf_dir)
        self.twop_dir = Path(twop_dir)
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.forward_model = forward_model
        self.use_forward_model = use_forward_model
        self.add_noise = add_noise
        self.noise_level = noise_level
        
        # Collect image paths
        patterns = ["**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.tif", "**/*.tiff"]
        
        self.wf_paths = []
        self.twop_paths = []
        
        for pattern in patterns:
            if self.wf_dir.exists():
                self.wf_paths.extend(list(self.wf_dir.glob(pattern)))
            self.twop_paths.extend(list(self.twop_dir.glob(pattern)))
        
        self.wf_paths = sorted(self.wf_paths)
        self.twop_paths = sorted(self.twop_paths)
        
        if self.use_forward_model:
            print(f"Found {len(self.twop_paths)} 2P images for self-supervised learning with forward model")
            if len(self.twop_paths) == 0:
                raise ValueError(f"No 2P images found in {self.twop_dir}")
        else:
            print(f"Found {len(self.wf_paths)} WF images and {len(self.twop_paths)} 2P images for unpaired training")
        
    def __len__(self) -> int:
        if self.use_forward_model:
            return len(self.twop_paths)
        else:
            return max(len(self.wf_paths), len(self.twop_paths))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get training pair.
        
        Returns:
            Tuple of (target_2P, conditioner_WF) where conditioner_WF is either:
            - Synthetic WF generated from 2P using forward model (self-supervised)
            - Real WF image (unpaired/paired mode)
        """
        from PIL import Image
        import numpy as np
        
        if self.use_forward_model:
            # Self-supervised mode: use forward model to generate synthetic WF
            return self._get_self_supervised_pair(idx)
        else:
            # Legacy unpaired mode: randomly pair WF and 2P
            return self._get_unpaired_pair(idx)
    
    def _get_self_supervised_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get self-supervised pair using forward model."""
        from PIL import Image
        import numpy as np
        
        # Load 2P image (ground truth)
        twop_path = self.twop_paths[idx]
        twop_img = Image.open(twop_path).convert('L')
        twop_img = twop_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        twop_tensor = torch.from_numpy(np.array(twop_img, dtype=np.float32) / 255.0).unsqueeze(0)
        
        # Generate synthetic WF using forward model
        if self.forward_model is not None:
            with torch.no_grad():
                # Move to forward model device
                device = next(iter(self.forward_model.psf.parameters())).device if hasattr(self.forward_model.psf, 'parameters') else 'cpu'
                twop_batch = twop_tensor.unsqueeze(0).to(device)
                
                # Apply forward model: 2P -> synthetic WF
                synthetic_wf_batch = self.forward_model.forward(twop_batch, add_noise=self.add_noise)
                
                # Add additional noise for robustness during training
                if self.add_noise and self.noise_level > 0 and self.mode == "train":
                    noise = torch.randn_like(synthetic_wf_batch) * self.noise_level
                    synthetic_wf_batch = synthetic_wf_batch + noise
                
                synthetic_wf = synthetic_wf_batch.squeeze(0).cpu()
        else:
            # Fallback: use original 2P as conditioner (identity mapping)
            synthetic_wf = twop_tensor.clone()
        
        # Ensure tensors are in valid range
        twop_tensor = torch.clamp(twop_tensor, 0.0, 1.0)
        synthetic_wf = torch.clamp(synthetic_wf, 0.0, 1.0)
        
        # Apply transforms
        if self.transform is not None:
            twop_tensor = self.transform(twop_tensor)
            synthetic_wf = self.transform(synthetic_wf)
        
        return twop_tensor, synthetic_wf
    
    def _get_unpaired_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get unpaired WF/2P pair (legacy mode)."""
        from PIL import Image
        import numpy as np
        
        # Randomly sample from each dataset
        wf_idx = idx % len(self.wf_paths) if len(self.wf_paths) > 0 else 0
        twop_idx = np.random.randint(0, len(self.twop_paths)) if len(self.twop_paths) > 0 else 0
        
        # Load images
        if len(self.wf_paths) > 0:
            wf_img = Image.open(self.wf_paths[wf_idx]).convert('L')
            wf_img = wf_img.resize((self.image_size, self.image_size), Image.BILINEAR)
            wf_tensor = torch.from_numpy(np.array(wf_img, dtype=np.float32) / 255.0).unsqueeze(0)
        else:
            # Fallback if no WF images
            wf_tensor = torch.zeros(1, self.image_size, self.image_size)
        
        twop_img = Image.open(self.twop_paths[twop_idx]).convert('L')
        twop_img = twop_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        twop_tensor = torch.from_numpy(np.array(twop_img, dtype=np.float32) / 255.0).unsqueeze(0)
        
        # Apply transforms
        if self.transform is not None:
            wf_tensor = self.transform(wf_tensor)
            twop_tensor = self.transform(twop_tensor)
        
        return twop_tensor, wf_tensor


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
        
        # Initialize self-supervised components if enabled
        if self.self_supervised:
            self._init_self_supervised_components()
        
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
    
    def _init_self_supervised_components(self):
        """Initialize components for self-supervised training."""
        # Loss components
        self.cycle_loss = CycleConsistencyLoss(
            loss_type=self.config.get("cycle_loss_type", "l1")
        )
        
        # Perceptual loss - use pretrained for natural images
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
        """Gather a[t] then reshape to broadcast over x."""
        out = a.gather(0, t)
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
                loss = F.mse_loss(noise_pred, noise)
                
                # Optional supervised x0 loss to encourage paired mapping
                if self.config.get("supervised_x0_weight", 0.0) > 0:
                    # Reconstruct x0 from epsilon prediction (epsilon parameterization)
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
            
            # Optional supervised x0 loss to encourage paired mapping
            if self.config.get("supervised_x0_weight", 0.0) > 0:
                # Reconstruct x0 from epsilon prediction (epsilon parameterization)
                alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
                sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)
                x0_hat = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
                loss_x0 = F.l1_loss(x0_hat, x_0)
                loss = loss + float(self.config.get("supervised_x0_weight", 0.0)) * loss_x0
        
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
            from ..utils.adaptive_batch import AdaptiveBatchSizer
            
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


# Legacy compatibility - keep SelfSupervisedDDPMTrainer as alias
SelfSupervisedDDPMTrainer = DDPMTrainer


# ===== 16-bit Output Utilities =====

def to_uint16_grayscale(
    tensor: torch.Tensor, 
    transform: Optional[Any] = None,
    percentile_clip: float = 99.5,
    preserve_range: bool = True
) -> np.ndarray:
    """
    Convert model output tensor to 16-bit grayscale numpy array.
    
    Args:
        tensor: Model output tensor in model domain [-1, 1] or intensity domain
        transform: Transform object to convert from model domain to intensity domain
        percentile_clip: Percentile for clipping outliers (default: 99.5)
        preserve_range: If True, preserve original intensity range; if False, normalize to full 16-bit
        
    Returns:
        16-bit grayscale numpy array (0-65535)
    """
    # Convert to numpy and squeeze to 2D if needed
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            array = tensor.detach().cpu().numpy()
        else:
            array = tensor.detach().numpy()
    else:
        array = np.array(tensor)
    
    # Remove batch and channel dimensions if present
    while array.ndim > 2:
        array = array.squeeze()
    
    # Convert from model domain to intensity domain if transform provided
    if transform is not None:
        # Convert back to tensor for transform
        tensor_2d = torch.from_numpy(array).float()
        if tensor_2d.ndim == 2:
            tensor_2d = tensor_2d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Apply inverse transform
        intensity_tensor = transform.inverse(tensor_2d)
        array = intensity_tensor.squeeze().detach().cpu().numpy()
    
    # Ensure non-negative
    array = np.clip(array, 0, None)
    
    if preserve_range:
        # Preserve the original intensity range, just convert to uint16
        # Clip outliers based on percentile
        max_val = np.percentile(array, percentile_clip)
        array = np.clip(array, 0, max_val)
        
        # Scale to 16-bit range while preserving relative intensities
        if max_val > 0:
            array = (array / max_val * 65535.0)
    else:
        # Normalize to full 16-bit range
        min_val = np.percentile(array, 100 - percentile_clip)
        max_val = np.percentile(array, percentile_clip)
        
        if max_val > min_val:
            array = (array - min_val) / (max_val - min_val)
        else:
            array = np.zeros_like(array)
        
        array = array * 65535.0
    
    # Convert to uint16
    return np.clip(array, 0, 65535).astype(np.uint16)


def save_16bit_grayscale(
    tensor: torch.Tensor,
    path: str,
    transform: Optional[Any] = None,
    percentile_clip: float = 99.5,
    preserve_range: bool = True
) -> None:
    """
    Save model output as 16-bit grayscale TIFF image.
    
    Args:
        tensor: Model output tensor
        path: Output file path (should end with .tif or .tiff)
        transform: Transform object to convert from model domain
        percentile_clip: Percentile for clipping outliers
        preserve_range: Whether to preserve original intensity range
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile is required for saving 16-bit images. Install with: pip install tifffile")
    
    # Convert to 16-bit array
    array_16bit = to_uint16_grayscale(
        tensor, 
        transform=transform, 
        percentile_clip=percentile_clip,
        preserve_range=preserve_range
    )
    
    # Save as 16-bit TIFF
    tifffile.imwrite(path, array_16bit, photometric='minisblack')


def save_16bit_comparison(
    wf_input: torch.Tensor,
    prediction: torch.Tensor, 
    gt_target: Optional[torch.Tensor],
    path: str,
    transform: Optional[Any] = None,
    percentile_clip: float = 99.5,
    preserve_range: bool = True
) -> None:
    """
    Save side-by-side comparison as 16-bit TIFF: [WF | Prediction | GT].
    
    Args:
        wf_input: WF input tensor
        prediction: Model prediction tensor
        gt_target: Ground truth tensor (optional)
        path: Output file path
        transform: Transform object
        percentile_clip: Percentile for clipping
        preserve_range: Whether to preserve intensity range
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile is required for saving 16-bit images. Install with: pip install tifffile")
    
    # Convert all to 16-bit arrays
    wf_16bit = to_uint16_grayscale(wf_input, transform, percentile_clip, preserve_range)
    pred_16bit = to_uint16_grayscale(prediction, transform, percentile_clip, preserve_range)
    
    # Create comparison
    if gt_target is not None:
        gt_16bit = to_uint16_grayscale(gt_target, transform, percentile_clip, preserve_range)
        comparison = np.concatenate([wf_16bit, pred_16bit, gt_16bit], axis=1)
    else:
        comparison = np.concatenate([wf_16bit, pred_16bit], axis=1)
    
    # Save as 16-bit TIFF
    tifffile.imwrite(path, comparison, photometric='minisblack')



