"""
Hydra Configuration Schemas for Evaluation

This module defines structured configuration schemas using Hydra's structured configs
to provide validation, type checking, and better IDE support.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from omegaconf import MISSING


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    device: str = "cuda"
    seed: int = 42
    name: str = "evaluation"
    output_dir: str = "outputs/evaluation"


@dataclass
class DataConfig:
    """Data configuration."""
    image_size: int = 256
    min_intensity: float = 0.0
    max_intensity: float = 1.0
    noise_model: str = "gaussian"  # gaussian, poisson, poisson_gaussian
    
    # Generalized Anscombe Transform parameters
    gat: Dict[str, float] = field(default_factory=lambda: {
        "alpha": 1.0,
        "mu": 0.0,
        "sigma": 0.0
    })


@dataclass
class ModelConfig:
    """Model configuration."""
    in_channels: int = 1
    out_channels: int = 1
    model_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: List[int] = field(default_factory=lambda: [16, 8])
    channel_mult: List[int] = field(default_factory=lambda: [1, 2, 2, 2])
    num_heads: int = 4
    use_scale_shift_norm: bool = True
    dropout: float = 0.0
    resblock_updown: bool = True
    use_new_attention_order: bool = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_timesteps: int = 1000
    use_conditioning: bool = False
    conditioning_type: str = "wf"  # wf (widefield) or other
    beta_schedule: str = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02


@dataclass
class PhysicsConfig:
    """Physics model configuration."""
    use_psf: bool = True
    psf_path: Optional[str] = None
    use_bead_psf: bool = False
    beads_dir: str = ""
    bead_mode: Optional[str] = None
    background: float = 0.0
    read_noise_sigma: float = 0.0


@dataclass
class PSFConfig:
    """PSF configuration for Gaussian PSF."""
    type: str = "gaussian"  # gaussian or file
    sigma_x: float = 2.0
    sigma_y: float = 2.0
    size: int = 21


@dataclass
class GuidanceConfig:
    """Guidance configuration."""
    epsilon: float = 1e-6
    lambda_base: float = 0.1
    
    # Schedule parameters
    schedule: Dict[str, Any] = field(default_factory=lambda: {
        "T_threshold": 800,
        "epsilon_lambda": 1e-3
    })


@dataclass
class InferenceConfig:
    """Inference configuration."""
    checkpoint_path: str = MISSING
    input_dir: str = MISSING
    gt_dir: str = MISSING
    mask_dir: str = ""
    output_dir: str = "outputs/evaluation"
    ddim_steps: int = 50
    eta: float = 0.0
    use_autocast: bool = True
    batch_size: int = 1


@dataclass
class MetricsConfig:
    """Metrics configuration."""
    # Which metrics to compute
    image_quality: List[str] = field(default_factory=lambda: ["psnr", "ssim", "frc"])
    perceptual: List[str] = field(default_factory=lambda: ["sar", "sharpness", "contrast"])
    downstream: List[str] = field(default_factory=lambda: ["cellpose_f1", "hausdorff_distance"])
    robustness: List[str] = field(default_factory=lambda: ["noise_robustness", "hallucination_score"])
    
    # Metric-specific parameters
    frc_threshold: float = 0.143
    noise_robustness_levels: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.1])
    
    # Output options
    save_individual_results: bool = True
    save_summary_plots: bool = True
    save_comparison_images: bool = True


@dataclass
class BaselineConfig:
    """Baseline methods configuration."""
    rcan_checkpoint: Optional[str] = None
    richardson_lucy_iterations: int = 30
    use_richardson_lucy: bool = True
    use_rcan: bool = False


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    mode: str = "online"  # online, offline, disabled
    project: str = "pkl-diffusion-evaluation"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    group: Optional[str] = None
    job_type: str = "evaluation"


@dataclass
class PathsConfig:
    """Paths configuration."""
    data: str = "data"
    checkpoints: str = "checkpoints"
    outputs: str = "outputs"
    logs: str = "logs"


@dataclass
class EvaluationConfig:
    """Main evaluation configuration."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    psf: PSFConfig = field(default_factory=PSFConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def validate_config(cfg: EvaluationConfig) -> None:
    """
    Validate the evaluation configuration.
    
    Args:
        cfg: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required paths exist
    if cfg.inference.checkpoint_path == MISSING:
        raise ValueError("inference.checkpoint_path is required")
    
    if cfg.inference.input_dir == MISSING:
        raise ValueError("inference.input_dir is required")
        
    if cfg.inference.gt_dir == MISSING:
        raise ValueError("inference.gt_dir is required")
    
    # Validate device
    if cfg.experiment.device not in ["cpu", "cuda", "auto"]:
        raise ValueError(f"Invalid device: {cfg.experiment.device}")
    
    # Validate noise model
    valid_noise_models = ["gaussian", "poisson", "poisson_gaussian"]
    if cfg.data.noise_model not in valid_noise_models:
        raise ValueError(f"Invalid noise_model: {cfg.data.noise_model}. Must be one of {valid_noise_models}")
    
    # Validate conditioning type
    valid_conditioning = ["wf", "none"]
    if cfg.training.conditioning_type not in valid_conditioning:
        raise ValueError(f"Invalid conditioning_type: {cfg.training.conditioning_type}")
    
    # Validate wandb mode
    valid_wandb_modes = ["online", "offline", "disabled"]
    if cfg.wandb.mode not in valid_wandb_modes:
        raise ValueError(f"Invalid wandb.mode: {cfg.wandb.mode}. Must be one of {valid_wandb_modes}")
    
    # Validate numeric ranges
    if cfg.inference.ddim_steps <= 0:
        raise ValueError("inference.ddim_steps must be positive")
    
    if cfg.inference.eta < 0 or cfg.inference.eta > 1:
        raise ValueError("inference.eta must be between 0 and 1")
    
    if cfg.training.num_timesteps <= 0:
        raise ValueError("training.num_timesteps must be positive")
    
    if cfg.guidance.lambda_base < 0:
        raise ValueError("guidance.lambda_base must be non-negative")
    
    # Validate metric lists contain valid metric names
    from ..metrics import list_metrics
    
    available_metrics = set(list_metrics())
    
    for category, metrics in [
        ("image_quality", cfg.metrics.image_quality),
        ("perceptual", cfg.metrics.perceptual), 
        ("downstream", cfg.metrics.downstream),
        ("robustness", cfg.metrics.robustness)
    ]:
        for metric in metrics:
            if metric not in available_metrics:
                raise ValueError(f"Unknown metric '{metric}' in metrics.{category}")
    
    print("✅ Configuration validation passed")
