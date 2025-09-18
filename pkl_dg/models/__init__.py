"""
Models package for PKL Diffusion Denoising.

This package provides a clean, organized structure for diffusion models:
- core: Essential DDPM training and sampling functionality
- unet: UNet architecture for noise prediction
- losses: Various loss functions for training
- schedulers: Noise scheduling strategies
- sampling: Different sampling methods (DDIM, DDPM, etc.)
- nn: Neural network building blocks
- progressive: Progressive training capabilities
- advanced: Advanced features (cascaded sampling, hierarchical training, etc.)
- registry: Component registry system for easy extension

Features:
- Registry-based component system following HuggingFace patterns
- Automatic component discovery and registration
- Type-safe component creation with validation
- Plugin architecture for easy extension
- Configuration-based component instantiation
"""

from .diffusion import DDPMTrainer as DDPMCore, create_enhanced_trainer, create_enhanced_config
from .unet import UNet

# Import registry system
from .registry import (
    ComponentRegistry,
    AutoRegistry,
    SCHEDULER_REGISTRY,
    SAMPLER_REGISTRY, 
    LOSS_REGISTRY,
    MODEL_REGISTRY,
    STRATEGY_REGISTRY,
    create_component,
    get_available_components,
    print_registry_status,
    register_scheduler,
    register_sampler,
    register_loss, 
    register_model,
    register_strategy,
)

from .losses import (
    MSELoss,
    L1Loss,
    HuberLoss,
    PerceptualLoss,
    FourierLoss,
    WaveletLoss,
    CycleConsistencyLoss,
    GradientLoss,
    CompositeLoss,
    create_loss_function,
)

# Import dual objective components
from .dual_objective_loss import (
    DualObjectiveLoss,
    IntensityMappingLoss,
    IntensityAugmentation,
    create_dual_objective_loss,
    create_dual_objective_training_config,
    create_progressive_training_strategy,
)

from .schedulers import (
    BaseScheduler,
    LinearScheduler,
    CosineScheduler,
    ExponentialScheduler,
    PolynomialScheduler,
    SigmoidScheduler,
    WarmupScheduler,
    AdaptiveScheduler,
    ResolutionAwareScheduler,
    LearnedScheduler,
    create_scheduler,
    get_default_scheduler_config,
)

from .sampler import (
    BaseSampler,
    DDIMSampler,
    DDPMSampler,
    AncestralSampler,
    create_sampler,
    get_default_sampler_config,
)

from .nn import (
    NNDefaults,
    ResNetBlock,
    SelfAttention2D,
    TimeEmbeddingMLP,
    ConvBlock,
    DoubleConvBlock,
    Downsample,
    Upsample,
    ZeroConv2d,
    get_activation,
    get_normalization,
    sinusoidal_position_embeddings,
    make_time_embedding,
)

from .progressive import (
    ProgressiveUNet,
    ProgressiveTrainer,
    ProgressiveDataLoader,
    create_progressive_config,
    run_progressive_training,
)

# Import cascaded sampling
try:
    from .cascaded_sampling import (
        CascadedSampler,
        HierarchicalCascadedSampler,
        MemoryEfficientCascadedSampler,
        create_cascaded_sampler,
    )
    CASCADED_SAMPLING_AVAILABLE = True
except ImportError:
    CASCADED_SAMPLING_AVAILABLE = False

# Import hierarchical strategies
try:
    from .hierarchical_strategy import (
        HierarchicalFeatureExtractor,
        HierarchicalNoiseScheduler,
        HierarchicalTrainer,
        HierarchicalSampler,
        create_hierarchical_trainer,
        create_hierarchical_sampler,
    )
    HIERARCHICAL_STRATEGIES_AVAILABLE = True
except ImportError:
    HIERARCHICAL_STRATEGIES_AVAILABLE = False


__all__ = [
    # Core components
    "DDPMCore",
    "UNet",
    
    # Registry system
    "ComponentRegistry",
    "AutoRegistry",
    "SCHEDULER_REGISTRY",
    "SAMPLER_REGISTRY", 
    "LOSS_REGISTRY",
    "MODEL_REGISTRY",
    "STRATEGY_REGISTRY",
    "create_component",
    "get_available_components",
    "print_registry_status",
    "register_scheduler",
    "register_sampler",
    "register_loss", 
    "register_model",
    "register_strategy",
    
    # Losses
    "MSELoss",
    "L1Loss",
    "HuberLoss", 
    "PerceptualLoss",
    "FourierLoss",
    "WaveletLoss",
    "CycleConsistencyLoss",
    "GradientLoss",
    "CompositeLoss",
    "create_loss_function",
    
    # Dual objective components
    "DualObjectiveLoss",
    "IntensityMappingLoss", 
    "IntensityAugmentation",
    "create_dual_objective_loss",
    "create_dual_objective_training_config",
    "create_progressive_training_strategy",
    "BaseScheduler",
    "LinearScheduler",
    "CosineScheduler",
    "ExponentialScheduler", 
    "PolynomialScheduler",
    "SigmoidScheduler",
    "WarmupScheduler",
    "AdaptiveScheduler",
    "ResolutionAwareScheduler",
    "LearnedScheduler",
    "create_scheduler",
    "get_default_scheduler_config",
    "BaseSampler",
    "DDIMSampler",
    "DDPMSampler", 
    "AncestralSampler",
    "CascadedSampler",
    "ProgressiveSampler",
    "create_sampler",
    "get_default_sampler_config",
    "NNDefaults",
    "ResNetBlock",
    "SelfAttention2D", 
    "TimeEmbeddingMLP",
    "ConvBlock",
    "DoubleConvBlock",
    "Downsample",
    "Upsample",
    "ZeroConv2d",
    "get_activation",
    "get_normalization",
    "sinusoidal_position_embeddings",
    "make_time_embedding",
    "ProgressiveUNet",
    "ProgressiveTrainer",
    "ProgressiveDataLoader", 
    "create_progressive_config",
    "run_progressive_training",
]

# Conditionally add advanced components if available
if CASCADED_SAMPLING_AVAILABLE:
    __all__.extend([
        "CascadedSampler",
        "HierarchicalCascadedSampler", 
        "MemoryEfficientCascadedSampler",
        "create_cascaded_sampler",
    ])

if HIERARCHICAL_STRATEGIES_AVAILABLE:
    __all__.extend([
        "HierarchicalFeatureExtractor",
        "HierarchicalNoiseScheduler",
        "HierarchicalTrainer",
        "HierarchicalSampler",
        "create_hierarchical_trainer",
        "create_hierarchical_sampler",
    ])

# Clean aliases
DDPMTrainer = DDPMCore


