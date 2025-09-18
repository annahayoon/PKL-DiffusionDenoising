"""
Utility functions and classes for PKL-DG.
"""

# I/O utilities
from .io import (
    CheckpointManager,
    setup_logging,
    setup_wandb,
    save_config,
    load_config,
    ensure_dir,
    get_latest_checkpoint,
    unified_training_loop,
    setup_distributed_training,
    create_distributed_model,
    cleanup_distributed,
    distributed_training_loop,
)

# Visualization utilities
from .visualization import (
    plot_training_curves,
    plot_sample_grid,
    plot_comparison_grid,
    plot_memory_usage,
    plot_interpolation_sequence,
    plot_psf_analysis,
    plot_evaluation_metrics,
    create_training_dashboard,
    log_to_wandb,
    save_figure_safely,
    close_all_figures,
)

# Configuration utilities
from .config import (
    ConfigValidator,
    create_training_config_validator,
    create_inference_config_validator,
    merge_configs,
    load_config_with_overrides,
    get_environment_config,
    adapt_config_for_environment,
    create_config_from_template,
    validate_and_complete_config,
    print_config_summary,
    export_config_for_reproducibility,
)

# Image processing utilities
from .image_processing import (
    normalize_16bit_to_model_input,
    denormalize_model_output_to_16bit,
    load_16bit_image,
    robust_normalize_16bit,
    convert_8bit_to_16bit_equivalent,
    adaptive_histogram_equalization,
    gamma_correction,
    bilateral_filter,
    gaussian_filter,
    unsharp_mask,
    resize_16bit_image,
    crop_center,
    pad_to_size,
    create_16bit_test_image,
    UINT16_MAX,
    UINT16_MIN,
    MODEL_RANGE_MIN,
    MODEL_RANGE_MAX,
)

# Data validation utilities
from .data_validation import (
    get_16bit_image_stats,
    validate_16bit_image,
    validate_model_input,
    validate_batch_consistency,
    validate_checkpoint_compatibility,
    validate_dataset_structure,
    validate_training_config,
    validate_gpu_memory_requirements,
    validate_inference_inputs,
    DataValidator,
)

# Memory and adaptive batch utilities
from .adaptive_batch import (
    AdaptiveBatchSizer,
    AdaptiveDataLoader,
    get_optimal_batch_size,
    create_adaptive_dataloader,
)

from .memory import (
    MemoryProfiler,
    MemorySnapshot,
    MemoryProfile,
    profile_memory,
    profile_memory_usage,
    get_memory_summary,
    cleanup_memory,
    monitor_training_memory,
)

# Interpolation utilities  
from .interpolation import (
    linear_interpolation,
    spherical_interpolation,
    noise_interpolation,
    latent_interpolation,
    diffusion_interpolation,
    morphing_sequence,
    create_interpolation_grid,
    save_interpolation_video,
    analyze_interpolation_smoothness,
    InterpolationPipeline,
)

# Keep backward compatibility with microscopy module
from .microscopy import Microscopy16BitProcessor

__all__ = [
    # I/O utilities
    "CheckpointManager",
    "setup_logging",
    "setup_wandb",
    "save_config",
    "load_config",
    "ensure_dir",
    "get_latest_checkpoint",
    "unified_training_loop",
    "setup_distributed_training",
    "create_distributed_model",
    "cleanup_distributed",
    "distributed_training_loop",
    
    # Visualization utilities
    "plot_training_curves",
    "plot_sample_grid",
    "plot_comparison_grid",
    "plot_memory_usage",
    "plot_interpolation_sequence",
    "plot_psf_analysis",
    "plot_evaluation_metrics",
    "create_training_dashboard",
    "log_to_wandb",
    "save_figure_safely",
    "close_all_figures",
    
    # Configuration utilities
    "ConfigValidator",
    "create_training_config_validator",
    "create_inference_config_validator",
    "merge_configs",
    "load_config_with_overrides",
    "get_environment_config",
    "adapt_config_for_environment",
    "create_config_from_template",
    "validate_and_complete_config",
    "print_config_summary",
    "export_config_for_reproducibility",
    
    # Image processing utilities
    "normalize_16bit_to_model_input",
    "denormalize_model_output_to_16bit",
    "load_16bit_image",
    "robust_normalize_16bit",
    "convert_8bit_to_16bit_equivalent",
    "adaptive_histogram_equalization",
    "gamma_correction",
    "bilateral_filter",
    "gaussian_filter",
    "unsharp_mask",
    "resize_16bit_image",
    "crop_center",
    "pad_to_size",
    "create_16bit_test_image",
    "UINT16_MAX",
    "UINT16_MIN",
    "MODEL_RANGE_MIN",
    "MODEL_RANGE_MAX",
    
    # Data validation utilities
    "get_16bit_image_stats",
    "validate_16bit_image",
    "validate_model_input",
    "validate_batch_consistency",
    "validate_checkpoint_compatibility",
    "validate_dataset_structure",
    "validate_training_config",
    "validate_gpu_memory_requirements",
    "validate_inference_inputs",
    "DataValidator",
    
    # Memory and adaptive batch utilities
    "AdaptiveBatchSizer",
    "AdaptiveDataLoader",
    "get_optimal_batch_size",
    "create_adaptive_dataloader",
    "MemoryProfiler",
    "MemorySnapshot",
    "MemoryProfile",
    "profile_memory",
    "profile_memory_usage",
    "get_memory_summary",
    "cleanup_memory",
    "monitor_training_memory",
    
    # Interpolation utilities
    "linear_interpolation",
    "spherical_interpolation", 
    "noise_interpolation",
    "latent_interpolation",
    "diffusion_interpolation",
    "morphing_sequence",
    "create_interpolation_grid",
    "save_interpolation_video",
    "analyze_interpolation_smoothness",
    "InterpolationPipeline",
    
    # Backward compatibility
    "Microscopy16BitProcessor",
]