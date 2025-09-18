"""
Utility functions and classes for PKL Diffusion Denoising.

This package provides a clean, organized structure for utilities:
- Core utilities: I/O, logging, config, device management (consolidated in utils.py)
- Data processing: Image processing, validation (moved to dataset.py)
- Visualization: Plotting, monitoring, and result visualization
- Interpolation: Interpolation and morphing utilities (specialized module)
"""

from .utils import (
    PathManager,
    ConfigManager,
    CheckpointManager,
    Logger,
    DeviceManager,
    Timer,
    IOManager,
    ensure_dir,
    setup_logging,
    setup_wandb,
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    get_environment_info,
    MemoryProfiler,
    MemorySnapshot,
    MemoryProfile,
    profile_memory,
    get_memory_summary,
    cleanup_memory,
    AdaptiveBatchSizer,
    AdaptiveDataLoader,
    get_optimal_batch_size,
    create_adaptive_dataloader,
    ConfigValidator,
    create_training_config_validator,
    merge_configs,
    validate_and_complete_config,
    print_config_summary,
)

UINT16_MAX = 65535
UINT16_MIN = 0
MODEL_RANGE_MIN = -1.0
MODEL_RANGE_MAX = 1.0

# 16-bit image processing functions moved to utils_16bit.py
# Import them here for backward compatibility
try:
    from .utils_16bit import (
        load_16bit_image,
        normalize_16bit_to_model_input, 
        denormalize_model_output_to_16bit,
        UINT16_MAX,
        UINT16_MIN,
        # Adaptive normalization classes
        NormalizationParams,
        AdaptiveNormalizer,
        create_normalization_params_from_metadata,
        analyze_current_normalization_issues
    )
except ImportError:
    # Fallback implementations if utils_16bit is not available
    def load_16bit_image(path: str):
        """Load 16-bit image from file."""
        try:
            import tifffile
            import numpy as np
            img = tifffile.imread(path)
            if img.dtype == np.uint16:
                return img.astype(np.float32)
            elif img.dtype == np.uint8:
                raise ValueError(f"8-bit images are not supported. Image {path} should be 16-bit TIFF format.")
            else:
                return img.astype(np.float32)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

    def normalize_16bit_to_model_input(image):
        """Normalize 16-bit image to model input range [-1, 1]."""
        import numpy as np
        if image is None:
            return None
        normalized = image / UINT16_MAX
        return (normalized * 2.0 - 1.0).astype(np.float32)

    def denormalize_model_output_to_16bit(image):
        """Denormalize model output from [-1, 1] to 16-bit range."""
        import numpy as np
        if image is None:
            return None
        normalized = (image + 1.0) / 2.0
        return (normalized * UINT16_MAX).astype(np.float32)

# Note: robust_normalize_16bit was removed as per requirements (no percentile normalization)

from .visualization import (
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
    PlotManager,
    TrainingVisualizer,
    ImageVisualizer,
    MetricsVisualizer,
    SamplingVisualizer, 
    WandBLogger,
    close_all_figures,
    save_figure_safely,
)

MICROSCOPY_AVAILABLE = False
__all__ = [
    "PathManager",
    "ConfigManager", 
    "CheckpointManager",
    "Logger",
    "DeviceManager",
    "Timer",
    "IOManager",
    "ensure_dir",
    "setup_logging",
    "setup_wandb",
    "save_json",
    "load_json", 
    "save_pickle",
    "load_pickle",
    "get_environment_info",
    "MemoryProfiler",
    "MemorySnapshot",
    "MemoryProfile",
    "profile_memory",
    "get_memory_summary",
    "cleanup_memory", 
    "AdaptiveBatchSizer",
    "AdaptiveDataLoader",
    "get_optimal_batch_size",
    "create_adaptive_dataloader",
    "ConfigValidator",
    "create_training_config_validator",
    "merge_configs",
    "validate_and_complete_config",
    "print_config_summary",
    
    "UINT16_MAX",
    "UINT16_MIN",
    "MODEL_RANGE_MIN", 
    "MODEL_RANGE_MAX",
    "load_16bit_image",
    "robust_normalize_16bit",
    "normalize_16bit_to_model_input",
    "denormalize_model_output_to_16bit",
    # Adaptive normalization
    "NormalizationParams",
    "AdaptiveNormalizer",
    "create_normalization_params_from_metadata",
    "analyze_current_normalization_issues",
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
    "PlotManager",
    "TrainingVisualizer",
    "ImageVisualizer",
    "MetricsVisualizer",
    "SamplingVisualizer", 
    "WandBLogger",
    "close_all_figures",
    "save_figure_safely",
]
def plot_training_curves(*args, **kwargs):
    """Convenience function for plotting training curves (backward compatibility)."""
    visualizer = TrainingVisualizer()
    return visualizer.plot_training_curves(*args, **kwargs)

def plot_sample_grid(*args, **kwargs):
    """Convenience function for plotting sample grids (backward compatibility).""" 
    visualizer = ImageVisualizer()
    return visualizer.plot_image_grid(*args, **kwargs)

def plot_comparison_grid(*args, **kwargs):
    """Convenience function for plotting comparison grids (backward compatibility)."""
    visualizer = ImageVisualizer()
    return visualizer.plot_comparison_grid(*args, **kwargs)
