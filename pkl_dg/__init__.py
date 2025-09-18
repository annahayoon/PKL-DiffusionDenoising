"""PKL-Diffusion Denoising core package.

This package provides physics-guided diffusion models for microscopy image denoising
using Poisson-Kullback-Leibler (PKL) guidance strategies.

The package is organized into the following modules:
- models: Neural network architectures, training, and sampling
- utils: Utilities for I/O, visualization, and data processing
- guidance: Physics-guided diffusion strategies
- physics: Physical models and forward operators
- data: Dataset handling and preprocessing
- evaluation: Metrics and evaluation tools
- baselines: Baseline methods for comparison
"""

__version__ = "0.2.0"  # Updated version for reorganization

# Import key components for easy access
from . import (
    data,
    physics,
    models, 
    utils,
)

# Core components are imported within their respective modules
# to avoid circular imports and maintain clean module boundaries

from .physics import (
    ForwardModel,
    PSF,
)

__all__ = [
    # Submodules
    "data",
    "physics", 
    "models",
    "utils",
    
    # Physics components
    "ForwardModel",
    "PSF",
]


