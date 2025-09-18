"""PKL-Diffusion Denoising core package.

This package provides physics-guided diffusion models for microscopy image denoising
using Poisson-Kullback-Leibler (PKL) guidance strategies.

Modules:
- models: Neural network architectures, training, and sampling
- utils: Utilities for I/O, visualization, and data processing
- guidance: Physics-guided diffusion strategies
- physics: Physical models and forward operators
- data: Dataset handling and preprocessing
- evaluation: Metrics and evaluation tools
- baselines: Baseline methods for comparison
"""

from typing import TYPE_CHECKING
import importlib

__version__ = "0.2.0"

__all__ = [
    "data",
    "physics",
    "models",
    "utils",
    "ForwardModel",
    "PSF",
]

if TYPE_CHECKING:
    from . import data as data  # noqa: F401
    from . import physics as physics  # noqa: F401
    from . import models as models  # noqa: F401
    from . import utils as utils  # noqa: F401
    from .physics import ForwardModel as ForwardModel  # noqa: F401
    from .physics import PSF as PSF  # noqa: F401


def __getattr__(name: str):
    if name in {"data", "physics", "models", "utils"}:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in {"ForwardModel", "PSF"}:
        physics_module = importlib.import_module(".physics", __name__)
        obj = getattr(physics_module, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + __all__)

