"""
Shared Test Utilities

This module consolidates common test helper functions to avoid duplication
across multiple test files.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional

# Import the ForwardModel - adjust import path as needed
try:
    from pkl_dg.physics import ForwardModel
except ImportError:
    # Fallback for tests that don't need ForwardModel
    ForwardModel = None


def make_tiny_dataset(root: Path, num_images: int = 4, size: int = 16) -> None:
    """Create a tiny synthetic dataset for testing.
    
    Args:
        root: Root directory for the dataset
        num_images: Number of images per split
        size: Image size (width and height)
    """
    for split in ["train", "val"]:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_images):
            arr = (np.random.rand(size, size) * 255).astype(np.uint8)
            Image.fromarray(arr).save(split_dir / f"img_{i}.png")


def make_forward_model(device: str = "cpu", background: float = 0.0) -> Optional[ForwardModel]:
    """Create a simple forward model for testing.
    
    Args:
        device: Device to create the model on
        background: Background level for the forward model
        
    Returns:
        ForwardModel instance, or None if ForwardModel is not available
    """
    if ForwardModel is None:
        return None
        
    psf = torch.ones(9, 9) / 81.0
    return ForwardModel(psf=psf, background=background, device=device)


# For backward compatibility, create aliases with the old function names
_make_tiny_dataset = make_tiny_dataset
_make_forward_model = make_forward_model


__all__ = [
    'make_tiny_dataset',
    'make_forward_model',
    '_make_tiny_dataset',  # backward compatibility
    '_make_forward_model'   # backward compatibility
]
