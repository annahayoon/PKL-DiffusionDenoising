"""
API Usage Examples and Correct Constructor Patterns

This module provides examples of correct API usage for the PKL-DiffusionDenoising
codebase, addressing common constructor parameter mismatches.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional

from ..models import UNet, DDPMTrainer
from ..models.factory import ModelFactory
from ..data import RealPairsDataset, IntensityToModel, Microscopy16BitToModel
from ..physics import PSF, ForwardModel
from ..guidance import PKLGuidance, L2Guidance


def create_unet_via_factory() -> UNet:
    """Create UNet using the ModelFactory (recommended approach)."""
    # RECOMMENDED: Use ModelFactory for sensible defaults
    config = {
        "sample_size": 256,
        "block_out_channels": [64, 128, 256],  # Smaller for testing
    }
    return ModelFactory.create_unet(config)


def create_real_pairs_dataset_correct(data_dir: str) -> RealPairsDataset:
    """Create RealPairsDataset with correct parameters."""
    # CORRECT: RealPairsDataset constructor signature
    return RealPairsDataset(
        data_dir=data_dir,                    # Required: root data directory
        split="train",                        # Required: train/val/test split
        transform=IntensityToModel(0, 255),   # Optional: data transform
        image_size=256,                       # Optional: target image size
        mode="train",                         # Optional: training mode
        align_pairs=True,                     # Optional: align WF to 2P
        max_shift=4,                          # Optional: max alignment shift
        use_16bit_normalization=True          # Optional: use 16-bit normalization
    )


def create_transforms_correct() -> Dict[str, Any]:
    """Create data transforms with correct parameters."""
    transforms = {}
    
    # Standard 8-bit intensity transform
    transforms['intensity_8bit'] = IntensityToModel(
        min_intensity=0,
        max_intensity=255
    )
    
    # 16-bit microscopy transform
    transforms['microscopy_16bit'] = Microscopy16BitToModel(
        min_intensity=0,
        max_intensity=65535
    )
    
    return transforms


def create_physics_components_correct() -> Dict[str, Any]:
    """Create physics components with correct parameters."""
    # PSF creation (no parameters needed for default)
    psf = PSF()
    
    # ForwardModel creation
    forward_model = ForwardModel(
        psf=psf.to_torch(device="cpu"),  # PSF as torch tensor
        background=0.0,                  # Background level
        device="cpu"                     # Device
    )
    
    return {
        "psf": psf,
        "forward_model": forward_model
    }


def create_guidance_correct(forward_model: ForwardModel) -> Dict[str, Any]:
    """Create guidance strategies with correct parameters."""
    guidance_strategies = {}
    
    # PKL Guidance (only needs epsilon parameter)
    guidance_strategies['pkl'] = PKLGuidance(
        epsilon=1e-6  # Optional: numerical stability parameter
    )
    
    # L2 Guidance 
    guidance_strategies['l2'] = L2Guidance()
    
    return guidance_strategies


def create_trainer_via_factory(model: Optional[UNet] = None) -> DDPMTrainer:
    """Create trainer using ModelFactory (recommended)."""
    # RECOMMENDED: Use ModelFactory
    config = {
        "num_timesteps": 1000,
        "beta_schedule": "cosine",
        "use_ema": True,
        "mixed_precision": True
    }
    
    return ModelFactory.create_trainer(
        model=model,
        scheduler_type="cosine",
        sampler_type="ddim",
        config=config
    )


def example_end_to_end_correct():
    """Complete example with correct API usage."""
    print("🧪 Demonstrating correct API usage...")
    
    # 1. Create model using factory (recommended)
    model = create_unet_via_factory()
    print("✅ UNet created correctly")
    
    # 2. Create physics components
    physics = create_physics_components_correct()
    forward_model = physics['forward_model']
    print("✅ Physics components created correctly")
    
    # 3. Create guidance
    guidance_dict = create_guidance_correct(forward_model)
    pkl_guidance = guidance_dict['pkl']
    print("✅ Guidance strategies created correctly")
    
    # 4. Create transforms
    transforms = create_transforms_correct()
    transform = transforms['intensity_8bit']
    print("✅ Data transforms created correctly")
    
    # 5. Test with synthetic data
    test_data = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    test_tensor = torch.from_numpy(test_data.astype(np.float32))
    transformed_data = transform(test_tensor)
    print("✅ Data transform works correctly")
    
    # 6. Create trainer using factory (recommended)
    trainer = create_trainer_via_factory(model)
    print("✅ Trainer created correctly")
    
    print("🎉 All components created with correct API usage!")
    return {
        "model": model,
        "forward_model": forward_model,
        "guidance": pkl_guidance,
        "transform": transform,
        "trainer": trainer
    }


# Common API mistakes and their corrections
API_CORRECTIONS = {
    "UNet": {
        "wrong": "UNet(in_channels=1, out_channels=1, model_channels=32)",
        "correct": "UNet({'in_channels': 1, 'out_channels': 1, 'block_out_channels': [32, 64]})",
        "recommended": "ModelFactory.create_unet({'sample_size': 256}) # ALWAYS USE FACTORY"
    },
    "PKLGuidance": {
        "wrong": "PKLGuidance(forward_model=fm, lambda_val=0.1)",
        "correct": "PKLGuidance(epsilon=1e-6)",
        "note": "forward_model is passed to compute_gradient method, not constructor"
    },
    "RealPairsDataset": {
        "wrong": "RealPairsDataset(root_dir='path', transform=t)",
        "correct": "RealPairsDataset(data_dir='path', split='train', transform=t)",
        "note": "Use data_dir and split parameters, not root_dir"
    },
    "IntensityToModel": {
        "wrong": "IntensityToModel(minIntensity=0, maxIntensity=255)",
        "correct": "IntensityToModel(min_intensity=0, max_intensity=255)",
        "note": "Use snake_case parameter names"
    }
}


if __name__ == "__main__":
    # Run the example to demonstrate correct usage
    example_end_to_end_correct()
    
    print("\n📚 API Corrections Summary:")
    for component, corrections in API_CORRECTIONS.items():
        print(f"\n{component}:")
        print(f"  ❌ Wrong: {corrections['wrong']}")
        print(f"  ✅ Correct: {corrections['correct']}")
        if 'recommended' in corrections:
            print(f"  🌟 Recommended: {corrections['recommended']}")
        if 'note' in corrections:
            print(f"  💡 Note: {corrections['note']}")
