"""
Enhanced Factory Functions for PKL Diffusion Models

This module provides comprehensive factory functions for creating and configuring
diffusion model components using the registry system. It follows the composition
patterns used by HuggingFace and other modern ML frameworks.

Features:
- Registry-based component creation
- Configuration validation and defaults
- Dependency injection and composition
- Plugin architecture support
- Type-safe component instantiation
"""

from typing import Dict, Any, Optional, List, Union, Type
import warnings
from pathlib import Path
import yaml
import json

from .registry import (
    SCHEDULER_REGISTRY,
    SAMPLER_REGISTRY,
    LOSS_REGISTRY,
    MODEL_REGISTRY,
    STRATEGY_REGISTRY,
    ComponentRegistry,
)
from .diffusion import DDPMTrainer
from .schedulers import BaseScheduler, create_scheduler
from .sampler import BaseSampler, create_sampler
from .unet import UNet


class ModelFactory:
    """Factory for creating complete diffusion model setups.
    
    This factory handles the composition of multiple components (model, scheduler, 
    sampler, etc.) into a complete diffusion training/inference setup.
    """
    
    @staticmethod
    def create_trainer(
        model: Optional[UNet] = None,
        scheduler_type: str = "cosine",
        sampler_type: str = "ddim",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> DDPMTrainer:
        """Create a complete DDPM trainer with all components.
        
        Args:
            model: UNet model (will create default if None)
            scheduler_type: Type of scheduler to use
            sampler_type: Type of sampler to use
            config: Configuration dictionary
            **kwargs: Additional configuration
            
        Returns:
            Configured DDPMTrainer
        """
        # Merge configurations
        final_config = {
            "num_timesteps": 1000,
            "beta_schedule": scheduler_type,
            "use_ema": True,
            "mixed_precision": True,
        }
        if config:
            final_config.update(config)
        final_config.update(kwargs)
        
        # Create model if not provided
        if model is None:
            model_config = final_config.get("model", {})
            model = ModelFactory.create_unet(model_config)
        
        # Create trainer
        trainer = DDPMTrainer(
            model=model,
            config=final_config,
            transform=final_config.get("transform"),
            forward_model=final_config.get("forward_model"),
        )
        
        return trainer
    
    @staticmethod
    def create_unet(config: Optional[Dict[str, Any]] = None) -> UNet:
        """Create UNet with sensible defaults.
        
        Args:
            config: UNet configuration
            
        Returns:
            Configured UNet model
        """
        default_config = {
            "in_channels": 1,
            "out_channels": 1,
            "sample_size": 256,
            "block_out_channels": [64, 128, 256, 512],
            "layers_per_block": 2,
            "attention_resolutions": [16, 32],
            "num_attention_heads": 8,
            "dropout": 0.1,
        }
        
        if config:
            default_config.update(config)
        
        return UNet(default_config)
    
    @staticmethod
    def create_inference_pipeline(
        model_path: Optional[str] = None,
        model: Optional[UNet] = None,
        sampler_type: str = "ddim",
        scheduler_type: str = "cosine",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create complete inference pipeline.
        
        Args:
            model_path: Path to saved model weights
            model: Pre-loaded model
            sampler_type: Type of sampler
            scheduler_type: Type of scheduler
            config: Pipeline configuration
            **kwargs: Additional configuration
            
        Returns:
            Dictionary containing all pipeline components
        """
        if model is None and model_path is None:
            raise ValueError("Either model or model_path must be provided")
        
        # Load model if path provided
        if model_path and model is None:
            # This would load from checkpoint - simplified for now
            model = ModelFactory.create_unet(config.get("model", {}) if config else {})
        
        # Create scheduler
        scheduler = create_scheduler(scheduler_type, **(config.get("scheduler", {}) if config else {}))
        
        # Create sampler
        sampler_config = config.get("sampler", {}) if config else {}
        sampler_config["model"] = model
        sampler = create_sampler(sampler_type, **sampler_config)
        
        return {
            "model": model,
            "scheduler": scheduler,
            "sampler": sampler,
            "config": config or {},
        }
    
    @staticmethod
    def from_config_file(config_path: Union[str, Path]) -> DDPMTrainer:
        """Create trainer from configuration file.
        
        Args:
            config_path: Path to YAML or JSON config file
            
        Returns:
            Configured trainer
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return ModelFactory.create_trainer(config=config)


class ComponentFactory:
    """Factory for individual component creation with validation."""
    
    @staticmethod
    def create_scheduler_with_validation(
        scheduler_type: str,
        num_timesteps: int = 1000,
        **kwargs
    ) -> BaseScheduler:
        """Create scheduler with parameter validation.
        
        Args:
            scheduler_type: Type of scheduler
            num_timesteps: Number of timesteps
            **kwargs: Additional parameters
            
        Returns:
            Validated scheduler instance
        """
        # Validate num_timesteps
        if not isinstance(num_timesteps, int) or num_timesteps <= 0:
            raise ValueError(f"num_timesteps must be positive integer, got {num_timesteps}")
        
        # Get default config and validate parameters
        try:
            default_config = SCHEDULER_REGISTRY.get_config(scheduler_type)
            
            # Merge with defaults
            final_config = default_config.copy()
            final_config.update(kwargs)
            final_config["num_timesteps"] = num_timesteps
            
            return create_scheduler(scheduler_type, **final_config)
        except Exception as e:
            raise RuntimeError(f"Failed to create scheduler '{scheduler_type}': {e}")
    
    @staticmethod
    def create_sampler_with_validation(
        sampler_type: str,
        model: UNet,
        **kwargs
    ) -> BaseSampler:
        """Create sampler with parameter validation.
        
        Args:
            sampler_type: Type of sampler
            model: Model to use
            **kwargs: Additional parameters
            
        Returns:
            Validated sampler instance
        """
        if not isinstance(model, UNet):
            raise TypeError(f"Expected UNet model, got {type(model)}")
        
        try:
            default_config = SAMPLER_REGISTRY.get_config(sampler_type)
            
            # Merge with defaults
            final_config = default_config.copy()
            final_config.update(kwargs)
            
            return create_sampler(sampler_type, model=model, **final_config)
        except Exception as e:
            raise RuntimeError(f"Failed to create sampler '{sampler_type}': {e}")


class PresetFactory:
    """Factory for creating preset configurations for common use cases."""
    
    @staticmethod
    def microscopy_2p_preset(
        image_size: int = 256,
        enable_guidance: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Preset for 2-photon microscopy denoising.
        
        Args:
            image_size: Image size
            enable_guidance: Enable physics guidance
            **kwargs: Override parameters
            
        Returns:
            Complete configuration dictionary
        """
        config = {
            "model": {
                "in_channels": 1,
                "out_channels": 1,
                "sample_size": image_size,
                "block_out_channels": [64, 128, 256, 512],
                "attention_resolutions": [16, 32],
            },
            "training": {
                "num_timesteps": 1000,
                "beta_schedule": "cosine",
                "learning_rate": 1e-4,
                "batch_size": 8,
                "use_ema": True,
                "mixed_precision": True,
            },
            "scheduler": {
                "type": "cosine",
                "s": 0.008,
                "max_beta": 0.999,
            },
            "sampler": {
                "type": "ddim",
                "ddim_steps": 100,
                "eta": 0.0,
                "clip_denoised": True,
            },
        }
        
        if enable_guidance:
            config["guidance"] = {
                "enable_physics_guidance": True,
                "guidance_scale": 1.0,
                "lambda_schedule": "adaptive",
            }
        
        # Apply overrides
        config.update(kwargs)
        
        return config
    
    @staticmethod
    def fast_inference_preset(
        image_size: int = 256,
        num_steps: int = 25,
        **kwargs
    ) -> Dict[str, Any]:
        """Preset for fast inference.
        
        Args:
            image_size: Image size
            num_steps: Number of inference steps
            **kwargs: Override parameters
            
        Returns:
            Fast inference configuration
        """
        config = {
            "model": {
                "sample_size": image_size,
                "block_out_channels": [64, 128, 256],  # Smaller model
            },
            "scheduler": {
                "type": "improved_cosine",
                "max_beta": 0.9999,
            },
            "sampler": {
                "type": "ddim",
                "ddim_steps": num_steps,
                "eta": 0.0,
            },
            "inference": {
                "use_karras_sigmas": True,
                "mixed_precision": True,
                "enable_memory_optimization": True,
            },
        }
        
        config.update(kwargs)
        return config
    
    @staticmethod
    def high_quality_preset(
        image_size: int = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """Preset for high-quality generation.
        
        Args:
            image_size: Image size
            **kwargs: Override parameters
            
        Returns:
            High-quality configuration
        """
        config = {
            "model": {
                "sample_size": image_size,
                "block_out_channels": [128, 256, 512, 768],  # Larger model
                "layers_per_block": 3,
                "attention_resolutions": [8, 16, 32],
                "num_attention_heads": 16,
            },
            "training": {
                "num_timesteps": 1000,
                "beta_schedule": "improved_cosine",
                "learning_rate": 5e-5,  # Lower LR for stability
                "use_ema": True,
                "ema_decay": 0.9999,
            },
            "scheduler": {
                "type": "improved_cosine",
                "cosine_power": 1.0,
                "max_beta": 0.9999,
            },
            "sampler": {
                "type": "ddim",
                "ddim_steps": 200,  # More steps for quality
                "eta": 0.0,
            },
            "advanced_features": {
                "enable_frequency_losses": True,
                "enable_cascaded_sampling": True,
                "frequency_loss_weight": 0.1,
            },
        }
        
        config.update(kwargs)
        return config


def create_from_preset(
    preset_name: str,
    **kwargs
) -> DDPMTrainer:
    """Create trainer from preset configuration.
    
    Args:
        preset_name: Name of preset ('microscopy_2p', 'fast_inference', 'high_quality')
        **kwargs: Override parameters
        
    Returns:
        Configured trainer
    """
    presets = {
        "microscopy_2p": PresetFactory.microscopy_2p_preset,
        "fast_inference": PresetFactory.fast_inference_preset,
        "high_quality": PresetFactory.high_quality_preset,
    }
    
    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    config = presets[preset_name](**kwargs)
    return ModelFactory.create_trainer(config=config)


def list_available_components() -> Dict[str, List[str]]:
    """List all available components in all registries.
    
    Returns:
        Dictionary mapping component types to available components
    """
    return {
        "schedulers": SCHEDULER_REGISTRY.list_components(),
        "samplers": SAMPLER_REGISTRY.list_components(),
        "losses": LOSS_REGISTRY.list_components(),
        "models": MODEL_REGISTRY.list_components(),
        "strategies": STRATEGY_REGISTRY.list_components(),
    }


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of warnings/errors.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation messages
    """
    messages = []
    
    # Check for required fields
    if "model" not in config:
        messages.append("WARNING: No model configuration found, using defaults")
    
    # Validate scheduler config
    scheduler_config = config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "cosine")
    if scheduler_type not in SCHEDULER_REGISTRY:
        available = SCHEDULER_REGISTRY.list_components()
        messages.append(f"ERROR: Unknown scheduler '{scheduler_type}'. Available: {available}")
    
    # Validate sampler config  
    sampler_config = config.get("sampler", {})
    sampler_type = sampler_config.get("type", "ddim")
    if sampler_type not in SAMPLER_REGISTRY:
        available = SAMPLER_REGISTRY.list_components()
        messages.append(f"ERROR: Unknown sampler '{sampler_type}'. Available: {available}")
    
    # Check for common misconfigurations
    training_config = config.get("training", {})
    if training_config.get("batch_size", 8) > 32:
        messages.append("WARNING: Large batch size may cause memory issues")
    
    if training_config.get("learning_rate", 1e-4) > 1e-3:
        messages.append("WARNING: High learning rate may cause training instability")
    
    return messages


__all__ = [
    "ModelFactory",
    "ComponentFactory", 
    "PresetFactory",
    "create_from_preset",
    "list_available_components",
    "validate_config",
]
