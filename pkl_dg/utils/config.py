"""
Configuration Management Utilities for PKL Diffusion Denoising

This module provides utilities for configuration validation, merging,
and environment-specific configuration management.
"""

import os
import copy
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import warnings

try:
    from omegaconf import OmegaConf, DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConfigValidator:
    """Validates configuration dictionaries against schemas."""
    
    def __init__(self):
        self.validation_rules = {}
        self.required_fields = {}
        self.default_values = {}
        
    def add_validation_rule(
        self, 
        field_path: str, 
        validator: Callable[[Any], bool],
        error_message: str
    ):
        """Add a validation rule for a specific field."""
        self.validation_rules[field_path] = (validator, error_message)
        
    def add_required_field(self, field_path: str, field_type: type = None):
        """Mark a field as required."""
        self.required_fields[field_path] = field_type
        
    def add_default_value(self, field_path: str, default_value: Any):
        """Add a default value for a field."""
        self.default_values[field_path] = default_value
        
    def validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration and return validated config with defaults."""
        config = copy.deepcopy(config)
        errors = []
        
        # Add default values
        for field_path, default_value in self.default_values.items():
            if not self._has_field(config, field_path):
                self._set_field(config, field_path, default_value)
        
        # Check required fields
        for field_path, field_type in self.required_fields.items():
            if not self._has_field(config, field_path):
                errors.append(f"Required field missing: {field_path}")
            elif field_type is not None:
                value = self._get_field(config, field_path)
                if not isinstance(value, field_type):
                    errors.append(f"Field {field_path} must be of type {field_type.__name__}, got {type(value).__name__}")
        
        # Apply validation rules
        for field_path, (validator, error_message) in self.validation_rules.items():
            if self._has_field(config, field_path):
                value = self._get_field(config, field_path)
                try:
                    if not validator(value):
                        errors.append(f"{field_path}: {error_message}")
                except Exception as e:
                    errors.append(f"{field_path}: Validation error - {str(e)}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
        
        return config
    
    def _has_field(self, config: Dict[str, Any], field_path: str) -> bool:
        """Check if field exists in nested config."""
        keys = field_path.split('.')
        current = config
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return False
            current = current[key]
        
        return True
    
    def _get_field(self, config: Dict[str, Any], field_path: str) -> Any:
        """Get field value from nested config."""
        keys = field_path.split('.')
        current = config
        
        for key in keys:
            current = current[key]
        
        return current
    
    def _set_field(self, config: Dict[str, Any], field_path: str, value: Any):
        """Set field value in nested config."""
        keys = field_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value


def create_training_config_validator() -> ConfigValidator:
    """Create validator for training configurations."""
    validator = ConfigValidator()
    
    # Required fields
    validator.add_required_field("model.sample_size", int)
    validator.add_required_field("model.in_channels", int)
    validator.add_required_field("model.out_channels", int)
    validator.add_required_field("training.learning_rate", (int, float))
    validator.add_required_field("training.batch_size", int)
    validator.add_required_field("training.num_epochs", int)
    
    # Default values
    validator.add_default_value("model.num_timesteps", 1000)
    validator.add_default_value("model.beta_schedule", "cosine")
    validator.add_default_value("training.weight_decay", 1e-6)
    validator.add_default_value("training.use_ema", True)
    validator.add_default_value("training.mixed_precision", False)
    validator.add_default_value("training.gradient_checkpointing", False)
    validator.add_default_value("logging.log_every", 100)
    validator.add_default_value("logging.save_every", 1000)
    
    # Validation rules
    validator.add_validation_rule(
        "model.sample_size",
        lambda x: x > 0 and (x & (x - 1)) == 0,  # Power of 2
        "sample_size must be a positive power of 2"
    )
    
    validator.add_validation_rule(
        "model.in_channels",
        lambda x: x > 0,
        "in_channels must be positive"
    )
    
    validator.add_validation_rule(
        "model.out_channels", 
        lambda x: x > 0,
        "out_channels must be positive"
    )
    
    validator.add_validation_rule(
        "training.learning_rate",
        lambda x: 0 < x < 1,
        "learning_rate must be between 0 and 1"
    )
    
    validator.add_validation_rule(
        "training.batch_size",
        lambda x: x > 0,
        "batch_size must be positive"
    )
    
    validator.add_validation_rule(
        "training.num_epochs",
        lambda x: x > 0,
        "num_epochs must be positive"
    )
    
    validator.add_validation_rule(
        "model.beta_schedule",
        lambda x: x in ["linear", "cosine", "squaredcos_cap_v2"],
        "beta_schedule must be one of: linear, cosine, squaredcos_cap_v2"
    )
    
    return validator


def create_inference_config_validator() -> ConfigValidator:
    """Create validator for inference configurations."""
    validator = ConfigValidator()
    
    # Required fields
    validator.add_required_field("model.checkpoint_path", str)
    validator.add_required_field("inference.num_inference_steps", int)
    
    # Default values
    validator.add_default_value("inference.guidance_scale", 1.0)
    validator.add_default_value("inference.use_ema", True)
    validator.add_default_value("inference.batch_size", 4)
    
    # Validation rules
    validator.add_validation_rule(
        "model.checkpoint_path",
        lambda x: Path(x).exists(),
        "checkpoint_path must point to existing file"
    )
    
    validator.add_validation_rule(
        "inference.num_inference_steps",
        lambda x: x > 0,
        "num_inference_steps must be positive"
    )
    
    validator.add_validation_rule(
        "inference.guidance_scale",
        lambda x: x >= 0,
        "guidance_scale must be non-negative"
    )
    
    return validator


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries."""
    if not configs:
        return {}
    
    result = copy.deepcopy(configs[0])
    
    for config in configs[1:]:
        result = _deep_merge(result, config)
    
    return result


def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = copy.deepcopy(dict1)
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def load_config_with_overrides(
    base_config_path: Union[str, Path],
    override_configs: Optional[List[Union[str, Path]]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load base config and apply overrides."""
    from .io import load_config
    
    # Load base config
    base_config = load_config(base_config_path)
    
    # Apply override configs
    configs_to_merge = [base_config]
    
    if override_configs:
        for override_path in override_configs:
            override_config = load_config(override_path)
            configs_to_merge.append(override_config)
    
    # Merge all configs
    merged_config = merge_configs(*configs_to_merge)
    
    # Apply CLI overrides
    if cli_overrides:
        merged_config = _deep_merge(merged_config, cli_overrides)
    
    return merged_config


def get_environment_config() -> Dict[str, Any]:
    """Get environment-specific configuration."""
    env_config = {}
    
    # GPU configuration
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        env_config["gpu_ids"] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    
    # Distributed training
    if os.environ.get("WORLD_SIZE"):
        env_config["distributed"] = {
            "world_size": int(os.environ["WORLD_SIZE"]),
            "rank": int(os.environ.get("RANK", 0)),
            "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        }
    
    # Compute environment
    if os.environ.get("SLURM_JOB_ID"):
        env_config["compute_environment"] = "slurm"
        env_config["slurm"] = {
            "job_id": os.environ["SLURM_JOB_ID"],
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "ntasks": int(os.environ.get("SLURM_NTASKS", 1)),
        }
    elif os.environ.get("AWS_BATCH_JOB_ID"):
        env_config["compute_environment"] = "aws_batch"
    else:
        env_config["compute_environment"] = "local"
    
    # Memory and performance hints
    import torch
    if torch.cuda.is_available():
        env_config["cuda_info"] = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
        }
    
    return env_config


def adapt_config_for_environment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt configuration based on environment."""
    env_config = get_environment_config()
    adapted_config = copy.deepcopy(config)
    
    # Adapt batch size based on available memory
    if "cuda_info" in env_config:
        total_memory_gb = env_config["cuda_info"]["memory_total"] / 1e9
        
        # Suggest batch size adjustments
        if total_memory_gb < 8:  # Low memory
            if adapted_config.get("training", {}).get("batch_size", 0) > 4:
                warnings.warn("Reducing batch size for low memory environment")
                adapted_config.setdefault("training", {})["batch_size"] = 4
                adapted_config.setdefault("training", {})["gradient_checkpointing"] = True
        elif total_memory_gb > 24:  # High memory
            adapted_config.setdefault("training", {})["mixed_precision"] = True
    
    # Adapt for distributed training
    if "distributed" in env_config:
        adapted_config.setdefault("training", {})["distributed"] = True
        adapted_config["training"].update(env_config["distributed"])
    
    # Add environment info
    adapted_config["environment"] = env_config
    
    return adapted_config


def create_config_from_template(
    template_name: str,
    custom_values: Optional[Dict[str, Any]] = None,
    config_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Create configuration from template."""
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "configs"
    else:
        config_dir = Path(config_dir)
    
    # Define templates
    templates = {
        "training_microscopy": {
            "model": {
                "sample_size": 256,
                "in_channels": 1,
                "out_channels": 1,
                "block_out_channels": [128, 256, 512, 512],
                "num_timesteps": 1000,
                "beta_schedule": "cosine"
            },
            "training": {
                "learning_rate": 1e-4,
                "batch_size": 8,
                "num_epochs": 100,
                "weight_decay": 1e-6,
                "use_ema": True,
                "mixed_precision": False
            },
            "data": {
                "image_size": 256,
                "normalize": True
            },
            "logging": {
                "log_every": 100,
                "save_every": 1000,
                "wandb_project": "pkl_diffusion"
            }
        },
        
        "inference_default": {
            "model": {
                "checkpoint_path": "checkpoints/best_model.pt"
            },
            "inference": {
                "num_inference_steps": 50,
                "guidance_scale": 1.0,
                "use_ema": True,
                "batch_size": 4
            },
            "data": {
                "image_size": 256
            }
        },
        
        "evaluation_default": {
            "model": {
                "checkpoint_path": "checkpoints/best_model.pt"
            },
            "evaluation": {
                "batch_size": 16,
                "num_samples": 1000,
                "compute_fid": True,
                "compute_is": True,
                "fid_batch_size": 50
            },
            "data": {
                "image_size": 256
            }
        }
    }
    
    if template_name not in templates:
        available = ", ".join(templates.keys())
        raise ValueError(f"Unknown template: {template_name}. Available: {available}")
    
    # Start with template
    config = copy.deepcopy(templates[template_name])
    
    # Apply custom values
    if custom_values:
        config = _deep_merge(config, custom_values)
    
    return config


def validate_and_complete_config(
    config: Dict[str, Any],
    config_type: str = "training"
) -> Dict[str, Any]:
    """Validate and complete configuration with defaults."""
    if config_type == "training":
        validator = create_training_config_validator()
    elif config_type == "inference":
        validator = create_inference_config_validator()
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    # Validate and add defaults
    validated_config = validator.validate(config)
    
    # Adapt for environment
    adapted_config = adapt_config_for_environment(validated_config)
    
    return adapted_config


def print_config_summary(config: Dict[str, Any], title: str = "Configuration Summary"):
    """Print a formatted configuration summary."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    def print_section(section_dict: Dict[str, Any], indent: int = 0):
        for key, value in section_dict.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_section(value, indent + 1)
            else:
                print(f"{prefix}{key}: {value}")
    
    print_section(config)
    print(f"{'='*60}\n")


def export_config_for_reproducibility(
    config: Dict[str, Any],
    output_path: Union[str, Path],
    include_environment: bool = True
):
    """Export configuration with environment info for reproducibility."""
    from .io import save_config
    import sys
    import torch
    import numpy as np
    from datetime import datetime
    
    export_config = copy.deepcopy(config)
    
    if include_environment:
        export_config["_reproducibility_info"] = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "environment_variables": {
                k: v for k, v in os.environ.items() 
                if k.startswith(("CUDA_", "PYTORCH_", "OMP_", "MKL_"))
            }
        }
    
    save_config(export_config, output_path)
    print(f"✅ Exported configuration to {output_path}")


# Hydra integration utilities
if HYDRA_AVAILABLE:
    def omegaconf_to_dict(cfg: DictConfig) -> Dict[str, Any]:
        """Convert OmegaConf DictConfig to regular dict."""
        return OmegaConf.to_container(cfg, resolve=True)
    
    def dict_to_omegaconf(config: Dict[str, Any]) -> DictConfig:
        """Convert regular dict to OmegaConf DictConfig."""
        return OmegaConf.create(config)
    
    def merge_with_hydra_config(
        hydra_cfg: DictConfig,
        additional_config: Dict[str, Any]
    ) -> DictConfig:
        """Merge additional config with Hydra config."""
        base_dict = omegaconf_to_dict(hydra_cfg)
        merged_dict = _deep_merge(base_dict, additional_config)
        return dict_to_omegaconf(merged_dict)
