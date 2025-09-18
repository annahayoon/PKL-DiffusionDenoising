"""
Data Validation Utilities for PKL Diffusion Denoising

This module provides comprehensive validation utilities for input data,
model outputs, and configuration parameters.
"""

import torch
import numpy as np
from typing import Union, Tuple, Dict, Any, List, Optional
from pathlib import Path
import warnings


# Constants
UINT16_MAX = 65535
UINT16_MIN = 0


def get_16bit_image_stats(image: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
    """Get statistics for a 16-bit image.
    
    Args:
        image: 16-bit image array or tensor
        
    Returns:
        Dictionary with image statistics
    """
    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().numpy()
    else:
        image_np = image
    
    return {
        "min": float(image_np.min()),
        "max": float(image_np.max()),
        "mean": float(image_np.mean()),
        "std": float(image_np.std()),
        "shape": image_np.shape,
        "dtype": str(image_np.dtype),
        "dynamic_range": float(image_np.max() - image_np.min()),
        "is_16bit_range": image_np.min() >= 0 and image_np.max() <= UINT16_MAX,
        "has_negative": image_np.min() < 0,
        "has_overflow": image_np.max() > UINT16_MAX,
        "zero_pixels": int(np.sum(image_np == 0)),
        "saturated_pixels": int(np.sum(image_np >= UINT16_MAX)),
        "non_finite": int(np.sum(~np.isfinite(image_np)))
    }


def validate_16bit_image(
    image: Union[np.ndarray, torch.Tensor], 
    name: str = "image",
    strict: bool = True
) -> bool:
    """Validate that an image is in proper 16-bit range.
    
    Args:
        image: Image to validate
        name: Name for error messages
        strict: If True, raise errors; if False, issue warnings
        
    Returns:
        True if valid, raises ValueError if not (in strict mode)
    """
    if isinstance(image, torch.Tensor):
        min_val = image.min().item()
        max_val = image.max().item()
        has_nan = torch.isnan(image).any().item()
        has_inf = torch.isinf(image).any().item()
    else:
        min_val = image.min()
        max_val = image.max()
        has_nan = np.isnan(image).any()
        has_inf = np.isinf(image).any()
    
    issues = []
    
    if has_nan:
        issues.append(f"{name} contains NaN values")
    
    if has_inf:
        issues.append(f"{name} contains infinite values")
    
    if min_val < UINT16_MIN:
        issues.append(f"{name} has values below 16-bit range: min={min_val}")
    
    if max_val > UINT16_MAX:
        issues.append(f"{name} has values above 16-bit range: max={max_val}")
    
    if issues:
        message = "; ".join(issues)
        if strict:
            raise ValueError(message)
        else:
            warnings.warn(message)
            return False
    
    return True


def validate_model_input(
    x: torch.Tensor,
    expected_range: Tuple[float, float] = (-1.0, 1.0),
    name: str = "model_input",
    strict: bool = True
) -> bool:
    """Validate model input tensor.
    
    Args:
        x: Input tensor to validate
        expected_range: Expected value range
        name: Name for error messages
        strict: If True, raise errors; if False, issue warnings
        
    Returns:
        True if valid
    """
    issues = []
    
    if not isinstance(x, torch.Tensor):
        issues.append(f"{name} must be a torch.Tensor, got {type(x)}")
    else:
        if torch.isnan(x).any():
            issues.append(f"{name} contains NaN values")
        
        if torch.isinf(x).any():
            issues.append(f"{name} contains infinite values")
        
        min_val = x.min().item()
        max_val = x.max().item()
        
        if min_val < expected_range[0]:
            issues.append(f"{name} has values below expected range: min={min_val} < {expected_range[0]}")
        
        if max_val > expected_range[1]:
            issues.append(f"{name} has values above expected range: max={max_val} > {expected_range[1]}")
        
        # Check for reasonable dynamic range
        dynamic_range = max_val - min_val
        if dynamic_range < 0.01:
            issues.append(f"{name} has very low dynamic range: {dynamic_range}")
    
    if issues:
        message = "; ".join(issues)
        if strict:
            raise ValueError(message)
        else:
            warnings.warn(message)
            return False
    
    return True


def validate_batch_consistency(
    batch: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
    name: str = "batch",
    strict: bool = True
) -> bool:
    """Validate batch consistency (shapes, dtypes, devices).
    
    Args:
        batch: Batch to validate
        name: Name for error messages
        strict: If True, raise errors; if False, issue warnings
        
    Returns:
        True if valid
    """
    issues = []
    
    if isinstance(batch, torch.Tensor):
        tensors = [batch]
    elif isinstance(batch, (tuple, list)):
        tensors = batch
    else:
        issues.append(f"{name} must be tensor, tuple, or list")
        tensors = []
    
    if not tensors:
        if strict:
            raise ValueError(f"Empty batch: {name}")
        return False
    
    # Check all tensors are actually tensors
    for i, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            issues.append(f"{name}[{i}] is not a tensor: {type(tensor)}")
    
    if not issues:  # Only proceed if all are tensors
        # Check batch dimensions
        batch_sizes = [t.shape[0] for t in tensors]
        if len(set(batch_sizes)) > 1:
            issues.append(f"{name} has inconsistent batch sizes: {batch_sizes}")
        
        # Check devices
        devices = [str(t.device) for t in tensors]
        if len(set(devices)) > 1:
            issues.append(f"{name} has tensors on different devices: {devices}")
        
        # Check dtypes
        dtypes = [str(t.dtype) for t in tensors]
        if len(set(dtypes)) > 1:
            issues.append(f"{name} has inconsistent dtypes: {dtypes}")
        
        # Check for NaN/Inf
        for i, tensor in enumerate(tensors):
            if torch.isnan(tensor).any():
                issues.append(f"{name}[{i}] contains NaN values")
            if torch.isinf(tensor).any():
                issues.append(f"{name}[{i}] contains infinite values")
    
    if issues:
        message = "; ".join(issues)
        if strict:
            raise ValueError(message)
        else:
            warnings.warn(message)
            return False
    
    return True


def validate_checkpoint_compatibility(
    checkpoint_path: Union[str, Path],
    model: torch.nn.Module,
    strict: bool = True
) -> bool:
    """Validate checkpoint compatibility with model.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to check compatibility with
        strict: If True, raise errors; if False, issue warnings
        
    Returns:
        True if compatible
    """
    issues = []
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        issues.append(f"Checkpoint file does not exist: {checkpoint_path}")
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            if "model_state_dict" not in checkpoint:
                issues.append("Checkpoint missing 'model_state_dict' key")
            else:
                checkpoint_keys = set(checkpoint["model_state_dict"].keys())
                model_keys = set(model.state_dict().keys())
                
                missing_keys = model_keys - checkpoint_keys
                unexpected_keys = checkpoint_keys - model_keys
                
                if missing_keys:
                    issues.append(f"Missing keys in checkpoint: {sorted(list(missing_keys))[:5]}...")
                
                if unexpected_keys:
                    issues.append(f"Unexpected keys in checkpoint: {sorted(list(unexpected_keys))[:5]}...")
                
                # Check tensor shapes for common keys
                common_keys = model_keys & checkpoint_keys
                for key in list(common_keys)[:10]:  # Check first 10 keys
                    model_shape = model.state_dict()[key].shape
                    checkpoint_shape = checkpoint["model_state_dict"][key].shape
                    
                    if model_shape != checkpoint_shape:
                        issues.append(f"Shape mismatch for {key}: model {model_shape} vs checkpoint {checkpoint_shape}")
        
        except Exception as e:
            issues.append(f"Failed to load checkpoint: {str(e)}")
    
    if issues:
        message = "; ".join(issues)
        if strict:
            raise ValueError(message)
        else:
            warnings.warn(message)
            return False
    
    return True


def validate_dataset_structure(
    dataset_path: Union[str, Path],
    required_subdirs: Optional[List[str]] = None,
    required_files: Optional[List[str]] = None,
    strict: bool = True
) -> bool:
    """Validate dataset directory structure.
    
    Args:
        dataset_path: Path to dataset directory
        required_subdirs: List of required subdirectories
        required_files: List of required files
        strict: If True, raise errors; if False, issue warnings
        
    Returns:
        True if valid
    """
    issues = []
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        issues.append(f"Dataset path does not exist: {dataset_path}")
    elif not dataset_path.is_dir():
        issues.append(f"Dataset path is not a directory: {dataset_path}")
    else:
        # Check required subdirectories
        if required_subdirs:
            for subdir in required_subdirs:
                subdir_path = dataset_path / subdir
                if not subdir_path.exists():
                    issues.append(f"Required subdirectory missing: {subdir}")
                elif not subdir_path.is_dir():
                    issues.append(f"Required path is not a directory: {subdir}")
        
        # Check required files
        if required_files:
            for file_name in required_files:
                file_path = dataset_path / file_name
                if not file_path.exists():
                    issues.append(f"Required file missing: {file_name}")
                elif not file_path.is_file():
                    issues.append(f"Required path is not a file: {file_name}")
    
    if issues:
        message = "; ".join(issues)
        if strict:
            raise ValueError(message)
        else:
            warnings.warn(message)
            return False
    
    return True


def validate_training_config(config: Dict[str, Any], strict: bool = True) -> bool:
    """Validate training configuration.
    
    Args:
        config: Configuration dictionary
        strict: If True, raise errors; if False, issue warnings
        
    Returns:
        True if valid
    """
    issues = []
    
    # Required top-level keys
    required_keys = ["model", "training"]
    for key in required_keys:
        if key not in config:
            issues.append(f"Required config section missing: {key}")
    
    # Model configuration validation
    if "model" in config:
        model_config = config["model"]
        required_model_keys = ["sample_size", "in_channels", "out_channels"]
        
        for key in required_model_keys:
            if key not in model_config:
                issues.append(f"Required model config missing: {key}")
            elif not isinstance(model_config[key], int) or model_config[key] <= 0:
                issues.append(f"Model config {key} must be positive integer")
    
    # Training configuration validation
    if "training" in config:
        training_config = config["training"]
        required_training_keys = ["learning_rate", "batch_size", "num_epochs"]
        
        for key in required_training_keys:
            if key not in training_config:
                issues.append(f"Required training config missing: {key}")
        
        # Validate specific training parameters
        if "learning_rate" in training_config:
            lr = training_config["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0 or lr >= 1:
                issues.append("learning_rate must be between 0 and 1")
        
        if "batch_size" in training_config:
            bs = training_config["batch_size"]
            if not isinstance(bs, int) or bs <= 0:
                issues.append("batch_size must be positive integer")
        
        if "num_epochs" in training_config:
            epochs = training_config["num_epochs"]
            if not isinstance(epochs, int) or epochs <= 0:
                issues.append("num_epochs must be positive integer")
    
    if issues:
        message = "; ".join(issues)
        if strict:
            raise ValueError(f"Configuration validation failed: {message}")
        else:
            warnings.warn(f"Configuration issues: {message}")
            return False
    
    return True


def validate_gpu_memory_requirements(
    model: torch.nn.Module,
    batch_size: int,
    image_size: int,
    safety_factor: float = 0.8,
    strict: bool = True
) -> bool:
    """Validate GPU memory requirements.
    
    Args:
        model: Model to check
        batch_size: Training batch size
        image_size: Input image size
        safety_factor: Safety factor for memory usage
        strict: If True, raise errors; if False, issue warnings
        
    Returns:
        True if requirements can be met
    """
    issues = []
    
    if not torch.cuda.is_available():
        issues.append("CUDA not available")
    else:
        # Get available memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = total_memory * safety_factor
        
        # Estimate memory requirements
        # This is a rough estimate - actual usage may vary
        param_memory = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per float32
        
        # Estimate activation memory (very rough)
        input_size = batch_size * 1 * image_size * image_size * 4  # Input tensor
        activation_memory = input_size * 10  # Rough multiplier for activations
        
        # Gradient memory (same as parameters)
        grad_memory = param_memory
        
        # Optimizer memory (AdamW needs ~2x parameter memory)
        optimizer_memory = param_memory * 2
        
        total_required = param_memory + activation_memory + grad_memory + optimizer_memory
        
        if total_required > available_memory:
            issues.append(
                f"Estimated memory requirement ({total_required / 1e9:.1f} GB) "
                f"exceeds available memory ({available_memory / 1e9:.1f} GB)"
            )
        
        # Check if batch size is reasonable
        if batch_size > 64:
            issues.append(f"Large batch size ({batch_size}) may cause memory issues")
    
    if issues:
        message = "; ".join(issues)
        if strict:
            raise ValueError(message)
        else:
            warnings.warn(message)
            return False
    
    return True


def validate_inference_inputs(
    source_images: torch.Tensor,
    model: torch.nn.Module,
    strict: bool = True
) -> bool:
    """Validate inputs for inference.
    
    Args:
        source_images: Input images for inference
        model: Model to use for inference
        strict: If True, raise errors; if False, issue warnings
        
    Returns:
        True if valid
    """
    issues = []
    
    # Validate input tensor
    if not isinstance(source_images, torch.Tensor):
        issues.append(f"source_images must be torch.Tensor, got {type(source_images)}")
    else:
        # Check dimensions
        if source_images.ndim != 4:
            issues.append(f"source_images must be 4D (B,C,H,W), got {source_images.ndim}D")
        
        # Check for NaN/Inf
        if torch.isnan(source_images).any():
            issues.append("source_images contains NaN values")
        
        if torch.isinf(source_images).any():
            issues.append("source_images contains infinite values")
        
        # Check value range (assuming model input range [-1, 1])
        min_val = source_images.min().item()
        max_val = source_images.max().item()
        
        if min_val < -1.1 or max_val > 1.1:  # Allow small tolerance
            issues.append(f"source_images values outside expected range [-1,1]: [{min_val:.3f}, {max_val:.3f}]")
    
    # Check model state
    if model.training:
        issues.append("Model is in training mode, should be in eval mode for inference")
    
    if issues:
        message = "; ".join(issues)
        if strict:
            raise ValueError(message)
        else:
            warnings.warn(message)
            return False
    
    return True


class DataValidator:
    """Comprehensive data validator for PKL diffusion training."""
    
    def __init__(self, strict: bool = True):
        """Initialize validator.
        
        Args:
            strict: If True, raise errors; if False, issue warnings
        """
        self.strict = strict
        self.validation_history = []
        
    def validate_all(
        self,
        images: Union[torch.Tensor, np.ndarray],
        config: Dict[str, Any],
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, bool]:
        """Run all validations and return results.
        
        Args:
            images: Input images to validate
            config: Configuration to validate
            model: Model to validate (optional)
            checkpoint_path: Checkpoint path to validate (optional)
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Image validation
        try:
            results["images"] = validate_16bit_image(images, strict=self.strict)
        except Exception as e:
            results["images"] = False
            if self.strict:
                raise
        
        # Config validation
        try:
            results["config"] = validate_training_config(config, strict=self.strict)
        except Exception as e:
            results["config"] = False
            if self.strict:
                raise
        
        # Model validation
        if model is not None:
            try:
                if isinstance(images, torch.Tensor):
                    batch_size = images.shape[0]
                    image_size = images.shape[-1]
                else:
                    batch_size = config.get("training", {}).get("batch_size", 8)
                    image_size = config.get("model", {}).get("sample_size", 256)
                
                results["gpu_memory"] = validate_gpu_memory_requirements(
                    model, batch_size, image_size, strict=self.strict
                )
            except Exception as e:
                results["gpu_memory"] = False
                if self.strict:
                    raise
        
        # Checkpoint validation
        if checkpoint_path is not None and model is not None:
            try:
                results["checkpoint"] = validate_checkpoint_compatibility(
                    checkpoint_path, model, strict=self.strict
                )
            except Exception as e:
                results["checkpoint"] = False
                if self.strict:
                    raise
        
        self.validation_history.append(results)
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed."""
        if not self.validation_history:
            return {"message": "No validations performed"}
        
        latest = self.validation_history[-1]
        total_checks = len(latest)
        passed_checks = sum(latest.values())
        
        return {
            "total_validations": len(self.validation_history),
            "latest_results": latest,
            "latest_summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "success_rate": passed_checks / total_checks if total_checks > 0 else 0
            }
        }
