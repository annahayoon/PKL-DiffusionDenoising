"""
Adaptive Batch Sizing Utilities for Memory-Efficient Training

This module provides utilities for automatically determining optimal batch sizes
and preventing out-of-memory (OOM) errors during training and inference.
"""

import torch
import torch.nn as nn
import time
import warnings
from typing import Dict, Any, Optional, Tuple, Callable, Union
import math


class AdaptiveBatchSizer:
    """Adaptive batch sizing utility for automatic OOM prevention and optimization.
    
    This class automatically determines the optimal batch size for a given model,
    input size, and available GPU memory. It includes safety margins and can
    adapt to different precision modes and gradient checkpointing settings.
    """
    
    def __init__(
        self,
        safety_factor: float = 0.8,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        test_iterations: int = 3,
        verbose: bool = True,
    ):
        """Initialize adaptive batch sizer.
        
        Args:
            safety_factor: Safety margin (0.8 = use 80% of available memory)
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            test_iterations: Number of test iterations for memory measurement
            verbose: Whether to print progress information
        """
        self.safety_factor = safety_factor
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.test_iterations = test_iterations
        self.verbose = verbose
        
        # Cache for storing batch size results
        self._cache = {}
        
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information in GB."""
        if not torch.cuda.is_available():
            return {"total": 0, "allocated": 0, "free": 0}
        
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        free = total - allocated
        
        return {
            "total": total,
            "allocated": allocated, 
            "free": free,
        }
    
    def get_current_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information (alias for get_gpu_memory_info)."""
        gpu_info = self.get_gpu_memory_info()
        
        # Add CPU memory info
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            gpu_info.update({
                "cpu_memory": memory_info.rss / 1e9,
                "cpu_percent": process.memory_percent(),
            })
        except ImportError:
            gpu_info.update({
                "cpu_memory": 0.0,
                "cpu_percent": 0.0,
            })
        
        # Rename for consistency
        gpu_info["gpu_allocated"] = gpu_info.pop("allocated")
        gpu_info["gpu_reserved"] = gpu_info.get("total", 0) - gpu_info.get("free", 0)
        gpu_info["gpu_free"] = gpu_info.pop("free")
        gpu_info["gpu_total"] = gpu_info.pop("total")
        
        return gpu_info
    
    def estimate_memory_per_sample(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        mixed_precision: bool = False,
        gradient_checkpointing: bool = False,
        device: str = "cuda",
    ) -> float:
        """Estimate memory usage per sample in GB.
        
        Args:
            model: The model to test
            input_shape: Input shape (C, H, W) without batch dimension
            mixed_precision: Whether mixed precision is enabled
            gradient_checkpointing: Whether gradient checkpointing is enabled
            device: Device to test on
            
        Returns:
            Estimated memory per sample in GB
        """
        if not torch.cuda.is_available():
            return 0.1  # Fallback estimate for CPU
        
        # Create cache key
        cache_key = (
            id(model), input_shape, mixed_precision, 
            gradient_checkpointing, device
        )
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        model = model.to(device)
        model.train()
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Test with batch size 1
            test_input = torch.randn(1, *input_shape, device=device, requires_grad=True)
            
            if hasattr(model, 'enable_gradient_checkpointing') and gradient_checkpointing:
                model.enable_gradient_checkpointing()
            
            # Forward pass
            if mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    output = model(test_input, torch.zeros(1, device=device, dtype=torch.long))
            else:
                output = model(test_input, torch.zeros(1, device=device, dtype=torch.long))
            
            # Backward pass
            loss = output.mean()
            loss.backward()
            
            # Measure peak memory
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            
            # Clean up
            del test_input, output, loss
            if hasattr(model, 'disable_gradient_checkpointing'):
                model.disable_gradient_checkpointing()
            torch.cuda.empty_cache()
            
            # Add overhead factor (activations, optimizer states, etc.)
            overhead_factor = 1.5 if not gradient_checkpointing else 1.2
            memory_per_sample = peak_memory * overhead_factor
            
            # Cache result
            self._cache[cache_key] = memory_per_sample
            
            if self.verbose:
                print(f"   Estimated memory per sample: {memory_per_sample:.3f} GB")
            
            return memory_per_sample
            
        except Exception as e:
            if self.verbose:
                print(f"   Memory estimation failed: {e}")
            # Fallback estimate based on input size
            return self._fallback_memory_estimate(input_shape, mixed_precision)
    
    def _fallback_memory_estimate(
        self, 
        input_shape: Tuple[int, ...], 
        mixed_precision: bool = False
    ) -> float:
        """Fallback memory estimation based on input size."""
        # Rough estimate: memory scales with input size
        total_elements = 1
        for dim in input_shape:
            total_elements *= dim
        
        # Base memory per element (bytes)
        bytes_per_element = 2 if mixed_precision else 4
        
        # Estimate total memory (input + activations + gradients + overhead)
        total_memory = total_elements * bytes_per_element * 10  # 10x overhead
        
        return total_memory / 1e9  # Convert to GB
    
    def find_optimal_batch_size(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        mixed_precision: bool = False,
        gradient_checkpointing: bool = False,
        device: str = "cuda",
        target_memory_usage: Optional[float] = None,
    ) -> int:
        """Find optimal batch size for given constraints.
        
        Args:
            model: The model to test
            input_shape: Input shape (C, H, W) without batch dimension
            mixed_precision: Whether mixed precision is enabled
            gradient_checkpointing: Whether gradient checkpointing is enabled
            device: Device to test on
            target_memory_usage: Target memory usage in GB (None = auto)
            
        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            if self.verbose:
                print("   CUDA not available, using batch size 4")
            return 4
        
        memory_info = self.get_gpu_memory_info()
        available_memory = memory_info["free"]
        
        if target_memory_usage is None:
            target_memory_usage = available_memory * self.safety_factor
        
        if self.verbose:
            print(f"   Available GPU memory: {available_memory:.2f} GB")
            print(f"   Target memory usage: {target_memory_usage:.2f} GB")
        
        # Estimate memory per sample
        memory_per_sample = self.estimate_memory_per_sample(
            model, input_shape, mixed_precision, gradient_checkpointing, device
        )
        
        if memory_per_sample == 0:
            return self.min_batch_size
        
        # Calculate theoretical maximum batch size
        theoretical_max = int(target_memory_usage / memory_per_sample)
        theoretical_max = max(self.min_batch_size, min(theoretical_max, self.max_batch_size))
        
        if self.verbose:
            print(f"   Theoretical max batch size: {theoretical_max}")
        
        # Binary search for actual maximum batch size
        optimal_batch_size = self._binary_search_batch_size(
            model, input_shape, theoretical_max, mixed_precision, 
            gradient_checkpointing, device
        )
        
        return optimal_batch_size
    
    def optimize_for_training(
        self,
        model: nn.Module,
        dataset,
        target_epochs: int = 1,
        mixed_precision: bool = False,
        gradient_checkpointing: bool = False,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Optimize batch size and data loading for training.
        
        Args:
            model: Model to optimize for
            dataset: Training dataset
            target_epochs: Number of training epochs
            mixed_precision: Whether using mixed precision
            gradient_checkpointing: Whether using gradient checkpointing
            device: Device to optimize for
            
        Returns:
            Dictionary with optimization recommendations
        """
        if self.verbose:
            print("🔧 Optimizing training configuration...")
        
        # Get sample from dataset
        try:
            sample = dataset[0]
            if isinstance(sample, (tuple, list)):
                input_shape = sample[0].shape if hasattr(sample[0], 'shape') else (1, 256, 256)
            else:
                input_shape = sample.shape if hasattr(sample, 'shape') else (1, 256, 256)
        except Exception:
            input_shape = (1, 256, 256)  # Default fallback
        
        # Get optimal batch size
        optimal_batch_size = self.get_optimal_batch_size(
            model=model,
            input_shape=input_shape,
            mixed_precision=mixed_precision,
            gradient_checkpointing=gradient_checkpointing,
            device=device,
        )
        
        # Calculate data loading parameters
        dataset_size = len(dataset)
        steps_per_epoch = math.ceil(dataset_size / optimal_batch_size)
        total_steps = steps_per_epoch * target_epochs
        
        # Optimize number of workers
        import os
        max_workers = min(8, os.cpu_count() or 1)
        if dataset_size < 1000:
            optimal_workers = min(2, max_workers)
        elif dataset_size < 10000:
            optimal_workers = min(4, max_workers)
        else:
            optimal_workers = max_workers
        
        # Memory-based prefetch factor
        memory_info = self.get_current_memory_info()
        available_memory_gb = memory_info.get("gpu_free", 8.0)
        
        if available_memory_gb > 16:
            prefetch_factor = 4
        elif available_memory_gb > 8:
            prefetch_factor = 2
        else:
            prefetch_factor = 1
        
        recommendations = {
            "batch_size": optimal_batch_size,
            "num_workers": optimal_workers,
            "prefetch_factor": prefetch_factor,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": optimal_workers > 0,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "estimated_memory_per_sample_gb": self.estimate_memory_per_sample(
                model, input_shape, mixed_precision, gradient_checkpointing, device
            ),
            "memory_info": memory_info,
        }
        
        if self.verbose:
            print(f"✅ Optimization complete:")
            print(f"   Batch size: {optimal_batch_size}")
            print(f"   Workers: {optimal_workers}")
            print(f"   Steps per epoch: {steps_per_epoch}")
            print(f"   Memory per sample: {recommendations['estimated_memory_per_sample_gb']:.3f} GB")
        
        return recommendations
    
    def create_optimized_dataloader(self, dataset, **optimization_kwargs):
        """Create an optimized DataLoader using the recommendations.
        
        Args:
            dataset: PyTorch dataset
            **optimization_kwargs: Arguments for optimize_for_training
            
        Returns:
            Optimized DataLoader
        """
        # Import here to avoid circular imports
        from ..data.dataset import create_optimized_dataloader
        
        # Get optimization recommendations
        recommendations = self.optimize_for_training(dataset=dataset, **optimization_kwargs)
        
        # Create optimized dataloader
        return create_optimized_dataloader(
            dataset=dataset,
            batch_size=recommendations["batch_size"],
            num_workers=recommendations["num_workers"],
            prefetch_factor=recommendations["prefetch_factor"],
            pin_memory=recommendations["pin_memory"],
            persistent_workers=recommendations["persistent_workers"],
        )
    
    def _binary_search_batch_size(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        max_batch_size: int,
        mixed_precision: bool,
        gradient_checkpointing: bool,
        device: str,
    ) -> int:
        """Binary search to find maximum working batch size."""
        model = model.to(device)
        
        if hasattr(model, 'enable_gradient_checkpointing') and gradient_checkpointing:
            model.enable_gradient_checkpointing()
        
        low, high = self.min_batch_size, max_batch_size
        best_batch_size = self.min_batch_size
        
        while low <= high:
            mid = (low + high) // 2
            
            if self._test_batch_size(model, input_shape, mid, mixed_precision, device):
                best_batch_size = mid
                low = mid + 1
                if self.verbose:
                    print(f"   ✅ Batch size {mid}: OK")
            else:
                high = mid - 1
                if self.verbose:
                    print(f"   ❌ Batch size {mid}: OOM")
        
        if hasattr(model, 'disable_gradient_checkpointing'):
            model.disable_gradient_checkpointing()
        
        if self.verbose:
            print(f"   🎯 Optimal batch size: {best_batch_size}")
        
        return best_batch_size
    
    def _test_batch_size(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int,
        mixed_precision: bool,
        device: str,
    ) -> bool:
        """Test if a specific batch size works without OOM."""
        try:
            model.train()
            
            # Clear memory
            torch.cuda.empty_cache()
            
            # Create test batch
            test_input = torch.randn(batch_size, *input_shape, device=device, requires_grad=True)
            test_timesteps = torch.randint(0, 1000, (batch_size,), device=device)
            
            # Test forward and backward pass
            for _ in range(self.test_iterations):
                if mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        output = model(test_input, test_timesteps)
                else:
                    output = model(test_input, test_timesteps)
                
                loss = output.mean()
                loss.backward()
                
                # Clear gradients
                test_input.grad = None
                for param in model.parameters():
                    param.grad = None
                
                torch.cuda.synchronize()
            
            # Clean up
            del test_input, test_timesteps, output, loss
            torch.cuda.empty_cache()
            
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False
            else:
                # Other error, re-raise
                raise e
    
    def get_recommended_config(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Get recommended configuration for optimal performance.
        
        Returns a dictionary with recommended settings for batch size,
        mixed precision, and gradient checkpointing based on available memory.
        """
        if not torch.cuda.is_available():
            return {
                "batch_size": 4,
                "mixed_precision": False,
                "gradient_checkpointing": False,
                "reason": "CUDA not available",
            }
        
        memory_info = self.get_gpu_memory_info()
        
        # Test different configurations
        configs = [
            {"mixed_precision": False, "gradient_checkpointing": False},
            {"mixed_precision": True, "gradient_checkpointing": False},
            {"mixed_precision": False, "gradient_checkpointing": True},
            {"mixed_precision": True, "gradient_checkpointing": True},
        ]
        
        best_config = None
        best_batch_size = 0
        
        for config in configs:
            try:
                batch_size = self.find_optimal_batch_size(
                    model, input_shape, 
                    config["mixed_precision"], 
                    config["gradient_checkpointing"], 
                    device
                )
                
                if batch_size > best_batch_size:
                    best_batch_size = batch_size
                    best_config = config.copy()
                    best_config["batch_size"] = batch_size
                
            except Exception as e:
                if self.verbose:
                    print(f"   Config {config} failed: {e}")
                continue
        
        if best_config is None:
            return {
                "batch_size": self.min_batch_size,
                "mixed_precision": False,
                "gradient_checkpointing": True,
                "reason": "Fallback configuration",
            }
        
        # Add performance reasoning
        if best_config["mixed_precision"] and best_config["gradient_checkpointing"]:
            best_config["reason"] = "Maximum memory efficiency"
        elif best_config["mixed_precision"]:
            best_config["reason"] = "Balanced speed and memory"
        elif best_config["gradient_checkpointing"]:
            best_config["reason"] = "Memory over speed"
        else:
            best_config["reason"] = "Maximum speed"
        
        return best_config


class AdaptiveDataLoader:
    """DataLoader wrapper with adaptive batch sizing capabilities."""
    
    def __init__(
        self,
        dataset,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_sizer: Optional[AdaptiveBatchSizer] = None,
        device: str = "cuda",
        **dataloader_kwargs
    ):
        """Initialize adaptive DataLoader.
        
        Args:
            dataset: Dataset to load from
            model: Model for batch size optimization
            input_shape: Input shape for memory estimation
            batch_sizer: Custom batch sizer (None = create default)
            device: Device for testing
            **dataloader_kwargs: Additional DataLoader arguments
        """
        self.dataset = dataset
        self.model = model
        self.input_shape = input_shape
        self.device = device
        
        if batch_sizer is None:
            batch_sizer = AdaptiveBatchSizer()
        self.batch_sizer = batch_sizer
        
        # Get optimal configuration
        self.config = self.batch_sizer.get_recommended_config(
            model, input_shape, device
        )
        
        # Update dataloader kwargs with optimal batch size
        dataloader_kwargs["batch_size"] = self.config["batch_size"]
        self.dataloader_kwargs = dataloader_kwargs
        
        # Create DataLoader
        self.dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
        
    def get_config(self) -> Dict[str, Any]:
        """Get the adaptive configuration used."""
        return self.config.copy()
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def get_optimal_batch_size(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cuda",
    mixed_precision: bool = False,
    gradient_checkpointing: bool = False,
    safety_factor: float = 0.8,
    verbose: bool = True,
) -> int:
    """Convenience function to get optimal batch size.
    
    Args:
        model: Model to test
        input_shape: Input shape (C, H, W) without batch dimension
        device: Device to test on
        mixed_precision: Whether mixed precision is enabled
        gradient_checkpointing: Whether gradient checkpointing is enabled
        safety_factor: Safety margin for memory usage
        verbose: Whether to print progress
        
    Returns:
        Optimal batch size
    """
    batch_sizer = AdaptiveBatchSizer(
        safety_factor=safety_factor,
        verbose=verbose
    )
    
    return batch_sizer.find_optimal_batch_size(
        model, input_shape, mixed_precision, gradient_checkpointing, device
    )


def create_adaptive_dataloader(
    dataset,
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cuda",
    **dataloader_kwargs
) -> Tuple[torch.utils.data.DataLoader, Dict[str, Any]]:
    """Create an adaptive DataLoader with optimal batch size.
    
    Args:
        dataset: Dataset to load from
        model: Model for optimization
        input_shape: Input shape for memory estimation
        device: Device for testing
        **dataloader_kwargs: Additional DataLoader arguments
        
    Returns:
        Tuple of (DataLoader, configuration_dict)
    """
    adaptive_loader = AdaptiveDataLoader(
        dataset, model, input_shape, device=device, **dataloader_kwargs
    )
    
    return adaptive_loader.dataloader, adaptive_loader.get_config()
