"""
Consolidated Utilities for PKL Diffusion Denoising

This module consolidates all core utilities into a single file:
- I/O operations and checkpoint management (from io.py)
- Memory profiling and monitoring (from memory.py) 
- Adaptive batch sizing (from adaptive_batch.py)
- Configuration management (from config.py)
"""

import os
import gc
import json
import yaml
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
import copy
import shutil
import socket
import psutil
import warnings
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple, Callable
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field

# Optional dependencies
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from omegaconf import OmegaConf, DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False


# =============================================================================
# I/O UTILITIES (from io.py)
# =============================================================================

class CheckpointManager:
    """Manages model checkpoints with automatic cleanup and versioning."""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best: bool = True,
        monitor_metric: str = "val/loss",
        mode: str = "min"
    ):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.checkpoint_history = []
        
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save a checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch:04d}_step_{step:06d}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics or {},
            "timestamp": timestamp,
        }
        
        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
            
        if extra_state is not None:
            checkpoint_data["extra_state"] = extra_state
            
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        self.checkpoint_history.append(checkpoint_path)
        
        # Check if this is the best checkpoint
        if self.save_best and metrics and self.monitor_metric in metrics:
            metric_value = metrics[self.monitor_metric]
            is_best = (
                (self.mode == "min" and metric_value < self.best_value) or
                (self.mode == "max" and metric_value > self.best_value)
            )
            
            if is_best:
                self.best_value = metric_value
                best_path = self.checkpoint_dir / "best_model.pt"
                shutil.copy2(checkpoint_path, best_path)
                
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        # Load scheduler state if provided
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        return checkpoint
    
    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu"
    ) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint if it exists."""
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            return self.load_checkpoint(best_path, model, optimizer, scheduler, device)
        return None
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        if len(self.checkpoint_history) > self.max_checkpoints:
            # Sort by modification time
            self.checkpoint_history.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest checkpoints
            while len(self.checkpoint_history) > self.max_checkpoints:
                old_checkpoint = self.checkpoint_history.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()


class PathManager:
    """Centralized path management for the project."""
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        if project_root is None:
            # Try to find project root automatically
            current = Path.cwd()
            while current != current.parent:
                if (current / "pkl_dg").exists():
                    project_root = current
                    break
                current = current.parent
            else:
                project_root = Path.cwd()
                
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"
        self.outputs_dir = self.project_root / "outputs"
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.configs_dir = self.project_root / "configs"
        
    def ensure_dirs(self):
        """Create all necessary directories."""
        for dir_path in [self.data_dir, self.logs_dir, self.outputs_dir, 
                        self.checkpoints_dir, self.configs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def get_run_dir(self, run_name: str) -> Path:
        """Get directory for a specific run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.outputs_dir / f"{run_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


class ConfigManager:
    """Enhanced configuration management utilities."""
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
                
        return config or {}
        
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
        """Save configuration to YAML or JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
                
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries."""
        result = base_config.copy()
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager.merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result


class DeviceManager:
    """Device management utilities for CUDA/CPU operations."""
    
    @staticmethod
    def get_device(prefer_cuda: bool = True) -> torch.device:
        """Get the best available device."""
        if prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
            
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get detailed device information."""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cpu_count": os.cpu_count(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_current_device": torch.cuda.current_device(),
                "cuda_device_name": torch.cuda.get_device_name(),
                "cuda_memory_allocated": torch.cuda.memory_allocated(),
                "cuda_memory_reserved": torch.cuda.memory_reserved(),
            })
            
        return info
        
    @staticmethod
    def cleanup_cuda_memory():
        """Clean up CUDA memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class Timer:
    """Simple timer utility for performance monitoring."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        self.end_time = time.time()
        if self.start_time is None:
            raise RuntimeError("Timer was not started")
        return self.end_time - self.start_time
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, *args):
        elapsed = self.stop()
        print(f"Elapsed time: {elapsed:.4f} seconds")


class Logger:
    """Enhanced logger with file and console output."""
    
    def __init__(self, name: str, log_file: Optional[Union[str, Path]] = None, level: int = 20):  # INFO level
        import logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
    def info(self, message: str):
        self.logger.info(message)
        
    def warning(self, message: str):
        self.logger.warning(message)
        
    def error(self, message: str):
        self.logger.error(message)
        
    def debug(self, message: str):
        self.logger.debug(message)


class IOManager:
    """Enhanced I/O utilities for various image formats."""
    
    @staticmethod
    def load_image(file_path: Union[str, Path], 
                  as_gray: bool = True) -> np.ndarray:
        """Load image from file with format detection."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
            
        # Try TIFF format first (common for microscopy)
        if TIFFFILE_AVAILABLE and file_path.suffix.lower() in ['.tif', '.tiff']:
            image = tifffile.imread(file_path)
            if len(image.shape) > 2 and as_gray:
                # Convert to grayscale if needed
                if image.shape[2] == 3:  # RGB
                    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                elif image.shape[2] == 4:  # RGBA
                    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
            return image
            
        # Try PIL for other formats
        elif PIL_AVAILABLE:
            pil_image = Image.open(file_path)
            if as_gray and pil_image.mode != 'L':
                pil_image = pil_image.convert('L')
            image = np.array(pil_image)
            return image
            
        else:
            raise ImportError("Either tifffile or PIL is required for image loading")
            
    @staticmethod
    def save_image(image: np.ndarray, 
                  file_path: Union[str, Path],
                  format_hint: Optional[str] = None) -> None:
        """Save image to file with format detection."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Detect format from extension or hint
        if format_hint:
            format_to_use = format_hint.lower()
        else:
            format_to_use = file_path.suffix.lower()
            
        # Save as TIFF for 16-bit images
        if format_to_use in ['.tif', '.tiff'] or image.dtype == np.uint16:
            if not TIFFFILE_AVAILABLE:
                raise ImportError("tifffile is required for TIFF saving")
            tifffile.imwrite(file_path, image)
            
        # Save using PIL for other formats
        elif PIL_AVAILABLE:
            if image.dtype == np.uint16:
                # Convert to 8-bit for PIL
                image_8bit = (image / 256).astype(np.uint8)
            else:
                image_8bit = image
                
            pil_image = Image.fromarray(image_8bit)
            pil_image.save(file_path)
            
        else:
            raise ImportError("Either tifffile or PIL is required for image saving")


# Convenience functions
def setup_logging(
    log_dir: Union[str, Path],
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
):
    """Setup logging configuration."""
    import logging
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("pkl_diffusion")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add file handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def setup_wandb(
    project_name: str,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None
) -> Optional[Any]:
    """Setup Weights & Biases logging."""
    if not WANDB_AVAILABLE:
        print("⚠️ wandb not available, skipping W&B setup")
        return None
    
    try:
        run = wandb.init(
            project=project_name,
            config=config,
            name=run_name,
            tags=tags,
            notes=notes
        )
        return run
    except Exception as e:
        print(f"⚠️ Failed to initialize wandb: {e}")
        return None


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_json(data: Dict[str, Any], file_path: Union[str, Path]):
    """Save data as JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, file_path: Union[str, Path]):
    """Save object using pickle."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path: Union[str, Path]) -> Any:
    """Load object using pickle."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def get_environment_info() -> Dict[str, Any]:
    """Get comprehensive environment information."""
    return {
        "hostname": socket.gethostname(),
        "python_version": os.sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# MEMORY UTILITIES (from memory.py)
# =============================================================================

@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point in time."""
    timestamp: float
    gpu_allocated: float  # GB
    gpu_reserved: float   # GB
    gpu_free: float      # GB
    gpu_total: float     # GB
    cpu_memory: float    # GB
    cpu_percent: float   # %
    step: Optional[int] = None
    phase: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class MemoryProfile:
    """Complete memory profiling results."""
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    peak_gpu_allocated: float = 0.0
    peak_gpu_reserved: float = 0.0
    peak_cpu_memory: float = 0.0
    total_duration: float = 0.0
    average_gpu_usage: float = 0.0
    memory_efficiency: float = 0.0  # allocated / reserved ratio
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the memory profile."""
        if not self.snapshots:
            return {"error": "No snapshots recorded"}
        
        gpu_allocated = [s.gpu_allocated for s in self.snapshots]
        gpu_reserved = [s.gpu_reserved for s in self.snapshots]
        cpu_memory = [s.cpu_memory for s in self.snapshots]
        
        return {
            "duration_seconds": self.total_duration,
            "num_snapshots": len(self.snapshots),
            "gpu_memory": {
                "peak_allocated_gb": self.peak_gpu_allocated,
                "peak_reserved_gb": self.peak_gpu_reserved,
                "average_allocated_gb": sum(gpu_allocated) / len(gpu_allocated),
                "average_reserved_gb": sum(gpu_reserved) / len(gpu_reserved),
                "efficiency_percent": self.memory_efficiency * 100,
            },
            "cpu_memory": {
                "peak_gb": self.peak_cpu_memory,
                "average_gb": sum(cpu_memory) / len(cpu_memory),
            },
            "memory_growth": {
                "gpu_allocated_growth_gb": gpu_allocated[-1] - gpu_allocated[0] if len(gpu_allocated) > 1 else 0,
                "gpu_reserved_growth_gb": gpu_reserved[-1] - gpu_reserved[0] if len(gpu_reserved) > 1 else 0,
                "cpu_growth_gb": cpu_memory[-1] - cpu_memory[0] if len(cpu_memory) > 1 else 0,
            }
        }


class MemoryProfiler:
    """Advanced memory profiler for deep learning training."""
    
    def __init__(
        self,
        interval: float = 1.0,
        track_cpu: bool = True,
        track_gpu: bool = True,
        auto_cleanup: bool = True,
        verbose: bool = False,
    ):
        """Initialize memory profiler."""
        self.interval = interval
        self.track_cpu = track_cpu
        self.track_gpu = track_gpu and torch.cuda.is_available()
        self.auto_cleanup = auto_cleanup
        self.verbose = verbose
        
        self.profile = MemoryProfile()
        self.start_time = None
        self.monitoring = False
        self.monitor_thread = None
        
        # Memory leak detection
        self.baseline_snapshot = None
        self.leak_threshold_gb = 0.1  # 100MB threshold for leak detection
        
    def get_current_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        info = {}
        
        if self.track_gpu and torch.cuda.is_available():
            # GPU memory in GB
            info.update({
                "gpu_allocated": torch.cuda.memory_allocated() / 1e9,
                "gpu_reserved": torch.cuda.memory_reserved() / 1e9,
                "gpu_free": (torch.cuda.get_device_properties(0).total_memory - 
                           torch.cuda.memory_reserved()) / 1e9,
                "gpu_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
            })
        else:
            info.update({
                "gpu_allocated": 0.0,
                "gpu_reserved": 0.0,
                "gpu_free": 0.0,
                "gpu_total": 0.0,
            })
        
        if self.track_cpu:
            # CPU memory in GB
            process = psutil.Process()
            memory_info = process.memory_info()
            info.update({
                "cpu_memory": memory_info.rss / 1e9,
                "cpu_percent": process.memory_percent(),
            })
        else:
            info.update({
                "cpu_memory": 0.0,
                "cpu_percent": 0.0,
            })
        
        return info
    
    def take_snapshot(
        self, 
        step: Optional[int] = None, 
        phase: Optional[str] = None,
        notes: Optional[str] = None
    ) -> MemorySnapshot:
        """Take a memory snapshot."""
        memory_info = self.get_current_memory_info()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            step=step,
            phase=phase,
            notes=notes,
            **memory_info
        )
        
        self.profile.snapshots.append(snapshot)
        
        # Update peak values
        self.profile.peak_gpu_allocated = max(
            self.profile.peak_gpu_allocated, snapshot.gpu_allocated
        )
        self.profile.peak_gpu_reserved = max(
            self.profile.peak_gpu_reserved, snapshot.gpu_reserved
        )
        self.profile.peak_cpu_memory = max(
            self.profile.peak_cpu_memory, snapshot.cpu_memory
        )
        
        if self.verbose:
            print(f"📊 Memory snapshot: GPU {snapshot.gpu_allocated:.2f}GB allocated, "
                  f"CPU {snapshot.cpu_memory:.2f}GB, Step {step}")
        
        return snapshot
    
    @contextmanager
    def profile_context(self, name: str = "operation"):
        """Context manager for profiling a specific operation."""
        start_snapshot = self.take_snapshot(phase=f"{name}_start")
        start_time = time.time()
        
        try:
            yield self
        finally:
            end_snapshot = self.take_snapshot(phase=f"{name}_end")
            duration = time.time() - start_time
            
            gpu_used = end_snapshot.gpu_allocated - start_snapshot.gpu_allocated
            cpu_used = end_snapshot.cpu_memory - start_snapshot.cpu_memory
            
            if self.verbose:
                print(f"🔍 {name}: {duration:.2f}s, "
                      f"GPU: {gpu_used:+.3f}GB, CPU: {cpu_used:+.3f}GB")


@contextmanager
def profile_memory(
    name: str = "operation",
    interval: float = 0.5,
    verbose: bool = True,
    auto_cleanup: bool = True,
) -> MemoryProfiler:
    """Convenience context manager for memory profiling."""
    profiler = MemoryProfiler(
        interval=interval,
        verbose=verbose,
        auto_cleanup=auto_cleanup,
    )
    
    with profiler.profile_context(name):
        yield profiler


def get_memory_summary() -> Dict[str, Any]:
    """Get current memory usage summary."""
    profiler = MemoryProfiler(verbose=False)
    memory_info = profiler.get_current_memory_info()
    
    summary = {
        "timestamp": time.time(),
        "gpu_available": torch.cuda.is_available(),
    }
    summary.update(memory_info)
    
    if torch.cuda.is_available():
        summary["gpu_utilization_percent"] = (
            memory_info["gpu_allocated"] / memory_info["gpu_total"] * 100
            if memory_info["gpu_total"] > 0 else 0
        )
    
    return summary


def cleanup_memory(verbose: bool = True):
    """Perform aggressive memory cleanup."""
    if verbose:
        before = get_memory_summary()
        print("🧹 Cleaning up memory...")
    
    # Python garbage collection
    collected = gc.collect()
    
    # PyTorch GPU cache cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if verbose:
        after = get_memory_summary()
        gpu_freed = before.get("gpu_allocated", 0) - after.get("gpu_allocated", 0)
        cpu_freed = before.get("cpu_memory", 0) - after.get("cpu_memory", 0)
        
        print(f"✅ Cleanup complete: {collected} objects collected, "
              f"GPU: {gpu_freed:+.3f}GB, CPU: {cpu_freed:+.3f}GB freed")


# =============================================================================
# ADAPTIVE BATCH SIZING (from adaptive_batch.py)
# =============================================================================

class AdaptiveBatchSizer:
    """Advanced adaptive batch sizing utility for automatic OOM prevention and optimization."""
    
    def __init__(
        self,
        safety_factor: float = 0.8,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        test_iterations: int = 3,
        verbose: bool = True,
        enable_dynamic_scaling: bool = True,
        memory_pressure_threshold: float = 0.9,
        performance_target_utilization: float = 0.85,
    ):
        """Initialize advanced adaptive batch sizer."""
        self.safety_factor = safety_factor
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.test_iterations = test_iterations
        self.verbose = verbose
        self.enable_dynamic_scaling = enable_dynamic_scaling
        self.memory_pressure_threshold = memory_pressure_threshold
        self.performance_target_utilization = performance_target_utilization
        
        # Cache for storing batch size results
        self._cache = {}
        
        # Dynamic scaling state
        self._current_batch_size = None
        self._performance_history = []
        self._memory_history = []
        self._oom_count = 0
        self._last_adjustment_step = 0
        
        # Performance monitoring
        self._timing_history = []
        self._throughput_history = []
        
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
    
    def estimate_memory_per_sample(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        mixed_precision: bool = False,
        gradient_checkpointing: bool = False,
        device: str = "cuda",
        for_training: bool = True,
    ) -> float:
        """Estimate memory usage per sample in GB."""
        if not torch.cuda.is_available():
            return 0.1  # Fallback estimate for CPU
        
        # Create cache key
        cache_key = (
            id(model), input_shape, mixed_precision, 
            gradient_checkpointing, device, for_training
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
            
            # Backward pass for training memory estimation
            if for_training:
                if not output.requires_grad:
                    output = output.detach().requires_grad_(True)
                
                loss = output.mean()
                try:
                    loss.backward()
                except RuntimeError as e:
                    if "does not require grad" in str(e):
                        if self.verbose:
                            print("   Warning: Cannot measure gradient memory, using forward-pass only")
                    else:
                        raise e
            
            # Measure peak memory
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            
            # Clean up
            del test_input, output
            if 'loss' in locals():
                del loss
            if hasattr(model, 'disable_gradient_checkpointing'):
                model.disable_gradient_checkpointing()
            torch.cuda.empty_cache()
            
            # Add overhead factor
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
                print(f"   Using fallback memory estimation...")
            
            # Clean up on error
            try:
                if hasattr(model, 'disable_gradient_checkpointing'):
                    model.disable_gradient_checkpointing()
                torch.cuda.empty_cache()
            except Exception:
                pass
            
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
        """Find optimal batch size for given constraints."""
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
            model, input_shape, mixed_precision, gradient_checkpointing, device, for_training=True
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
                
                # Only compute gradients if output requires them
                if output.requires_grad:
                    loss = output.mean()
                    loss.backward()
                else:
                    # Just measure forward pass memory if no gradients
                    loss = output.mean().detach()
                
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


def get_optimal_batch_size(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cuda",
    mixed_precision: bool = False,
    gradient_checkpointing: bool = False,
    safety_factor: float = 0.8,
    verbose: bool = True,
) -> int:
    """Convenience function to get optimal batch size."""
    batch_sizer = AdaptiveBatchSizer(
        safety_factor=safety_factor,
        verbose=verbose
    )
    
    return batch_sizer.find_optimal_batch_size(
        model, input_shape, mixed_precision, gradient_checkpointing, device
    )


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
        """Initialize adaptive DataLoader."""
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
        ) if hasattr(batch_sizer, 'get_recommended_config') else {
            "batch_size": batch_sizer.find_optimal_batch_size(model, input_shape, device=device)
        }
        
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


def create_adaptive_dataloader(
    dataset,
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cuda",
    enable_dynamic_scaling: bool = False,
    **dataloader_kwargs
) -> torch.utils.data.DataLoader:
    """Create an adaptive DataLoader with optimal batch size."""
    adaptive_loader = AdaptiveDataLoader(
        dataset, model, input_shape, device=device, **dataloader_kwargs
    )
    return adaptive_loader.dataloader


# =============================================================================
# CONFIGURATION UTILITIES (from config.py)
# =============================================================================

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
        "training.learning_rate",
        lambda x: 0 < x < 1,
        "learning_rate must be between 0 and 1"
    )
    
    validator.add_validation_rule(
        "training.batch_size",
        lambda x: x > 0,
        "batch_size must be positive"
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


def validate_and_complete_config(
    config: Dict[str, Any],
    config_type: str = "training"
) -> Dict[str, Any]:
    """Validate and complete configuration with defaults."""
    if config_type == "training":
        validator = create_training_config_validator()
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    # Validate and add defaults
    validated_config = validator.validate(config)
    
    return validated_config


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


# =============================================================================
# 16-BIT MICROSCOPY PATCH PROCESSING - Import from utils_16bit
# =============================================================================

# Import 16-bit processing functions from utils_16bit.py to avoid duplication
try:
    from .utils_16bit import save_16bit_patch, extract_and_save_patches_16bit
except ImportError:
    # Fallback implementations if utils_16bit is not available
    def save_16bit_patch(*args, **kwargs):
        raise ImportError("utils_16bit module not available")
    
    def extract_and_save_patches_16bit(*args, **kwargs):
        raise ImportError("utils_16bit module not available")


def process_microscopy_dataset_16bit(
    wf_image_path: Union[str, Path],
    tp_image_path: Union[str, Path], 
    output_dir: Union[str, Path],
    patch_size: int = 256,
    stride: int = None,
    train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    min_intensity_threshold: float = None,
    preserve_range: bool = True,
    verbose: bool = True
) -> Dict[str, List[str]]:
    """Process paired WF/2P microscopy images into 16-bit TIFF patches.
    
    Args:
        wf_image_path: Path to wide-field TIFF image
        tp_image_path: Path to two-photon TIFF image
        output_dir: Output directory for processed patches
        patch_size: Size of patches to extract
        stride: Stride between patches (defaults to patch_size)
        train_val_test_split: Tuple of (train, val, test) split ratios
        min_intensity_threshold: Minimum intensity to keep a patch
        preserve_range: Whether to preserve original intensity ranges
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with lists of patch paths for each split
    """
    if not TIFFFILE_AVAILABLE:
        raise ImportError("tifffile is required for processing microscopy images. Install with: pip install tifffile")
    
    output_dir = Path(output_dir)
    
    if verbose:
        print(f"🔬 Processing microscopy dataset to 16-bit TIFF patches")
        print(f"WF image: {wf_image_path}")
        print(f"2P image: {tp_image_path}")
        print(f"Output: {output_dir}")
    
    # Load images
    wf_image = tifffile.imread(str(wf_image_path)).astype(np.float32)
    tp_image = tifffile.imread(str(tp_image_path)).astype(np.float32)
    
    if verbose:
        print(f"WF image shape: {wf_image.shape}, dtype: {wf_image.dtype}")
        print(f"2P image shape: {tp_image.shape}, dtype: {tp_image.dtype}")
        print(f"WF range: [{wf_image.min():.1f}, {wf_image.max():.1f}]")
        print(f"2P range: [{tp_image.min():.1f}, {tp_image.max():.1f}]")
    
    # Handle multi-frame images
    if wf_image.ndim == 3:
        if verbose:
            print(f"Processing {wf_image.shape[0]} frames...")
        
        all_patches = {"train": [], "val": [], "test": []}
        
        for frame_idx in range(wf_image.shape[0]):
            wf_frame = wf_image[frame_idx]
            tp_frame = tp_image[frame_idx] if tp_image.ndim == 3 else tp_image
            
            # Extract patches for this frame
            frame_output_dir = output_dir / f"frame_{frame_idx:03d}"
            
            wf_patches = extract_and_save_patches_16bit(
                wf_frame,
                frame_output_dir / "wf",
                patch_size=patch_size,
                stride=stride,
                prefix=f"wf_frame{frame_idx:03d}",
                preserve_range=preserve_range,
                min_intensity_threshold=min_intensity_threshold,
                verbose=False
            )
            
            tp_patches = extract_and_save_patches_16bit(
                tp_frame,
                frame_output_dir / "2p", 
                patch_size=patch_size,
                stride=stride,
                prefix=f"tp_frame{frame_idx:03d}",
                preserve_range=preserve_range,
                min_intensity_threshold=min_intensity_threshold,
                verbose=False
            )
            
            # Ensure matching pairs
            if len(wf_patches) != len(tp_patches):
                print(f"Warning: Frame {frame_idx} has mismatched patch counts: WF={len(wf_patches)}, 2P={len(tp_patches)}")
                min_patches = min(len(wf_patches), len(tp_patches))
                wf_patches = wf_patches[:min_patches]
                tp_patches = tp_patches[:min_patches]
            
            # Split patches for this frame
            num_patches = len(wf_patches)
            train_end = int(num_patches * train_val_test_split[0])
            val_end = train_end + int(num_patches * train_val_test_split[1])
            
            # Random shuffle for better distribution
            indices = np.random.permutation(num_patches)
            
            all_patches["train"].extend([(wf_patches[i], tp_patches[i]) for i in indices[:train_end]])
            all_patches["val"].extend([(wf_patches[i], tp_patches[i]) for i in indices[train_end:val_end]])
            all_patches["test"].extend([(wf_patches[i], tp_patches[i]) for i in indices[val_end:]])
            
            if verbose:
                print(f"  Frame {frame_idx}: {len(wf_patches)} patch pairs")
        
        # Create final split directories and move patches
        splits_dir = output_dir / "splits"
        result_paths = {"train": [], "val": [], "test": []}
        
        for split_name, patch_pairs in all_patches.items():
            wf_split_dir = splits_dir / split_name / "wf"
            tp_split_dir = splits_dir / split_name / "2p"
            wf_split_dir.mkdir(parents=True, exist_ok=True)
            tp_split_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, (wf_path, tp_path) in enumerate(patch_pairs):
                # Copy to final split directory
                new_wf_path = wf_split_dir / f"{split_name}_wf_{idx:06d}.tif"
                new_tp_path = tp_split_dir / f"{split_name}_tp_{idx:06d}.tif"
                
                shutil.copy2(wf_path, new_wf_path)
                shutil.copy2(tp_path, new_tp_path)
                
                result_paths[split_name].append((str(new_wf_path), str(new_tp_path)))
        
        if verbose:
            print(f"✅ Final splits:")
            for split_name, pairs in result_paths.items():
                print(f"  {split_name}: {len(pairs)} patch pairs")
        
        return result_paths
    
    else:
        # Single frame processing
        wf_patches = extract_and_save_patches_16bit(
            wf_image,
            output_dir / "wf",
            patch_size=patch_size,
            stride=stride,
            prefix="wf",
            preserve_range=preserve_range,
            min_intensity_threshold=min_intensity_threshold,
            verbose=verbose
        )
        
        tp_patches = extract_and_save_patches_16bit(
            tp_image,
            output_dir / "2p",
            patch_size=patch_size,
            stride=stride,
            prefix="tp",
            preserve_range=preserve_range,
            min_intensity_threshold=min_intensity_threshold,
            verbose=verbose
        )
        
        return {"wf": wf_patches, "2p": tp_patches}


# =============================================================================
# CONSOLIDATED EXPORTS
# =============================================================================

__all__ = [
    # I/O utilities
    "CheckpointManager",
    "PathManager", 
    "ConfigManager",
    "DeviceManager",
    "Timer",
    "Logger",
    "IOManager",
    "setup_logging",
    "setup_wandb",
    "ensure_dir",
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
    "get_environment_info",
    
    # Memory utilities
    "MemorySnapshot",
    "MemoryProfile",
    "MemoryProfiler",
    "profile_memory",
    "get_memory_summary",
    "cleanup_memory",
    
    # Adaptive batch sizing
    "AdaptiveBatchSizer",
    "AdaptiveDataLoader",
    "get_optimal_batch_size",
    "create_adaptive_dataloader",
    
    # Configuration utilities
    "ConfigValidator",
    "create_training_config_validator",
    "merge_configs",
    "validate_and_complete_config",
    "print_config_summary",
    
    # 16-bit microscopy patch processing
    "save_16bit_patch",
    "extract_and_save_patches_16bit", 
    "process_microscopy_dataset_16bit",
]
