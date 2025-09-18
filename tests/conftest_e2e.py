"""
Pytest configuration and fixtures for end-to-end pipeline tests.

This module provides shared fixtures, markers, and configuration
for comprehensive E2E testing of the PKL-Guided Diffusion system.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator, Optional

import pytest
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Pytest markers for test organization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end pipeline test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running (>30s)"
    )
    config.addinivalue_line(
        "markers", "memory_intensive: mark test as memory intensive"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as performance benchmark"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark all E2E tests
        if "test_e2e_" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
        
        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in [
            "training", "inference", "integration", "workflow"
        ]):
            item.add_marker(pytest.mark.slow)
        
        # Mark memory intensive tests
        if any(keyword in item.name.lower() for keyword in [
            "memory", "scaling", "batch", "performance"
        ]):
            item.add_marker(pytest.mark.memory_intensive)
        
        # Mark GPU tests
        if "gpu" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Mark benchmarks
        if "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.benchmark)


@pytest.fixture(scope="session")
def project_root():
    """Project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session") 
def test_device():
    """Determine test device (CPU/CUDA)."""
    if torch.cuda.is_available() and os.getenv("USE_GPU_TESTS", "false").lower() == "true":
        return "cuda"
    return "cpu"


@pytest.fixture(scope="class")
def temp_workspace():
    """Create temporary workspace for testing."""
    temp_dir = tempfile.mkdtemp(prefix="pkl_e2e_workspace_")
    workspace = Path(temp_dir)
    
    # Create standard directory structure
    directories = [
        "data/train/classless",
        "data/val/classless",
        "data/synth/train",
        "data/synth/val", 
        "checkpoints",
        "outputs",
        "logs",
        "inference_input",
        "inference_output"
    ]
    
    for dir_path in directories:
        (workspace / dir_path).mkdir(parents=True, exist_ok=True)
    
    yield workspace
    
    # Cleanup
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture(scope="class")
def minimal_test_config(temp_workspace, test_device):
    """Minimal configuration optimized for fast testing."""
    config = {
        "experiment": {
            "name": "e2e_test",
            "seed": 42,
            "device": test_device,
            "mixed_precision": False
        },
        "paths": {
            "data": str(temp_workspace / "data"),
            "checkpoints": str(temp_workspace / "checkpoints"),
            "logs": str(temp_workspace / "logs"),
            "outputs": str(temp_workspace / "outputs")
        },
        "model": {
            "sample_size": 32,
            "in_channels": 1,
            "out_channels": 1,
            "layers_per_block": 1,
            "block_out_channels": [32, 32, 64],
            "down_block_types": ["DownBlock2D", "DownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "UpBlock2D", "UpBlock2D"],
            "attention_head_dim": 8
        },
        "data": {
            "image_size": 32,
            "min_intensity": 0.0,
            "max_intensity": 1000.0
        },
        "physics": {
            "background": 10.0,
            "psf_path": None
        },
        "training": {
            "num_timesteps": 50,
            "max_epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "num_workers": 0,
            "use_ema": False,
            "beta_schedule": "cosine",
            "accumulate_grad_batches": 1,
            "gradient_clip": 0.0
        },
        "guidance": {
            "type": "pkl",
            "lambda_base": 0.1,
            "epsilon": 1e-6,
            "schedule": {
                "T_threshold": 40,
                "epsilon_lambda": 1e-3
            }
        },
        "inference": {
            "ddim_steps": 5,
            "eta": 0.0,
            "use_autocast": False,
            "checkpoint_path": None,
            "input_dir": None,
            "output_dir": None
        },
        "wandb": {
            "mode": "disabled"
        }
    }
    
    return OmegaConf.create(config)


@pytest.fixture(scope="class")
def standard_test_config(temp_workspace, test_device):
    """Standard configuration for more realistic testing."""
    config = {
        "experiment": {
            "name": "e2e_standard_test",
            "seed": 42,
            "device": test_device,
            "mixed_precision": test_device == "cuda"
        },
        "paths": {
            "data": str(temp_workspace / "data"),
            "checkpoints": str(temp_workspace / "checkpoints"),
            "logs": str(temp_workspace / "logs"),
            "outputs": str(temp_workspace / "outputs")
        },
        "model": {
            "sample_size": 64,
            "in_channels": 1,
            "out_channels": 1,
            "layers_per_block": 2,
            "block_out_channels": [64, 128, 256],
            "down_block_types": ["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
            "up_block_types": ["AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
            "attention_head_dim": 8
        },
        "data": {
            "image_size": 64,
            "min_intensity": 0.0,
            "max_intensity": 1000.0
        },
        "physics": {
            "background": 10.0,
            "psf_path": None
        },
        "training": {
            "num_timesteps": 200,
            "max_epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "num_workers": 0,
            "use_ema": True,
            "beta_schedule": "cosine",
            "accumulate_grad_batches": 1,
            "gradient_clip": 1.0
        },
        "guidance": {
            "type": "pkl",
            "lambda_base": 0.1,
            "epsilon": 1e-6,
            "schedule": {
                "T_threshold": 150,
                "epsilon_lambda": 1e-3
            }
        },
        "inference": {
            "ddim_steps": 20,
            "eta": 0.0,
            "use_autocast": test_device == "cuda",
            "checkpoint_path": None,
            "input_dir": None,
            "output_dir": None
        },
        "wandb": {
            "mode": "disabled"
        }
    }
    
    return OmegaConf.create(config)


@pytest.fixture
def synthetic_image_generator():
    """Generator for creating synthetic test images."""
    def _generate_images(
        output_dir: Path, 
        num_images: int = 5,
        image_size: int = 32,
        format: str = "png"
    ) -> list[Path]:
        """Generate synthetic test images."""
        output_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []
        
        np.random.seed(42)  # Reproducible images
        
        for i in range(num_images):
            # Create synthetic microscopy-like image
            image = np.zeros((image_size, image_size), dtype=np.float32)
            
            # Add bright spots (simulating cells/structures)
            num_spots = np.random.randint(2, 6)
            for _ in range(num_spots):
                y, x = np.random.randint(5, image_size-5, 2)
                intensity = np.random.uniform(100, 255)
                size = np.random.uniform(1.0, 3.0)
                
                # Create Gaussian spot
                y_grid, x_grid = np.meshgrid(
                    np.arange(image_size) - y,
                    np.arange(image_size) - x,
                    indexing='ij'
                )
                spot = intensity * np.exp(-(y_grid**2 + x_grid**2) / (2 * size**2))
                image += spot
            
            # Add background and noise
            image += np.random.uniform(5, 15)  # Background
            image += np.random.normal(0, 5, image.shape)  # Noise
            image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Save image
            if format.lower() == "png":
                from PIL import Image as PILImage
                img_path = output_dir / f"test_image_{i:03d}.png"
                PILImage.fromarray(image).save(str(img_path))
            elif format.lower() in ["tif", "tiff"]:
                import tifffile
                img_path = output_dir / f"test_image_{i:03d}.tif"
                tifffile.imwrite(str(img_path), image.astype(np.float32))
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            image_paths.append(img_path)
        
        return image_paths
    
    return _generate_images


@pytest.fixture
def measurement_generator():
    """Generator for creating test measurement data."""
    def _generate_measurements(
        output_dir: Path,
        num_measurements: int = 3,
        image_size: int = 32,
        noise_level: float = 0.1,
        background: float = 10.0
    ) -> list[Path]:
        """Generate synthetic measurement data."""
        output_dir.mkdir(parents=True, exist_ok=True)
        measurement_paths = []
        
        np.random.seed(123)  # Different seed from images
        
        for i in range(num_measurements):
            # Create noisy measurement (simulating WF microscopy)
            measurement = np.random.poisson(
                lam=50, size=(image_size, image_size)
            ).astype(np.float32)
            
            # Add background
            measurement += background
            
            # Add Gaussian noise
            measurement += np.random.normal(0, noise_level * background, measurement.shape)
            
            # Ensure non-negative
            measurement = np.clip(measurement, 0, None)
            
            # Save as TIFF
            import tifffile
            tiff_path = output_dir / f"measurement_{i:03d}.tif"
            tifffile.imwrite(str(tiff_path), measurement)
            measurement_paths.append(tiff_path)
        
        return measurement_paths
    
    return _generate_measurements


@pytest.fixture
def performance_monitor():
    """Monitor for tracking test performance metrics."""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
            self.start_times = {}
        
        def start_timer(self, name: str):
            """Start timing a operation."""
            self.start_times[name] = time.time()
        
        def end_timer(self, name: str) -> float:
            """End timing and return elapsed time."""
            if name not in self.start_times:
                raise ValueError(f"Timer '{name}' was not started")
            
            elapsed = time.time() - self.start_times[name]
            self.metrics[name] = elapsed
            del self.start_times[name]
            return elapsed
        
        def get_memory_usage(self) -> Dict[str, float]:
            """Get current memory usage."""
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_mb = memory_info.rss / 1024 / 1024
            
            gpu_memory = None
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            
            return {
                "cpu_memory_mb": memory_mb,
                "gpu_memory_mb": gpu_memory
            }
        
        def get_metrics(self) -> Dict[str, Any]:
            """Get all collected metrics."""
            return {
                "timing": self.metrics.copy(),
                "memory": self.get_memory_usage()
            }
    
    import time
    return PerformanceMonitor()


# Skip conditions for different test types
def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )


def skip_if_slow_tests_disabled():
    """Skip test if slow tests are disabled."""
    return pytest.mark.skipif(
        os.getenv("SKIP_SLOW_TESTS", "false").lower() == "true",
        reason="Slow tests disabled"
    )


def skip_if_memory_limited():
    """Skip test if system has limited memory."""
    import psutil
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    return pytest.mark.skipif(
        total_memory_gb < 8,
        reason="Insufficient system memory"
    )


# Pytest hooks for test reporting
def pytest_runtest_setup(item):
    """Setup for each test."""
    # Print test name for verbose output
    if hasattr(item.config.option, 'verbose') and item.config.option.verbose:
        print(f"\n🧪 Starting: {item.name}")


def pytest_runtest_teardown(item, nextitem):
    """Teardown for each test."""
    # Force garbage collection after each test
    import gc
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    if hasattr(session.config.option, 'verbose') and session.config.option.verbose:
        print(f"\n📊 Test session finished with exit status: {exitstatus}")
