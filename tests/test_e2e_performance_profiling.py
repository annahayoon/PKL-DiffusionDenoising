"""
Performance profiling and benchmarking tests for PKL-Guided Diffusion.

This module provides detailed performance analysis including:
1. Memory usage profiling
2. Execution time benchmarking  
3. GPU/CPU utilization analysis
4. Scalability testing
5. Bottleneck identification
"""

import gc
import time
import psutil
import threading
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from pkl_dg.models.sampler import DDIMSampler
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.models.unet import UNet
from pkl_dg.guidance import PKLGuidance, L2Guidance, AnscombeGuidance, AdaptiveSchedule
from pkl_dg.physics import ForwardModel, PSF
from pkl_dg.data import IntensityToModel, RealPairsDataset


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float
    peak_memory_mb: float
    avg_memory_mb: float
    cpu_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None


class MemoryProfiler:
    """Context manager for memory profiling."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.start_memory = 0
        self.peak_memory = 0
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def __enter__(self):
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()
        else:
            self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        gc.collect()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _monitor_memory(self):
        """Monitor memory usage in background thread."""
        while self.monitoring:
            if self.device == "cuda" and torch.cuda.is_available():
                current = torch.cuda.memory_allocated() / 1024 / 1024
                peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                current = psutil.Process().memory_info().rss / 1024 / 1024
                peak = current
            
            self.memory_samples.append(current)
            self.peak_memory = max(self.peak_memory, peak)
            time.sleep(0.1)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get memory profiling metrics."""
        if not self.memory_samples:
            return {"peak_mb": 0, "avg_mb": 0, "delta_mb": 0}
        
        avg_memory = sum(self.memory_samples) / len(self.memory_samples)
        delta_memory = self.peak_memory - self.start_memory
        
        return {
            "peak_mb": self.peak_memory,
            "avg_mb": avg_memory,
            "delta_mb": max(0, delta_memory)
        }


@contextmanager
def performance_timer():
    """Context manager for timing execution."""
    start_time = time.time()
    start_cpu = psutil.Process().cpu_percent()
    
    yield
    
    end_time = time.time()
    end_cpu = psutil.Process().cpu_percent()
    
    execution_time = end_time - start_time
    cpu_usage = (start_cpu + end_cpu) / 2
    
    return execution_time, cpu_usage


class TestPerformanceProfiling:
    """Performance profiling tests."""
    
    @pytest.fixture(scope="class")
    def temp_workspace(self):
        """Create temporary workspace."""
        temp_dir = tempfile.mkdtemp(prefix="pkl_perf_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture(scope="class")
    def minimal_model_components(self):
        """Create minimal model components for testing."""
        # UNet configuration for testing
        unet_config = {
            "sample_size": 64,
            "in_channels": 1,
            "out_channels": 1,
            "layers_per_block": 1,
            "block_out_channels": [32, 64, 128],
            "down_block_types": ["DownBlock2D", "DownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "UpBlock2D", "UpBlock2D"],
            "attention_head_dim": 8
        }
        
        # Training configuration
        training_config = {
            "num_timesteps": 100,
            "learning_rate": 1e-4,
            "use_ema": False,
            "beta_schedule": "cosine"
        }
        
        # Create components
        unet = UNet(unet_config)
        trainer = DDPMTrainer(model=unet, config=training_config)
        
        psf = PSF()
        forward_model = ForwardModel(
            psf=psf.to_torch(device="cpu"),
            background=10.0,
            device="cpu"
        )
        
        guidance = PKLGuidance(epsilon=1e-6)
        schedule = AdaptiveSchedule(
            lambda_base=0.1,
            T_threshold=80,
            epsilon_lambda=1e-3,
            T_total=100
        )
        
        transform = IntensityToModel(min_intensity=0.0, max_intensity=1000.0)
        
        return {
            "trainer": trainer,
            "forward_model": forward_model,
            "guidance": guidance,
            "schedule": schedule,
            "transform": transform
        }
    
    def test_inference_memory_scaling(self, minimal_model_components):
        """Test memory usage scaling with image size."""
        components = minimal_model_components
        
        # Test different image sizes
        image_sizes = [32, 64, 128]
        memory_results = {}
        
        for size in image_sizes:
            # Create test measurement
            y = torch.randn(1, 1, size, size) * 10 + 50
            y = torch.clamp(y, 0, None)
            
            # Create sampler
            sampler = DDIMSampler(
                model=components["trainer"],
                forward_model=components["forward_model"],
                guidance_strategy=components["guidance"],
                schedule=components["schedule"],
                transform=components["transform"],
                num_timesteps=50,  # Reduced for faster testing
                ddim_steps=5,
                eta=0.0,
            )
            
            # Profile memory usage
            with MemoryProfiler(device="cpu") as profiler:
                result = sampler.sample(
                    y=y,
                    shape=y.shape,
                    device="cpu",
                    verbose=False,
                    return_intermediates=False
                )
            
            memory_metrics = profiler.get_metrics()
            memory_results[size] = memory_metrics
            
            # Verify result
            assert result.shape == y.shape
            assert torch.isfinite(result).all()
        
        # Analyze memory scaling
        print("\nMemory Scaling Results:")
        for size, metrics in memory_results.items():
            print(f"Size {size}x{size}: Peak={metrics['peak_mb']:.1f}MB, "
                  f"Avg={metrics['avg_mb']:.1f}MB, Delta={metrics['delta_mb']:.1f}MB")
        
        # Memory usage should increase with image size
        sizes = sorted(memory_results.keys())
        for i in range(1, len(sizes)):
            prev_size, curr_size = sizes[i-1], sizes[i]
            prev_memory = memory_results[prev_size]['peak_mb']
            curr_memory = memory_results[curr_size]['peak_mb']
            
            # Memory should increase (allowing for some measurement noise)
            assert curr_memory >= prev_memory * 0.8, \
                f"Memory didn't scale as expected: {prev_size}x{prev_size}={prev_memory}MB, " \
                f"{curr_size}x{curr_size}={curr_memory}MB"
    
    def test_guidance_strategy_performance(self, minimal_model_components):
        """Compare performance of different guidance strategies."""
        components = minimal_model_components
        
        # Test measurement
        y = torch.randn(1, 1, 64, 64) * 10 + 50
        y = torch.clamp(y, 0, None)
        
        guidance_strategies = {
            "pkl": PKLGuidance(epsilon=1e-6),
            "l2": L2Guidance(),
            "anscombe": AnscombeGuidance(epsilon=1e-6)
        }
        
        performance_results = {}
        
        for name, guidance in guidance_strategies.items():
            sampler = DDIMSampler(
                model=components["trainer"],
                forward_model=components["forward_model"],
                guidance_strategy=guidance,
                schedule=components["schedule"],
                transform=components["transform"],
                num_timesteps=50,
                ddim_steps=10,
                eta=0.0,
            )
            
            # Profile performance
            with MemoryProfiler(device="cpu") as profiler:
                start_time = time.time()
                
                result = sampler.sample(
                    y=y,
                    shape=y.shape,
                    device="cpu",
                    verbose=False
                )
                
                end_time = time.time()
            
            execution_time = end_time - start_time
            memory_metrics = profiler.get_metrics()
            
            performance_results[name] = {
                "execution_time": execution_time,
                "peak_memory": memory_metrics["peak_mb"],
                "avg_memory": memory_metrics["avg_mb"]
            }
            
            # Verify result
            assert result.shape == y.shape
            assert torch.isfinite(result).all()
        
        # Report performance comparison
        print("\nGuidance Strategy Performance:")
        for name, metrics in performance_results.items():
            print(f"{name}: Time={metrics['execution_time']:.2f}s, "
                  f"Peak Memory={metrics['peak_memory']:.1f}MB")
        
        # All strategies should complete in reasonable time
        for name, metrics in performance_results.items():
            assert metrics["execution_time"] < 120, \
                f"{name} guidance took too long: {metrics['execution_time']:.2f}s"
    
    def test_batch_size_scaling(self, minimal_model_components):
        """Test performance scaling with batch size."""
        components = minimal_model_components
        
        batch_sizes = [1, 2, 4]
        scaling_results = {}
        
        for batch_size in batch_sizes:
            # Create batch measurement
            y = torch.randn(batch_size, 1, 32, 32) * 10 + 50
            y = torch.clamp(y, 0, None)
            
            sampler = DDIMSampler(
                model=components["trainer"],
                forward_model=components["forward_model"],
                guidance_strategy=components["guidance"],
                schedule=components["schedule"],
                transform=components["transform"],
                num_timesteps=50,
                ddim_steps=5,  # Fewer steps for faster testing
                eta=0.0,
            )
            
            # Profile batch processing
            results = []
            total_time = 0
            peak_memory = 0
            
            with MemoryProfiler(device="cpu") as profiler:
                start_time = time.time()
                
                # Process each item in batch individually
                for i in range(batch_size):
                    y_single = y[i:i+1]
                    result = sampler.sample(
                        y=y_single,
                        shape=y_single.shape,
                        device="cpu",
                        verbose=False
                    )
                    results.append(result)
                
                end_time = time.time()
                total_time = end_time - start_time
            
            memory_metrics = profiler.get_metrics()
            
            scaling_results[batch_size] = {
                "total_time": total_time,
                "time_per_sample": total_time / batch_size,
                "peak_memory": memory_metrics["peak_mb"]
            }
            
            # Verify all results
            assert len(results) == batch_size
            for result in results:
                assert result.shape == (1, 1, 32, 32)
                assert torch.isfinite(result).all()
        
        # Report scaling results
        print("\nBatch Size Scaling:")
        for batch_size, metrics in scaling_results.items():
            print(f"Batch {batch_size}: Total={metrics['total_time']:.2f}s, "
                  f"Per Sample={metrics['time_per_sample']:.2f}s, "
                  f"Memory={metrics['peak_memory']:.1f}MB")
        
        # Time per sample should be relatively consistent
        times_per_sample = [metrics["time_per_sample"] for metrics in scaling_results.values()]
        min_time = min(times_per_sample)
        max_time = max(times_per_sample)
        
        # Allow for some variation but shouldn't be dramatically different
        assert max_time / min_time < 2.0, \
            f"Time per sample varies too much: {min_time:.2f}s to {max_time:.2f}s"
    
    def test_ddim_steps_performance_tradeoff(self, minimal_model_components):
        """Test performance vs quality tradeoff with different DDIM steps."""
        components = minimal_model_components
        
        # Test measurement
        y = torch.randn(1, 1, 64, 64) * 10 + 50
        y = torch.clamp(y, 0, None)
        
        ddim_steps_list = [5, 10, 20, 50]
        tradeoff_results = {}
        
        for ddim_steps in ddim_steps_list:
            sampler = DDIMSampler(
                model=components["trainer"],
                forward_model=components["forward_model"],
                guidance_strategy=components["guidance"],
                schedule=components["schedule"],
                transform=components["transform"],
                num_timesteps=100,
                ddim_steps=ddim_steps,
                eta=0.0,
            )
            
            # Profile performance
            with MemoryProfiler(device="cpu") as profiler:
                start_time = time.time()
                
                result = sampler.sample(
                    y=y,
                    shape=y.shape,
                    device="cpu",
                    verbose=False
                )
                
                end_time = time.time()
            
            execution_time = end_time - start_time
            memory_metrics = profiler.get_metrics()
            
            # Simple quality metric (variance as proxy for detail)
            result_variance = torch.var(result).item()
            
            tradeoff_results[ddim_steps] = {
                "execution_time": execution_time,
                "peak_memory": memory_metrics["peak_mb"],
                "result_variance": result_variance
            }
            
            # Verify result
            assert result.shape == y.shape
            assert torch.isfinite(result).all()
        
        # Report tradeoff analysis
        print("\nDDIM Steps Performance Tradeoff:")
        for steps, metrics in tradeoff_results.items():
            print(f"{steps} steps: Time={metrics['execution_time']:.2f}s, "
                  f"Memory={metrics['peak_memory']:.1f}MB, "
                  f"Variance={metrics['result_variance']:.3f}")
        
        # Execution time should generally increase with more steps
        steps_list = sorted(tradeoff_results.keys())
        for i in range(1, len(steps_list)):
            prev_steps, curr_steps = steps_list[i-1], steps_list[i]
            prev_time = tradeoff_results[prev_steps]["execution_time"]
            curr_time = tradeoff_results[curr_steps]["execution_time"]
            
            # More steps should generally take more time (with some tolerance)
            assert curr_time >= prev_time * 0.8, \
                f"Execution time didn't increase with steps: {prev_steps}={prev_time:.2f}s, " \
                f"{curr_steps}={curr_time:.2f}s"
    
    def test_memory_leak_detection(self, minimal_model_components):
        """Test for memory leaks during repeated inference."""
        components = minimal_model_components
        
        # Create sampler
        sampler = DDIMSampler(
            model=components["trainer"],
            forward_model=components["forward_model"],
            guidance_strategy=components["guidance"],
            schedule=components["schedule"],
            transform=components["transform"],
            num_timesteps=50,
            ddim_steps=5,
            eta=0.0,
        )
        
        # Test measurement
        y = torch.randn(1, 1, 32, 32) * 10 + 50
        y = torch.clamp(y, 0, None)
        
        # Monitor memory usage over multiple runs
        memory_samples = []
        num_runs = 10
        
        for run in range(num_runs):
            # Force garbage collection before measurement
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Measure memory before inference
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated() / 1024 / 1024
            else:
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Run inference
            result = sampler.sample(
                y=y,
                shape=y.shape,
                device="cpu",
                verbose=False
            )
            
            # Force cleanup
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Measure memory after cleanup
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024 / 1024
            else:
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            memory_samples.append(memory_after - memory_before)
        
        # Analyze memory leak
        avg_memory_growth = sum(memory_samples) / len(memory_samples)
        max_memory_growth = max(memory_samples)
        
        print(f"\nMemory Leak Analysis:")
        print(f"Average memory growth per run: {avg_memory_growth:.2f}MB")
        print(f"Maximum memory growth: {max_memory_growth:.2f}MB")
        print(f"Memory samples: {[f'{m:.1f}' for m in memory_samples]}")
        
        # Memory growth should be minimal (allowing for some measurement noise)
        assert avg_memory_growth < 5.0, \
            f"Potential memory leak detected: {avg_memory_growth:.2f}MB average growth"
        assert max_memory_growth < 20.0, \
            f"Large memory spike detected: {max_memory_growth:.2f}MB maximum growth"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_performance_comparison(self, minimal_model_components):
        """Compare CPU vs GPU performance (if available)."""
        components = minimal_model_components
        
        # Test measurement
        y = torch.randn(1, 1, 64, 64) * 10 + 50
        y = torch.clamp(y, 0, None)
        
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        
        device_results = {}
        
        for device in devices:
            # Move components to device
            trainer = components["trainer"].to(device)
            
            # Clear dynamic caches to avoid device mismatch
            def clear_block_caches(block):
                if hasattr(block, '_norm_cache'):
                    block._norm_cache.clear()
                if hasattr(block, '_conv_cache'):
                    block._conv_cache.clear()
                if hasattr(block, '_skip_cache'):
                    block._skip_cache.clear()
            
            if hasattr(trainer, 'model'):
                model = trainer.model
                # Clear down blocks
                if hasattr(model, 'down_blocks'):
                    for block_group in model.down_blocks:
                        if hasattr(block_group, '__iter__'):
                            for block in block_group:
                                clear_block_caches(block)
                
                # Clear up blocks  
                if hasattr(model, 'up_blocks'):
                    for block_group in model.up_blocks:
                        if hasattr(block_group, '__iter__'):
                            for block in block_group:
                                clear_block_caches(block)
                
                # Clear middle blocks
                if hasattr(model, 'mid_block1') and model.mid_block1 is not None:
                    clear_block_caches(model.mid_block1)
                if hasattr(model, 'mid_block2') and model.mid_block2 is not None:
                    clear_block_caches(model.mid_block2)
            
            # Create device-specific forward model
            psf = PSF()
            forward_model = ForwardModel(
                psf=psf.to_torch(device=device),
                background=10.0,
                device=device
            )
            
            sampler = DDIMSampler(
                model=trainer,
                forward_model=forward_model,
                guidance_strategy=components["guidance"],
                schedule=components["schedule"],
                transform=components["transform"],
                num_timesteps=50,
                ddim_steps=10,
                eta=0.0,
            )
            
            # Move measurement to device
            y_device = y.to(device)
            
            # Profile performance
            with MemoryProfiler(device=device) as profiler:
                start_time = time.time()
                
                result = sampler.sample(
                    y=y_device,
                    shape=y_device.shape,
                    device=device,
                    verbose=False
                )
                
                end_time = time.time()
            
            execution_time = end_time - start_time
            memory_metrics = profiler.get_metrics()
            
            device_results[device] = {
                "execution_time": execution_time,
                "peak_memory": memory_metrics["peak_mb"]
            }
            
            # Verify result
            assert result.shape == y.shape
            assert torch.isfinite(result).all()
        
        # Report device comparison
        print("\nDevice Performance Comparison:")
        for device, metrics in device_results.items():
            print(f"{device.upper()}: Time={metrics['execution_time']:.2f}s, "
                  f"Memory={metrics['peak_memory']:.1f}MB")
        
        # If both devices available, GPU should generally be faster for larger models
        if len(device_results) > 1:
            cpu_time = device_results["cpu"]["execution_time"]
            gpu_time = device_results["cuda"]["execution_time"]
            
            print(f"GPU Speedup: {cpu_time / gpu_time:.2f}x")


class TestPerformanceBenchmarks:
    """Standardized performance benchmarks."""
    
    def test_standard_inference_benchmark(self):
        """Standard inference benchmark for comparison."""
        # Standard benchmark configuration
        benchmark_config = {
            "image_size": 256,
            "batch_size": 1,
            "ddim_steps": 50,
            "num_timesteps": 1000,
            "guidance": "pkl"
        }
        
        # Create components
        unet_config = {
            "sample_size": benchmark_config["image_size"],
            "in_channels": 1,
            "out_channels": 1,
            "layers_per_block": 2,
            "block_out_channels": [128, 256, 512],
            "down_block_types": ["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
            "up_block_types": ["AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
            "attention_head_dim": 8
        }
        
        training_config = {
            "num_timesteps": benchmark_config["num_timesteps"],
            "learning_rate": 1e-4,
            "use_ema": True,
            "beta_schedule": "cosine"
        }
        
        # Skip this test if it would be too slow/memory intensive
        if benchmark_config["image_size"] > 128:
            pytest.skip("Skipping large benchmark test to avoid excessive resource usage")
        
        # Create minimal version for testing
        unet_config["sample_size"] = 64
        unet_config["block_out_channels"] = [32, 64, 128]
        benchmark_config["image_size"] = 64
        
        unet = UNet(unet_config)
        trainer = DDPMTrainer(model=unet, config=training_config)
        
        psf = PSF()
        forward_model = ForwardModel(
            psf=psf.to_torch(device="cpu"),
            background=10.0,
            device="cpu"
        )
        
        guidance = PKLGuidance(epsilon=1e-6)
        schedule = AdaptiveSchedule(
            lambda_base=0.1,
            T_threshold=800,
            epsilon_lambda=1e-3,
            T_total=benchmark_config["num_timesteps"]
        )
        
        transform = IntensityToModel(min_intensity=0.0, max_intensity=1000.0)
        
        sampler = DDIMSampler(
            model=trainer,
            forward_model=forward_model,
            guidance_strategy=guidance,
            schedule=schedule,
            transform=transform,
            num_timesteps=benchmark_config["num_timesteps"],
            ddim_steps=benchmark_config["ddim_steps"],
            eta=0.0,
        )
        
        # Create benchmark measurement
        size = benchmark_config["image_size"]
        y = torch.randn(benchmark_config["batch_size"], 1, size, size) * 10 + 50
        y = torch.clamp(y, 0, None)
        
        # Run benchmark
        with MemoryProfiler(device="cpu") as profiler:
            start_time = time.time()
            
            result = sampler.sample(
                y=y,
                shape=y.shape,
                device="cpu",
                verbose=False
            )
            
            end_time = time.time()
        
        execution_time = end_time - start_time
        memory_metrics = profiler.get_metrics()
        
        # Report benchmark results
        print(f"\nStandard Inference Benchmark Results:")
        print(f"Configuration: {benchmark_config}")
        print(f"Execution Time: {execution_time:.2f}s")
        print(f"Peak Memory: {memory_metrics['peak_mb']:.1f}MB")
        print(f"Average Memory: {memory_metrics['avg_mb']:.1f}MB")
        print(f"Memory Delta: {memory_metrics['delta_mb']:.1f}MB")
        
        # Verify result
        assert result.shape == y.shape
        assert torch.isfinite(result).all()
        assert torch.all(result >= 0)
        
        # Basic performance assertions
        assert execution_time < 300, f"Benchmark too slow: {execution_time:.2f}s"
        assert memory_metrics["peak_mb"] < 2000, f"Memory usage too high: {memory_metrics['peak_mb']:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
