"""
Memory Profiling Utilities for Deep Learning Training

This module provides comprehensive memory profiling and monitoring utilities
for tracking GPU memory usage, detecting memory leaks, and optimizing
memory consumption during training and inference.
"""

import torch
import time
import gc
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import warnings


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
        """Initialize memory profiler.
        
        Args:
            interval: Sampling interval in seconds
            track_cpu: Whether to track CPU memory
            track_gpu: Whether to track GPU memory
            auto_cleanup: Whether to automatically run garbage collection
            verbose: Whether to print verbose output
        """
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
    
    def start_monitoring(self):
        """Start continuous memory monitoring in background thread."""
        if self.monitoring:
            warnings.warn("Memory monitoring already started")
            return
        
        self.monitoring = True
        self.start_time = time.time()
        
        # Take initial snapshot
        self.baseline_snapshot = self.take_snapshot(phase="start", notes="Baseline")
        
        def monitor_loop():
            while self.monitoring:
                time.sleep(self.interval)
                if self.monitoring:  # Check again in case it was stopped
                    self.take_snapshot(phase="monitoring")
                    
                    if self.auto_cleanup:
                        # Periodic cleanup
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        if self.verbose:
            print("🔍 Started memory monitoring")
    
    def stop_monitoring(self) -> MemoryProfile:
        """Stop memory monitoring and return profile."""
        if not self.monitoring:
            warnings.warn("Memory monitoring not started")
            return self.profile
        
        self.monitoring = False
        
        # Wait for monitor thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # Take final snapshot
        final_snapshot = self.take_snapshot(phase="end", notes="Final")
        
        # Calculate summary statistics
        if self.start_time:
            self.profile.total_duration = time.time() - self.start_time
        
        if self.profile.snapshots:
            gpu_allocated = [s.gpu_allocated for s in self.profile.snapshots]
            gpu_reserved = [s.gpu_reserved for s in self.profile.snapshots]
            
            self.profile.average_gpu_usage = sum(gpu_allocated) / len(gpu_allocated)
            
            # Calculate memory efficiency (allocated / reserved)
            avg_reserved = sum(gpu_reserved) / len(gpu_reserved)
            if avg_reserved > 0:
                self.profile.memory_efficiency = self.profile.average_gpu_usage / avg_reserved
        
        if self.verbose:
            print("🏁 Stopped memory monitoring")
            self.print_summary()
        
        return self.profile
    
    def detect_memory_leaks(self) -> Dict[str, Any]:
        """Detect potential memory leaks by comparing with baseline."""
        if not self.baseline_snapshot or not self.profile.snapshots:
            return {"error": "Insufficient data for leak detection"}
        
        latest_snapshot = self.profile.snapshots[-1]
        
        gpu_leak = latest_snapshot.gpu_allocated - self.baseline_snapshot.gpu_allocated
        cpu_leak = latest_snapshot.cpu_memory - self.baseline_snapshot.cpu_memory
        
        leaks_detected = []
        
        if gpu_leak > self.leak_threshold_gb:
            leaks_detected.append({
                "type": "GPU",
                "leak_gb": gpu_leak,
                "baseline_gb": self.baseline_snapshot.gpu_allocated,
                "current_gb": latest_snapshot.gpu_allocated,
            })
        
        if cpu_leak > self.leak_threshold_gb:
            leaks_detected.append({
                "type": "CPU", 
                "leak_gb": cpu_leak,
                "baseline_gb": self.baseline_snapshot.cpu_memory,
                "current_gb": latest_snapshot.cpu_memory,
            })
        
        return {
            "leaks_detected": len(leaks_detected) > 0,
            "leaks": leaks_detected,
            "threshold_gb": self.leak_threshold_gb,
            "duration_seconds": self.profile.total_duration,
        }
    
    def print_summary(self):
        """Print a formatted summary of memory usage."""
        summary = self.profile.get_summary()
        
        if "error" in summary:
            print(f"❌ {summary['error']}")
            return
        
        print("\n" + "="*60)
        print("📊 MEMORY PROFILING SUMMARY")
        print("="*60)
        
        print(f"⏱️  Duration: {summary['duration_seconds']:.1f}s")
        print(f"📸 Snapshots: {summary['num_snapshots']}")
        
        gpu_info = summary['gpu_memory']
        print(f"\n🎮 GPU Memory:")
        print(f"   Peak Allocated: {gpu_info['peak_allocated_gb']:.2f} GB")
        print(f"   Peak Reserved:  {gpu_info['peak_reserved_gb']:.2f} GB")
        print(f"   Avg Allocated:  {gpu_info['average_allocated_gb']:.2f} GB")
        print(f"   Efficiency:     {gpu_info['efficiency_percent']:.1f}%")
        
        cpu_info = summary['cpu_memory']
        print(f"\n💻 CPU Memory:")
        print(f"   Peak:     {cpu_info['peak_gb']:.2f} GB")
        print(f"   Average:  {cpu_info['average_gb']:.2f} GB")
        
        growth = summary['memory_growth']
        print(f"\n📈 Memory Growth:")
        print(f"   GPU Allocated: {growth['gpu_allocated_growth_gb']:+.3f} GB")
        print(f"   GPU Reserved:  {growth['gpu_reserved_growth_gb']:+.3f} GB")
        print(f"   CPU:           {growth['cpu_growth_gb']:+.3f} GB")
        
        # Memory leak detection
        leak_info = self.detect_memory_leaks()
        if leak_info.get("leaks_detected", False):
            print(f"\n⚠️  MEMORY LEAKS DETECTED:")
            for leak in leak_info["leaks"]:
                print(f"   {leak['type']}: {leak['leak_gb']:.3f} GB leak")
        else:
            print(f"\n✅ No significant memory leaks detected")
        
        print("="*60)
    
    def reset(self):
        """Reset the profiler state."""
        self.profile = MemoryProfile()
        self.baseline_snapshot = None
        self.start_time = None
        
        if self.monitoring:
            self.stop_monitoring()
    
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
    """Convenience context manager for memory profiling.
    
    Args:
        name: Name of the operation being profiled
        interval: Sampling interval in seconds
        verbose: Whether to print verbose output
        auto_cleanup: Whether to automatically run garbage collection
        
    Yields:
        MemoryProfiler instance
        
    Example:
        with profile_memory("training_step") as profiler:
            # Your training code here
            loss = model(batch)
            loss.backward()
    """
    profiler = MemoryProfiler(
        interval=interval,
        verbose=verbose,
        auto_cleanup=auto_cleanup,
    )
    
    with profiler.profile_context(name):
        yield profiler


def profile_memory_usage(
    func: Callable,
    *args,
    name: Optional[str] = None,
    verbose: bool = True,
    **kwargs
) -> tuple:
    """Decorator/function wrapper for memory profiling.
    
    Args:
        func: Function to profile
        *args: Arguments to pass to function
        name: Name for profiling (defaults to function name)
        verbose: Whether to print results
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Tuple of (function_result, memory_profile)
        
    Example:
        result, profile = profile_memory_usage(train_step, batch, model)
        
        # Or as decorator:
        @profile_memory_usage
        def train_step(batch, model):
            return model(batch)
    """
    if name is None:
        name = getattr(func, '__name__', 'function')
    
    profiler = MemoryProfiler(verbose=verbose)
    
    with profiler.profile_context(name):
        result = func(*args, **kwargs)
    
    return result, profiler.profile


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


def monitor_training_memory(
    trainer,
    dataloader,
    num_steps: int = 10,
    interval: float = 0.1,
    verbose: bool = True,
) -> MemoryProfile:
    """Monitor memory usage during training steps.
    
    Args:
        trainer: Training model/trainer instance
        dataloader: DataLoader for training data
        num_steps: Number of training steps to monitor
        interval: Memory sampling interval
        verbose: Whether to print progress
        
    Returns:
        MemoryProfile with detailed memory usage
    """
    profiler = MemoryProfiler(interval=interval, verbose=verbose)
    profiler.start_monitoring()
    
    try:
        trainer.train()
        
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break
            
            profiler.take_snapshot(step=step, phase="training_step")
            
            # Perform training step
            loss = trainer.training_step(batch, step)
            
            if verbose and step % max(1, num_steps // 5) == 0:
                print(f"Step {step}/{num_steps}: loss = {loss:.4f}")
    
    finally:
        profile = profiler.stop_monitoring()
    
    return profile