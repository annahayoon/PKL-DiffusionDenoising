#!/usr/bin/env python3
"""
Example usage of advanced schedulers for microscopy diffusion models.

This script demonstrates how to use the new schedulers implemented in schedulers.py
for improved sampling quality and speed in microscopy applications.
"""

import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pkl_dg.models.schedulers import (
    create_scheduler,
    DPMSolverScheduler,
    EulerScheduler,
    MicroscopyOptimizedScheduler,
    ConsistencyTrainingScheduler
)


def demo_fast_sampling_schedulers():
    """Demonstrate fast sampling schedulers for real-time microscopy."""
    print("🚀 Fast Sampling Schedulers for Real-Time Microscopy")
    print("=" * 60)
    
    # DPM-Solver++: 10-20 steps with high quality
    dpm_scheduler = create_scheduler("dpm_solver", num_timesteps=1000, solver_order=2)
    dpm_timesteps = dpm_scheduler.get_dpm_timesteps(num_inference_steps=20)
    print(f"✅ DPM-Solver++: {len(dpm_timesteps)} steps for high-quality sampling")
    print(f"   Timesteps: {dpm_timesteps[:5].tolist()}... (first 5)")
    
    # Euler: Stable and fast
    euler_scheduler = create_scheduler("euler", use_karras_sigmas=True)
    karras_sigmas = euler_scheduler.get_karras_sigmas(num_inference_steps=25)
    print(f"✅ Euler with Karras sigmas: 25 steps with improved quality")
    
    # PNDM: High quality in 50 steps
    pndm_scheduler = create_scheduler("pndm", skip_prk_steps=True)
    print(f"✅ PNDM: High-quality sampling in ~50 steps")
    print()


def demo_microscopy_optimized_scheduler():
    """Demonstrate the microscopy-specific scheduler."""
    print("🔬 Microscopy-Optimized Scheduler")
    print("=" * 40)
    
    # Standard cosine vs microscopy-optimized
    standard_scheduler = create_scheduler("cosine", num_timesteps=1000)
    standard_betas = standard_scheduler.get_betas()
    
    micro_scheduler = create_scheduler(
        "microscopy_optimized",
        num_timesteps=1000,
        preserve_fine_details=True,
        edge_enhancement=0.1,
        low_intensity_protection=0.05
    )
    micro_betas = micro_scheduler.get_betas()
    
    print(f"Standard cosine - Early beta range: {standard_betas[:10].min():.6f} to {standard_betas[:10].max():.6f}")
    print(f"Microscopy opt - Early beta range: {micro_betas[:10].min():.6f} to {micro_betas[:10].max():.6f}")
    print(f"                 (Lower = better fine detail preservation)")
    print()
    
    print(f"Standard cosine - Late beta range: {standard_betas[-10:].min():.6f} to {standard_betas[-10:].max():.6f}")
    print(f"Microscopy opt - Late beta range: {micro_betas[-10:].min():.6f} to {micro_betas[-10:].max():.6f}")
    print(f"                 (Lower = better low-intensity protection)")
    print()


def demo_consistency_model_scheduler():
    """Demonstrate consistency model training scheduler."""
    print("⚡ Consistency Model Training")
    print("=" * 35)
    
    consistency_scheduler = create_scheduler(
        "consistency_training",
        num_timesteps=1000,
        distillation_steps=18,
        consistency_weight=1.0
    )
    
    # Get timesteps for consistency distillation
    ct_timesteps = consistency_scheduler.get_consistency_timesteps()
    print(f"✅ Consistency training with {len(ct_timesteps)} distillation steps")
    print(f"   Timesteps: {ct_timesteps.tolist()}")
    print(f"   Benefits: Single-step generation (1000x faster inference!)")
    print()


def integration_example():
    """Show how to integrate with your existing DDIMSampler."""
    print("🔗 Integration with Existing Code")
    print("=" * 35)
    
    print("To use in your run_microscopy.py:")
    print()
    print("# For fast inference (10-20 steps):")
    print("sampler = DDIMSampler(")
    print("    model=ddpm_trainer.model,")
    print("    guidance=guidance,")
    print("    transform=transform,")
    print("    ddim_steps=20,  # Much faster than 50-100")
    print("    scheduler='dpm_solver'  # Use DPM-Solver++")
    print(")")
    print()
    
    print("# For microscopy-optimized quality:")
    print("sampler = DDIMSampler(")
    print("    model=ddpm_trainer.model,")
    print("    guidance=guidance,")
    print("    transform=transform,")
    print("    ddim_steps=50,")
    print("    scheduler='microscopy_optimized'  # Preserve fine details")
    print(")")
    print()
    
    print("# For consistency model training:")
    print("trainer = DDPMTrainer(")
    print("    model=unet,")
    print("    config={...},")
    print("    scheduler='consistency_training'  # Enable single-step generation")
    print(")")
    print()


if __name__ == "__main__":
    print("🧬 Advanced Schedulers for Microscopy Diffusion Models")
    print("=" * 60)
    print()
    
    demo_fast_sampling_schedulers()
    demo_microscopy_optimized_scheduler()
    demo_consistency_model_scheduler()
    integration_example()
    
    print("🎯 Recommendations for Your Microscopy Application:")
    print("=" * 50)
    print("1. Use 'dmp_solver' for real-time inference (10-20 steps)")
    print("2. Use 'microscopy_optimized' for best image quality")
    print("3. Use 'consistency_training' for single-step generation")
    print("4. Use 'euler' with Karras sigmas for stable sampling")
    print()
    print("Your physics-guided diffusion + these schedulers = 🏆 State-of-the-art!")
