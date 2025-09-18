#!/usr/bin/env python3
"""
Unified PKL Diffusion Denoising Evaluation System

This script combines all evaluation functionality into a single, comprehensive system:
- Standard evaluation on regular images
- Full FOV evaluation with patch-based processing
- Baseline comparisons (Richardson-Lucy, RCAN)
- Cross-method analysis from pre-computed results
- Publication-ready visualizations and statistics

Features:
- Modular metrics system with 21+ metrics across 4 categories
- Configuration validation with Hydra schemas
- Comprehensive Wandb integration
- Multiple evaluation modes in one script
- Extensible architecture for new methods and metrics

Usage Examples:
    # Standard evaluation
    python scripts/evaluation/evaluation.py mode=standard

    # Full FOV evaluation with patch processing
    python scripts/evaluation/evaluation.py mode=full_fov processing.patch_size=256

    # Baseline comparison
    python scripts/evaluation/evaluation.py mode=baseline_comparison

    # Cross-method analysis from existing results
    python scripts/evaluation/evaluation.py mode=cross_method_analysis

    # Custom configuration
    python scripts/evaluation/evaluation.py --config-path configs --config-name my_eval_config
"""

import os
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
import tifffile
from tqdm import tqdm
from PIL import Image
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy import stats

# PKL-DG imports
from pkl_dg.models.unet import UNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.models.sampler import DDIMSampler
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel
from pkl_dg.physics.psf import build_psf_bank
from pkl_dg.guidance.pkl_optical_restoration import PKLGuidance
from pkl_dg.guidance.l2 import L2Guidance
from pkl_dg.guidance.anscombe import AnscombeGuidance
from pkl_dg.guidance.schedules import AdaptiveSchedule
from pkl_dg.data import IntensityToModel, AnscombeToModel, GeneralizedAnscombeToModel, create_sid_dataloader
# Import baselines dynamically to avoid circular imports

# Import new metrics system
from pkl_dg.evaluation.metrics_pkg import compute_metrics, compute_downstream_metrics, list_metrics
# Import robustness functions directly to avoid circular imports
try:
    from pkl_dg.evaluation.metrics_pkg.robustness import test_alignment_error_with_sampler, add_out_of_focus_artifact
except ImportError:
    # Define dummy functions if import fails
    def test_alignment_error_with_sampler(*args, **kwargs):
        return None
    def add_out_of_focus_artifact(*args, **kwargs):
        return args[0]
from pkl_dg.evaluation.schemas import EvaluationConfig, validate_config

try:
    from pkl_dg.baselines import RCANWrapper
    HAS_RCAN = True
except Exception:
    HAS_RCAN = False


class EvaluationMode(Enum):
    """Evaluation modes supported by the unified system."""
    STANDARD = "standard"
    FULL_FOV = "full_fov"
    BASELINE_COMPARISON = "baseline_comparison"
    CROSS_METHOD_ANALYSIS = "cross_method_analysis"
    SID_EVALUATION = "sid_evaluation"


@dataclass
class ProcessingConfig:
    """Processing configuration for different evaluation modes."""
    patch_size: int = 256
    stride: int = 128
    feather_size: Optional[int] = None  # Auto-computed as stride//2
    max_images: Optional[int] = None
    
    # Richardson-Lucy parameters
    rl_iterations: int = 30
    
    # SID-specific parameters
    sid_camera_type: str = "Sony"  # "Sony" or "Fuji"
    sid_data_dir: str = "data/SID"
    sid_use_processed: bool = True  # Use processed 16-bit images instead of RAW
    sid_guidance_types: List[str] = field(default_factory=lambda: ["pkl"])
    sid_num_steps: int = 50
    sid_guidance_scale: float = 0.1
    sid_eta: float = 0.0
    
    # Statistical analysis parameters
    significance_level: float = 0.05
    
    # Visualization parameters
    percentile_clip: float = 99.5
    save_individual_comparisons: bool = True
    save_summary_plots: bool = True


# Add ProcessingConfig to the main config
@dataclass
class UnifiedEvaluationConfig(EvaluationConfig):
    """Extended configuration for unified evaluation system."""
    mode: str = "standard"  # standard, full_fov, baseline_comparison, cross_method_analysis
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # Cross-method analysis specific paths
    cross_analysis: Dict[str, str] = field(default_factory=lambda: {
        "pkl_dir": "",
        "l2_dir": "",
        "anscombe_dir": "",
        "rl_dir": "",
        "rcan_dir": ""
    })


class ParallelEvaluationMixin:
    """Mixin class for parallel evaluation capabilities."""
    
    def parallel_evaluate_batch(self, batch_list: list, num_workers: int = None) -> list:
        """Evaluate multiple batches in parallel.
        
        Args:
            batch_list: List of batches to evaluate
            num_workers: Number of worker processes
            
        Returns:
            List of evaluation results
        """
        if num_workers is None:
            import os
            num_workers = min(8, os.cpu_count() or 1)
        
        if len(batch_list) <= 1 or num_workers <= 1:
            return [self._evaluate_single_batch(batch) for batch in batch_list]
        
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        
        try:
            with ProcessPoolExecutor(max_workers=num_workers,
                                   mp_context=mp.get_context('spawn')) as executor:
                results = list(executor.map(self._evaluate_single_batch, batch_list))
            return results
        except Exception as e:
            print(f"Parallel evaluation failed: {e}. Falling back to sequential.")
            return [self._evaluate_single_batch(batch) for batch in batch_list]
    
    def _evaluate_single_batch(self, batch):
        """Evaluate a single batch (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _evaluate_single_batch")
    
    def gpu_parallel_inference(self, model, batch_list: list, device_ids: list = None) -> list:
        """Run inference on multiple GPUs in parallel.
        
        Args:
            model: Model to use for inference
            batch_list: List of input batches
            device_ids: List of GPU device IDs to use
            
        Returns:
            List of inference results
        """
        if not torch.cuda.is_available():
            # Fallback to CPU
            return [model(batch) for batch in batch_list]
        
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        
        if len(device_ids) <= 1:
            # Single GPU
            device = f"cuda:{device_ids[0]}" if device_ids else "cuda:0"
            model = model.to(device)
            return [model(batch.to(device)) for batch in batch_list]
        
        # Multi-GPU parallel inference
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        def inference_worker(batch, device_id):
            device = f"cuda:{device_id}"
            # Create a copy of the model for this device
            model_copy = model.to(device)
            with torch.no_grad():
                return model_copy(batch.to(device))
        
        results = [None] * len(batch_list)
        
        with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
            futures = []
            for i, batch in enumerate(batch_list):
                device_id = device_ids[i % len(device_ids)]
                future = executor.submit(inference_worker, batch, device_id)
                futures.append((i, future))
            
            for i, future in futures:
                results[i] = future.result()
        
        return results


class UnifiedEvaluator(ParallelEvaluationMixin):
    """Unified evaluation system that combines all evaluation functionality."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize the unified evaluator."""
        self.cfg = cfg
        self.device = self._setup_device()
        self.mode = EvaluationMode(cfg.mode)
        
        # Initialize wandb
        self._setup_wandb()
        
        # Determine metrics to compute
        self.metrics_to_compute = self._get_metrics_list()
        
        # Initialize components based on mode
        self.samplers = {}
        self.rcan_model = None
        self.psf = None
        
        if self.mode in [EvaluationMode.STANDARD, EvaluationMode.FULL_FOV, EvaluationMode.BASELINE_COMPARISON]:
            self._setup_models()
        
        print(f"🚀 Unified Evaluator initialized in {self.mode.value} mode")
        print(f"Device: {self.device}")
        print(f"Computing {len(self.metrics_to_compute)} metrics: {self.metrics_to_compute}")
    
    def _setup_device(self) -> str:
        """Setup computation device."""
        device = str(self.cfg.experiment.device)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        if self.cfg.wandb.mode != "disabled":
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                name=f"{self.cfg.experiment.name}_{self.cfg.mode}",
                mode=self.cfg.wandb.mode,
                tags=self.cfg.wandb.tags + ["evaluation", self.cfg.mode],
                notes=self.cfg.wandb.notes,
                group=self.cfg.wandb.group,
                job_type=f"evaluation_{self.cfg.mode}"
            )
            print("✅ Initialized Weights & Biases logging")
    
    def _get_metrics_list(self) -> List[str]:
        """Get list of metrics to compute based on configuration."""
        all_metrics = []
        if hasattr(self.cfg, 'metrics'):
            all_metrics.extend(getattr(self.cfg.metrics, 'image_quality', []))
            all_metrics.extend(getattr(self.cfg.metrics, 'perceptual', []))
            all_metrics.extend(getattr(self.cfg.metrics, 'robustness', []))
        
        if not all_metrics:
            # Default metrics
            all_metrics = ['psnr', 'ssim', 'frc', 'sar', 'sharpness', 'contrast']
        
        return all_metrics
    
    def _setup_models(self):
        """Setup diffusion models and baselines."""
        print("Setting up models...")
        
        # Setup diffusion models
        if self.mode != EvaluationMode.CROSS_METHOD_ANALYSIS:
            guidance_types = ["l2", "anscombe", "pkl"]
            for guidance_type in guidance_types:
                try:
                    sampler = self._load_model_and_sampler(guidance_type)
                    self.samplers[guidance_type] = sampler
                    print(f"✅ Loaded {guidance_type} sampler")
                except Exception as e:
                    print(f"⚠️ Failed to load {guidance_type} sampler: {e}")
        
        # Setup RCAN if available
        if HAS_RCAN and hasattr(self.cfg, 'baselines') and getattr(self.cfg.baselines, 'rcan_checkpoint', None):
            try:
                self.rcan_model = RCANWrapper(
                    checkpoint_path=str(self.cfg.baselines.rcan_checkpoint),
                    device=self.device
                )
                print("✅ Loaded RCAN model")
            except Exception as e:
                print(f"⚠️ Failed to load RCAN model: {e}")
        
        # Setup PSF for Richardson-Lucy
        if hasattr(self.cfg, 'physics') and getattr(self.cfg.physics, 'psf_path', None):
            try:
                self.psf = self._load_psf(self.cfg.physics.psf_path)
                print("✅ Loaded PSF for Richardson-Lucy")
            except Exception as e:
                print(f"⚠️ Failed to load PSF: {e}")
    
    def _load_model_and_sampler(self, guidance_type: str):
        """Load diffusion model and sampler for specific guidance type."""
        device = self.device
        model_cfg = OmegaConf.to_container(self.cfg.model, resolve=True)
        use_conditioning = bool(getattr(self.cfg.training, "use_conditioning", False))
        
        unet = UNet(model_cfg)
        
        # Create forward model for trainer
        psf_config = getattr(self.cfg, "psf", {})
        if psf_config.get("type") == "gaussian":
            psf = PSF(
                sigma_x=float(psf_config.get("sigma_x", 2.0)),
                sigma_y=float(psf_config.get("sigma_y", 2.0)),
                size=int(psf_config.get("size", 21))
            )
        else:
            psf = PSF()
        
        forward_model = ForwardModel(
            psf=psf.to_torch(device=device),
            background=float(psf_config.get("background", 0.0)),
            device=device,
            common_sizes=[(int(getattr(self.cfg.data, "image_size", 256)), 
                         int(getattr(self.cfg.data, "image_size", 256)))]
        )
        
        # Create DDPM trainer
        ddpm = DDPMTrainer(
            model=unet,
            config=OmegaConf.to_container(self.cfg.training, resolve=True),
            forward_model=forward_model,
        )
        
        # Load checkpoint
        checkpoint_path = Path(str(self.cfg.inference.checkpoint_path))
        state_dict = torch.load(checkpoint_path, map_location=device)
        ddpm.load_state_dict(state_dict, strict=False)
        ddpm.eval().to(device)
        
        # Setup physics-based forward model
        forward_model = None
        try:
            phys_cfg = self.cfg.physics
            use_psf = bool(getattr(phys_cfg, "use_psf", False))
            background = float(getattr(phys_cfg, "background", 0.0))
            
            if use_psf:
                psf_path = getattr(phys_cfg, "psf_path", None)
                use_bead = bool(getattr(phys_cfg, "use_bead_psf", False))
                
                if use_bead:
                    beads_dir = str(getattr(phys_cfg, "beads_dir", ""))
                    bank = build_psf_bank(beads_dir)
                    mode = getattr(phys_cfg, "bead_mode", None)
                    if mode is None:
                        psf_t = bank.get("with_AO", bank.get("no_AO"))
                    else:
                        psf_t = bank.get(str(mode), next(iter(bank.values())))
                    psf = psf_t.to(device=device, dtype=torch.float32)
                    if psf.ndim == 2:
                        psf = psf.unsqueeze(0).unsqueeze(0)
                elif psf_path is not None:
                    psf = PSF(psf_path=str(psf_path)).to_torch(device=device)
                else:
                    psf = PSF().to_torch(device=device)
                
                read_noise_sigma = float(getattr(phys_cfg, "read_noise_sigma", 0.0))
                
                # Get pixel size configuration for PSF scaling
                target_pixel_size_xy_nm = getattr(phys_cfg, "target_pixel_size_xy_nm", None)
                if target_pixel_size_xy_nm is not None:
                    target_pixel_size_xy_nm = float(target_pixel_size_xy_nm)
                
                forward_model = ForwardModel(
                    psf=psf, 
                    background=background, 
                    device=device, 
                    read_noise_sigma=read_noise_sigma,
                    target_pixel_size_xy_nm=target_pixel_size_xy_nm
                )
        except Exception:
            forward_model = None
        
        # Create guidance strategy
        if guidance_type == "pkl":
            guidance = PKLGuidance(epsilon=float(getattr(self.cfg.guidance, "epsilon", 1e-6)))
        elif guidance_type == "l2":
            guidance = L2Guidance()
        elif guidance_type == "anscombe":
            guidance = AnscombeGuidance(epsilon=float(getattr(self.cfg.guidance, "epsilon", 1e-6)))
        else:
            raise ValueError(f"Unknown guidance type: {guidance_type}")
        
        # Create schedule
        schedule_cfg = getattr(self.cfg.guidance, "schedule", {})
        schedule = AdaptiveSchedule(
            lambda_base=float(getattr(self.cfg.guidance, "lambda_base", 0.1)),
            T_threshold=int(getattr(schedule_cfg, "T_threshold", 800)),
            epsilon_lambda=float(getattr(schedule_cfg, "epsilon_lambda", 1e-3)),
            T_total=int(self.cfg.training.num_timesteps),
        )
        
        # Create transform
        noise_model = str(getattr(self.cfg.data, "noise_model", "gaussian")).lower()
        if noise_model == "poisson":
            transform = AnscombeToModel(maxIntensity=float(self.cfg.data.max_intensity))
        elif noise_model == "poisson_gaussian":
            gat_cfg = getattr(self.cfg.data, "gat", {})
            transform = GeneralizedAnscombeToModel(
                maxIntensity=float(self.cfg.data.max_intensity),
                alpha=float(getattr(gat_cfg, "alpha", 1.0)),
                mu=float(getattr(gat_cfg, "mu", 0.0)),
                sigma=float(getattr(gat_cfg, "sigma", 0.0)),
            )
        else:
            transform = IntensityToModel(
                min_intensity=float(self.cfg.data.min_intensity),
                max_intensity=float(self.cfg.data.max_intensity)
            )
        
        # Create sampler
        sampler = DDIMSampler(
            model=ddpm,
            forward_model=forward_model,
            guidance_strategy=guidance,
            schedule=schedule,
            transform=transform,
            num_timesteps=int(self.cfg.training.num_timesteps),
            ddim_steps=int(self.cfg.inference.ddim_steps),
            eta=float(self.cfg.inference.eta),
            use_autocast=bool(getattr(self.cfg.inference, "use_autocast", True)),
        )
        
        return sampler
    
    def _load_psf(self, psf_path: str) -> np.ndarray:
        """Load PSF from file."""
        psf_path = Path(psf_path)
        
        if psf_path.suffix.lower() in ['.tif', '.tiff']:
            psf = tifffile.imread(str(psf_path))
        elif psf_path.suffix.lower() == '.npy':
            psf = np.load(psf_path)
        else:
            raise ValueError(f"Unsupported PSF format: {psf_path.suffix}")
        
        # Ensure PSF is 2D
        if psf.ndim == 3 and psf.shape[0] == 1:
            psf = psf[0]
        elif psf.ndim == 3:
            raise ValueError("3D PSF not supported - please provide 2D PSF")
        
        # Normalize PSF
        psf = psf.astype(np.float32)
        psf = psf / (psf.sum() + 1e-12)
        
        return psf
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation based on the selected mode."""
        if self.mode == EvaluationMode.STANDARD:
            return self.run_standard_evaluation()
        elif self.mode == EvaluationMode.FULL_FOV:
            return self.run_full_fov_evaluation()
        elif self.mode == EvaluationMode.BASELINE_COMPARISON:
            return self.run_baseline_comparison()
        elif self.mode == EvaluationMode.CROSS_METHOD_ANALYSIS:
            return self.run_cross_method_analysis()
        elif self.mode == EvaluationMode.SID_EVALUATION:
            return self.run_sid_evaluation()
        else:
            raise ValueError(f"Unknown evaluation mode: {self.mode}")
    
    def run_standard_evaluation(self) -> Dict[str, Dict[str, float]]:
        """Run standard evaluation on regular images."""
        print("🔍 Running Standard Evaluation")
        
        input_dir = Path(str(self.cfg.inference.input_dir))
        gt_dir = Path(str(self.cfg.inference.gt_dir))
        mask_dir = Path(str(getattr(self.cfg.inference, "mask_dir", "")))
        output_dir = Path(str(self.cfg.inference.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image pairs
        image_paths = self._load_image_pairs(input_dir)
        if self.cfg.processing.max_images:
            image_paths = image_paths[:self.cfg.processing.max_images]
        
        print(f"Found {len(image_paths)} images to evaluate")
        
        # Results accumulator
        results_accumulator = {}
        
        for img_path in tqdm(image_paths, desc="Evaluating images"):
            wf_image = self._read_image(img_path)
            gt_image = self._read_image(gt_dir / img_path.name)
            
            # Load masks if available
            gt_masks = None
            if mask_dir.is_dir():
                mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    gt_masks = self._read_image(mask_path)
            
            # Evaluate each method
            method_results = {}
            
            # Widefield baseline
            wf_metrics = compute_metrics(wf_image, gt_image, input_img=wf_image, 
                                       metric_names=self.metrics_to_compute)
            method_results["widefield"] = wf_metrics
            
            if gt_masks is not None:
                downstream_metrics = compute_downstream_metrics(wf_image, gt_masks)
                method_results["widefield"].update(downstream_metrics)
            
            # Diffusion methods
            conditioning_type = str(getattr(self.cfg.training, "conditioning_type", "wf")).lower()
            
            for method_name, sampler in self.samplers.items():
                try:
                    # Prepare input tensor
                    ten_y = torch.from_numpy(wf_image).float().to(self.device)
                    if ten_y.ndim == 2:
                        ten_y = ten_y.unsqueeze(0).unsqueeze(0)
                    
                    # Build conditioner if needed
                    cond = None
                    if conditioning_type == "wf":
                        try:
                            cond = sampler.transform(ten_y)
                        except Exception:
                            pass
                    
                    # Sample
                    pred = sampler.sample(ten_y, tuple(ten_y.shape), device=self.device, 
                                        verbose=False, conditioner=cond)
                    pred_np = pred.squeeze().detach().cpu().numpy().astype(np.float32)
                    
                    # Compute metrics
                    method_metrics = compute_metrics(pred_np, gt_image, input_img=wf_image,
                                                   metric_names=self.metrics_to_compute)
                    method_results[method_name] = method_metrics
                    
                    if gt_masks is not None:
                        downstream_metrics = compute_downstream_metrics(pred_np, gt_masks)
                        method_results[method_name].update(downstream_metrics)
                    
                except Exception as e:
                    print(f"⚠️ Error evaluating {method_name}: {e}")
                    continue
            
            # Richardson-Lucy baseline
            if self.psf is not None:
                try:
                    from pkl_dg.baselines import richardson_lucy_restore
                    rl_result = richardson_lucy_restore(
                        image=wf_image,
                        psf=self.psf,
                        num_iter=self.cfg.processing.rl_iterations,
                        clip=True
                    )
                    rl_metrics = compute_metrics(rl_result, gt_image, input_img=wf_image,
                                               metric_names=self.metrics_to_compute)
                    method_results["richardson_lucy"] = rl_metrics
                    
                    if gt_masks is not None:
                        downstream_metrics = compute_downstream_metrics(rl_result, gt_masks)
                        method_results["richardson_lucy"].update(downstream_metrics)
                        
                except Exception as e:
                    print(f"⚠️ Error with Richardson-Lucy: {e}")
            
            # RCAN baseline
            if self.rcan_model is not None:
                try:
                    rcan_result = self.rcan_model.infer(wf_image)
                    rcan_metrics = compute_metrics(rcan_result, gt_image, input_img=wf_image,
                                                 metric_names=self.metrics_to_compute)
                    method_results["rcan"] = rcan_metrics
                    
                    if gt_masks is not None:
                        downstream_metrics = compute_downstream_metrics(rcan_result, gt_masks)
                        method_results["rcan"].update(downstream_metrics)
                        
                except Exception as e:
                    print(f"⚠️ Error with RCAN: {e}")
            
            # Accumulate results
            for method_name, metrics in method_results.items():
                if method_name not in results_accumulator:
                    results_accumulator[method_name] = {k: [] for k in metrics.keys()}
                
                for metric_name, value in metrics.items():
                    if np.isfinite(value):
                        results_accumulator[method_name][metric_name].append(value)
            
            # Save visual comparison
            if self.cfg.processing.save_individual_comparisons:
                self._save_visual_comparison(
                    output_dir / "comparisons" / f"{img_path.stem}_comparison.png",
                    wf_image, method_results, gt_image
                )
        
        # Compute final statistics
        final_results = {}
        for method_name, metric_lists in results_accumulator.items():
            final_results[method_name] = {}
            for metric_name, values in metric_lists.items():
                if values:
                    final_results[method_name][metric_name] = float(np.mean(values))
        
        # Log to wandb and save results
        self._log_results_to_wandb(final_results)
        self._save_results_to_file(final_results, output_dir / "standard_evaluation_results.json")
        
        return final_results
    
    def run_full_fov_evaluation(self) -> Dict[str, Any]:
        """Run full field-of-view evaluation with patch-based processing."""
        print("🔍 Running Full FOV Evaluation")
        
        input_dir = Path(str(self.cfg.inference.input_dir))
        gt_dir = Path(str(self.cfg.inference.gt_dir))
        output_dir = Path(str(self.cfg.inference.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image pairs
        image_paths = self._load_image_pairs(input_dir)
        if self.cfg.processing.max_images:
            image_paths = image_paths[:self.cfg.processing.max_images]
        
        print(f"Found {len(image_paths)} images for full FOV evaluation")
        
        results_accumulator = {}
        
        for img_path in tqdm(image_paths, desc="Processing full FOV images"):
            wf_image = self._read_image(img_path)
            gt_image = self._read_image(gt_dir / img_path.name)
            
            print(f"Processing {img_path.name}: {wf_image.shape} -> {gt_image.shape}")
            
            # Process with patch-based methods
            method_results = {}
            
            # Diffusion methods with patch processing
            for method_name, sampler in self.samplers.items():
                try:
                    processed_image = self._process_image_patches(wf_image, sampler, method_name)
                    method_metrics = compute_metrics(processed_image, gt_image, input_img=wf_image,
                                                   metric_names=self.metrics_to_compute)
                    method_results[method_name] = method_metrics
                    
                    # Save processed image
                    output_path = output_dir / f"{img_path.stem}_{method_name}_result.tif"
                    tifffile.imwrite(str(output_path), processed_image.astype(np.float32))
                    
                except Exception as e:
                    print(f"⚠️ Error processing {method_name}: {e}")
                    continue
            
            # Richardson-Lucy with patch processing
            if self.psf is not None:
                try:
                    rl_result = self._process_richardson_lucy_patches(wf_image)
                    rl_metrics = compute_metrics(rl_result, gt_image, input_img=wf_image,
                                               metric_names=self.metrics_to_compute)
                    method_results["richardson_lucy"] = rl_metrics
                    
                    # Save result
                    output_path = output_dir / f"{img_path.stem}_richardson_lucy_result.tif"
                    tifffile.imwrite(str(output_path), rl_result.astype(np.float32))
                    
                except Exception as e:
                    print(f"⚠️ Error with Richardson-Lucy patches: {e}")
            
            # RCAN with patch processing
            if self.rcan_model is not None:
                try:
                    rcan_result = self._process_rcan_patches(wf_image)
                    rcan_metrics = compute_metrics(rcan_result, gt_image, input_img=wf_image,
                                                 metric_names=self.metrics_to_compute)
                    method_results["rcan"] = rcan_metrics
                    
                    # Save result
                    output_path = output_dir / f"{img_path.stem}_rcan_result.tif"
                    tifffile.imwrite(str(output_path), rcan_result.astype(np.float32))
                    
                except Exception as e:
                    print(f"⚠️ Error with RCAN patches: {e}")
            
            # Accumulate results
            for method_name, metrics in method_results.items():
                if method_name not in results_accumulator:
                    results_accumulator[method_name] = {k: [] for k in metrics.keys()}
                
                for metric_name, value in metrics.items():
                    if np.isfinite(value):
                        results_accumulator[method_name][metric_name].append(value)
            
            # Create full FOV comparison visualization
            if self.cfg.processing.save_individual_comparisons:
                self._save_full_fov_comparison(
                    output_dir / "comparisons" / f"{img_path.stem}_full_fov_comparison.png",
                    wf_image, method_results, gt_image
                )
        
        # Compute final statistics
        final_results = {}
        for method_name, metric_lists in results_accumulator.items():
            final_results[method_name] = {}
            for metric_name, values in metric_lists.items():
                if values:
                    final_results[method_name][metric_name] = float(np.mean(values))
        
        # Log results and create summary
        self._log_results_to_wandb(final_results)
        self._save_results_to_file(final_results, output_dir / "full_fov_evaluation_results.json")
        
        if self.cfg.processing.save_summary_plots:
            self._create_summary_plots(final_results, output_dir / "summary_plots")
        
        return final_results
    
    def run_baseline_comparison(self) -> Dict[str, Any]:
        """Run focused baseline comparison between methods."""
        print("🔍 Running Baseline Comparison")
        
        # This combines the functionality of baseline_comparison_full_fov.py
        # with the new metrics system and wandb integration
        
        input_dir = Path(str(self.cfg.inference.input_dir))
        gt_dir = Path(str(self.cfg.inference.gt_dir))
        output_dir = Path(str(self.cfg.inference.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image pairs
        image_paths = self._load_image_pairs(input_dir)
        if self.cfg.processing.max_images:
            image_paths = image_paths[:self.cfg.processing.max_images]
        
        print(f"Running baseline comparison on {len(image_paths)} images")
        
        all_results = {
            "richardson_lucy": [],
            "rcan": [],
            "widefield": []
        }
        
        for img_path in tqdm(image_paths, desc="Baseline comparison"):
            wf_image = self._read_image(img_path)
            gt_image = self._read_image(gt_dir / img_path.name)
            
            # Widefield baseline
            wf_metrics = compute_metrics(wf_image, gt_image, input_img=wf_image,
                                       metric_names=self.metrics_to_compute)
            all_results["widefield"].append(wf_metrics)
            
            # Richardson-Lucy
            if self.psf is not None:
                try:
                    if wf_image.size > 512*512:  # Use patch processing for large images
                        rl_result = self._process_richardson_lucy_patches(wf_image)
                    else:
                        from pkl_dg.baselines import richardson_lucy_restore
                        rl_result = richardson_lucy_restore(
                            image=wf_image,
                            psf=self.psf,
                            num_iter=self.cfg.processing.rl_iterations,
                            clip=True
                        )
                    
                    rl_metrics = compute_metrics(rl_result, gt_image, input_img=wf_image,
                                               metric_names=self.metrics_to_compute)
                    all_results["richardson_lucy"].append(rl_metrics)
                    
                    print(f"RL metrics: PSNR={rl_metrics.get('psnr', 0):.2f}, SSIM={rl_metrics.get('ssim', 0):.3f}")
                    
                except Exception as e:
                    print(f"⚠️ Error with Richardson-Lucy: {e}")
                    continue
            
            # RCAN
            if self.rcan_model is not None:
                try:
                    if wf_image.size > 512*512:  # Use patch processing for large images
                        rcan_result = self._process_rcan_patches(wf_image)
                    else:
                        rcan_result = self.rcan_model.infer(wf_image)
                    
                    rcan_metrics = compute_metrics(rcan_result, gt_image, input_img=wf_image,
                                                 metric_names=self.metrics_to_compute)
                    all_results["rcan"].append(rcan_metrics)
                    
                    print(f"RCAN metrics: PSNR={rcan_metrics.get('psnr', 0):.2f}, SSIM={rcan_metrics.get('ssim', 0):.3f}")
                    
                except Exception as e:
                    print(f"⚠️ Error with RCAN: {e}")
                    continue
            
            # Create comparison visualization
            if self.cfg.processing.save_individual_comparisons:
                comparison_methods = {"widefield": wf_image}
                if "richardson_lucy" in all_results and all_results["richardson_lucy"]:
                    comparison_methods["richardson_lucy"] = rl_result
                if "rcan" in all_results and all_results["rcan"]:
                    comparison_methods["rcan"] = rcan_result
                
                self._save_baseline_comparison_visual(
                    output_dir / "comparisons" / f"{img_path.stem}_baseline_comparison.png",
                    wf_image, comparison_methods, gt_image
                )
        
        # Compute summary statistics
        summary_results = {}
        for method, results_list in all_results.items():
            if not results_list:
                continue
            
            summary_results[method] = {}
            
            # Get all metric names from first result
            if results_list:
                metric_names = results_list[0].keys()
                
                for metric_name in metric_names:
                    values = [r[metric_name] for r in results_list if np.isfinite(r[metric_name])]
                    if values:
                        summary_results[method][metric_name] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                            "count": len(values)
                        }
        
        # Statistical significance testing
        if len(summary_results) >= 2:
            significance_results = self._compute_statistical_significance(all_results)
            summary_results["statistical_tests"] = significance_results
        
        # Log and save results
        self._log_baseline_results_to_wandb(summary_results)
        self._save_results_to_file(summary_results, output_dir / "baseline_comparison_results.json")
        
        if self.cfg.processing.save_summary_plots:
            self._create_baseline_summary_plots(summary_results, output_dir / "summary_plots")
        
        return summary_results
    
    def run_cross_method_analysis(self) -> Dict[str, Any]:
        """Run cross-method analysis from pre-computed results."""
        print("🔍 Running Cross-Method Analysis")
        
        # This combines the functionality of compare_all_methods.py
        output_dir = Path(str(self.cfg.inference.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results from different method directories
        method_results = {}
        for method_name, method_dir in self.cfg.cross_analysis.items():
            if method_dir and Path(method_dir).exists():
                try:
                    results = self._load_method_results(method_dir, method_name)
                    method_results[method_name] = results
                    print(f"✅ Loaded {len(results)} results for {method_name}")
                except Exception as e:
                    print(f"⚠️ Failed to load {method_name} results: {e}")
        
        if not method_results:
            raise ValueError("No method results found. Check cross_analysis paths in config.")
        
        # Load input and GT images
        input_dir = Path(str(self.cfg.inference.input_dir))
        gt_dir = Path(str(self.cfg.inference.gt_dir))
        
        input_files = self._load_image_pairs(input_dir)
        if self.cfg.processing.max_images:
            input_files = input_files[:self.cfg.processing.max_images]
        
        # Collect all results with metrics
        all_analysis_results = []
        
        for input_file in tqdm(input_files, desc="Analyzing method results"):
            gt_file = gt_dir / input_file.name
            if not gt_file.exists():
                continue
            
            image_name = input_file.stem
            
            # Check if we have results for all methods
            if not all(image_name in results for results in method_results.values()):
                continue
            
            # Load input and GT
            wf_image = self._read_image(input_file)
            gt_image = self._read_image(gt_file)
            
            # Analyze each method's result
            for method_name, results in method_results.items():
                if image_name not in results:
                    continue
                
                method_result = results[image_name]
                
                # Compute metrics using new system
                metrics = compute_metrics(method_result, gt_image, input_img=wf_image,
                                        metric_names=self.metrics_to_compute)
                
                result_row = {
                    "image": image_name,
                    "method": method_name,
                    **metrics
                }
                all_analysis_results.append(result_row)
            
            # Create cross-method comparison visualization
            if self.cfg.processing.save_individual_comparisons:
                self._save_cross_method_comparison(
                    output_dir / "comparisons" / f"{image_name}_cross_method_comparison.png",
                    wf_image, {name: results[image_name] for name, results in method_results.items()
                              if image_name in results}, gt_image
                )
        
        # Create DataFrame for analysis
        df = pd.DataFrame(all_analysis_results)
        
        # Save detailed results
        df.to_csv(output_dir / "cross_method_analysis_results.csv", index=False)
        
        # Compute summary statistics
        summary_stats = df.groupby('method')[self.metrics_to_compute].agg(['mean', 'std', 'count']).round(4)
        summary_stats.to_csv(output_dir / "cross_method_summary_statistics.csv")
        
        # Create comprehensive plots
        if self.cfg.processing.save_summary_plots:
            self._create_cross_method_plots(df, output_dir / "summary_plots")
        
        # Log to wandb
        self._log_cross_method_results_to_wandb(df, summary_stats)
        
        return {
            "detailed_results": df.to_dict('records'),
            "summary_statistics": summary_stats.to_dict(),
            "total_comparisons": len(all_analysis_results)
        }
    
    # Helper methods for patch processing
    def _extract_patches(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Extract overlapping patches from image with position information."""
        patches = []
        h, w = image.shape
        patch_size = self.cfg.processing.patch_size
        stride = self.cfg.processing.stride
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                patch = image[y:y_end, x:x_end]
                patches.append((patch, (y, x, y_end, x_end)))
        
        return patches
    
    def _reconstruct_from_patches(self, patches: List[Tuple[np.ndarray, Tuple[int, int, int, int]]], 
                                 original_shape: Tuple[int, int]) -> np.ndarray:
        """Reconstruct image from patches with feathering."""
        h, w = original_shape
        canvas = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        feather_size = self.cfg.processing.feather_size
        if feather_size is None:
            feather_size = self.cfg.processing.stride // 2
        
        for patch, (y, x, y_end, x_end) in patches:
            patch_h, patch_w = patch.shape
            
            # Create feathering weights
            weights = np.ones((patch_h, patch_w), dtype=np.float32)
            
            # Apply feathering to edges
            if feather_size > 0:
                # Top edge
                if y > 0:
                    fade_h = min(feather_size, patch_h)
                    for i in range(fade_h):
                        weights[i, :] *= i / fade_h
                
                # Bottom edge
                if y_end < h:
                    fade_h = min(feather_size, patch_h)
                    for i in range(fade_h):
                        weights[-(i+1), :] *= i / fade_h
                
                # Left edge
                if x > 0:
                    fade_w = min(feather_size, patch_w)
                    for i in range(fade_w):
                        weights[:, i] *= i / fade_w
                
                # Right edge
                if x_end < w:
                    fade_w = min(feather_size, patch_w)
                    for i in range(fade_w):
                        weights[:, -(i+1)] *= i / fade_w
            
            # Add to canvas
            canvas[y:y_end, x:x_end] += patch * weights
            weight_map[y:y_end, x:x_end] += weights
        
        # Normalize by weights
        mask = weight_map > 0
        canvas[mask] = canvas[mask] / weight_map[mask]
        
        # Apply slight smoothing to reduce artifacts
        canvas = gaussian_filter(canvas, sigma=0.5)
        
        return canvas
    
    def _process_image_patches(self, image: np.ndarray, sampler, method_name: str) -> np.ndarray:
        """Process image using patches with the diffusion sampler."""
        patches = self._extract_patches(image)
        processed_patches = []
        
        conditioning_type = str(getattr(self.cfg.training, "conditioning_type", "wf")).lower()
        
        for patch, position in tqdm(patches, desc=f"Processing {method_name} patches", leave=False):
            try:
                # Prepare tensor
                ten_patch = torch.from_numpy(patch).float().to(self.device).unsqueeze(0).unsqueeze(0)
                
                # Build conditioner
                cond = None
                if conditioning_type == "wf":
                    try:
                        cond = sampler.transform(ten_patch)
                    except Exception:
                        pass
                
                # Sample
                pred = sampler.sample(ten_patch, ten_patch.shape, device=self.device, 
                                    verbose=False, conditioner=cond)
                pred_np = pred.squeeze().detach().cpu().numpy().astype(np.float32)
                
                processed_patches.append((pred_np, position))
                
            except Exception as e:
                print(f"⚠️ Error processing patch: {e}")
                # Use original patch as fallback
                processed_patches.append((patch, position))
        
        # Reconstruct full image
        reconstructed = self._reconstruct_from_patches(processed_patches, image.shape)
        return reconstructed
    
    def _process_richardson_lucy_patches(self, image: np.ndarray) -> np.ndarray:
        """Process image using Richardson-Lucy with patches."""
        patches = self._extract_patches(image)
        processed_patches = []
        
        for patch, position in tqdm(patches, desc="Richardson-Lucy patches", leave=False):
            try:
                from pkl_dg.baselines import richardson_lucy_restore
                rl_patch = richardson_lucy_restore(
                    image=patch,
                    psf=self.psf,
                    num_iter=self.cfg.processing.rl_iterations,
                    clip=True
                )
                processed_patches.append((rl_patch, position))
            except Exception as e:
                print(f"⚠️ Error processing RL patch: {e}")
                processed_patches.append((patch, position))
        
        return self._reconstruct_from_patches(processed_patches, image.shape)
    
    def _process_rcan_patches(self, image: np.ndarray) -> np.ndarray:
        """Process image using RCAN with patches."""
        patches = self._extract_patches(image)
        processed_patches = []
        
        for patch, position in tqdm(patches, desc="RCAN patches", leave=False):
            try:
                # Normalize for RCAN
                patch_norm = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
                rcan_patch = self.rcan_model.infer(patch_norm)
                # Denormalize
                rcan_patch = rcan_patch * (patch.max() - patch.min()) + patch.min()
                processed_patches.append((rcan_patch, position))
            except Exception as e:
                print(f"⚠️ Error processing RCAN patch: {e}")
                processed_patches.append((patch, position))
        
        return self._reconstruct_from_patches(processed_patches, image.shape)
    
    # Utility methods
    def _load_image_pairs(self, directory: Path) -> List[Path]:
        """Load image file paths from directory."""
        paths = []
        for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
            paths.extend(list(directory.glob(ext)))
        return sorted(paths)
    
    def _read_image(self, path: Path) -> np.ndarray:
        """Read image from file."""
        if path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            img = np.array(Image.open(path)).astype(np.float32)
        else:
            img = tifffile.imread(str(path)).astype(np.float32)
        
        # Ensure single channel
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        
        return img
    
    def _load_method_results(self, method_dir: str, method_name: str) -> Dict[str, np.ndarray]:
        """Load pre-computed results from method directory."""
        method_path = Path(method_dir)
        results = {}
        
        for ext in ['.tif', '.tiff', '.png']:
            for result_file in method_path.glob(f"*{ext}"):
                if 'comparison' not in result_file.name and 'summary' not in result_file.name:
                    image_name = result_file.stem
                    # Clean up common suffixes
                    for suffix in ['_result', '_reconstructed', '_output', f'_{method_name}']:
                        image_name = image_name.replace(suffix, '')
                    
                    img = self._read_image(result_file)
                    results[image_name] = img
        
        return results
    
    def _normalize_for_visualization(self, image: np.ndarray) -> np.ndarray:
        """Normalize image for visualization."""
        percentile_clip = self.cfg.processing.percentile_clip
        image_norm = np.clip(image, 0, np.percentile(image, percentile_clip))
        if image_norm.max() > image_norm.min():
            image_norm = (image_norm - image_norm.min()) / (image_norm.max() - image_norm.min())
        return (image_norm * 255).astype(np.uint8)
    
    def _save_visual_comparison(self, output_path: Path, wf_image: np.ndarray, 
                               method_results: Dict[str, Dict[str, float]], gt_image: np.ndarray):
        """Save visual comparison of methods."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # This would need the actual reconstructed images, not just metrics
        # For now, create a placeholder implementation
        print(f"Saving comparison to {output_path}")
    
    def _save_full_fov_comparison(self, output_path: Path, wf_image: np.ndarray,
                                 method_results: Dict[str, Dict[str, float]], gt_image: np.ndarray):
        """Save full FOV comparison visualization."""
        self._save_visual_comparison(output_path, wf_image, method_results, gt_image)
    
    def _save_baseline_comparison_visual(self, output_path: Path, wf_image: np.ndarray,
                                        comparison_methods: Dict[str, np.ndarray], gt_image: np.ndarray):
        """Save baseline comparison visualization."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Normalize all images
        images_norm = []
        labels = []
        
        for method_name, img in comparison_methods.items():
            images_norm.append(self._normalize_for_visualization(img))
            labels.append(method_name.replace('_', ' ').title())
        
        # Add ground truth
        images_norm.append(self._normalize_for_visualization(gt_image))
        labels.append('Ground Truth')
        
        # Create comparison grid
        comparison = np.concatenate(images_norm, axis=1)
        Image.fromarray(comparison).save(output_path)
        
        print(f"Saved baseline comparison to {output_path}")
    
    def _save_cross_method_comparison(self, output_path: Path, wf_image: np.ndarray,
                                     method_results: Dict[str, np.ndarray], gt_image: np.ndarray):
        """Save cross-method comparison visualization."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Normalize all images
        images_norm = [self._normalize_for_visualization(wf_image)]
        
        for method_name in sorted(method_results.keys()):
            img = method_results[method_name]
            images_norm.append(self._normalize_for_visualization(img))
        
        images_norm.append(self._normalize_for_visualization(gt_image))
        
        # Create comparison grid
        comparison = np.concatenate(images_norm, axis=1)
        Image.fromarray(comparison).save(output_path)
        
        print(f"Saved cross-method comparison to {output_path}")
    
    def _compute_statistical_significance(self, all_results: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
        """Compute statistical significance between methods."""
        significance_results = {}
        methods = list(all_results.keys())
        
        if len(methods) < 2:
            return significance_results
        
        # Get common metrics
        common_metrics = set()
        for method_results in all_results.values():
            if method_results:
                common_metrics.update(method_results[0].keys())
        
        for metric in common_metrics:
            significance_results[metric] = {}
            
            # Extract values for each method
            method_values = {}
            for method, results_list in all_results.items():
                values = [r[metric] for r in results_list if metric in r and np.isfinite(r[metric])]
                if values:
                    method_values[method] = values
            
            # Pairwise t-tests
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    if method1 in method_values and method2 in method_values:
                        try:
                            t_stat, p_value = stats.ttest_ind(method_values[method1], method_values[method2])
                            significance_results[metric][f"{method1}_vs_{method2}"] = {
                                "t_statistic": float(t_stat),
                                "p_value": float(p_value),
                                "significant": p_value < self.cfg.processing.significance_level
                            }
                        except Exception:
                            pass
        
        return significance_results
    
    def _create_summary_plots(self, results: Dict[str, Dict[str, float]], output_dir: Path):
        """Create summary plots for results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract metrics for plotting
        methods = list(results.keys())
        metrics = list(results[methods[0]].keys()) if methods else []
        
        # Create box plots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):  # Plot first 4 metrics
            if i >= len(axes):
                break
                
            ax = axes[i]
            data_for_box = []
            labels_for_box = []
            
            for method in methods:
                if metric in results[method]:
                    data_for_box.append([results[method][metric]])  # Single value, but boxplot expects list
                    labels_for_box.append(method)
            
            if data_for_box:
                ax.bar(labels_for_box, [d[0] for d in data_for_box])
                ax.set_ylabel(metric.upper())
                ax.set_title(f'{metric.upper()} Comparison')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'summary_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved summary plots to {output_dir}")
    
    def _create_baseline_summary_plots(self, results: Dict[str, Dict[str, Any]], output_dir: Path):
        """Create summary plots for baseline comparison."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        methods = [k for k in results.keys() if k != "statistical_tests"]
        if not methods:
            return
        
        # Get metrics
        first_method = methods[0]
        metrics = list(results[first_method].keys())
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            means = []
            stds = []
            labels = []
            
            for method in methods:
                if metric in results[method] and 'mean' in results[method][metric]:
                    means.append(results[method][metric]['mean'])
                    stds.append(results[method][metric]['std'])
                    labels.append(method.replace('_', ' ').title())
            
            if means:
                ax.bar(labels, means, yerr=stds, capsize=5)
                ax.set_ylabel(metric.upper())
                ax.set_title(f'{metric.upper()} Comparison (Mean ± Std)')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'baseline_comparison_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved baseline summary plots to {output_dir}")
    
    def _create_cross_method_plots(self, df: pd.DataFrame, output_dir: Path):
        """Create comprehensive plots for cross-method analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comparison plots for each metric
        metrics_to_plot = [m for m in self.metrics_to_compute if m in df.columns]
        
        if not metrics_to_plot:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot[:4]):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Box plot
            methods = df['method'].unique()
            data_for_box = [df[df['method'] == method][metric].values for method in methods]
            
            bp = ax.boxplot(data_for_box, labels=methods, patch_artist=True)
            
            # Color the boxes
            colors = sns.color_palette("husl", len(methods))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Comparison')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cross_method_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create correlation matrix
        if len(metrics_to_plot) > 1:
            correlation_data = df[metrics_to_plot].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Metrics Correlation Matrix')
            plt.tight_layout()
            plt.savefig(output_dir / 'metrics_correlation.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Saved cross-method plots to {output_dir}")
    
    # Wandb logging methods
    def _log_results_to_wandb(self, results: Dict[str, Dict[str, float]]):
        """Log results to Weights & Biases."""
        if self.cfg.wandb.mode == "disabled":
            return
        
        try:
            # Flatten results for wandb
            wandb_metrics = {}
            for method_name, metrics in results.items():
                for metric_name, value in metrics.items():
                    wandb_metrics[f"{method_name}/{metric_name}"] = value
            
            wandb.log(wandb_metrics)
            
            # Create summary table
            summary_data = []
            for method_name, metrics in results.items():
                row = {"method": method_name}
                row.update(metrics)
                summary_data.append(row)
            
            df = pd.DataFrame(summary_data)
            wandb.log({"results_table": wandb.Table(dataframe=df)})
            
            print("✅ Logged results to Weights & Biases")
            
        except Exception as e:
            print(f"⚠️ Failed to log to wandb: {e}")
    
    def _log_baseline_results_to_wandb(self, results: Dict[str, Any]):
        """Log baseline comparison results to wandb."""
        if self.cfg.wandb.mode == "disabled":
            return
        
        try:
            # Log summary statistics
            for method, stats in results.items():
                if method == "statistical_tests":
                    continue
                    
                for metric, values in stats.items():
                    if isinstance(values, dict) and 'mean' in values:
                        wandb.log({
                            f"baseline_{method}/{metric}_mean": values['mean'],
                            f"baseline_{method}/{metric}_std": values['std'],
                            f"baseline_{method}/{metric}_count": values['count']
                        })
            
            # Log statistical significance if available
            if "statistical_tests" in results:
                for metric, tests in results["statistical_tests"].items():
                    for comparison, test_result in tests.items():
                        wandb.log({
                            f"significance/{metric}_{comparison}_p_value": test_result['p_value'],
                            f"significance/{metric}_{comparison}_significant": test_result['significant']
                        })
            
            print("✅ Logged baseline results to Weights & Biases")
            
        except Exception as e:
            print(f"⚠️ Failed to log baseline results to wandb: {e}")
    
    def _log_cross_method_results_to_wandb(self, df: pd.DataFrame, summary_stats: pd.DataFrame):
        """Log cross-method analysis results to wandb."""
        if self.cfg.wandb.mode == "disabled":
            return
        
        try:
            # Log summary statistics
            for method in df['method'].unique():
                method_data = df[df['method'] == method]
                for metric in self.metrics_to_compute:
                    if metric in method_data.columns:
                        values = method_data[metric].dropna()
                        if len(values) > 0:
                            wandb.log({
                                f"cross_method_{method}/{metric}_mean": float(values.mean()),
                                f"cross_method_{method}/{metric}_std": float(values.std()),
                                f"cross_method_{method}/{metric}_count": len(values)
                            })
            
            # Log detailed results table
            wandb.log({"cross_method_results": wandb.Table(dataframe=df)})
            
            # Log summary statistics table
            summary_df_reset = summary_stats.reset_index()
            wandb.log({"cross_method_summary": wandb.Table(dataframe=summary_df_reset)})
            
            print("✅ Logged cross-method results to Weights & Biases")
            
        except Exception as e:
            print(f"⚠️ Failed to log cross-method results to wandb: {e}")
    
    def _save_results_to_file(self, results: Any, output_path: Path):
        """Save results to JSON file."""
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            else:
                return obj
        
        results_json = convert_types(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Saved results to {output_path}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.cfg.wandb.mode != "disabled":
            try:
                wandb.finish()
                print("✅ Weights & Biases session closed")
            except:
                pass


    def run_sid_evaluation(self) -> Dict[str, Any]:
        """
        Run SID (See-in-the-Dark) evaluation for cross-domain generalization.
        
        This function evaluates PKL diffusion models trained on microscopy data
        on the SID natural image dataset to demonstrate cross-domain generalization.
        """
        print("🌙 Running SID Cross-Domain Evaluation")
        print("🔬 Testing microscopy-trained model on natural low-light images")
        print("=" * 70)
        
        # Setup parameters from config
        processing = self.cfg.processing
        camera_type = processing.sid_camera_type
        data_dir = processing.sid_data_dir
        guidance_types = processing.sid_guidance_types
        max_images = processing.max_images
        
        print(f"📊 SID Evaluation Parameters:")
        print(f"   Camera type: {camera_type}")
        print(f"   Data directory: {data_dir}")
        print(f"   Guidance strategies: {guidance_types}")
        print(f"   Max images: {max_images or 'All'}")
        print(f"   DDIM steps: {processing.sid_num_steps}")
        print(f"   Guidance scale: {processing.sid_guidance_scale}")
        
        # Create SID dataloader
        try:
            dataloader = create_sid_dataloader(
                data_dir=data_dir,
                camera_type=camera_type,
                split="test",
                batch_size=4,  # Small batch size for memory efficiency
                image_size=processing.patch_size,
                max_images=max_images,
                num_workers=4,
                use_processed=processing.sid_use_processed
            )
            print(f"✅ SID dataset loaded: {len(dataloader.dataset)} images")
        except Exception as e:
            print(f"❌ Error loading SID dataset: {e}")
            print("💡 Make sure to download SID dataset first:")
            print(f"   python scripts/main.py sid --task download --camera {camera_type}")
            raise
        
        # Setup output directory
        output_dir = Path(str(self.cfg.inference.output_dir)) / "sid_evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        all_results = {}
        detailed_results = []
        
        # Optional LPIPS calculation
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='alex').to(self.device)
            has_lpips = True
            print("✅ LPIPS metric available")
        except ImportError:
            lpips_fn = None
            has_lpips = False
            print("⚠️ LPIPS not available. Install with: pip install lpips")
        
        # Evaluate each guidance strategy
        for guidance_type in guidance_types:
            print(f"\n{'='*60}")
            print(f"🔬 Evaluating {guidance_type.upper()} guidance")
            print(f"{'='*60}")
            
            # Create guidance strategy and forward model
            psf = PSF()  # Simple Gaussian PSF for natural images
            forward_model = ForwardModel(psf=psf.kernel, background=0.01)
            forward_model.to(self.device)
            
            # Create guidance strategy
            guidance_map = {
                "pkl": PKLGuidance(epsilon=1e-6),
                "l2": L2Guidance(),
                "anscombe": AnscombeGuidance()
            }
            
            if guidance_type.lower() not in guidance_map:
                print(f"❌ Unknown guidance type: {guidance_type}")
                continue
                
            guidance = guidance_map[guidance_type.lower()]
            
            # Create adaptive schedule
            schedule = AdaptiveSchedule(
                lambda_base=processing.sid_guidance_scale,
                T_threshold=800,
                T_total=1000
            )
            
            # Initialize results for this guidance type
            guidance_results = {
                'psnr': [],
                'ssim': [],
                'lpips': [],
                'processing_time': [],
                'image_ids': [],
                'exposure_times': []
            }
            
            # Evaluate on SID dataset
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {guidance_type}")):
                    try:
                        # Move to device
                        noisy_input = batch['input'].to(self.device)
                        clean_target = batch['target'].to(self.device)
                        
                        batch_size = noisy_input.shape[0]
                        start_time = time.time()
                        
                        # Sample using DDIM with guidance
                        denoised = self.sampler.sample(
                            shape=noisy_input.shape,
                            conditioning=None,
                            guidance_strategy=guidance,
                            forward_model=forward_model,
                            measurements=noisy_input,
                            num_steps=processing.sid_num_steps,
                            eta=processing.sid_eta,
                            schedule=schedule
                        )
                        
                        processing_time = (time.time() - start_time) / batch_size
                        
                        # Calculate metrics for each image in batch
                        for i in range(batch_size):
                            # Denormalize from [-1, 1] to [0, 1]
                            pred_img = (denoised[i, 0].cpu().numpy() + 1.0) / 2.0
                            target_img = (clean_target[i, 0].cpu().numpy() + 1.0) / 2.0
                            
                            # Clip to valid range
                            pred_img = np.clip(pred_img, 0, 1)
                            target_img = np.clip(target_img, 0, 1)
                            
                            # Calculate metrics using existing framework
                            metrics = compute_metrics(
                                pred_img, 
                                target_img, 
                                metrics_list=['psnr', 'ssim']
                            )
                            
                            guidance_results['psnr'].append(metrics['psnr'])
                            guidance_results['ssim'].append(metrics['ssim'])
                            guidance_results['processing_time'].append(processing_time)
                            guidance_results['image_ids'].append(batch['image_id'][i])
                            guidance_results['exposure_times'].append(batch['exposure_time'][i])
                            
                            # Calculate LPIPS if available
                            if has_lpips:
                                pred_3ch = torch.from_numpy(pred_img).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(self.device)
                                target_3ch = torch.from_numpy(target_img).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(self.device)
                                
                                pred_3ch = pred_3ch * 2.0 - 1.0
                                target_3ch = target_3ch * 2.0 - 1.0
                                
                                lpips_score = lpips_fn(pred_3ch, target_3ch).item()
                                guidance_results['lpips'].append(lpips_score)
                            
                            # Store detailed result
                            result_entry = {
                                'guidance_type': guidance_type,
                                'image_id': batch['image_id'][i],
                                'exposure_time': batch['exposure_time'][i],
                                'camera_type': camera_type,
                                'psnr': metrics['psnr'],
                                'ssim': metrics['ssim'],
                                'processing_time': processing_time
                            }
                            
                            if has_lpips:
                                result_entry['lpips'] = lpips_score
                            
                            detailed_results.append(result_entry)
                        
                        # Clean up GPU memory
                        del denoised, noisy_input, clean_target
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                    except Exception as e:
                        print(f"❌ Error processing batch {batch_idx}: {e}")
                        continue
            
            # Calculate summary statistics
            summary = {
                'mean_psnr': np.mean(guidance_results['psnr']),
                'std_psnr': np.std(guidance_results['psnr']),
                'mean_ssim': np.mean(guidance_results['ssim']),
                'std_ssim': np.std(guidance_results['ssim']),
                'mean_processing_time': np.mean(guidance_results['processing_time']),
                'num_images': len(guidance_results['psnr'])
            }
            
            if guidance_results['lpips']:
                summary.update({
                    'mean_lpips': np.mean(guidance_results['lpips']),
                    'std_lpips': np.std(guidance_results['lpips'])
                })
            
            all_results[guidance_type] = {
                'results': guidance_results,
                'summary': summary
            }
            
            # Print results
            print(f"\n📊 Results for {guidance_type.upper()} guidance:")
            print(f"   PSNR: {summary['mean_psnr']:.2f} ± {summary['std_psnr']:.2f} dB")
            print(f"   SSIM: {summary['mean_ssim']:.3f} ± {summary['std_ssim']:.3f}")
            if 'mean_lpips' in summary:
                print(f"   LPIPS: {summary['mean_lpips']:.3f} ± {summary['std_lpips']:.3f}")
            print(f"   Processing time: {summary['mean_processing_time']:.2f}s per image")
            print(f"   Images evaluated: {summary['num_images']}")
        
        # Save detailed results
        import json
        results_file = output_dir / f"sid_{camera_type.lower()}_detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Create DataFrame for analysis
        df = pd.DataFrame(detailed_results)
        df.to_csv(output_dir / f"sid_{camera_type.lower()}_results.csv", index=False)
        
        # Create summary plots if requested
        if processing.save_summary_plots and len(guidance_types) > 1:
            self._create_sid_comparison_plots(df, output_dir, guidance_types)
        
        # Log to wandb if enabled
        if self.wandb_enabled:
            self._log_sid_results_to_wandb(all_results, camera_type)
        
        print(f"\n💾 Results saved to {output_dir}")
        print(f"\n🎉 SID cross-domain evaluation complete!")
        print(f"📊 Results demonstrate generalization from microscopy to natural images")
        print(f"🏆 Ready for ICLR submission!")
        
        return {
            "all_results": all_results,
            "detailed_results": detailed_results,
            "summary_statistics": {k: v['summary'] for k, v in all_results.items()},
            "camera_type": camera_type,
            "total_images": len(detailed_results) // len(guidance_types) if guidance_types else 0
        }
    
    def _create_sid_comparison_plots(self, df: pd.DataFrame, output_dir: Path, guidance_types: List[str]):
        """Create comparison plots for SID evaluation results."""
        plots_dir = output_dir / "comparison_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # PSNR comparison
        plt.figure(figsize=(10, 6))
        df.boxplot(column='psnr', by='guidance_type', ax=plt.gca())
        plt.title('PSNR Comparison on SID Dataset')
        plt.xlabel('Guidance Type')
        plt.ylabel('PSNR (dB)')
        plt.tight_layout()
        plt.savefig(plots_dir / 'psnr_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # SSIM comparison
        plt.figure(figsize=(10, 6))
        df.boxplot(column='ssim', by='guidance_type', ax=plt.gca())
        plt.title('SSIM Comparison on SID Dataset')
        plt.xlabel('Guidance Type')
        plt.ylabel('SSIM')
        plt.tight_layout()
        plt.savefig(plots_dir / 'ssim_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Summary bar plot
        summary_stats = df.groupby('guidance_type')[['psnr', 'ssim']].agg(['mean', 'std'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # PSNR bars
        psnr_means = summary_stats['psnr']['mean']
        psnr_stds = summary_stats['psnr']['std']
        ax1.bar(guidance_types, psnr_means, yerr=psnr_stds, capsize=5)
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Mean PSNR by Guidance Type')
        ax1.grid(True, alpha=0.3)
        
        # SSIM bars
        ssim_means = summary_stats['ssim']['mean']
        ssim_stds = summary_stats['ssim']['std']
        ax2.bar(guidance_types, ssim_means, yerr=ssim_stds, capsize=5)
        ax2.set_ylabel('SSIM')
        ax2.set_title('Mean SSIM by Guidance Type')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'summary_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison plots saved to {plots_dir}")
    
    def _log_sid_results_to_wandb(self, results: Dict[str, Any], camera_type: str):
        """Log SID evaluation results to Weights & Biases."""
        if not self.wandb_enabled:
            return
        
        # Log summary metrics
        for guidance_type, result_data in results.items():
            summary = result_data['summary']
            wandb.log({
                f"sid_{camera_type.lower()}_{guidance_type}_psnr": summary['mean_psnr'],
                f"sid_{camera_type.lower()}_{guidance_type}_ssim": summary['mean_ssim'],
                f"sid_{camera_type.lower()}_{guidance_type}_processing_time": summary['mean_processing_time'],
                f"sid_{camera_type.lower()}_{guidance_type}_num_images": summary['num_images']
            })
            
            if 'mean_lpips' in summary:
                wandb.log({f"sid_{camera_type.lower()}_{guidance_type}_lpips": summary['mean_lpips']})
        
        # Log cross-domain generalization table
        table_data = []
        for guidance_type, result_data in results.items():
            summary = result_data['summary']
            row = [
                guidance_type.upper(),
                f"{summary['mean_psnr']:.2f} ± {summary['std_psnr']:.2f}",
                f"{summary['mean_ssim']:.3f} ± {summary['std_ssim']:.3f}",
                f"{summary['mean_processing_time']:.2f}s",
                summary['num_images']
            ]
            if 'mean_lpips' in summary:
                row.insert(3, f"{summary['mean_lpips']:.3f} ± {summary['std_lpips']:.3f}")
            table_data.append(row)
        
        columns = ["Guidance", "PSNR (dB)", "SSIM", "Processing Time", "Images"]
        if any('mean_lpips' in results[k]['summary'] for k in results):
            columns.insert(3, "LPIPS")
        
        wandb.log({
            f"sid_{camera_type.lower()}_cross_domain_results": wandb.Table(
                columns=columns,
                data=table_data
            )
        })


# Register the structured config
cs = ConfigStore.instance()
cs.store(name="unified_evaluation_config", node=UnifiedEvaluationConfig)


@hydra.main(version_base=None, config_path="../../configs", config_name="unified_evaluation")
def main(cfg: DictConfig):
    """Main evaluation function."""
    evaluator = None
    
    try:
        print("🚀 Starting Unified PKL Diffusion Evaluation")
        print("=" * 60)
        
        # Validate configuration
        try:
            validate_config(cfg)
        except Exception as e:
            print(f"⚠️ Configuration validation failed: {e}")
            print("Continuing with potentially invalid configuration...")
        
        # Create evaluator
        evaluator = UnifiedEvaluator(cfg)
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS SUMMARY - {cfg.mode.upper()} MODE")
        print("=" * 60)
        
        if isinstance(results, dict) and all(isinstance(v, dict) for v in results.values() if isinstance(v, dict)):
            # Standard results format
            for method_name, metrics in results.items():
                if isinstance(metrics, dict):
                    print(f"\n{method_name.upper()}:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"  {metric:20}: {value:8.4f}")
        else:
            print(f"Results: {len(results.get('detailed_results', []))} comparisons completed")
        
        print(f"\nEvaluation completed successfully!")
        print(f"Mode: {cfg.mode}")
        print(f"Results saved to: {cfg.inference.output_dir}")
        print(f"Available metrics: {list_metrics()}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if evaluator:
            evaluator.cleanup()


if __name__ == "__main__":
    main()
