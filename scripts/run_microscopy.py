#!/usr/bin/env python3
"""
Unified Training and Evaluation Script for PKL Diffusion Denoising on Microscopy Data

This script combines both training and evaluation workflows in a single interface,
similar to other diffusion repositories. It supports:
- Training diffusion models on microscopy data
- Evaluating trained models with comprehensive metrics
- Baseline comparisons (Richardson-Lucy, RCAN)
- Both paired and unpaired (self-supervised) training modes

Usage:
    # Training
    python scripts/run_microscopy.py --mode train --config configs/config_real.yaml
    
    # Evaluation
    python scripts/run_microscopy.py --mode eval --checkpoint checkpoints/best_model.pt --input-dir data/test/wf --gt-dir data/test/2p
    
    # Train then evaluate
    python scripts/run_microscopy.py --mode train_eval --config configs/config_real.yaml --eval-input data/test/wf --eval-gt data/test/2p

Examples:
    # Quick training run
    python scripts/run_microscopy.py --mode train --config configs/config_real.yaml --max-epochs 10
    
    # Training with adaptive normalization for better dynamic range
    python scripts/run_microscopy.py --mode train --config configs/config_real.yaml --use-adaptive-normalization --adaptive-percentiles 0.1 99.9
    
    # Full pipeline with evaluation
    python scripts/run_microscopy.py --mode train_eval --config configs/config_real.yaml --eval-input data/test/wf --eval-gt data/test/2p
    
    # Full pipeline with adaptive normalization
    python scripts/run_microscopy.py --mode train_eval --config configs/config_real.yaml --use-adaptive-normalization --eval-input data/test/wf --eval-gt data/test/2p
    
    # Evaluation only with baselines
    python scripts/run_microscopy.py --mode eval --checkpoint checkpoints/best_model.pt --input-dir data/test/wf --gt-dir data/test/2p --include-baselines
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings

import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
import numpy as np
import random
import wandb
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image
import tifffile
import json

# PyTorch Lightning imports
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import WandbLogger, CSVLogger
    HAS_LIGHTNING = True
    print(f"✅ PyTorch Lightning {pl.__version__} imported successfully")
except ImportError as e:
    HAS_LIGHTNING = False
    print(f"❌ PyTorch Lightning import failed: {e}")
    import traceback
    traceback.print_exc()

# Debug: print(f"🔍 HAS_LIGHTNING = {HAS_LIGHTNING}")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core imports
from pkl_dg.models.unet import UNet
from pkl_dg.models.diffusion import DDPMTrainer, UnpairedDataset
from pkl_dg.models.sampler import DDIMSampler
from pkl_dg.data import RealPairsDataset, IntensityToModel, Microscopy16BitToModel

# Optional Zarr import
try:
    from pkl_dg.data import ZarrPatchesDataset
    HAS_ZARR = True
except ImportError:
    ZarrPatchesDataset = None
    HAS_ZARR = False
from pkl_dg.data.adaptive_dataset import create_adaptive_datasets, AdaptiveRealPairsDataset
from pkl_dg.physics import PSF, build_psf_bank, ForwardModel
# Guidance is a single module, not a package
try:
    from pkl_dg.guidance import PKLGuidance, L2Guidance, AnscombeGuidance
    HAS_GUIDANCE = True
except ImportError:
    PKLGuidance = L2Guidance = AnscombeGuidance = None
    HAS_GUIDANCE = False
from pkl_dg.evaluation import Metrics
# from pkl_dg.evaluation.robustness import RobustnessTests
# from pkl_dg.evaluation.hallucination import HallucinationTests
# from pkl_dg.evaluation.tasks import DownstreamTasks
# Baselines import
try:
    from pkl_dg.baseline import richardson_lucy_restore
    HAS_BASELINES = True
except ImportError:
    richardson_lucy_restore = None
    HAS_BASELINES = False
# Memory cleanup import
try:
    from pkl_dg.utils import cleanup_memory
except ImportError:
    def cleanup_memory():
        pass
from pkl_dg.utils import (
    print_config_summary,
    validate_and_complete_config,
    setup_logging
)

# Optional imports
try:
    from pkl_dg.baseline import RCANWrapper
    HAS_RCAN = True
except ImportError:
    HAS_RCAN = False


def setup_experiment(args, cfg: Optional[DictConfig] = None) -> DictConfig:
    """Setup experiment configuration and logging."""
    if cfg is None:
        # Use Hydra to properly load and resolve configuration
        config_path = Path(args.config)
        config_name = config_path.stem
        
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # For configs in the configs directory, use relative path
        if config_path.is_absolute():
            config_path = config_path.relative_to(Path.cwd())
        
        config_dir = str(config_path.parent)
        
        # Initialize Hydra with the config directory (relative to script location)
        # Script is in scripts/, config is in configs/, so we need ../configs
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name=config_name)
            
        print(f"✅ Loaded configuration from: {args.config}")
    
    # Ensure cfg is a DictConfig
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)
    
    # Override config with command line arguments
    if args.max_epochs is not None:
        cfg.training.max_epochs = args.max_epochs
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
        cfg.training.max_epochs = -1  # Disable epoch limit when using steps
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate
    if args.device is not None:
        cfg.experiment.device = args.device
    if args.seed is not None:
        cfg.experiment.seed = args.seed
    
    # Override data configuration with command line arguments
    if hasattr(args, 'use_adaptive_normalization') and args.use_adaptive_normalization:
        cfg.data.use_adaptive_normalization = True
        if not hasattr(cfg.data, 'adaptive'):
            cfg.data.adaptive = {}
        cfg.data.adaptive.percentiles = args.adaptive_percentiles
        
    # Setup paths
    if args.data_dir is not None:
        cfg.paths.data = args.data_dir
    if args.checkpoint_dir is not None:
        cfg.paths.checkpoints = args.checkpoint_dir
    if args.output_dir is not None:
        cfg.paths.outputs = args.output_dir
    
    # Setup logging
    if hasattr(args, 'wandb_mode') and args.wandb_mode is not None:
        cfg.wandb.mode = args.wandb_mode
    
    # Print configuration summary
    if args.verbose:
        print_config_summary(cfg, "Experiment Configuration")
    
    return cfg


def run_training(cfg: DictConfig, args) -> DDPMTrainer:
    """Run training workflow."""
    print("🚀 Starting Training Phase")
    print("=" * 50)
    
    # Set seed for reproducibility
    seed = int(cfg.experiment.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Clear GPU memory and optimize for A40 Tensor Cores
    if torch.cuda.is_available():
        # Clean GPU memory before training
        import gc
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        
        # Optimize for A40 Tensor Cores and performance
        torch.set_float32_matmul_precision('medium')  # Optimize for A40 Tensor Cores
        torch.backends.cudnn.benchmark = True  # Optimize cuDNN for consistent input sizes
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cuDNN
    
    # Initialize W&B if enabled
    if cfg.wandb.mode != "disabled":
        # Create run name with config name and date
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = Path(args.config).stem if args.config else "default"
        run_name = f"{config_name}_{cfg.experiment.name}_{timestamp}"
        
        # Ensure wandb uses project root directory instead of current working directory
        wandb_dir = project_root / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            mode=cfg.wandb.mode,
            dir=str(wandb_dir.parent),  # Set to project root so wandb creates wandb/ subdirectory there
            tags=[config_name, cfg.experiment.name, f"seed_{seed}"]
        )

    # Setup device and paths
    device = str(cfg.experiment.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup paths - ensure all paths are relative to project root
    if Path(cfg.paths.data).is_absolute():
        data_dir = Path(cfg.paths.data)
    else:
        data_dir = project_root / cfg.paths.data
        
    if Path(cfg.paths.checkpoints).is_absolute():
        checkpoint_dir = Path(cfg.paths.checkpoints)
    else:
        checkpoint_dir = project_root / cfg.paths.checkpoints
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup timestamp and config name for consistent naming
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = Path(args.config).stem if args.config else "default"
    
    # Setup file logging - ensure logs go to project root instead of current working directory
    # Convert relative path to absolute path based on project root
    if Path(cfg.paths.logs).is_absolute():
        logs_dir = Path(cfg.paths.logs)
    else:
        logs_dir = project_root / cfg.paths.logs
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Use same naming convention as W&B (timestamp and config_name already defined above)
    log_file = logs_dir / f"training_{config_name}_{cfg.experiment.name}_{timestamp}_seed{seed}.log"
    
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with config: {cfg.experiment.name}")
    print(f"✅ Initialized file logging: {log_file}")

    # If W&B is active, ensure the run name matches the file log stem for consistency
    try:
        if cfg.wandb.mode != "disabled" and getattr(wandb, "run", None) is not None:
            wandb.run.name = log_file.stem
    except Exception:
        pass
    
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Training logs: {log_file}")

    # Setup outputs directory for saving reconstructions/visualizations
    if Path(cfg.paths.outputs).is_absolute():
        outputs_dir = Path(cfg.paths.outputs)
    else:
        outputs_dir = project_root / cfg.paths.outputs
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Create transform based on noise model
    noise_model = str(getattr(cfg.data, "noise_model", "gaussian")).lower()
    if noise_model == "microscopy_16bit":
        transform = Microscopy16BitToModel(
            max_intensity=float(getattr(cfg.data, "max_intensity", 65535)),
            min_intensity=float(getattr(cfg.data, "min_intensity", 0))
        )
        print("✅ Using 16-bit microscopy transform")
    else:
        # Default to intensity normalization for all noise models
        transform = IntensityToModel(
            min_intensity=float(getattr(cfg.data, "min_intensity", 0)),
            max_intensity=float(getattr(cfg.data, "max_intensity", 65535)),
        )
        print(f"✅ Using intensity normalization for {noise_model} noise")

    # Create forward model for cycle consistency (self-supervised training)
    # Check physics config first for PSF settings
    physics_config = getattr(cfg, "physics", {})
    use_psf = getattr(physics_config, "use_psf", False)
    use_bead_psf = getattr(physics_config, "use_bead_psf", False)
    
    if use_psf and use_bead_psf:
        # Load PSF from bead data
        beads_dir = getattr(physics_config, "beads_dir", "data/beads")
        try:
            psf_bank = build_psf_bank(beads_dir)
            bead_mode = getattr(physics_config, "bead_mode", None)
            
            if bead_mode and bead_mode in psf_bank:
                psf_tensor = psf_bank[bead_mode]
                print(f"✅ Loaded PSF from bead data (mode: {bead_mode})")
            elif "with_AO" in psf_bank:
                psf_tensor = psf_bank["with_AO"]
                print("✅ Loaded PSF from bead data (mode: with_AO)")
            elif "no_AO" in psf_bank:
                psf_tensor = psf_bank["no_AO"]
                print("✅ Loaded PSF from bead data (mode: no_AO)")
            else:
                # Use first available PSF
                psf_tensor = next(iter(psf_bank.values()))
                print(f"✅ Loaded PSF from bead data (mode: {list(psf_bank.keys())[0]})")
            
            # Extract sigma values from PSF for logging
            from pkl_dg.physics import psf_params_from_tensor
            sigma_x, sigma_y = psf_params_from_tensor(psf_tensor)
            
            # Extract pixel size information from PSF tensor
            psf_pixel_size_xy_nm = getattr(psf_tensor, 'pixel_size_xy_nm', None)
            target_pixel_size_xy_nm = getattr(physics_config, "target_pixel_size_xy_nm", None)
            
            if psf_pixel_size_xy_nm is not None:
                print(f"   PSF pixel size: {psf_pixel_size_xy_nm:.2f} nm")
                print(f"   PSF parameters (in pixels): σx={sigma_x:.2f}, σy={sigma_y:.2f}")
                
                # Convert sigma to physical units (nm)
                sigma_x_nm = sigma_x * psf_pixel_size_xy_nm
                sigma_y_nm = sigma_y * psf_pixel_size_xy_nm
                print(f"   PSF parameters (in nm): σx={sigma_x_nm:.1f} nm, σy={sigma_y_nm:.1f} nm")
                
                if target_pixel_size_xy_nm is not None:
                    target_pixel_size_xy_nm = float(target_pixel_size_xy_nm)
                    print(f"   Target pixel size: {target_pixel_size_xy_nm:.2f} nm")
                    # Sigma in target pixels will be calculated by ForwardModel scaling
                    target_sigma_x = sigma_x_nm / target_pixel_size_xy_nm
                    target_sigma_y = sigma_y_nm / target_pixel_size_xy_nm
                    print(f"   Target PSF parameters (in target pixels): σx={target_sigma_x:.2f}, σy={target_sigma_y:.2f}")
            else:
                print(f"   PSF parameters (in pixels): σx={sigma_x:.2f}, σy={sigma_y:.2f}")
            
            # Convert to PSF object, preserving pixel size info
            psf_array = psf_tensor.detach().cpu().numpy().astype(np.float32)
            psf = PSF(psf_array=psf_array, pixel_size_xy_nm=psf_pixel_size_xy_nm)
            
        except Exception as e:
            print(f"⚠️ Failed to load PSF from bead data: {e}")
            print("   Falling back to Gaussian PSF")
            # Fallback to Gaussian PSF
            psf_config = getattr(cfg, "psf", {})
            size = int(psf_config.get("size", 21))
            sigma_x = float(psf_config.get("sigma_x", 2.0))
            sigma_y = float(psf_config.get("sigma_y", 2.0))
            
            x = np.arange(size) - size // 2
            y = np.arange(size) - size // 2
            xx, yy = np.meshgrid(x, y)
            psf_array = np.exp(-(xx ** 2 / (2 * sigma_x ** 2) + yy ** 2 / (2 * sigma_y ** 2)))
            psf_array = psf_array.astype(np.float32)
            
            psf = PSF(psf_array=psf_array)
            print(f"✅ Created fallback Gaussian PSF (σx={sigma_x}, σy={sigma_y})")
            
    elif use_psf:
        # Load PSF from file path if specified
        psf_path = getattr(physics_config, "psf_path", None)
        if psf_path:
            psf = PSF(psf_path=psf_path)
            print(f"✅ Loaded PSF from file: {psf_path}")
        else:
            psf = PSF()
            print("✅ Created default PSF")
    else:
        # Fallback to config-based Gaussian PSF
        psf_config = getattr(cfg, "psf", {})
        if psf_config.get("type") == "gaussian":
            size = int(psf_config.get("size", 21))
            sigma_x = float(psf_config.get("sigma_x", 2.0))
            sigma_y = float(psf_config.get("sigma_y", 2.0))
            
            x = np.arange(size) - size // 2
            y = np.arange(size) - size // 2
            xx, yy = np.meshgrid(x, y)
            psf_array = np.exp(-(xx ** 2 / (2 * sigma_x ** 2) + yy ** 2 / (2 * sigma_y ** 2)))
            psf_array = psf_array.astype(np.float32)
            
            psf = PSF(psf_array=psf_array)
            print(f"✅ Created Gaussian PSF (σx={sigma_x}, σy={sigma_y})")
        else:
            psf = PSF()
            print("✅ Created default PSF")
    
    # Get target pixel size for PSF scaling
    target_pixel_size_xy_nm = getattr(physics_config, "target_pixel_size_xy_nm", None)
    if target_pixel_size_xy_nm is not None:
        target_pixel_size_xy_nm = float(target_pixel_size_xy_nm)
    
    forward_model = ForwardModel(
        psf=psf.to_torch(device=device),
        background=float(physics_config.get("background", 0.0)),
        device=device,
        common_sizes=[(int(cfg.data.image_size), int(cfg.data.image_size))],
        target_pixel_size_xy_nm=target_pixel_size_xy_nm
    )

    # Create datasets
    use_zarr = bool(getattr(cfg.data, "use_zarr", False))
    use_self_supervised = bool(getattr(cfg.data, "use_self_supervised", True))
    use_adaptive_normalization = bool(getattr(cfg.data, "use_adaptive_normalization", False))
    
    if use_adaptive_normalization:
        # Use adaptive normalization for better dynamic range utilization
        print("✅ Using adaptive normalization datasets")
        
        # Get adaptive normalization config
        adaptive_cfg = getattr(cfg.data, "adaptive", {})
        percentiles = tuple(getattr(adaptive_cfg, "percentiles", [0.1, 99.9]))
        
        # Resolve data directory path
        if Path(cfg.paths.data).is_absolute():
            adaptive_data_dir = Path(cfg.paths.data)
        else:
            adaptive_data_dir = project_root / cfg.paths.data
            
        # Create adaptive datasets
        datasets = create_adaptive_datasets(
            data_dir=str(adaptive_data_dir),
            batch_size=int(cfg.training.batch_size),
            num_workers=int(cfg.training.num_workers),
            percentiles=percentiles,
            transform=None  # Adaptive normalization handles the normalization
        )
        
        train_dataset = datasets['train_dataset']
        val_dataset = datasets['val_dataset']
        
        # Get normalization parameters for logging
        params = datasets['normalization_params']
        print(f"📊 Adaptive Normalization Parameters:")
        print(f"  WF: [{params.wf_min:.1f}, {params.wf_max:.1f}] -> [-1, 1]")
        print(f"  2P: [{params.tp_min:.1f}, {params.tp_max:.1f}] -> [-1, 1]")
        print(f"✅ Benefits: Better dynamic range utilization, preserved pixel intensity recovery")
        
        # Demonstrate the benefits (similar to the training script)
        print(f"\n✅ Benefits for DDPM Training:")
        
        # Calculate dynamic range improvements
        wf_old_range = 65535.0 - 0.0  # Typical 16-bit range
        tp_old_range = 65535.0 - 0.0
        wf_new_range = params.wf_max - params.wf_min
        tp_new_range = params.tp_max - params.tp_min
        
        wf_improvement = 2.0 / (wf_new_range / wf_old_range) if wf_new_range > 0 else 1.0
        tp_improvement = 2.0 / (tp_new_range / tp_old_range) if tp_new_range > 0 else 1.0
        
        print(f"  • {tp_improvement:.1f}x better dynamic range for 2P data")
        print(f"  • {wf_improvement:.1f}x better dynamic range for WF data")
        print(f"  • Full utilization of [-1, 1] input range")
        print(f"  • Better gradient flow and numerical stability")
        print(f"  • Preserved ability to recover exact pixel intensities")
        
        # Log to W&B if enabled
        if cfg.wandb.mode != "disabled":
            wandb.log({
                "normalization/wf_min": params.wf_min,
                "normalization/wf_max": params.wf_max,
                "normalization/tp_min": params.tp_min,
                "normalization/tp_max": params.tp_max,
                "normalization/wf_range": params.wf_max - params.wf_min,
                "normalization/tp_range": params.tp_max - params.tp_min,
                "normalization/wf_improvement": wf_improvement,
                "normalization/tp_improvement": tp_improvement
            })
            
    elif use_zarr:
        # Use Zarr format for large datasets
        if not HAS_ZARR:
            raise ImportError("Zarr support not available. Install zarr package to use this feature.")
        zarr_train = data_dir / "zarr" / "train.zarr"
        zarr_val = data_dir / "zarr" / "val.zarr"
        train_dataset = ZarrPatchesDataset(str(zarr_train), transform=transform)
        val_dataset = ZarrPatchesDataset(str(zarr_val), transform=transform)
        print("✅ Using Zarr datasets")
    elif use_self_supervised:
        # Use true self-supervised learning with forward model
        # Check for proper train/val directory structure
        train_wf_dir = data_dir / "train" / "wf"
        train_twop_dir = data_dir / "train" / "2p"
        val_wf_dir = data_dir / "val" / "wf"
        val_twop_dir = data_dir / "val" / "2p"
        
        if not all([train_wf_dir.exists(), train_twop_dir.exists(), val_wf_dir.exists(), val_twop_dir.exists()]):
            raise ValueError(f"Missing train/val directories. Expected: {data_dir}/{{train,val}}/{{wf,2p}}")
        
        # Check if we should use paired data for comparison
        use_paired = bool(getattr(cfg.data, "use_paired_for_validation", False))
        use_forward_model = not use_paired  # Use forward model unless explicitly using paired data
        
        train_dataset = UnpairedDataset(
            wf_dir=str(train_wf_dir),
            twop_dir=str(train_twop_dir),
            transform=transform,
            image_size=int(cfg.data.image_size),
            mode="train",
            forward_model=forward_model,
            use_forward_model=use_forward_model,
            add_noise=True,
            noise_level=0.05,
            use_16bit_normalization=bool(getattr(cfg.data, "use_16bit_normalization", True))
        )
        
        val_dataset = UnpairedDataset(
            wf_dir=str(val_wf_dir),
            twop_dir=str(val_twop_dir),
            transform=transform,
            image_size=int(cfg.data.image_size),
            mode="val",
            forward_model=forward_model,
            use_forward_model=use_forward_model,
            add_noise=False,  # No noise during validation
            noise_level=0.0,
            use_16bit_normalization=bool(getattr(cfg.data, "use_16bit_normalization", True))
        )
        
        if use_paired:
            print("✅ Using paired WF/2P datasets for training")
        else:
            print("✅ Using true self-supervised learning (forward model generates synthetic WF)")
    else:
        # Check for unpaired directory structure (fallback)
        train_wf_dir = data_dir / "train" / "wf"
        train_twop_dir = data_dir / "train" / "2p"
        val_wf_dir = data_dir / "val" / "wf"
        val_twop_dir = data_dir / "val" / "2p"
        
        if all([train_wf_dir.exists(), train_twop_dir.exists(), val_wf_dir.exists(), val_twop_dir.exists()]):
            # Check if we should use forward model for true self-supervised learning
            use_forward_model = bool(getattr(cfg.data, "use_forward_model", True))
            
            if use_forward_model:
                print("✅ Using self-supervised learning with forward model")
            else:
                print("✅ Using unpaired self-supervised learning")
                
            train_dataset = UnpairedDataset(
                wf_dir=str(train_wf_dir),
                twop_dir=str(train_twop_dir),
                transform=transform,
                image_size=int(cfg.data.image_size),
                mode="train",
                forward_model=forward_model,
                use_forward_model=use_forward_model,
                add_noise=True,
                noise_level=float(getattr(cfg.data, "noise_level", 0.05)),
                use_16bit_normalization=bool(getattr(cfg.data, "use_16bit_normalization", True))
            )
            val_dataset = UnpairedDataset(
                wf_dir=str(val_wf_dir),
                twop_dir=str(val_twop_dir),
                transform=transform,
                image_size=int(cfg.data.image_size),
                mode="val",
                forward_model=forward_model,
                use_forward_model=use_forward_model,
                add_noise=False,  # No additional noise for validation
                noise_level=0.0,
                use_16bit_normalization=bool(getattr(cfg.data, "use_16bit_normalization", True))
            )
        else:
            # Fallback to paired data structure using Hydra instantiation
            print("✅ Using paired data structure")
            
            # Use Hydra to instantiate datasets with proper configuration
            from hydra.utils import instantiate
            
            # Create train dataset config with resolved paths
            train_cfg = OmegaConf.create(cfg.data)
            train_cfg.split = "train"
            train_cfg.transform = None  # Will be set after instantiation
            train_cfg.mode = "train"
            # Directly set the resolved data_dir path
            train_cfg.data_dir = str(Path(cfg.paths.data) / "real_microscopy")
            
            # Create val dataset config with resolved paths
            val_cfg = OmegaConf.create(cfg.data)
            val_cfg.split = "val"
            val_cfg.transform = None  # Will be set after instantiation
            val_cfg.mode = "val"
            # Directly set the resolved data_dir path
            val_cfg.data_dir = str(Path(cfg.paths.data) / "real_microscopy")
            
            # Instantiate datasets
            train_dataset = instantiate(train_cfg)
            val_dataset = instantiate(val_cfg)
            
            # Set transforms after instantiation
            train_dataset.transform = transform
            val_dataset.transform = transform

    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")

    # Create data loaders - use existing ones if adaptive normalization already created them
    if use_adaptive_normalization:
        # Adaptive datasets already created data loaders
        train_loader = datasets['train_loader']
        val_loader = datasets['val_loader']
        print("✅ Using data loaders from adaptive dataset creation")
    else:
        # Create data loaders for other dataset types
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=True,
            num_workers=int(cfg.training.num_workers),
            pin_memory=True,
            persistent_workers=bool(getattr(cfg.training, "persistent_workers", True)),
            prefetch_factor=int(getattr(cfg.training, "prefetch_factor", 4)),
            drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=int(cfg.training.batch_size),
            shuffle=False,
            num_workers=int(cfg.training.num_workers),
            pin_memory=True,
            persistent_workers=bool(getattr(cfg.training, "persistent_workers", True)),
            prefetch_factor=int(getattr(cfg.training, "prefetch_factor", 4)),
        )

    # Create model
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    use_conditioning = bool(getattr(cfg.training, "use_conditioning", True))
    
    # Ensure correct input channels for conditioning
    if use_conditioning and int(model_cfg.get("in_channels", 1)) == 1:
        model_cfg["in_channels"] = 2  # x_t + WF conditioner

    unet = UNet(model_cfg).to(device)
    print(f"✅ Created U-Net with {sum(p.numel() for p in unet.parameters()):,} parameters")

    # Create trainer
    training_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    
    # Add self-supervised specific config if not present
    if "ddpm_loss_weight" not in training_cfg:
        training_cfg["ddpm_loss_weight"] = 1.0
    if "cycle_loss_weight" not in training_cfg:
        training_cfg["cycle_loss_weight"] = 0.1
    # Ensure perceptual loss is disabled (pure self-supervised approach)
    if "perceptual_loss_weight" not in training_cfg:
        training_cfg["perceptual_loss_weight"] = 0.0

    ddpm_trainer = DDPMTrainer(
        model=unet,
        config=training_cfg,
        forward_model=forward_model,
        transform=transform
    ).to(device)

    # Setup optimizer and training parameters
    optimizer = torch.optim.AdamW(
        ddpm_trainer.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(getattr(cfg.training, "weight_decay", 1e-4))
    )

    # Configure automatic mixed precision (AMP) based on config
    precision_str = str(getattr(cfg.training, "precision", "")).lower()
    use_amp = (
        device == "cuda"
        and (
            ("16" in precision_str) or ("mixed" in precision_str)
            or bool(getattr(cfg.experiment, "mixed_precision", False))
        )
    )
    amp_dtype = torch.bfloat16 if "bf16" in precision_str else torch.float16
    scaler = GradScaler(enabled=use_amp and ("bf16" not in precision_str))
    if use_amp:
        print(f"✅ Using AMP with dtype: {'bf16' if amp_dtype == torch.bfloat16 else 'fp16'}")
    
    max_epochs = int(cfg.training.max_epochs)
    save_every = int(getattr(cfg.training, "save_every_n_epochs", 10))
    grad_clip_val = float(getattr(cfg.training, "gradient_clip_val", 1.0))
    
    print(f"🚀 Starting training for {max_epochs} epochs")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Learning rate: {cfg.training.learning_rate}")

    # Cleanup any self-referential or broken last*.ckpt symlinks to avoid IOError
    try:
        for p in checkpoint_dir.glob('last*.ckpt'):
            if p.is_symlink():
                target = os.readlink(p)
                # Remove if points to itself or points to a missing file
                if target == p.name or not (checkpoint_dir / target).exists():
                    p.unlink(missing_ok=True)
    except Exception:
        pass

    # Use PyTorch Lightning training if available
    if HAS_LIGHTNING:
        print("🚀 Using PyTorch Lightning training with automatic early stopping")
        
        # Setup Lightning logger
        wandb_logger = None
        if cfg.wandb.mode != "disabled":
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_name = Path(args.config).stem if args.config else "default"
            # Use the same naming as the file logger
            run_name = log_file.stem
            
            wandb_logger = WandbLogger(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=run_name,
                mode=cfg.wandb.mode,
                tags=[config_name, cfg.experiment.name, f"seed_{cfg.experiment.seed}"]
            )
            print(f"✅ Initialized Lightning W&B logger: {run_name}")

        # Always add a CSV logger with the same run name so metrics.csv path matches
        csv_logger = CSVLogger(
            save_dir=str(project_root / "lightning_logs"),
            name=log_file.stem
        )

        # Setup Lightning callbacks
        callbacks = []
        
        class EpochMetricsPrinter(pl.Callback):
            def __init__(self, file_logger):
                super().__init__()
                self._logger = file_logger
            
            def on_validation_epoch_end(self, trainer, pl_module):
                metrics = trainer.callback_metrics
                
                def _extract(keys):
                    for key in keys:
                        if key in metrics:
                            val = metrics[key]
                            try:
                                return float(val)
                            except Exception:
                                return val.item() if hasattr(val, 'item') else val
                    # Fallback: search for aggregated epoch metrics
                    for key in metrics.keys():
                        if 'train/loss' in key and 'epoch' in key:
                            val = metrics[key]
                            try:
                                return float(val)
                            except Exception:
                                return val.item() if hasattr(val, 'item') else val
                    return None
                
                train_loss = _extract(['train/loss_epoch', 'train/loss'])
                val_loss = _extract(['val/loss_epoch', 'val/loss'])
                if train_loss is not None or val_loss is not None:
                    msg = (
                        f"Epoch {trainer.current_epoch + 1} - "
                        f"Train Loss: {train_loss if train_loss is not None else 'N/A'}, "
                        f"Val Loss: {val_loss if val_loss is not None else 'N/A'}"
                    )
                    print(msg)
                    self._logger.info(msg)

        class PeriodicStepMetricsLogger(pl.Callback):
            """Log train/val losses every fixed number of training steps (default: 5000)."""
            def __init__(self, file_logger, step_interval: int = 5000):
                super().__init__()
                self._logger = file_logger
                self.step_interval = int(step_interval)

            def on_validation_end(self, trainer, pl_module):
                if hasattr(trainer, 'is_global_zero') and not trainer.is_global_zero:
                    return
                # Called whenever validation runs; align validation to the desired step cadence
                step = int(getattr(trainer, 'global_step', 0))
                if self.step_interval > 0 and step > 0 and step % self.step_interval == 0:
                    metrics = trainer.callback_metrics
                    # Extract best-effort train/val losses
                    def _get(metrics_dict, keys):
                        for k in keys:
                            if k in metrics_dict:
                                v = metrics_dict[k]
                                try:
                                    return float(v)
                                except Exception:
                                    return v.item() if hasattr(v, 'item') else None
                        return None
                    # Prefer per-step training loss if available
                    train_loss = _get(metrics, ['train/loss_step', 'train/loss_epoch', 'train/loss'])
                    val_loss = _get(metrics, ['val/loss_epoch', 'val/loss'])
                    msg = (
                        f"Step {step} (Epoch {trainer.current_epoch + 1}) - "
                        f"Train Loss: {train_loss if train_loss is not None else 'N/A'}, "
                        f"Val Loss: {val_loss if val_loss is not None else 'N/A'}"
                    )
                    print(msg)
                    self._logger.info(msg)

        class PeriodicFullFrameReconstructionSaver(pl.Callback):
            """Save a full-frame reconstruction every fixed number of training steps (default: 10000)."""
            def __init__(self, 
                         file_logger,
                         outputs_path: Path,
                         data_root: Path,
                         cfg: DictConfig,
                         step_interval: int = 10000):
                super().__init__()
                self._logger = file_logger
                self.outputs_path = outputs_path
                self.data_root = data_root
                self.cfg = cfg
                self.step_interval = int(step_interval)

            def _find_validation_frame(self) -> Optional[Path]:
                # Try to find a reasonably large validation WF image
                candidates = []
                for rel in [
                    Path('val_full') / 'wf',
                    Path('val') / 'wf',
                ]:
                    search_dir = (self.data_root / rel)
                    if search_dir.exists():
                        for ext in ('*.tif', '*.tiff', '*.png', '*.jpg'):
                            candidates.extend(sorted(search_dir.glob(ext)))
                return candidates[0] if candidates else None

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                if hasattr(trainer, 'is_global_zero') and not trainer.is_global_zero:
                    return
                step = int(getattr(trainer, 'global_step', 0))
                if self.step_interval <= 0 or step == 0 or step % self.step_interval != 0:
                    return

                try:
                    import tifffile
                    from PIL import Image
                    import numpy as np
                    import torch
                except Exception as e:
                    self._logger.warning(f"Reconstruction deps missing at step {step}: {e}")
                    return

                frame_path = self._find_validation_frame()
                if frame_path is None:
                    self._logger.info(f"[Reconstruct @ {step}] No validation frame found; skipping.")
                    return

                # Load frame
                try:
                    if frame_path.suffix.lower() in ['.tif', '.tiff']:
                        wf = tifffile.imread(str(frame_path)).astype(np.float32)
                    else:
                        wf = np.array(Image.open(frame_path)).astype(np.float32)
                except Exception as e:
                    self._logger.warning(f"[Reconstruct @ {step}] Failed to read {frame_path}: {e}")
                    return

                # Prepare tensor (1,1,H,W)
                if wf.ndim == 2:
                    wf_t = torch.from_numpy(wf)[None, None]
                elif wf.ndim == 3 and wf.shape[0] in (1, 3):
                    wf_t = torch.from_numpy(wf[:1])[None]
                else:
                    # Fallback to single-channel
                    wf_t = torch.from_numpy(wf[..., :1].squeeze(-1))[None, None]

                device = next(pl_module.parameters()).device
                wf_t = wf_t.to(device=device, dtype=torch.float32)

                # Build sampler from trainer (guidance optional)
                try:
                    from pkl_dg.models.sampler import DDIMSampler
                    sampler = DDIMSampler.from_ddpm_trainer(
                        trainer=pl_module,
                        forward_model=getattr(pl_module, 'forward_model', None),
                        guidance_strategy=None,  # Optional; falls back to unguided if None
                        schedule=None,
                        ddim_steps=int(getattr(self.cfg.inference, 'ddim_steps', 50)),
                        eta=float(getattr(self.cfg.inference, 'eta', 0.0)),
                    )
                except Exception as e:
                    self._logger.warning(f"[Reconstruct @ {step}] Failed to create sampler: {e}")
                    return

                # Run sampling (may be heavy for large frames; best-effort only)
                try:
                    with torch.no_grad():
                        pred = sampler.sample(
                            y=wf_t,
                            shape=tuple(wf_t.shape),
                            device=device,
                            verbose=False,
                        )
                        pred_np = pred.squeeze().detach().cpu().numpy().astype(np.float32)
                        # Normalize to >=0 for 16-bit save
                        pred_np = np.clip(pred_np, a_min=0.0, a_max=None)
                except Exception as e:
                    self._logger.warning(f"[Reconstruct @ {step}] Sampling failed: {e}")
                    return

                # Save as 16-bit TIFF
                try:
                    import tifffile
                    save_dir = self.outputs_path / 'reconstructions'
                    save_dir.mkdir(parents=True, exist_ok=True)
                    out_path = save_dir / f"reconstruction_step_{step:06d}.tif"
                    # Scale to 16-bit range using min-max per frame
                    pmin, pmax = float(pred_np.min()), float(pred_np.max())
                    if pmax > pmin:
                        pred_norm = (pred_np - pmin) / (pmax - pmin)
                    else:
                        pred_norm = np.zeros_like(pred_np)
                    tifffile.imwrite(str(out_path), (pred_norm * 65535).astype(np.uint16))
                    self._logger.info(f"[Reconstruct @ {step}] Saved full-frame reconstruction to {out_path}")
                except Exception as e:
                    self._logger.warning(f"[Reconstruct @ {step}] Failed to save reconstruction: {e}")
        
        # Model checkpoint callback - step-based saving (DDPM standard)
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='ddpm-{step:06d}-{val/loss:.4f}',
            monitor='val/loss',  # Lightning uses 'val/loss' format
            mode='min',
            save_top_k=3,
            save_last=True,
            every_n_train_steps=10000,  # Save every 10K steps (main checkpoints)
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Intermediate checkpoint callback - more frequent saves for recovery
        intermediate_checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir / "intermediate",
            filename='ddpm-intermediate-{step:06d}',
            every_n_train_steps=5000,  # Intermediate saves every 5K steps
            save_top_k=-1,  # Save all checkpoints
            save_last=False,  # Don't duplicate last checkpoint
            verbose=False  # Less verbose for intermediate saves
        )
        callbacks.append(intermediate_checkpoint_callback)
        
        # Early stopping callback - step-based
        early_stop_callback = EarlyStopping(
            monitor='val/loss',  # Lightning uses 'val/loss' format
            patience=cfg.training.get('early_stopping_patience_steps', 10),  # 10 validation cycles
            min_delta=cfg.training.get('early_stopping_min_delta', 1e-5),
            mode='min',
            verbose=True,
            check_on_train_epoch_end=False  # Check on validation, not epoch end
        )
        callbacks.append(early_stop_callback)
        
        # Always print end-of-epoch train/val losses
        callbacks.append(EpochMetricsPrinter(logger))
        
        # Setup Lightning trainer with performance optimizations (multi-GPU aware)
        hw_cfg = getattr(cfg, 'hardware', {})
        accelerator_arg = hw_cfg.get('accelerator', 'gpu' if device == 'cuda' else 'cpu')
        devices_arg = int(hw_cfg.get('devices', 1))
        strategy_arg = hw_cfg.get('strategy', 'auto')
        sync_bn = bool(hw_cfg.get('sync_batchnorm', False))

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            max_steps=(int(getattr(cfg.training, 'max_steps', 0)) if int(getattr(cfg.training, 'max_steps', 0)) > 0 else None),
            accelerator=accelerator_arg,
            devices=devices_arg,
            precision='16-mixed' if use_amp else '32-true',
            check_val_every_n_epoch=None,  # Use step-based validation across total training steps
            callbacks=callbacks + [
                PeriodicStepMetricsLogger(logger, step_interval=int(getattr(cfg.training, 'metrics_log_steps', 5000))),
                PeriodicFullFrameReconstructionSaver(
                    file_logger=logger,
                    outputs_path=outputs_dir,
                    data_root=data_dir,
                    cfg=cfg,
                    step_interval=int(getattr(cfg.training, 'reconstruction_save_steps', 10000))
                ),
            ],
            logger=[l for l in [wandb_logger, csv_logger] if l is not None],
            gradient_clip_val=grad_clip_val,
            accumulate_grad_batches=int(cfg.training.get('accumulate_grad_batches', 1)),
            log_every_n_steps=int(cfg.training.get('log_every_n_steps', 100)),
            # Run validation every fixed number of training steps (int) or epoch fraction (float)
            val_check_interval=(
                int(cfg.training.get('val_check_interval', cfg.training.get('val_check_interval_steps', 5000)))
                if isinstance(cfg.training.get('val_check_interval', None), int)
                else float(cfg.training.get('val_check_interval', 1.0))
            ),
            limit_val_batches=float(cfg.training.get('limit_val_batches', 1.0)),  # Limit validation batches for speed
            enable_progress_bar=True,
            enable_model_summary=False,  # Disable for speed
            deterministic=False,  # For performance
            benchmark=True,  # Enable cuDNN benchmarking
            inference_mode=False,  # Keep gradients for validation
            sync_batchnorm=sync_bn,
            num_sanity_val_steps=0,  # Skip sanity check for faster startup
            strategy=strategy_arg,
        )
        
        # Train the model
        trainer.fit(ddpm_trainer, train_loader, val_loader)
        
        print(f"✅ Lightning training completed!")
        print(f"📁 Checkpoints saved to: {checkpoint_dir}")
        best_path = checkpoint_callback.best_model_path or "None"
        best_score = checkpoint_callback.best_model_score
        print(f"🏆 Best model path: {best_path}")
        if best_score is not None:
            try:
                print(f"🏆 Best model score: {float(best_score):.6f}")
            except Exception:
                print(f"🏆 Best model score: {best_score}")
        else:
            print("🏆 Best model score: None")
        
        logger.info(f"Lightning training completed!")
        logger.info(f"Best model path: {checkpoint_callback.best_model_path}")
        try:
            best_score_val = (
                float(checkpoint_callback.best_model_score)
                if checkpoint_callback.best_model_score is not None else None
            )
        except Exception:
            best_score_val = None
        logger.info(
            f"Best model score: {best_score_val:.6f}" if best_score_val is not None else "Best model score: None"
        )
        
    else:
        # Fallback to manual training loop if Lightning is not available
        print("⚠️ PyTorch Lightning not available, using manual training loop")
        print("🚀 Starting manual training with manual early stopping")
        
        best_loss = float('inf')
        patience = cfg.training.get('early_stopping_patience', 25)
        min_delta = cfg.training.get('early_stopping_min_delta', 1e-5)
        epochs_without_improvement = 0
        
        for epoch in range(max_epochs):
            ddpm_trainer.train()
            epoch_loss = 0.0
            num_batches = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)
            
            for batch_idx, batch in enumerate(progress):
                # Move batch to device
                if isinstance(batch, (tuple, list)):
                    batch = [b.to(device, non_blocking=True) for b in batch]
                else:
                    batch = batch.to(device, non_blocking=True)

                optimizer.zero_grad()

                # Forward pass with mixed precision when enabled
                if use_amp:
                    # Prefer new torch.amp.autocast API when available, else fall back
                    try:
                        with torch.amp.autocast('cuda', dtype=amp_dtype):
                            loss = ddpm_trainer.training_step(batch, batch_idx)
                    except Exception:
                        # Fallback API: torch.cuda.amp.autocast (no device_type kw)
                        with autocast(dtype=amp_dtype):
                            loss = ddpm_trainer.training_step(batch, batch_idx)
                    
                    scaler.scale(loss).backward()
                    
                    if grad_clip_val > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(ddpm_trainer.parameters(), grad_clip_val)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = ddpm_trainer.training_step(batch, batch_idx)
                    loss.backward()
                    
                    if grad_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(ddpm_trainer.parameters(), grad_clip_val)
                    
                    optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Update progress bar
                progress.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'avg_loss': f'{epoch_loss/num_batches:.6f}'
                })

                # Memory cleanup
                if batch_idx % 50 == 0:
                    cleanup_memory()

            # Calculate average training loss
            avg_train_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {avg_train_loss:.6f}")
            logger.info(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {avg_train_loss:.6f}")

            # Update EMA if enabled
            if hasattr(ddpm_trainer, 'ema_model') and ddpm_trainer.ema_model is not None:
                ddpm_trainer.update_ema()

            # Run validation at end of epoch
            ddpm_trainer.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                val_progress = tqdm(val_loader, desc=f"Validation {epoch+1}/{max_epochs}", leave=False)
                for val_batch_idx, val_batch in enumerate(val_progress):
                    # Move batch to device
                    if isinstance(val_batch, (tuple, list)):
                        val_batch = [b.to(device, non_blocking=True) for b in val_batch]
                    else:
                        val_batch = val_batch.to(device, non_blocking=True)
                    
                    # Validation step
                    val_step_loss = ddpm_trainer.validation_step(val_batch, val_batch_idx)
                    val_loss += val_step_loss.item()
                    val_batches += 1
                    
                    val_progress.set_postfix({'val_loss': f'{val_step_loss.item():.6f}'})
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            logger.info(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            # Early stopping logic
            if avg_val_loss < best_loss - min_delta:
                best_loss = avg_val_loss
                epochs_without_improvement = 0
                is_best = True
                logger.info(f"New best model found with val loss: {best_loss:.6f}")
            else:
                epochs_without_improvement += 1
                is_best = False
                logger.info(f"No improvement for {epochs_without_improvement}/{patience} epochs")
                
            # Check early stopping
            if epochs_without_improvement >= patience:
                print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
                print(f"Best validation loss: {best_loss:.6f}")
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                logger.info(f"Best validation loss: {best_loss:.6f}")
                break

            if (epoch + 1) % save_every == 0 or is_best or epoch == max_epochs - 1:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': ddpm_trainer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'best_val_loss': best_loss,
                    'config': cfg,
                    'args': vars(args),
                    'timestamp': str(Path(__file__).stat().st_mtime),  # Training script timestamp
                }
                
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
                
                if is_best:
                    best_path = checkpoint_dir / 'best_model.pt'
                    torch.save(checkpoint, best_path)
                    print(f"💾 Saved best model: {best_path}")
                    logger.info(f"Saved best model: {best_path}")
                
                latest_path = checkpoint_dir / 'latest_checkpoint.pt'
                torch.save(checkpoint, latest_path)

            # W&B logging for manual training
            if cfg.wandb.mode != "disabled":
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "best_val_loss": best_loss,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

            # Memory cleanup
            cleanup_memory()

        print(f"✅ Manual training completed! Best loss: {best_loss:.6f}")
        print(f"📁 Checkpoints saved to: {checkpoint_dir}")
        logger.info(f"Manual training completed! Best loss: {best_loss:.6f}")
        
        # Close W&B run if enabled
        if cfg.wandb.mode != "disabled":
            wandb.finish()
        
    return ddpm_trainer


def load_model_and_sampler(cfg: DictConfig, checkpoint_path: str, guidance_type: str, device: str):
    """Load trained model and create sampler with specified guidance."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    use_conditioning = bool(getattr(cfg.training, "use_conditioning", False))

    unet = UNet(model_cfg).to(device)
    
    # Load model weights
    ddpm_trainer = DDPMTrainer(model=unet, config={})
    ddpm_trainer.load_state_dict(checkpoint['model_state_dict'])
    ddpm_trainer.eval()
    
    # Create transform
    noise_model = str(getattr(cfg.data, "noise_model", "gaussian")).lower()
    if noise_model == "microscopy_16bit":
        transform = Microscopy16BitToModel(
            max_intensity=float(getattr(cfg.data, "max_intensity", 65535)),
            min_intensity=float(getattr(cfg.data, "min_intensity", 0))
        )
    else:
        # Default to intensity normalization for all noise models
        transform = IntensityToModel(
            min_intensity=float(getattr(cfg.data, "min_intensity", 0)),
            max_intensity=float(getattr(cfg.data, "max_intensity", 65535)),
        )
    
    # Create guidance - use same PSF logic as training
    physics_config = getattr(cfg, "physics", {})
    use_psf = getattr(physics_config, "use_psf", False)
    use_bead_psf = getattr(physics_config, "use_bead_psf", False)
    
    if use_psf and use_bead_psf:
        # Load PSF from bead data
        beads_dir = getattr(physics_config, "beads_dir", "data/beads")
        try:
            psf_bank = build_psf_bank(beads_dir)
            bead_mode = getattr(physics_config, "bead_mode", None)
            
            if bead_mode and bead_mode in psf_bank:
                psf_tensor = psf_bank[bead_mode]
            elif "with_AO" in psf_bank:
                psf_tensor = psf_bank["with_AO"]
            elif "no_AO" in psf_bank:
                psf_tensor = psf_bank["no_AO"]
            else:
                psf_tensor = next(iter(psf_bank.values()))
            
            psf_array = psf_tensor.detach().cpu().numpy().astype(np.float32)
            psf = PSF(psf_array=psf_array)
            
        except Exception:
            # Fallback to Gaussian PSF
            psf_config = getattr(cfg, "psf", {})
            size = int(psf_config.get("size", 21))
            sigma_x = float(psf_config.get("sigma_x", 2.0))
            sigma_y = float(psf_config.get("sigma_y", 2.0))
            
            x = np.arange(size) - size // 2
            y = np.arange(size) - size // 2
            xx, yy = np.meshgrid(x, y)
            psf_array = np.exp(-(xx ** 2 / (2 * sigma_x ** 2) + yy ** 2 / (2 * sigma_y ** 2)))
            psf_array = psf_array.astype(np.float32)
            
            psf = PSF(psf_array=psf_array)
            
    elif use_psf:
        psf_path = getattr(physics_config, "psf_path", None)
        if psf_path:
            psf = PSF(psf_path=psf_path)
        else:
            psf = PSF()
    else:
        # Fallback to config-based Gaussian PSF
        psf_config = getattr(cfg, "psf", {})
        if psf_config.get("type") == "gaussian":
            size = int(psf_config.get("size", 21))
            sigma_x = float(psf_config.get("sigma_x", 2.0))
            sigma_y = float(psf_config.get("sigma_y", 2.0))
            
            x = np.arange(size) - size // 2
            y = np.arange(size) - size // 2
            xx, yy = np.meshgrid(x, y)
            psf_array = np.exp(-(xx ** 2 / (2 * sigma_x ** 2) + yy ** 2 / (2 * sigma_y ** 2)))
            psf_array = psf_array.astype(np.float32)
            
            psf = PSF(psf_array=psf_array)
        else:
            psf = PSF()
    
    # Get target pixel size for PSF scaling
    target_pixel_size_xy_nm = getattr(physics_config, "target_pixel_size_xy_nm", None)
    if target_pixel_size_xy_nm is not None:
        target_pixel_size_xy_nm = float(target_pixel_size_xy_nm)
    
    forward_model = ForwardModel(
        psf=psf.to_torch(device=device),
        background=float(physics_config.get("background", 0.0)),
        device=device,
        target_pixel_size_xy_nm=target_pixel_size_xy_nm
    )
    
    guidance_cfg = getattr(cfg, "guidance", {})
    if guidance_type == "pkl":
        guidance = PKLGuidance(forward_model, **guidance_cfg)
    elif guidance_type == "anscombe":
        guidance = AnscombeGuidance(forward_model, **guidance_cfg)
    else:  # l2
        guidance = L2Guidance(forward_model, **guidance_cfg)
    
    # Create sampler
    sampler = DDIMSampler(
        ddpm_trainer.model,
        guidance=guidance,
        transform=transform,
        num_timesteps=int(getattr(cfg.training, "num_timesteps", 1000))
    )
    
    return sampler


# Import consolidated metrics function
    from pkl_dg.metrics import compute_standard_metrics as compute_evaluation_metrics


def compute_baseline_metrics(wf_input: np.ndarray, gt: np.ndarray, psf: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """Compute baseline method results using centralized metrics."""
    # Import centralized metrics to avoid duplication
    from pkl_dg.metrics import compute_standard_metrics
    
    results = {}
    
    # Richardson-Lucy baseline
    if psf is not None:
        try:
            rl_result = richardson_lucy_restore(wf_input, psf, iterations=30)
            results["richardson_lucy"] = compute_standard_metrics(rl_result, gt)
        except Exception as e:
            print(f"⚠️ Richardson-Lucy failed: {e}")
            results["richardson_lucy"] = {"psnr": float('nan'), "ssim": float('nan'), "frc": float('nan')}
    
    # RCAN baseline (if available)
    if HAS_RCAN:
        try:
            # Note: This would require a trained RCAN model
            # results["rcan"] = compute_standard_metrics(rcan_result, gt)
            pass
        except Exception:
            pass
    
    return results


def run_evaluation(cfg: DictConfig, args) -> Dict[str, Dict[str, float]]:
    """Run evaluation workflow."""
    print("🔍 Starting Evaluation Phase")
    print("=" * 50)
    
    device = str(cfg.experiment.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    input_dir = Path(args.input_dir)
    gt_dir = Path(args.gt_dir)
    
    print(f"Device: {device}")
    print(f"Input directory: {input_dir}")
    print(f"Ground truth directory: {gt_dir}")
    
    # Load checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = Path(cfg.paths.checkpoints) / "best_model.pt"
    
    print(f"Model checkpoint: {checkpoint_path}")
    
    # Get image pairs
    image_paths = []
    for ext in ['*.tif', '*.tiff', '*.png', '*.jpg']:
        image_paths.extend(input_dir.glob(ext))
    
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")
    
    print(f"Found {len(image_paths)} images to evaluate")
    
    # Load models and samplers for different guidance types
    guidance_types = ['l2', 'anscombe', 'pkl']
    samplers = {}
    
    for guidance_type in guidance_types:
        try:
            samplers[guidance_type] = load_model_and_sampler(cfg, checkpoint_path, guidance_type, device)
            print(f"✅ Loaded {guidance_type.upper()} sampler")
        except Exception as e:
            print(f"⚠️ Failed to load {guidance_type} sampler: {e}")
    
    # Initialize results accumulator
    results = {}
    counts = {}
    
    def accumulate_results(method_name: str, metrics: Dict[str, float]):
        if method_name not in results:
            results[method_name] = {k: 0.0 for k in metrics}
            counts[method_name] = 0
        
        for k, v in metrics.items():
            if np.isfinite(v):
                results[method_name][k] += float(v)
        counts[method_name] += 1
    
    # Load PSF for baselines if available
    psf = None
    if args.include_baselines:
        physics_config = getattr(cfg, "physics", {})
        use_psf = getattr(physics_config, "use_psf", False)
        use_bead_psf = getattr(physics_config, "use_bead_psf", False)
        
        if use_psf and use_bead_psf:
            # Load PSF from bead data
            beads_dir = getattr(physics_config, "beads_dir", "data/beads")
            try:
                psf_bank = build_psf_bank(beads_dir)
                bead_mode = getattr(physics_config, "bead_mode", None)
                
                if bead_mode and bead_mode in psf_bank:
                    psf_tensor = psf_bank[bead_mode]
                elif "with_AO" in psf_bank:
                    psf_tensor = psf_bank["with_AO"]
                elif "no_AO" in psf_bank:
                    psf_tensor = psf_bank["no_AO"]
                else:
                    psf_tensor = next(iter(psf_bank.values()))
                
                psf_obj = PSF(psf_array=psf_tensor.detach().cpu().numpy().astype(np.float32))
                psf = psf_obj.psf
                
            except Exception:
                # Fallback to Gaussian PSF
                psf_config = getattr(cfg, "psf", {})
                if psf_config.get("type") == "gaussian":
                    size = int(psf_config.get("size", 21))
                    sigma_x = float(psf_config.get("sigma_x", 2.0))
                    sigma_y = float(psf_config.get("sigma_y", 2.0))
                    
                    x = np.arange(size) - size // 2
                    y = np.arange(size) - size // 2
                    xx, yy = np.meshgrid(x, y)
                    psf_array = np.exp(-(xx ** 2 / (2 * sigma_x ** 2) + yy ** 2 / (2 * sigma_y ** 2)))
                    psf_array = psf_array.astype(np.float32)
                    
                    psf_obj = PSF(psf_array=psf_array)
                    psf = psf_obj.psf
                    
        elif use_psf:
            psf_path = getattr(physics_config, "psf_path", None)
            if psf_path:
                psf_obj = PSF(psf_path=psf_path)
                psf = psf_obj.psf
        else:
            # Fallback to config-based Gaussian PSF
            psf_config = getattr(cfg, "psf", {})
            if psf_config.get("type") == "gaussian":
                size = int(psf_config.get("size", 21))
                sigma_x = float(psf_config.get("sigma_x", 2.0))
                sigma_y = float(psf_config.get("sigma_y", 2.0))
                
                x = np.arange(size) - size // 2
                y = np.arange(size) - size // 2
                xx, yy = np.meshgrid(x, y)
                psf_array = np.exp(-(xx ** 2 / (2 * sigma_x ** 2) + yy ** 2 / (2 * sigma_y ** 2)))
                psf_array = psf_array.astype(np.float32)
                
                psf_obj = PSF(psf_array=psf_array)
                psf = psf_obj.psf
    
    # Import centralized metrics once
    from pkl_dg.metrics import compute_standard_metrics as compute_evaluation_metrics

    # Process each image
    for img_path in tqdm(image_paths, desc="Evaluating"):
        # Load input and ground truth
        if img_path.suffix.lower() in ['.tif', '.tiff']:
            wf_input = tifffile.imread(str(img_path)).astype(np.float32)
        else:
            wf_input = np.array(Image.open(img_path)).astype(np.float32)
        
        gt_path = gt_dir / img_path.name
        if not gt_path.exists():
            print(f"⚠️ Ground truth not found for {img_path.name}, skipping")
            continue
            
        if gt_path.suffix.lower() in ['.tif', '.tiff']:
            gt = tifffile.imread(str(gt_path)).astype(np.float32)
        else:
            gt = np.array(Image.open(gt_path)).astype(np.float32)
        
        # Normalize inputs
        wf_input = wf_input / np.max(wf_input)
        gt = gt / np.max(gt)
        
        # Wide-field baseline
        wf_metrics = compute_evaluation_metrics(wf_input, gt)
        accumulate_results("wide_field", wf_metrics)
        
        # Baseline methods
        if args.include_baselines:
            baseline_results = compute_baseline_metrics(wf_input, gt, psf)
            for method, metrics in baseline_results.items():
                accumulate_results(method, metrics)
        
        # Diffusion methods
        for guidance_type, sampler in samplers.items():
            try:
                # Convert to tensor
                wf_tensor = torch.from_numpy(wf_input).float().to(device)
                if wf_tensor.ndim == 2:
                    wf_tensor = wf_tensor.unsqueeze(0).unsqueeze(0)
                
                # Create conditioner if needed
                conditioning_type = str(getattr(cfg.training, "conditioning_type", "wf")).lower()
                conditioner = sampler.transform(wf_tensor) if conditioning_type == "wf" else None
                
                # Sample
                pred_tensor = sampler.sample(
                    wf_tensor, 
                    tuple(wf_tensor.shape), 
                    device=device, 
                    verbose=False, 
                    conditioner=conditioner
                )
                
                pred = pred_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
                
                # Compute metrics
                diffusion_metrics = compute_evaluation_metrics(pred, gt)
                accumulate_results(f"diffusion_{guidance_type}", diffusion_metrics)
                
            except Exception as e:
                print(f"⚠️ Failed to evaluate {guidance_type}: {e}")
    
    # Compute average results
    final_results = {}
    for method_name, accumulated in results.items():
        if counts[method_name] > 0:
            final_results[method_name] = {
                k: v / counts[method_name] for k, v in accumulated.items()
            }
    
    # Print results
    print("\n📊 Evaluation Results")
    print("=" * 50)
    for method, metrics in final_results.items():
        print(f"{method}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()
    
    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"💾 Results saved to: {results_path}")
    
    return final_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Training and Evaluation Script for PKL Diffusion Denoising",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', '-m',
        choices=['train', 'eval', 'train_eval'],
        required=True,
        help='Operation mode: train, eval, or train_eval (train then evaluate)'
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (required for training)'
    )
    
    # Training arguments
    parser.add_argument('--max-epochs', type=int, help='Maximum training epochs')
    parser.add_argument('--max-steps', type=int, help='Maximum training steps (overrides max-epochs)')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, help='Device (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    # Data paths
    parser.add_argument('--data-dir', type=str, help='Training data directory')
    parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--use-adaptive-normalization', action='store_true', 
                       help='Use adaptive normalization for better dynamic range utilization')
    parser.add_argument('--adaptive-percentiles', nargs=2, type=float, default=[0.1, 99.9],
                       help='Percentile range for adaptive normalization (default: 0.1 99.9)')
    
    # Evaluation arguments
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint for evaluation')
    parser.add_argument('--input-dir', type=str, help='Input directory for evaluation')
    parser.add_argument('--gt-dir', type=str, help='Ground truth directory for evaluation')
    parser.add_argument('--eval-input', type=str, help='Input directory for train_eval mode')
    parser.add_argument('--eval-gt', type=str, help='Ground truth directory for train_eval mode')
    parser.add_argument('--include-baselines', action='store_true', help='Include baseline methods in evaluation')
    
    # Logging
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], help='W&B logging mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['train', 'train_eval'] and not args.config:
        parser.error("--config is required for training modes")
    
    if args.mode in ['eval'] and not (args.input_dir and args.gt_dir):
        parser.error("--input-dir and --gt-dir are required for evaluation mode")
    
    if args.mode == 'train_eval' and not (args.eval_input and args.eval_gt):
        parser.error("--eval-input and --eval-gt are required for train_eval mode")
    
    try:
        # Setup experiment
        cfg = None
        if args.config:
            cfg = setup_experiment(args)
        
        # Run training
        if args.mode in ['train', 'train_eval']:
            trainer = run_training(cfg, args)
            
            if args.mode == 'train':
                print("🎉 Training completed successfully!")
                return 0
        
        # Run evaluation
        if args.mode in ['eval', 'train_eval']:
            if args.mode == 'train_eval':
                # Use train_eval specific arguments
                args.input_dir = args.eval_input
                args.gt_dir = args.eval_gt
                # Use the best checkpoint from training
                args.checkpoint = str(Path(cfg.paths.checkpoints) / "best_model.pt")
            
            if not cfg:
                # Load config from checkpoint for eval-only mode
                checkpoint = torch.load(args.checkpoint, map_location='cpu')
                cfg = checkpoint['config']
            
            results = run_evaluation(cfg, args)
            print("🎉 Evaluation completed successfully!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        cleanup_memory()


if __name__ == "__main__":
    sys.exit(main())
