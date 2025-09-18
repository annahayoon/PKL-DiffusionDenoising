#!/usr/bin/env python3
"""
Unified Training and Evaluation Script for PKL Diffusion Denoising on Microscopy Data

This script combines both training and evaluation workflows in a single interface,
similar to other diffusion repositories. It supports:
- Training diffusion models on microscopy data
- Evaluating trained models with comprehensive metrics
- Baseline comparison (Richardson-Lucy)
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
import itertools
from datetime import datetime
import csv

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
from contextlib import nullcontext
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
from pkl_dg.data.adaptive_dataset import create_adaptive_datasets, AdaptiveRealPairsDataset
from pkl_dg.physics import PSF, ForwardModel, psf_from_config
# Guidance is a single module, not a package
try:
    from pkl_dg.guidance import PKLGuidance, KLGuidance, L2Guidance, AnscombeGuidance, AdaptiveSchedule
    HAS_GUIDANCE = True
except ImportError:
    PKLGuidance = KLGuidance = L2Guidance = AnscombeGuidance = AdaptiveSchedule = None
    HAS_GUIDANCE = False
from pkl_dg.evaluation import Metrics
try:
    from pkl_dg.evaluation import EvaluationSuite
    HAS_EVAL_SUITE = True
except Exception:
    EvaluationSuite = None
    HAS_EVAL_SUITE = False
# (Optional evaluation extras available in docs)
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

#


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
    print("🚀 Starting Training Phase\n" + "=" * 50)
    
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
    # Honor configured logging level if provided
    log_level_str = str(getattr(getattr(cfg, 'logging', {}), 'log_level', 'INFO')).upper()
    try:
        resolved_log_level = getattr(logging, log_level_str)
    except Exception:
        resolved_log_level = logging.INFO
    logging.basicConfig(
        level=resolved_log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with config: {cfg.experiment.name}")
    print(f"✅ File logging: {log_file}")

    # If W&B is active, ensure the run name matches the file log stem for consistency
    try:
        if cfg.wandb.mode != "disabled" and getattr(wandb, "run", None) is not None:
            wandb.run.name = log_file.stem
    except Exception:
        pass
    
    print(f"Device: {device}\nData: {data_dir}\nCheckpoints: {checkpoint_dir}\nLogs: {log_file}")

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
    physics_config = getattr(cfg, "physics", {})
    psf_obj, target_pixel_size_xy_nm = psf_from_config(physics_config)
    forward_model = ForwardModel(
        psf=psf_obj.to_torch(device=device),
        background=float(getattr(physics_config, "background", 0.0)),
        device=device,
        common_sizes=[(int(cfg.data.image_size), int(cfg.data.image_size))],
        read_noise_sigma=float(getattr(physics_config, "read_noise_sigma", 0.0)),
        target_pixel_size_xy_nm=target_pixel_size_xy_nm
    )

    # Create datasets
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
        # Build DataLoader kwargs honoring pinning and pin_memory_device if available (PyTorch >= 2.3)
        _pin_memory = bool(getattr(cfg.training, "pin_memory", True))
        _pin_mem_dev = getattr(cfg.training, "dataloader_pin_memory_device", None)
        _torch_version = tuple(int(p) for p in str(torch.__version__).split("+")[0].split(".")[:2])
        _num_workers = int(cfg.training.num_workers)
        _prefetch = int(getattr(cfg.training, "prefetch_factor", 4)) if _num_workers > 0 else None
        _persistent = bool(getattr(cfg.training, "persistent_workers", True)) if _num_workers > 0 else False
        dl_common_kwargs = dict(
            batch_size=int(cfg.training.batch_size),
            shuffle=True,
            num_workers=_num_workers,
            pin_memory=_pin_memory,
            drop_last=True,
        )
        if _persistent:
            dl_common_kwargs["persistent_workers"] = True
        if _prefetch is not None:
            dl_common_kwargs["prefetch_factor"] = _prefetch
        if _pin_mem_dev and _torch_version >= (2, 3) and torch.cuda.is_available():
            dl_common_kwargs["pin_memory_device"] = str(_pin_mem_dev)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            **dl_common_kwargs,
        )

        dl_val_kwargs = dict(
            batch_size=int(cfg.training.batch_size),
            shuffle=False,
            num_workers=_num_workers,
            pin_memory=_pin_memory,
        )
        if _persistent:
            dl_val_kwargs["persistent_workers"] = True
        if _prefetch is not None:
            dl_val_kwargs["prefetch_factor"] = _prefetch
        if _pin_mem_dev and _torch_version >= (2, 3) and torch.cuda.is_available():
            dl_val_kwargs["pin_memory_device"] = str(_pin_mem_dev)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            **dl_val_kwargs,
        )

    # Create model
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    use_conditioning = bool(getattr(cfg.training, "use_conditioning", False))
    
    # Ensure correct input channels for conditioning
    if use_conditioning and int(model_cfg.get("in_channels", 1)) == 1:
        model_cfg["in_channels"] = 2  # x_t + WF conditioner

    # Propagate gradient checkpointing preference from training/optimization config into model config
    try:
        use_gc = bool(
            getattr(cfg.training, "gradient_checkpointing", False)
            or getattr(getattr(cfg, "optimization", {}), "use_gradient_checkpointing", False)
            or getattr(getattr(cfg, "optimization", {}), "gradient_checkpointing", False)
        )
    except Exception:
        use_gc = False
    if use_gc:
        model_cfg["gradient_checkpointing"] = True

    unet = UNet(model_cfg).to(device)
    print(f"✅ Created U-Net with {sum(p.numel() for p in unet.parameters()):,} parameters")

    # Optionally compile the model with PyTorch 2.x
    try:
        if bool(getattr(cfg.training, "compile_model", False)):
            compile_mode = getattr(cfg.training, "compile_mode", "reduce-overhead")
            unet = torch.compile(unet, mode=compile_mode)
            print(f"✅ torch.compile enabled (mode={compile_mode})")
    except Exception as e:
        print(f"⚠️ torch.compile failed or unavailable: {e}")

    # Create trainer
    training_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    # Propagate channels-last preference into training config for use in training steps
    try:
        if bool(getattr(getattr(cfg, "optimization", {}), "channels_last", False)):
            training_cfg["channels_last"] = True
    except Exception:
        pass
    
    # Add self-supervised specific config if not present
    if "ddpm_loss_weight" not in training_cfg:
        training_cfg["ddpm_loss_weight"] = 1.0
    if "cycle_loss_weight" not in training_cfg:
        training_cfg["cycle_loss_weight"] = 0.1
    # Ensure perceptual loss is disabled (pure self-supervised approach)
    if "perceptual_loss_weight" not in training_cfg:
        training_cfg["perceptual_loss_weight"] = 0.0

    # Instantiate trainer (ProgressiveTrainer if enabled in cfg.model.multi_resolution.progressive)
    use_progressive = False
    progressive_section = {}
    try:
        mr_cfg = getattr(cfg.model, 'multi_resolution', {})
        prog_cfg = getattr(mr_cfg, 'progressive', {})
        use_progressive = bool(getattr(prog_cfg, 'enabled', False))
        if use_progressive:
            # Map YAML fields to trainer's progressive config
            progressive_section = {
                'enabled': True,
                'max_resolution': int(getattr(prog_cfg, 'max_resolution', getattr(mr_cfg, 'target_resolution', getattr(cfg.model, 'sample_size', 128)))) ,
                'start_resolution': int(getattr(mr_cfg, 'base_resolution', 64)),
                'curriculum_type': str(getattr(prog_cfg, 'curriculum_type', 'linear')),
                # epochs_per_resolution can be derived from steps_per_resolution if provided in YAML; fallback to small ints
                # If steps_per_resolution provided and steps_per_epoch configured, map steps->epochs
                'epochs_per_resolution': (lambda _prog=prog_cfg: (
                    list(getattr(_prog, 'epochs_per_resolution', []))
                    if list(getattr(_prog, 'epochs_per_resolution', []))
                    else (
                        (lambda steps, spe: [max(1, int(round(s / max(1, spe)))) for s in steps])(
                            list(getattr(_prog, 'steps_per_resolution', [])),
                            int(getattr(cfg.training, 'steps_per_epoch', 0))
                        ) if list(getattr(_prog, 'steps_per_resolution', [])) else [10, 15, 20, 25]
                    )
                ))(),
                'smooth_transitions': bool(getattr(prog_cfg, 'smooth_transitions', True)),
                # YAML uses transition_steps; trainer expects transition_epochs. Map using steps_per_epoch when available
                'transition_epochs': (lambda te_steps, spe: (
                    int(max(1, int(round(te_steps / max(1, spe))))) if te_steps is not None and spe > 0 else int(getattr(prog_cfg, 'transition_epochs', 2))
                ))(getattr(prog_cfg, 'transition_steps', None), int(getattr(cfg.training, 'steps_per_epoch', 0))),
                'blend_mode': str(getattr(prog_cfg, 'blend_mode', 'alpha')),
                'lr_scaling': bool(getattr(prog_cfg, 'lr_scaling', True)),
                'lr_curriculum': str(getattr(prog_cfg, 'lr_curriculum', 'sqrt')),
                'batch_scaling': bool(getattr(prog_cfg, 'batch_scaling', True)),
                'adaptive_batch_scaling': bool(getattr(prog_cfg, 'adaptive_batch_scaling', True)),
                'cross_resolution_consistency': bool(getattr(prog_cfg, 'cross_resolution_consistency', True)),
                'consistency_weight': float(getattr(prog_cfg, 'consistency_weight', 0.1)),
                # Pass through explicit resolution schedule if YAML provides one
                'resolution_schedule': list(getattr(mr_cfg, 'resolutions', [])) or None,
            }
            # Also allow resolutions list from YAML to seed schedule in ProgressiveUNet via UNet.sample_size and trainer logic
            # Note: ProgressiveTrainer computes schedule from max_resolution; we log YAML for info
            print(f"✅ Progressive training enabled. YAML schedule: {list(getattr(mr_cfg, 'resolutions', []))}")
    except Exception as e:
        use_progressive = False
        progressive_section = {}
        print(f"⚠️ Progressive config parsing failed: {e}")

    # Merge progressive section into training config for trainer consumption
    if use_progressive:
        training_cfg = {**training_cfg, 'progressive': progressive_section}
        try:
            from pkl_dg.models.progressive import ProgressiveTrainer
            ddpm_trainer = ProgressiveTrainer(
                model=unet,
                config=training_cfg,
                forward_model=forward_model,
                transform=transform
            ).to(device)
            print("✅ Using ProgressiveTrainer")
        except Exception as e:
            print(f"⚠️ Failed to initialize ProgressiveTrainer, falling back to DDPMTrainer: {e}")
            ddpm_trainer = DDPMTrainer(
                model=unet,
                config=training_cfg,
                forward_model=forward_model,
                transform=transform
            ).to(device)
    else:
        ddpm_trainer = DDPMTrainer(
            model=unet,
            config=training_cfg,
            forward_model=forward_model,
            transform=transform
        ).to(device)

    # Optionally wrap dataloaders with ProgressiveDataLoader if enabled in data cfg
    try:
        data_loading_cfg = getattr(cfg.data, 'data_loading', {})
        mr_dl_cfg = getattr(data_loading_cfg, 'multi_resolution', {})
        use_progressive_data = bool(getattr(mr_dl_cfg, 'enabled', False))
    except Exception:
        use_progressive_data = False

    if use_progressive_data:
        try:
            from pkl_dg.models.progressive import ProgressiveDataLoader
            # Build a resolution schedule from YAML if available
            try:
                mr_cfg = getattr(cfg.model, 'multi_resolution', {})
                yaml_schedule = list(getattr(mr_cfg, 'resolutions', []))
            except Exception:
                yaml_schedule = []

            # Prefer schedule passed to trainer via training_cfg['progressive'] if present
            schedule_from_progressive = list(
                training_cfg.get('progressive', {}).get('resolution_schedule', [])
            ) if isinstance(training_cfg.get('progressive', {}), dict) else []

            resolution_schedule = schedule_from_progressive or yaml_schedule
            # Fallback to [cfg.model.sample_size] if nothing provided
            if not resolution_schedule:
                try:
                    # Start at base_resolution if available
                    base_res = int(getattr(getattr(cfg.model, 'multi_resolution', {}), 'base_resolution', 64))
                    target_res = int(getattr(cfg.model, 'sample_size', 128))
                    # Build doubling schedule up to target
                    s = []
                    cur = max(8, base_res)
                    while cur < target_res:
                        s.append(cur)
                        cur *= 2
                    s.append(target_res)
                    resolution_schedule = s
                except Exception:
                    resolution_schedule = [int(getattr(cfg.model, 'sample_size', 128))]

            # Wrap loaders
            train_loader = ProgressiveDataLoader(train_loader, resolution_schedule)
            val_loader = ProgressiveDataLoader(val_loader, resolution_schedule)
            print(f"✅ Using ProgressiveDataLoader with schedule: {resolution_schedule}")
        except Exception as e:
            print(f"⚠️ Failed to enable ProgressiveDataLoader: {e}")

    # Setup optimizer and training parameters
    # Optimizer with optional fused AdamW
    _use_fused = bool(getattr(cfg.training, "use_fused_adam", False)) and torch.cuda.is_available()
    try:
        if _use_fused:
            optimizer = torch.optim.AdamW(
                ddpm_trainer.parameters(),
                lr=float(cfg.training.learning_rate),
                betas=(0.9, 0.999),
                weight_decay=float(getattr(cfg.training, "weight_decay", 1e-6)),
                fused=True,
            )
            print("✅ Using fused AdamW optimizer")
        else:
            optimizer = torch.optim.AdamW(
                ddpm_trainer.parameters(),
                lr=float(cfg.training.learning_rate),
                betas=(0.9, 0.999),
                weight_decay=float(getattr(cfg.training, "weight_decay", 1e-6))
            )
    except TypeError:
        # Older PyTorch without fused kwarg support
        optimizer = torch.optim.AdamW(
            ddpm_trainer.parameters(),
            lr=float(cfg.training.learning_rate),
            betas=(0.9, 0.999),
            weight_decay=float(getattr(cfg.training, "weight_decay", 1e-6))
        )
        if _use_fused:
            print("⚠️ 'fused' AdamW not supported in this PyTorch version; falling back to standard AdamW")

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
                    with torch.inference_mode():
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
        
        # Model checkpoint callback(s) - honor logging.* and training.checkpoint_config
        logging_cfg = getattr(cfg, 'logging', {})
        ckpt_cfg = getattr(cfg.training, 'checkpoint_config', {})
        main_ckpt_cfg = getattr(ckpt_cfg, 'main_checkpoint', {})
        inter_ckpt_cfg = getattr(ckpt_cfg, 'intermediate_checkpoint', {})
        steps_list = getattr(cfg.training, 'checkpoint_every_n_steps', None)
        save_intermediate = bool(getattr(cfg.training, 'save_intermediate_checkpoints', True))

        # Resolve monitor metric with normalization (e.g., val_loss -> val/loss)
        monitor_metric = str(
            getattr(main_ckpt_cfg, 'monitor', None)
            or getattr(logging_cfg, 'monitor_metric', 'val/loss')
        )
        if '/' not in monitor_metric and 'loss' in monitor_metric:
            monitor_metric = monitor_metric.replace('_', '/')

        # Resolve main checkpoint params
        main_every_n = getattr(main_ckpt_cfg, 'every_n_train_steps', None)
        if main_every_n is None:
            if isinstance(steps_list, (list, tuple)) and len(steps_list) > 0:
                try:
                    main_every_n = int(max(steps_list))
                except Exception:
                    main_every_n = 10000
            else:
                main_every_n = 10000
        main_save_top_k = int(
            getattr(main_ckpt_cfg, 'save_top_k', None)
            if getattr(main_ckpt_cfg, 'save_top_k', None) is not None
            else getattr(logging_cfg, 'save_top_k', 3)
        )
        main_mode = str(getattr(main_ckpt_cfg, 'mode', 'min'))
        main_filename = str(
            getattr(main_ckpt_cfg, 'filename', f"ddpm-{{step:06d}}-{{{monitor_metric}:.4f}}")
        )
        main_save_last = bool(getattr(main_ckpt_cfg, 'save_last', True))

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=main_filename,
            monitor=monitor_metric,
            mode=main_mode,
            save_top_k=main_save_top_k,
            save_last=main_save_last,
            every_n_train_steps=int(main_every_n),
            verbose=True
        )
        callbacks.append(checkpoint_callback)

        # Intermediate checkpoint (optional)
        if save_intermediate:
            inter_every_n = getattr(inter_ckpt_cfg, 'every_n_train_steps', None)
            if inter_every_n is None:
                if isinstance(steps_list, (list, tuple)) and len(steps_list) > 0:
                    try:
                        inter_every_n = int(min(steps_list))
                    except Exception:
                        inter_every_n = 5000
                else:
                    inter_every_n = 5000
            inter_save_top_k = int(getattr(inter_ckpt_cfg, 'save_top_k', -1))
            inter_save_last = bool(getattr(inter_ckpt_cfg, 'save_last', False))
            inter_filename = str(getattr(inter_ckpt_cfg, 'filename', 'ddpm-intermediate-{step:06d}'))

            intermediate_checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir / "intermediate",
                filename=inter_filename,
                every_n_train_steps=int(inter_every_n),
                save_top_k=inter_save_top_k,
                save_last=inter_save_last,
                verbose=False
            )
            callbacks.append(intermediate_checkpoint_callback)
        
        # Early stopping callback - step-based (monitor same metric as checkpoints)
        early_stop_callback = EarlyStopping(
            monitor=monitor_metric,
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

        # Prefer logging.log_every_n_steps if provided
        _log_every = int(
            getattr(getattr(cfg, 'logging', {}), 'log_every_n_steps',
                    int(cfg.training.get('log_every_n_steps', 100)))
        )

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
            log_every_n_steps=_log_every,
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
            # Update progressive data loader phase if enabled
            try:
                from pkl_dg.models.progressive import ProgressiveDataLoader as _PDL
                if isinstance(train_loader, _PDL):
                    # Ensure dataloader matches trainer's current phase
                    train_loader.set_phase(getattr(ddpm_trainer, 'current_phase', 0))
                if isinstance(val_loader, _PDL):
                    val_loader.set_phase(getattr(ddpm_trainer, 'current_phase', 0))
            except Exception:
                pass
            epoch_loss = 0.0
            num_batches = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)
            
            for batch_idx, batch in enumerate(progress):
                # Move batch to device
                if isinstance(batch, (tuple, list)):
                    batch = [b.to(device, non_blocking=True) for b in batch]
                else:
                    batch = batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

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
            
            with torch.inference_mode():
                amp_cm = (autocast(dtype=amp_dtype) if use_amp else nullcontext())
                val_progress = tqdm(val_loader, desc=f"Validation {epoch+1}/{max_epochs}", leave=False)
                for val_batch_idx, val_batch in enumerate(val_progress):
                    # Move batch to device
                    if isinstance(val_batch, (tuple, list)):
                        val_batch = [b.to(device, non_blocking=True) for b in val_batch]
                    else:
                        val_batch = val_batch.to(device, non_blocking=True)
                    
                    # Validation step
                    with amp_cm:
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
    
    # Create guidance - use same PSF logic as training via centralized helper
    physics_config = getattr(cfg, "physics", {})
    _psf_obj, target_pixel_size_xy_nm = psf_from_config(physics_config)
    forward_model = ForwardModel(
        psf=_psf_obj.to_torch(device=device),
        background=float(getattr(physics_config, "background", 0.0)),
        device=device,
        read_noise_sigma=float(getattr(physics_config, "read_noise_sigma", 0.0)),
        target_pixel_size_xy_nm=target_pixel_size_xy_nm
    )
    
    guidance_cfg = getattr(cfg, "guidance", {})
    # Build guidance strategy from type; constructors expect only their own params
    gtype = str(guidance_type).lower()
    if gtype == "pkl":
        epsilon = float(guidance_cfg.get("epsilon", 1e-6))
        guidance = PKLGuidance(epsilon=epsilon)
    elif gtype == "kl":
        sigma2 = float(guidance_cfg.get("sigma2", 1.0))
        guidance = KLGuidance(sigma2=sigma2)
    elif gtype == "anscombe":
        epsilon = float(guidance_cfg.get("epsilon", 1e-6))
        guidance = AnscombeGuidance(epsilon=epsilon)
    else:  # l2
        guidance = L2Guidance()

    # Build schedule (adaptive by default)
    schedule_type = str(guidance_cfg.get("schedule_type", "adaptive")).lower()
    if schedule_type == "adaptive":
        lambda_base = float(guidance_cfg.get("lambda_base", 0.1))
        T_threshold = int(guidance_cfg.get("schedule", {}).get("T_threshold", 800))
        epsilon_lambda = float(guidance_cfg.get("schedule", {}).get("epsilon_lambda", 1e-3))
        T_total = int(getattr(cfg.training, "num_timesteps", 1000))
        schedule = AdaptiveSchedule(
            lambda_base=lambda_base,
            T_threshold=T_threshold,
            epsilon_lambda=epsilon_lambda,
            T_total=T_total,
        )
    else:
        schedule = None
    
    # Create sampler
    sampler = DDIMSampler(
        model=ddpm_trainer,
        forward_model=forward_model,
        guidance_strategy=guidance,
        schedule=schedule,
        transform=transform,
        num_timesteps=int(getattr(cfg.training, "num_timesteps", 1000)),
        ddim_steps=int(getattr(cfg.inference, "ddim_steps", 50)),
        eta=float(getattr(cfg.inference, "eta", 0.0)),
        use_autocast=bool(getattr(cfg.inference, "use_autocast", True)),
        clip_denoised=True,
        v_parameterization=bool(getattr(cfg.model, "learned_variance", False))
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
    
    #
    
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
    # Respect cfg.guidance.type if specified, otherwise evaluate all
    cfg_type = str(getattr(getattr(cfg, 'guidance', {}), 'type', '')).lower()
    if cfg_type in ['pkl', 'l2', 'anscombe', 'kl']:
        guidance_types = [cfg_type]
    else:
        guidance_types = ['l2', 'anscombe', 'kl', 'pkl']
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
    
    # Load PSF for baselines and robustness if available (centralized helper)
    psf = None
    physics_config = getattr(cfg, "physics", {})
    try:
        _psf_obj, _ = psf_from_config(physics_config)
        psf = _psf_obj.psf
    except Exception:
        psf = None
    
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
                
                # Optional robustness tests (misalignment and PSF broadening) per guidance
                if getattr(args, 'include_robustness_tests', False):
                    # Misalignment: shift measurement by 1 pixel
                    try:
                        wf_shift = torch.roll(wf_tensor, shifts=(1, 1), dims=(-2, -1))
                        pred_shift = sampler.sample(wf_shift, tuple(wf_shift.shape), device=device, verbose=False, conditioner=conditioner)
                        pred_shift_np = pred_shift.squeeze().detach().cpu().numpy().astype(np.float32)
                        mis_metrics = compute_evaluation_metrics(pred_shift_np, gt)
                        accumulate_results(f"diffusion_{guidance_type}_misalign", mis_metrics)
                    except Exception as e:
                        print(f"⚠️ Misalignment test failed for {guidance_type}: {e}")

                    # PSF broadening: lightly blur PSF before guidance
                    try:
                        if psf is not None:
                            import torch.nn.functional as Fnn
                            psf_t = torch.from_numpy(psf if isinstance(psf, np.ndarray) else np.array(psf)).float()
                            if psf_t.ndim == 2:
                                psf_t = psf_t.unsqueeze(0).unsqueeze(0)
                            kernel = torch.ones((1, 1, 5, 5), device=psf_t.device, dtype=psf_t.dtype) / 25.0
                            psf_blur = Fnn.conv2d(psf_t, kernel, padding=2)
                            # Normalize
                            psf_blur = psf_blur / (psf_blur.sum() + 1e-8)
                            from pkl_dg.physics import ForwardModel
                            fm_broad = ForwardModel(
                                psf=psf_blur.squeeze(0).squeeze(0).to(device),
                                background=float(getattr(physics_config, "background", 0.0)),
                                device=device,
                                read_noise_sigma=float(getattr(physics_config, "read_noise_sigma", 0.0))
                            )
                            # Temporarily swap forward model
                            original_fm = getattr(sampler, 'forward_model', None)
                            sampler.forward_model = fm_broad
                            try:
                                pred_broad = sampler.sample(wf_tensor, tuple(wf_tensor.shape), device=device, verbose=False, conditioner=conditioner)
                                pred_broad_np = pred_broad.squeeze().detach().cpu().numpy().astype(np.float32)
                                broad_metrics = compute_evaluation_metrics(pred_broad_np, gt)
                                accumulate_results(f"diffusion_{guidance_type}_psf_broaden", broad_metrics)
                            finally:
                                sampler.forward_model = original_fm
                    except Exception as e:
                        print(f"⚠️ PSF broadening test failed for {guidance_type}: {e}")

                # Optional downstream: Cellpose F1 and morphology (requires masks and cellpose)
                if getattr(args, 'include_cellpose', False) and getattr(args, 'gt_masks_dir', None):
                    try:
                        from cellpose import models as _cp_models
                        masks_dir = Path(args.gt_masks_dir)
                        mask_path = masks_dir / img_path.name
                        if mask_path.exists():
                            gt_masks = tifffile.imread(str(mask_path)) if mask_path.suffix.lower() in ['.tif','.tiff'] else np.array(Image.open(mask_path))
                            cp_model = _cp_models.Cellpose(model_type='cyto')
                            pred_masks_list, _, _, _ = cp_model.eval([pred], diameter=None, channels=[0,0])
                            pred_masks = pred_masks_list[0]
                            # Compute F1 at IoU 0.5 (greedy matching) as fallback if average_precision is unavailable
                            def _f1_iou50(gt_m, pr_m):
                                gt_labels = np.unique(gt_m)[1:]
                                pr_labels = np.unique(pr_m)[1:]
                                if gt_labels.size == 0 and pr_labels.size == 0:
                                    return 1.0
                                if gt_labels.size == 0 or pr_labels.size == 0:
                                    return 0.0
                                # Build IoU matrix
                                iou = np.zeros((gt_labels.size, pr_labels.size), dtype=np.float32)
                                for i, gl in enumerate(gt_labels):
                                    g = (gt_m == gl)
                                    g_sum = g.sum()
                                    if g_sum == 0:
                                        continue
                                    for j, pl in enumerate(pr_labels):
                                        p = (pr_m == pl)
                                        inter = np.logical_and(g, p).sum()
                                        union = g_sum + p.sum() - inter
                                        iou[i, j] = inter / union if union > 0 else 0.0
                                # Greedy match
                                matched_gt = set()
                                matched_pr = set()
                                tp = 0
                                while True:
                                    idx = np.unravel_index(np.argmax(iou), iou.shape)
                                    if iou[idx] < 0.5:
                                        break
                                    if idx[0] in matched_gt or idx[1] in matched_pr:
                                        iou[idx] = -1
                                        continue
                                    matched_gt.add(idx[0])
                                    matched_pr.add(idx[1])
                                    tp += 1
                                    iou[idx] = -1
                                fp = pr_labels.size - tp
                                fn = gt_labels.size - tp
                                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                                return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                            f1 = float(_f1_iou50(gt_masks, pred_masks))
                            # Hausdorff distance
                            try:
                                from pkl_dg.evaluation import DownstreamTasks as _DT
                                hd = _DT.hausdorff_distance(pred_masks, gt_masks)
                            except Exception:
                                hd = float('nan')
                            # Record
                            accumulate_results(f"diffusion_{guidance_type}_cellpose_f1", {"psnr": f1, "ssim": float('nan'), "frc": float('nan')})
                            accumulate_results(f"diffusion_{guidance_type}_hausdorff", {"psnr": hd, "ssim": float('nan'), "frc": float('nan')})
                    except Exception as e:
                        print(f"⚠️ Cellpose/morphology metrics failed: {e}")

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
        
        # Also save CSV summary for convenience
        csv_path = output_dir / "evaluation_results.csv"
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header
                header = ["method", "psnr", "ssim", "frc"]
                writer.writerow(header)
                # Rows
                for method, metrics in final_results.items():
                    writer.writerow([
                        method,
                        metrics.get("psnr", float('nan')),
                        metrics.get("ssim", float('nan')),
                        metrics.get("frc", float('nan')),
                    ])
            print(f"💾 CSV saved to: {csv_path}")
        except Exception as e:
            print(f"⚠️ Failed to save CSV: {e}")
    
    return final_results


def _parse_bool_list(value: Optional[str]) -> Optional[List[bool]]:
    if value is None:
        return None
    tokens = [t.strip().lower() for t in value.split(',') if t.strip() != ""]
    mapping = {"true": True, "t": True, "1": True, "yes": True, "y": True, "false": False, "f": False, "0": False, "no": False, "n": False}
    result: List[bool] = []
    for t in tokens:
        if t not in mapping:
            raise ValueError(f"Invalid boolean token in sweep list: {t}")
        result.append(mapping[t])
    return result


def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    return [int(t.strip()) for t in value.split(',') if t.strip() != ""]


def _parse_float_list(value: Optional[str]) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(t.strip()) for t in value.split(',') if t.strip() != ""]


def _parse_str_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [t.strip().lower() for t in value.split(',') if t.strip() != ""]


def run_ablation(cfg: DictConfig, args) -> List[Dict[str, Union[str, float, int, bool]]]:
    """Run ablation sweeps over requested configuration axes and save aggregated CSV/JSON.

    The following axes are supported via CLI:
      - guidance types
      - PSF source (beads vs gaussian)
      - conditioning on/off
      - adaptive normalization on/off
      - num_timesteps (ints)
      - learned_variance on/off
      - EMA on/off
      - cycle_loss_weight (floats)
    """
    print("🧪 Starting Ablation Sweeps")
    print("=" * 50)

    # Resolve base values from cfg if sweeps not provided
    guidance_list = _parse_str_list(getattr(args, 'sweep_guidance', None))
    if not guidance_list:
        # Evaluate all by default if not specified
        guidance_list = ['l2', 'anscombe', 'kl', 'pkl']

    psf_sources = _parse_str_list(getattr(args, 'sweep_psf_source', None))
    if not psf_sources:
        use_bead = bool(getattr(cfg.physics, 'use_bead_psf', False)) if hasattr(cfg, 'physics') else False
        psf_sources = ['beads' if use_bead else 'gaussian']

    conditioning_list = _parse_bool_list(getattr(args, 'sweep_conditioning', None))
    if conditioning_list is None:
        conditioning_list = [bool(getattr(cfg.training, 'use_conditioning', False))]

    adaptive_norm_list = _parse_bool_list(getattr(args, 'sweep_adaptive_normalization', None))
    if adaptive_norm_list is None:
        adaptive_norm_list = [bool(getattr(cfg.data, 'use_adaptive_normalization', False))]

    num_timesteps_list = _parse_int_list(getattr(args, 'sweep_num_timesteps', None))
    if num_timesteps_list is None:
        num_timesteps_list = [int(getattr(cfg.training, 'num_timesteps', 1000))]

    learned_variance_list = _parse_bool_list(getattr(args, 'sweep_learned_variance', None))
    if learned_variance_list is None:
        learned_variance_list = [bool(getattr(cfg.model, 'learned_variance', False))]

    ema_list = _parse_bool_list(getattr(args, 'sweep_ema', None))
    if ema_list is None:
        ema_list = [bool(getattr(cfg.training, 'use_ema', True))]

    cycle_weight_list = _parse_float_list(getattr(args, 'sweep_cycle_weight', None))
    if cycle_weight_list is None:
        cycle_weight_list = [float(getattr(cfg.training, 'cycle_loss_weight', 0.1))]

    # Prepare output directory and run stamp
    base_output = Path(args.output_dir) if args.output_dir else (Path(getattr(cfg.paths, 'outputs', 'outputs')))
    ablate_dir = base_output / 'ablations'
    ablate_dir.mkdir(parents=True, exist_ok=True)
    config_name = Path(args.config).stem if args.config else 'default'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Cartesian product of all axes
    # Optional guidance parameter sweeps
    lambda_list = _parse_float_list(getattr(args, 'sweep_guidance_lambda', None)) or [float(getattr(getattr(cfg, 'guidance', {}), 'lambda_base', 0.1))]
    Tthr_list = _parse_int_list(getattr(args, 'sweep_guidance_Tthr', None)) or [int(getattr(getattr(getattr(cfg, 'guidance', {}), 'schedule', {}), 'T_threshold', 800))]
    epslambda_list = _parse_float_list(getattr(args, 'sweep_guidance_epslambda', None)) or [float(getattr(getattr(getattr(cfg, 'guidance', {}), 'schedule', {}), 'epsilon_lambda', 1e-3))]

    axes = [
        ('guidance', guidance_list),
        ('psf_source', psf_sources),
        ('conditioning', conditioning_list),
        ('adaptive_norm', adaptive_norm_list),
        ('num_timesteps', num_timesteps_list),
        ('learned_variance', learned_variance_list),
        ('use_ema', ema_list),
        ('cycle_loss_weight', cycle_weight_list),
        ('lambda_base', lambda_list),
        ('T_threshold', Tthr_list),
        ('epsilon_lambda', epslambda_list),
    ]
    key_names = [k for k, _ in axes]
    value_lists = [v for _, v in axes]

    results_rows: List[Dict[str, Union[str, float, int, bool]]] = []

    combos = itertools.product(*value_lists)
    for combo in combos:
        # Build variant configuration
        variant_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        variant = dict(zip(key_names, combo))

        # Apply variant settings
        # Guidance
        if not hasattr(variant_cfg, 'guidance'):
            variant_cfg.guidance = {}
        variant_cfg.guidance.type = str(variant['guidance'])
        variant_cfg.guidance.lambda_base = float(variant['lambda_base'])
        if not hasattr(variant_cfg.guidance, 'schedule'):
            variant_cfg.guidance.schedule = {}
        variant_cfg.guidance.schedule.T_threshold = int(variant['T_threshold'])
        variant_cfg.guidance.schedule.epsilon_lambda = float(variant['epsilon_lambda'])

        # PSF source
        if not hasattr(variant_cfg, 'physics'):
            variant_cfg.physics = {}
        if str(variant['psf_source']) == 'beads':
            variant_cfg.physics.use_bead_psf = True
        else:
            variant_cfg.physics.use_bead_psf = False
            # Ensure PSF type is gaussian when not using beads
            if not hasattr(variant_cfg, 'psf'):
                variant_cfg.psf = {}
            variant_cfg.psf.type = 'gaussian'

        # Conditioning
        if not hasattr(variant_cfg, 'training'):
            variant_cfg.training = {}
        variant_cfg.training.use_conditioning = bool(variant['conditioning'])

        # Adaptive normalization
        if not hasattr(variant_cfg, 'data'):
            variant_cfg.data = {}
        variant_cfg.data.use_adaptive_normalization = bool(variant['adaptive_norm'])

        # Timesteps
        variant_cfg.training.num_timesteps = int(variant['num_timesteps'])

        # Learned variance
        if not hasattr(variant_cfg, 'model'):
            variant_cfg.model = {}
        variant_cfg.model.learned_variance = bool(variant['learned_variance'])

        # EMA
        variant_cfg.training.use_ema = bool(variant['use_ema'])

        # Cycle loss weight
        variant_cfg.training.cycle_loss_weight = float(variant['cycle_loss_weight'])

        # Prepare per-variant output directory with stamped, config-named run directory
        run_name = (
            f"{config_name}__guid_{variant['guidance']}__psf_{variant['psf_source']}"
            f"__cond_{int(bool(variant['conditioning']))}__adapt_{int(bool(variant['adaptive_norm']))}"
            f"__T_{variant['num_timesteps']}__lv_{int(bool(variant['learned_variance']))}"
            f"__ema_{int(bool(variant['use_ema']))}__cycle_{variant['cycle_loss_weight']}"
            f"__lam_{variant['lambda_base']}__Tthr_{variant['T_threshold']}__epsl_{variant['epsilon_lambda']}__{stamp}"
        )
        out_dir_variant = ablate_dir / run_name
        out_dir_variant.mkdir(parents=True, exist_ok=True)

        # Run evaluation for this variant
        prev_output_dir = args.output_dir
        prev_guidance_type = getattr(args, 'guidance_type', None)
        try:
            args.output_dir = str(out_dir_variant)
            # Ensure we do not override guidance via CLI; use cfg's guidance.type
            args.guidance_type = None
            eval_results = run_evaluation(variant_cfg, args)
        finally:
            args.output_dir = prev_output_dir
            args.guidance_type = prev_guidance_type

        # Flatten results into rows (one row per method)
        for method, metrics in eval_results.items():
            row: Dict[str, Union[str, float, int, bool]] = {
                'run_name': run_name,
                'method': method,
                'guidance': str(variant['guidance']),
                'psf_source': str(variant['psf_source']),
                'conditioning': bool(variant['conditioning']),
                'adaptive_norm': bool(variant['adaptive_norm']),
                'num_timesteps': int(variant['num_timesteps']),
                'learned_variance': bool(variant['learned_variance']),
                'use_ema': bool(variant['use_ema']),
                'cycle_loss_weight': float(variant['cycle_loss_weight']),
                'psnr': float(metrics.get('psnr', float('nan'))),
                'ssim': float(metrics.get('ssim', float('nan'))),
                'frc': float(metrics.get('frc', float('nan'))),
                'lambda_base': float(variant['lambda_base']),
                'T_threshold': int(variant['T_threshold']),
                'epsilon_lambda': float(variant['epsilon_lambda']),
            }
            results_rows.append(row)

    # Save aggregated ablation CSV and JSON
    agg_csv = ablate_dir / f"ablations_{config_name}_{stamp}.csv"
    try:
        with open(agg_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"💾 Ablation CSV saved to: {agg_csv}")
    except Exception as e:
        print(f"⚠️ Failed to save ablation CSV: {e}")

    agg_json = ablate_dir / f"ablations_{config_name}_{stamp}.json"
    try:
        with open(agg_json, 'w') as f:
            json.dump(results_rows, f, indent=2)
        print(f"💾 Ablation JSON saved to: {agg_json}")
    except Exception as e:
        print(f"⚠️ Failed to save ablation JSON: {e}")

    return results_rows

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
        choices=['train', 'eval', 'train_eval', 'ablate'],
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
    parser.add_argument('--guidance-type', type=str, choices=['pkl','kl','l2','anscombe'], help='Override guidance.type for inference/eval')
    
    # Ablation sweep arguments (comma-separated lists for values)
    parser.add_argument('--sweep-guidance', type=str, help='Comma-separated guidance types to sweep (e.g., pkl,kl,l2,anscombe)')
    parser.add_argument('--sweep-psf-source', type=str, help="Comma-separated PSF sources to sweep: 'beads' or 'gaussian'")
    parser.add_argument('--sweep-conditioning', type=str, help='Comma-separated booleans for conditioning (e.g., true,false)')
    parser.add_argument('--sweep-adaptive-normalization', type=str, help='Comma-separated booleans for adaptive normalization (e.g., true,false)')
    parser.add_argument('--sweep-num-timesteps', type=str, help='Comma-separated timesteps (e.g., 250,500,1000)')
    parser.add_argument('--sweep-learned-variance', type=str, help='Comma-separated booleans for learned variance (e.g., true,false)')
    parser.add_argument('--sweep-ema', type=str, help='Comma-separated booleans for EMA (e.g., true,false)')
    parser.add_argument('--sweep-cycle-weight', type=str, help='Comma-separated cycle loss weights (e.g., 0.05,0.1,0.2)')
    # Guidance parameter sweeps
    parser.add_argument('--sweep-guidance-lambda', type=str, help='Comma-separated lambda_base values for guidance (e.g., 0.05,0.1,0.2)')
    parser.add_argument('--sweep-guidance-Tthr', type=str, help='Comma-separated T_threshold values (e.g., 600,800,900)')
    parser.add_argument('--sweep-guidance-epslambda', type=str, help='Comma-separated epsilon_lambda values (e.g., 1e-4,1e-3,1e-2)')
    # Optional robustness and downstream flags
    parser.add_argument('--include-robustness-tests', action='store_true', help='Run misalignment and PSF-broadening tests during evaluation')
    parser.add_argument('--include-cellpose', action='store_true', help='Run Cellpose/morphology metrics if masks available')
    parser.add_argument('--gt-masks-dir', type=str, help='Directory with ground-truth masks for Cellpose/morphology metrics')
    
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
    
    if args.mode == 'ablate':
        if not args.config:
            parser.error("--config is required for ablate mode")
        if not (args.input_dir and args.gt_dir):
            parser.error("--input-dir and --gt-dir are required for ablate mode")
    
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
            
            # Optional guidance-type override
            if args.guidance_type:
                if not hasattr(cfg, 'guidance'):
                    cfg.guidance = {}
                cfg.guidance.type = args.guidance_type
            
            results = run_evaluation(cfg, args)
            print("🎉 Evaluation completed successfully!")
        
        # Run ablations
        if args.mode == 'ablate':
            # Always load cfg from provided config for ablations
            cfg = setup_experiment(args)
            run_ablation(cfg, args)
            print("🎉 Ablations completed successfully!")
        
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
