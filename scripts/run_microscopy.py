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
    
    # Full pipeline with evaluation
    python scripts/run_microscopy.py --mode train_eval --config configs/config_real.yaml --eval-input data/test/wf --eval-gt data/test/2p
    
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core imports
from pkl_dg.models.unet import UNet
from pkl_dg.models.diffusion import DDPMTrainer, UnpairedDataset
from pkl_dg.models.sampler import DDIMSampler
from pkl_dg.data import RealPairsDataset, IntensityToModel, AnscombeToModel, GeneralizedAnscombeToModel
from pkl_dg.data import ZarrPatchesDataset
from pkl_dg.physics.psf import PSF, build_psf_bank
from pkl_dg.physics.forward_model import ForwardModel
from pkl_dg.guidance.pkl_optical_restoration import PKLGuidance
from pkl_dg.guidance.l2 import L2Guidance
from pkl_dg.guidance.anscombe import AnscombeGuidance
from pkl_dg.guidance.schedules import AdaptiveSchedule
from pkl_dg.evaluation.metrics import Metrics
# from pkl_dg.evaluation.robustness import RobustnessTests
# from pkl_dg.evaluation.hallucination import HallucinationTests
# from pkl_dg.evaluation.tasks import DownstreamTasks
from pkl_dg.baselines import richardson_lucy_restore
from pkl_dg.utils.memory import cleanup_memory
from pkl_dg.utils import (
    load_config, 
    print_config_summary,
    validate_and_complete_config,
    setup_logging
)

# Optional imports
try:
    from pkl_dg.baselines import RCANWrapper
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
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate
    if args.device is not None:
        cfg.experiment.device = args.device
    if args.seed is not None:
        cfg.experiment.seed = args.seed
        
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
    
    # Initialize W&B if enabled
    if cfg.wandb.mode != "disabled":
        # Create run name with config name and date
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = Path(args.config).stem if args.config else "default"
        run_name = f"{config_name}_{cfg.experiment.name}_{timestamp}"
        
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            mode=cfg.wandb.mode,
            tags=[config_name, cfg.experiment.name, f"seed_{seed}"]
        )
        print(f"✅ Initialized Weights & Biases logging: {run_name}")

    # Setup device and paths
    device = str(cfg.experiment.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_dir = Path(cfg.paths.data)
    checkpoint_dir = Path(cfg.paths.checkpoints)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup timestamp and config name for consistent naming
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = Path(args.config).stem if args.config else "default"
    
    # Setup file logging
    logs_dir = Path(cfg.paths.logs)
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
    
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Training logs: {log_file}")

    # Create transform based on noise model
    noise_model = str(getattr(cfg.data, "noise_model", "gaussian")).lower()
    if noise_model == "poisson":
        transform = AnscombeToModel(max_intensity=float(cfg.data.max_intensity))
        print("✅ Using Anscombe transform for Poisson noise")
    elif noise_model == "poisson_gaussian":
        gat_cfg = getattr(cfg.data, "gat", {})
        transform = GeneralizedAnscombeToModel(
            max_intensity=float(cfg.data.max_intensity),
            alpha=float(getattr(gat_cfg, "alpha", 1.0)),
            mu=float(getattr(gat_cfg, "mu", 0.0)),
            sigma=float(getattr(gat_cfg, "sigma", 0.0)),
        )
        print("✅ Using Generalized Anscombe transform for Poisson-Gaussian noise")
    else:
        transform = IntensityToModel(
            min_intensity=float(cfg.data.min_intensity),
            max_intensity=float(cfg.data.max_intensity),
        )
        print("✅ Using intensity normalization for Gaussian noise")

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
            from pkl_dg.physics.psf import psf_params_from_tensor
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
    
    if use_zarr:
        # Use Zarr format for large datasets
        zarr_train = data_dir / "zarr" / "train.zarr"
        zarr_val = data_dir / "zarr" / "val.zarr"
        train_dataset = ZarrPatchesDataset(str(zarr_train), transform=transform)
        val_dataset = ZarrPatchesDataset(str(zarr_val), transform=transform)
        print("✅ Using Zarr datasets")
    elif use_self_supervised:
        # Use true self-supervised learning with forward model
        # Check for paired directory structure first
        wf_dir = data_dir / "wf" 
        twop_dir = data_dir / "2p"
        
        if not wf_dir.exists() or not twop_dir.exists():
            raise ValueError(f"For self-supervised learning, both 'wf' and '2p' directories must exist in {data_dir}")
        
        # Check if we should use paired data for comparison
        use_paired = bool(getattr(cfg.data, "use_paired_for_validation", False))
        use_forward_model = not use_paired  # Use forward model unless explicitly using paired data
        
        train_dataset = UnpairedDataset(
            wf_dir=str(wf_dir),
            twop_dir=str(twop_dir),
            transform=transform,
            image_size=int(cfg.data.image_size),
            mode="train",
            forward_model=forward_model,
            use_forward_model=use_forward_model,
            add_noise=True,
            noise_level=0.05
        )
        
        val_dataset = UnpairedDataset(
            wf_dir=str(wf_dir),
            twop_dir=str(twop_dir),
            transform=transform,
            image_size=int(cfg.data.image_size),
            mode="val",
            forward_model=forward_model,
            use_forward_model=use_forward_model,
            add_noise=False,  # No noise during validation
            noise_level=0.0
        )
        
        if use_paired:
            print("✅ Using paired WF/2P datasets for training")
        else:
            print("✅ Using true self-supervised learning (forward model generates synthetic WF)")
    else:
        # Check for unpaired directory structure (legacy)
        wf_dir = data_dir / "wf"
        twop_dir = data_dir / "2p"
        
        if wf_dir.exists() and twop_dir.exists():
            # Check if we should use forward model for true self-supervised learning
            use_forward_model = bool(getattr(cfg.data, "use_forward_model", True))
            
            if use_forward_model:
                print("✅ Using self-supervised learning with forward model")
            else:
                print("✅ Using unpaired self-supervised learning (legacy)")
                
            train_dataset = UnpairedDataset(
                wf_dir=str(wf_dir),
                twop_dir=str(twop_dir),
                transform=transform,
                image_size=int(cfg.data.image_size),
                mode="train",
                forward_model=forward_model,
                use_forward_model=use_forward_model,
                add_noise=True,
                noise_level=float(getattr(cfg.data, "noise_level", 0.05))
            )
            val_dataset = UnpairedDataset(
                wf_dir=str(wf_dir),
                twop_dir=str(twop_dir),
                transform=transform,
                image_size=int(cfg.data.image_size),
                mode="val",
                forward_model=forward_model,
                use_forward_model=use_forward_model,
                add_noise=False,  # No additional noise for validation
                noise_level=0.0
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

    # Create data loaders
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
    if "perceptual_loss_weight" not in training_cfg:
        training_cfg["perceptual_loss_weight"] = 0.01

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

    scaler = GradScaler() if cfg.training.get("mixed_precision", False) and device == "cuda" else None
    
    max_epochs = int(cfg.training.max_epochs)
    save_every = int(getattr(cfg.training, "save_every_n_epochs", 10))
    grad_clip_val = float(getattr(cfg.training, "gradient_clip_val", 1.0))
    
    print(f"🚀 Starting training for {max_epochs} epochs")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Learning rate: {cfg.training.learning_rate}")

    # Training loop
    best_loss = float('inf')
    
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

            # Forward pass with mixed precision
            if scaler is not None:
                with autocast():
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

        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{max_epochs} - Average Loss: {avg_loss:.6f}")
        logger.info(f"Epoch {epoch+1}/{max_epochs} - Average Loss: {avg_loss:.6f}")

        # Update EMA if enabled
        if hasattr(ddpm_trainer, 'ema_model') and ddpm_trainer.ema_model is not None:
            ddpm_trainer.update_ema()

        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            logger.info(f"New best model found with loss: {best_loss:.6f}")

        if (epoch + 1) % save_every == 0 or is_best or epoch == max_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': ddpm_trainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': avg_loss,
                'best_loss': best_loss,
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

        # Enhanced W&B logging
        if cfg.wandb.mode != "disabled":
            log_dict = {
                "epoch": epoch,
                "train_loss": avg_loss,
                "best_loss": best_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "batch_size": cfg.training.batch_size,
                "device": device,
            }
            
            # Log gradient norms and parameter statistics
            total_grad_norm = 0
            total_param_norm = 0
            for name, param in ddpm_trainer.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2)
                    total_grad_norm += param_grad_norm.item() ** 2
                    log_dict[f"grad_norm/{name}"] = param_grad_norm.item()
                
                param_norm = param.data.norm(2)
                total_param_norm += param_norm.item() ** 2
                log_dict[f"param_norm/{name}"] = param_norm.item()
            
            log_dict["total_grad_norm"] = total_grad_norm ** 0.5
            log_dict["total_param_norm"] = total_param_norm ** 0.5
            
            # Log parameter histograms to W&B every 10 epochs
            if (epoch + 1) % 10 == 0:
                for name, param in ddpm_trainer.named_parameters():
                    if param.grad is not None:
                        log_dict[f"gradients/{name}"] = wandb.Histogram(param.grad.detach().cpu().numpy())
                    log_dict[f"parameters/{name}"] = wandb.Histogram(param.detach().cpu().numpy())
            
            # Log memory usage if CUDA is available
            if device == "cuda" and torch.cuda.is_available():
                log_dict["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
                log_dict["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3  # GB
            
            wandb.log(log_dict)

        # Memory cleanup
        cleanup_memory()

    print(f"✅ Training completed! Best loss: {best_loss:.6f}")
    print(f"📁 Checkpoints saved to: {checkpoint_dir}")
    logger.info(f"Training completed! Best loss: {best_loss:.6f}")
    
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
    if noise_model == "poisson":
        transform = AnscombeToModel(max_intensity=float(cfg.data.max_intensity))
    elif noise_model == "poisson_gaussian":
        gat_cfg = getattr(cfg.data, "gat", {})
        transform = GeneralizedAnscombeToModel(
            max_intensity=float(cfg.data.max_intensity),
            alpha=float(getattr(gat_cfg, "alpha", 1.0)),
            mu=float(getattr(gat_cfg, "mu", 0.0)),
            sigma=float(getattr(gat_cfg, "sigma", 0.0)),
        )
    else:
        transform = IntensityToModel(
            min_intensity=float(cfg.data.min_intensity),
            max_intensity=float(cfg.data.max_intensity),
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


def compute_evaluation_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics between prediction and ground truth."""
    try:
        # Use the direct Metrics class
        return {
            "psnr": Metrics.psnr(pred, gt),
            "ssim": Metrics.ssim(pred, gt, data_range=1.0),
            "frc": Metrics.frc(pred, gt, threshold=0.143)
        }
    except Exception as e:
        print(f"⚠️ Error computing metrics: {e}")
        return {
            "psnr": float('nan'),
            "ssim": float('nan'), 
            "frc": float('nan')
        }


def compute_baseline_metrics(wf_input: np.ndarray, gt: np.ndarray, psf: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    """Compute baseline method results."""
    results = {}
    
    # Richardson-Lucy baseline
    if psf is not None:
        try:
            rl_result = richardson_lucy_restore(wf_input, psf, iterations=30)
            results["richardson_lucy"] = compute_evaluation_metrics(rl_result, gt)
        except Exception as e:
            print(f"⚠️ Richardson-Lucy failed: {e}")
            results["richardson_lucy"] = {"psnr": float('nan'), "ssim": float('nan'), "frc": float('nan')}
    
    # RCAN baseline (if available)
    if HAS_RCAN:
        try:
            # Note: This would require a trained RCAN model
            # results["rcan"] = compute_metrics(rcan_result, gt)
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
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, help='Device (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    # Data paths
    parser.add_argument('--data-dir', type=str, help='Training data directory')
    parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint directory')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    
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
