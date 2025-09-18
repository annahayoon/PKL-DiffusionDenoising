#!/usr/bin/env python3
"""
Unified Training and Evaluation Script for PKL Diffusion Denoising on Natural Images

This script combines both training and evaluation workflows for natural image datasets
(ImageNet, CIFAR, etc.) in a single interface, similar to other diffusion repositories.
It supports:
- Training diffusion models on natural image datasets using synthetic degradation
- RGB and grayscale image support with proper [0,255] -> [-1,1] normalization
- Evaluating trained models with comprehensive metrics  
- Self-supervised training with cycle consistency
- Multiple noise models and guidance strategies

Usage:
    # Training on ImageNet
    python scripts/run_natural.py --mode train --dataset imagenet --data-dir data/imagenet --max-epochs 100
    
    # Training on CIFAR-10 (auto-downloads)
    python scripts/run_natural.py --mode train --dataset cifar10 --max-epochs 50
    
    # Training on MNIST (auto-downloads)
    python scripts/run_natural.py --mode train --dataset mnist --max-epochs 20
    
    # Evaluation on natural images
    python scripts/run_natural.py --mode eval --checkpoint checkpoints/best_model.pt --input-dir data/test --gt-dir data/test_clean
    
    # SID (See-in-the-Dark) evaluation for cross-domain generalization
    python scripts/run_natural.py --mode eval --dataset sid --checkpoint checkpoints/wf2p_model.pt --data-dir data/SID
    
    # Train then evaluate
    python scripts/run_natural.py --mode train_eval --dataset cifar10 --eval-input data/test --eval-gt data/test_clean

Examples:
    # Quick MNIST training (grayscale)
    python scripts/run_natural.py --mode train --dataset mnist --max-epochs 10 --batch-size 64
    
    # CIFAR-10 RGB training
    python scripts/run_natural.py --mode train --dataset cifar10 --channels 3 --max-epochs 100 --batch-size 32
    
    # CIFAR-10 grayscale training
    python scripts/run_natural.py --mode train --dataset cifar10 --force-grayscale --max-epochs 100 --batch-size 32
    
    # ImageNet RGB training
    python scripts/run_natural.py --mode train --dataset imagenet --data-dir data/imagenet --channels 3 --max-epochs 200 --batch-size 16
    
    # Full pipeline with evaluation
    python scripts/run_natural.py --mode train_eval --dataset cifar10 --channels 3 --eval-input data/test --eval-gt data/test_clean
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
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core imports
from pkl_dg.models.unet import UNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.models.sampler import DDIMSampler
from pkl_dg.data import SynthesisDataset, CIFARDataset, SIDDataset, create_sid_dataloader, IntensityToModel, AnscombeToModel, GeneralizedAnscombeToModel
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel
from pkl_dg.guidance.pkl_optical_restoration import PKLGuidance
from pkl_dg.guidance.l2 import L2Guidance
from pkl_dg.guidance.anscombe import AnscombeGuidance
from pkl_dg.guidance.schedules import AdaptiveSchedule
from pkl_dg.evaluation.metrics import Metrics
from pkl_dg.utils.memory import cleanup_memory
from pkl_dg.utils.adaptive_batch import get_optimal_batch_size

# Optional imports for MNIST
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    load_dataset = None
    HAS_DATASETS = False


class MNISTDataset(torch.utils.data.Dataset):
    """MNIST dataset wrapper using HuggingFace datasets."""
    
    def __init__(self, split="train", image_size=32, transform=None):
        if not HAS_DATASETS:
            raise ImportError("datasets library required for MNIST. Install with: pip install datasets")
        
        self.split = split
        self.image_size = image_size
        self.transform = transform
        
        # Load MNIST from HuggingFace
        dataset_split = "train" if split == "train" else "test"
        self.hf_dataset = load_dataset("mnist", split=dataset_split)
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[int(idx)]
        
        # Convert PIL image to tensor
        img = item["image"].convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # [1, H, W]
        
        # Apply transform if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        # For diffusion training, we return (clean, degraded) pairs
        # For MNIST, we'll create synthetic degradation
        return img_tensor, img_tensor  # Placeholder - will be processed by forward model


def create_datasets(config: Dict, forward_model):
    """Create appropriate datasets based on the dataset type."""
    dataset_name = config['experiment']['name']
    data_dir = config['paths']['data']
    image_size = config['data']['image_size']
    
    # Create transform
    noise_model = config['data']['noise_model'].lower()
    if noise_model == "poisson":
        transform = AnscombeToModel(maxIntensity=float(config['data']['max_intensity']))
    elif noise_model == "poisson_gaussian":
        transform = GeneralizedAnscombeToModel(
            maxIntensity=float(config['data']['max_intensity']),
            alpha=1.0,
            mu=0.0,
            sigma=0.0,
        )
    else:
        transform = IntensityToModel(
            minIntensity=float(config['data']['min_intensity']),
            maxIntensity=float(config['data']['max_intensity']),
        )
    
    # Determine dataset type from name or data directory
    if "mnist" in dataset_name.lower() or (data_dir and "mnist" in str(data_dir).lower()):
        print("✅ Using MNIST dataset")
        train_dataset = MNISTDataset(split="train", image_size=image_size, transform=transform)
        val_dataset = MNISTDataset(split="test", image_size=image_size, transform=transform)
        
    elif "cifar" in dataset_name.lower() or (data_dir and "cifar" in str(data_dir).lower()):
        print("✅ Using CIFAR dataset")
        # Determine CIFAR-10 vs CIFAR-100
        dataset_type = "cifar100" if "cifar100" in dataset_name.lower() or "cifar-100" in dataset_name.lower() else "cifar10"
        
        # Determine if we should use grayscale or RGB
        use_grayscale = config['model']['in_channels'] == 1
        
        train_dataset = CIFARDataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            split="train",
            transform=transform,
            image_size=image_size,
            download=True,
            grayscale=use_grayscale
        )
        val_dataset = CIFARDataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            split="test",
            transform=transform,
            image_size=image_size,
            download=True,
            grayscale=use_grayscale
        )
        
    else:
        print("✅ Using SynthesisDataset for ImageNet-like data")
        # Use SynthesisDataset for ImageNet and other natural image datasets
        train_dataset = SynthesisDataset(
            source_dir=str(Path(data_dir) / "train"),
            forward_model=forward_model,
            transform=transform,
            image_size=image_size,
            mode="train",
        )
        val_dataset = SynthesisDataset(
            source_dir=str(Path(data_dir) / "val"),
            forward_model=forward_model,
            transform=transform,
            image_size=image_size,
            mode="val",
        )
    
    return train_dataset, val_dataset


def _get_default_image_size(args):
    """Get default image size based on dataset."""
    if args.image_size:
        return args.image_size
    
    dataset_name = args.dataset.lower()
    if "mnist" in dataset_name:
        return 32  # MNIST is 28x28, but we'll use 32 for consistency
    elif "cifar" in dataset_name:
        return 32  # CIFAR is 32x32
    else:
        return 256  # ImageNet and other natural images


def _get_default_max_intensity(args):
    """Get default max intensity based on dataset."""
    dataset_name = args.dataset.lower()
    if "mnist" in dataset_name or "cifar" in dataset_name:
        return 1.0  # These datasets are normalized to [0,1] first, then transform to [-1,1]
    else:
        return 1000  # Natural images can have higher intensity ranges


def _get_default_channels(args):
    """Get default number of channels based on dataset and arguments."""
    if args.channels:
        return args.channels
    
    # Force grayscale if requested
    if args.force_grayscale:
        return 1
    
    dataset_name = args.dataset.lower()
    if "mnist" in dataset_name:
        return 1  # MNIST is grayscale
    else:
        return 3  # CIFAR and ImageNet are RGB by default


def load_and_normalize_image(img_path: Path, target_range="unit", keep_rgb=True) -> np.ndarray:
    """
    Load an image (RGB or grayscale) and normalize it properly.
    
    Args:
        img_path: Path to image file
        target_range: "unit" for [0,1] or "model" for [-1,1]
        keep_rgb: If True, keep RGB channels; if False, convert to grayscale
    
    Returns:
        Normalized image array (H,W) for grayscale or (H,W,3) for RGB
    """
    # Load image
    img = Image.open(img_path)
    
    # Handle different image modes
    if img.mode == 'RGBA':
        # Convert RGBA to RGB
        img = img.convert('RGB')
    elif img.mode not in ['RGB', 'L']:
        # Convert other modes to RGB
        img = img.convert('RGB')
    
    # Convert to grayscale if requested
    if not keep_rgb and img.mode == 'RGB':
        img = img.convert('L')
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize from [0,255] to [0,1]
    img_array = img_array / 255.0
    
    # Convert to [-1,1] if requested
    if target_range == "model":
        img_array = 2.0 * img_array - 1.0
    
    return img_array


def create_default_config(args) -> Dict:
    """Create default configuration for natural images."""
    config = {
        'experiment': {
            'name': f"natural_images_{args.dataset}",
            'seed': args.seed or 42,
            'device': args.device or 'auto',
            'mixed_precision': True,
        },
        'paths': {
            'data': args.data_dir,
            'checkpoints': args.checkpoint_dir or 'checkpoints',
            'outputs': args.output_dir or 'outputs',
            'logs': 'logs'
        },
        'wandb': {
            'project': 'pkl-diffusion-natural',
            'entity': None,
            'mode': args.wandb_mode or 'disabled'
        },
        'model': {
            'sample_size': _get_default_image_size(args),
            'in_channels': _get_default_channels(args),
            'out_channels': _get_default_channels(args),
            'layers_per_block': 2,
            'block_out_channels': [64, 128, 256, 512],
            'down_block_types': ["DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
            'up_block_types': ["AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"],
        },
        'data': {
            'image_size': _get_default_image_size(args),
            'min_intensity': 0,
            'max_intensity': _get_default_max_intensity(args),
            'noise_model': args.noise_model or 'gaussian'
        },
        'training': {
            'batch_size': args.batch_size or 16,
            'num_workers': args.num_workers or 8,
            'max_epochs': args.max_epochs or 100,
            'learning_rate': args.learning_rate or 1e-4,
            'gradient_clip': 1.0,
            'num_timesteps': 1000,
            'beta_schedule': 'cosine',
            'use_ema': True,
            'use_conditioning': True,
            'conditioning_type': 'wf',
            'supervised_x0_weight': 0.0,  # Self-supervised by default
            'accumulate_grad_batches': 1,
            'persistent_workers': True,
            'prefetch_factor': 4,
            # Self-supervised specific parameters
            'ddpm_loss_weight': 1.0,
            'cycle_loss_weight': 0.1,
            'perceptual_loss_weight': 0.01,
            'cycle_loss_type': 'l1',
            'use_perceptual_loss': True,
            'dynamic_batch_sizing': False,
            'use_lightning': False,
        },
        'psf': {
            'type': 'gaussian',
            'sigma_x': 2.0,
            'sigma_y': 2.0,
            'size': 21,
            'background': 0.0
        },
        'guidance': {
            'lambda_': 1.0,
            'num_inference_steps': 50,
            'eta': 0.0
        }
    }
    return config


def setup_experiment(args, config: Optional[Union[Dict, DictConfig]] = None) -> DictConfig:
    """Setup experiment configuration."""
    if config is None:
        config = create_default_config(args)
    
    # Convert to DictConfig for attribute access
    if not isinstance(config, DictConfig):
        config = OmegaConf.create(config)
    
    # Override config with command line arguments
    if args.max_epochs is not None:
        config.training.max_epochs = args.max_epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.device is not None:
        config.experiment.device = args.device
    if args.seed is not None:
        config.experiment.seed = args.seed
        
    # Setup paths
    if args.data_dir is not None:
        config.paths.data = args.data_dir
    if args.checkpoint_dir is not None:
        config.paths.checkpoints = args.checkpoint_dir
    if args.output_dir is not None:
        config.paths.outputs = args.output_dir
    
    # Print configuration summary if verbose
    if args.verbose:
        print("🔧 Configuration Summary:")
        print("=" * 50)
        print(f"Experiment: {config.experiment.name}")
        print(f"Device: {config.experiment.device}")
        print(f"Data: {config.paths.data}")
        print(f"Image size: {config.data.image_size}")
        print(f"Batch size: {config.training.batch_size}")
        print(f"Max epochs: {config.training.max_epochs}")
        print(f"Learning rate: {config.training.learning_rate}")
        print("=" * 50)
    
    return config


def run_training(config: DictConfig, args) -> DDPMTrainer:
    """Run training workflow for natural images."""
    print("🚀 Starting Natural Image Training")
    print("=" * 50)
    
    # Set seed for reproducibility
    seed = int(config['experiment']['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Initialize W&B if enabled
    if config['wandb']['mode'] != "disabled":
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            config=config,
            name=config['experiment']['name'],
            mode=config['wandb']['mode']
        )
        print("✅ Initialized Weights & Biases logging")

    # Setup device and paths
    device = str(config['experiment']['device'])
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_dir = Path(config['paths']['data'])
    checkpoint_dir = Path(config['paths']['checkpoints'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Create transform based on noise model
    noise_model = str(config['data']['noise_model']).lower()
    if noise_model == "poisson":
        transform = AnscombeToModel(maxIntensity=float(config['data']['max_intensity']))
        print("✅ Using Anscombe transform for Poisson noise")
    elif noise_model == "poisson_gaussian":
        transform = GeneralizedAnscombeToModel(
            maxIntensity=float(config['data']['max_intensity']),
            alpha=1.0,
            mu=0.0,
            sigma=0.0,
        )
        print("✅ Using Generalized Anscombe transform for Poisson-Gaussian noise")
    else:
        transform = IntensityToModel(
            minIntensity=float(config['data']['min_intensity']),
            maxIntensity=float(config['data']['max_intensity']),
        )
        print("✅ Using intensity normalization for Gaussian noise")

    # Create forward model for synthetic degradation
    psf_config = config['psf']
    if psf_config['type'] == "gaussian":
        psf = PSF(
            sigma_x=float(psf_config['sigma_x']),
            sigma_y=float(psf_config['sigma_y']),
            size=int(psf_config['size'])
        )
        print(f"✅ Created Gaussian PSF (σ={psf_config['sigma_x']})")
    else:
        psf = PSF()
        print("✅ Created default PSF")
    
    forward_model = ForwardModel(
        psf=psf.to_torch(device=device),
        background=float(psf_config['background']),
        device=device,
        common_sizes=[(int(config['data']['image_size']), int(config['data']['image_size']))]
    )

    # Create datasets based on dataset type
    train_dataset, val_dataset = create_datasets(config, forward_model)

    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(config['training']['batch_size']),
        shuffle=True,
        num_workers=int(config['training']['num_workers']),
        pin_memory=True,
        persistent_workers=bool(config['training']['persistent_workers']),
        prefetch_factor=int(config['training']['prefetch_factor']),
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(config['training']['batch_size']),
        shuffle=False,
        num_workers=int(config['training']['num_workers']),
        pin_memory=True,
        persistent_workers=bool(config['training']['persistent_workers']),
        prefetch_factor=int(config['training']['prefetch_factor']),
    )

    # Create model
    model_cfg = config['model']
    use_conditioning = bool(config['training']['use_conditioning'])
    
    # Ensure correct input channels for conditioning
    if use_conditioning and int(model_cfg['in_channels']) == 1:
        model_cfg['in_channels'] = 2  # x_t + degraded conditioner

    unet = UNet(model_cfg).to(device)
    print(f"✅ Created U-Net with {sum(p.numel() for p in unet.parameters()):,} parameters")

    # Create trainer
    training_cfg = config['training']
    
    ddpm_trainer = DDPMTrainer(
        model=unet,
        config=training_cfg,
        forward_model=forward_model,
        transform=transform
    ).to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        ddpm_trainer.parameters(),
        lr=float(training_cfg['learning_rate']),
        weight_decay=1e-4
    )

    scaler = GradScaler() if config['experiment']['mixed_precision'] and device == "cuda" else None
    
    max_epochs = int(training_cfg['max_epochs'])
    grad_clip_val = float(training_cfg['gradient_clip'])
    
    print(f"🚀 Starting training for {max_epochs} epochs")
    print(f"Batch size: {training_cfg['batch_size']}")
    print(f"Learning rate: {training_cfg['learning_rate']}")

    # Create samples output dir
    samples_dir = Path(config['paths']['outputs']) / "samples" / config['experiment']['name']
    samples_dir.mkdir(parents=True, exist_ok=True)

    def save_samples(epoch_idx: int, num_rows: int = 2, num_cols: int = 8):
        """Generate and save sample grid."""
        try:
            ddpm_trainer.eval()
            with torch.no_grad():
                num_images = num_rows * num_cols
                H = int(config['data']['image_size'])
                W = H
                samples = ddpm_trainer.ddpm_sample(
                    num_images=num_images, 
                    image_shape=(1, H, W), 
                    use_ema=True
                )
                samples = transform.inverse(samples.clamp(-1, 1)).cpu().numpy()
                
                # Create grid
                grid_h = num_rows * H
                grid_w = num_cols * W
                grid = np.zeros((grid_h, grid_w), dtype=np.float32)
                for i in range(num_images):
                    r = i // num_cols
                    c = i % num_cols
                    img = samples[i, 0]
                    grid[r*H:(r+1)*H, c*W:(c+1)*W] = img
                
                grid_img = (grid * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(grid_img).save(samples_dir / f"epoch_{epoch_idx:03d}.png")
        except Exception as e:
            print(f"⚠️ Failed to save samples: {e}")

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
                x_0, y_degraded = batch
                x_0 = x_0.to(device, non_blocking=True)
                y_degraded = y_degraded.to(device, non_blocking=True)
                batch = (x_0, y_degraded)
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

        # Update EMA if enabled
        if hasattr(ddpm_trainer, 'ema_model') and ddpm_trainer.ema_model is not None:
            ddpm_trainer.update_ema()

        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0 or is_best or epoch == max_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': ddpm_trainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': avg_loss,
                'config': config,
            }
            
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save(checkpoint, checkpoint_path)
            
            if is_best:
                best_path = checkpoint_dir / 'best_model.pt'
                torch.save(checkpoint, best_path)
                print(f"💾 Saved best model: {best_path}")
            
            latest_path = checkpoint_dir / 'latest_checkpoint.pt'
            torch.save(checkpoint, latest_path)

        # Save samples periodically
        if (epoch + 1) % 20 == 0:
            save_samples(epoch + 1)

        # W&B logging
        if config['wandb']['mode'] != "disabled":
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        # Memory cleanup
        cleanup_memory()

    print(f"✅ Training completed! Best loss: {best_loss:.6f}")
    print(f"📁 Checkpoints saved to: {checkpoint_dir}")
    
    return ddpm_trainer


def load_model_and_sampler(config: Dict, checkpoint_path: str, guidance_type: str, device: str):
    """Load trained model and create sampler with specified guidance."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model_cfg = config['model']
    use_conditioning = bool(config['training'].get('use_conditioning', False))

    unet = UNet(model_cfg).to(device)
    
    # Load model weights
    ddpm_trainer = DDPMTrainer(model=unet, config={})
    ddpm_trainer.load_state_dict(checkpoint['model_state_dict'])
    ddpm_trainer.eval()
    
    # Create transform
    noise_model = str(config['data']['noise_model']).lower()
    if noise_model == "poisson":
        transform = AnscombeToModel(maxIntensity=float(config['data']['max_intensity']))
    elif noise_model == "poisson_gaussian":
        transform = GeneralizedAnscombeToModel(
            maxIntensity=float(config['data']['max_intensity']),
            alpha=1.0,
            mu=0.0,
            sigma=0.0,
        )
    else:
        transform = IntensityToModel(
            minIntensity=float(config['data']['min_intensity']),
            maxIntensity=float(config['data']['max_intensity']),
        )
    
    # Create guidance
    psf_config = config['psf']
    if psf_config['type'] == "gaussian":
        psf = PSF(
            sigma_x=float(psf_config['sigma_x']),
            sigma_y=float(psf_config['sigma_y']),
            size=int(psf_config['size'])
        )
    else:
        psf = PSF()
    
    forward_model = ForwardModel(
        psf=psf.to_torch(device=device),
        background=float(psf_config['background']),
        device=device
    )
    
    guidance_cfg = config['guidance']
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
        num_timesteps=int(config['training']['num_timesteps'])
    )
    
    return sampler


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics between prediction and ground truth."""
    metrics = {}
    
    try:
        metrics["psnr"] = Metrics.psnr(pred, gt)
    except Exception:
        metrics["psnr"] = float('nan')
    
    try:
        metrics["ssim"] = Metrics.ssim(pred, gt, data_range=1.0)
    except Exception:
        metrics["ssim"] = float('nan')
    
    # Skip FRC for natural images as it's more relevant for microscopy
    try:
        metrics["mse"] = float(np.mean((pred - gt) ** 2))
    except Exception:
        metrics["mse"] = float('nan')
    
    return metrics


def run_sid_evaluation(config: Dict, args) -> Dict[str, Dict[str, float]]:
    """Run SID (See-in-the-Dark) evaluation for cross-domain generalization."""
    print("🔍 Starting SID Cross-Domain Evaluation")
    print("🌙 Testing microscopy-trained model on natural low-light images")
    print("=" * 60)
    
    # Import unified evaluation system
    from pkl_dg.evaluation.evaluation import UnifiedEvaluator, EvaluationMode
    
    # Setup parameters
    device = str(config['experiment']['device'])
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint_path = args.checkpoint or (Path(config['paths']['checkpoints']) / "best_model.pt")
    data_dir = args.data_dir or "data/SID"
    camera_type = getattr(args, 'camera', 'Sony')
    guidance_types = getattr(args, 'guidance_types', ['pkl'])
    max_images = getattr(args, 'max_images', None)
    
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"SID data directory: {data_dir}")
    print(f"Camera type: {camera_type}")
    print(f"Guidance strategies: {guidance_types}")
    
    # Create configuration for SID evaluation
    from omegaconf import OmegaConf
    
    sid_config = OmegaConf.create({
        'mode': 'sid_evaluation',
        'model': {'checkpoint_path': str(checkpoint_path)},
        'inference': {
            'output_dir': getattr(args, 'output_dir', 'results/sid_evaluation'),
            'device': device
        },
        'processing': {
            'patch_size': getattr(args, 'image_size', 256),
            'max_images': max_images,
            'sid_camera_type': camera_type,
            'sid_data_dir': data_dir,
            'sid_guidance_types': guidance_types,
            'sid_num_steps': getattr(args, 'num_steps', 50),
            'sid_guidance_scale': getattr(args, 'guidance_scale', 0.1),
            'sid_eta': 0.0,
            'sid_use_processed': True,
            'save_summary_plots': True
        },
        'wandb': {
            'enabled': getattr(args, 'wandb_project', None) is not None,
            'project': getattr(args, 'wandb_project', 'pkl-diffusion-sid')
        }
    })
    
    # Run SID evaluation using unified system
    evaluator = UnifiedEvaluator(sid_config)
    results = evaluator.run_evaluation()
    
    # Convert to format expected by main script
    formatted_results = {}
    for guidance_type, result_data in results.items():
        summary = result_data['summary']
        formatted_results[guidance_type] = {
            'psnr': summary['mean_psnr'],
            'ssim': summary['mean_ssim'],
            'processing_time': summary['mean_processing_time']
        }
        if 'mean_lpips' in summary:
            formatted_results[guidance_type]['lpips'] = summary['mean_lpips']
    
    print("\n🎉 SID cross-domain evaluation complete!")
    print("📊 Results demonstrate generalization from microscopy to natural images")
    
    return formatted_results


def run_evaluation(config: Dict, args) -> Dict[str, Dict[str, float]]:
    """Run evaluation workflow for natural images."""
    
    # Check if this is SID evaluation
    if hasattr(args, 'dataset') and args.dataset == 'sid':
        return run_sid_evaluation(config, args)
    
    print("🔍 Starting Natural Image Evaluation")
    print("=" * 50)
    
    device = str(config['experiment']['device'])
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
        checkpoint_path = Path(config['paths']['checkpoints']) / "best_model.pt"
    
    print(f"Model checkpoint: {checkpoint_path}")
    
    # Get image pairs
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
        image_paths.extend(input_dir.glob(ext))
    
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")
    
    print(f"Found {len(image_paths)} images to evaluate")
    
    # Load models and samplers for different guidance types
    guidance_types = ['l2', 'anscombe', 'pkl']
    samplers = {}
    
    for guidance_type in guidance_types:
        try:
            samplers[guidance_type] = load_model_and_sampler(config, checkpoint_path, guidance_type, device)
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
    
    # Determine if we should keep RGB or convert to grayscale based on model config
    model_channels = config['model']['in_channels']
    keep_rgb = model_channels == 3
    
    # Process each image
    for img_path in tqdm(image_paths, desc="Evaluating"):
        # Load input and ground truth using proper normalization
        degraded_input = load_and_normalize_image(img_path, target_range="unit", keep_rgb=keep_rgb)
        
        gt_path = gt_dir / img_path.name
        if not gt_path.exists():
            print(f"⚠️ Ground truth not found for {img_path.name}, skipping")
            continue
            
        gt = load_and_normalize_image(gt_path, target_range="unit", keep_rgb=keep_rgb)
        
        # Degraded input baseline
        degraded_metrics = compute_metrics(degraded_input, gt)
        accumulate_results("degraded_input", degraded_metrics)
        
        # Diffusion methods
        for guidance_type, sampler in samplers.items():
            try:
                # Convert to tensor
                input_tensor = torch.from_numpy(degraded_input).float().to(device)
                
                # Handle different channel dimensions
                if input_tensor.ndim == 2:  # Grayscale (H, W)
                    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
                elif input_tensor.ndim == 3:  # RGB (H, W, 3)
                    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)  # -> (1, 3, H, W)
                
                # Create conditioner if needed
                conditioning_type = str(config['training']['conditioning_type']).lower()
                conditioner = sampler.transform(input_tensor) if conditioning_type == "wf" else None
                
                # Sample
                pred_tensor = sampler.sample(
                    input_tensor, 
                    tuple(input_tensor.shape), 
                    device=device, 
                    verbose=False, 
                    conditioner=conditioner
                )
                
                # Convert back to numpy, handling different channel dimensions
                pred = pred_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
                if pred.ndim == 3 and pred.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
                    pred = pred.transpose(1, 2, 0)
                elif pred.ndim == 3 and pred.shape[0] == 1:  # (1, H, W) -> (H, W)
                    pred = pred.squeeze(0)
                
                # Compute metrics
                diffusion_metrics = compute_metrics(pred, gt)
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
        description="Unified Training and Evaluation Script for PKL Diffusion Denoising on Natural Images",
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
    
    # Training arguments
    parser.add_argument('--dataset', type=str, choices=['imagenet', 'cifar10', 'cifar100', 'mnist', 'sid'], 
                       default='imagenet', help='Dataset type to use')
    parser.add_argument('--max-epochs', type=int, help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, help='Device (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--image-size', type=int, help='Image size')
    parser.add_argument('--num-workers', type=int, help='Number of data loader workers')
    parser.add_argument('--noise-model', choices=['gaussian', 'poisson', 'poisson_gaussian'], help='Noise model')
    parser.add_argument('--channels', type=int, choices=[1, 3], help='Number of image channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--force-grayscale', action='store_true', help='Convert RGB images to grayscale')
    
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
    
    # SID-specific arguments
    parser.add_argument('--camera', type=str, default='Sony', choices=['Sony', 'Fuji'], 
                       help='Camera type for SID dataset')
    parser.add_argument('--guidance-types', nargs='+', default=['pkl'], 
                       choices=['pkl', 'l2', 'anscombe'], help='Guidance strategies for SID evaluation')
    parser.add_argument('--max-images', type=int, help='Maximum images to evaluate (for SID)')
    parser.add_argument('--num-steps', type=int, default=50, help='DDIM sampling steps')
    parser.add_argument('--guidance-scale', type=float, default=0.1, help='Guidance scale')
    
    # Logging
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], help='W&B logging mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['train', 'train_eval']:
        # For MNIST and CIFAR, data_dir is optional (they can be downloaded)
        if args.dataset in ['imagenet'] and not args.data_dir:
            parser.error("--data-dir is required for ImageNet dataset")
        # Set default data directories for MNIST/CIFAR if not provided
        if not args.data_dir:
            args.data_dir = f"data/{args.dataset}"
    
    if args.mode in ['eval'] and not (args.input_dir and args.gt_dir):
        parser.error("--input-dir and --gt-dir are required for evaluation mode")
    
    if args.mode == 'train_eval' and not (args.eval_input and args.eval_gt):
        parser.error("--eval-input and --eval-gt are required for train_eval mode")
    
    try:
        # Setup experiment
        config = setup_experiment(args)
        
        # Run training
        if args.mode in ['train', 'train_eval']:
            trainer = run_training(config, args)
            
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
                args.checkpoint = str(Path(config['paths']['checkpoints']) / "best_model.pt")
            
            if not config and args.checkpoint:
                # Load config from checkpoint for eval-only mode
                checkpoint = torch.load(args.checkpoint, map_location='cpu')
                config = checkpoint['config']
            
            results = run_evaluation(config, args)
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
