# Phase 8: Training Script (Week 5-6)

### Step 8.1: Create Training Script
```python
# scripts/train_diffusion.py
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.data.synthesis import SynthesisDataset
from pkl_dg.data.transforms import IntensityToModel
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    """Train diffusion model."""
    
    # Set seed
    pl.seed_everything(cfg.experiment.seed)
    
    # Initialize W&B
    if cfg.wandb.mode != 'disabled':
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.experiment.name
        )
        logger = WandbLogger()
    else:
        logger = None
    
    # Setup paths
    data_dir = cfg.paths.data
    checkpoint_dir = cfg.paths.checkpoints
    
    # Create forward model for data synthesis
    psf = PSF(cfg.physics.psf_path)
    forward_model = ForwardModel(
        psf=psf.to_torch(device='cpu'),
        background=cfg.physics.background,
        device='cpu'
    )
    
    # Create transform
    transform = IntensityToModel(
        min_intensity=cfg.data.min_intensity,
        max_intensity=cfg.data.max_intensity
    )
    
    # Create datasets
    train_dataset = SynthesisDataset(
        source_dir=f"{data_dir}/train",
        forward_model=forward_model,
        transform=transform,
        image_size=cfg.data.image_size,
        mode='train'
    )
    
    val_dataset = SynthesisDataset(
        source_dir=f"{data_dir}/val",
        forward_model=forward_model,
        transform=transform,
        image_size=cfg.data.image_size,
        mode='val'
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    
    # Create model
    unet = DenoisingUNet(cfg.model)
    
    # Create trainer module
    ddpm_trainer = DDPMTrainer(
        model=unet,
        config=cfg.training,
        transform=transform
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='ddpm-{epoch:02d}-{val_loss:.4f}',
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor='val/loss',
            patience=cfg.training.early_stopping_patience,
            mode='min'
        )
    ]
    
    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=cfg.training.num_gpus,
        precision=16 if cfg.experiment.mixed_precision else 32,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=cfg.training.gradient_clip,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        log_every_n_steps=10,
        val_check_interval=cfg.training.val_check_interval
    )
    
    # Train
    trainer.fit(ddpm_trainer, train_loader, val_loader)
    
    # Save final model
    torch.save(ddpm_trainer.state_dict(), f"{checkpoint_dir}/final_model.pt")
    
    if cfg.wandb.mode != 'disabled':
        wandb.finish()

if __name__ == "__main__":
    train()
```
