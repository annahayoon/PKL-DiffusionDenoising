"""
I/O Utilities for PKL Diffusion Denoising

This module centralizes file I/O operations, checkpoint management,
logging setup, and integration with external services like Weights & Biases.
"""

import os
import json
import yaml
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import logging
from datetime import datetime
import shutil

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from omegaconf import OmegaConf, DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False


class CheckpointManager:
    """Manages model checkpoints with automatic cleanup and versioning."""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best: bool = True,
        monitor_metric: str = "val/loss",
        mode: str = "min"
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Whether to save best checkpoint separately
            monitor_metric: Metric to monitor for best checkpoint
            mode: "min" or "max" for best checkpoint selection
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.checkpoint_history = []
        
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save a checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch:04d}_step_{step:06d}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics or {},
            "timestamp": timestamp,
        }
        
        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
            
        if extra_state is not None:
            checkpoint_data["extra_state"] = extra_state
            
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        self.checkpoint_history.append(checkpoint_path)
        
        # Check if this is the best checkpoint
        if self.save_best and metrics and self.monitor_metric in metrics:
            metric_value = metrics[self.monitor_metric]
            is_best = (
                (self.mode == "min" and metric_value < self.best_value) or
                (self.mode == "max" and metric_value > self.best_value)
            )
            
            if is_best:
                self.best_value = metric_value
                best_path = self.checkpoint_dir / "best_model.pt"
                shutil.copy2(checkpoint_path, best_path)
                
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        # Load scheduler state if provided
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        return checkpoint
    
    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu"
    ) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint if it exists."""
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            return self.load_checkpoint(best_path, model, optimizer, scheduler, device)
        return None
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        if len(self.checkpoint_history) > self.max_checkpoints:
            # Sort by modification time
            self.checkpoint_history.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest checkpoints
            while len(self.checkpoint_history) > self.max_checkpoints:
                old_checkpoint = self.checkpoint_history.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()


def setup_logging(
    log_dir: Union[str, Path],
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("pkl_diffusion")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add file handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def setup_wandb(
    project_name: str,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None
) -> Optional[Any]:
    """Setup Weights & Biases logging."""
    if not WANDB_AVAILABLE:
        print("⚠️ wandb not available, skipping W&B setup")
        return None
    
    try:
        run = wandb.init(
            project=project_name,
            config=config,
            name=run_name,
            tags=tags,
            notes=notes
        )
        return run
    except Exception as e:
        print(f"⚠️ Failed to initialize wandb: {e}")
        return None


def save_config(config: Dict[str, Any], file_path: Union[str, Path]):
    """Save configuration to file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_path.suffix in ['.yaml', '.yml']:
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif file_path.suffix == '.json':
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {file_path.suffix}")


def load_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    if file_path.suffix in ['.yaml', '.yml']:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
    elif file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {file_path.suffix}")
    
    return config


def save_numpy(array: np.ndarray, file_path: Union[str, Path]):
    """Save numpy array to file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, array)


def load_numpy(file_path: Union[str, Path]) -> np.ndarray:
    """Load numpy array from file."""
    return np.load(file_path)


def save_tensor(tensor: torch.Tensor, file_path: Union[str, Path]):
    """Save torch tensor to file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, file_path)


def load_tensor(file_path: Union[str, Path], device: str = "cpu") -> torch.Tensor:
    """Load torch tensor from file."""
    return torch.load(file_path, map_location=device)


def save_pickle(obj: Any, file_path: Union[str, Path]):
    """Save object using pickle."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path: Union[str, Path]) -> Any:
    """Load object using pickle."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """Get the latest checkpoint in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None
    
    # Sort by modification time, return latest
    latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
    return latest


def copy_file(src: Union[str, Path], dst: Union[str, Path]):
    """Copy file with directory creation."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def backup_file(file_path: Union[str, Path], backup_suffix: str = ".bak"):
    """Create backup of a file."""
    file_path = Path(file_path)
    if file_path.exists():
        backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
        shutil.copy2(file_path, backup_path)
        return backup_path
    return None


def clean_directory(dir_path: Union[str, Path], pattern: str = "*", keep_latest: int = 0, 
                   safe_mode: bool = True):
    """
    Clean directory keeping only latest N files matching pattern.
    
    Args:
        dir_path: Directory path to clean
        pattern: Glob pattern to match files/directories
        keep_latest: Number of latest files to keep (0 = remove all matching)
        safe_mode: If True, avoid cleaning important directories like 'data'
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return
    
    # Safety check to avoid accidentally cleaning important directories
    if safe_mode:
        dangerous_patterns = ['data', 'dataset', 'datasets']
        dir_name = dir_path.name.lower()
        if any(pattern in dir_name for pattern in dangerous_patterns):
            print(f"⚠️  WARNING: Skipping cleanup of potentially important directory: {dir_path}")
            print("   Use safe_mode=False to override this protection")
            return
    
    files = list(dir_path.glob(pattern))
    if len(files) <= keep_latest:
        return
    
    # Sort by modification time, keep latest
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    files_to_remove = files[keep_latest:]
    
    print(f"Cleaning {len(files_to_remove)} files from {dir_path} (keeping {keep_latest} latest)")
    
    for file_path in files_to_remove:
        try:
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Warning: Could not remove {file_path}: {e}")


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes."""
    return Path(file_path).stat().st_size


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def list_files_with_info(dir_path: Union[str, Path], pattern: str = "*") -> List[Dict[str, Any]]:
    """List files with size and modification time info."""
    dir_path = Path(dir_path)
    files_info = []
    
    for file_path in dir_path.glob(pattern):
        if file_path.is_file():
            stat = file_path.stat()
            files_info.append({
                "path": file_path,
                "size_bytes": stat.st_size,
                "size_formatted": format_file_size(stat.st_size),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "name": file_path.name
            })
    
    return sorted(files_info, key=lambda x: x["modified"], reverse=True)


def unified_training_loop(
    model,
    train_loader,
    optimizer,
    max_epochs: int,
    device: str = "cuda",
    scaler=None,
    grad_clip_val: float = 0.0,
    checkpoint_manager=None,
    wandb_log: bool = False,
    verbose: bool = True,
):
    """Unified training loop to reduce code duplication between scripts.
    
    Args:
        model: Model to train (should have training_step method)
        train_loader: Training data loader
        optimizer: Optimizer
        max_epochs: Number of epochs to train
        device: Device to use
        scaler: GradScaler for mixed precision (optional)
        grad_clip_val: Gradient clipping value (0 = no clipping)
        checkpoint_manager: CheckpointManager for saving checkpoints
        wandb_log: Whether to log to wandb
        verbose: Whether to show progress bars
        
    Returns:
        Dictionary with training statistics
    """
    from torch.cuda.amp import autocast
    import torch.nn.utils
    from tqdm import tqdm
    
    training_stats = {
        "epoch_losses": [],
        "best_loss": float('inf'),
        "total_steps": 0,
    }
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False) if verbose else train_loader
        
        for batch_idx, batch in enumerate(progress):
            # Move batch to device - handle different batch structures
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2 and hasattr(batch[0], 'to') and hasattr(batch[1], 'to'):
                    # Handle (x_0, y_degraded) format from natural images
                    x_0, y_degraded = batch
                    x_0 = x_0.to(device, non_blocking=True)
                    y_degraded = y_degraded.to(device, non_blocking=True)
                    batch = (x_0, y_degraded)
                else:
                    # Handle general list/tuple format
                    batch = [b.to(device, non_blocking=True) for b in batch if hasattr(b, 'to')]
            else:
                batch = batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass with mixed precision support
            if scaler is not None:
                with autocast():
                    loss = model.training_step(batch, batch_idx)
                
                scaler.scale(loss).backward()
                
                if grad_clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model.training_step(batch, batch_idx)
                loss.backward()
                
                if grad_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            training_stats["total_steps"] += 1

            # Update progress bar
            if verbose:
                progress.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'avg_loss': f'{epoch_loss/num_batches:.6f}'
                })

            # Memory cleanup every 50 batches
            if batch_idx % 50 == 0:
                try:
                    from .memory import cleanup_memory
                    cleanup_memory()
                except ImportError:
                    # Fallback memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        training_stats["epoch_losses"].append(avg_loss)
        
        if avg_loss < training_stats["best_loss"]:
            training_stats["best_loss"] = avg_loss

        if verbose:
            print(f"Epoch {epoch+1}/{max_epochs} - Average Loss: {avg_loss:.6f}")

        # Update EMA if available
        if hasattr(model, 'ema_model') and model.ema_model is not None:
            if hasattr(model, 'update_ema'):
                model.update_ema()

        # Save checkpoint
        if checkpoint_manager is not None:
            is_best = avg_loss < training_stats["best_loss"]
            checkpoint_manager.save_checkpoint(
                model, optimizer, epoch, avg_loss, is_best=is_best
            )

        # Log to wandb
        if wandb_log:
            try:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "best_loss": training_stats["best_loss"],
                })
            except ImportError:
                pass

    return training_stats


def setup_distributed_training(rank: int = None, world_size: int = None, backend: str = 'nccl'):
    """Setup distributed training environment.
    
    Args:
        rank: Process rank (auto-detected if None)
        world_size: Total number of processes (auto-detected if None)
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
        
    Returns:
        Dictionary with distributed training info
    """
    import os
    
    # Auto-detect distributed environment
    if rank is None:
        rank = int(os.environ.get('LOCAL_RANK', 0))
    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    distributed_info = {
        "rank": rank,
        "world_size": world_size,
        "backend": backend,
        "distributed": world_size > 1,
    }
    
    if world_size > 1:
        try:
            import torch.distributed as dist
            
            # Initialize process group
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=backend,
                    rank=rank,
                    world_size=world_size
                )
            
            # Set device for current process
            if torch.cuda.is_available() and backend == 'nccl':
                device_id = rank % torch.cuda.device_count()
                torch.cuda.set_device(device_id)
                distributed_info["device_id"] = device_id
                distributed_info["device"] = f"cuda:{device_id}"
            else:
                distributed_info["device"] = "cpu"
            
            print(f"Initialized distributed training: rank {rank}/{world_size}")
            
        except Exception as e:
            print(f"Failed to initialize distributed training: {e}")
            distributed_info["distributed"] = False
    
    return distributed_info


def create_distributed_model(model, device_ids: list = None, find_unused_parameters: bool = False):
    """Wrap model for distributed training.
    
    Args:
        model: PyTorch model
        device_ids: List of device IDs (auto-detected if None)
        find_unused_parameters: Whether to find unused parameters
        
    Returns:
        Wrapped model for distributed training
    """
    try:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        if not dist.is_initialized():
            print("Distributed training not initialized. Returning original model.")
            return model
        
        # Auto-detect device IDs
        if device_ids is None and torch.cuda.is_available():
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device_ids = [local_rank]
        
        # Move model to appropriate device
        if device_ids and torch.cuda.is_available():
            model = model.to(f"cuda:{device_ids[0]}")
        
        # Wrap with DDP
        ddp_model = DDP(
            model,
            device_ids=device_ids,
            find_unused_parameters=find_unused_parameters
        )
        
        print(f"Created distributed model on devices: {device_ids}")
        return ddp_model
        
    except Exception as e:
        print(f"Failed to create distributed model: {e}")
        return model


def cleanup_distributed():
    """Cleanup distributed training environment."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
            print("Cleaned up distributed training")
    except Exception:
        pass


def distributed_training_loop(
    model,
    train_loader,
    optimizer,
    max_epochs: int,
    device: str = "cuda",
    scaler=None,
    grad_clip_val: float = 0.0,
    checkpoint_manager=None,
    wandb_log: bool = False,
    verbose: bool = True,
    rank: int = 0,
):
    """Distributed training loop with proper synchronization.
    
    Args:
        model: Distributed model (wrapped with DDP)
        train_loader: Training data loader
        optimizer: Optimizer
        max_epochs: Number of epochs to train
        device: Device to use
        scaler: GradScaler for mixed precision
        grad_clip_val: Gradient clipping value
        checkpoint_manager: CheckpointManager for saving checkpoints
        wandb_log: Whether to log to wandb (only on rank 0)
        verbose: Whether to show progress bars (only on rank 0)
        rank: Process rank
        
    Returns:
        Dictionary with training statistics
    """
    from torch.cuda.amp import autocast
    import torch.nn.utils
    from tqdm import tqdm
    
    training_stats = {
        "epoch_losses": [],
        "best_loss": float('inf'),
        "total_steps": 0,
    }
    
    for epoch in range(max_epochs):
        model.train()
        
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        num_batches = 0

        # Only show progress on rank 0
        if rank == 0 and verbose:
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)
        else:
            progress = train_loader
        
        for batch_idx, batch in enumerate(progress):
            # Move batch to device
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2 and hasattr(batch[0], 'to') and hasattr(batch[1], 'to'):
                    x_0, y_degraded = batch
                    x_0 = x_0.to(device, non_blocking=True)
                    y_degraded = y_degraded.to(device, non_blocking=True)
                    batch = (x_0, y_degraded)
                else:
                    batch = [b.to(device, non_blocking=True) for b in batch if hasattr(b, 'to')]
            else:
                batch = batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass with mixed precision support
            if scaler is not None:
                with autocast():
                    loss = model.module.training_step(batch, batch_idx) if hasattr(model, 'module') else model.training_step(batch, batch_idx)
                
                scaler.scale(loss).backward()
                
                if grad_clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model.module.training_step(batch, batch_idx) if hasattr(model, 'module') else model.training_step(batch, batch_idx)
                loss.backward()
                
                if grad_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            training_stats["total_steps"] += 1

            # Update progress bar (only on rank 0)
            if rank == 0 and verbose:
                progress.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'avg_loss': f'{epoch_loss/num_batches:.6f}'
                })

            # Memory cleanup every 50 batches
            if batch_idx % 50 == 0:
                try:
                    from .memory import cleanup_memory
                    cleanup_memory()
                except ImportError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        training_stats["epoch_losses"].append(avg_loss)
        
        # Synchronize loss across all processes
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                loss_tensor = torch.tensor(avg_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = loss_tensor.item() / dist.get_world_size()
        except Exception:
            pass
        
        if avg_loss < training_stats["best_loss"]:
            training_stats["best_loss"] = avg_loss

        # Only print and log on rank 0
        if rank == 0:
            if verbose:
                print(f"Epoch {epoch+1}/{max_epochs} - Average Loss: {avg_loss:.6f}")

            # Save checkpoint (only on rank 0)
            if checkpoint_manager is not None:
                is_best = avg_loss < training_stats["best_loss"]
                # Save the underlying model, not the DDP wrapper
                model_to_save = model.module if hasattr(model, 'module') else model
                checkpoint_manager.save_checkpoint(
                    model_to_save, optimizer, epoch, avg_loss, is_best=is_best
                )

            # Log to wandb (only on rank 0)
            if wandb_log:
                try:
                    import wandb
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": avg_loss,
                        "best_loss": training_stats["best_loss"],
                    })
                except ImportError:
                    pass

    return training_stats
