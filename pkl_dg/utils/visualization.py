"""
Visualization Utilities for PKL Diffusion Denoising

This module provides comprehensive plotting and visualization utilities
for training monitoring, sample generation, and result analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from typing import List, Optional, Union, Tuple, Dict, Any
from pathlib import Path
import seaborn as sns
from PIL import Image
import glob
import os

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_matplotlib_style():
    """Setup consistent matplotlib style."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Set default parameters
    plt.rcParams.update({
        'figure.figsize': (10, 8),
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3
    })


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """Plot training and validation curves."""
    setup_matplotlib_style()
    
    # Separate train and validation metrics
    train_metrics = {k: v for k, v in metrics_history.items() if k.startswith('train/')}
    val_metrics = {k: v for k, v in metrics_history.items() if k.startswith('val/')}
    
    # Determine number of subplots needed
    unique_metrics = set()
    for key in metrics_history.keys():
        metric_name = key.split('/')[-1]  # Remove train/val prefix
        unique_metrics.add(metric_name)
    
    n_metrics = len(unique_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_metrics > 1 else [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, metric_name in enumerate(sorted(unique_metrics)):
        ax = axes[i] if i < len(axes) else axes[-1]
        
        # Plot training curve
        train_key = f'train/{metric_name}'
        if train_key in train_metrics:
            ax.plot(train_metrics[train_key], label=f'Train {metric_name}', 
                   linewidth=2, alpha=0.8)
        
        # Plot validation curve
        val_key = f'val/{metric_name}'
        if val_key in val_metrics:
            ax.plot(val_metrics[val_key], label=f'Val {metric_name}', 
                   linewidth=2, alpha=0.8, linestyle='--')
        
        ax.set_title(f'{metric_name.capitalize()} Over Time')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_sample_grid(
    samples: torch.Tensor,
    nrow: int = 8,
    ncol: Optional[int] = None,
    titles: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = 'gray',
    title: str = "Sample Grid"
) -> plt.Figure:
    """Plot a grid of samples."""
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    
    batch_size = samples.shape[0]
    if ncol is None:
        ncol = min(nrow, batch_size)
    nrow = min(nrow, (batch_size + ncol - 1) // ncol)
    
    if figsize is None:
        figsize = (ncol * 3, nrow * 3)
    
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    if nrow == 1 and ncol == 1:
        axes = [axes]
    elif nrow == 1 or ncol == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i in range(nrow * ncol):
        ax = axes[i] if i < len(axes) else axes[-1]
        
        if i < batch_size:
            # Handle different tensor shapes
            sample = samples[i]
            if sample.ndim == 3:  # CHW
                if sample.shape[0] == 1:  # Grayscale
                    sample = sample[0]
                else:  # RGB
                    sample = np.transpose(sample, (1, 2, 0))
            
            ax.imshow(sample, cmap=cmap)
            ax.axis('off')
            
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=10)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comparison_grid(
    source_images: torch.Tensor,
    target_images: torch.Tensor,
    generated_images: torch.Tensor,
    nrow: int = 4,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Source → Target → Generated Comparison"
) -> plt.Figure:
    """Plot comparison grid showing source, target, and generated images."""
    # Convert to numpy
    if isinstance(source_images, torch.Tensor):
        source_images = source_images.detach().cpu().numpy()
    if isinstance(target_images, torch.Tensor):
        target_images = target_images.detach().cpu().numpy()
    if isinstance(generated_images, torch.Tensor):
        generated_images = generated_images.detach().cpu().numpy()
    
    batch_size = min(source_images.shape[0], target_images.shape[0], generated_images.shape[0])
    nrow = min(nrow, batch_size)
    
    fig, axes = plt.subplots(nrow, 3, figsize=(12, nrow * 4))
    if nrow == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i in range(nrow):
        # Source image
        source = source_images[i]
        if source.ndim == 3 and source.shape[0] == 1:
            source = source[0]
        axes[i, 0].imshow(source, cmap='gray')
        axes[i, 0].set_title('Source (WF)' if i == 0 else '')
        axes[i, 0].axis('off')
        
        # Target image
        target = target_images[i]
        if target.ndim == 3 and target.shape[0] == 1:
            target = target[0]
        axes[i, 1].imshow(target, cmap='gray')
        axes[i, 1].set_title('Target (2P)' if i == 0 else '')
        axes[i, 1].axis('off')
        
        # Generated image
        generated = generated_images[i]
        if generated.ndim == 3 and generated.shape[0] == 1:
            generated = generated[0]
        axes[i, 2].imshow(generated, cmap='gray')
        axes[i, 2].set_title('Generated' if i == 0 else '')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_memory_usage(
    memory_snapshots: List[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Memory Usage Over Time"
) -> plt.Figure:
    """Plot memory usage over time."""
    setup_matplotlib_style()
    
    timestamps = [s['timestamp'] for s in memory_snapshots]
    gpu_allocated = [s.get('gpu_allocated', 0) for s in memory_snapshots]
    gpu_reserved = [s.get('gpu_reserved', 0) for s in memory_snapshots]
    cpu_memory = [s.get('cpu_memory', 0) for s in memory_snapshots]
    
    # Convert timestamps to relative time
    start_time = timestamps[0] if timestamps else 0
    relative_times = [(t - start_time) / 60 for t in timestamps]  # Convert to minutes
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # GPU Memory plot
    ax1.plot(relative_times, gpu_allocated, label='GPU Allocated', linewidth=2)
    ax1.plot(relative_times, gpu_reserved, label='GPU Reserved', linewidth=2, alpha=0.7)
    ax1.set_title('GPU Memory Usage')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Memory (GB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # CPU Memory plot
    ax2.plot(relative_times, cpu_memory, label='CPU Memory', color='orange', linewidth=2)
    ax2.set_title('CPU Memory Usage')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Memory (GB)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_loss_landscape(
    losses: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Loss Landscape"
) -> plt.Figure:
    """Plot 2D loss landscape."""
    setup_matplotlib_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create contour plot
    x = np.linspace(x_range[0], x_range[1], losses.shape[1])
    y = np.linspace(y_range[0], y_range[1], losses.shape[0])
    X, Y = np.meshgrid(x, y)
    
    contour = ax.contourf(X, Y, losses, levels=50, cmap='viridis')
    ax.contour(X, Y, losses, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Loss Value')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_interpolation_sequence(
    interpolation_sequence: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Interpolation Sequence",
    cmap: str = 'gray'
) -> plt.Figure:
    """Plot interpolation sequence between two images."""
    if isinstance(interpolation_sequence, torch.Tensor):
        sequence = interpolation_sequence.detach().cpu().numpy()
    else:
        sequence = interpolation_sequence
    
    n_steps = sequence.shape[0]
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 3, 4))
    
    if n_steps == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes):
        image = sequence[i]
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]
        
        ax.imshow(image, cmap=cmap)
        ax.set_title(f'Step {i}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_psf_analysis(
    psf: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "PSF Analysis"
) -> plt.Figure:
    """Plot PSF analysis including 2D view and cross-sections."""
    if isinstance(psf, torch.Tensor):
        psf_np = psf.detach().cpu().numpy()
    else:
        psf_np = psf
    
    if psf_np.ndim == 3:
        psf_np = psf_np[0]  # Remove channel dimension
    
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # 2D PSF view
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(psf_np, cmap='hot')
    ax1.set_title('2D PSF')
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, shrink=0.8)
    
    # Horizontal cross-section
    ax2 = fig.add_subplot(gs[1])
    center_y = psf_np.shape[0] // 2
    ax2.plot(psf_np[center_y, :])
    ax2.set_title('Horizontal Cross-section')
    ax2.set_xlabel('Pixel')
    ax2.set_ylabel('Intensity')
    ax2.grid(True, alpha=0.3)
    
    # Vertical cross-section
    ax3 = fig.add_subplot(gs[2])
    center_x = psf_np.shape[1] // 2
    ax3.plot(psf_np[:, center_x])
    ax3.set_title('Vertical Cross-section')
    ax3.set_xlabel('Pixel')
    ax3.set_ylabel('Intensity')
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_evaluation_metrics(
    metrics_dict: Dict[str, float],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Evaluation Metrics"
) -> plt.Figure:
    """Plot evaluation metrics as bar chart."""
    setup_matplotlib_style()
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(metrics, values, color=sns.color_palette("husl", len(metrics)))
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel('Metric Value')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_training_dashboard(
    metrics_history: Dict[str, List[float]],
    sample_images: Optional[torch.Tensor] = None,
    memory_snapshots: Optional[List[Dict[str, Any]]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Create comprehensive training dashboard."""
    setup_matplotlib_style()
    
    # Determine layout based on available data
    n_plots = 1  # Always have metrics
    if sample_images is not None:
        n_plots += 1
    if memory_snapshots is not None:
        n_plots += 1
    
    if n_plots == 1:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        axes = [ax]
    elif n_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
    
    fig.suptitle('Training Dashboard', fontsize=20, fontweight='bold')
    
    plot_idx = 0
    
    # Plot training curves
    ax = axes[plot_idx]
    plot_idx += 1
    
    for metric_name, values in metrics_history.items():
        ax.plot(values, label=metric_name, linewidth=2)
    
    ax.set_title('Training Metrics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot sample images if available
    if sample_images is not None and plot_idx < len(axes):
        # Create subplot for samples within the dashboard
        # This is simplified - for full grid, use plot_sample_grid separately
        ax = axes[plot_idx]
        plot_idx += 1
        
        if isinstance(sample_images, torch.Tensor):
            sample = sample_images[0].detach().cpu().numpy()
        else:
            sample = sample_images[0]
        
        if sample.ndim == 3 and sample.shape[0] == 1:
            sample = sample[0]
        
        ax.imshow(sample, cmap='gray')
        ax.set_title('Latest Sample')
        ax.axis('off')
    
    # Plot memory usage if available
    if memory_snapshots is not None and plot_idx < len(axes):
        ax = axes[plot_idx]
        plot_idx += 1
        
        timestamps = [s['timestamp'] for s in memory_snapshots]
        gpu_allocated = [s.get('gpu_allocated', 0) for s in memory_snapshots]
        
        if timestamps:
            start_time = timestamps[0]
            relative_times = [(t - start_time) / 60 for t in timestamps]
            ax.plot(relative_times, gpu_allocated, linewidth=2)
            ax.set_title('GPU Memory Usage')
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Memory (GB)')
            ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def log_to_wandb(
    metrics: Dict[str, float],
    images: Optional[Dict[str, Union[torch.Tensor, np.ndarray, plt.Figure]]] = None,
    step: Optional[int] = None
):
    """Log metrics and images to Weights & Biases."""
    if not WANDB_AVAILABLE:
        print("⚠️ wandb not available, skipping logging")
        return
    
    try:
        log_dict = metrics.copy()
        
        if images:
            for name, image in images.items():
                if isinstance(image, plt.Figure):
                    log_dict[name] = wandb.Image(image)
                elif isinstance(image, torch.Tensor):
                    log_dict[name] = wandb.Image(image.detach().cpu().numpy())
                elif isinstance(image, np.ndarray):
                    log_dict[name] = wandb.Image(image)
        
        wandb.log(log_dict, step=step)
        
    except Exception as e:
        print(f"⚠️ Failed to log to wandb: {e}")


def save_figure_safely(fig: plt.Figure, save_path: Union[str, Path], dpi: int = 300):
    """Save figure with error handling."""
    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved figure to {save_path}")
    except Exception as e:
        print(f"⚠️ Failed to save figure to {save_path}: {e}")


def close_all_figures():
    """Close all matplotlib figures to free memory."""
    plt.close('all')


def to_uint8(a: np.ndarray) -> np.ndarray:
    """Convert array to uint8 with percentile-based normalization."""
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, (1, 99))
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        a = (a - lo) / (hi - lo)
    a = np.clip(a, 0.0, 1.0)
    return (a * 255).astype(np.uint8)


def read_tif(path: str) -> np.ndarray:
    """Read TIFF file using tifffile if available, otherwise PIL."""
    if TIFFFILE_AVAILABLE:
        return tifffile.imread(path)
    else:
        # Fallback to PIL
        with Image.open(path) as img:
            return np.array(img)


def create_comparison_previews(
    wf_dir: Union[str, Path],
    pred_dir: Union[str, Path], 
    gt_dir: Union[str, Path],
    out_dir: Union[str, Path],
    max_n: int = 24,
    file_pattern: str = '*.tif'
) -> int:
    """
    Create comparison preview images showing widefield, prediction, and ground truth side-by-side.
    
    Args:
        wf_dir: Directory containing widefield TIFF files
        pred_dir: Directory containing prediction files (TIFF or PNG)
        gt_dir: Directory containing ground truth TIFF files
        out_dir: Output directory for preview images
        max_n: Maximum number of previews to generate (0 for unlimited)
        file_pattern: Glob pattern for widefield files
        
    Returns:
        Number of preview images generated
    """
    # Convert to Path objects
    wf_dir = Path(wf_dir)
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    out_dir = Path(out_dir)
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find widefield files
    wf_files = sorted(wf_dir.glob(file_pattern))
    if max_n > 0:
        wf_files = wf_files[:max_n]
    
    count = 0
    for wf_path in wf_files:
        stem = wf_path.stem
        
        # Look for prediction file (support both .tif and .png)
        pred_path = pred_dir / f'{stem}_reconstructed.tif'
        if not pred_path.exists():
            pred_path = pred_dir / f'{stem}_reconstructed.png'
        if not pred_path.exists():
            continue
            
        try:
            # Read and normalize images
            wf = to_uint8(read_tif(str(wf_path)))
            pr = to_uint8(read_tif(str(pred_path)))
            panels = [wf, pr]
            
            # Add ground truth if available
            gt_path = gt_dir / f'{stem}.tif'
            if gt_path.exists():
                gt = to_uint8(read_tif(str(gt_path)))
                panels.append(gt)
            
            # Create horizontal strip
            strip = np.concatenate(panels, axis=1)
            
            # Save as PNG
            output_path = out_dir / f'{stem}_wf_pred_gt.png'
            Image.fromarray(strip).save(output_path)
            count += 1
            
        except Exception as e:
            print(f"⚠️ Failed to process {wf_path.name}: {e}")
            continue
    
    print(f"✅ Generated {count} preview(s) in {out_dir}")
    return count


# Set default style when module is imported
setup_matplotlib_style()
