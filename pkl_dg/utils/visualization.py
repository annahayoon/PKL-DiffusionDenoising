"""
Visualization utilities for PKL Diffusion Denoising.

This module provides comprehensive plotting and visualization functions
for training monitoring, result analysis, and research presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch
import torch.nn.functional as F
import math
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from pathlib import Path
import warnings

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

# Set style
plt.style.use('default')
sns.set_palette("husl")


# =============================================================================
# Interpolation Utilities (from interpolation.py)
# =============================================================================

def linear_interpolation(
    start: torch.Tensor,
    end: torch.Tensor,
    num_steps: int = 10,
    include_endpoints: bool = True
) -> torch.Tensor:
    """Linear interpolation between two tensors.
    
    Args:
        start: Starting tensor
        end: Ending tensor
        num_steps: Number of interpolation steps
        include_endpoints: Whether to include start and end points
        
    Returns:
        Tensor of interpolated values [num_steps, *tensor_shape]
    """
    if start.shape != end.shape:
        raise ValueError(f"Shape mismatch: start {start.shape} vs end {end.shape}")
    
    device = start.device
    dtype = start.dtype
    
    if include_endpoints:
        alphas = torch.linspace(0, 1, num_steps, device=device, dtype=dtype)
    else:
        alphas = torch.linspace(0, 1, num_steps + 2, device=device, dtype=dtype)[1:-1]
    
    # Reshape alphas for broadcasting
    for _ in range(start.ndim):
        alphas = alphas.unsqueeze(-1)
    
    # Interpolate
    interpolated = start.unsqueeze(0) * (1 - alphas) + end.unsqueeze(0) * alphas
    
    return interpolated


def spherical_interpolation(
    start: torch.Tensor,
    end: torch.Tensor,
    num_steps: int = 10,
    include_endpoints: bool = True,
    eps: float = 1e-7
) -> torch.Tensor:
    """Spherical linear interpolation (SLERP) between two tensors.
    
    This is particularly useful for interpolating in latent spaces where
    the vectors lie on a sphere.
    
    Args:
        start: Starting tensor
        end: Ending tensor  
        num_steps: Number of interpolation steps
        include_endpoints: Whether to include start and end points
        eps: Small value to avoid division by zero
        
    Returns:
        Tensor of interpolated values [num_steps, *tensor_shape]
    """
    if start.shape != end.shape:
        raise ValueError(f"Shape mismatch: start {start.shape} vs end {end.shape}")
    
    device = start.device
    dtype = start.dtype
    
    # Flatten tensors for computation
    start_flat = start.flatten()
    end_flat = end.flatten()
    
    # Normalize vectors
    start_norm = F.normalize(start_flat, dim=0)
    end_norm = F.normalize(end_flat, dim=0)
    
    # Compute angle between vectors
    dot_product = torch.dot(start_norm, end_norm)
    dot_product = torch.clamp(dot_product, -1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot_product)
    
    if include_endpoints:
        t_values = torch.linspace(0, 1, num_steps, device=device, dtype=dtype)
    else:
        t_values = torch.linspace(0, 1, num_steps + 2, device=device, dtype=dtype)[1:-1]
    
    # Handle case where vectors are nearly parallel
    if omega.abs() < eps:
        # Fall back to linear interpolation
        return linear_interpolation(start, end, num_steps, include_endpoints)
    
    sin_omega = torch.sin(omega)
    
    interpolated_list = []
    for t in t_values:
        # SLERP formula
        weight_start = torch.sin((1 - t) * omega) / sin_omega
        weight_end = torch.sin(t * omega) / sin_omega
        
        interpolated_flat = weight_start * start_flat + weight_end * end_flat
        interpolated = interpolated_flat.reshape(start.shape)
        interpolated_list.append(interpolated)
    
    return torch.stack(interpolated_list, dim=0)


def noise_interpolation(
    start_noise: torch.Tensor,
    end_noise: torch.Tensor,
    num_steps: int = 10,
    interpolation_type: str = "linear",
    include_endpoints: bool = True
) -> torch.Tensor:
    """Interpolate between noise tensors for diffusion models.
    
    Args:
        start_noise: Starting noise tensor
        end_noise: Ending noise tensor
        num_steps: Number of interpolation steps
        interpolation_type: "linear" or "spherical"
        include_endpoints: Whether to include start and end points
        
    Returns:
        Interpolated noise tensors
    """
    if interpolation_type == "linear":
        return linear_interpolation(start_noise, end_noise, num_steps, include_endpoints)
    elif interpolation_type == "spherical":
        return spherical_interpolation(start_noise, end_noise, num_steps, include_endpoints)
    else:
        raise ValueError(f"Unknown interpolation type: {interpolation_type}")


def latent_interpolation(
    model: torch.nn.Module,
    start_latent: torch.Tensor,
    end_latent: torch.Tensor,
    num_steps: int = 10,
    interpolation_type: str = "spherical",
    decode_fn: Optional[Callable] = None,
    include_endpoints: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interpolate in latent space and optionally decode to image space.
    
    Args:
        model: Model for decoding (if needed)
        start_latent: Starting latent vector
        end_latent: Ending latent vector
        num_steps: Number of interpolation steps
        interpolation_type: "linear" or "spherical"
        decode_fn: Optional function to decode latents to images
        include_endpoints: Whether to include start and end points
        
    Returns:
        Tuple of (interpolated_latents, decoded_images)
        decoded_images is None if decode_fn is not provided
    """
    # Interpolate in latent space
    if interpolation_type == "linear":
        interpolated_latents = linear_interpolation(
            start_latent, end_latent, num_steps, include_endpoints
        )
    elif interpolation_type == "spherical":
        interpolated_latents = spherical_interpolation(
            start_latent, end_latent, num_steps, include_endpoints
        )
    else:
        raise ValueError(f"Unknown interpolation type: {interpolation_type}")
    
    # Decode to image space if function provided
    decoded_images = None
    if decode_fn is not None:
        model.eval()
        with torch.no_grad():
            decoded_list = []
            for latent in interpolated_latents:
                decoded = decode_fn(model, latent.unsqueeze(0))
                decoded_list.append(decoded.squeeze(0))
            decoded_images = torch.stack(decoded_list, dim=0)
    
    return interpolated_latents, decoded_images


def diffusion_interpolation(
    diffusion_model: torch.nn.Module,
    start_image: torch.Tensor,
    end_image: torch.Tensor,
    num_steps: int = 10,
    timestep: int = 500,
    interpolation_type: str = "spherical",
    include_endpoints: bool = True,
    guidance_scale: float = 1.0
) -> torch.Tensor:
    """Interpolate through diffusion process.
    
    This adds noise to both images, interpolates in noisy space,
    then denoises the interpolated sequence.
    
    Args:
        diffusion_model: Trained diffusion model
        start_image: Starting image
        end_image: Ending image
        num_steps: Number of interpolation steps
        timestep: Diffusion timestep for noise addition
        interpolation_type: "linear" or "spherical"
        include_endpoints: Whether to include start and end points
        guidance_scale: Guidance scale for denoising
        
    Returns:
        Interpolated image sequence
    """
    device = start_image.device
    
    # Add noise to both images at specified timestep
    noise_start = torch.randn_like(start_image)
    noise_end = torch.randn_like(end_image)
    
    # Get noise schedule parameters
    if hasattr(diffusion_model, 'sqrt_alphas_cumprod'):
        sqrt_alpha = diffusion_model.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus_alpha = diffusion_model.sqrt_one_minus_alphas_cumprod[timestep]
    else:
        # Fallback values
        sqrt_alpha = 0.5
        sqrt_one_minus_alpha = 0.866
    
    # Add noise
    noisy_start = sqrt_alpha * start_image + sqrt_one_minus_alpha * noise_start
    noisy_end = sqrt_alpha * end_image + sqrt_one_minus_alpha * noise_end
    
    # Interpolate in noisy space
    if interpolation_type == "linear":
        noisy_interpolated = linear_interpolation(
            noisy_start, noisy_end, num_steps, include_endpoints
        )
    elif interpolation_type == "spherical":
        noisy_interpolated = spherical_interpolation(
            noisy_start, noisy_end, num_steps, include_endpoints
        )
    else:
        raise ValueError(f"Unknown interpolation type: {interpolation_type}")
    
    # Denoise interpolated sequence
    diffusion_model.eval()
    with torch.no_grad():
        denoised_list = []
        t_tensor = torch.full((1,), timestep, device=device, dtype=torch.long)
        
        for noisy_img in noisy_interpolated:
            # Single denoising step (could be extended to full sampling)
            noise_pred = diffusion_model(noisy_img.unsqueeze(0), t_tensor)
            
            # Predict original image (simplified)
            predicted_original = (noisy_img.unsqueeze(0) - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
            predicted_original = torch.clamp(predicted_original, -1, 1)
            
            denoised_list.append(predicted_original.squeeze(0))
    
    return torch.stack(denoised_list, dim=0)


def morphing_sequence(
    images: List[torch.Tensor],
    steps_between: int = 10,
    interpolation_type: str = "linear",
    loop: bool = False
) -> torch.Tensor:
    """Create morphing sequence between multiple images.
    
    Args:
        images: List of images to morph between
        steps_between: Number of steps between each pair
        interpolation_type: "linear" or "spherical"
        loop: Whether to loop back to first image
        
    Returns:
        Complete morphing sequence
    """
    if len(images) < 2:
        raise ValueError("Need at least 2 images for morphing")
    
    # Add first image to end if looping
    if loop:
        images = images + [images[0]]
    
    sequence_parts = []
    
    for i in range(len(images) - 1):
        start_img = images[i]
        end_img = images[i + 1]
        
        if interpolation_type == "linear":
            interpolated = linear_interpolation(
                start_img, end_img, steps_between, include_endpoints=False
            )
        elif interpolation_type == "spherical":
            interpolated = spherical_interpolation(
                start_img, end_img, steps_between, include_endpoints=False
            )
        else:
            raise ValueError(f"Unknown interpolation type: {interpolation_type}")
        
        # Add the start image and interpolated frames
        if i == 0:
            sequence_parts.append(start_img.unsqueeze(0))
        sequence_parts.append(interpolated)
    
    # Add final image if not looping
    if not loop:
        sequence_parts.append(images[-1].unsqueeze(0))
    
    return torch.cat(sequence_parts, dim=0)


def create_interpolation_grid(
    corner_images: List[torch.Tensor],
    grid_size: Tuple[int, int] = (5, 5),
    interpolation_type: str = "linear"
) -> torch.Tensor:
    """Create 2D interpolation grid between 4 corner images.
    
    Args:
        corner_images: List of 4 corner images [top_left, top_right, bottom_left, bottom_right]
        grid_size: (height, width) of interpolation grid
        interpolation_type: "linear" or "spherical"
        
    Returns:
        Grid of interpolated images [grid_height, grid_width, *image_shape]
    """
    if len(corner_images) != 4:
        raise ValueError("Need exactly 4 corner images")
    
    top_left, top_right, bottom_left, bottom_right = corner_images
    grid_h, grid_w = grid_size
    
    # Create coordinate grids
    y_coords = torch.linspace(0, 1, grid_h)
    x_coords = torch.linspace(0, 1, grid_w)
    
    grid_images = []
    
    for i, y in enumerate(y_coords):
        row_images = []
        
        for j, x in enumerate(x_coords):
            # Bilinear interpolation in 2D
            # First interpolate along top and bottom edges
            if interpolation_type == "linear":
                top_interp = linear_interpolation(
                    top_left, top_right, 2, include_endpoints=True
                )[1 if x > 0 else 0]  # Simplified for demo
                
                bottom_interp = linear_interpolation(
                    bottom_left, bottom_right, 2, include_endpoints=True
                )[1 if x > 0 else 0]
                
                # Then interpolate vertically
                final_interp = linear_interpolation(
                    top_interp, bottom_interp, 2, include_endpoints=True
                )[1 if y > 0 else 0]
            else:
                # For spherical, use proper 2D interpolation
                # Weights for bilinear interpolation
                w_tl = (1 - x) * (1 - y)
                w_tr = x * (1 - y)
                w_bl = (1 - x) * y
                w_br = x * y
                
                final_interp = (w_tl * top_left + w_tr * top_right + 
                               w_bl * bottom_left + w_br * bottom_right)
            
            row_images.append(final_interp)
        
        grid_images.append(torch.stack(row_images, dim=0))
    
    return torch.stack(grid_images, dim=0)


def save_interpolation_video(
    interpolation_sequence: torch.Tensor,
    output_path: Union[str, Path],
    fps: int = 30,
    loop_count: int = 1,
    normalize: bool = True
):
    """Save interpolation sequence as video.
    
    Args:
        interpolation_sequence: Sequence of images [T, C, H, W]
        output_path: Output video path
        fps: Frames per second
        loop_count: Number of times to loop the sequence
        normalize: Whether to normalize images to [0, 1]
    """
    if not IMAGEIO_AVAILABLE:
        warnings.warn("imageio not available, cannot save video")
        return
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy and normalize
    sequence_np = interpolation_sequence.detach().cpu().numpy()
    
    if normalize:
        # Normalize to [0, 1] range
        sequence_np = (sequence_np + 1) / 2  # Assuming input in [-1, 1]
        sequence_np = np.clip(sequence_np, 0, 1)
    
    # Convert to uint8
    sequence_uint8 = (sequence_np * 255).astype(np.uint8)
    
    # Handle different image formats
    if sequence_uint8.shape[1] == 1:  # Grayscale
        sequence_uint8 = sequence_uint8.squeeze(1)
    elif sequence_uint8.shape[1] == 3:  # RGB
        sequence_uint8 = np.transpose(sequence_uint8, (0, 2, 3, 1))
    
    # Repeat sequence for looping
    if loop_count > 1:
        sequence_uint8 = np.tile(sequence_uint8, (loop_count, 1, 1, 1))
    
    # Save video
    import imageio
    imageio.mimsave(output_path, sequence_uint8, fps=fps)
    print(f"✅ Saved interpolation video to {output_path}")


def analyze_interpolation_smoothness(
    interpolation_sequence: torch.Tensor,
    metric: str = "mse"
) -> Dict[str, float]:
    """Analyze smoothness of interpolation sequence.
    
    Args:
        interpolation_sequence: Sequence of interpolated images
        metric: Metric to use ("mse", "l1", "cosine", "perceptual")
        
    Returns:
        Dictionary with smoothness metrics
    """
    if len(interpolation_sequence) < 2:
        return {"error": "Need at least 2 frames"}
    
    differences = []
    
    for i in range(len(interpolation_sequence) - 1):
        frame1 = interpolation_sequence[i]
        frame2 = interpolation_sequence[i + 1]
        
        if metric == "mse":
            diff = F.mse_loss(frame1, frame2).item()
        elif metric == "l1":
            diff = F.l1_loss(frame1, frame2).item()
        elif metric == "cosine":
            frame1_flat = frame1.flatten()
            frame2_flat = frame2.flatten()
            cosine_sim = F.cosine_similarity(frame1_flat, frame2_flat, dim=0)
            diff = 1 - cosine_sim.item()  # Convert similarity to distance
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        differences.append(diff)
    
    differences = np.array(differences)
    
    return {
        "mean_difference": float(differences.mean()),
        "std_difference": float(differences.std()),
        "max_difference": float(differences.max()),
        "min_difference": float(differences.min()),
        "total_variation": float(differences.sum()),
        "smoothness_score": 1.0 / (1.0 + differences.mean())  # Higher is smoother
    }


# =============================================================================
# Visualization Classes
# =============================================================================

class PlotManager:
    """Manager for consistent plot styling and saving."""
    
    def __init__(self, 
                 style: str = "seaborn-v0_8",
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 300,
                 save_format: str = "png"):
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        self.save_format = save_format
        
        # Apply style
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default if style not available
            plt.style.use('default')
            
    def create_figure(self, 
                     figsize: Optional[Tuple[int, int]] = None,
                     **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Create a figure with consistent styling."""
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi, **kwargs)
        return fig, ax
        
    def save_figure(self, 
                   fig: plt.Figure,
                   filepath: Union[str, Path],
                   close_fig: bool = True,
                   **kwargs):
        """Save figure with consistent settings."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(
            filepath,
            format=self.save_format,
            dpi=self.dpi,
            bbox_inches='tight',
            facecolor='white',
            **kwargs
        )
        
        if close_fig:
            plt.close(fig)


class TrainingVisualizer:
    """Visualizations for training monitoring."""
    
    def __init__(self, plot_manager: Optional[PlotManager] = None):
        self.plot_manager = plot_manager or PlotManager()
        
    def plot_training_curves(self, 
                           metrics: Dict[str, List[float]],
                           title: str = "Training Curves",
                           save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Plot training and validation curves."""
        n_metrics = len(metrics)
        if n_metrics == 0:
            return None
            
        # Create subplots
        fig, axes = plt.subplots(
            nrows=(n_metrics + 1) // 2, 
            ncols=2,
            figsize=(15, 5 * ((n_metrics + 1) // 2)),
            dpi=self.plot_manager.dpi
        )
        
        if n_metrics == 1:
            axes = [axes]
        elif n_metrics > 2:
            axes = axes.flatten()
            
        # Plot each metric
        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx] if n_metrics > 1 else axes
            
            epochs = list(range(1, len(values) + 1))
            ax.plot(epochs, values, linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} Over Time')
            ax.grid(True, alpha=0.3)
            
            # Add trend line if enough data points
            if len(values) > 5:
                z = np.polyfit(epochs, values, 1)
                p = np.poly1d(z)
                ax.plot(epochs, p(epochs), "--", alpha=0.7, color='red')
                
        # Hide unused subplots
        if n_metrics % 2 == 1 and n_metrics > 1:
            axes[-1].set_visible(False)
            
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            self.plot_manager.save_figure(fig, save_path, close_fig=False)
            
        return fig
        
    def plot_loss_components(self,
                           loss_components: Dict[str, List[float]],
                           title: str = "Loss Components",
                           save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Plot breakdown of loss components."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.plot_manager.dpi)
        
        # Plot individual components
        for component, values in loss_components.items():
            epochs = list(range(1, len(values) + 1))
            ax1.plot(epochs, values, label=component, linewidth=2, marker='o', markersize=3)
            
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss Value')
        ax1.set_title('Individual Loss Components')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot stacked components
        epochs = list(range(1, len(next(iter(loss_components.values()))) + 1))
        bottom = np.zeros(len(epochs))
        
        for component, values in loss_components.items():
            ax2.bar(epochs, values, bottom=bottom, label=component, alpha=0.7)
            bottom += np.array(values)
            
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Cumulative Loss')
        ax2.set_title('Stacked Loss Components')
        ax2.legend()
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            self.plot_manager.save_figure(fig, save_path, close_fig=False)
            
        return fig
        
    def plot_learning_rate_schedule(self,
                                  learning_rates: List[float],
                                  title: str = "Learning Rate Schedule",
                                  save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Plot learning rate schedule."""
        fig, ax = self.plot_manager.create_figure(figsize=(10, 6))
        
        epochs = list(range(1, len(learning_rates) + 1))
        ax.plot(epochs, learning_rates, linewidth=2, color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        if save_path:
            self.plot_manager.save_figure(fig, save_path, close_fig=False)
            
        return fig


class ImageVisualizer:
    """Visualizations for image analysis."""
    
    def __init__(self, plot_manager: Optional[PlotManager] = None):
        self.plot_manager = plot_manager or PlotManager()
        
    def plot_image_grid(self,
                       images: List[np.ndarray],
                       titles: Optional[List[str]] = None,
                       grid_size: Optional[Tuple[int, int]] = None,
                       figsize: Optional[Tuple[int, int]] = None,
                       cmap: str = 'gray',
                       save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Plot a grid of images."""
        n_images = len(images)
        
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
        else:
            rows, cols = grid_size
            
        if figsize is None:
            figsize = (4 * cols, 4 * rows)
            
        fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=self.plot_manager.dpi)
        
        if n_images == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        for idx in range(rows * cols):
            ax = axes[idx]
            
            if idx < n_images:
                image = images[idx]
                
                # Handle different image formats
                if len(image.shape) == 3 and image.shape[0] in [1, 3]:  # CHW format
                    if image.shape[0] == 1:
                        image = image[0]
                    else:
                        image = np.transpose(image, (1, 2, 0))
                        
                im = ax.imshow(image, cmap=cmap)
                
                if titles and idx < len(titles):
                    ax.set_title(titles[idx])
                    
                # Add colorbar for single-channel images
                if len(image.shape) == 2:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    
            ax.axis('off')
            
        plt.tight_layout()
        
        if save_path:
            self.plot_manager.save_figure(fig, save_path, close_fig=False)
            
        return fig
        
    def plot_comparison_grid(self,
                           image_sets: Dict[str, List[np.ndarray]],
                           titles: Optional[List[str]] = None,
                           cmap: str = 'gray',
                           save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Plot comparison grid of different image sets."""
        set_names = list(image_sets.keys())
        n_sets = len(set_names)
        n_images = len(image_sets[set_names[0]])
        
        fig, axes = plt.subplots(
            n_sets, n_images, 
            figsize=(4 * n_images, 4 * n_sets),
            dpi=self.plot_manager.dpi
        )
        
        if n_sets == 1:
            axes = [axes]
        if n_images == 1:
            axes = [[ax] for ax in axes]
            
        for set_idx, set_name in enumerate(set_names):
            images = image_sets[set_name]
            
            for img_idx, image in enumerate(images):
                ax = axes[set_idx][img_idx]
                
                # Handle different image formats
                if len(image.shape) == 3 and image.shape[0] in [1, 3]:
                    if image.shape[0] == 1:
                        image = image[0]
                    else:
                        image = np.transpose(image, (1, 2, 0))
                        
                im = ax.imshow(image, cmap=cmap)
                
                # Set titles
                if set_idx == 0 and titles and img_idx < len(titles):
                    ax.set_title(titles[img_idx])
                    
                if img_idx == 0:
                    ax.set_ylabel(set_name, fontsize=12, fontweight='bold')
                    
                ax.axis('off')
                
                # Add colorbar for single-channel images
                if len(image.shape) == 2:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    
        plt.tight_layout()
        
        if save_path:
            self.plot_manager.save_figure(fig, save_path, close_fig=False)
            
        return fig
        
    def plot_histogram_comparison(self,
                                images: Dict[str, np.ndarray],
                                bins: int = 50,
                                title: str = "Intensity Histograms",
                                save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Plot histogram comparison of images."""
        fig, ax = self.plot_manager.create_figure(figsize=(10, 6))
        
        for name, image in images.items():
            ax.hist(image.flatten(), bins=bins, alpha=0.7, label=name, density=True)
            
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            self.plot_manager.save_figure(fig, save_path, close_fig=False)
            
        return fig


class MetricsVisualizer:
    """Visualizations for evaluation metrics."""
    
    def __init__(self, plot_manager: Optional[PlotManager] = None):
        self.plot_manager = plot_manager or PlotManager()
        
    def plot_metrics_comparison(self,
                              metrics_data: Dict[str, Dict[str, float]],
                              title: str = "Metrics Comparison",
                              save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Plot comparison of metrics across different methods."""
        methods = list(metrics_data.keys())
        metric_names = list(next(iter(metrics_data.values())).keys())
        
        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6), dpi=self.plot_manager.dpi)
        
        if n_metrics == 1:
            axes = [axes]
            
        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            
            values = [metrics_data[method][metric_name] for method in methods]
            
            bars = ax.bar(methods, values, alpha=0.7)
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
                       
            # Rotate x-axis labels if needed
            if len(max(methods, key=len)) > 8:
                ax.tick_params(axis='x', rotation=45)
                
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            self.plot_manager.save_figure(fig, save_path, close_fig=False)
            
        return fig
        
    def plot_metrics_heatmap(self,
                           metrics_data: Dict[str, Dict[str, float]],
                           title: str = "Metrics Heatmap",
                           save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Plot metrics as a heatmap."""
        # Convert to matrix format
        methods = list(metrics_data.keys())
        metric_names = list(next(iter(metrics_data.values())).keys())
        
        matrix = np.array([[metrics_data[method][metric] for metric in metric_names] 
                          for method in methods])
        
        fig, ax = self.plot_manager.create_figure(figsize=(len(metric_names) * 2, len(methods) * 0.8))
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(metric_names)))
        ax.set_yticks(range(len(methods)))
        ax.set_xticklabels([name.replace('_', ' ').title() for name in metric_names])
        ax.set_yticklabels(methods)
        
        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(metric_names)):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                             ha="center", va="center", color="white")
                             
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        
        if save_path:
            self.plot_manager.save_figure(fig, save_path, close_fig=False)
            
        return fig


class SamplingVisualizer:
    """Visualizations for sampling process."""
    
    def __init__(self, plot_manager: Optional[PlotManager] = None):
        self.plot_manager = plot_manager or PlotManager()
        
    def plot_sampling_process(self,
                            intermediate_samples: List[np.ndarray],
                            timesteps: Optional[List[int]] = None,
                            title: str = "Sampling Process",
                            save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """Plot the denoising process during sampling."""
        n_samples = len(intermediate_samples)
        
        if timesteps is None:
            timesteps = list(range(n_samples))
            
        # Show subset of samples if too many
        if n_samples > 10:
            indices = np.linspace(0, n_samples - 1, 10, dtype=int)
            samples_to_show = [intermediate_samples[i] for i in indices]
            timesteps_to_show = [timesteps[i] for i in indices]
        else:
            samples_to_show = intermediate_samples
            timesteps_to_show = timesteps
            
        fig, axes = plt.subplots(
            2, len(samples_to_show) // 2 + len(samples_to_show) % 2,
            figsize=(3 * len(samples_to_show), 6),
            dpi=self.plot_manager.dpi
        )
        
        axes = axes.flatten()
        
        for idx, (sample, t) in enumerate(zip(samples_to_show, timesteps_to_show)):
            ax = axes[idx]
            
            # Handle different formats
            if len(sample.shape) == 3 and sample.shape[0] == 1:
                sample = sample[0]
                
            ax.imshow(sample, cmap='gray')
            ax.set_title(f't = {t}')
            ax.axis('off')
            
        # Hide unused subplots
        for idx in range(len(samples_to_show), len(axes)):
            axes[idx].set_visible(False)
            
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            self.plot_manager.save_figure(fig, save_path, close_fig=False)
            
        return fig


class WandBLogger:
    """Weights & Biases logging utilities."""
    
    def __init__(self, project_name: str, run_name: Optional[str] = None):
        if not WANDB_AVAILABLE:
            warnings.warn("wandb not available, logging will be skipped")
            self.enabled = False
            return
            
        self.enabled = True
        wandb.init(project=project_name, name=run_name)
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to wandb."""
        if self.enabled:
            wandb.log(metrics, step=step)
            
    def log_images(self, images: Dict[str, np.ndarray], step: Optional[int] = None):
        """Log images to wandb."""
        if not self.enabled:
            return
            
        wandb_images = {}
        for name, image in images.items():
            # Convert to wandb Image format
            if len(image.shape) == 3 and image.shape[0] == 1:
                image = image[0]
            wandb_images[name] = wandb.Image(image)
            
        wandb.log(wandb_images, step=step)
        
    def log_figure(self, fig: plt.Figure, name: str, step: Optional[int] = None):
        """Log matplotlib figure to wandb."""
        if self.enabled:
            wandb.log({name: wandb.Image(fig)}, step=step)


class InterpolationPipeline:
    """Pipeline for creating and analyzing interpolation sequences."""
    
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize interpolation pipeline.
        
        Args:
            model: Optional model for advanced interpolations
            device: Device to run on
        """
        self.model = model
        self.device = device
        
    def create_sequence(
        self,
        start_image: torch.Tensor,
        end_image: torch.Tensor,
        num_steps: int = 10,
        method: str = "linear",
        **kwargs
    ) -> torch.Tensor:
        """Create interpolation sequence.
        
        Args:
            start_image: Starting image
            end_image: Ending image
            num_steps: Number of steps
            method: Interpolation method
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Interpolated sequence
        """
        start_image = start_image.to(self.device)
        end_image = end_image.to(self.device)
        
        if method == "linear":
            return linear_interpolation(start_image, end_image, num_steps, **kwargs)
        elif method == "spherical":
            return spherical_interpolation(start_image, end_image, num_steps, **kwargs)
        elif method == "diffusion":
            if self.model is None:
                raise ValueError("Model required for diffusion interpolation")
            return diffusion_interpolation(
                self.model, start_image, end_image, num_steps, **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def create_morphing_sequence(
        self,
        images: List[torch.Tensor],
        steps_between: int = 10,
        method: str = "linear",
        **kwargs
    ) -> torch.Tensor:
        """Create morphing sequence between multiple images."""
        images = [img.to(self.device) for img in images]
        return morphing_sequence(images, steps_between, method, **kwargs)
    
    def analyze_smoothness(
        self,
        sequence: torch.Tensor,
        metric: str = "mse"
    ) -> Dict[str, float]:
        """Analyze smoothness of sequence."""
        return analyze_interpolation_smoothness(sequence, metric)
    
    def save_video(
        self,
        sequence: torch.Tensor,
        output_path: Union[str, Path],
        **kwargs
    ):
        """Save sequence as video."""
        save_interpolation_video(sequence, output_path, **kwargs)


# Convenience functions
def close_all_figures():
    """Close all matplotlib figures to free memory."""
    plt.close('all')


def save_figure_safely(fig: plt.Figure, 
                      filepath: Union[str, Path],
                      **kwargs):
    """Save figure with error handling."""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, **kwargs)
    except Exception as e:
        warnings.warn(f"Failed to save figure to {filepath}: {e}")


__all__ = [
    # Interpolation utilities
    "linear_interpolation",
    "spherical_interpolation",
    "noise_interpolation",
    "latent_interpolation",
    "diffusion_interpolation",
    "morphing_sequence",
    "create_interpolation_grid",
    "save_interpolation_video",
    "analyze_interpolation_smoothness",
    "InterpolationPipeline",
    
    # Visualization classes
    "PlotManager",
    "TrainingVisualizer",
    "ImageVisualizer",
    "MetricsVisualizer", 
    "SamplingVisualizer",
    "WandBLogger",
    
    # Convenience functions
    "close_all_figures",
    "save_figure_safely",
]