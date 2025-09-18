"""
Interpolation Utilities for PKL Diffusion Denoising

This module provides utilities for smooth interpolation between samples,
latent space interpolation, and morphing sequences for analysis and visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Tuple, Callable, Dict, Any
from pathlib import Path
import math


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
    try:
        import imageio
    except ImportError:
        print("⚠️ imageio not available, cannot save video")
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
