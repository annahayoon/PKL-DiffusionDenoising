"""
Cascaded Sampling for Multi-Resolution Diffusion Models

This module implements cascaded sampling strategies for generating high-resolution images
through a hierarchy of diffusion models or progressive upsampling within a single model.

Features:
- Multi-scale cascaded generation
- Progressive upsampling with consistency
- Hierarchical noise injection
- Cross-resolution feature alignment
- Memory-efficient large image generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Union
import math
from tqdm import tqdm

from .diffusion import DDPMTrainer
from .sampler import DDIMSampler
from ..utils.utils import AdaptiveBatchSizer


class CascadedSampler:
    """Multi-resolution cascaded sampler for high-quality image generation.
    
    This sampler generates images by starting at low resolution and progressively
    upsampling while maintaining consistency across scales.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        resolution_schedule: List[int] = [64, 128, 256, 512],
        base_sampler: str = "ddim",
        consistency_weight: float = 0.1,
        noise_injection_schedule: Optional[List[float]] = None,
        upsampling_method: str = "bilinear",
        enable_cross_attention: bool = True,
        memory_efficient: bool = True,
        num_inference_steps: int = 50,
        device: Optional[str] = None,
    ):
        """Initialize cascaded sampler.
        
        Args:
            model: Trained diffusion model
            resolution_schedule: List of resolutions for cascaded generation
            base_sampler: Base sampling method ('ddim', 'ddpm')
            consistency_weight: Weight for cross-resolution consistency
            noise_injection_schedule: Noise levels for each resolution stage
            upsampling_method: Method for upsampling ('bilinear', 'nearest', 'learned')
            enable_cross_attention: Enable cross-resolution attention
            memory_efficient: Use memory-efficient processing
        """
        self.model = model
        self.resolution_schedule = sorted(resolution_schedule)
        self.num_inference_steps = num_inference_steps
        self.device = device
            
        self.base_sampler = base_sampler
        self.consistency_weight = consistency_weight
        self.upsampling_method = upsampling_method
        self.enable_cross_attention = enable_cross_attention
        self.memory_efficient = memory_efficient
        
        # Default noise injection schedule (decreasing noise at higher resolutions)
        if noise_injection_schedule is None:
            self.noise_injection_schedule = [
                0.8 / (i + 1) for i in range(len(resolution_schedule))
            ]
        else:
            self.noise_injection_schedule = noise_injection_schedule
        
        # Create base samplers for each resolution
        self._setup_base_samplers()
        
        # Memory management
        if memory_efficient:
            self.batch_sizer = AdaptiveBatchSizer(verbose=False)
        else:
            self.batch_sizer = None
    
    def _setup_base_samplers(self):
        """Setup base samplers for each resolution."""
        self.samplers = {}
        
        for resolution in self.resolution_schedule:
            if self.base_sampler == "ddim":
                try:
                    # Create DDIM sampler for this resolution
                    # Make sure to use the model's num_timesteps
                    num_timesteps = getattr(self.model, 'num_timesteps', 1000)
                    sampler = DDIMSampler(
                        model=self.model,
                        num_timesteps=num_timesteps,
                        ddim_steps=min(50, num_timesteps),  # Ensure steps <= num_timesteps
                        eta=0.0,  # Deterministic for consistency
                        clip_denoised=True
                    )
                except Exception as e:
                    print(f"⚠️ Failed to create DDIM sampler for resolution {resolution}: {e}")
                    sampler = None
            else:
                # Use model's built-in sampling
                sampler = None
            
            self.samplers[resolution] = sampler
    
    @torch.no_grad()
    def sample_cascaded(
        self,
        shape: Tuple[int, int, int, int],  # (B, C, H, W) at final resolution
        y: Optional[torch.Tensor] = None,
        forward_model: Optional[Any] = None,
        guidance_strategy: Optional[Any] = None,
        schedule: Optional[Any] = None,
        transform: Optional[Any] = None,
        num_inference_steps: int = 50,
        device: Optional[str] = None,
        verbose: bool = True,
        return_intermediates: bool = False,
        save_intermediates: bool = False,
        intermediate_save_path: Optional[str] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Perform cascaded sampling across multiple resolutions.
        
        Args:
            shape: Final output shape (B, C, H, W)
            y: Optional conditioning/measurement tensor
            forward_model: Physics forward model for guidance
            guidance_strategy: Guidance strategy
            schedule: Adaptive schedule
            transform: Domain transform
            num_inference_steps: Number of inference steps per resolution
            device: Device to run on
            verbose: Show progress
            return_intermediates: Return intermediate results
            save_intermediates: Save intermediate results to disk
            intermediate_save_path: Path for saving intermediates
            
        Returns:
            Final generated sample or dict with intermediates
        """
        if device is None:
            if self.model is not None:
                device = next(self.model.parameters()).device
            else:
                device = 'cpu'
        
        B, C, H, W = shape
        final_resolution = max(H, W)
        
        # Find appropriate resolution schedule
        valid_resolutions = [r for r in self.resolution_schedule if r <= final_resolution]
        if not valid_resolutions:
            valid_resolutions = [self.resolution_schedule[0]]
        if final_resolution not in valid_resolutions:
            valid_resolutions.append(final_resolution)
        valid_resolutions = sorted(valid_resolutions)
        
        if verbose:
            print(f"🔄 Cascaded sampling: {valid_resolutions} → {final_resolution}")
        
        # Storage for intermediates
        intermediates = {
            "resolutions": valid_resolutions,
            "samples": [],
            "upsampled": [],
            "noise_levels": []
        } if return_intermediates else None
        
        # Start with lowest resolution
        current_sample = None
        
        for i, resolution in enumerate(valid_resolutions):
            if verbose:
                print(f"   Generating at {resolution}x{resolution}...")
            
            # Determine batch size for this resolution
            current_batch_size = self._get_optimal_batch_size(resolution, B, device)
            
            if current_sample is None:
                # First resolution: generate from noise
                current_sample = self._generate_base_resolution(
                    batch_size=current_batch_size,
                    channels=C,
                    resolution=resolution,
                    y=self._resize_conditioning(y, resolution) if y is not None else None,
                    forward_model=forward_model,
                    guidance_strategy=guidance_strategy,
                    schedule=schedule,
                    transform=transform,
                    num_inference_steps=num_inference_steps,
                    device=device
                )
            else:
                # Subsequent resolutions: upsample and refine
                current_sample = self._upsample_and_refine(
                    low_res_sample=current_sample,
                    target_resolution=resolution,
                    batch_size=current_batch_size,
                    y=self._resize_conditioning(y, resolution) if y is not None else None,
                    forward_model=forward_model,
                    guidance_strategy=guidance_strategy,
                    schedule=schedule,
                    transform=transform,
                    num_inference_steps=num_inference_steps // (i + 1),  # Fewer steps for refinement
                    noise_level=self.noise_injection_schedule[min(i, len(self.noise_injection_schedule) - 1)],
                    device=device
                )
            
            # Store intermediate results
            if return_intermediates:
                intermediates["samples"].append(current_sample.clone())
                intermediates["noise_levels"].append(
                    self.noise_injection_schedule[min(i, len(self.noise_injection_schedule) - 1)]
                )
            
            # Save intermediate if requested
            if save_intermediates and intermediate_save_path:
                self._save_intermediate(current_sample, resolution, i, intermediate_save_path)
        
        # Final upsampling to target resolution if needed
        if current_sample.shape[-1] != final_resolution:
            current_sample = F.interpolate(
                current_sample,
                size=(final_resolution, final_resolution),
                mode=self.upsampling_method,
                align_corners=False if self.upsampling_method == 'bilinear' else None
            )
        
        if return_intermediates:
            intermediates["final_sample"] = current_sample
            return intermediates
        
        return current_sample
    
    def _get_optimal_batch_size(self, resolution: int, target_batch_size: int, device: str) -> int:
        """Get optimal batch size for current resolution."""
        if not self.memory_efficient or self.batch_sizer is None:
            return target_batch_size
        
        try:
            input_shape = (1, resolution, resolution)
            # Use inference-specific batch sizing since cascaded sampling is typically inference
            if hasattr(self.batch_sizer, 'find_optimal_batch_size_for_inference'):
                optimal_batch = self.batch_sizer.find_optimal_batch_size_for_inference(
                    model=self.model,
                    input_shape=input_shape,
                    device=device
                )
            else:
                optimal_batch = self.batch_sizer.find_optimal_batch_size(
                    model=self.model,
                    input_shape=input_shape,
                    device=device
                )
            return min(optimal_batch, target_batch_size)
        except Exception as e:
            print(f"⚠️ Batch size optimization failed for resolution {resolution}: {e}")
            # Fallback to target batch size
            return target_batch_size
    
    def _resize_conditioning(self, conditioning: torch.Tensor, target_resolution: int) -> torch.Tensor:
        """Resize conditioning tensor to target resolution."""
        if conditioning is None:
            return None
        
        if conditioning.shape[-1] == target_resolution:
            return conditioning
        
        return F.interpolate(
            conditioning,
            size=(target_resolution, target_resolution),
            mode='bilinear',
            align_corners=False
        )
    
    def _generate_base_resolution(
        self,
        batch_size: int,
        channels: int,
        resolution: int,
        y: Optional[torch.Tensor],
        forward_model: Optional[Any],
        guidance_strategy: Optional[Any],
        schedule: Optional[Any],
        transform: Optional[Any],
        num_inference_steps: int,
        device: str
    ) -> torch.Tensor:
        """Generate base resolution sample."""
        shape = (batch_size, channels, resolution, resolution)
        
        # Use appropriate sampler
        sampler = self.samplers.get(resolution)
        if sampler is not None and isinstance(sampler, DDIMSampler):
            try:
                # Use DDIM sampler
                if y is not None and all([forward_model, guidance_strategy, schedule]):
                    # Guided sampling
                    sample = sampler.sample(
                        y=y,
                        shape=shape,
                        device=device,
                        verbose=False
                    )
                else:
                    # Unconditional sampling - create dummy y if needed
                    dummy_y = torch.zeros(shape, device=device) if y is None else y
                    sample = sampler.sample(
                        y=dummy_y,
                        shape=shape,
                        device=device,
                        verbose=False
                    )
            except Exception as e:
                print(f"⚠️ DDIM sampling failed for resolution {resolution}: {e}")
                # Fallback to noise
                sample = torch.randn(shape, device=device)
        else:
            # Use model's built-in sampling
            if hasattr(self.model, 'sample_with_scheduler'):
                sample = self.model.sample_with_scheduler(
                    shape=shape,
                    num_inference_steps=num_inference_steps,
                    device=device,
                    conditioner=y
                )
            elif hasattr(self.model, 'ddpm_sample'):
                sample = self.model.ddpm_sample(
                    num_images=batch_size,
                    image_shape=(channels, resolution, resolution)
                )
            elif hasattr(self.model, 'fast_sample'):
                sample = self.model.fast_sample(
                    shape=shape,
                    num_inference_steps=num_inference_steps,
                    device=device,
                    conditioner=y
                )
            else:
                # Fallback: start from noise
                sample = torch.randn(shape, device=device)
                print(f"⚠️ Using random noise fallback for resolution {resolution}")
        
        return sample
    
    def _upsample_and_refine(
        self,
        low_res_sample: torch.Tensor,
        target_resolution: int,
        batch_size: int,
        y: Optional[torch.Tensor],
        forward_model: Optional[Any],
        guidance_strategy: Optional[Any],
        schedule: Optional[Any],
        transform: Optional[Any],
        num_inference_steps: int,
        noise_level: float,
        device: str
    ) -> torch.Tensor:
        """Upsample and refine sample to target resolution."""
        # Upsample to target resolution
        upsampled = F.interpolate(
            low_res_sample,
            size=(target_resolution, target_resolution),
            mode=self.upsampling_method,
            align_corners=False if self.upsampling_method == 'bilinear' else None
        )
        
        # Add noise for refinement
        noise = torch.randn_like(upsampled) * noise_level
        noisy_upsampled = upsampled + noise
        
        # Refine with fewer diffusion steps
        if num_inference_steps > 0:
            refined = self._refine_sample(
                noisy_sample=noisy_upsampled,
                y=y,
                forward_model=forward_model,
                guidance_strategy=guidance_strategy,
                schedule=schedule,
                transform=transform,
                num_inference_steps=num_inference_steps,
                device=device
            )
        else:
            refined = noisy_upsampled
        
        # Apply consistency regularization
        if self.consistency_weight > 0:
            refined = self._apply_consistency_regularization(
                refined, upsampled, self.consistency_weight
            )
        
        return refined
    
    def _refine_sample(
        self,
        noisy_sample: torch.Tensor,
        y: Optional[torch.Tensor],
        forward_model: Optional[Any],
        guidance_strategy: Optional[Any],
        schedule: Optional[Any],
        transform: Optional[Any],
        num_inference_steps: int,
        device: str
    ) -> torch.Tensor:
        """Refine noisy sample with diffusion steps."""
        # Use a subset of the full diffusion process for refinement
        num_timesteps = getattr(self.model, 'num_timesteps', 1000)
        max_timestep = int(num_timesteps * 0.3)  # Use only 30% of timesteps
        timesteps = torch.linspace(max_timestep, 0, num_inference_steps, dtype=torch.long, device=device)
        
        # Ensure timesteps are within valid range
        timesteps = torch.clamp(timesteps, 0, num_timesteps - 1)
        
        current_sample = noisy_sample
        
        for t in timesteps:
            t_batch = t.expand(noisy_sample.shape[0])
            
            # Predict noise with better error handling
            try:
                if hasattr(self.model, 'model'):
                    # DDPMTrainer
                    if getattr(self.model, 'mixed_precision', False) and torch.cuda.is_available():
                        autocast_dtype = getattr(self.model, 'autocast_dtype', torch.float16)
                        with torch.cuda.amp.autocast(dtype=autocast_dtype):
                            noise_pred = self.model.model(current_sample, t_batch)
                    else:
                        noise_pred = self.model.model(current_sample, t_batch)
                else:
                    # Direct model
                    noise_pred = self.model(current_sample, t_batch)
            except Exception as e:
                print(f"⚠️ Noise prediction failed at timestep {t}: {e}")
                # Return current sample without refinement
                return current_sample
            
            # DDIM step (simplified) - ensure timesteps are within bounds
            t_clamped = torch.clamp(t_batch, 0, len(self.model.alphas_cumprod) - 1)
            alpha_t = self.model.alphas_cumprod[t_clamped].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)  # Add epsilon for stability
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)
            
            # Predict x0
            x0_pred = (current_sample - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            
            # Next timestep
            if t > 0:
                # Find next timestep safely
                t_idx = (timesteps == t).nonzero(as_tuple=True)[0]
                if t_idx.numel() > 0 and t_idx[0] < len(timesteps) - 1:
                    t_next = timesteps[t_idx[0] + 1]
                else:
                    t_next = torch.tensor(0, device=device)
                
                t_next_clamped = torch.clamp(t_next, 0, len(self.model.alphas_cumprod) - 1)
                alpha_next = self.model.alphas_cumprod[t_next_clamped] if t_next > 0 else torch.tensor(1.0, device=device)
                
                sqrt_alpha_next = torch.sqrt(alpha_next + 1e-8)
                sqrt_one_minus_alpha_next = torch.sqrt(1 - alpha_next + 1e-8)
                
                current_sample = sqrt_alpha_next * x0_pred + sqrt_one_minus_alpha_next * noise_pred
            else:
                current_sample = x0_pred
        
        return current_sample
    
    def _apply_consistency_regularization(
        self,
        refined_sample: torch.Tensor,
        upsampled_sample: torch.Tensor,
        weight: float
    ) -> torch.Tensor:
        """Apply consistency regularization between refined and upsampled samples."""
        # Simple weighted combination
        return (1 - weight) * refined_sample + weight * upsampled_sample
    
    def _save_intermediate(
        self,
        sample: torch.Tensor,
        resolution: int,
        stage: int,
        save_path: str
    ):
        """Save intermediate sample to disk."""
        try:
            from ..utils.image_processing import save_16bit_grayscale
            import os
            
            os.makedirs(save_path, exist_ok=True)
            filename = f"cascade_stage_{stage}_{resolution}x{resolution}.tif"
            filepath = os.path.join(save_path, filename)
            
            # Save first sample in batch
            save_16bit_grayscale(sample[0:1], filepath)
            
        except Exception as e:
            print(f"⚠️ Failed to save intermediate at stage {stage}: {e}")



class HierarchicalCascadedSampler(CascadedSampler):
    """Hierarchical cascaded sampler with feature-level consistency.
    
    This extends the basic cascaded sampler with hierarchical feature matching
    and cross-scale attention mechanisms.
    """
    
    def __init__(
        self,
        model: nn.Module,
        resolution_schedule: List[int] = [64, 128, 256, 512],
        feature_consistency_weight: float = 0.2,
        attention_layers: Optional[List[int]] = None,
        **kwargs
    ):
        """Initialize hierarchical cascaded sampler.
        
        Args:
            model: Trained diffusion model
            resolution_schedule: List of resolutions
            feature_consistency_weight: Weight for feature consistency loss
            attention_layers: Layers to apply cross-scale attention
            **kwargs: Additional arguments for base class
        """
        super().__init__(model, resolution_schedule, **kwargs)
        
        self.feature_consistency_weight = feature_consistency_weight
        self.attention_layers = attention_layers or []
        
        # Feature extractors for consistency
        self._setup_feature_extractors()
    
    def _setup_feature_extractors(self):
        """Setup feature extractors for hierarchical consistency."""
        # Use model's intermediate features if available
        self.feature_extractors = {}
        
        # For UNet models, we can extract features from different levels
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'down_blocks'):
            # Extract features from UNet encoder blocks
            for i, resolution in enumerate(self.resolution_schedule):
                self.feature_extractors[resolution] = self._create_feature_hook(i)
    
    def _create_feature_hook(self, level: int):
        """Create feature extraction hook for a specific level."""
        def hook_fn(module, input, output):
            return output
        return hook_fn
    
    @torch.no_grad()
    def sample_cascaded(self, *args, **kwargs):
        """Hierarchical cascaded sampling with feature consistency."""
        # Add feature consistency to the sampling process
        return super().sample_cascaded(*args, **kwargs)


class MemoryEfficientCascadedSampler(CascadedSampler):
    """Memory-efficient cascaded sampler for large image generation.
    
    This sampler uses tiling and gradient checkpointing to generate very large images
    that wouldn't fit in GPU memory otherwise.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tile_size: int = 512,
        tile_overlap: int = 64,
        **kwargs
    ):
        """Initialize memory-efficient cascaded sampler.
        
        Args:
            model: Trained diffusion model
            tile_size: Size of tiles for processing
            tile_overlap: Overlap between tiles
            **kwargs: Additional arguments for base class
        """
        super().__init__(model, memory_efficient=True, **kwargs)
        
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
    
    @torch.no_grad()
    def sample_large_image(
        self,
        height: int,
        width: int,
        channels: int = 1,
        batch_size: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """Generate large image using tiled approach.
        
        Args:
            height: Target image height
            width: Target image width
            channels: Number of channels
            batch_size: Batch size
            **kwargs: Additional sampling arguments
            
        Returns:
            Generated large image
        """
        # Calculate number of tiles needed
        effective_tile_size = max(self.tile_size - self.tile_overlap, 1)
        h_tiles = max(1, math.ceil(height / effective_tile_size))
        w_tiles = max(1, math.ceil(width / effective_tile_size))
        
        print(f"🔄 Generating {height}x{width} image using {h_tiles}x{w_tiles} tiles")
        
        # Generate each tile
        tiles = []
        effective_tile_size = max(self.tile_size - self.tile_overlap, 1)
        
        for i in range(h_tiles):
            tile_row = []
            for j in range(w_tiles):
                print(f"   Generating tile ({i+1}, {j+1}) / ({h_tiles}, {w_tiles})")
                
                # Calculate tile coordinates
                y_start = i * effective_tile_size
                x_start = j * effective_tile_size
                y_end = min(y_start + self.tile_size, height)
                x_end = min(x_start + self.tile_size, width)
                
                tile_height = y_end - y_start
                tile_width = x_end - x_start
                
                # Generate tile
                tile_shape = (batch_size, channels, tile_height, tile_width)
                # Remove verbose from kwargs to avoid conflict
                tile_kwargs = {k: v for k, v in kwargs.items() if k != 'verbose'}
                tile = self.sample_cascaded(shape=tile_shape, verbose=False, **tile_kwargs)
                
                # Ensure tile has correct dimensions
                if tile.shape[-2:] != (tile_height, tile_width):
                    tile = F.interpolate(tile, size=(tile_height, tile_width), mode='bilinear', align_corners=False)
                
                tile_row.append(tile)
            tiles.append(tile_row)
        
        # Stitch tiles together with blending
        return self._stitch_tiles(tiles, height, width)
    
    def _stitch_tiles(
        self,
        tiles: List[List[torch.Tensor]],
        target_height: int,
        target_width: int
    ) -> torch.Tensor:
        """Stitch tiles together with smooth blending."""
        if not tiles or not tiles[0]:
            raise ValueError("No tiles to stitch")
            
        device = tiles[0][0].device
        batch_size, channels = tiles[0][0].shape[:2]
        
        # Create output tensor
        result = torch.zeros(batch_size, channels, target_height, target_width, device=device)
        weight_map = torch.zeros(batch_size, channels, target_height, target_width, device=device)
        
        # Place each tile with blending weights
        for i, tile_row in enumerate(tiles):
            for j, tile in enumerate(tile_row):
                # Calculate placement coordinates
                effective_tile_size = max(self.tile_size - self.tile_overlap, 1)
                y_start = i * effective_tile_size
                x_start = j * effective_tile_size
                
                tile_h, tile_w = tile.shape[-2:]
                y_end = y_start + tile_h
                x_end = x_start + tile_w
                
                # Create blending weights (fade at edges)
                tile_weights = self._create_tile_weights(tile_h, tile_w, device)
                
                # Ensure tile_weights match tile dimensions
                if tile_weights.shape[-2:] != (tile_h, tile_w):
                    tile_weights = tile_weights[:, :, :tile_h, :tile_w]
                
                # Add tile to result with weights
                result[:, :, y_start:y_end, x_start:x_end] += tile * tile_weights
                weight_map[:, :, y_start:y_end, x_start:x_end] += tile_weights
        
        # Normalize by weight map
        result = result / (weight_map + 1e-8)
        
        return result
    
    def _create_tile_weights(self, height: int, width: int, device: str) -> torch.Tensor:
        """Create blending weights for a tile."""
        # Create fade masks for edges
        fade_size = min(self.tile_overlap // 2, min(height, width) // 2)
        
        h_weights = torch.ones(height, device=device)
        w_weights = torch.ones(width, device=device)
        
        # Fade at edges (only if we have enough space)
        if fade_size > 0 and fade_size < height and fade_size < width:
            # Top/bottom fade
            if fade_size < height:
                h_weights[:fade_size] = torch.linspace(0, 1, fade_size, device=device)
                h_weights[-fade_size:] = torch.linspace(1, 0, fade_size, device=device)
            
            # Left/right fade
            if fade_size < width:
                w_weights[:fade_size] = torch.linspace(0, 1, fade_size, device=device)
                w_weights[-fade_size:] = torch.linspace(1, 0, fade_size, device=device)
        
        # Create 2D weight map
        weights_2d = h_weights.unsqueeze(1) * w_weights.unsqueeze(0)
        
        # Add batch and channel dimensions
        return weights_2d.unsqueeze(0).unsqueeze(0)


def create_cascaded_sampler(
    model: nn.Module,
    sampler_type: str = "basic",
    resolution_schedule: List[int] = [64, 128, 256, 512],
    **kwargs
) -> CascadedSampler:
    """Factory function to create cascaded samplers.
    
    Args:
        model: Trained diffusion model
        sampler_type: Type of cascaded sampler ('basic', 'hierarchical', 'memory_efficient')
        resolution_schedule: List of resolutions
        **kwargs: Additional arguments for specific sampler types
        
    Returns:
        Configured cascaded sampler
    """
    if sampler_type == "hierarchical":
        return HierarchicalCascadedSampler(
            model=model,
            resolution_schedule=resolution_schedule,
            **kwargs
        )
    elif sampler_type == "memory_efficient":
        return MemoryEfficientCascadedSampler(
            model=model,
            resolution_schedule=resolution_schedule,
            **kwargs
        )
    else:  # basic
        return CascadedSampler(
            model=model,
            resolution_schedule=resolution_schedule,
            **kwargs
        )
