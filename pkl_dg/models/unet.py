from typing import Any, Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import normalize_16bit_to_model_input, denormalize_model_output_to_16bit
from .nn import (
    NNDefaults,
    ResNetBlock,
    SelfAttention2D,
    TimeEmbeddingMLP,
    Downsample,
    Upsample,
    sinusoidal_position_embeddings,
    ZeroConv2d,
    DoubleConvBlock,
    get_normalization,
)




class UNet(nn.Module):
    """Diffusion UNet for microscopy image denoising research.
    
    Pure diffusion model with timestep conditioning for self-supervised learning.
    
    Designed for:
    - 16-bit grayscale microscopy images (0-65535 range)
    - Physics-guided diffusion denoising
    - Single channel input/output (grayscale)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Parse config with standardized defaults
        self.in_channels = config.get("in_channels", 1)
        self.out_channels = config.get("out_channels", 1)
        self.sample_size = config.get("sample_size", 256)
        self.block_out_channels = config.get("block_out_channels", NNDefaults.UNET_CHANNELS)
        self.layers_per_block = config.get("layers_per_block", NNDefaults.UNET_LAYERS_PER_BLOCK)
        self.attention_resolutions = config.get("attention_resolutions", [16, 32])
        self.num_attention_heads = config.get("num_attention_heads", 8)
        self.dropout = config.get("dropout", NNDefaults.DROPOUT_RATE)
        
        # Pure diffusion mode only
        self.mode = "diffusion"
        self.use_time_conditioning = True
        
        # Input channels (no PSF conditioning)
        actual_in_channels = self.in_channels
        
        # Time embedding (required for diffusion)
        self.time_embedding = TimeEmbeddingMLP(
            time_dim=self.block_out_channels[0],
            emb_dim=self.block_out_channels[0] * NNDefaults.TIME_EMB_MULT,
        )
        time_emb_dim = self.time_embedding.mlp[-1].out_features
        
        # Input projection
        self.conv_in = nn.Conv2d(actual_in_channels, self.block_out_channels[0], 3, padding=1)
        
        # Down path
        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        in_ch = self.block_out_channels[0]
        
        # Diffusion mode: ResNet blocks with time conditioning
        for i, out_ch in enumerate(self.block_out_channels[1:]):
            # ResNet blocks using standardized component
            blocks = nn.ModuleList([
                ResNetBlock(
                    in_ch if j == 0 else out_ch, 
                    out_ch, 
                    time_emb_dim=time_emb_dim,
                    dropout=self.dropout
                )
                for j in range(self.layers_per_block)
            ])
            self.down_blocks.append(blocks)
            
            # Attention using standardized component
            current_resolution = self.sample_size // (2 ** (i + 1))
            if current_resolution in self.attention_resolutions:
                self.down_attentions.append(SelfAttention2D(out_ch, self.num_attention_heads))
            else:
                self.down_attentions.append(nn.Identity())
            
            # Downsampling using standardized component
            self.down_samples.append(Downsample(out_ch, method="conv"))
            in_ch = out_ch
            
        
        # Middle blocks - ResNet blocks with time conditioning
        mid_ch = self.block_out_channels[-1]
        self.mid_block1 = ResNetBlock(mid_ch, mid_ch, time_emb_dim=time_emb_dim, dropout=self.dropout)
        self.mid_attn = SelfAttention2D(mid_ch, self.num_attention_heads)
        self.mid_block2 = ResNetBlock(mid_ch, mid_ch, time_emb_dim=time_emb_dim, dropout=self.dropout)
        
        # Up path
        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        reversed_channels = list(reversed(self.block_out_channels))
        
        if self.mode == "supervised":
            # Supervised mode: Use DoubleConv decoder blocks
            for i, out_ch in enumerate(reversed_channels[1:]):  # Skip first (bottleneck)
                in_ch = reversed_channels[i]
                
                # Upsampling
                self.up_samples.append(Upsample(in_ch, method="conv_transpose"))
                
                # Decoder block (handles skip connection concatenation)
                decoder_in_ch = in_ch + out_ch  # upsampled + skip connection
                decoder = DoubleConvBlock(decoder_in_ch, out_ch, dropout=self.dropout)
                self.up_blocks.append(decoder)
                
                # Attention at specified resolutions
                current_resolution = self.sample_size // (2 ** (len(reversed_channels) - 2 - i))
                if current_resolution in self.attention_resolutions:
                    self.up_attentions.append(SelfAttention2D(out_ch, self.num_attention_heads))
                else:
                    self.up_attentions.append(nn.Identity())
        else:
            # Diffusion mode: Use ResNet blocks with time conditioning
            for i, out_ch in enumerate(reversed_channels[1:]):
                in_ch = reversed_channels[i]
                
                # Upsampling using standardized component
                self.up_samples.append(Upsample(in_ch, method="conv_transpose"))
                
                # ResNet blocks - first block handles concatenated input
                blocks = nn.ModuleList()
                for j in range(self.layers_per_block + 1):  # +1 for up blocks
                    if j == 0:
                        # First block gets upsampled + skip connection
                        block_in_ch = out_ch * 2
                    else:
                        block_in_ch = out_ch
                    blocks.append(ResNetBlock(
                        block_in_ch, 
                        out_ch, 
                        time_emb_dim=time_emb_dim, 
                        dropout=self.dropout
                    ))
                self.up_blocks.append(blocks)
                
                # Attention using standardized component
                current_resolution = self.sample_size // (2 ** (len(reversed_channels) - 2 - i))
                if current_resolution in self.attention_resolutions:
                    self.up_attentions.append(SelfAttention2D(out_ch, self.num_attention_heads))
                else:
                    self.up_attentions.append(nn.Identity())
        
        # Output projection - zero-initialized conv for diffusion
        self.conv_norm_out = get_normalization("groupnorm", self.block_out_channels[0])
        self.conv_out = ZeroConv2d(self.block_out_channels[0], self.out_channels, 3, padding=1)
        
        # Memory optimization settings
        self.gradient_checkpointing = config.get("gradient_checkpointing", False)
        self.memory_efficient_attention = config.get("memory_efficient_attention", True)
        
        # Enable memory optimizations if requested
        if self.gradient_checkpointing:
            self.enable_gradient_checkpointing()
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        # Apply to all attention blocks for maximum memory savings
        for module in self.modules():
            if isinstance(module, SelfAttention2D):
                module.enable_gradient_checkpointing()
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
        for module in self.modules():
            if isinstance(module, SelfAttention2D):
                module.disable_gradient_checkpointing()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
        return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0}
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize 16-bit input images to model range [-1, 1]."""
        return normalize_16bit_to_model_input(x)
    
    def denormalize_output(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize model output back to 16-bit range [0, 65535]."""
        return denormalize_model_output_to_16bit(x)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                normalize_input: bool = True, denormalize_output: bool = False) -> torch.Tensor:
        """Forward pass with optional 16-bit normalization.
        
        Args:
            x: Input tensor [B, C, H, W] - expected in 16-bit range [0, 65535] if normalize_input=True
            t: Timestep tensor [B] - required for diffusion
            normalize_input: Whether to normalize 16-bit input to [-1, 1]
            denormalize_output: Whether to denormalize output back to [0, 65535]
            
        Returns:
            Predicted tensor [B, C, H, W] - in [-1, 1] range unless denormalize_output=True
        """
        # Normalize 16-bit input to model range if requested
        if normalize_input:
            x = self.normalize_input(x)
        
        # Time embedding (only for diffusion mode)
        if self.use_time_conditioning:
            if t is None:
                raise ValueError("Timestep t is required for diffusion mode")
            time_emb = self.time_embedding(t)
        else:
            time_emb = None
        
        # Input projection
        h = self.conv_in(x)
        
        # Down path with gradient checkpointing support
        skip_connections = []
        if self.mode == "supervised":
            # Supervised mode: Simple encoder path
            for i, (encoder, attention, downsample) in enumerate(zip(self.down_blocks, self.down_attentions, self.down_samples)):
                if self.gradient_checkpointing and self.training:
                    h = torch.utils.checkpoint.checkpoint(encoder, h, use_reentrant=False)
                else:
                    h = encoder(h)
                
                if not isinstance(attention, nn.Identity):
                    if self.gradient_checkpointing and self.training:
                        h = torch.utils.checkpoint.checkpoint(attention, h, use_reentrant=False)
                    else:
                        h = attention(h)
                
                # Store skip connection before downsampling (except for last layer)
                if not isinstance(downsample, nn.Identity) and i < len(self.down_blocks) - 1:
                    skip_connections.append(h)
                
                h = downsample(h)
        else:
            # Diffusion mode: ResNet blocks with time conditioning
            for blocks, attention, downsample in zip(self.down_blocks, self.down_attentions, self.down_samples):
                for block in blocks:
                    if self.gradient_checkpointing and self.training:
                        h = torch.utils.checkpoint.checkpoint(block, h, time_emb, use_reentrant=False)
                    else:
                        h = block(h, time_emb)
                
                if not isinstance(attention, nn.Identity):
                    if self.gradient_checkpointing and self.training:
                        h = torch.utils.checkpoint.checkpoint(attention, h, use_reentrant=False)
                    else:
                        h = attention(h)
                skip_connections.append(h)
                h = downsample(h)
        
        # Middle blocks (only for diffusion mode) with gradient checkpointing
        if self.mode == "diffusion":
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(self.mid_block1, h, time_emb, use_reentrant=False)
                h = torch.utils.checkpoint.checkpoint(self.mid_attn, h, use_reentrant=False)
                h = torch.utils.checkpoint.checkpoint(self.mid_block2, h, time_emb, use_reentrant=False)
            else:
                h = self.mid_block1(h, time_emb)
                h = self.mid_attn(h)
                h = self.mid_block2(h, time_emb)
        
        # Up path with gradient checkpointing support
        if self.mode == "supervised":
            # Supervised mode: Simple decoder path
            for decoder, attention, upsample in zip(self.up_blocks, self.up_attentions, self.up_samples):
                # Upsample
                h = upsample(h)
                
                # Get skip connection (pop from end for reverse order)
                if len(skip_connections) > 0:
                    skip = skip_connections.pop()
                    if skip.shape[2:] != h.shape[2:]:
                        skip = F.interpolate(skip, size=h.shape[2:], mode='bilinear', align_corners=False)
                    h = torch.cat([h, skip], dim=1)
                
                # Process decoder block with checkpointing
                if self.gradient_checkpointing and self.training:
                    h = torch.utils.checkpoint.checkpoint(decoder, h, use_reentrant=False)
                else:
                    h = decoder(h)
                
                # Apply attention with checkpointing
                if not isinstance(attention, nn.Identity):
                    if self.gradient_checkpointing and self.training:
                        h = torch.utils.checkpoint.checkpoint(attention, h, use_reentrant=False)
                    else:
                        h = attention(h)
        else:
            # Diffusion mode: ResNet blocks with time conditioning
            for blocks, attention, upsample in zip(self.up_blocks, self.up_attentions, self.up_samples):
                # Upsample using standardized component
                h = upsample(h)
                
                # Get skip connection
                if len(skip_connections) > 0:
                    skip = skip_connections.pop()
                    if skip.shape[2:] != h.shape[2:]:
                        skip = F.interpolate(skip, size=h.shape[2:], mode='bilinear', align_corners=False)
                    h = torch.cat([h, skip], dim=1)
                
                # Process blocks with checkpointing
                for block in blocks:
                    if self.gradient_checkpointing and self.training:
                        h = torch.utils.checkpoint.checkpoint(block, h, time_emb, use_reentrant=False)
                    else:
                        h = block(h, time_emb)
                
                if not isinstance(attention, nn.Identity):
                    if self.gradient_checkpointing and self.training:
                        h = torch.utils.checkpoint.checkpoint(attention, h, use_reentrant=False)
                    else:
                        h = attention(h)
        
        # Output
        h = self.conv_norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        # Denormalize output back to 16-bit range if requested
        if denormalize_output:
            h = self.denormalize_output(h)
        
        return h
