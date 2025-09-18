"""
Neural Network Building Blocks and Defaults for PKL-Diffusion Models.

This module provides standardized neural network components with consistent
defaults across the PKL-Diffusion codebase. It helps maintain consistency
and reduces code duplication.
"""

from typing import Optional, Tuple, Union, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Default Configuration
# =============================================================================

class NNDefaults:
    """Default values for neural network components."""
    
    # Activation functions
    ACTIVATION = nn.SiLU  # Swish activation (standard for diffusion models)
    
    # Normalization
    NORM_NUM_GROUPS = 8  # For GroupNorm
    NORM_EPS = 1e-6
    
    # Dropout
    DROPOUT_RATE = 0.1
    
    # Convolution defaults
    CONV_KERNEL_SIZE = 3
    CONV_PADDING = 1
    CONV_BIAS = True
    
    # Attention defaults
    ATTENTION_HEAD_DIM = 64
    ATTENTION_DROPOUT = 0.1
    
    # Time embedding defaults
    TIME_EMB_MULT = 4  # time_emb_dim = channels * TIME_EMB_MULT
    
    # Channel progression for UNets
    UNET_CHANNELS = [64, 128, 256, 512]
    UNET_LAYERS_PER_BLOCK = 2
    
    # Initialization
    ZERO_INIT_FINAL_CONV = True
    ZERO_INIT_RESIDUAL = True


# =============================================================================
# Utility Functions
# =============================================================================

def get_activation(name: str = "silu") -> nn.Module:
    """Get activation function by name."""
    activations = {
        "silu": nn.SiLU,
        "swish": nn.SiLU,  # Alias for SiLU
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "mish": nn.Mish,
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    return activations[name.lower()]()


def get_normalization(
    norm_type: str, 
    num_channels: int, 
    num_groups: Optional[int] = None,
    eps: float = NNDefaults.NORM_EPS
) -> nn.Module:
    """Get normalization layer by type."""
    if norm_type.lower() == "groupnorm":
        if num_groups is None:
            num_groups = min(NNDefaults.NORM_NUM_GROUPS, num_channels)
        return nn.GroupNorm(num_groups, num_channels, eps=eps)
    elif norm_type.lower() == "batchnorm":
        return nn.BatchNorm2d(num_channels, eps=eps)
    elif norm_type.lower() == "layernorm":
        return nn.LayerNorm(num_channels, eps=eps)
    else:
        raise ValueError(f"Unknown normalization: {norm_type}")


def sinusoidal_position_embeddings(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal position embeddings for timesteps.
    
    Args:
        timesteps: Tensor of shape [batch_size]
        dim: Embedding dimension
        
    Returns:
        Embeddings of shape [batch_size, dim]
    """
    device = timesteps.device
    half_dim = dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings = timesteps[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    return embeddings


# =============================================================================
# Basic Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """Standard convolution block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = NNDefaults.CONV_KERNEL_SIZE,
        padding: int = NNDefaults.CONV_PADDING,
        stride: int = 1,
        norm_type: str = "groupnorm",
        activation: str = "silu",
        dropout: float = 0.0,
        bias: bool = NNDefaults.CONV_BIAS,
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=bias
        )
        self.norm = get_normalization(norm_type, out_channels)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DoubleConvBlock(nn.Module):
    """Double convolution block (conv-norm-act-conv-norm-act)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        norm_type: str = "groupnorm",
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.conv1 = ConvBlock(
            in_channels, mid_channels, norm_type=norm_type, 
            activation=activation, dropout=dropout
        )
        self.conv2 = ConvBlock(
            mid_channels, out_channels, norm_type=norm_type, 
            activation=activation, dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TimeEmbeddingMLP(nn.Module):
    """Time embedding MLP for diffusion models."""
    
    def __init__(
        self,
        time_dim: int,
        emb_dim: Optional[int] = None,
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if emb_dim is None:
            emb_dim = time_dim * NNDefaults.TIME_EMB_MULT
        
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, emb_dim),
            get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(emb_dim, emb_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Convert timesteps to sinusoidal embeddings
        if t.dtype in [torch.int32, torch.int64]:
            t = sinusoidal_position_embeddings(t, self.mlp[0].in_features)
        return self.mlp(t)


# =============================================================================
# Attention Mechanisms
# =============================================================================

class SelfAttention2D(nn.Module):
    """2D Self-attention block for diffusion models with memory optimizations."""
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = NNDefaults.ATTENTION_DROPOUT,
        norm_type: str = "groupnorm",
        memory_efficient: bool = True,
    ):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        if head_dim is None:
            head_dim = channels // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        inner_dim = head_dim * num_heads
        
        self.norm = get_normalization(norm_type, channels)
        self.to_qkv = nn.Linear(channels, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, channels),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        
        # Memory optimization settings
        self.memory_efficient = memory_efficient
        self.gradient_checkpointing = False
        
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for this attention block."""
        self.gradient_checkpointing = True
        
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing for this attention block."""
        self.gradient_checkpointing = False
    
    def _attention_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Core attention computation separated for gradient checkpointing."""
        batch, channels, height, width = x.shape
        
        # Normalize and reshape
        x = self.norm(x)
        x = x.view(batch, channels, height * width).transpose(1, 2)
        
        # Get Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(batch, height * width, self.num_heads, self.head_dim).transpose(1, 2),
            qkv
        )
        
        # Memory-efficient attention computation
        if self.memory_efficient and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch 2.0+ native SDPA if available (more memory efficient)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        else:
            # Fallback to standard attention with chunked computation for memory efficiency
            seq_len = q.shape[2]
            chunk_size = min(seq_len, 1024)  # Process in chunks to save memory
            
            if seq_len > chunk_size:
                out_chunks = []
                for i in range(0, seq_len, chunk_size):
                    end_i = min(i + chunk_size, seq_len)
                    q_chunk = q[:, :, i:end_i]
                    
                    # Compute attention for this chunk
                    attn = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
                    attn = F.softmax(attn, dim=-1)
                    out_chunk = torch.matmul(attn, v)
                    out_chunks.append(out_chunk)
                
                out = torch.cat(out_chunks, dim=2)
            else:
                # Standard attention for small sequences
                attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn = F.softmax(attn, dim=-1)
                out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).reshape(batch, height * width, -1)
        out = self.to_out(out)
        
        # Reshape back
        out = out.transpose(1, 2).view(batch, channels, height, width)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory during training
            out = torch.utils.checkpoint.checkpoint(
                self._attention_forward, x, use_reentrant=False
            )
        else:
            out = self._attention_forward(x)
        
        return out + residual


# =============================================================================
# ResNet Blocks
# =============================================================================

class ResNetBlock(nn.Module):
    """ResNet block with time embedding for diffusion models.
    
    Handles dynamic input channels for skip connections by creating
    normalization layers on-the-fly when needed.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: Optional[int] = None,
        norm_type: str = "groupnorm",
        activation: str = "silu",
        dropout: float = NNDefaults.DROPOUT_RATE,
        zero_init: bool = NNDefaults.ZERO_INIT_RESIDUAL,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.activation_name = activation
        
        # Time embedding projection
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                get_activation(activation),
                nn.Linear(time_emb_dim, out_channels),
            )
        else:
            self.time_mlp = None
        
        # Second conv block (always fixed channels)
        self.norm2 = get_normalization(norm_type, out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Activation
        self.activation = get_activation(activation)
        
        # Zero initialization
        if zero_init:
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)
        
        # Cache for dynamic layers
        self._norm_cache = {}
        self._conv_cache = {}
        self._skip_cache = {}
    
    def _get_norm1(self, channels: int, device: torch.device) -> nn.Module:
        """Get or create norm layer for given channels."""
        if channels not in self._norm_cache:
            norm = get_normalization(self.norm_type, channels)
            norm = norm.to(device)
            self._norm_cache[channels] = norm
        return self._norm_cache[channels]
    
    def _get_conv1(self, in_channels: int, device: torch.device) -> nn.Module:
        """Get or create conv1 layer for given input channels."""
        if in_channels not in self._conv_cache:
            conv = nn.Conv2d(in_channels, self.out_channels, 3, padding=1)
            conv = conv.to(device)
            self._conv_cache[in_channels] = conv
        return self._conv_cache[in_channels]
    
    def _get_skip_connection(self, in_channels: int, device: torch.device) -> nn.Module:
        """Get or create skip connection for given input channels."""
        if in_channels not in self._skip_cache:
            if in_channels != self.out_channels:
                skip = nn.Conv2d(in_channels, self.out_channels, 1)
                skip = skip.to(device)
            else:
                skip = nn.Identity()
            self._skip_cache[in_channels] = skip
        return self._skip_cache[in_channels]
    
    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        actual_in_channels = x.shape[1]
        device = x.device
        
        # First conv block with dynamic normalization and convolution
        norm1 = self._get_norm1(actual_in_channels, device)
        h = norm1(x)
        h = self.activation(h)
        
        conv1 = self._get_conv1(actual_in_channels, device)
        h = conv1(h)
        
        # Add time embedding
        if time_emb is not None and self.time_mlp is not None:
            time_proj = self.time_mlp(time_emb)
            h = h + time_proj[:, :, None, None]
        
        # Second conv block
        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Dynamic residual connection
        skip_connection = self._get_skip_connection(actual_in_channels, device)
        return h + skip_connection(residual)


# =============================================================================
# Upsampling and Downsampling
# =============================================================================

class Downsample(nn.Module):
    """Downsampling layer."""
    
    def __init__(
        self, 
        channels: int, 
        method: str = "conv",
        factor: int = 2,
    ):
        super().__init__()
        
        if method == "conv":
            self.downsample = nn.Conv2d(channels, channels, 3, stride=factor, padding=1)
        elif method == "avgpool":
            self.downsample = nn.AvgPool2d(factor)
        elif method == "maxpool":
            self.downsample = nn.MaxPool2d(factor)
        else:
            raise ValueError(f"Unknown downsampling method: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)


class Upsample(nn.Module):
    """Upsampling layer."""
    
    def __init__(
        self, 
        channels: int, 
        method: str = "conv_transpose",
        factor: int = 2,
    ):
        super().__init__()
        
        if method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(channels, channels, 4, stride=factor, padding=1)
        elif method == "interpolate":
            self.upsample = lambda x: F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False)
        else:
            raise ValueError(f"Unknown upsampling method: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


# =============================================================================
# Specialized Components
# =============================================================================

class GroupNormDynamic(nn.Module):
    """GroupNorm with dynamic group calculation based on input channels."""
    
    def __init__(self, max_groups: int = NNDefaults.NORM_NUM_GROUPS, eps: float = NNDefaults.NORM_EPS):
        super().__init__()
        self.max_groups = max_groups
        self.eps = eps
        self._cached_norms = {}  # Cache for different channel sizes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        
        if channels not in self._cached_norms:
            num_groups = min(self.max_groups, channels)
            # Ensure num_groups divides channels
            while channels % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            
            norm = nn.GroupNorm(num_groups, channels, eps=self.eps)
            norm = norm.to(x.device)
            self._cached_norms[channels] = norm
        
        return self._cached_norms[channels](x)


class ZeroConv2d(nn.Module):
    """Convolution initialized to zero (useful for residual connections)."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# =============================================================================
# Factory Functions
# =============================================================================

def make_time_embedding(dim: int, activation: str = "silu") -> TimeEmbeddingMLP:
    """Factory function for creating time embeddings."""
    return TimeEmbeddingMLP(dim, activation=activation)


def make_resnet_block(
    in_channels: int,
    out_channels: int,
    time_emb_dim: Optional[int] = None,
    **kwargs
) -> ResNetBlock:
    """Factory function for creating ResNet blocks."""
    return ResNetBlock(in_channels, out_channels, time_emb_dim, **kwargs)


def make_attention_block(channels: int, num_heads: int = 8, **kwargs) -> SelfAttention2D:
    """Factory function for creating attention blocks."""
    return SelfAttention2D(channels, num_heads, **kwargs)


def make_conv_sequence(
    in_channels: int,
    out_channels: int,
    num_layers: int = 2,
    activation: str = "silu",
    norm_type: str = "groupnorm",
    dropout: float = 0.0,
) -> nn.Sequential:
    """Create a sequence of convolution blocks."""
    layers = []
    
    for i in range(num_layers):
        layer_in = in_channels if i == 0 else out_channels
        layers.append(ConvBlock(
            layer_in, out_channels,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout,
        ))
    
    return nn.Sequential(*layers)
