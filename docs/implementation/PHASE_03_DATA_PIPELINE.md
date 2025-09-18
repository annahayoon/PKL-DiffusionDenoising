# Phase 3: Data Pipeline (Week 2)

### Step 3.1: Implement Data Synthesis
```python
# pkl_dg/data/synthesis.py
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image

class SynthesisDataset(Dataset):
    """Dataset for synthesizing training pairs from external image sources."""
    
    def __init__(
        self,
        source_dir: str,
        forward_model: 'ForwardModel',
        transform: Optional[T.Compose] = None,
        image_size: int = 256,
        mode: str = 'train'
    ):
        """
        Initialize synthesis dataset.
        
        Args:
            source_dir: Directory with source images
            forward_model: Forward model for WF simulation
            transform: Additional transforms
            image_size: Target image size
            mode: 'train', 'val', or 'test'
        """
        self.source_dir = Path(source_dir)
        self.forward_model = forward_model
        self.image_size = image_size
        self.mode = mode
        
        # Collect image paths
        self.image_paths = list(self.source_dir.glob("**/*.png"))
        self.image_paths += list(self.source_dir.glob("**/*.jpg"))
        self.image_paths += list(self.source_dir.glob("**/*.tif"))
        
        # Basic transforms
        self.base_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
        ])
        
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get synthesized training pair.
        
        Returns:
            Tuple of (2P-like clean image, WF-like noisy image)
        """
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        
        # Apply base transforms
        img = self.base_transform(img)
        
        # Create "2P-like" clean image (mild processing)
        x_2p = self._create_2p_like(img)
        
        # Create "WF-like" measurement
        with torch.no_grad():
            # Apply forward model
            y_wf = self.forward_model.forward(x_2p.unsqueeze(0), add_noise=True)
            y_wf = y_wf.squeeze(0)
            
            # Add Gaussian background
            if self.mode == 'train':
                background_noise = torch.randn_like(y_wf) * 0.1
                y_wf = y_wf + torch.abs(background_noise)
        
        # Apply additional transforms if any
        if self.transform:
            x_2p = self.transform(x_2p)
            y_wf = self.transform(y_wf)
        
        return x_2p, y_wf
    
    def _create_2p_like(self, img: torch.Tensor) -> torch.Tensor:
        """
        Process image to be more 2P-like.
        
        Args:
            img: Input image
        
        Returns:
            Processed image resembling 2P microscopy
        """
        # Apply mild sharpening and contrast enhancement
        # This simulates the better resolution of 2P
        
        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Enhance contrast (gamma correction)
        img = torch.pow(img, 0.8)
        
        # Scale to realistic photon counts (e.g., 10-1000 photons)
        img = img * 500 + 10
        
        return img
```

### Step 3.2: Implement Data Transforms
```python
# pkl_dg/data/transforms.py
import torch
import numpy as np

class Normalize:
    """Normalize images to [-1, 1] for diffusion model."""
    
    def __init__(self, mean: float = 0.5, std: float = 0.5):
        self.mean = mean
        self.std = std
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

class IntensityToModel:
    """Convert intensity domain to model domain [-1, 1]."""
    
    def __init__(self, min_intensity: float = 0, max_intensity: float = 1000):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Clip to valid range
        x = torch.clamp(x, self.min_intensity, self.max_intensity)
        # Scale to [0, 1]
        x = (x - self.min_intensity) / (self.max_intensity - self.min_intensity)
        # Scale to [-1, 1]
        return 2 * x - 1
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # From [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # From [0, 1] to intensity
        x = x * (self.max_intensity - self.min_intensity) + self.min_intensity
        return torch.clamp(x, min=0)  # Ensure non-negative

class RandomCrop:
    """Random crop for data augmentation."""
    
    def __init__(self, size: int):
        self.size = size
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape
        if h > self.size and w > self.size:
            top = torch.randint(0, h - self.size + 1, (1,)).item()
            left = torch.randint(0, w - self.size + 1, (1,)).item()
            x = x[:, top:top+self.size, left:left+self.size]
        return x
```
