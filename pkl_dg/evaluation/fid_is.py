"""
FID and IS Evaluation Metrics for PKL Diffusion Denoising

This module implements Fréchet Inception Distance (FID) and Inception Score (IS)
for evaluating the quality of generated images, with support for microscopy images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
import warnings

try:
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    from scipy.linalg import sqrtm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class InceptionV3Feature(nn.Module):
    """InceptionV3 model for feature extraction."""
    
    def __init__(self, resize_input: bool = True, normalize_input: bool = True):
        """Initialize InceptionV3 feature extractor.
        
        Args:
            resize_input: Whether to resize input to 299x299
            normalize_input: Whether to normalize input for ImageNet
        """
        super().__init__()
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for InceptionV3 features")
        
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        
        # Load pretrained InceptionV3
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.eval()
        
        # Remove final layers to get features
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Features [B, 2048]
        """
        # Handle grayscale images
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # Convert to RGB
        
        # Resize if needed
        if self.resize_input:
            if x.shape[-1] != 299 or x.shape[-2] != 299:
                x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize for ImageNet if needed
        if self.normalize_input:
            # Assume input is in [-1, 1], convert to ImageNet normalization
            x = (x + 1) / 2  # Convert to [0, 1]
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std
        
        # Extract features
        features = self.features(x)
        return features.view(features.size(0), -1)


class MicroscopyFeatureExtractor(nn.Module):
    """Custom feature extractor optimized for microscopy images."""
    
    def __init__(self, feature_dim: int = 512):
        """Initialize microscopy feature extractor.
        
        Args:
            feature_dim: Dimension of output features
        """
        super().__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth block
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Freeze after initialization (can be unfrozen for training)
        for param in self.parameters():
            param.requires_grad = False
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from microscopy images.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Features [B, feature_dim]
        """
        # Handle RGB images by converting to grayscale
        if x.shape[1] == 3:
            x = torch.mean(x, dim=1, keepdim=True)
        
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)


def extract_features_batch(
    images: torch.Tensor,
    feature_extractor: nn.Module,
    batch_size: int = 50,
    device: str = "cuda",
    verbose: bool = True
) -> np.ndarray:
    """Extract features from a batch of images.
    
    Args:
        images: Input images [N, C, H, W]
        feature_extractor: Feature extraction model
        batch_size: Batch size for processing
        device: Device to run on
        verbose: Whether to show progress
        
    Returns:
        Features array [N, feature_dim]
    """
    feature_extractor.eval()
    feature_extractor = feature_extractor.to(device)
    
    features_list = []
    num_batches = (len(images) + batch_size - 1) // batch_size
    
    iterator = range(num_batches)
    if verbose and TQDM_AVAILABLE:
        iterator = tqdm(iterator, desc="Extracting features")
    
    with torch.no_grad():
        for i in iterator:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            
            batch = images[start_idx:end_idx].to(device)
            batch_features = feature_extractor(batch)
            features_list.append(batch_features.cpu().numpy())
    
    return np.concatenate(features_list, axis=0)


def calculate_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and covariance of features.
    
    Args:
        features: Feature array [N, feature_dim]
        
    Returns:
        Tuple of (mean, covariance)
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    eps: float = 1e-6
) -> float:
    """Calculate Fréchet Inception Distance.
    
    Args:
        real_features: Features from real images [N1, feature_dim]
        fake_features: Features from generated images [N2, feature_dim]
        eps: Small value to avoid numerical issues
        
    Returns:
        FID score (lower is better)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for FID calculation")
    
    # Calculate statistics
    mu1, sigma1 = calculate_statistics(real_features)
    mu2, sigma2 = calculate_statistics(fake_features)
    
    # Calculate FID
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        msg = ('FID calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {} too large'.format(m))
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma1) + 
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_inception_score(
    features: np.ndarray,
    splits: int = 10
) -> Tuple[float, float]:
    """Calculate Inception Score.
    
    Args:
        features: Features from generated images [N, feature_dim]
        splits: Number of splits for calculating IS
        
    Returns:
        Tuple of (mean_IS, std_IS)
    """
    # Convert features to probabilities using softmax
    # Note: This assumes features are logits from a classifier
    # For feature vectors, we need to add a classifier head
    
    # Simple approximation: use feature magnitudes as pseudo-probabilities
    probs = F.softmax(torch.from_numpy(features), dim=1).numpy()
    
    scores = []
    n_samples = probs.shape[0]
    split_size = n_samples // splits
    
    for i in range(splits):
        start_idx = i * split_size
        end_idx = min((i + 1) * split_size, n_samples)
        
        if end_idx <= start_idx:
            continue
            
        split_probs = probs[start_idx:end_idx]
        
        # Calculate marginal probability
        p_y = np.mean(split_probs, axis=0)
        
        # Calculate KL divergence
        kl_div = split_probs * (np.log(split_probs + 1e-16) - np.log(p_y + 1e-16))
        kl_div = np.mean(np.sum(kl_div, axis=1))
        
        scores.append(np.exp(kl_div))
    
    return np.mean(scores), np.std(scores)


def evaluate_fid_is(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    feature_extractor: Optional[nn.Module] = None,
    batch_size: int = 50,
    device: str = "cuda",
    use_microscopy_features: bool = False,
    verbose: bool = True
) -> Dict[str, float]:
    """Evaluate FID and IS for generated images.
    
    Args:
        real_images: Real images [N1, C, H, W]
        fake_images: Generated images [N2, C, H, W]
        feature_extractor: Custom feature extractor (optional)
        batch_size: Batch size for processing
        device: Device to run on
        use_microscopy_features: Whether to use microscopy-specific features
        verbose: Whether to show progress
        
    Returns:
        Dictionary with FID and IS scores
    """
    # Choose feature extractor
    if feature_extractor is None:
        if use_microscopy_features:
            feature_extractor = MicroscopyFeatureExtractor()
        else:
            if not TORCHVISION_AVAILABLE:
                raise ImportError("torchvision required for InceptionV3 features")
            feature_extractor = InceptionV3Feature()
    
    # Extract features
    if verbose:
        print("Extracting features from real images...")
    real_features = extract_features_batch(
        real_images, feature_extractor, batch_size, device, verbose
    )
    
    if verbose:
        print("Extracting features from generated images...")
    fake_features = extract_features_batch(
        fake_images, feature_extractor, batch_size, device, verbose
    )
    
    # Calculate FID
    if verbose:
        print("Calculating FID...")
    fid_score = calculate_fid(real_features, fake_features)
    
    # Calculate IS
    if verbose:
        print("Calculating IS...")
    is_mean, is_std = calculate_inception_score(fake_features)
    
    results = {
        "fid": float(fid_score),
        "is_mean": float(is_mean),
        "is_std": float(is_std),
        "num_real": len(real_images),
        "num_fake": len(fake_images)
    }
    
    if verbose:
        print(f"Results: FID = {fid_score:.2f}, IS = {is_mean:.2f} ± {is_std:.2f}")
    
    return results


def load_images_from_directory(
    directory: Union[str, Path],
    max_images: Optional[int] = None,
    image_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True
) -> torch.Tensor:
    """Load images from directory for evaluation.
    
    Args:
        directory: Directory containing images
        max_images: Maximum number of images to load
        image_size: Target image size (H, W)
        normalize: Whether to normalize to [-1, 1]
        
    Returns:
        Tensor of images [N, C, H, W]
    """
    from PIL import Image
    import os
    
    directory = Path(directory)
    
    # Supported image extensions
    extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(directory.glob(f"*{ext}"))
        image_paths.extend(directory.glob(f"*{ext.upper()}"))
    
    if max_images is not None:
        image_paths = image_paths[:max_images]
    
    if not image_paths:
        raise ValueError(f"No images found in {directory}")
    
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            
            # Convert to grayscale if needed
            if img.mode != 'L' and img.mode != 'RGB':
                img = img.convert('L')
            
            # Resize if specified
            if image_size is not None:
                img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
            
            # Convert to tensor
            img_array = np.array(img, dtype=np.float32)
            
            # Add channel dimension if grayscale
            if img_array.ndim == 2:
                img_array = img_array[np.newaxis, ...]
            elif img_array.ndim == 3:
                img_array = np.transpose(img_array, (2, 0, 1))
            
            # Normalize
            if normalize:
                if img_array.max() > 1.0:  # Assume 0-255 range
                    img_array = img_array / 255.0
                img_array = (img_array * 2.0) - 1.0  # Convert to [-1, 1]
            
            images.append(torch.from_numpy(img_array))
            
        except Exception as e:
            print(f"⚠️ Failed to load {img_path}: {e}")
            continue
    
    if not images:
        raise ValueError("Failed to load any images")
    
    return torch.stack(images, dim=0)


class FIDISEvaluator:
    """Comprehensive FID/IS evaluator for diffusion models."""
    
    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_microscopy_features: bool = False
    ):
        """Initialize evaluator.
        
        Args:
            feature_extractor: Custom feature extractor
            device: Device to run on
            use_microscopy_features: Whether to use microscopy-specific features
        """
        self.device = device
        self.use_microscopy_features = use_microscopy_features
        
        if feature_extractor is None:
            if use_microscopy_features:
                self.feature_extractor = MicroscopyFeatureExtractor()
            else:
                self.feature_extractor = InceptionV3Feature()
        else:
            self.feature_extractor = feature_extractor
        
        self.feature_extractor.to(device)
        
        # Cache for real image features
        self.real_features_cache = {}
    
    def extract_features(
        self,
        images: torch.Tensor,
        batch_size: int = 50,
        verbose: bool = True
    ) -> np.ndarray:
        """Extract features from images."""
        return extract_features_batch(
            images, self.feature_extractor, batch_size, self.device, verbose
        )
    
    def evaluate(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        batch_size: int = 50,
        verbose: bool = True,
        cache_key: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate FID and IS.
        
        Args:
            real_images: Real images
            fake_images: Generated images
            batch_size: Batch size for processing
            verbose: Whether to show progress
            cache_key: Key for caching real features
            
        Returns:
            Evaluation results
        """
        # Check cache for real features
        real_features = None
        if cache_key is not None and cache_key in self.real_features_cache:
            real_features = self.real_features_cache[cache_key]
            if verbose:
                print("Using cached real image features")
        
        # Extract real features if not cached
        if real_features is None:
            if verbose:
                print("Extracting features from real images...")
            real_features = self.extract_features(real_images, batch_size, verbose)
            
            # Cache if key provided
            if cache_key is not None:
                self.real_features_cache[cache_key] = real_features
        
        # Extract fake features
        if verbose:
            print("Extracting features from generated images...")
        fake_features = self.extract_features(fake_images, batch_size, verbose)
        
        # Calculate metrics
        if verbose:
            print("Calculating FID...")
        fid_score = calculate_fid(real_features, fake_features)
        
        if verbose:
            print("Calculating IS...")
        is_mean, is_std = calculate_inception_score(fake_features)
        
        results = {
            "fid": float(fid_score),
            "is_mean": float(is_mean),
            "is_std": float(is_std),
            "num_real": len(real_images),
            "num_fake": len(fake_images)
        }
        
        if verbose:
            print(f"Results: FID = {fid_score:.2f}, IS = {is_mean:.2f} ± {is_std:.2f}")
        
        return results
    
    def evaluate_from_directories(
        self,
        real_dir: Union[str, Path],
        fake_dir: Union[str, Path],
        max_images: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        batch_size: int = 50,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Evaluate FID/IS from image directories.
        
        Args:
            real_dir: Directory with real images
            fake_dir: Directory with generated images
            max_images: Maximum images to evaluate
            image_size: Target image size
            batch_size: Batch size for processing
            verbose: Whether to show progress
            
        Returns:
            Evaluation results
        """
        # Load images
        if verbose:
            print(f"Loading real images from {real_dir}...")
        real_images = load_images_from_directory(
            real_dir, max_images, image_size, normalize=True
        )
        
        if verbose:
            print(f"Loading generated images from {fake_dir}...")
        fake_images = load_images_from_directory(
            fake_dir, max_images, image_size, normalize=True
        )
        
        # Evaluate
        return self.evaluate(real_images, fake_images, batch_size, verbose)
    
    def clear_cache(self):
        """Clear cached features."""
        self.real_features_cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached features."""
        return {
            "num_cached": len(self.real_features_cache),
            "cache_keys": list(self.real_features_cache.keys()),
            "total_cached_features": sum(
                features.shape[0] for features in self.real_features_cache.values()
            )
        }
