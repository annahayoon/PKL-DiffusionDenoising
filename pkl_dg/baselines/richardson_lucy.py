#!/usr/bin/env python3
"""
Richardson-Lucy Deconvolution for PKL-DG Evaluation

This module provides both the core Richardson-Lucy algorithm implementation
and a comprehensive evaluation pipeline that includes:
1. PSF extraction from bead data
2. Richardson-Lucy deconvolution processing
3. Evaluation against ground truth
4. Visualization and results export

Usage:
    # Core algorithm only
    from pkl_dg.baselines.richardson_lucy import richardson_lucy_restore
    result = richardson_lucy_restore(image, psf, num_iter=30)

    # Full pipeline with PSF extraction
    python -m pkl_dg.baselines.richardson_lucy \
        --bead-dir data/real_microscopy/beads \
        --input-dir data/real_microscopy/splits/test/wf \
        --gt-dir data/real_microscopy/splits/test/2p \
        --output-dir outputs/rl_baseline

    # Use existing PSF
    python -m pkl_dg.baselines.richardson_lucy \
        --psf-path data/psf/psf.tif \
        --input-dir data/real_microscopy/splits/test/wf \
        --gt-dir data/real_microscopy/splits/test/2p \
        --output-dir outputs/rl_baseline
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import numpy as np
import torch
import tifffile
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# PKL-DG imports
from ..physics.psf import PSF, build_psf_bank, estimate_psf_from_beads, _load_grayscale_images
from ..evaluation.metrics import Metrics


def richardson_lucy_restore(
    image: np.ndarray,
    psf: np.ndarray,
    num_iter: int = 30,
    clip: bool = True,
) -> np.ndarray:
    """Richardson–Lucy deconvolution baseline using scikit-image if available.

    Falls back to a minimal NumPy implementation if scikit-image is not installed.
    Expects single-channel 2D arrays.
    
    Args:
        image: Input degraded image
        psf: Point spread function
        num_iter: Number of iterations
        clip: Whether to clip negative values
        
    Returns:
        Deconvolved image
    """
    try:
        from skimage.restoration import richardson_lucy  # type: ignore

        return richardson_lucy(image, psf, iterations=num_iter, clip=clip)
    except Exception:
        # Simple fallback RL without acceleration
        from scipy.signal import fftconvolve  # type: ignore

        img = image.astype(np.float32)
        kernel = psf.astype(np.float32)
        kernel = kernel / (kernel.sum() + 1e-12)
        estimate = np.maximum(img, 1e-12)
        psf_mirror = kernel[::-1, ::-1]
        for _ in tqdm(range(num_iter), desc="Richardson-Lucy iterations"):
            conv = fftconvolve(estimate, kernel, mode="same")
            relative_blur = img / (conv + 1e-12)
            estimate *= fftconvolve(relative_blur, psf_mirror, mode="same")
            if clip:
                estimate = np.clip(estimate, 0, None)
        return estimate


class RichardsonLucyComprehensive:
    """Comprehensive Richardson-Lucy baseline for PKL-DG evaluation."""
    
    def __init__(
        self,
        psf_path: Optional[str] = None,
        bead_dir: Optional[str] = None,
        device: str = "cuda",
        patch_size: int = 256,
        stride: int = 128,
        iterations: int = 30,
        clip: bool = True,
        psf_method: str = "average"
    ):
        """
        Initialize Richardson-Lucy comprehensive baseline.
        
        Args:
            psf_path: Path to existing PSF file (optional if bead_dir provided)
            bead_dir: Directory containing bead data for PSF extraction
            device: Computation device
            patch_size: Size of patches for processing
            stride: Stride between patches
            iterations: Number of RL iterations
            clip: Clip negative values
            psf_method: PSF extraction method ("average" or "bank")
        """
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        self.iterations = iterations
        self.clip = clip
        self.psf_method = psf_method
        
        # Load or extract PSF
        if psf_path and Path(psf_path).exists():
            print(f"Loading existing PSF from {psf_path}")
            self.psf = self._load_psf(psf_path)
        elif bead_dir:
            print(f"Extracting PSF from bead data in {bead_dir}")
            self.psf = self._extract_psf_from_beads(bead_dir)
        else:
            raise ValueError("Either psf_path or bead_dir must be provided")
        
        print(f"PSF loaded: shape={self.psf.shape}, sum={self.psf.sum():.6f}")
    
    def _load_psf(self, psf_path: str) -> np.ndarray:
        """Load PSF from file using PSF class."""
        # Use the PSF class from psf.py for consistent loading
        psf_obj = PSF(psf_path=psf_path)
        psf_array = psf_obj.psf
        
        print(f"PSF loaded: shape={psf_array.shape}, sum={psf_array.sum():.6f}")
        print(f"PSF range: {psf_array.min():.6f} - {psf_array.max():.6f}")
        
        return psf_array
    
    def _extract_psf_from_beads(self, bead_dir: str) -> np.ndarray:
        """Extract PSF from bead directory using psf.py functionality."""
        bead_path = Path(bead_dir)
        
        print(f"Extracting PSF from bead data in {bead_path}...")
        
        try:
            # Try to use build_psf_bank first (handles multiple subdirs)
            psf_bank = build_psf_bank(bead_path)
            
            # Use the first available PSF from the bank
            psf_key = list(psf_bank.keys())[0]
            psf_tensor = psf_bank[psf_key]
            psf_array = psf_tensor.detach().cpu().numpy().astype(np.float32)
            
            print(f"PSF extracted using build_psf_bank (mode: {psf_key})")
            
        except Exception as e:
            print(f"build_psf_bank failed: {e}")
            print("Trying direct image loading...")
            
            # Fallback: load images directly from directory
            bead_images = _load_grayscale_images(bead_path)
            
            if not bead_images:
                raise ValueError(f"No bead images found in {bead_path}")
            
            psf_array = estimate_psf_from_beads(bead_images, crop_size=33)
            print(f"PSF extracted from {len(bead_images)} bead images")
        
        return psf_array
    
    def save_psf(self, output_path: str) -> None:
        """Save PSF to file for future use."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save PSF as TIFF
        tifffile.imwrite(output_path, self.psf)
        
        # Also save as PNG for visualization
        png_path = output_path.with_suffix('.png')
        psf_norm = (self.psf - self.psf.min()) / (self.psf.max() - self.psf.min())
        psf_uint8 = (psf_norm * 255).astype(np.uint8)
        Image.fromarray(psf_uint8).save(png_path)
        
        print(f"PSF saved to {output_path}")
        print(f"PSF visualization saved to {png_path}")
        print(f"PSF shape: {self.psf.shape}, sum: {self.psf.sum():.6f}")
        print(f"PSF range: {self.psf.min():.6f} - {self.psf.max():.6f}")
    
    def _extract_patches(self, image: np.ndarray) -> Dict[int, np.ndarray]:
        """Extract overlapping patches from image."""
        patches = {}
        patch_id = 0
        
        h, w = image.shape
        patches_y = (h - self.patch_size) // self.stride + 1
        patches_x = (w - self.patch_size) // self.stride + 1
        
        print(f"Extracting patches: {patches_y} rows × {patches_x} cols = {patches_y * patches_x} patches")
        
        for row in range(patches_y):
            for col in range(patches_x):
                y_start = row * self.stride
                x_start = col * self.stride
                y_end = y_start + self.patch_size
                x_end = x_start + self.patch_size
                
                patch = image[y_start:y_end, x_start:x_end]
                patches[patch_id] = patch
                patch_id += 1
        
        return patches
    
    def _reconstruct_from_patches(
        self, 
        patches: Dict[int, np.ndarray], 
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Reconstruct full FOV image from patches with seamless blending."""
        h, w = original_shape
        patches_y = (h - self.patch_size) // self.stride + 1
        patches_x = (w - self.patch_size) // self.stride + 1
        
        # Initialize canvas
        canvas = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        # Process all patches
        for patch_id, patch_data in patches.items():
            row = patch_id // patches_x
            col = patch_id % patches_x
            
            # Calculate position in the original image
            y_start = row * self.stride
            x_start = col * self.stride
            y_end = y_start + self.patch_size
            x_end = x_start + self.patch_size
            
            # Create feathering weights for seamless blending
            patch_weight = np.ones((self.patch_size, self.patch_size), dtype=np.float32)
            
            # Feather the edges to create smooth blending
            feather_size = self.stride // 2
            
            # Top edge feathering
            if row > 0:
                patch_weight[:feather_size, :] *= np.linspace(0, 1, feather_size)[:, np.newaxis]
            
            # Bottom edge feathering
            if row < patches_y - 1:
                patch_weight[-feather_size:, :] *= np.linspace(1, 0, feather_size)[:, np.newaxis]
            
            # Left edge feathering
            if col > 0:
                patch_weight[:, :feather_size] *= np.linspace(0, 1, feather_size)[np.newaxis, :]
            
            # Right edge feathering
            if col < patches_x - 1:
                patch_weight[:, -feather_size:] *= np.linspace(1, 0, feather_size)[np.newaxis, :]
            
            # Add patch to canvas
            canvas[y_start:y_end, x_start:x_end] += patch_data * patch_weight
            weight_map[y_start:y_end, x_start:x_end] += patch_weight
        
        # Normalize by weights
        mask = weight_map > 0
        canvas[mask] = canvas[mask] / weight_map[mask]
        
        # Apply slight smoothing to reduce any remaining artifacts
        canvas = gaussian_filter(canvas, sigma=0.5)
        
        return canvas
    
    def process_image(self, wf_image: np.ndarray) -> np.ndarray:
        """Process full FOV image using Richardson-Lucy deconvolution."""
        print(f"Processing image with Richardson-Lucy ({self.iterations} iterations)")
        
        # Extract patches
        patches = self._extract_patches(wf_image)
        
        # Process each patch
        processed_patches = {}
        
        for patch_id, patch in tqdm(patches.items(), desc="RL patches"):
            processed_patches[patch_id] = richardson_lucy_restore(
                image=patch,
                psf=self.psf,
                num_iter=self.iterations,
                clip=self.clip
            )
        
        # Reconstruct full FOV
        reconstructed = self._reconstruct_from_patches(processed_patches, wf_image.shape)
        
        return reconstructed
    
    def compute_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        data_range = float(target.max() - target.min()) if target.size > 0 else 1.0
        
        return {
            "psnr": Metrics.psnr(pred, target, data_range=data_range),
            "ssim": Metrics.ssim(pred, target, data_range=data_range),
            "frc": Metrics.frc(pred, target, threshold=0.143),
        }
    
    def normalize_for_visualization(self, image: np.ndarray, percentile_clip: float = 99.5) -> np.ndarray:
        """Normalize image for visualization."""
        image_norm = np.clip(image, 0, np.percentile(image, percentile_clip))
        if image_norm.max() > image_norm.min():
            image_norm = (image_norm - image_norm.min()) / (image_norm.max() - image_norm.min())
        return (image_norm * 255).astype(np.uint8)
    
    def process_dataset(
        self,
        input_dir: str,
        gt_dir: str,
        output_dir: str,
        max_images: Optional[int] = None,
        create_visualizations: bool = True
    ) -> List[Dict[str, float]]:
        """
        Process a full dataset of images.
        
        Args:
            input_dir: Directory containing WF input images
            gt_dir: Directory containing ground truth 2P images
            output_dir: Directory to save results
            max_images: Maximum number of images to process
            create_visualizations: Whether to create comparison visualizations
            
        Returns:
            List of results dictionaries with metrics for each image
        """
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image pairs
        input_dir = Path(input_dir)
        gt_dir = Path(gt_dir)
        
        # Find matching image pairs
        input_files = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.tif"))
        if max_images:
            input_files = input_files[:max_images]
        
        print(f"Found {len(input_files)} input images")
        
        # Results storage
        all_results = []
        
        # Process each image pair
        for input_path in tqdm(input_files, desc="Processing images"):
            # Find corresponding GT file
            gt_path = gt_dir / input_path.name
            if not gt_path.exists():
                print(f"Warning: No GT file found for {input_path.name}")
                continue
            
            # Load images
            if input_path.suffix.lower() in ['.png']:
                wf_image = np.array(Image.open(input_path)).astype(np.float32)
                gt_image = np.array(Image.open(gt_path)).astype(np.float32)
            else:
                wf_image = tifffile.imread(str(input_path)).astype(np.float32)
                gt_image = tifffile.imread(str(gt_path)).astype(np.float32)
            
            # Ensure single channel
            if wf_image.ndim == 3 and wf_image.shape[0] == 1:
                wf_image = wf_image[0]
            if gt_image.ndim == 3 and gt_image.shape[0] == 1:
                gt_image = gt_image[0]
            
            print(f"Processing {input_path.name}: {wf_image.shape} -> {gt_image.shape}")
            
            # Process with Richardson-Lucy
            try:
                rl_result = self.process_image(wf_image)
                rl_metrics = self.compute_metrics(rl_result, gt_image)
                
                # Store results
                result = {
                    "image": input_path.name,
                    "method": "richardson_lucy",
                    **rl_metrics
                }
                all_results.append(result)
                
                print(f"RL metrics: PSNR={rl_metrics['psnr']:.2f}, SSIM={rl_metrics['ssim']:.3f}, FRC={rl_metrics['frc']:.3f}")
                
                # Create visualization if requested
                if create_visualizations:
                    # Normalize images for visualization
                    wf_norm = self.normalize_for_visualization(wf_image)
                    rl_norm = self.normalize_for_visualization(rl_result)
                    gt_norm = self.normalize_for_visualization(gt_image)
                    
                    # Create comparison: WF | RL | GT
                    comparison = np.concatenate([wf_norm, rl_norm, gt_norm], axis=1)
                    
                    # Save comparison
                    comparison_path = output_dir / f"{input_path.stem}_rl_comparison.png"
                    Image.fromarray(comparison).save(comparison_path)
                    print(f"Comparison saved to {comparison_path}")
                
                # Save individual result
                result_path = output_dir / f"{input_path.stem}_rl_result.tif"
                tifffile.imwrite(result_path, rl_result.astype(np.float32))
                
            except Exception as e:
                print(f"Error processing {input_path.name}: {e}")
                continue
        
        return all_results
    
    def save_results(self, results: List[Dict[str, float]], output_dir: str) -> None:
        """Save results and summary statistics."""
        output_dir = Path(output_dir)
        
        if not results:
            print("No results to save.")
            return
        
        import pandas as pd
        
        df = pd.DataFrame(results)
        
        # Compute summary statistics
        summary = df[['psnr', 'ssim', 'frc']].agg(['mean', 'std', 'min', 'max'])
        
        print("\n" + "="*60)
        print("RICHARDSON-LUCY COMPREHENSIVE RESULTS")
        print("="*60)
        print("\nSummary Statistics:")
        print(summary)
        
        # Save detailed results
        results_path = output_dir / "rl_results.csv"
        df.to_csv(results_path, index=False)
        print(f"\nDetailed results saved to {results_path}")
        
        # Save summary
        summary_path = output_dir / "rl_summary.csv"
        summary.to_csv(summary_path)
        print(f"Summary statistics saved to {summary_path}")
        
        # Print individual results
        print("\nIndividual Results:")
        for result in results:
            print(f"  {result['image']}: PSNR={result['psnr']:.2f}, SSIM={result['ssim']:.3f}, FRC={result['frc']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Richardson-Lucy Deconvolution Baseline")
    
    # PSF options (mutually exclusive)
    psf_group = parser.add_mutually_exclusive_group(required=True)
    psf_group.add_argument("--psf-path", help="Path to existing PSF file")
    psf_group.add_argument("--bead-dir", help="Directory containing bead data for PSF extraction")
    
    # Required arguments
    parser.add_argument("--input-dir", required=True, help="Input WF images directory")
    parser.add_argument("--gt-dir", required=True, help="Ground truth 2P images directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    # Processing options
    parser.add_argument("--device", default="cuda", help="Computation device")
    parser.add_argument("--patch-size", type=int, default=256, help="Patch size")
    parser.add_argument("--stride", type=int, default=128, help="Stride between patches")
    parser.add_argument("--iterations", type=int, default=30, help="RL iterations")
    parser.add_argument("--clip", action="store_true", default=True, help="Clip negative values")
    parser.add_argument("--max-images", type=int, help="Maximum number of images to process")
    parser.add_argument("--create-visualizations", action="store_true", default=True, 
                       help="Create comparison visualizations")
    
    # PSF extraction options
    parser.add_argument("--psf-method", default="average", 
                       choices=["average", "bank"],
                       help="PSF extraction method")
    parser.add_argument("--save-psf", help="Save extracted PSF to this path")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize comprehensive RL baseline
    rl_comprehensive = RichardsonLucyComprehensive(
        psf_path=args.psf_path,
        bead_dir=args.bead_dir,
        device=args.device,
        patch_size=args.patch_size,
        stride=args.stride,
        iterations=args.iterations,
        clip=args.clip,
        psf_method=args.psf_method
    )
    
    # Save PSF if requested
    if args.save_psf:
        rl_comprehensive.save_psf(args.save_psf)
    elif args.bead_dir:
        # Auto-save PSF when extracted from beads
        psf_path = output_dir / "extracted_psf.tif"
        rl_comprehensive.save_psf(str(psf_path))
    
    # Process dataset
    print("\nProcessing dataset with Richardson-Lucy...")
    results = rl_comprehensive.process_dataset(
        input_dir=args.input_dir,
        gt_dir=args.gt_dir,
        output_dir=str(output_dir),
        max_images=args.max_images,
        create_visualizations=args.create_visualizations
    )
    
    # Save results
    rl_comprehensive.save_results(results, str(output_dir))
    
    # Summary
    print(f"\n{'='*60}")
    print("RICHARDSON-LUCY BASELINE COMPLETED!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Individual comparisons: {output_dir}/*_rl_comparison.png")
    print(f"Summary results: {output_dir}/rl_results.csv")
    print(f"Summary statistics: {output_dir}/rl_summary.csv")
    
    if args.bead_dir:
        print(f"Extracted PSF saved to: {output_dir}/extracted_psf.tif")
    
    print(f"\nNext steps:")
    print(f"1. Review the results in {output_dir}")
    print(f"2. Compare with other baseline methods")
    print(f"3. Run comprehensive comparison with PKL-DG methods")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
