import torch
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm

try:
    import kornia.filters as K_filters
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False


class ForwardModel:
    """WF to 2P forward model with PSF convolution and noise."""

    def __init__(self, psf: torch.Tensor, background: float = 0.0, device: str = "cuda", 
                 common_sizes: Optional[list] = None, read_noise_sigma: float = 0.0,
                 psf_pixel_size_xy_nm: float = None, target_pixel_size_xy_nm: float = None):
        """
        Initialize forward model.

        Args:
            psf: Point spread function tensor
            background: Background intensity level
            device: Computation device
            common_sizes: List of (height, width) tuples for pre-computing FFTs
            read_noise_sigma: Read noise standard deviation
            psf_pixel_size_xy_nm: Original PSF pixel size in nm (if known)
            target_pixel_size_xy_nm: Target image pixel size in nm (if different from PSF)
        """
        self.device = device
        self.background = background
        # Read noise standard deviation (counts). When > 0, can be used by guidance.
        self.read_noise_sigma = float(read_noise_sigma)
        
        # Store pixel size information
        self.psf_pixel_size_xy_nm = psf_pixel_size_xy_nm
        self.target_pixel_size_xy_nm = target_pixel_size_xy_nm

        # Handle PSF scaling if pixel sizes are different
        processed_psf = self._process_psf_for_pixel_size(psf)
        
        # Ensure PSF is shape [1, 1, H, W]
        self.psf = processed_psf.to(device)
        if self.psf.ndim == 2:
            self.psf = self.psf.unsqueeze(0).unsqueeze(0)
        # Device/dtype-aware cache: (H, W, dtype, device_str) -> psf_fft tensor
        self._psf_fft_cache = {}
        # Device-agnostic base cache to avoid recomputing FFTs when dtype changes:
        # (H, W) -> psf_fft tensor stored on CPU in high precision
        self._psf_fft_cache_base = {}
        
        # Pre-compute FFTs for common image sizes
        if common_sizes is None:
            # Default common sizes for microscopy (powers of 2 and common patch sizes)
            common_sizes = [(256, 256), (512, 512), (128, 128), (64, 64), (1024, 1024)]
        
        self._precompute_common_ffts(common_sizes)
    
    def _process_psf_for_pixel_size(self, psf: torch.Tensor) -> torch.Tensor:
        """Process PSF to handle pixel size scaling if needed."""
        # Extract pixel size info from PSF tensor if available
        if hasattr(psf, 'pixel_size_xy_nm'):
            self.psf_pixel_size_xy_nm = psf.pixel_size_xy_nm
        
        # If we have both PSF pixel size and target pixel size, scale if needed
        if (self.psf_pixel_size_xy_nm is not None and 
            self.target_pixel_size_xy_nm is not None and
            abs(self.psf_pixel_size_xy_nm - self.target_pixel_size_xy_nm) > 0.01):
            
            print(f"ForwardModel: Scaling PSF from {self.psf_pixel_size_xy_nm} nm "
                  f"to {self.target_pixel_size_xy_nm} nm")
            
            # Convert to numpy, scale, convert back
            from .psf import PSF
            psf_np = psf.detach().cpu().numpy()
            psf_obj = PSF(psf_array=psf_np, pixel_size_xy_nm=self.psf_pixel_size_xy_nm)
            scaled_psf_obj = psf_obj.scale_for_pixel_size(self.target_pixel_size_xy_nm)
            return torch.from_numpy(scaled_psf_obj.psf)
        
        return psf


    def _precompute_common_ffts(self, common_sizes: list) -> None:
        """Pre-compute PSF FFTs for common image sizes to improve runtime performance."""
        # Only precompute base FFTs (device-agnostic, high precision)
        # Device-specific FFTs will be materialized on-demand
        for height, width in tqdm(common_sizes, desc="Pre-computing PSF FFTs", leave=False):
            try:
                # Pre-compute base FFT (CPU, float32) for this size
                base_key = (height, width)
                if base_key not in self._psf_fft_cache_base:
                    # Force computation of base FFT
                    self._get_psf_fft(height, width, torch.float32, torch.device('cpu'))
            except Exception:
                # Skip if there are memory or compatibility issues
                continue

    def set_psf(self, psf: torch.Tensor, common_sizes: Optional[list] = None) -> None:
        """Update PSF and clear cached FFTs to maintain correctness.

        Args:
            psf: New PSF tensor, will be moved to model device and shaped to [1,1,H,W]
            common_sizes: Optional list of sizes to pre-compute FFTs for
        """
        self.psf = psf.to(self.device)
        if self.psf.ndim == 2:
            self.psf = self.psf.unsqueeze(0).unsqueeze(0)
        # Invalidate all caches since kernel changed
        self._psf_fft_cache.clear()
        self._psf_fft_cache_base.clear()
        
        # Re-precompute common FFTs for new PSF
        if common_sizes is None:
            common_sizes = [(256, 256), (512, 512), (128, 128), (64, 64), (1024, 1024)]
        self._precompute_common_ffts(common_sizes)

    def _get_psf_fft(self, height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return cached PSF FFT for given size/dtype or compute and store it.

        Optimizations:
        - Maintain a high-precision CPU base cache per (H, W)
        - Materialize device/dtype-specific tensors from the base cache to avoid recomputation
        """
        device_key = (height, width, str(dtype), str(device))
        cached_dev = self._psf_fft_cache.get(device_key)
        if cached_dev is not None:
            return cached_dev

        # Try base cache first (CPU, high precision)
        base_key = (height, width)
        base_fft = self._psf_fft_cache_base.get(base_key)

        if base_fft is None:
            # Build base FFT on CPU in float32 for stability
            psf_h, psf_w = self.psf.shape[-2:]
            psf_padded = torch.zeros((1, 1, height, width), device="cpu", dtype=torch.float32)
            psf_src = self.psf.to(device="cpu", dtype=torch.float32)

            # If image smaller than PSF, center-crop PSF; otherwise place PSF then roll to center
            if height < psf_h or width < psf_w:
                start_h = max((psf_h - height) // 2, 0)
                start_w = max((psf_w - width) // 2, 0)
                end_h = start_h + min(psf_h, height)
                end_w = start_w + min(psf_w, width)
                cropped = psf_src[..., start_h:end_h, start_w:end_w]
                ph, pw = cropped.shape[-2:]
                psf_padded[..., :ph, :pw] = cropped
                psf_padded = torch.roll(psf_padded, shifts=(-(ph // 2), -(pw // 2)), dims=(-2, -1))
            else:
                psf_padded[..., :psf_h, :psf_w] = psf_src
                psf_padded = torch.roll(psf_padded, shifts=(-(psf_h // 2), -(psf_w // 2)), dims=(-2, -1))

            base_fft = torch.fft.rfft2(psf_padded)
            self._psf_fft_cache_base[base_key] = base_fft

        # Materialize for target device (cast at multiplication time if needed)
        dev_fft = base_fft.to(device)
        self._psf_fft_cache[device_key] = dev_fft
        return dev_fft

    def _fft_convolve(self, x: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        """
        Circular convolution via FFT with kernel centered at origin.
        Ensures discrete adjointness with correlation for apply_psf_adjoint.
        """
        _, _, img_h, img_w = x.shape
        x_fft = torch.fft.rfft2(x)
        psf_fft = self._get_psf_fft(img_h, img_w, x.dtype, x.device)
        y_fft = x_fft * psf_fft
        y = torch.fft.irfft2(y_fft, s=(img_h, img_w))
        return y

    def apply_psf(self, x: torch.Tensor, use_kornia: bool = False) -> torch.Tensor:
        """Apply PSF convolution using FFT with padding or Kornia filters."""
        if use_kornia and KORNIA_AVAILABLE:
            return self._kornia_convolve(x, self.psf)
        return self._fft_convolve(x, self.psf)

    def _kornia_convolve(self, x: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        """Alternative convolution using Kornia for differentiable operations."""
        if not KORNIA_AVAILABLE:
            raise ImportError("Kornia not available. Use FFT convolution instead.")
        
        # Kornia expects kernel in shape [1, 1, H, W] 
        kernel = psf if psf.ndim == 4 else psf.unsqueeze(0).unsqueeze(0)
        
        # Use kornia's filter2d which handles padding automatically
        return K_filters.filter2d(x, kernel, border_type='reflect')

    def apply_psf_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint (correlation) using conjugate multiplication in Fourier domain."""
        _, _, img_h, img_w = y.shape
        y_fft = torch.fft.rfft2(y)
        psf_fft = self._get_psf_fft(img_h, img_w, y.dtype, y.device)
        at_fft = y_fft * torch.conj(psf_fft)
        at = torch.fft.irfft2(at_fft, s=(img_h, img_w))
        return at

    def forward(self, x: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        """
        Full forward model: A(x) + B [+ Poisson noise].
        """
        y = self.apply_psf(x)
        y = y + self.background
        if add_noise:
            y = torch.poisson(torch.clamp(y, min=0))
        return y

    def clear_cache(self) -> None:
        """Clear all FFT caches to free memory."""
        self._psf_fft_cache.clear()
        self._psf_fft_cache_base.clear()
    
    def get_cache_stats(self) -> dict:
        """Return cache statistics for testing and diagnostics."""
        # Calculate memory usage
        base_memory = sum(
            fft.numel() * fft.element_size() 
            for fft in self._psf_fft_cache_base.values()
        ) / 1e6  # MB
        
        device_memory = sum(
            fft.numel() * fft.element_size() 
            for fft in self._psf_fft_cache.values()
        ) / 1e6  # MB
        
        return {
            "base_entries": len(self._psf_fft_cache_base),
            "device_entries": len(self._psf_fft_cache),
            "base_memory_mb": base_memory,
            "device_memory_mb": device_memory,
            "total_memory_mb": base_memory + device_memory,
        }
    
    def optimize_cache_memory(self, max_memory_mb: float = 512) -> None:
        """Optimize cache memory usage by removing least recently used entries."""
        current_stats = self.get_cache_stats()
        
        if current_stats["total_memory_mb"] > max_memory_mb:
            # Simple strategy: clear device cache (can be regenerated from base)
            num_cleared = len(self._psf_fft_cache)
            self._psf_fft_cache.clear()
            print(f"Cleared {num_cleared} device FFT cache entries to save memory")
    
    def batch_apply_psf(self, x_batch: torch.Tensor, use_kornia: bool = False, 
                       chunk_size: int = None) -> torch.Tensor:
        """Apply PSF to a batch of images efficiently with optional chunking.
        
        Args:
            x_batch: Input batch [B, C, H, W]
            use_kornia: Whether to use Kornia for convolution
            chunk_size: Process in chunks to manage memory (None = auto-detect)
            
        Returns:
            PSF-convolved batch [B, C, H, W]
        """
        if x_batch.dim() != 4:
            raise ValueError(f"Expected 4D batch tensor, got {x_batch.dim()}D")
        
        batch_size = x_batch.shape[0]
        
        if batch_size == 1:
            # Single image - use regular method
            return self.apply_psf(x_batch, use_kornia)
        
        # Auto-detect chunk size based on available memory
        if chunk_size is None:
            if torch.cuda.is_available():
                # Estimate memory per image and available memory
                img_memory = x_batch[0].numel() * x_batch.element_size() / 1e9  # GB
                available_memory = torch.cuda.get_device_properties(0).total_memory * 0.5 / 1e9
                chunk_size = max(1, int(available_memory / (img_memory * 4)))  # 4x overhead
                chunk_size = min(chunk_size, batch_size)
            else:
                chunk_size = min(8, batch_size)  # Conservative CPU default
        
        # Process in chunks if needed
        if chunk_size < batch_size:
            results = []
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk = x_batch[i:end_idx]
                result_chunk = self._batch_apply_psf_chunk(chunk, use_kornia)
                results.append(result_chunk)
            return torch.cat(results, dim=0)
        else:
            return self._batch_apply_psf_chunk(x_batch, use_kornia)
    
    def _batch_apply_psf_chunk(self, x_batch: torch.Tensor, use_kornia: bool = False) -> torch.Tensor:
        """Apply PSF to a chunk of images."""
        # For batch processing, we can use the same FFT for all images in the batch
        if use_kornia and KORNIA_AVAILABLE:
            return self._kornia_convolve(x_batch, self.psf)
        
        # Batch FFT convolution - vectorized for efficiency
        _, _, img_h, img_w = x_batch.shape
        
        # Use torch.fft.rfft2 which is optimized for batch operations
        x_fft = torch.fft.rfft2(x_batch, dim=(-2, -1))
        psf_fft = self._get_psf_fft(img_h, img_w, x_batch.dtype, x_batch.device)
        
        # Broadcast PSF FFT to batch dimension efficiently
        y_fft = x_fft * psf_fft.unsqueeze(0)
        y = torch.fft.irfft2(y_fft, s=(img_h, img_w), dim=(-2, -1))
        return y
    
    def parallel_apply_psf_cpu(self, x_list: list, num_workers: int = None) -> list:
        """Apply PSF to multiple images in parallel using CPU multiprocessing.
        
        Args:
            x_list: List of input tensors
            num_workers: Number of worker processes
            
        Returns:
            List of PSF-convolved tensors
        """
        if num_workers is None:
            import os
            num_workers = min(8, os.cpu_count() or 1)
        
        if len(x_list) <= 1 or num_workers <= 1:
            return [self.apply_psf(x) for x in x_list]
        
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        
        # Create a worker function that can be pickled
        def worker_apply_psf(args):
            x, psf_data, background, device_str = args
            # Recreate forward model in worker process
            worker_fm = ForwardModel(psf_data, background, device_str)
            return worker_fm.apply_psf(x)
        
        # Prepare arguments for workers
        psf_data = self.psf.cpu()  # Move PSF to CPU for serialization
        args_list = [(x.cpu(), psf_data, self.background, "cpu") for x in x_list]
        
        # Process in parallel
        try:
            with ProcessPoolExecutor(max_workers=num_workers, 
                                   mp_context=mp.get_context('spawn')) as executor:
                results = list(executor.map(worker_apply_psf, args_list))
            
            # Move results back to original device if needed
            if x_list[0].device != torch.device('cpu'):
                results = [r.to(x_list[0].device) for r in results]
            
            return results
        except Exception as e:
            # Fallback to sequential processing
            print(f"Parallel processing failed: {e}. Falling back to sequential.")
            return [self.apply_psf(x) for x in x_list]


