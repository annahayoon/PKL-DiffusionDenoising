import torch
from typing import Any


class RobustnessTests:
    """Robustness evaluation tests."""

    @staticmethod
    def psf_mismatch_test(
        sampler: "DDIMSampler",
        y: torch.Tensor,
        psf_true: "PSF",
        mismatch_factor: float = 1.1,
    ) -> torch.Tensor:
        """
        Test robustness to PSF mismatch by broadening the PSF used in the forward model.

        Args:
            sampler: DDIM sampler instance
            y: Measurement with true PSF
            psf_true: True PSF object
            mismatch_factor: PSF broadening factor

        Returns:
            Reconstruction with mismatched PSF as a torch.Tensor
        """
        # Create mismatched PSF
        psf_mismatched = psf_true.broaden(mismatch_factor)

        # Swap PSF in sampler's forward model using setter to clear cache
        original_psf = sampler.forward_model.psf
        sampler.forward_model.set_psf(psf_mismatched.to_torch(
            device=sampler.forward_model.device
        ))

        try:
            # Run reconstruction
            shape = (y.shape[0], 1, y.shape[-2], y.shape[-1]) if y.dim() == 3 else y.shape
            reconstruction = sampler.sample(y, shape, device=sampler.forward_model.device, verbose=False)
        finally:
            # Restore original PSF using setter to clear cache
            sampler.forward_model.set_psf(original_psf.squeeze(0).squeeze(0))

        return reconstruction

    @staticmethod
    def alignment_error_test(
        sampler: "DDIMSampler",
        y: torch.Tensor,
        shift_pixels: float = 0.5,
    ) -> torch.Tensor:
        """
        Test robustness to alignment errors by applying a small affine shift.

        Args:
            sampler: DDIM sampler instance
            y: Original measurement
            shift_pixels: Subpixel shift amount (in pixels)

        Returns:
            Reconstruction with shifted input
        """
        try:
            import kornia  # type: ignore
            use_kornia = True
        except Exception:
            use_kornia = False

        # Build affine matrix (normalized translation)
        theta = torch.tensor(
            [
                [1, 0, shift_pixels / y.shape[-1]],
                [0, 1, shift_pixels / y.shape[-2]],
            ],
            dtype=torch.float32,
            device=y.device,
        ).unsqueeze(0)

        # Prepare input shape [B, C, H, W]
        if y.dim() == 3:
            y_input = y.unsqueeze(0)  # [1, C, H, W]
        else:
            y_input = y

        if use_kornia:
            # Apply shift via kornia warp_affine
            y_shifted = kornia.geometry.transform.warp_affine(
                y_input,
                theta,
                dsize=(y.shape[-2], y.shape[-1]),
                mode="bilinear",
                padding_mode="border",
            )
        else:
            # Fallback: use torch grid_sample with affine_grid
            import torch.nn.functional as F
            grid = F.affine_grid(theta, size=y_input.size(), align_corners=False)
            y_shifted = F.grid_sample(y_input, grid, mode="bilinear", padding_mode="border", align_corners=False)

        # Run reconstruction
        shape = y_shifted.shape
        reconstruction = sampler.sample(y_shifted, shape, device=y_shifted.device, verbose=False)

        return reconstruction


