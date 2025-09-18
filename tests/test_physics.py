import torch
import numpy as np

from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel
from pkl_dg.physics.noise import PoissonNoise, GaussianBackground


def test_psf_normalization():
    psf = PSF()
    assert np.abs(psf.psf.sum() - 1.0) < 1e-6


def test_forward_model_shape():
    psf = PSF()
    forward_model = ForwardModel(psf=psf.to_torch(), background=0.0, device="cpu")

    x = torch.randn(2, 1, 64, 64)
    y = forward_model.forward(x)

    assert y.shape == x.shape


def test_adjoint_operator_inner_product():
    psf = PSF()
    forward_model = ForwardModel(psf=psf.to_torch(), background=0.0, device="cpu")

    # Test <Ax, y> = <x, A^T y>
    x = torch.randn(1, 1, 64, 64)
    y = torch.randn(1, 1, 64, 64)

    Ax = forward_model.apply_psf(x)
    ATy = forward_model.apply_psf_adjoint(y)

    inner1 = (Ax * y).sum()
    inner2 = (x * ATy).sum()

    rel_err = torch.abs(inner1 - inner2) / (torch.abs(inner1) + 1e-12)
    assert rel_err < 1e-4


@torch.no_grad()
def test_poisson_noise_statistics():
    # For a constant signal, Poisson variance ~ mean/gain
    torch.manual_seed(0)
    signal_mean = 10.0
    gain = 2.0
    signal = torch.full((1000, 1, 8, 8), signal_mean)
    noisy = PoissonNoise.add_noise(signal, gain=gain)

    empirical_mean = noisy.mean().item()
    empirical_var = noisy.var(unbiased=False).item()

    # Mean preserved approximately
    assert abs(empirical_mean - signal_mean) / signal_mean < 0.05
    # Variance approximately mean/gain
    expected_var = signal_mean / gain
    assert abs(empirical_var - expected_var) / expected_var < 0.2


@torch.no_grad()
def test_gaussian_background_statistics():
    torch.manual_seed(0)
    base = torch.zeros(2000, 1, 4, 4)
    mean = 1.5
    std = 0.7
    noisy = GaussianBackground.add_background(base, mean=mean, std=std)
    empirical_mean = noisy.mean().item()
    empirical_std = noisy.std(unbiased=False).item()

    assert abs(empirical_mean - mean) / mean < 0.1
    assert abs(empirical_std - std) / std < 0.1


def test_psf_3d_array_central_slice():
    # Create a 3D PSF with known central slice
    z, h, w = 5, 9, 9
    central = np.zeros((h, w), dtype=np.float32)
    central[h // 2, w // 2] = 1.0
    stack = np.random.rand(z, h, w).astype(np.float32) * 0.01
    stack[z // 2] = central

    psf = PSF(psf_array=stack)
    # PSF should have taken central slice and normalized
    assert psf.psf.shape == (h, w)
    np.testing.assert_allclose(psf.psf.sum(), 1.0, rtol=1e-6, atol=1e-6)


