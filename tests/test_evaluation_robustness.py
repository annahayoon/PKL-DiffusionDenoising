import torch

from pkl_dg.evaluation.robustness import RobustnessTests
from pkl_dg.models.sampler import DDIMSampler
from pkl_dg.data import IntensityToModel
from pkl_dg.guidance import PKLGuidance, AdaptiveSchedule
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel


class _DummyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _DummyTrainer(torch.nn.Module):
    def __init__(self, T: int = 50):
        super().__init__()
        self.model = _DummyNet()
        alphas = torch.linspace(1.0, 1e-3, T)
        self.register_buffer("alphas_cumprod", alphas)


def _make_sampler(device: str = "cpu") -> DDIMSampler:
    trainer = _DummyTrainer(T=50).to(device)
    psf_t = torch.ones(9, 9) / 81.0
    fm = ForwardModel(psf=psf_t, background=0.0, device=device)
    guidance = PKLGuidance()
    schedule = AdaptiveSchedule(lambda_base=0.05, T_threshold=40, T_total=50)
    transform = IntensityToModel(minIntensity=0, maxIntensity=1000)
    sampler = DDIMSampler(
        model=trainer,
        forward_model=fm,
        guidance_strategy=guidance,
        schedule=schedule,
        transform=transform,
        num_timesteps=50,
        ddim_steps=5,
        eta=0.0,
    )
    return sampler


def test_psf_mismatch_test_runs_and_restores_psf():
    device = "cpu"
    sampler = _make_sampler(device)
    original_psf_ref = sampler.forward_model.psf.clone()
    y = torch.rand(1, 1, 16, 16, device=device) + 0.1
    psf_true = PSF()

    out = RobustnessTests.psf_mismatch_test(sampler, y, psf_true, mismatch_factor=1.2)
    assert out.shape == (1, 1, 16, 16)
    # Ensure PSF restored
    assert torch.allclose(sampler.forward_model.psf, original_psf_ref)


def test_alignment_error_test_runs():
    device = "cpu"
    sampler = _make_sampler(device)
    y = torch.rand(1, 1, 16, 16, device=device) + 0.2
    out = RobustnessTests.alignment_error_test(sampler, y, shift_pixels=0.5)
    assert out.shape == (1, 1, 16, 16)


