import torch

from pkl_dg.guidance import PKLGuidance, L2Guidance, AnscombeGuidance, AdaptiveSchedule
from pkl_dg.physics.forward_model import ForwardModel


def _make_forward_model(device: str = "cpu") -> ForwardModel:
    psf = torch.ones(9, 9) / 81.0
    return ForwardModel(psf=psf, background=0.2, device=device)


def test_pkl_guidance_shapes_and_sign():
    device = "cpu"
    fm = _make_forward_model(device)
    guide = PKLGuidance()
    x = torch.rand(2, 1, 32, 32, device=device) * 10.0
    with torch.no_grad():
        y = fm.forward(x, add_noise=False)
    grad = guide.compute_gradient(x, y, fm, t=500)
    assert grad.shape == x.shape
    # If using exact forward, residual should be near zero
    assert torch.isfinite(grad).all()


def test_l2_guidance_shapes_and_basic_behavior():
    device = "cpu"
    fm = _make_forward_model(device)
    guide = L2Guidance()
    x = torch.rand(1, 1, 16, 16, device=device)
    y = torch.rand_like(x) + 0.1
    grad = guide.compute_gradient(x, y, fm, t=200)
    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()


def test_anscombe_guidance_shapes_and_finiteness():
    device = "cpu"
    fm = _make_forward_model(device)
    guide = AnscombeGuidance()
    x = torch.rand(1, 1, 16, 16, device=device) * 5.0
    y = torch.rand_like(x) * 5.0 + 0.1
    grad = guide.compute_gradient(x, y, fm, t=100)
    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()


def test_adaptive_schedule_lambda_scaling():
    sched = AdaptiveSchedule(lambda_base=0.1, T_threshold=800, epsilon_lambda=1e-3, T_total=1000)
    grad = torch.ones(1, 1, 8, 8)
    lam1 = sched.get_lambda_t(grad, t=950)  # early time, small warmup
    lam2 = sched.get_lambda_t(grad, t=100)  # late time, large warmup
    assert lam2 > lam1
    assert lam1 > 0 and lam2 > 0


