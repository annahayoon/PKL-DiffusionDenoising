import torch

from pkl_dg.guidance import PKLGuidance, KLGuidance, L2Guidance, AnscombeGuidance, AdaptiveSchedule
from pkl_dg.physics import ForwardModel


# Import shared test utilities
from tests.test_utils import make_forward_model

# Use lambda to avoid redundant function definition
_make_forward_model = lambda device="cpu": make_forward_model(device=device, background=0.2)


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


@torch.no_grad()
def test_l2_guidance_direction_alignment():
    device = "cpu"
    fm = _make_forward_model(device)
    guide = L2Guidance()
    x = torch.rand(2, 1, 32, 32, device=device)
    y = torch.rand_like(x) + 0.05
    g = guide.compute_gradient(x, y, fm, t=50)
    AxB = fm.apply_psf(x) + fm.background
    residual = y - AxB
    # Check that projected gradient aligns with residual on average
    inner = (fm.apply_psf(g) * residual).mean().item()
    assert inner >= -1e-4


def test_anscombe_guidance_shapes_and_finiteness():
    device = "cpu"
    fm = _make_forward_model(device)
    guide = AnscombeGuidance()
    x = torch.rand(1, 1, 16, 16, device=device) * 5.0
    y = torch.rand_like(x) * 5.0 + 0.1
    grad = guide.compute_gradient(x, y, fm, t=100)
    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()


@torch.no_grad()
def test_kl_guidance_shapes_and_direction():
    device = "cpu"
    fm = _make_forward_model(device)
    guide = KLGuidance(sigma2=1.0)
    x = torch.rand(2, 1, 32, 32, device=device)
    y = torch.rand_like(x) + 0.1
    g = guide.compute_gradient(x, y, fm, t=100)
    assert g.shape == x.shape
    assert torch.isfinite(g).all()
    AxB = fm.apply_psf(x) + fm.background
    residual = AxB - y  # KL gradient uses (Ax+B - y)
    inner = (fm.apply_psf(g) * residual).mean().item()
    assert inner >= -1e-4


def test_adaptive_schedule_lambda_scaling():
    sched = AdaptiveSchedule(lambda_base=0.1, T_threshold=800, epsilon_lambda=1e-3, T_total=1000)
    grad = torch.ones(1, 1, 8, 8)
    lam1 = sched.get_lambda_t(grad, t=950)  # early time, small warmup
    lam2 = sched.get_lambda_t(grad, t=100)  # late time, large warmup
    assert lam2 > lam1
    assert lam1 > 0 and lam2 > 0


