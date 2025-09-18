import math

import torch

from pkl_dg.guidance import (
    AdaptiveSchedule,
    AnscombeGuidance,
    L2Guidance,
    PKLGuidance,
)


class MockForwardModel:
    def __init__(self, background: float = 0.0):
        self.background = background

    def apply_psf(self, x: torch.Tensor) -> torch.Tensor:
        # Identity operator for tests
        return x

    def apply_psf_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        # Adjoint of identity is identity
        return y


def test_apply_guidance_step_subtracts_gradient():
    g = PKLGuidance()
    x0_hat = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    grad = torch.ones_like(x0_hat) * 0.5
    lam = 0.2
    x_next = g.apply_guidance(x0_hat, grad, lam)
    expected = x0_hat - lam * grad
    assert torch.allclose(x_next, expected, atol=1e-7)


def test_pkl_guidance_gradient_identity_forward():
    fm = MockForwardModel(background=0.1)
    g = PKLGuidance(epsilon=1e-6)
    x0_hat = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    y = torch.tensor([[[[0.9, 1.8], [2.7, 3.6]]]], dtype=torch.float32)

    grad = g.compute_gradient(x0_hat, y, fm, t=500)

    Ax = x0_hat  # identity PSF
    Ax_plus_B = Ax + fm.background
    ratio = y / (Ax_plus_B + g.epsilon)
    expected = 1.0 - ratio

    assert grad.shape == x0_hat.shape
    assert torch.allclose(grad, expected, atol=1e-6)


def test_l2_guidance_gradient_identity_forward():
    fm = MockForwardModel(background=0.2)
    g = L2Guidance()
    x0_hat = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    y = torch.tensor([[[[0.8, 1.7], [2.9, 3.5]]]], dtype=torch.float32)

    grad = g.compute_gradient(x0_hat, y, fm, t=100)

    Ax_plus_B = x0_hat + fm.background
    expected = -(Ax_plus_B - y)  # L2 guidance returns negative gradient

    assert grad.shape == x0_hat.shape
    assert torch.allclose(grad, expected, atol=1e-6)


def test_anscombe_guidance_gradient_identity_forward():
    fm = MockForwardModel(background=0.0)
    g = AnscombeGuidance(epsilon=1e-6)
    # Positive intensities to avoid sqrt issues
    x0_hat = torch.tensor([[[[1.5, 2.5], [3.5, 4.5]]]], dtype=torch.float32)
    y = torch.tensor([[[[1.2, 2.4], [3.4, 4.4]]]], dtype=torch.float32)

    grad = g.compute_gradient(x0_hat, y, fm, t=250)

    Ax_plus_B = x0_hat
    y_a = g.anscombe_transform(y)
    Ax_a = g.anscombe_transform(Ax_plus_B)
    residual_a = Ax_a - y_a
    chain = residual_a * g.anscombe_derivative(Ax_plus_B)
    expected = chain  # Adjoint of identity is identity

    assert torch.allclose(grad, expected, atol=1e-6)


def test_adaptive_schedule_behavior():
    sched = AdaptiveSchedule(lambda_base=0.1, T_threshold=800, epsilon_lambda=1e-3, T_total=1000)
    grad = torch.ones((1, 1, 2, 2), dtype=torch.float32)  # L2 norm = 2

    # For t < T_threshold, warmup should be 1.0
    lam_early = sched.get_lambda_t(grad, t=750)
    expected_step = 0.1 / (2.0 + 1e-3)
    assert math.isclose(lam_early, expected_step, rel_tol=1e-6, abs_tol=1e-6)

    # For t midway between threshold and total, warmup scales linearly
    lam_mid = sched.get_lambda_t(grad, t=900)
    assert math.isclose(lam_mid, 0.5 * expected_step, rel_tol=1e-6, abs_tol=1e-6)

    # At t == T_total, warmup is 0
    lam_end = sched.get_lambda_t(grad, t=1000)
    assert math.isclose(lam_end, 0.0, abs_tol=1e-12)


