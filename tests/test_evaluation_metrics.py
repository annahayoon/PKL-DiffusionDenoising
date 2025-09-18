import numpy as np
import importlib.util
import sys
from pathlib import Path

# Import Metrics class directly from the .py file (not the directory)
spec = importlib.util.spec_from_file_location(
    "metrics_module", 
    Path(__file__).parent.parent / "pkl_dg" / "evaluation" / "metrics.py"
)
metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_module)
Metrics = metrics_module.Metrics


def test_psnr_basic_invariance_and_range():
    target = np.ones((32, 32), dtype=np.float32)
    pred_same = np.ones_like(target)
    pred_noisy = target + 0.1 * np.random.RandomState(0).randn(*target.shape).astype(np.float32)

    psnr_same = Metrics.psnr(pred_same, target)
    psnr_noisy = Metrics.psnr(pred_noisy, target)

    assert np.isfinite(psnr_same)
    assert psnr_same > psnr_noisy
    assert psnr_same > 50  # identical images have very high PSNR


def test_ssim_monotonicity():
    target = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    pred_same = target.copy()
    pred_blur = (target + np.roll(target, 1, axis=0)) / 2

    ssim_same = Metrics.ssim(pred_same, target)
    ssim_blur = Metrics.ssim(pred_blur, target)

    assert 0 <= ssim_blur <= 1
    assert ssim_same >= ssim_blur


def test_frc_threshold_behavior():
    # Construct target and a progressively noisier pred to force FRC drop sooner
    rng = np.random.RandomState(0)
    target = rng.rand(64, 64).astype(np.float32)
    pred = (target * 0.5 + 0.5 * rng.rand(64, 64)).astype(np.float32)

    res_0143 = Metrics.frc(pred, target, threshold=0.143)
    res_0200 = Metrics.frc(pred, target, threshold=0.2)

    # Higher threshold should be crossed earlier (smaller radius index)
    assert res_0200 <= res_0143


def test_sar_and_hausdorff_distance():
    img = np.zeros((32, 32), dtype=np.float32)
    img[10:20, 10:20] = 1.0

    artifact_mask = np.zeros_like(img, dtype=bool)
    artifact_mask[0:5, 0:5] = True

    sar = Metrics.sar(img, artifact_mask)
    assert np.isfinite(sar)
    assert sar > 0  # signal power > artifact power

    pred_mask = np.zeros_like(img, dtype=bool)
    pred_mask[10:20, 10:20] = True
    target_mask = np.zeros_like(img, dtype=bool)
    target_mask[12:22, 12:22] = True

    hd = Metrics.hausdorff_distance(pred_mask, target_mask)
    assert np.isfinite(hd)
    assert hd >= 0


