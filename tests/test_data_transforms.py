import torch

from pkl_dg.data import Normalize, IntensityToModel, RandomCrop


def test_normalize_and_inverse():
    x = torch.tensor([[[0.0, 1.0], [0.5, 0.25]]])
    norm = Normalize(mean=0.5, std=0.5)
    y = norm(x)
    x_rec = norm.inverse(y)
    assert torch.allclose(x, x_rec, atol=1e-6)


def test_intensity_to_model_and_inverse():
    x = torch.tensor([[[0.0, 1000.0], [250.0, 750.0]]])
    tf = IntensityToModel(min_intensity=0, max_intensity=1000)
    y = tf(x)
    assert torch.all(y >= -1.0) and torch.all(y <= 1.0)
    x_rec = tf.inverse(y)
    assert torch.allclose(x, x_rec, atol=1e-5)


def test_random_crop_smaller_input():
    x = torch.randn(1, 16, 16)
    rc = RandomCrop(size=32)
    y = rc(x)
    # Should be unchanged because input is smaller than crop
    assert y.shape == x.shape


def test_random_crop_applied():
    x = torch.randn(1, 64, 64)
    rc = RandomCrop(size=32)
    y = rc(x)
    assert y.shape == (1, 32, 32)


