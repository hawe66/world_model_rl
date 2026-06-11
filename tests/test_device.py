import torch

from wm.common.device import get_device, set_seed


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_follows_priority():
    """Priority: cuda > xpu > mps > cpu."""
    device = get_device()
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        assert device.type == "xpu"
    elif torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"


def test_set_seed_makes_torch_reproducible():
    set_seed(42)
    a = torch.randn(8)
    set_seed(42)
    b = torch.randn(8)
    assert torch.allclose(a, b), "Same seed must produce identical random tensors"


def test_set_seed_different_seeds_differ():
    set_seed(0)
    a = torch.randn(8)
    set_seed(1)
    b = torch.randn(8)
    assert not torch.allclose(a, b), "Different seeds must produce different tensors"
