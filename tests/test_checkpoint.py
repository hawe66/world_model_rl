import numpy as np
import pytest
import torch
from torch import nn

from wm.common.checkpoint import load_checkpoint, save_checkpoint


def _make_model_and_opt():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


def _step_optimizer(model, optimizer):
    """Run one fwd/bwd/step so the optimizer holds non-trivial state."""
    x = torch.randn(3, 4)  # (B, in_features)
    loss = model(x).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_roundtrip_model_optimizer_step_extra(tmp_path):
    model, optimizer = _make_model_and_opt()
    _step_optimizer(model, optimizer)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        step=12345,
        model=model,
        optimizer=optimizer,
        extra={"best_score": 21.0},
    )

    new_model, new_optimizer = _make_model_and_opt()
    # Sanity: fresh model params differ before loading.
    assert not all(
        torch.allclose(a, b)
        for a, b in zip(model.parameters(), new_model.parameters(), strict=True)
    )

    meta = load_checkpoint(path, model=new_model, optimizer=new_optimizer)

    assert meta["step"] == 12345
    assert meta["extra"]["best_score"] == 21.0
    for a, b in zip(model.parameters(), new_model.parameters(), strict=True):
        assert torch.allclose(a, b)
    # Optimizer exp_avg states must match.
    s_old = optimizer.state_dict()["state"]
    s_new = new_optimizer.state_dict()["state"]
    assert s_old.keys() == s_new.keys()
    for k in s_old:
        assert torch.allclose(s_old[k]["exp_avg"], s_new[k]["exp_avg"])


def test_rng_state_is_restored(tmp_path):
    model, _ = _make_model_and_opt()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, step=0, model=model)

    # Draw the "ground truth" sequence right after saving.
    torch_after = torch.randn(5)
    np_after = np.random.rand(5)

    # Perturb RNG, then restore from checkpoint and redraw.
    torch.randn(100)
    np.random.rand(100)
    load_checkpoint(path, restore_rng=True)

    assert torch.allclose(torch.randn(5), torch_after)
    assert np.allclose(np.random.rand(5), np_after)


def test_restore_rng_false_does_not_touch_rng(tmp_path):
    model, _ = _make_model_and_opt()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, step=0, model=model)

    torch.manual_seed(7)
    expected = torch.randn(5)
    torch.manual_seed(7)
    load_checkpoint(path, restore_rng=False)
    assert torch.allclose(torch.randn(5), expected)


def test_map_location_cpu(tmp_path):
    model, optimizer = _make_model_and_opt()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, step=1, model=model, optimizer=optimizer)

    new_model, _ = _make_model_and_opt()
    load_checkpoint(path, model=new_model, map_location="cpu")
    for p in new_model.parameters():
        assert p.device.type == "cpu"


def test_save_is_atomic_overwrite(tmp_path):
    model, _ = _make_model_and_opt()
    path = tmp_path / "latest.pt"

    save_checkpoint(path, step=1, model=model)
    save_checkpoint(path, step=2, model=model)  # overwrite existing file

    meta = load_checkpoint(path)
    assert meta["step"] == 2
    # No leftover temp files in the directory.
    assert list(tmp_path.glob("*.tmp")) == []


def test_replay_buffer_roundtrip(tmp_path):
    class FakeBuffer:
        def __init__(self, n):
            self.n = n

        def state_dict(self):
            return {"n": self.n}

        def load_state_dict(self, sd):
            self.n = sd["n"]

    model, _ = _make_model_and_opt()
    buf = FakeBuffer(99)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, step=0, model=model, replay_buffer=buf)

    new_buf = FakeBuffer(0)
    load_checkpoint(path, replay_buffer=new_buf)
    assert new_buf.n == 99


def test_load_returns_meta_without_targets(tmp_path):
    model, _ = _make_model_and_opt()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, step=42, model=model, extra={"k": "v"})

    meta = load_checkpoint(path)
    assert meta["step"] == 42
    assert meta["extra"] == {"k": "v"}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")
