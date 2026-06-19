import numpy as np
import pytest

from wm.common.buffer import SequenceReplayBuffer


def _fill(buf, n, *, img_shape=(4, 4, 1), boundaries=(0,)):
    """Add a deterministic stream: reward == step index (unique id per step)."""
    for i in range(n):
        buf.add(
            obs={"image": np.full(img_shape, i % 256, np.uint8)},
            action=np.int64(i % 3),
            reward=np.float32(i),
            is_first=(i in boundaries),
            is_last=False,
            is_terminal=False,
        )


def test_gather_returns_consecutive_in_insertion_order():
    buf = SequenceReplayBuffer(capacity=100)
    _fill(buf, 10, boundaries=(0,))
    batch = buf.gather(starts=[0, 3], seq_len=4)
    # (B, L) rewards must be the consecutive stream slices.
    assert np.array_equal(batch["reward"], np.array([[0, 1, 2, 3], [3, 4, 5, 6]], np.float32))
    assert batch["image"].shape == (2, 4, 4, 4, 1)  # (B, L, H, W, C)
    assert batch["image"].dtype == np.uint8


def test_sample_shapes_and_dtypes():
    buf = SequenceReplayBuffer(capacity=100, seed=0)
    _fill(buf, 20)
    batch = buf.sample(batch_size=5, seq_len=4)
    assert batch["image"].shape == (5, 4, 4, 4, 1)
    assert batch["reward"].shape == (5, 4) and batch["reward"].dtype == np.float32
    assert batch["action"].shape == (5, 4)
    assert batch["is_first"].dtype == np.bool_


def test_sample_to_float_normalizes_images():
    buf = SequenceReplayBuffer(capacity=100, seed=0)
    _fill(buf, 20)
    batch = buf.sample(batch_size=2, seq_len=4, to_float=True)
    assert batch["image"].dtype == np.float32
    assert batch["image"].max() <= 1.0 and batch["image"].min() >= 0.0


def test_sample_requires_enough_data():
    buf = SequenceReplayBuffer(capacity=100)
    _fill(buf, 3)
    with pytest.raises(ValueError):
        buf.sample(batch_size=2, seq_len=8)


def test_ring_eviction_drops_oldest():
    buf = SequenceReplayBuffer(capacity=5)
    _fill(buf, 8)  # rewards 0..7; only the last 5 survive
    assert len(buf) == 5
    batch = buf.gather(starts=[0], seq_len=5)
    assert np.array_equal(batch["reward"][0], np.array([3, 4, 5, 6, 7], np.float32))


def test_is_first_mask_marks_episode_starts_across_boundary():
    buf = SequenceReplayBuffer(capacity=100)
    _fill(buf, 10, boundaries=(0, 5))  # second episode begins at index 5
    batch = buf.gather(starts=[3], seq_len=4)  # window covers 3,4,5,6
    assert np.array_equal(batch["is_first"][0], np.array([False, False, True, False]))


def test_state_dict_roundtrip():
    buf = SequenceReplayBuffer(capacity=50)
    _fill(buf, 30, boundaries=(0, 10, 20))
    sd = buf.state_dict()
    buf2 = SequenceReplayBuffer(capacity=50)
    buf2.load_state_dict(sd)
    assert len(buf2) == len(buf)
    a = buf.gather(starts=[2, 11], seq_len=6)
    b = buf2.gather(starts=[2, 11], seq_len=6)
    for k in a:
        assert np.array_equal(a[k], b[k])


def test_from_npz_loads_stream(tmp_path):
    n = 12
    path = tmp_path / "mini.npz"
    np.savez(
        path,
        image=np.arange(n * 4, dtype=np.uint8).reshape(n, 2, 2, 1),
        action=np.arange(n, dtype=np.int64),
        reward=np.arange(n, dtype=np.float32),
        is_first=np.array([i == 0 for i in range(n)]),
        is_last=np.zeros(n, bool),
        is_terminal=np.zeros(n, bool),
        episode=np.zeros(n, np.int32),
    )
    buf = SequenceReplayBuffer.from_npz(path)
    assert len(buf) == n
    batch = buf.gather(starts=[0], seq_len=5)
    assert np.array_equal(batch["reward"][0], np.array([0, 1, 2, 3, 4], np.float32))


def test_checkpoint_integration(tmp_path):
    from torch import nn

    from wm.common.checkpoint import load_checkpoint, save_checkpoint

    buf = SequenceReplayBuffer(capacity=50)
    _fill(buf, 25)
    model = nn.Linear(2, 2)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, step=1, model=model, replay_buffer=buf)

    buf2 = SequenceReplayBuffer(capacity=50)
    load_checkpoint(path, replay_buffer=buf2)
    assert len(buf2) == len(buf)
    a = buf.gather(starts=[5], seq_len=4)
    b = buf2.gather(starts=[5], seq_len=4)
    assert np.array_equal(a["reward"], b["reward"])
