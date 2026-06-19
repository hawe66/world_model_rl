import json

import numpy as np

from wm.common.envs.base import Transition
from wm.common.eval import evaluate
from wm.common.logger import Logger
from wm.common.viz import frames_to_gif, frames_to_grid, open_loop_grid


# --------------------------------------------------------------------------- #
# Logger                                                                       #
# --------------------------------------------------------------------------- #


def test_logger_writes_jsonl(tmp_path):
    with Logger(tmp_path) as log:
        log.log_scalar("loss", 1.5, step=10)
        log.log_scalar("ret", 2.0, step=10)

    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    records = [json.loads(line) for line in lines]
    assert {"step": 10, "loss": 1.5} in records
    assert {"step": 10, "ret": 2.0} in records


def test_logger_creates_tensorboard_events(tmp_path):
    with Logger(tmp_path) as log:
        log.log_scalar("x", 1.0, step=0)
    assert any(p.name.startswith("events.out.tfevents") for p in tmp_path.iterdir())


# --------------------------------------------------------------------------- #
# Eval                                                                         #
# --------------------------------------------------------------------------- #


class _FakeEnv:
    """Fixed-length episode env; reward 1 per step => return == ep_len."""

    def __init__(self, ep_len=5):
        self.ep_len = ep_len
        self._t = 0

    class _Space:
        shape = (1,)

        def sample(self, rng=None):
            return np.zeros(1, np.float32)

    @property
    def action_space(self):
        return self._Space()

    def reset(self):
        self._t = 0
        return Transition({"state": np.zeros(2, np.float32)}, np.zeros(1, np.float32), 0.0,
                          True, False, False)

    def step(self, action):
        self._t += 1
        return Transition({"state": np.zeros(2, np.float32)}, action, 1.0,
                          False, self._t >= self.ep_len, False)

    def close(self):
        pass


def test_evaluate_random_policy_returns_stats():
    env = _FakeEnv(ep_len=5)

    def policy(obs, mode="stochastic"):
        return np.zeros(1, np.float32)

    stats = evaluate(env, policy, episodes=3)
    assert stats["return_mean"] == 5.0  # 5 steps * reward 1
    assert len(stats["returns"]) == 3
    assert stats["length_mean"] == 5.0


# --------------------------------------------------------------------------- #
# Viz                                                                          #
# --------------------------------------------------------------------------- #


def _frames(t=4, h=8, w=8, c=1):
    return (np.arange(t * h * w * c) % 256).astype(np.uint8).reshape(t, h, w, c)


def test_frames_to_grid_single_row_shape(tmp_path):
    grid = frames_to_grid(_frames(t=4, h=8, w=8, c=1), path=tmp_path / "grid.png")
    assert grid.shape[0] == 8 and grid.shape[1] == 8 * 4  # one row of 4 frames
    assert (tmp_path / "grid.png").exists()


def test_frames_to_gif_writes_file(tmp_path):
    out = tmp_path / "roll.gif"
    frames_to_gif(_frames(t=4), out, fps=5)
    assert out.exists() and out.stat().st_size > 0


def test_open_loop_grid_pred_equals_real(tmp_path):
    real = _frames(t=4)
    out = open_loop_grid(real, real, path=tmp_path / "ol.png")
    # real row stacked above pred row -> double height of a single grid row.
    assert out.shape[0] == 8 * 2
    assert (tmp_path / "ol.png").exists()
