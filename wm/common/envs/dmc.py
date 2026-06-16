"""DeepMind Control Suite wrapper producing the common :class:`Transition`.

Supports two observation modes:
- proprio (default): ``obs["state"]`` = flattened physics features (float32),
  concatenated in sorted-key order for a stable layout.
- vision: ``obs["image"]`` = offscreen-rendered 64x64 RGB (uint8 HWC). Needs an
  OpenGL backend; unavailable on driverless machines (rendering raises there).

Episode flags (per project decision): DMC tasks are fixed-length with no true
terminal, so ``is_terminal`` is always False; ``is_last`` is True only when the
underlying time limit is reached (``TimeStep.last()``). One agent step repeats
the action ``action_repeat`` times and sums the reward.
"""

from __future__ import annotations

import numpy as np
from dm_control import suite

from .base import Env, Transition


class _BoxActionSpace:
    """Minimal continuous action space exposing shape/bounds and sample()."""

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        self.low = low.astype(np.float32)
        self.high = high.astype(np.float32)
        self.shape = low.shape

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        return rng.uniform(self.low, self.high).astype(np.float32)


def _flatten_obs(obs: dict) -> np.ndarray:
    # Sorted-key concat -> deterministic (T,) vector independent of dict order.
    parts = [np.asarray(obs[k], dtype=np.float32).ravel() for k in sorted(obs)]
    return np.concatenate(parts).astype(np.float32)


class DMCEnv(Env):
    def __init__(
        self,
        domain: str,
        task: str,
        *,
        vision: bool = False,
        image_size: int = 64,
        action_repeat: int = 2,
        camera_id: int = 0,
        seed: int | None = None,
    ) -> None:
        self.domain = domain
        self.task = task
        self.vision = vision
        self.image_size = image_size
        self.action_repeat = action_repeat
        self.camera_id = camera_id
        self._env = suite.load(domain, task, task_kwargs={"random": seed})
        spec = self._env.action_spec()
        self._action_space = _BoxActionSpace(np.asarray(spec.minimum), np.asarray(spec.maximum))
        self._control_steps = 0

    @property
    def action_space(self) -> _BoxActionSpace:
        return self._action_space

    @property
    def observation_space(self):
        return self._action_space  # placeholder; obs schema is dict-based

    @property
    def frames_per_step(self) -> int:
        """Control steps consumed per agent step (= action_repeat)."""
        return self.action_repeat

    @property
    def control_steps(self) -> int:
        """Underlying control steps taken since construction."""
        return self._control_steps

    def _obs(self, time_step) -> dict[str, np.ndarray]:
        if self.vision:
            img = self._env.physics.render(
                self.image_size, self.image_size, camera_id=self.camera_id
            )
            return {"image": np.asarray(img, dtype=np.uint8)}  # (H, W, 3)
        return {"state": _flatten_obs(time_step.observation)}  # (D,) float32

    def reset(self) -> Transition:
        time_step = self._env.reset()
        return Transition(
            obs=self._obs(time_step),
            action=np.zeros(self._action_space.shape, dtype=np.float32),
            reward=0.0,
            is_first=True,
            is_last=False,
            is_terminal=False,
        )

    def step(self, action) -> Transition:
        action = np.asarray(action, dtype=np.float32)
        total_reward = 0.0
        time_step = None
        for _ in range(self.action_repeat):
            time_step = self._env.step(action)
            self._control_steps += 1
            total_reward += float(time_step.reward or 0.0)
            if time_step.last():
                break

        return Transition(
            obs=self._obs(time_step),
            action=action,
            reward=total_reward,
            is_first=False,
            is_last=bool(time_step.last()),
            is_terminal=False,  # DMC: no true terminal, only time-limit truncation
        )

    def close(self) -> None:
        self._env.close()
