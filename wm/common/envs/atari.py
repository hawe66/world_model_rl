"""Atari (gymnasium ALE) wrapper producing the common :class:`Transition`.

Preprocessing follows the Atari-100k standard:
- base env runs at ``frameskip=1`` and ``repeat_action_probability=0.0``
  (sticky actions OFF), so this wrapper controls action-repeat itself;
- one agent step repeats the action ``action_repeat`` times, sums reward, and
  max-pools the last two raw RGB frames (removes flicker);
- the pooled frame is resized to ``image_size`` and optionally grayscaled,
  stored uint8 HWC.

Episode flags (per project decision):
- ``is_last``     = terminated OR truncated
- ``is_terminal`` = terminated  (time-limit truncation is NOT terminal)
- optional life-loss (``life_loss=True``): losing a life sets
  ``is_terminal=True`` while keeping ``is_last=False`` (episode continues);
  real game-over sets both True.
"""

from __future__ import annotations

import ale_py
import gymnasium as gym
import numpy as np
from PIL import Image

from .base import Env, Transition

gym.register_envs(ale_py)


class AtariEnv(Env):
    def __init__(
        self,
        env_id: str,
        *,
        image_size: int = 64,
        action_repeat: int = 4,
        grayscale: bool = True,
        life_loss: bool = False,
        seed: int | None = None,
    ) -> None:
        self.env_id = env_id
        self.image_size = image_size
        self.action_repeat = action_repeat
        self.grayscale = grayscale
        self.life_loss = life_loss
        self._seed = seed
        # frameskip=1: we do action-repeat ourselves; sticky actions OFF.
        self._env = gym.make(
            env_id,
            frameskip=1,
            repeat_action_probability=0.0,
        )
        self._lives = 0
        channels = 1 if grayscale else 3
        self._obs_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(image_size, image_size, channels),  # (H, W, C)
            dtype=np.uint8,
        )

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def frames_per_step(self) -> int:
        """Raw ALE frames consumed per agent step (= action_repeat)."""
        return self.action_repeat

    @property
    def episode_frame_number(self) -> int:
        """Raw ALE frame count since the last reset (frameskip=1)."""
        return int(self._env.unwrapped.ale.getEpisodeFrameNumber())

    def _process(self, frame: np.ndarray) -> np.ndarray:
        # frame: (210, 160, 3) uint8 RGB.
        img = Image.fromarray(frame)
        if self.grayscale:
            img = img.convert("L")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        out = np.asarray(img, dtype=np.uint8)
        if self.grayscale:
            out = out[:, :, None]  # (H, W, 1)
        return out  # (H, W, C) uint8

    def reset(self, seed: int | None = None) -> Transition:
        seed = self._seed if seed is None else seed
        frame, info = self._env.reset(seed=seed)
        self._lives = info.get("lives", 0)
        return Transition(
            obs={"image": self._process(frame)},
            action=np.zeros((), dtype=np.int64),
            reward=0.0,
            is_first=True,
            is_last=False,
            is_terminal=False,
        )

    def step(self, action) -> Transition:
        total_reward = 0.0
        terminated = truncated = False
        frames: list[np.ndarray] = []
        info: dict = {}
        for _ in range(self.action_repeat):
            frame, reward, terminated, truncated, info = self._env.step(int(action))
            total_reward += float(reward)
            frames.append(frame)
            if terminated or truncated:
                break

        # Max-pool the last two raw frames to remove sprite flicker.
        pooled = frames[-1] if len(frames) == 1 else np.maximum(frames[-1], frames[-2])

        is_last = bool(terminated or truncated)
        is_terminal = bool(terminated)

        lives = info.get("lives", self._lives)
        if self.life_loss and lives < self._lives and not is_last:
            is_terminal = True  # life loss => terminal, but episode continues
        self._lives = lives

        return Transition(
            obs={"image": self._process(pooled)},
            action=np.asarray(action, dtype=np.int64),
            reward=total_reward,
            is_first=False,
            is_last=is_last,
            is_terminal=is_terminal,
        )

    def close(self) -> None:
        self._env.close()
