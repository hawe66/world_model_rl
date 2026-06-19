"""Sequence replay buffer (Dreamer-style stream sampling).

The buffer is a fixed-capacity ring over a contiguous transition *stream*
(insertion order). Sampling draws length-``L`` windows from a uniform start in
the stream; windows MAY cross episode boundaries, and the per-step ``is_first``
flag marks where a new episode begins inside a window so the consumer can reset
its recurrent state there (DreamerV2/V3 convention).

Images are stored uint8 (HWC) to save memory; ``sample(..., to_float=True)``
returns float32 in [0, 1] for convenience. The consumer handles CHW/device.

Shapes:
- stored per field: ``(capacity, *feature_dims)``
- sampled batch:    ``(B, L, *feature_dims)``  e.g. image ``(B, L, H, W, C)``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# npz keys that are observation modalities (everything else is scalar-per-step).
_OBS_KEYS = ("image", "state")
_FLAG_KEYS = ("is_first", "is_last", "is_terminal")


class SequenceReplayBuffer:
    def __init__(self, capacity: int, *, seed: int | None = None) -> None:
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._size = 0
        self._write = 0  # next slot to write
        self._start = 0  # physical index of the oldest (logical 0) element
        self._obs: dict[str, np.ndarray] = {}
        self._action: np.ndarray | None = None
        self._reward: np.ndarray | None = None
        self._flags: dict[str, np.ndarray] = {}
        self._obs_keys: tuple[str, ...] = ()

    def __len__(self) -> int:
        return self._size

    # -- allocation -------------------------------------------------------- #

    def _allocate(self, obs: dict[str, np.ndarray], action: np.ndarray) -> None:
        self._obs_keys = tuple(k for k in obs)
        for k in self._obs_keys:
            arr = np.asarray(obs[k])
            self._obs[k] = np.empty((self.capacity, *arr.shape), dtype=arr.dtype)
        a = np.asarray(action)
        self._action = np.empty((self.capacity, *a.shape), dtype=a.dtype)
        self._reward = np.empty((self.capacity,), dtype=np.float32)
        for k in _FLAG_KEYS:
            self._flags[k] = np.empty((self.capacity,), dtype=np.bool_)

    # -- writing ----------------------------------------------------------- #

    def add(
        self,
        *,
        obs: dict[str, np.ndarray],
        action: Any,
        reward: float,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
    ) -> None:
        if self._reward is None:
            self._allocate(obs, np.asarray(action))
        p = self._write
        for k in self._obs_keys:
            self._obs[k][p] = obs[k]
        self._action[p] = action
        self._reward[p] = reward
        self._flags["is_first"][p] = is_first
        self._flags["is_last"][p] = is_last
        self._flags["is_terminal"][p] = is_terminal

        self._write = (self._write + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1
        else:
            # full: oldest just got overwritten, advance logical origin.
            self._start = (self._start + 1) % self.capacity

    def add_transition(self, t) -> None:
        """Convenience: add a wm.common.envs.base.Transition."""
        self.add(
            obs=t.obs,
            action=t.action,
            reward=t.reward,
            is_first=t.is_first,
            is_last=t.is_last,
            is_terminal=t.is_terminal,
        )

    # -- npz --------------------------------------------------------------- #

    def load_npz(self, path: str | Path) -> None:
        data = np.load(path)
        n = len(data["reward"])
        obs_keys = [k for k in _OBS_KEYS if k in data]
        for i in range(n):
            self.add(
                obs={k: data[k][i] for k in obs_keys},
                action=data["action"][i],
                reward=float(data["reward"][i]),
                is_first=bool(data["is_first"][i]),
                is_last=bool(data["is_last"][i]),
                is_terminal=bool(data["is_terminal"][i]),
            )

    @classmethod
    def from_npz(
        cls, path: str | Path, *, capacity: int | None = None, seed: int | None = None
    ) -> SequenceReplayBuffer:
        data = np.load(path)
        cap = capacity if capacity is not None else len(data["reward"])
        buf = cls(capacity=cap, seed=seed)
        buf.load_npz(path)
        return buf

    # -- sampling ---------------------------------------------------------- #

    def _physical(self, starts: np.ndarray, seq_len: int) -> np.ndarray:
        # (B, L) physical indices for logical windows [start, start+L).
        offsets = np.arange(seq_len)  # (L,)
        logical = starts[:, None] + offsets[None, :]  # (B, L)
        return (self._start + logical) % self.capacity

    def gather(self, starts, seq_len: int, *, to_float: bool = False) -> dict[str, np.ndarray]:
        starts = np.asarray(starts, dtype=np.int64)
        if np.any(starts < 0) or np.any(starts + seq_len > self._size):
            raise ValueError("requested window falls outside the stored stream")
        idx = self._physical(starts, seq_len)  # (B, L)

        batch: dict[str, np.ndarray] = {}
        for k in self._obs_keys:
            obs = self._obs[k][idx]  # (B, L, *feature)
            if to_float and k == "image":
                obs = obs.astype(np.float32) / 255.0
            batch[k] = obs
        batch["action"] = self._action[idx]
        batch["reward"] = self._reward[idx]
        for k in _FLAG_KEYS:
            batch[k] = self._flags[k][idx]
        return batch

    def sample(
        self, batch_size: int, seq_len: int, *, to_float: bool = False
    ) -> dict[str, np.ndarray]:
        if self._size < seq_len:
            raise ValueError(f"buffer has {self._size} steps < seq_len {seq_len}")
        starts = self._rng.integers(0, self._size - seq_len + 1, size=batch_size)
        return self.gather(starts, seq_len, to_float=to_float)

    # -- checkpoint -------------------------------------------------------- #

    def _logical_indices(self) -> np.ndarray:
        return (self._start + np.arange(self._size)) % self.capacity

    def state_dict(self) -> dict[str, Any]:
        """Compact, insertion-ordered snapshot (origin reset to 0)."""
        if self._reward is None:
            return {"capacity": self.capacity, "size": 0, "obs_keys": ()}
        idx = self._logical_indices()
        return {
            "capacity": self.capacity,
            "size": self._size,
            "obs_keys": self._obs_keys,
            "obs": {k: self._obs[k][idx] for k in self._obs_keys},
            "action": self._action[idx],
            "reward": self._reward[idx],
            "flags": {k: self._flags[k][idx] for k in _FLAG_KEYS},
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.capacity = sd["capacity"]
        self._size = sd["size"]
        self._start = 0
        self._write = self._size % self.capacity
        self._obs_keys = tuple(sd["obs_keys"])
        if self._size == 0:
            return
        self._obs = {}
        for k in self._obs_keys:
            self._obs[k] = np.empty((self.capacity, *sd["obs"][k].shape[1:]), sd["obs"][k].dtype)
            self._obs[k][: self._size] = sd["obs"][k]
        self._action = np.empty((self.capacity, *sd["action"].shape[1:]), sd["action"].dtype)
        self._action[: self._size] = sd["action"]
        self._reward = np.empty((self.capacity,), np.float32)
        self._reward[: self._size] = sd["reward"]
        self._flags = {}
        for k in _FLAG_KEYS:
            self._flags[k] = np.empty((self.capacity,), np.bool_)
            self._flags[k][: self._size] = sd["flags"][k]
