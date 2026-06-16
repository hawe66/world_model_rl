"""Common environment interface shared by all wrappers.

Every wrapper (Atari, DMC, ...) returns a :class:`Transition` from ``reset`` and
``step`` so downstream code (buffer, world models, actor-critic) consumes one
uniform structure. The boundary flags follow Dreamer-family episode semantics:

- ``is_first``   : first transition of an episode (the ``reset`` output).
- ``is_last``    : episode ended this step (terminal OR time-limit truncation).
- ``is_terminal``: a *true* terminal state (used to zero the value bootstrap).
                   Time-limit truncation is ``is_last=True`` but
                   ``is_terminal=False`` so the critic still bootstraps.

Images live in ``obs["image"]`` as uint8 HWC; float/CHW conversion happens at
training time. Proprioceptive states live in ``obs["state"]`` as float32.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Transition:
    """One environment transition in the common interface.

    Attributes:
        obs: Observation dict — ``"image"`` (uint8, HWC) and/or ``"state"``
            (float32, 1-D).
        action: Action taken to reach this transition (zeros on ``reset``).
        reward: Scalar reward (summed over action-repeat frames).
        is_first: True only for the ``reset`` transition.
        is_last: True when the episode ends this step.
        is_terminal: True only for a genuine terminal (not time-limit).
    """

    obs: dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    is_first: bool
    is_last: bool
    is_terminal: bool


class Env(ABC):
    """Abstract environment returning :class:`Transition` objects."""

    @abstractmethod
    def reset(self) -> Transition:
        """Start a new episode; returns the ``is_first=True`` transition."""

    @abstractmethod
    def step(self, action: np.ndarray) -> Transition:
        """Advance one agent step (action repeated internally)."""

    @property
    @abstractmethod
    def action_space(self) -> Any:
        """Action space exposing ``sample()`` (gymnasium/dm_env style)."""

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        """Observation space describing ``obs`` contents."""

    def close(self) -> None:  # noqa: B027 - optional override
        """Release env resources (no-op by default)."""
