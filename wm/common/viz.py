"""Visualization helpers for world-model rollouts.

- ``frames_to_grid``: tile ``(T, H, W, C)`` frames into a filmstrip/grid image.
- ``frames_to_gif``: write an animated GIF.
- ``open_loop_grid``: real frames (top row) above predicted frames (bottom row)
  for open-loop prediction inspection. Until a world model exists (Task 4), the
  demo/tests pass ``pred=real``; the RSSM decoder later supplies ``pred``.

Grayscale frames (C=1) are squeezed for image/GIF output.
"""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def _to_uint8(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:  # float in [0, 1] -> uint8
        frames = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
    return frames


def frames_to_grid(
    frames: np.ndarray, *, path: str | Path | None = None, ncol: int | None = None
) -> np.ndarray:
    """Tile ``(T, H, W, C)`` into ``(rows*H, cols*W, C)`` and optionally save."""
    frames = _to_uint8(frames)
    t, h, w, c = frames.shape  # (T, H, W, C)
    ncol = t if ncol is None else ncol
    nrow = math.ceil(t / ncol)
    canvas = np.zeros((nrow * h, ncol * w, c), dtype=np.uint8)
    for i in range(t):
        r, col = divmod(i, ncol)
        canvas[r * h : (r + 1) * h, col * w : (col + 1) * w] = frames[i]
    if path is not None:
        img = canvas[:, :, 0] if c == 1 else canvas  # squeeze grayscale for PIL
        Image.fromarray(img).save(path)
    return canvas


def frames_to_gif(frames: np.ndarray, path: str | Path, *, fps: int = 10) -> None:
    """Write ``(T, H, W, C)`` frames as an animated GIF."""
    frames = _to_uint8(frames)
    seq = [f[:, :, 0] if f.shape[-1] == 1 else f for f in frames]  # squeeze grayscale
    imageio.mimsave(path, seq, duration=1000 / fps)  # per-frame ms (fps kw deprecated)


def open_loop_grid(
    real: np.ndarray, pred: np.ndarray, *, path: str | Path | None = None
) -> np.ndarray:
    """Stack the real filmstrip above the predicted one: ``(2*rows*H, T*W, C)``."""
    real_grid = frames_to_grid(real)
    pred_grid = frames_to_grid(pred)
    canvas = np.concatenate([real_grid, pred_grid], axis=0)  # vertical stack
    if path is not None:
        img = canvas[:, :, 0] if canvas.shape[-1] == 1 else canvas
        Image.fromarray(img).save(path)
    return canvas
