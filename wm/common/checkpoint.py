"""Checkpoint save/load utilities for --resume support.

Stores everything needed to resume a training run inside a single file written
atomically (temp file -> os.replace): model/optimizer state, the env step
counter, RNG state (python/numpy/torch, plus cuda/xpu when available), an
optional replay buffer, and an arbitrary ``extra`` dict. Device-agnostic: pass
``map_location`` to move tensors between CPU and an accelerator (cloud CUDA <->
local CPU resume).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _collect_rng_state() -> dict[str, Any]:
    """Snapshot RNG state across python/numpy/torch (+cuda/xpu if present)."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),  # uint8 CPU tensor
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        state["xpu"] = torch.xpu.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state saved by :func:`_collect_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch expects a uint8 ByteTensor on CPU regardless of training device.
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "xpu" in state and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_rng_state_all(state["xpu"])


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    replay_buffer: Any | None = None,
    rng: bool = True,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically save a training checkpoint to ``path``.

    Args:
        path: Destination file (parent dirs are created as needed).
        step: Env step counter (= agent step) marking training progress.
        model: Module whose ``state_dict`` is saved.
        optimizer: Optional optimizer whose ``state_dict`` is saved.
        replay_buffer: Optional object exposing ``state_dict()``.
        rng: If True, snapshot python/numpy/torch RNG state.
        extra: Arbitrary picklable metadata (e.g. best eval score).
    """
    path = Path(path)
    payload: dict[str, Any] = {
        "step": step,
        "model": model.state_dict(),
        "extra": extra if extra is not None else {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if replay_buffer is not None:
        payload["replay_buffer"] = replay_buffer.state_dict()
    if rng:
        payload["rng"] = _collect_rng_state()

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file, then atomically replace the target so an
    # interrupted save never corrupts an existing checkpoint.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    replay_buffer: Any | None = None,
    map_location: str | torch.device | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, restoring into the given objects in-place.

    Args:
        path: Checkpoint file written by :func:`save_checkpoint`.
        model: If given, ``load_state_dict`` is called on it.
        optimizer: If given, ``load_state_dict`` is called on it.
        replay_buffer: If given, ``load_state_dict`` is called on it.
        map_location: Passed to ``torch.load`` to relocate tensors
            (e.g. ``"cpu"`` to resume a CUDA run on a CPU-only machine).
        restore_rng: If True and RNG state was saved, restore it.

    Returns:
        Metadata dict ``{"step", "extra", ...}`` (the full saved payload).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")

    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if replay_buffer is not None and "replay_buffer" in payload:
        replay_buffer.load_state_dict(payload["replay_buffer"])
    if restore_rng and "rng" in payload:
        _restore_rng_state(payload["rng"])

    return payload
