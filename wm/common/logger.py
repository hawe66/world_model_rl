"""Dual logger: TensorBoard + metrics.jsonl, keyed on env step.

One JSON object per ``log_scalar`` call (``{"step": s, "<key>": value}``) is
appended to ``metrics.jsonl``; the same scalar is mirrored to TensorBoard. The
step axis is always env step (= agent step) per project convention.
"""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, logdir: str | Path, *, use_tensorboard: bool = True) -> None:
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(self.logdir / "metrics.jsonl", "a", encoding="utf-8")
        self._writer = SummaryWriter(str(self.logdir)) if use_tensorboard else None

    def log_scalar(self, key: str, value: float, step: int) -> None:
        value = float(value)
        self._jsonl.write(json.dumps({"step": int(step), key: value}) + "\n")
        if self._writer is not None:
            self._writer.add_scalar(key, value, step)

    def log_scalars(self, values: dict[str, float], step: int) -> None:
        for key, value in values.items():
            self.log_scalar(key, value, step)

    def flush(self) -> None:
        self._jsonl.flush()
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        self._jsonl.close()
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> Logger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
