# Task 0 — Repo Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan subtask-by-subtask. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap an empty repo into a working skeleton where `uv run pytest` passes and `uv run python scripts/smoke.py` executes a forward/backward pass on the best available accelerator.

**Architecture:** uv-managed Python project with `wm/` as the top-level package, `wm/common/` for shared utilities, and `wm/algos/` for per-algorithm code. No algorithm logic yet — only the directory skeleton, device utilities, and a smoke test.

**Tech Stack:** Python 3.11, uv, PyTorch 2.5+ (XPU wheel), ruff (lint+format), pre-commit, pytest

**Device priority:** `cuda → xpu → mps → cpu`
- This machine uses Intel Core Ultra integrated Arc (Meteor Lake). PyTorch 2.5+ native XPU backend supports it on Windows.
- IPEX is archived (2026-03-30) — do not use. torch-directml is not recommended (stale, low op coverage).
- When running on a CUDA machine (Task 5+), the same `get_device()` call returns `cuda` automatically.

---

## File Structure

| Path | Action | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Create | uv project config, all runtime + dev deps, XPU index |
| `.python-version` | Create | Pin Python 3.11 for uv |
| `.pre-commit-config.yaml` | Create | ruff format + lint hooks |
| `wm/__init__.py` | Create | Package root (empty) |
| `wm/common/__init__.py` | Create | Subpackage marker (empty) |
| `wm/common/device.py` | Create | `get_device()`, `set_seed()` |
| `wm/common/envs/__init__.py` | Create | Placeholder for Task 2 |
| `wm/algos/__init__.py` | Create | Subpackage marker (empty) |
| `wm/algos/rssm/.gitkeep` | Create | Placeholder dir |
| `wm/algos/dreamer_v1/.gitkeep` | Create | Placeholder dir |
| `wm/algos/dreamer_v2/.gitkeep` | Create | Placeholder dir |
| `wm/algos/dreamer_v3/.gitkeep` | Create | Placeholder dir |
| `wm/algos/tdmpc2/.gitkeep` | Create | Placeholder dir |
| `wm/algos/iris/.gitkeep` | Create | Placeholder dir |
| `wm/algos/storm/.gitkeep` | Create | Placeholder dir |
| `wm/algos/diamond/.gitkeep` | Create | Placeholder dir (stretch) |
| `wm/configs/.gitkeep` | Create | Placeholder for algorithm YAML configs |
| `scripts/__init__.py` | Create | Makes scripts importable under uv |
| `scripts/smoke.py` | Create | Dummy forward/backward on best device |
| `results/.gitkeep` | Create | Placeholder for training outputs |
| `tests/__init__.py` | Create | Test package marker |
| `tests/test_device.py` | Create | Unit tests for device utils |

---

## Subtask 1: Initialize uv Project

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`

- [ ] **Step 1: Create `.python-version`**

File content (one line):
```
3.11
```

Save as `.python-version` at repo root (`D:\world_model_rl\.python-version`).

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "world-model-zoo"
version = "0.1.0"
description = "World model RL lineage reproduction"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.5",
    "torchvision>=0.20",
    "gymnasium[atari,accept-rom-license]>=0.29",
    "dm_control>=1.0",
    "tensorboard>=2.16",
    "einops>=0.7",
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "numpy>=1.26",
    "imageio>=2.34",
    "Pillow>=10.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
    "pre-commit>=3.7",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["wm"]

# Route torch/torchvision to the PyTorch XPU index (includes CUDA + XPU builds)
[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-xpu" }
torchvision = { index = "pytorch-xpu" }

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Initialize git and install the project**

Run from `D:\world_model_rl`:
```
git init
uv sync --extra dev
```

Expected: uv resolves all packages (torch comes from pytorch.org/whl/xpu) and installs into `.venv/`. No error output.

> **If dm_control fails on Windows:** Install the Visual C++ Redistributable from Microsoft, then re-run `uv sync --extra dev`.

- [ ] **Step 4: Verify Python and XPU availability**

Run:
```
uv run python --version
uv run python -c "import torch; print('xpu:', torch.xpu.is_available()); print('cuda:', torch.cuda.is_available())"
```

Expected:
```
Python 3.11.x
xpu: True
cuda: False
```

- [ ] **Step 5: Commit**

```
git add pyproject.toml .python-version uv.lock
git commit -m "chore: init uv project with PyTorch XPU wheel"
```

---

## Subtask 2: Directory Skeleton

**Files:**
- Create all `__init__.py` and `.gitkeep` files listed in the file structure table

- [ ] **Step 1: Create the wm package and common subpackage**

Create these three empty files:
- `wm/__init__.py`
- `wm/common/__init__.py`
- `wm/common/envs/__init__.py`

All are empty (zero bytes).

- [ ] **Step 2: Create the algos subpackage and per-algorithm placeholders**

Create `wm/algos/__init__.py` (empty).

Create one empty `.gitkeep` file in each directory (creating the dirs):
- `wm/algos/rssm/.gitkeep`
- `wm/algos/dreamer_v1/.gitkeep`
- `wm/algos/dreamer_v2/.gitkeep`
- `wm/algos/dreamer_v3/.gitkeep`
- `wm/algos/tdmpc2/.gitkeep`
- `wm/algos/iris/.gitkeep`
- `wm/algos/storm/.gitkeep`
- `wm/algos/diamond/.gitkeep`

- [ ] **Step 3: Create configs, scripts, results, tests directories**

Create empty files:
- `wm/configs/.gitkeep`
- `scripts/__init__.py`
- `results/.gitkeep`
- `tests/__init__.py`

- [ ] **Step 4: Verify wm package imports**

Run:
```
uv run python -c "import wm; import wm.common; import wm.algos; print('imports OK')"
```

Expected: `imports OK`

- [ ] **Step 5: Commit**

```
git add wm/ scripts/ results/ tests/
git commit -m "chore: create wm package skeleton and placeholder dirs"
```

---

## Subtask 3: Device Utilities (TDD)

**Files:**
- Create: `tests/test_device.py`
- Create: `wm/common/device.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_device.py`:

```python
import torch

from wm.common.device import get_device, set_seed


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_follows_priority():
    """Priority: cuda > xpu > mps > cpu."""
    device = get_device()
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        assert device.type == "xpu"
    elif torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"


def test_set_seed_makes_torch_reproducible():
    set_seed(42)
    a = torch.randn(8)
    set_seed(42)
    b = torch.randn(8)
    assert torch.allclose(a, b), "Same seed must produce identical random tensors"


def test_set_seed_different_seeds_differ():
    set_seed(0)
    a = torch.randn(8)
    set_seed(1)
    b = torch.randn(8)
    assert not torch.allclose(a, b), "Different seeds must produce different tensors"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
uv run pytest tests/test_device.py -v
```

Expected: `ImportError` — `wm.common.device` does not exist yet.

- [ ] **Step 3: Implement `wm/common/device.py`**

```python
import random

import numpy as np
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
uv run pytest tests/test_device.py -v
```

Expected:
```
tests/test_device.py::test_get_device_returns_torch_device PASSED
tests/test_device.py::test_get_device_follows_priority PASSED
tests/test_device.py::test_set_seed_makes_torch_reproducible PASSED
tests/test_device.py::test_set_seed_different_seeds_differ PASSED
4 passed
```

- [ ] **Step 5: Commit**

```
git add wm/common/device.py tests/test_device.py
git commit -m "feat: add get_device (cuda>xpu>mps>cpu) and set_seed utilities"
```

---

## Subtask 4: Pre-commit Setup (ruff)

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.9
    hooks:
      - id: ruff
        args: ["--fix"]
      - id: ruff-format
```

- [ ] **Step 2: Install pre-commit hooks**

Run:
```
uv run pre-commit install
```

Expected: `pre-commit installed at .git/hooks/pre-commit`

- [ ] **Step 3: Run pre-commit against all files**

Run:
```
uv run pre-commit run --all-files
```

Expected: all checks pass. If ruff auto-fixes files on the first run (exit code 1), re-run once — it should then pass cleanly.

- [ ] **Step 4: Commit**

```
git add .pre-commit-config.yaml
git commit -m "chore: add pre-commit with ruff format and lint"
```

---

## Subtask 5: Smoke Test Script

**Files:**
- Create: `scripts/smoke.py`

- [ ] **Step 1: Create `scripts/smoke.py`**

```python
"""Smoke test: one dummy forward/backward pass on the best available device."""

import torch
import torch.nn as nn

from wm.common.device import get_device, set_seed


def main() -> None:
    device = get_device()
    print(f"device: {device}")
    set_seed(0)

    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    ).to(device)

    x = torch.randn(8, 64, device=device)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()

    grad_norm = sum(
        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
    ) ** 0.5

    print(f"loss={loss.item():.6f}  grad_norm={grad_norm:.6f}")
    print("smoke test PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run smoke test**

Run:
```
uv run python scripts/smoke.py
```

Expected output (on this machine):
```
device: xpu
loss=<any float>  grad_norm=<any float>
smoke test PASSED
```

The first line should say `device: xpu` (or `cuda` on a CUDA machine). If it says `cpu`, XPU is not detected — check that Intel GPU drivers are up to date, then re-run.

- [ ] **Step 3: Run full test suite**

Run:
```
uv run pytest -v
```

Expected: `4 passed, 0 failed, 0 errors`

- [ ] **Step 4: Commit**

```
git add scripts/smoke.py
git commit -m "feat: add smoke test script"
```

---

## Subtask 6: DoD Verification

- [ ] **Step 1: Full test suite**

Run:
```
uv run pytest -v
```

Expected: `4 passed, 0 failed, 0 errors`

- [ ] **Step 2: Smoke test on accelerator**

Run:
```
uv run python scripts/smoke.py
```

Expected: first line is `device: xpu` (or `cuda`), last line is `smoke test PASSED`.

- [ ] **Step 3: Pre-commit clean pass**

Run:
```
uv run pre-commit run --all-files
```

Expected: all hooks pass with no file modifications.

- [ ] **Step 4: Commit any remaining changes**

Run `git status`. If modified files exist:
```
git add -A
git commit -m "chore: pre-commit formatting fixes"
```

- [ ] **Step 5: Tag Task 0 complete**

```
git tag task0-complete
```

---

## Self-Review

**Spec coverage:**
- [x] uv-based `pyproject.toml` with torch, gymnasium[atari,accept-rom-license], dm_control, tensorboard, einops, hydra-core, pytest, ruff
- [x] Directory structure matches roadmap exactly (all `wm/algos/<name>/` dirs created)
- [x] GPU detection util → `get_device()` with `cuda → xpu → mps → cpu` priority
- [x] Global seed util → `set_seed()` covering cuda + xpu
- [x] pre-commit ruff format/lint
- [x] `uv run pytest` passing (4 tests in `tests/test_device.py`)
- [x] `scripts/smoke.py` — dummy forward/backward on best available device

**Placeholder scan:** None found.

**Type consistency:** `get_device() -> torch.device` and `set_seed(seed: int) -> None` used identically in implementation and tests.
