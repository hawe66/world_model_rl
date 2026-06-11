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

    grad_norm = (
        sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    )

    print(f"loss={loss.item():.6f}  grad_norm={grad_norm:.6f}")
    print("smoke test PASSED")


if __name__ == "__main__":
    main()
