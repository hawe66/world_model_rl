"""Demo: npz -> replay buffer -> (B, L) sequence batch -> viz grid/GIF.

Verifies the Task 3 infra end-to-end on CPU:

    uv run python scripts/demo_buffer_viz.py --npz results/random/pong.npz \
        --seq-len 16 --batch 4 --out results/demo

Writes a filmstrip PNG, an open-loop grid (pred=real placeholder until Task 4),
and a GIF of the first sampled sequence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wm.common.buffer import SequenceReplayBuffer
from wm.common.viz import frames_to_gif, frames_to_grid, open_loop_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Buffer + viz demo.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/demo"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    buf = SequenceReplayBuffer.from_npz(args.npz, seed=args.seed)
    batch = buf.sample(args.batch, args.seq_len)  # (B, L, ...)
    if "image" not in batch:
        raise SystemExit("npz has no 'image' modality (proprio-only) — nothing to visualize")

    seq = batch["image"][0]  # (L, H, W, C) first sequence
    args.out.mkdir(parents=True, exist_ok=True)
    frames_to_grid(seq, path=args.out / "grid.png")
    frames_to_gif(seq, args.out / "rollout.gif", fps=10)
    open_loop_grid(seq, seq, path=args.out / "open_loop.png")  # pred=real placeholder

    print(
        f"buffer steps={len(buf)}  batch image {batch['image'].shape}\n"
        f"saved grid.png / rollout.gif / open_loop.png -> {args.out}"
    )


if __name__ == "__main__":
    main()
