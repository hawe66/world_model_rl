"""Train the RSSM (PlaNet) world model from a collected npz dataset.

Pure sequence modeling (no RL). Supports --resume (model/optimizer/replay
buffer/step/RNG) and a tiny --smoke mode for a CPU sanity run.

    uv run python scripts/train_rssm.py --smoke --npz results/random/pong.npz
    uv run python scripts/train_rssm.py --config wm/configs/rssm/default.yaml \
        --npz results/random/walker_walk.npz --run-dir runs/rssm/walker --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from wm.algos.rssm.rssm import RSSM
from wm.common.buffer import SequenceReplayBuffer
from wm.common.checkpoint import load_checkpoint, save_checkpoint
from wm.common.device import get_device, set_seed
from wm.common.logger import Logger
from wm.common.viz import frames_to_gif, open_loop_grid


def _inspect_npz(path: Path) -> tuple[int, bool, int]:
    """Return (image_channels, action_is_discrete, action_dim) from the npz."""
    data = np.load(path)
    channels = int(data["image"].shape[-1])  # (N, H, W, C)
    action = data["action"]
    if action.ndim == 1:  # discrete: scalar action index per step
        return channels, True, int(action.max()) + 1
    return channels, False, int(action.shape[-1])  # continuous vector


def _prepare_actions(action: np.ndarray, discrete: bool, action_dim: int) -> torch.Tensor:
    a = torch.as_tensor(action)
    if discrete:  # (B, L) int -> (B, L, A) one-hot
        return F.one_hot(a.long(), action_dim).float()
    return a.float()  # (B, L, A)


def _to_batch(raw: dict, discrete: bool, action_dim: int, device) -> dict[str, torch.Tensor]:
    image = torch.as_tensor(raw["image"]).float()  # (B, L, H, W, C) in [0,1]
    image = einops.rearrange(image, "b l h w c -> b l c h w")
    return {
        "image": image.to(device),
        "action": _prepare_actions(raw["action"], discrete, action_dim).to(device),
        "reward": torch.as_tensor(raw["reward"]).float().to(device),
        "is_first": torch.as_tensor(raw["is_first"]).bool().to(device),
    }


def _save_open_loop(model, batch, cfg, out_dir: Path, step: int) -> None:
    ctx, hor = cfg.open_loop.context, cfg.open_loop.horizon
    if batch["image"].shape[1] < ctx + hor:
        return
    real, pred = model.open_loop(batch, ctx, hor)  # (B, hor, C, H, W)

    def to_frames(x):  # first batch elem -> (T, H, W, C) uint8
        x = x[0].clamp(0, 1).cpu()
        x = einops.rearrange(x, "t c h w -> t h w c")
        return (x.numpy() * 255).astype(np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    open_loop_grid(to_frames(real), to_frames(pred), path=out_dir / f"open_loop_{step}.png")
    frames_to_gif(to_frames(pred), out_dir / f"open_loop_{step}.gif", fps=10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RSSM (PlaNet).")
    parser.add_argument("--config", type=Path, default=Path("wm/configs/rssm/default.yaml"))
    parser.add_argument("--npz", type=Path, default=None, help="override data.npz")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/rssm"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny dims + few steps (CPU)")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.smoke:
        cfg = OmegaConf.merge(cfg, cfg.smoke)  # smoke section overrides top-level
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    if args.npz is not None:
        cfg.data.npz = str(args.npz)

    set_seed(cfg.seed)
    device = get_device()
    print(f"device: {device}  npz: {cfg.data.npz}")

    channels, discrete, action_dim = _inspect_npz(Path(cfg.data.npz))
    buffer = SequenceReplayBuffer.from_npz(
        cfg.data.npz, capacity=cfg.buffer_capacity, seed=cfg.seed
    )

    model = RSSM(
        action_dim=action_dim,
        deter=cfg.model.deter,
        stoch=cfg.model.stoch,
        hidden=cfg.model.hidden,
        image_shape=(channels, 64, 64),
        cnn_depth=cfg.model.cnn_depth,
        reward_units=cfg.model.reward_units,
        reward_layers=cfg.model.reward_layers,
        min_std=cfg.model.min_std,
        free_nats=cfg.loss.free_nats,
        kl_scale=cfg.loss.kl_scale,
        overshoot_d=cfg.loss.overshoot_d,
        overshoot_beta=cfg.loss.overshoot_beta,
        recon_scale=cfg.loss.recon_scale,
        reward_scale=cfg.loss.reward_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr, eps=cfg.optim.eps)

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoints" / "latest.pt"
    start_step = 0
    if args.resume and ckpt_path.exists():
        meta = load_checkpoint(ckpt_path, model=model, optimizer=optimizer,
                               replay_buffer=buffer, map_location=device)
        start_step = meta["step"]
        print(f"[resume] from step {start_step}")

    logger = Logger(run_dir)
    for step in range(start_step, cfg.train.steps):
        raw = buffer.sample(cfg.batch_size, cfg.seq_len, to_float=cfg.to_float)
        batch = _to_batch(raw, discrete, action_dim, device)
        loss, metrics = model.loss(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        optimizer.step()

        if step % cfg.train.log_every == 0:
            logger.log_scalars(metrics, step)
            print(f"step {step}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if step > 0 and step % cfg.open_loop.every == 0:
            _save_open_loop(model, batch, cfg, run_dir / "open_loop", step)
        if step > 0 and step % cfg.train.ckpt_every == 0:
            save_checkpoint(ckpt_path, step=step, model=model, optimizer=optimizer,
                            replay_buffer=buffer, extra={"metrics": metrics})

    # Final checkpoint + open-loop.
    save_checkpoint(ckpt_path, step=cfg.train.steps, model=model, optimizer=optimizer,
                    replay_buffer=buffer)
    raw = buffer.sample(cfg.batch_size, cfg.seq_len, to_float=cfg.to_float)
    _save_open_loop(model, _to_batch(raw, discrete, action_dim, device), cfg,
                    run_dir / "open_loop", cfg.train.steps)
    logger.close()
    print(f"done. checkpoints + open-loop in {run_dir}")


if __name__ == "__main__":
    main()
