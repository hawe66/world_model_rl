"""Collect random-policy episodes and save them to a single npz file.

Works for both domains via the common env interface:

    uv run python scripts/collect_random.py --env atari --game pong \
        --episodes 2 --seed 0 --out results/random/pong.npz
    uv run python scripts/collect_random.py --env dmc --game walker-walk \
        --episodes 2 --seed 0 --out results/random/walker_walk.npz

Images are stored uint8 HWC; proprio states float32. No --resume (collection is
short). Episode boundaries are recoverable from the per-step ``episode`` index
and the ``is_first`` flag.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wm.common.device import get_device, set_seed

# Friendly game name -> ALE env id (sticky actions OFF set inside the wrapper).
ATARI_GAMES = {
    "pong": "ALE/Pong-v5",
    "breakout": "ALE/Breakout-v5",
    "boxing": "ALE/Boxing-v5",
}


def _build_env(args):
    if args.env == "atari":
        from wm.common.envs.atari import AtariEnv

        env_id = ATARI_GAMES.get(args.game, args.game)
        return AtariEnv(
            env_id,
            image_size=args.image_size,
            action_repeat=args.action_repeat or 4,
            grayscale=not args.rgb,
            life_loss=args.life_loss,
            seed=args.seed,
        )
    from wm.common.envs.dmc import DMCEnv

    domain, task = args.game.split("-", 1)
    return DMCEnv(
        domain,
        task,
        vision=args.vision,
        image_size=args.image_size,
        action_repeat=args.action_repeat or 2,
        seed=args.seed,
    )


def _sample_action(action_space, rng: np.random.Generator):
    if hasattr(action_space, "n"):  # gymnasium Discrete (Atari)
        return int(rng.integers(action_space.n))
    return action_space.sample(rng)  # continuous (DMC)


def collect(args) -> dict[str, np.ndarray]:
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = _build_env(args)

    buffers: dict[str, list] = {
        "action": [],
        "reward": [],
        "is_first": [],
        "is_last": [],
        "is_terminal": [],
        "episode": [],
    }
    obs_key = "image" if (args.env == "atari" or args.vision) else "state"
    buffers[obs_key] = []

    def record(t, ep: int) -> None:
        buffers[obs_key].append(t.obs[obs_key])
        buffers["action"].append(np.asarray(t.action))
        buffers["reward"].append(np.float32(t.reward))
        buffers["is_first"].append(t.is_first)
        buffers["is_last"].append(t.is_last)
        buffers["is_terminal"].append(t.is_terminal)
        buffers["episode"].append(np.int32(ep))

    for ep in range(args.episodes):
        t = env.reset()
        record(t, ep)
        for _ in range(args.max_steps):
            t = env.step(_sample_action(env.action_space, rng))
            record(t, ep)
            if t.is_last:
                break
    env.close()

    out = {k: np.stack(v) for k, v in buffers.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect random-policy episodes to npz.")
    parser.add_argument("--env", choices=["atari", "dmc"], required=True)
    parser.add_argument("--game", required=True, help="e.g. 'pong' or 'walker-walk'")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--action-repeat", type=int, default=0, help="0 = domain default")
    parser.add_argument("--max-steps", type=int, default=1000, help="safety cap per episode")
    parser.add_argument("--rgb", action="store_true", help="Atari: keep RGB instead of grayscale")
    parser.add_argument("--vision", action="store_true", help="DMC: render images instead of state")
    parser.add_argument("--life-loss", action="store_true", help="Atari: life loss => is_terminal")
    args = parser.parse_args()

    # Device is irrelevant to env stepping; report it for run-log consistency.
    print(f"device: {get_device()}")
    data = collect(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    obs_key = "image" if "image" in data else "state"
    print(
        f"saved {len(data['reward'])} steps from {args.episodes} episode(s) -> {args.out}\n"
        f"  obs['{obs_key}'] {data[obs_key].shape} {data[obs_key].dtype}"
    )


if __name__ == "__main__":
    main()
