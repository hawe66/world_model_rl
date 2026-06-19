"""Fixed evaluation protocol.

``evaluate`` rolls a policy in a *separate* eval env (caller constructs it with
its own seed, kept apart from the training env) for a fixed number of episodes
and reports return statistics. The policy is any callable ``policy(obs, mode)``
returning an action; ``mode`` selects deterministic vs stochastic behaviour and
is forwarded to the policy. World-model/actor code is not required — a random
policy works, which is how this is unit-tested.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

Policy = Callable[[dict, str], Any]


def evaluate(
    env,
    policy: Policy,
    episodes: int = 10,
    *,
    mode: str = "stochastic",
    max_steps: int = 100_000,
) -> dict[str, Any]:
    """Run ``episodes`` rollouts; return return/length statistics.

    Returns:
        dict with ``return_mean``, ``return_std``, ``returns`` (list),
        ``length_mean``, ``lengths`` (list).
    """
    returns: list[float] = []
    lengths: list[int] = []
    for _ in range(episodes):
        t = env.reset()
        ep_return = 0.0
        steps = 0
        while True:
            action = policy(t.obs, mode)
            t = env.step(action)
            ep_return += float(t.reward)
            steps += 1
            if t.is_last or steps >= max_steps:
                break
        returns.append(ep_return)
        lengths.append(steps)

    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "returns": returns,
        "length_mean": float(np.mean(lengths)),
        "lengths": lengths,
    }
