"""RSSM (PlaNet) world model — pure sequence modeling, no RL.

State factorization (PlaNet): a deterministic GRU path ``h_t`` plus a stochastic
Gaussian latent ``z_t``; the *model state* fed to decoder/reward is the concat
``[h_t, z_t]``.

    h_t      = GRU(h_{t-1}, MLP([z_{t-1}, a_{t-1}]))
    prior    p(z_t | h_t)        = N(mu, sigma) from h_t
    posterior q(z_t | h_t, e_t)  = N(mu, sigma) from [h_t, e_t]   (e_t = encoder embed)

Loss = reconstruction NLL (unit-variance Gaussian) + reward NLL + KL term.
The KL term is PlaNet latent overshooting: for each t and distance d in [1, D],
roll the prior d steps from the posterior at t and add KL(sg(q[t+d]) || prior_d),
averaged by 1/D, with free nats. Episode boundaries (``is_first``) stop a roll.

KL balancing (DreamerV2) is intentionally absent here (see docs/notes/rssm.md).
"""

from __future__ import annotations

import einops
import torch
import torch.nn.functional as F
from torch import nn

from .modules import MLP, ConvDecoder, ConvEncoder

State = dict[str, torch.Tensor]


def kl_normal_sum(
    q_mean: torch.Tensor, q_std: torch.Tensor, p_mean: torch.Tensor, p_std: torch.Tensor
) -> torch.Tensor:
    """KL( N(q) || N(p) ) summed over the last (latent) dim. Returns (...,)."""
    var_ratio = (q_std / p_std).pow(2)
    t1 = ((q_mean - p_mean) / p_std).pow(2)
    kl = 0.5 * (var_ratio + t1 - 1 - var_ratio.log())  # (..., stoch)
    return kl.sum(-1)


class RSSM(nn.Module):
    def __init__(
        self,
        action_dim: int,
        *,
        deter: int = 200,
        stoch: int = 30,
        hidden: int = 200,
        image_shape: tuple[int, int, int] = (3, 64, 64),
        cnn_depth: int = 32,
        reward_units: int = 200,
        reward_layers: int = 2,
        min_std: float = 0.1,
        free_nats: float = 3.0,
        kl_scale: float = 1.0,
        overshoot_d: int = 0,
        overshoot_beta: float = 1.0,
        recon_scale: float = 1.0,
        reward_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.deter, self.stoch, self.action_dim = deter, stoch, action_dim
        self.min_std = min_std
        self.free_nats = free_nats
        self.kl_scale = kl_scale
        self.overshoot_d = overshoot_d
        self.overshoot_beta = overshoot_beta
        self.recon_scale = recon_scale
        self.reward_scale = reward_scale

        self.encoder = ConvEncoder(image_shape, cnn_depth)
        embed = self.encoder.embed_dim
        feat = deter + stoch
        self.decoder = ConvDecoder(feat, image_shape, cnn_depth)
        self.reward_head = MLP(feat, reward_units, reward_layers, 1)

        # RSSM cell.
        self._cell_in = nn.Sequential(nn.Linear(stoch + action_dim, hidden), nn.ReLU())
        self._gru = nn.GRUCell(hidden, deter)
        self._prior_net = nn.Sequential(
            nn.Linear(deter, hidden), nn.ReLU(), nn.Linear(hidden, 2 * stoch)
        )
        self._post_net = nn.Sequential(
            nn.Linear(deter + embed, hidden), nn.ReLU(), nn.Linear(hidden, 2 * stoch)
        )

    # -- state helpers ----------------------------------------------------- #

    def initial(self, batch: int, device: torch.device) -> State:
        z = torch.zeros(batch, self.stoch, device=device)
        return {
            "deter": torch.zeros(batch, self.deter, device=device),
            "stoch": z,
            "mean": z,
            "std": torch.ones(batch, self.stoch, device=device),
        }

    def _to_dist(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = params.chunk(2, dim=-1)
        std = F.softplus(std) + self.min_std
        return mean, std

    def feat(self, state: State) -> torch.Tensor:
        # [h_t, z_t] model state. (..., deter + stoch)
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    # -- one-step transitions --------------------------------------------- #

    def img_step(self, prev: State, prev_action: torch.Tensor) -> State:
        # prior step: (B, *) -> prior State at next t.
        x = torch.cat([prev["stoch"], prev_action], dim=-1)  # (B, stoch + A)
        deter = self._gru(self._cell_in(x), prev["deter"])  # (B, deter)
        mean, std = self._to_dist(self._prior_net(deter))
        stoch = mean + std * torch.randn_like(std)  # reparameterized sample
        return {"deter": deter, "stoch": stoch, "mean": mean, "std": std}

    def obs_step(
        self, prev: State, prev_action: torch.Tensor, embed: torch.Tensor
    ) -> tuple[State, State]:
        prior = self.img_step(prev, prev_action)
        x = torch.cat([prior["deter"], embed], dim=-1)  # (B, deter + embed)
        mean, std = self._to_dist(self._post_net(x))
        stoch = mean + std * torch.randn_like(std)
        post = {"deter": prior["deter"], "stoch": stoch, "mean": mean, "std": std}
        return post, prior

    # -- sequence rollout (buffer contract: t=0 init + is_first reset) ----- #

    def observe(
        self, embed: torch.Tensor, action: torch.Tensor, is_first: torch.Tensor
    ) -> tuple[State, State]:
        # embed (B, L, E), action (B, L, A), is_first (B, L) -> post/prior (B, L, *)
        b, length = embed.shape[:2]
        state = self.initial(b, embed.device)  # rule 1: every window inits at t=0
        posts: list[State] = []
        priors: list[State] = []
        for t in range(length):
            # rule 2: reset prev state + action wherever a new episode begins.
            keep = (1.0 - is_first[:, t].float())[:, None]  # (B, 1)
            state = {k: v * keep for k, v in state.items()}
            prev_action = action[:, t] * keep
            post, prior = self.obs_step(state, prev_action, embed[:, t])
            posts.append(post)
            priors.append(prior)
            state = post
        return self._stack(posts), self._stack(priors)

    @staticmethod
    def _stack(states: list[State]) -> State:
        return {k: torch.stack([s[k] for s in states], dim=1) for k in states[0]}

    # -- image / reward heads (handle (B, L, ...) by folding) -------------- #

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        # image (B, L, C, H, W) -> embed (B, L, E)
        b, length = image.shape[:2]
        x = einops.rearrange(image, "b l c h w -> (b l) c h w")
        e = self.encoder(x)
        return einops.rearrange(e, "(b l) e -> b l e", b=b, l=length)

    def decode(self, feat: torch.Tensor) -> torch.Tensor:
        # feat (B, L, F) -> image (B, L, C, H, W)
        b, length = feat.shape[:2]
        x = einops.rearrange(feat, "b l f -> (b l) f")
        img = self.decoder(x)
        return einops.rearrange(img, "(b l) c h w -> b l c h w", b=b, l=length)

    def predict_reward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat (B, L, F) -> reward (B, L)
        return self.reward_head(feat).squeeze(-1)

    # -- losses ------------------------------------------------------------ #

    def _overshooting_kl(
        self, post: State, action: torch.Tensor, is_first: torch.Tensor
    ) -> torch.Tensor:
        # Latent overshooting: roll prior d steps from posterior@t, KL to sg(post@t+d).
        b, length = action.shape[:2]
        device = action.device
        big_d = self.overshoot_d if self.overshoot_d > 0 else 1
        kl_sum = torch.zeros((), device=device)
        count = torch.zeros((), device=device)
        for t in range(length - 1):
            state = {"deter": post["deter"][:, t], "stoch": post["stoch"][:, t]}
            alive = torch.ones(b, device=device)
            for d in range(1, big_d + 1):
                idx = t + d
                if idx >= length:
                    break
                alive = alive * (1.0 - is_first[:, idx].float())  # stop at boundary
                state = self.img_step(state, action[:, idx])  # prior at idx
                tgt_mean = post["mean"][:, idx].detach()
                tgt_std = post["std"][:, idx].detach()
                kl = kl_normal_sum(tgt_mean, tgt_std, state["mean"], state["std"])  # (B,)
                kl = torch.clamp(kl, min=self.free_nats)
                kl_sum = kl_sum + (kl * alive).sum()
                count = count + alive.sum()
        if count == 0:
            return torch.zeros((), device=device)
        return kl_sum / count

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        image = batch["image"]  # (B, L, C, H, W) float [0, 1]
        action = batch["action"].float()  # (B, L, A)
        reward = batch["reward"].float()  # (B, L)
        is_first = batch["is_first"]  # (B, L) bool

        embed = self.encode(image)
        post, prior = self.observe(embed, action, is_first)
        feat = self.feat(post)

        # Reconstruction NLL: unit-variance Gaussian -> 0.5 * sum_pixels (x - mu)^2.
        recon_mean = self.decode(feat)  # (B, L, C, H, W)
        recon_nll = 0.5 * (recon_mean - image).pow(2).sum(dim=[2, 3, 4]).mean()

        # Reward NLL: unit-variance Gaussian -> 0.5 * (r - mu)^2.
        reward_mean = self.predict_reward(feat)  # (B, L)
        reward_nll = 0.5 * (reward_mean - reward).pow(2).mean()

        # Monitoring: raw 1-step KL(post || prior) (no clamp/sg) for collapse watch.
        kl_raw = kl_normal_sum(post["mean"], post["std"], prior["mean"], prior["std"]).mean()
        kl_term = self._overshooting_kl(post, action, is_first)

        loss = (
            self.recon_scale * recon_nll
            + self.reward_scale * reward_nll
            + self.kl_scale * self.overshoot_beta * kl_term
        )
        metrics = {
            "loss": float(loss.detach()),
            "recon": float(recon_nll.detach()),
            "reward_mse": float((reward_mean - reward).pow(2).mean().detach()),
            "kl": float(kl_raw.detach()),
            "overshoot": float(kl_term.detach()),
        }
        return loss, metrics

    # -- open-loop prediction --------------------------------------------- #

    @torch.no_grad()
    def open_loop(
        self, batch: dict[str, torch.Tensor], context: int, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Context via posterior, then prior-only rollout (actions only, no obs).
        image = batch["image"]  # (B, L, C, H, W)
        action = batch["action"].float()
        is_first = batch["is_first"]
        embed = self.encode(image[:, :context])
        post, _ = self.observe(embed, action[:, :context], is_first[:, :context])
        state = {"deter": post["deter"][:, -1], "stoch": post["stoch"][:, -1]}

        preds: list[torch.Tensor] = []
        for t in range(context, context + horizon):
            state = self.img_step(state, action[:, t])  # prior only, no observation
            preds.append(self.decoder(self.feat(state)))  # (B, C, H, W)
        pred = torch.stack(preds, dim=1)  # (B, horizon, C, H, W)
        real = image[:, context : context + horizon]
        return real, pred
