import torch

from wm.algos.rssm.rssm import RSSM, kl_normal_sum


def _rssm(**kw):
    defaults = dict(
        action_dim=3,
        deter=8,
        stoch=4,
        hidden=8,
        image_shape=(1, 64, 64),  # (C, H, W)
        cnn_depth=4,
        reward_units=8,
        reward_layers=2,
        min_std=0.1,
        free_nats=1.0,
        overshoot_d=3,
    )
    defaults.update(kw)
    return RSSM(**defaults)


def _batch(b=2, length=3, c=1):
    is_first = torch.zeros(b, length, dtype=torch.bool)
    is_first[:, 0] = True
    return {
        "image": torch.rand(b, length, c, 64, 64),  # (B, L, C, H, W) float [0,1]
        "action": torch.randn(b, length, 3),  # (B, L, A)
        "reward": torch.randn(b, length),  # (B, L)
        "is_first": is_first,
    }


def test_encode_observe_decode_shapes():
    m = _rssm()
    batch = _batch()
    embed = m.encode(batch["image"])  # (B, L, E)
    assert embed.shape[:2] == (2, 3)
    post, prior = m.observe(embed, batch["action"], batch["is_first"])
    assert post["deter"].shape == (2, 3, 8)
    assert post["stoch"].shape == (2, 3, 4)
    assert prior["mean"].shape == (2, 3, 4)
    feat = m.feat(post)  # (B, L, deter+stoch)
    assert feat.shape == (2, 3, 12)
    recon = m.decode(feat)  # (B, L, C, H, W)
    assert recon.shape == (2, 3, 1, 64, 64)
    rew = m.predict_reward(feat)  # (B, L)
    assert rew.shape == (2, 3)


def test_loss_backward_no_nan():
    m = _rssm()
    loss, metrics = m.loss(_batch())
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    for g in grads:
        if g is not None:
            assert torch.isfinite(g).all()
    assert "kl" in metrics and "recon" in metrics


def test_is_first_resets_deterministic_state():
    m = _rssm()
    torch.manual_seed(0)
    a = torch.randn(2, 3, 3)
    e1 = torch.randn(2, 3, m.encoder.embed_dim)
    e2 = e1.clone()
    e2[:, 0] = torch.randn(2, m.encoder.embed_dim)  # different history at t=0 only

    first_on = torch.zeros(2, 3, dtype=torch.bool)
    first_on[:, 0] = True
    first_on[:, 1] = True  # reset at t=1 too
    p1, _ = m.observe(e1, a, first_on)
    p2, _ = m.observe(e2, a, first_on)
    # deter at the reset step must NOT depend on pre-reset history.
    assert torch.allclose(p1["deter"][:, 1], p2["deter"][:, 1], atol=1e-6)

    first_off = torch.zeros(2, 3, dtype=torch.bool)
    first_off[:, 0] = True  # no reset at t=1
    q1, _ = m.observe(e1, a, first_off)
    q2, _ = m.observe(e2, a, first_off)
    # without reset, t=1 deter carries history -> differs.
    assert not torch.allclose(q1["deter"][:, 1], q2["deter"][:, 1], atol=1e-6)


def test_kl_normal_sum_and_free_nats():
    mean = torch.zeros(2, 4)
    std = torch.ones(2, 4)
    kl = kl_normal_sum(mean, std, mean, std)  # identical -> 0
    assert torch.allclose(kl, torch.zeros(2))
    free_nats = 1.5
    clamped = torch.clamp(kl, min=free_nats)
    assert torch.allclose(clamped, torch.full((2,), free_nats))


def test_overshoot_produces_finite_metric():
    m = _rssm(overshoot_d=3)
    _, metrics = m.loss(_batch(length=4))
    assert "overshoot" in metrics
    assert torch.isfinite(torch.tensor(metrics["overshoot"]))


def test_open_loop_shapes():
    m = _rssm()
    real, pred = m.open_loop(_batch(length=6), context=2, horizon=3)
    assert real.shape == pred.shape == (2, 3, 1, 64, 64)  # (B, horizon, C, H, W)
