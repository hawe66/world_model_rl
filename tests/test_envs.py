import warnings

import numpy as np
import pytest

from wm.common.envs.base import Transition


def test_transition_holds_required_fields():
    t = Transition(
        obs={"image": np.zeros((64, 64, 3), dtype=np.uint8)},
        action=np.zeros(1, dtype=np.float32),
        reward=0.0,
        is_first=True,
        is_last=False,
        is_terminal=False,
    )
    assert t.obs["image"].dtype == np.uint8
    assert t.reward == 0.0
    assert t.is_first and not t.is_last and not t.is_terminal


# --------------------------------------------------------------------------- #
# Atari                                                                        #
# --------------------------------------------------------------------------- #


def _atari(**kwargs):
    from wm.common.envs.atari import AtariEnv

    defaults = dict(env_id="ALE/Pong-v5", image_size=64, action_repeat=4, seed=0)
    defaults.update(kwargs)
    return AtariEnv(**defaults)


def test_atari_reset_grayscale_shape_dtype():
    env = _atari()
    t = env.reset()
    assert t.obs["image"].shape == (64, 64, 1)  # (H, W, C) grayscale
    assert t.obs["image"].dtype == np.uint8
    assert t.is_first and not t.is_last and not t.is_terminal
    assert t.reward == 0.0
    env.close()


def test_atari_rgb_config_gives_three_channels():
    env = _atari(grayscale=False)
    t = env.reset()
    assert t.obs["image"].shape == (64, 64, 3)  # (H, W, C) RGB
    env.close()


def test_atari_step_returns_valid_transition():
    env = _atari()
    env.reset()
    t = env.step(env.action_space.sample())
    assert t.obs["image"].shape == (64, 64, 1)
    assert t.obs["image"].dtype == np.uint8
    assert isinstance(t.reward, float)
    assert isinstance(t.is_last, bool) and isinstance(t.is_terminal, bool)
    assert not t.is_first
    env.close()


def test_atari_action_repeat_advances_four_frames():
    env = _atari(action_repeat=4)
    env.reset()
    env.step(env.action_space.sample())
    # One agent step must consume action_repeat raw ALE frames.
    assert env.frames_per_step == 4
    assert env.episode_frame_number == 4
    env.close()


def test_atari_determinism_same_seed():
    env1, env2 = _atari(seed=123), _atari(seed=123)
    t1, t2 = env1.reset(), env2.reset()
    assert np.array_equal(t1.obs["image"], t2.obs["image"])
    rng = np.random.default_rng(0)
    for _ in range(12):
        a = int(rng.integers(env1.action_space.n))
        s1, s2 = env1.step(a), env2.step(a)
        assert np.array_equal(s1.obs["image"], s2.obs["image"])
        assert s1.reward == s2.reward
        assert s1.is_last == s2.is_last and s1.is_terminal == s2.is_terminal
    env1.close()
    env2.close()


def test_atari_terminal_implies_last_over_rollout():
    # Invariant: a true terminal must also end the episode (is_terminal -> is_last).
    env = _atari()
    env.reset()
    rng = np.random.default_rng(1)
    for _ in range(30):
        t = env.step(int(rng.integers(env.action_space.n)))
        if t.is_terminal:
            assert t.is_last
        if t.is_last:
            break
    env.close()


# --------------------------------------------------------------------------- #
# DMC                                                                          #
# --------------------------------------------------------------------------- #


def _dmc(**kwargs):
    from wm.common.envs.dmc import DMCEnv

    defaults = dict(domain="walker", task="walk", action_repeat=2, seed=0)
    defaults.update(kwargs)
    return DMCEnv(**defaults)


_RENDER_AVAILABLE: bool | None = None


def _dmc_render_available() -> bool:
    """DMC vision needs an OpenGL backend; skip vision tests where absent.

    Cached so the (noisy on headless machines) probe runs at most once.
    """
    global _RENDER_AVAILABLE
    if _RENDER_AVAILABLE is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                env = _dmc(vision=True)
                env.reset()
                env.close()
                _RENDER_AVAILABLE = True
            except Exception:
                _RENDER_AVAILABLE = False
    return _RENDER_AVAILABLE


def test_dmc_proprio_reset_shape_dtype():
    env = _dmc()
    t = env.reset()
    assert t.obs["state"].dtype == np.float32
    assert t.obs["state"].shape == (24,)  # walker-walk proprio dim
    assert t.is_first and not t.is_last and not t.is_terminal
    assert t.reward == 0.0
    env.close()


def test_dmc_proprio_step_returns_valid_transition():
    env = _dmc()
    env.reset()
    t = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert t.obs["state"].shape == (24,)
    assert isinstance(t.reward, float)
    assert not t.is_first
    assert not t.is_terminal  # DMC has no true terminal
    env.close()


def test_dmc_cheetah_proprio_dim():
    env = _dmc(domain="cheetah", task="run")
    t = env.reset()
    assert t.obs["state"].shape == (17,)  # cheetah-run proprio dim
    env.close()


def test_dmc_action_repeat_consumes_two_control_steps():
    env = _dmc(action_repeat=2)
    env.reset()
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    env.step(a)
    env.step(a)
    assert env.frames_per_step == 2
    assert env.control_steps == 4  # 2 agent steps * action_repeat 2
    env.close()


def test_dmc_determinism_same_seed():
    env1, env2 = _dmc(seed=0), _dmc(seed=0)
    t1, t2 = env1.reset(), env2.reset()
    assert np.allclose(t1.obs["state"], t2.obs["state"])
    rng = np.random.default_rng(2)
    for _ in range(40):
        a = rng.uniform(-1.0, 1.0, size=env1.action_space.shape).astype(np.float32)
        s1, s2 = env1.step(a), env2.step(a)
        assert np.allclose(s1.obs["state"], s2.obs["state"], atol=1e-5)
        assert s1.reward == s2.reward
        assert s1.is_last == s2.is_last
    env1.close()
    env2.close()


def test_dmc_is_terminal_always_false():
    env = _dmc()
    env.reset()
    rng = np.random.default_rng(3)
    for _ in range(50):
        t = env.step(rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32))
        assert not t.is_terminal
    env.close()


@pytest.mark.skipif(not _dmc_render_available(), reason="no OpenGL backend for DMC rendering")
def test_dmc_vision_obs_shape():
    env = _dmc(vision=True, image_size=64)
    t = env.reset()
    assert t.obs["image"].shape == (64, 64, 3)  # (H, W, C) RGB
    assert t.obs["image"].dtype == np.uint8
    env.close()
