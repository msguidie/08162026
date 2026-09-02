"""The PPO fallback learner (``learner: ppo``).

These cover the parts that are implemented: the GAE/return maths over per-seat
decision points, the margin-scaled terminal reward, and that ``PPOLearner``
satisfies the orchestrator's learner interface so ``train.py`` can drive it
without a PPO-specific branch.  The search-free actor that would feed it is a
documented TODO at the top of ``selfplay/ppo_learner.py``.
"""

import numpy as np
import pytest

from splendor_ai.selfplay.config import load_config
from splendor_ai.selfplay.ppo_learner import (PPOConfig, PPOLearner, compute_gae,
                                              terminal_reward)


def _batch(n=8, obs_dim=None):
    from splendor_ai.encode import OBS_DIM

    obs_dim = obs_dim or OBS_DIM
    rng = np.random.default_rng(0)
    mask = np.zeros((n, 65), dtype=bool)
    mask[:, :5] = True
    policy = np.zeros((n, 65), dtype=np.float32)
    policy[:, :5] = 1.0 / 5
    return {
        "obs": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "mask": mask,
        "policy_target": policy,
        "z": rng.uniform(-1, 1, size=(n, 4)).astype(np.float32),
        "z_valid": np.tile(np.array([1, 1, 0, 0], dtype=np.float32), (n, 1)),
        "z_weight": np.ones(n, dtype=np.float32),
        "score_target": np.zeros((n, 4), dtype=np.float32),
        "stuck_target": np.zeros((n, 4), dtype=np.float32),
    }


def test_gae_on_a_terminal_only_reward_discounts_backwards():
    values = np.zeros(4, dtype=np.float32)
    rewards = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    dones = np.array([0, 0, 0, 1], dtype=bool)
    adv, ret = compute_gae(rewards, values, dones, gamma=0.9, lam=1.0)
    assert adv[-1] == pytest.approx(1.0)
    assert adv[-2] == pytest.approx(0.9)
    assert adv[0] == pytest.approx(0.9 ** 3)
    assert np.allclose(ret, adv + values)


def test_gae_stops_at_a_decision_point_boundary():
    """A ``done`` marks the end of one seat's chain; credit must not leak past
    it into the next game's decisions."""
    rewards = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
    values = np.zeros(4, dtype=np.float32)
    dones = np.array([0, 1, 0, 1], dtype=bool)
    adv, _ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    assert adv[1] == pytest.approx(1.0)
    assert adv[3] == pytest.approx(-1.0)
    assert adv[0] == pytest.approx(0.99 * 0.95 * 1.0, rel=1e-5)
    assert adv[2] == pytest.approx(0.99 * 0.95 * -1.0, rel=1e-5)


def test_terminal_reward_scales_with_the_margin():
    values = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32)
    close = terminal_reward(values, [9, 8], margin_scale=0.5)
    blowout = terminal_reward(values, [20, 2], margin_scale=0.5)
    assert np.sign(close[0]) == 1 and np.sign(close[1]) == -1
    assert abs(blowout[0]) > abs(close[0])
    assert abs(blowout[1]) > abs(close[1])
    zero = terminal_reward(np.zeros(4, dtype=np.float32), [5, 5])
    assert np.allclose(zero, 0.0)


def test_ppo_learner_satisfies_the_orchestrator_interface(tmp_path):
    cfg = load_config(None, [f"run_dir={tmp_path}/run", "net.width=32",
                             "net.blocks=1", "learner.algorithm=ppo",
                             "learner.ppo_experimental=true",
                             "learner.batch=8", "replay.min_samples=8"])
    cfg.make_dirs()
    learner = PPOLearner(cfg, ppo=PPOConfig(epochs=2, minibatch=4))
    for attr in ("ready", "train_step", "publish", "save_generation",
                 "state_dict", "load_state_dict", "warm_start", "local_batch",
                 "step", "generation", "samples_consumed"):
        assert hasattr(learner, attr)
    assert learner.ready(samples_produced=100, buffer_size=100)

    batch = _batch(8)
    before = [p.detach().clone() for p in learner.model.parameters()]
    metrics = learner.train_step(batch)
    for key in ("policy", "value", "entropy", "approx_kl", "clip_frac"):
        assert key in metrics
    assert learner.step == 1 and learner.samples_consumed == 8
    after = list(learner.model.parameters())
    assert any(not p.equal(q) for p, q in zip(before, after)), "no update happened"

    path = learner.publish()
    from splendor_ai.model import load_checkpoint

    _model, ckpt = load_checkpoint(path)
    assert ckpt["meta"]["learner"] == "ppo"
    state = learner.state_dict()
    fresh = PPOLearner(cfg)
    fresh.load_state_dict(state)
    assert fresh.step == learner.step


def test_train_py_selects_the_ppo_learner(tmp_path):
    from splendor_ai.selfplay.train import Trainer

    cfg = load_config(None, [f"run_dir={tmp_path}/run", "net.width=32",
                             "net.blocks=1", "learner.algorithm=ppo",
                             "learner.ppo_experimental=true",
                             "selfplay.actors=1", "eval.enabled=false"])
    trainer = Trainer(cfg)
    assert isinstance(trainer.learner, PPOLearner)
    trainer.metrics.close()
