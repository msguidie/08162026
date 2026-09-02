"""End-to-end trainer: the loop closes, and it survives a restart.

The heavy case runs ``configs/smoke_short.yaml`` — a ~2.5 minute version of the
G3 smoke gate — and only asks whether the machinery closes: records arrive, the
buffer fills, generations seal, weights get published, an evaluation returns and
checkpoint/resume restores the counters.  Whether anything is *learned* is
``configs/smoke_cpu.yaml``'s job (``scripts/smoke_cpu.sh``, ~21 minutes).

It is opt-in because it costs minutes of CPU:

    SPLENDOR_SELFPLAY_SMOKE=1 pytest splendor_ai/tests/test_selfplay_loop.py -q
    pytest -m selfplay_smoke ...          # the same tests, once enabled
"""

import json
import os

import numpy as np
import pytest
import torch

from splendor_ai.model import SplendorNet, load_checkpoint, save_checkpoint
from splendor_ai.selfplay.config import load_config
from splendor_ai.selfplay.learner import Learner, lr_at
from splendor_ai.selfplay.replay import ReplayBuffer
from splendor_ai.selfplay.train import Trainer, main

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "configs")
SMOKE = os.environ.get("SPLENDOR_SELFPLAY_SMOKE", "") not in ("", "0", "false")
requires_smoke = pytest.mark.skipif(
    not SMOKE, reason="set SPLENDOR_SELFPLAY_SMOKE=1 to run the ~3 minute loop")


def _read(path):
    return [json.loads(line) for line in open(path)]


# ── cheap checks (always run) ─────────────────────────────────────────────

def test_lr_schedule_warms_up_then_decays():
    assert lr_at(0, 1e-3, 1e-4, 100, 1000) == pytest.approx(1e-5, rel=1e-6)
    assert lr_at(99, 1e-3, 1e-4, 100, 1000) == pytest.approx(1e-3, rel=1e-6)
    assert lr_at(1000, 1e-3, 1e-4, 100, 1000) == pytest.approx(1e-4, rel=1e-3)
    mid = lr_at(550, 1e-3, 1e-4, 100, 1000)
    assert 1e-4 < mid < 1e-3


def test_learner_rejects_a_leaking_policy_target(tmp_path):
    cfg = load_config(None, [f"run_dir={tmp_path}/run", "net.width=32",
                             "net.blocks=1", "learner.batch=4"])
    cfg.make_dirs()
    learner = Learner(cfg)
    batch = {
        "obs": np.zeros((4, cfg.net.obs_dim), dtype=np.float32),
        "mask": np.zeros((4, 65), dtype=bool),
        "policy_target": np.zeros((4, 65), dtype=np.float32),
        "z": np.zeros((4, 4), dtype=np.float32),
        "z_valid": np.ones((4, 4), dtype=np.float32),
        "z_weight": np.ones(4, dtype=np.float32),
        "score_target": np.zeros((4, 4), dtype=np.float32),
        "stuck_target": np.zeros((4, 4), dtype=np.float32),
    }
    batch["mask"][:, :3] = True
    batch["policy_target"][:, 0] = 1.0
    learner.train_step(batch)                       # legal target: fine
    batch["policy_target"][:] = 0.0
    batch["policy_target"][:, 40] = 1.0             # illegal action
    with pytest.raises(AssertionError):
        learner.train_step(batch)
    batch["policy_target"][:] = 0.0                 # does not sum to 1
    batch["policy_target"][:, 0] = 0.5
    with pytest.raises(AssertionError):
        learner.train_step(batch)


def test_learner_replay_ratio_throttle(tmp_path):
    cfg = load_config(None, [f"run_dir={tmp_path}/run", "net.width=32",
                             "net.blocks=1", "learner.batch=64",
                             "learner.replay_ratio=2.0", "replay.min_samples=64"])
    cfg.make_dirs()
    learner = Learner(cfg)
    assert not learner.ready(samples_produced=1000, buffer_size=8)   # too small
    assert learner.ready(samples_produced=1000, buffer_size=1000)
    learner.samples_consumed = 2000
    assert not learner.ready(samples_produced=1000, buffer_size=1000)


def test_obs_version_gate_fires_on_a_tampered_checkpoint(tmp_path):
    cfg = load_config(None, [f"run_dir={tmp_path}/run", "net.width=32",
                             "net.blocks=1"])
    cfg.make_dirs()
    path = save_checkpoint(cfg.latest_weights, SplendorNet(cfg.net),
                           {"step": 1, "generation": 0})
    load_checkpoint(path)                                   # fine as written
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["obs_version"] = 999
    torch.save(payload, path)
    with pytest.raises(RuntimeError) as exc:
        load_checkpoint(path)
    assert "obs_version" in str(exc.value) and "999" in str(exc.value)

    payload["obs_version"] = 1
    payload["action_version"] = 999
    torch.save(payload, path)
    with pytest.raises(RuntimeError) as exc:
        load_checkpoint(path)
    assert "action_version" in str(exc.value)


def test_trainer_state_gate_rejects_a_foreign_encoder(tmp_path):
    cfg = load_config(None, [f"run_dir={tmp_path}/run", "net.width=32",
                             "net.blocks=1"])
    cfg.make_dirs()
    learner = Learner(cfg)
    state = learner.state_dict()
    state["obs_version"] = 42
    with pytest.raises(RuntimeError):
        Learner(cfg).load_state_dict(state)


# ── the real loop (opt-in) ────────────────────────────────────────────────

@pytest.mark.selfplay_smoke
@requires_smoke
def test_smoke_short_loop_closes(tmp_path):
    run_dir = str(tmp_path / "smoke_short")
    rc = main(["--config", os.path.join(CONFIG_DIR, "smoke_short.yaml"),
               "--set", f"run_dir={run_dir}", "--set", "max_seconds=140"])
    assert rc == 0

    rows = _read(os.path.join(run_dir, "metrics.jsonl"))
    kinds = {r["kind"] for r in rows}
    assert {"run", "learner", "generation", "eval", "summary"} <= kinds
    summary = [r for r in rows if r["kind"] == "summary"][-1]
    assert summary["games_done"] > 0
    assert summary["steps"] > 0
    assert summary["generations"] >= 1
    assert summary["actor_restarts"] == 0
    assert summary["buffer"] > 0

    # weights were published, and they load through the version gate
    weights = os.path.join(run_dir, "weights", "latest.pt")
    assert os.path.exists(weights)
    _model, ckpt = load_checkpoint(weights)
    assert ckpt["step"] == summary["steps"]

    # a generation checkpoint exists for the opponent pool / arena
    gens = sorted(os.listdir(os.path.join(run_dir, "checkpoints")))
    assert gens and gens[0].startswith("gen_")

    # the replay buffer round-trips
    replay = ReplayBuffer().load(os.path.join(run_dir, "replay.npz"))
    assert len(replay) > 0
    batch = replay.batch(32)
    assert batch["obs"].shape[0] == 32

    # every evaluation returned a number
    evals = [r for r in rows if r["kind"] == "eval"]
    assert evals and all("error" not in e for e in evals)
    assert all(0.0 <= e["net_vs_random"] <= 1.0 for e in evals)


@pytest.mark.selfplay_smoke
@requires_smoke
def test_resume_restores_counters(tmp_path):
    run_dir = str(tmp_path / "resume")
    args = ["--config", os.path.join(CONFIG_DIR, "smoke_short.yaml"),
            "--set", f"run_dir={run_dir}", "--set", "max_seconds=40",
            "--set", "eval.enabled=false"]
    assert main(args) == 0
    first = [r for r in _read(os.path.join(run_dir, "metrics.jsonl"))
             if r["kind"] == "summary"][-1]
    assert first["steps"] > 0

    assert main(["--resume", run_dir, "--set", "max_seconds=30",
                 "--set", "eval.enabled=false"]) == 0
    rows = [r for r in _read(os.path.join(run_dir, "metrics.jsonl"))
            if r["kind"] == "summary"]
    second = rows[-1]
    assert second["steps"] > first["steps"]
    assert second["games_done"] > first["games_done"]
    assert second["generations"] >= first["generations"]
