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
import time

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

    # max_seconds is a RUN budget restored from the checkpoint, so the resumed
    # link needs a bigger one than the ~40 s the first link already spent.
    assert main(["--resume", run_dir, "--set", "max_seconds=80",
                 "--set", "eval.enabled=false"]) == 0
    rows = [r for r in _read(os.path.join(run_dir, "metrics.jsonl"))
            if r["kind"] == "summary"]
    second = rows[-1]
    assert second["steps"] > first["steps"]
    assert second["games_done"] > first["games_done"]
    assert second["generations"] >= first["generations"]


# ── shutdown order, resume bookkeeping, actor supervision (cheap) ─────────

def _trainer_cfg(tmp_path, *extra):
    base = [f"run_dir={tmp_path}/run", "net.width=32", "net.blocks=1",
            "selfplay.actors=1", "selfplay.games_per_actor=2",
            "learner.batch=8", "replay.min_samples=8",
            "eval.enabled=false", "selfplay.win_threshold=5"]
    cfg = load_config(None, base + list(extra))
    cfg.make_dirs()
    return cfg


def _stub_eval(monkeypatch, seen, state_path):
    """Replace the real evaluation and record whether it ran before the save."""
    import splendor_ai.selfplay.train as train_mod

    def fake(cfg, weights, generation=0):
        seen.append({"checkpoint_existed": os.path.exists(state_path),
                     "generation": generation})
        return {"generation": generation, "net_vs_random": 0.5,
                "net_vs_greedy": 0.5, "search_vs_greedy": 0.5}

    monkeypatch.setattr(train_mod, "evaluate_weights", fake)


def test_shutdown_checkpoints_before_it_evaluates(tmp_path, monkeypatch):
    cfg = _trainer_cfg(tmp_path, "eval.enabled=true", "eval.async_process=false")
    trainer = Trainer(cfg)
    seen = []
    _stub_eval(monkeypatch, seen, cfg.state_path)
    summary = trainer.shutdown()

    assert os.path.exists(cfg.state_path)                   # resumable
    assert os.path.exists(cfg.latest_weights)               # published
    assert len(seen) == 1 and seen[0]["checkpoint_existed"] is True
    assert summary["evals"] and summary["evals"][-1]["final"] is True


def test_a_signal_skips_the_final_evaluation_but_never_the_checkpoint(
        tmp_path, monkeypatch):
    """The PBS case: SIGINT, then SIGKILL ten minutes later."""
    cfg = _trainer_cfg(tmp_path, "eval.enabled=true", "eval.async_process=false",
                       "eval.final_eval_seconds=0")
    trainer = Trainer(cfg)
    seen = []
    _stub_eval(monkeypatch, seen, cfg.state_path)
    trainer.stopped_by_signal = True                        # as the handler does
    summary = trainer.shutdown()

    assert seen == []                                       # no evaluation at all
    assert summary["stopped_by_signal"] is True
    assert not summary["evals"]
    # ...and the checkpoint is complete enough to resume from.
    assert os.path.exists(cfg.state_path) and os.path.exists(cfg.replay_path)
    state = torch.load(cfg.state_path, map_location="cpu", weights_only=False)
    assert {"learner", "replay", "games_done", "generation", "run_instance",
            "elapsed_s"} <= set(state)
    rows = _read(cfg.metrics_path)
    assert any(r["kind"] == "eval_skipped" for r in rows)
    assert any(r["kind"] == "checkpoint" and r["final"] for r in rows)


def test_resume_bumps_the_instance_and_keeps_the_clock(tmp_path):
    cfg = _trainer_cfg(tmp_path)
    first = Trainer(cfg)
    first.games_done = 120
    first.records_seen = 900
    first.elapsed_base = 0.0
    first.t0 = time.monotonic() - 30.0
    first.checkpoint()
    launch_one = {first._next_instance() for _ in range(4)}
    first.metrics.close()

    second = Trainer(load_config(None, [f"run_dir={cfg.run_dir}",
                                        "net.width=32", "net.blocks=1",
                                        "eval.enabled=false"]), resume=True)
    try:
        assert second.run_instance == first.run_instance + 1
        # counters restored, and remembered as the base for this launch
        assert second.games_done == 120 and second.base_games == 120
        assert second.records_seen == 900 and second.base_records == 900
        assert second.elapsed_base >= 30.0                  # cumulative clock
        launch_two = {second._next_instance() for _ in range(4)}
        assert not (launch_one & launch_two)                # disjoint nonces
    finally:
        second.metrics.close()


def test_max_seconds_is_a_whole_run_budget(tmp_path):
    cfg = _trainer_cfg(tmp_path, "max_seconds=100")
    trainer = Trainer(cfg)
    try:
        assert trainer._should_stop() is False
        trainer.elapsed_base = 90.0                         # an earlier link
        assert trainer._should_stop() is False
        trainer.elapsed_base = 101.0
        assert trainer._should_stop() is True               # budget spent
    finally:
        trainer.metrics.close()


def test_throughput_separates_this_launch_from_the_lifetime(tmp_path):
    cfg = _trainer_cfg(tmp_path)
    trainer = Trainer(cfg)
    try:
        trainer.base_games, trainer.games_done = 10_000, 10_100
        trainer.base_records, trainer.records_seen = 500_000, 505_000
        trainer.base_steps = 0
        trainer.elapsed_base = 3_600.0
        trainer.t0 = time.monotonic() - 10.0
        tp = trainer._throughput()
        # 100 games in this launch's 10 s, not 10,100 of them.
        assert tp["games_per_s"] == pytest.approx(10.0, rel=0.2)
        assert tp["games_this_launch"] == 100
        assert tp["lifetime_s"] >= 3_610.0
        assert tp["lifetime_games_per_s"] == pytest.approx(10_100 / 3_610.0,
                                                           rel=0.2)
    finally:
        trainer.metrics.close()


def test_actor_restarts_back_off_and_then_abort_the_run(tmp_path):
    cfg = _trainer_cfg(tmp_path, "selfplay.restart_budget=3",
                       "selfplay.restart_window_s=600",
                       "selfplay.restart_backoff_s=2",
                       "selfplay.restart_backoff_max_s=5")
    trainer = Trainer(cfg)
    try:
        now = 1_000.0
        assert trainer._note_actor_death(0, 1, now) == now + 2      # 2 * 2**0
        assert trainer._note_actor_death(0, 1, now) == now + 4      # 2 * 2**1
        assert trainer._note_actor_death(0, 1, now) == now + 5      # capped
        with pytest.raises(RuntimeError) as exc:
            trainer._note_actor_death(0, 1, now)            # over budget
        assert "restart_budget" in str(exc.value)
        assert "died 4 times" in str(exc.value)
        # A death long ago does not count against the budget.
        trainer._restart_times[1] = [now - 10_000.0] * 9
        assert trainer._note_actor_death(1, 1, now) == now + 2
        rows = _read(cfg.metrics_path)
        assert sum(1 for r in rows if r["kind"] == "actor_restart") == 5
    finally:
        trainer.metrics.close()


def test_a_replacement_actor_gets_a_new_nonce_and_a_drained_queue(tmp_path):
    """`_start_actor` is the only place both halves of the fix can be seen."""
    import queue as _queue
    import types

    cfg = _trainer_cfg(tmp_path, "inference.mode=server",
                       "inference.devices=[cpu]")
    trainer = Trainer(cfg)
    spawned = []

    class _Proc:
        def __init__(self, **kw):
            self.kw = kw
            spawned.append(kw)

        def start(self):
            pass

        def is_alive(self):
            return True

    trainer.ctx = types.SimpleNamespace(Process=lambda **kw: _Proc(**kw))
    trainer.request_qs = [_queue.Queue()]
    trainer.response_qs = {0: _queue.Queue()}
    trainer.response_qs[0].put(("a stale reply", b"", b"", 0))
    trainer.response_qs[0].put(("another", b"", b"", 0))
    trainer.games_done = 4_242

    try:
        trainer._start_actor(0)
        assert trainer.response_qs[0].empty()               # drained
        assert trainer._stale_replies[0] == 2
        first = spawned[-1]["kwargs"]
        assert first["games_offset"] == 4_242                # curriculum offset
        trainer._start_actor(0)                             # the restart
        second = spawned[-1]["kwargs"]
        assert second["instance"] != first["instance"]
    finally:
        trainer.metrics.close()


def test_trainer_publishes_run_global_progress(tmp_path):
    cfg = _trainer_cfg(tmp_path)
    trainer = Trainer(cfg)
    try:
        trainer.games_done = 777
        trainer.generation = 5
        trainer._write_progress()
        data = json.load(open(cfg.progress_path))
        assert data["games_done"] == 777 and data["generation"] == 5
        assert data["instance"] == trainer.run_instance
        # ...and it is exactly what an actor reads for its curriculum phase.
        from splendor_ai.selfplay.actor import read_progress

        assert read_progress(cfg.progress_path)["games_done"] == 777
    finally:
        trainer.metrics.close()


def test_checkpoint_retention_keeps_the_pool_and_the_milestones(tmp_path):
    cfg = load_config(None, [f"run_dir={tmp_path}/run", "net.width=32",
                             "net.blocks=1", "selfplay.historical_pool_size=3",
                             "learner.checkpoint_retention=true"])
    cfg.make_dirs()
    learner = Learner(cfg)
    for gen in range(12):
        open(os.path.join(cfg.checkpoints_dir, f"gen_{gen:04d}.pt"), "wb").close()
    learner.save_generation(12)

    kept = sorted(int(n[4:-3]) for n in os.listdir(cfg.checkpoints_dir))
    # the live opponent pool (10, 11, 12) plus the milestone ladder (5, 10)
    assert kept == [5, 10, 11, 12]
    # a milestone far out is kept too, and the pool floor moves with it
    for gen in (39, 40, 41):
        open(os.path.join(cfg.checkpoints_dir, f"gen_{gen:04d}.pt"), "wb").close()
    learner.save_generation(42)
    kept = sorted(int(n[4:-3]) for n in os.listdir(cfg.checkpoints_dir))
    assert kept == [5, 10, 40, 41, 42]


def test_retention_milestones():
    from splendor_ai.selfplay.learner import is_milestone_generation

    assert [g for g in range(200) if is_milestone_generation(g)] == \
        [5, 10, 20, 40, 80, 120, 160]
    assert not is_milestone_generation(0)


@pytest.mark.selfplay_smoke
@requires_smoke
def test_sigint_leaves_a_complete_checkpoint_and_runs_no_eval(tmp_path):
    """The PBS chain's stop, reproduced: SIGINT to the whole process group.

    `timeout --signal=INT ... --kill-after=10m` signals every process in the
    job, so this test does the same (`killpg`).  What must survive it: a
    resumable `trainer_state.pt`, published weights, the games that had
    finished — and no final evaluation, because the SIGKILL is ten minutes out.
    """
    import signal
    import subprocess
    import sys

    run_dir = str(tmp_path / "sigint")
    proc = subprocess.Popen(
        [sys.executable, "-m", "splendor_ai.selfplay.train",
         "--config", os.path.join(CONFIG_DIR, "smoke_short.yaml"),
         "--set", f"run_dir={run_dir}", "--set", "max_seconds=600"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)                             # its own group
    try:
        metrics = os.path.join(run_dir, "metrics.jsonl")
        deadline = time.time() + 240
        games = 0
        while time.time() < deadline:
            if os.path.exists(metrics):
                rows = _read(metrics)
                games = max([r.get("games_done", 0) for r in rows
                             if r["kind"] in ("generation", "checkpoint",
                                              "progress")] or [0])
                if games > 0:
                    break
            time.sleep(1.0)
        assert games > 0, "the run produced no games before the signal"
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        out, _ = proc.communicate(timeout=240)
    finally:
        if proc.poll() is None:                             # pragma: no cover
            proc.kill()
            proc.communicate(timeout=30)

    assert proc.returncode == 0, out[-4000:]
    assert "signal 2: stopping (checkpoint first)" in out
    assert "final evaluation skipped" in out
    assert "did not stop within" not in out                 # no actor killed

    rows = _read(os.path.join(run_dir, "metrics.jsonl"))
    summary = [r for r in rows if r["kind"] == "summary"][-1]
    assert summary["stopped_by_signal"] is True
    assert not [e for e in summary["evals"] if e.get("final")]
    assert any(r["kind"] == "eval_skipped" for r in rows)
    assert summary["games_done"] >= games                   # nothing lost
    assert summary["actor_restarts"] == 0

    # The checkpoint is complete: it resumes, with its counters intact.
    state = torch.load(os.path.join(run_dir, "trainer_state.pt"),
                       map_location="cpu", weights_only=False)
    assert state["games_done"] == summary["games_done"]
    assert state["run_instance"] == 1
    assert os.path.exists(os.path.join(run_dir, "weights", "latest.pt"))
    replay = ReplayBuffer().load(os.path.join(run_dir, "replay.npz"))
    assert len(replay) > 0
    assert main(["--resume", run_dir, "--set", "max_seconds=1",
                 "--set", "eval.enabled=false"]) == 0


def test_replay_checkpoints_in_the_background_but_never_on_the_way_out(tmp_path):
    cfg = _trainer_cfg(tmp_path, "replay.checkpoint_async=true")
    trainer = Trainer(cfg)
    try:
        trainer.checkpoint()
        assert trainer.replay.join_save(timeout=60.0)
        trainer.checkpoint(final=True)
        rows = [r for r in _read(cfg.metrics_path) if r["kind"] == "checkpoint"]
        assert [r["replay_save"] for r in rows] == ["async", "sync"]
        assert os.path.exists(cfg.replay_path)
    finally:
        trainer.metrics.close()


def test_a_truncated_replay_window_is_warned_about_once(tmp_path, capsys):
    cfg = _trainer_cfg(tmp_path, "replay.window_start=8", "replay.window_end=8",
                       "replay.max_samples=32")
    trainer = Trainer(cfg)
    try:
        from splendor_ai.selfplay.sample import empty

        for _ in range(4):
            trainer.replay.add(empty(20))
            trainer.replay.close_generation()
        assert trainer.replay.window_truncated()
        trainer._warn_if_window_truncated()
        trainer._warn_if_window_truncated()                 # once, not per gen
        out = capsys.readouterr().out
        assert out.count("is truncating the window") == 1
        rows = [r for r in _read(cfg.metrics_path)
                if r["kind"] == "replay_window_truncated"]
        assert len(rows) == 1
        assert rows[0]["window_retained"] < rows[0]["window"]
    finally:
        trainer.metrics.close()


def test_progress_logging_aggregates_the_per_mode_counters(tmp_path):
    cfg = _trainer_cfg(tmp_path)
    trainer = Trainer(cfg)
    try:
        trainer.actor_stats = {
            0: {"actor": 0, "sims": 100.0, "moves": 10.0, "games": 4.0,
                "stuck_rate": 0.0, "truncation_rate": 0.25,
                "mode_plies": {"ind2": 30.0}, "mode_games": {"ind2": 4},
                "mode_truncations": {"ind2": 1}, "global_games": 40,
                "weight_generation": 2, "eval_stale": 0, "historical_loads": 1},
            1: {"actor": 1, "sims": 50.0, "moves": 5.0, "games": 2.0,
                "stuck_rate": 0.5, "truncation_rate": 0.0,
                "mode_plies": {"ovt": 44.0}, "mode_games": {"ovt": 2, "ind2": 1},
                "mode_truncations": {}, "global_games": 38,
                "weight_generation": 3, "eval_stale": 2, "historical_loads": 5},
        }
        trainer._log_progress()
        row = [r for r in _read(cfg.metrics_path) if r["kind"] == "progress"][-1]
        assert row["mode_games"] == {"ind2": 5, "ovt": 2}
        assert row["mode_truncations"] == {"ind2": 1}
        assert row["mode_truncation_rate"]["ind2"] == pytest.approx(0.2)
        assert row["mode_truncation_rate"]["ovt"] == 0.0
        assert row["stuck_rate"] == pytest.approx(0.25)
        assert row["truncation_rate"] == pytest.approx(0.125)
        assert row["weight_generation"] == 3                # the newest seen
        assert row["eval_stale"] == 2 and row["historical_loads"] == 6
        assert row["actor_global_games"] == 40
    finally:
        trainer.metrics.close()


def test_the_learner_refuses_a_world_size_it_cannot_honour(tmp_path, monkeypatch):
    from splendor_ai.selfplay.learner import DDP_NOT_WIRED

    cfg = load_config(None, [f"run_dir={tmp_path}/run", "net.width=32",
                             "net.blocks=1"])
    cfg.make_dirs()
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(RuntimeError) as exc:
        Learner(cfg)
    assert DDP_NOT_WIRED in str(exc.value)
    assert "LOCAL_RANK" in str(exc.value) and "rank 0" in str(exc.value)
    monkeypatch.setenv("WORLD_SIZE", "1")
    learner = Learner(cfg)
    assert learner.local_batch == cfg.learner.batch          # never sharded


def test_retention_never_deletes_the_checkpoint_it_just_wrote(tmp_path):
    """`historical_pool_size: 0` disables the pool, not the arena history."""
    cfg = load_config(None, [f"run_dir={tmp_path}/run", "net.width=32",
                             "net.blocks=1", "selfplay.historical_pool_size=0",
                             "learner.checkpoint_retention=true"])
    cfg.make_dirs()
    learner = Learner(cfg)
    for gen in range(4):
        open(os.path.join(cfg.checkpoints_dir, f"gen_{gen:04d}.pt"), "wb").close()
    path = learner.save_generation(4)
    assert os.path.exists(path)
    assert sorted(os.listdir(cfg.checkpoints_dir)) == ["gen_0004.pt"]
