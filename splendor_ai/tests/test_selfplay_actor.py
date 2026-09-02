"""The actor: PCR, recording, seat conventions, stuck/truncation, mixing."""

import queue
import random

import numpy as np
import pytest

from splendor_ai.rules import engine as E
from splendor_ai.selfplay import sample as S
from splendor_ai.selfplay.actor import Actor
from splendor_ai.selfplay.config import load_config
from splendor_ai.selfplay.replay import make_batch
from splendor_ai.model import SplendorNet, save_checkpoint


def _cfg(tmp_path, **overrides):
    base = [f"run_dir={tmp_path}/run", "net.width=32", "net.blocks=1",
            "selfplay.games_per_actor=4", "selfplay.win_threshold=5",
            # short games on purpose: a record only ships when its game ends,
            # so every slot must finish at least one game inside the wave budget
            "selfplay.max_plies=30", "selfplay.mixed_game_frac=0.0",
            "search_full.sims=8", "search_full.universes=1",
            "search_full.forced_playouts_k=0.0",
            "search_full.prune_policy_target=false",
            "search_fast.sims=4", "search_fast.universes=1",
            "inference.mode=inproc"]
    base += [f"{k}={v}" for k, v in overrides.items()]
    cfg = load_config(None, base)
    cfg.make_dirs()
    save_checkpoint(cfg.latest_weights, SplendorNet(cfg.net),
                    {"step": 0, "generation": 0})
    return cfg


def _run(cfg, waves=60, actor_id=0):
    out, stats = queue.Queue(), queue.Queue()
    actor = Actor(cfg, actor_id, out, stats)
    actor.run(max_waves=waves)
    payloads = []
    while not out.empty():
        payloads.append(out.get())
    records = [S.records_from_bytes(p["buf"]) for p in payloads if p.get("n")]
    return actor, (np.concatenate(records) if records else S.empty(0)), payloads


def test_actor_produces_valid_records(tmp_path):
    cfg = _cfg(tmp_path)
    actor, records, payloads = _run(cfg, waves=60)
    assert actor.moves > 0 and actor.sims > 0
    assert actor.games_finished > 0
    assert len(records) > 0
    S.check_records(records)
    assert sum(p["games"] for p in payloads) == actor.games_finished
    # C5 augmentation: five rotations of every recorded position
    assert len(records) % cfg.selfplay.augment_rotations == 0
    assert set(np.unique(records["rot"]).tolist()) <= set(range(5))


def test_pcr_only_records_full_searches(tmp_path):
    """With ``pcr_full_prob = 0`` nothing may be recorded; with 1.0 the number
    of recorded positions must track the number of net moves."""
    cfg = _cfg(tmp_path, **{"selfplay.pcr_full_prob": 0.0})
    actor, records, _ = _run(cfg, waves=60)
    assert actor.moves > 0
    assert len(records) == 0

    cfg = _cfg(tmp_path, **{"selfplay.pcr_full_prob": 1.0})
    actor, records, _ = _run(cfg, waves=60)
    assert len(records) > 0
    assert actor.records_made == len(records)


def test_recorded_value_targets_match_the_game_outcome(tmp_path):
    """Column 0 of the learner's ``z`` must be the acting seat's own result."""
    cfg = _cfg(tmp_path, **{"selfplay.pcr_full_prob": 1.0})
    _actor, records, _ = _run(cfg, waves=80)
    assert len(records) > 0
    batch = make_batch(records)
    for i in range(len(records)):
        seat = int(records["seat"][i])
        z_abs = np.asarray(records["z"][i], dtype=np.float32)
        assert abs(batch["z"][i, 0] - z_abs[seat]) < 1e-2
    # 2p INDIVIDUAL is zero sum: the pair of entries must be +-1 (or 0 on a tie)
    z_pairs = np.asarray(records["z"], dtype=np.float32)[:, :2]
    assert np.allclose(z_pairs.sum(axis=1), 0.0, atol=1e-2)
    assert set(np.unique(np.abs(z_pairs)).tolist()) <= {0.0, 1.0}


def test_truncation_downweights_the_value_target(tmp_path):
    cfg = _cfg(tmp_path, **{"selfplay.pcr_full_prob": 1.0,
                            "selfplay.max_plies": 6})
    actor, records, _ = _run(cfg, waves=40)
    assert actor.truncations > 0
    weights = np.unique(np.asarray(records["z_weight"], dtype=np.float32))
    assert weights.size == 1
    assert abs(float(weights[0]) - cfg.selfplay.truncation_z_weight) < 1e-2
    assert (np.asarray(records["plies"]) <= 6).all()


def test_mixed_games_only_record_current_net_seats(tmp_path):
    cfg = _cfg(tmp_path, **{"selfplay.pcr_full_prob": 1.0,
                            "selfplay.mixed_game_frac": 1.0,
                            "selfplay.opponent_weights": "{anchor: 1.0}"})
    actor, records, _ = _run(cfg, waves=60)
    assert len(records) > 0
    # every game has exactly one greedy seat here, and no record may carry it
    per_game = {}
    for i in range(len(records)):
        per_game.setdefault(int(records["game_id"][i]), set()).add(
            int(records["seat"][i]))
    assert all(len(seats) <= 1 for seats in per_game.values()), per_game


def test_actor_reports_throughput_and_diagnostics(tmp_path):
    cfg = _cfg(tmp_path, **{"selfplay.stats_every_s": 0.0,
                            "selfplay.pcr_full_prob": 1.0})
    out, stats = queue.Queue(), queue.Queue()
    actor = Actor(cfg, 0, out, stats)
    actor.run(max_waves=12)
    msgs = []
    while not stats.empty():
        msgs.append(stats.get())
    assert msgs
    last = msgs[-1]
    for key in ("sims_per_s", "moves_per_s", "games_per_s", "stuck_rate",
                "truncation_rate", "disagreement", "mode_plies"):
        assert key in last
    assert last["sims_per_s"] > 0
    assert 0.0 <= last["disagreement"] <= 1.0
