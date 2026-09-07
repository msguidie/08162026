"""The actor: PCR, recording, seat conventions, stuck/truncation, mixing."""

import json
import os
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


# ── the curriculum is a function of RUN-GLOBAL progress ───────────────────

def _phased_cfg(tmp_path, **overrides):
    """A config whose phase 0 is 2p and whose phase 1 is 4p."""
    from splendor_ai.selfplay.config import PhaseConfig

    cfg = _cfg(tmp_path, **overrides)
    cfg.selfplay.phases = [
        PhaseConfig(until_games=1000, mixture={"ind2": 1.0}, sims_full=4),
        PhaseConfig(until_games=None, mixture={"ind4": 1.0}, sims_full=6),
    ]
    return cfg


def _fresh_actor(cfg, instance=0, games_offset=0, actor_id=0):
    return Actor(cfg, actor_id, queue.Queue(), queue.Queue(),
                 instance=instance, games_offset=games_offset)


def test_curriculum_phase_comes_from_run_global_games_not_the_actors_own(tmp_path):
    """A resumed actor used to restart the whole curriculum at phase 0.

    Its own counter starts at 0 every time the process is spawned, so deriving
    the phase from it put the node back in the warm-up phase on every link of
    the PBS chain and after every actor restart.
    """
    cfg = _phased_cfg(tmp_path)

    # A fresh run: no progress file, no offset -> phase 0.
    actor = _fresh_actor(cfg)
    assert actor.global_games() == 0
    assert actor._phase().mixture == {"ind2": 1.0}
    assert actor._sample_mode() == "ind2"

    # The trainer hands a restored count at spawn: phase 1 straight away.
    resumed = _fresh_actor(cfg, games_offset=50_000)
    assert resumed._phase().mixture == {"ind4": 1.0}
    assert resumed._sample_mode() == "ind4"
    assert resumed._search_cfg(full=True).sims == 6

    # ...and a running actor picks the boundary up from progress.json.
    with open(cfg.progress_path, "w") as fh:
        json.dump({"games_done": 4_000, "generation": 7, "instance": 3}, fh)
    assert actor._sync_progress(force=True) is True
    assert actor.global_games() >= 4_000
    assert actor._phase().mixture == {"ind4": 1.0}


def test_progress_is_never_read_backwards_or_fatal(tmp_path):
    cfg = _phased_cfg(tmp_path)
    actor = _fresh_actor(cfg, games_offset=9_000)
    # A stale progress.json (it is written on a timer) must not rewind us.
    with open(cfg.progress_path, "w") as fh:
        json.dump({"games_done": 12}, fh)
    actor._sync_progress(force=True)
    assert actor.global_games() >= 9_000
    # A half-written / empty file is not worth crashing an actor over.
    open(cfg.progress_path, "w").close()
    assert actor._sync_progress(force=True) is False
    assert actor.global_games() >= 9_000


# ── per-launch nonce: no replayed deals, no colliding game ids ────────────

def test_the_instance_nonce_changes_the_deals_and_the_game_ids(tmp_path):
    """Two launches of the same run must not play the same games again."""
    cfg = _cfg(tmp_path)

    def deal(instance):
        actor = _fresh_actor(cfg, instance=instance)
        for slot in actor.slots:
            actor._start_game(slot)
        return ([s.game_id for s in actor.slots],
                [bytes(s.state.to_bytes()) for s in actor.slots])

    ids_a, states_a = deal(1)
    ids_b, states_b = deal(2)
    assert len(set(ids_a)) == len(ids_a)                    # unique in a launch
    assert not (set(ids_a) & set(ids_b))                    # ...and across them
    assert states_a != states_b                             # different deals
    assert all(i > 0 for i in ids_a + ids_b)                # positive int64

    # Same instance, different actors: still disjoint ids.
    other_seat = _fresh_actor(cfg, instance=1, actor_id=3)
    for slot in other_seat.slots:
        other_seat._start_game(slot)
    assert not (set(ids_a) & {s.game_id for s in other_seat.slots})


def test_game_id_layout_packs_instance_actor_and_index():
    from splendor_ai.selfplay.actor import make_game_id

    assert make_game_id(0, 0, 0) == 0
    assert make_game_id(0, 0, 7) == 7
    assert make_game_id(0, 1, 0) == 1 << 32
    assert make_game_id(1, 0, 0) == 1 << 42
    assert make_game_id(2 ** 20 - 1, 2 ** 10 - 1, 2 ** 32 - 1) < 2 ** 63


# ── historical opponents: cached listing, LRU of loaded nets ──────────────

def _write_pool(cfg, generations):
    from splendor_ai.model import SplendorNet, save_checkpoint

    paths = []
    for gen in generations:
        path = os.path.join(cfg.checkpoints_dir, f"gen_{gen:04d}.pt")
        save_checkpoint(path, SplendorNet(cfg.net), {"step": gen,
                                                     "generation": gen})
        paths.append(path)
    return paths


def test_pool_listing_is_cached_and_loaded_nets_are_an_lru(tmp_path):
    cfg = _cfg(tmp_path, **{"selfplay.historical_pool_size": 4,
                            "selfplay.historical_cache": 2,
                            "selfplay.historical_pool_refresh_s": 3600.0})
    paths = _write_pool(cfg, range(4))
    actor = _fresh_actor(cfg)

    assert actor._historical_pool() == paths
    # Cached: a checkpoint pruned by the learner after the listing does not
    # change what this actor sees until the cache expires (it falls back to the
    # anchor if the load then fails, rather than dying).
    os.remove(paths[0])
    assert actor._historical_pool() == paths
    actor._pool_cache_t = -1e18                             # expire it
    assert actor._historical_pool() == paths[1:]

    # LRU over loaded nets, capped at historical_cache.
    a, b, c = paths[1], paths[2], paths[3]
    assert actor._historical_evaluator(a) is not None
    assert actor._historical_evaluator(b) is not None
    assert actor.historical_loads == 2
    actor._historical_evaluator(a)                          # touch a
    assert actor.historical_loads == 2                      # served from cache
    actor._historical_evaluator(c)                          # evicts b, not a
    assert set(actor._historical) == {a, c}
    assert actor.historical_loads == 3


def test_an_unreadable_checkpoint_drops_out_of_the_pool(tmp_path):
    cfg = _cfg(tmp_path, **{"selfplay.historical_pool_size": 4,
                            "selfplay.historical_cache": 2})
    paths = _write_pool(cfg, range(2))
    actor = _fresh_actor(cfg)
    actor._historical_pool()
    with open(paths[0], "wb") as fh:                        # truncate it
        fh.write(b"not a checkpoint")
    assert actor._historical_evaluator(paths[0]) is None
    assert paths[0] in actor._pool_failed
    assert paths[0] not in actor._pool_cache
    # ...and a seat that drew it falls back rather than killing the actor.
    assert actor._evaluator_for(paths[0]) is actor.evaluator


def test_historical_reuse_prefers_an_already_loaded_checkpoint(tmp_path):
    cfg = _cfg(tmp_path, **{"selfplay.historical_pool_size": 8,
                            "selfplay.historical_cache": 2,
                            "selfplay.historical_reuse_prob": 1.0})
    paths = _write_pool(cfg, range(8))
    actor = _fresh_actor(cfg)
    first = actor._pick_historical(actor._historical_pool())
    assert first in paths
    loads = actor.historical_loads
    for _ in range(10):
        assert actor._pick_historical(actor._historical_pool()) == first
    assert actor.historical_loads == loads                  # no reloading


# ── per-mode instrumentation the trainer aggregates (§1.8) ────────────────

def test_actor_counts_games_and_truncations_per_mode(tmp_path):
    cfg = _cfg(tmp_path, **{"selfplay.max_plies": 12})      # force truncations
    actor, _records, _payloads = _run(cfg, waves=40)
    assert sum(actor.mode_games.values()) == actor.games_finished
    assert set(actor.mode_games) <= set(actor.mode_plies)
    assert sum(actor.mode_truncations.values()) == actor.truncations
    actor._maybe_stats(force=True)
    msg = actor.stats_queue.get_nowait()
    assert msg["mode_games"] == actor.mode_games
    assert msg["mode_truncations"] == actor.mode_truncations
    assert msg["global_games"] == actor.global_games()
