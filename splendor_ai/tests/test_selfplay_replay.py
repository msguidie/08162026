"""Replay buffer: generational window, sampling, batch decoding, persistence."""

import random

import numpy as np
import pytest

from splendor_ai.encode import OBS_DIM
from splendor_ai.rules import engine as E
from splendor_ai.selfplay import sample as S
from splendor_ai.selfplay.replay import ReplayBuffer, make_batch, seat_relative_rows


def _game_records(seed=0, n_records=6, num_players=2, mode="ind2"):
    players, engine_mode, layout = {
        "ind2": (2, "INDIVIDUAL", None), "ind4": (4, "INDIVIDUAL", None),
        "ovt": (3, "ONE_V_TWO", None), "team_adj": (4, "TEAM", "ADJACENT"),
    }[mode]
    state = E.new_game(players, engine_mode, layout, rng=random.Random(seed))
    rng = random.Random(seed)
    records = []
    while len(records) < n_records and state.phase == E.PHASE_PLAYING:
        mask = E.legal_mask(state)
        legal = [i for i, v in enumerate(mask) if v]
        if not legal:
            break
        policy = np.zeros(65, dtype=np.float32)
        policy[legal] = 1.0 / len(legal)
        records.append(S.make_record(state, state.current_player, policy, mask,
                                     mode, seed, len(records),
                                     root_value=[0.2, -0.2, 0.0, 0.0]))
        E.apply(state, rng.choice(legal))
    z = np.zeros(4, dtype=np.float32)
    z[:players] = np.linspace(1.0, -1.0, players)
    S.finish_game_records(records, z, 1.0,
                          [p.score for p in state.players] + [0] * (4 - players),
                          [0, 0, 0, 0], len(records))
    return records, z, players


def test_window_trims_whole_generations_and_ramps():
    buf = ReplayBuffer(window_start=2, window_end=4, window_ramp_generations=4,
                       max_samples=10 ** 6)
    records, _z, _n = _game_records()
    assert buf.window_size(0) == 2
    assert buf.window_size(4) == 4
    for _ in range(8):
        buf.add(np.array(records, dtype=S.RECORD_DTYPE))
        buf.close_generation()
    assert len(buf.generations) == 4                        # ramped window
    assert buf.total_dropped > 0
    sizes = {len(block) for _gen, block in buf.generations}
    assert sizes == {len(records)}                          # never a partial gen


def test_max_samples_cap_drops_oldest():
    records, _z, _n = _game_records(n_records=10)
    buf = ReplayBuffer(window_start=10, window_end=10, max_samples=25)
    for _ in range(5):
        buf.add(np.array(records, dtype=S.RECORD_DTYPE))
        buf.close_generation()
    assert len(buf) <= 25 + len(records)


def test_batch_shapes_and_masking():
    records, _z, _n = _game_records(n_records=8)
    buf = ReplayBuffer(window_start=2, window_end=2)
    buf.add(np.array(records, dtype=S.RECORD_DTYPE))
    buf.close_generation()
    batch = buf.batch(16)
    assert batch["obs"].shape == (16, OBS_DIM)
    assert batch["mask"].shape == (16, 65) and batch["mask"].dtype == bool
    assert np.allclose(batch["policy_target"].sum(axis=1), 1.0, atol=1e-3)
    assert float((batch["policy_target"] * ~batch["mask"]).sum()) < 1e-5
    assert batch["mask"].any(axis=1).all()
    assert (batch["z_valid"][:, 2:] == 0).all()             # 2p game


def test_value_target_is_rotated_to_the_acting_seat():
    """The one sign convention the whole run depends on: column 0 of ``z`` is
    the acting seat's own outcome, in every mode."""
    for mode in ("ind2", "ind4", "ovt", "team_adj"):
        records, z_abs, players = _game_records(seed=3, n_records=8, mode=mode)
        batch = make_batch(np.array(records, dtype=S.RECORD_DTYPE))
        for i, rec in enumerate(records):
            seat = int(rec["seat"])
            assert abs(batch["z"][i, 0] - z_abs[seat]) < 1e-2, (mode, seat)
            for j in range(players):
                assert abs(batch["z"][i, j] - z_abs[(j + seat) % players]) < 1e-2
            assert (batch["z"][i, players:] == 0).all()


def test_value_blend_mixes_root_value():
    records, _z, _n = _game_records(n_records=4)
    plain = make_batch(np.array(records, dtype=S.RECORD_DTYPE), value_blend=0.0)
    blended = make_batch(np.array(records, dtype=S.RECORD_DTYPE), value_blend=0.5)
    seats = np.array([int(r["seat"]) for r in records])
    root = seat_relative_rows(
        np.array([np.asarray(r["root_value"], dtype=np.float32) for r in records]),
        seats, np.full(len(records), 2))
    assert np.allclose(blended["z"], 0.5 * plain["z"] + 0.5 * root, atol=1e-3)


def test_save_and_load_restores_window(tmp_path):
    records, _z, _n = _game_records(n_records=5)
    buf = ReplayBuffer(window_start=3, window_end=3)
    for _ in range(3):
        buf.add(np.array(records, dtype=S.RECORD_DTYPE))
        buf.close_generation()
    buf.add(np.array(records, dtype=S.RECORD_DTYPE))        # an open generation
    path = buf.save(str(tmp_path / "replay.npz"))
    restored = ReplayBuffer(window_start=3, window_end=3).load(path)
    assert len(restored) == len(buf)
    assert restored.generation == buf.generation
    assert len(restored.generations) == len(buf.generations)
    assert restored._pending_n == buf._pending_n
    batch = restored.batch(8)
    assert batch["obs"].shape[0] == 8


def test_sampling_covers_every_generation():
    buf = ReplayBuffer(window_start=4, window_end=4)
    for gen in range(3):
        records, _z, _n = _game_records(seed=gen, n_records=4)
        buf.add(np.array(records, dtype=S.RECORD_DTYPE))
        buf.close_generation()
    drawn = buf.sample(400)
    assert len(np.unique(drawn["game_id"])) == 3


def test_empty_buffer_raises():
    with pytest.raises(ValueError):
        ReplayBuffer().sample(4)
