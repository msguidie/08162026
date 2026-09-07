"""Training records: packing, invariants and C5 colour augmentation."""

import random

import numpy as np
import pytest

from splendor_ai.rules import engine as E
from splendor_ai.selfplay import sample as S
from splendor_ai.symmetry import rotate_action, rotate_state


def _position(seed=0, plies=6, num_players=2, mode="INDIVIDUAL", layout=None):
    state = E.new_game(num_players, mode, layout, rng=random.Random(seed))
    rng = random.Random(seed + 1)
    for _ in range(plies):
        actions = E.legal_actions(state)
        if not actions or state.phase != E.PHASE_PLAYING:
            break
        E.apply(state, rng.choice(actions))
    return state


def _uniform_record(state, mode="ind2", game_id=1, ply=0):
    mask = E.legal_mask(state)
    legal = [i for i, v in enumerate(mask) if v]
    policy = np.zeros(65, dtype=np.float32)
    policy[legal] = 1.0 / len(legal)
    return S.make_record(state, state.current_player, policy, mask, mode,
                         game_id, ply), np.asarray(mask, dtype=bool), policy


def test_record_roundtrip_and_size():
    state = _position()
    rec, mask, policy = _uniform_record(state)
    assert S.RECORD_DTYPE.itemsize < 512
    raw = bytes(rec["state"][:rec["nbytes"]])
    assert E.GameState.from_bytes(raw).to_bytes() == state.to_bytes()
    assert (S.unpack_mask(rec["mask"]) == mask).all()
    dense = S.densify_policy(rec["policy_idx"], rec["policy_prob"], rec["policy_n"])
    assert np.allclose(dense, policy, atol=1e-3)


def test_state_bytes_slot_covers_every_mode():
    for players, mode, layout in ((2, "INDIVIDUAL", None), (4, "INDIVIDUAL", None),
                                  (3, "ONE_V_TWO", None), (4, "TEAM", "ADJACENT")):
        for seed in range(3):
            state = _position(seed, 40, players, mode, layout)
            assert len(state.to_bytes()) <= S.STATE_BYTES


def test_policy_invariants_are_enforced():
    state = _position()
    mask = np.asarray(E.legal_mask(state), dtype=bool)
    bad = np.zeros(65, dtype=np.float32)
    bad[np.flatnonzero(mask)[0]] = 0.5              # does not sum to 1
    with pytest.raises(ValueError):
        S.pack_policy(bad, mask)
    illegal = np.zeros(65, dtype=np.float32)
    illegal[np.flatnonzero(~mask)[0]] = 1.0         # mass on an illegal action
    with pytest.raises(ValueError):
        S.pack_policy(illegal, mask)


def test_topk_truncation_renormalises():
    policy = np.zeros(65, dtype=np.float32)
    policy[:64] = 1.0 / 64
    mask = np.ones(65, dtype=bool)
    idx, prob, n = S.pack_policy(policy, mask)
    assert n == S.POLICY_TOPK
    assert abs(float(np.asarray(prob, dtype=np.float64).sum()) - 1.0) < 1e-2


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_augment_matches_the_symmetry_group(k):
    state = _position(seed=5, plies=10)
    rec, mask, _policy = _uniform_record(state)
    out = S.augment(rec, k)
    rotated = E.GameState.from_bytes(bytes(out["state"][:out["nbytes"]]))
    assert rotated.to_bytes() == rotate_state(state, k).to_bytes()
    assert (S.unpack_mask(out["mask"])
            == np.asarray(E.legal_mask(rotated), dtype=bool)).all()
    assert out["seat"] == rec["seat"]
    assert (np.asarray(out["z"]) == np.asarray(rec["z"])).all()


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_augmented_policy_indices_play_the_same_move(k):
    """The mass on action ``a`` must land on the action of the rotated state
    that leads to the rotated successor."""
    state = _position(seed=9, plies=12)
    mask = E.legal_mask(state)
    legal = [i for i, v in enumerate(mask) if v]
    policy = np.zeros(65, dtype=np.float32)
    policy[legal[0]] = 1.0
    rec = S.make_record(state, state.current_player, policy, mask, "ind2", 1, 0)
    out = S.augment(rec, k)
    dense = S.densify_policy(out["policy_idx"], out["policy_prob"], out["policy_n"])
    moved = int(np.argmax(dense))
    assert moved == rotate_action(legal[0], k)

    direct = state.clone()
    E.apply(direct, legal[0])
    rotated = rotate_state(state, k)
    E.apply(rotated, moved)
    assert rotate_state(direct, k).to_bytes() == rotated.to_bytes()


def test_augment_many_and_check_records():
    state = _position(seed=11, plies=4)
    rec, _mask, _policy = _uniform_record(state)
    S.finish_game_records([rec], np.array([1.0, -1.0, 0, 0], dtype=np.float32),
                          1.0, [8, 3], [0, 0], 42)
    packed = S.augment_many([rec], 5)
    assert len(packed) == 5
    report = S.check_records(packed)
    assert report["mask_leak"] == 0.0 and report["empty_masks"] == 0.0
    assert (packed["plies"] == 42).all()
    # transport round trip
    assert (S.records_from_bytes(S.records_to_bytes(packed))["nbytes"]
            == packed["nbytes"]).all()


def test_check_records_rejects_padding_values():
    state = _position(seed=13, plies=4)
    rec, _mask, _policy = _uniform_record(state)
    rec["z"] = np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float16)   # 2p game
    with pytest.raises(AssertionError):
        S.check_records(np.array([rec], dtype=S.RECORD_DTYPE))
