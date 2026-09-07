"""C5 colour symmetry (docs/AI_DESIGN.md §1.4) and gate G1.

The property test walks random legal games in every mode and checks, for all
five rotations of every position reached:

* the card and tile tables are closed under the rotation (checked from the
  tables, not assumed from ``addCycle``),
* ``legal_mask`` permutes exactly by ``action_perm(k)``,
* applying the rotated action to the rotated state gives the rotated
  successor (state equality via ``GameState.to_bytes()``),
* ``terminal_values`` is invariant,
* the encoder is exactly equivariant: ``encode(rotate_state(s, k), seat) ==
  encode(s, seat)[feature_perm(k)]``.

``SPLENDOR_SYM_STATES`` sets the size of the default run;
``SPLENDOR_SYM_FULL=1`` enables the 100k-state gate run.
"""

import os
import random

import numpy as np
import pytest

from splendor_ai.rules import engine as E
from splendor_ai.rules.cards import (CARDS, TILES, NUM_COLORS, rotate_id,
                                     rotate_tile_id)
from splendor_ai import encode as EN
from splendor_ai import symmetry as SY
from splendor_ai import values as V

MODES = [("INDIVIDUAL", 2, None), ("INDIVIDUAL", 3, None),
         ("INDIVIDUAL", 4, None), ("ONE_V_TWO", 3, None),
         ("TEAM", 4, "ADJACENT"), ("TEAM", 4, "OPPOSITE")]

DEFAULT_STATES = int(os.environ.get("SPLENDOR_SYM_STATES", "3000"))
FULL_STATES = int(os.environ.get("SPLENDOR_SYM_FULL_STATES", "100000"))


# ── the tables ────────────────────────────────────────────────────────────

def test_card_and_tile_tables_are_closed():
    """The whole C5 group, verified card by card and tile by tile."""
    for k in range(NUM_COLORS):
        images = set()
        for c in CARDS:
            r = CARDS[rotate_id(c.id, k)]
            images.add(r.id)
            assert r.tier == c.tier and r.points == c.points
            assert r.reward == (c.reward + k) % NUM_COLORS
            assert r.cost == tuple(c.cost[(j - k) % NUM_COLORS]
                                   for j in range(NUM_COLORS))
        assert len(images) == len(CARDS)          # a bijection, nothing lost

        images = set()
        for t in TILES:
            r = TILES[rotate_tile_id(t.id, k)]
            images.add(r.id)
            assert r.points == t.points
            assert r.requirement == tuple(t.requirement[(j - k) % NUM_COLORS]
                                          for j in range(NUM_COLORS))
        assert len(images) == len(TILES)
    assert SY.closed_rotations() == (0, 1, 2, 3, 4)


def test_rotation_is_a_group_action_on_ids():
    for a in range(NUM_COLORS):
        for b in range(NUM_COLORS):
            for cid in range(len(CARDS)):
                assert (rotate_id(rotate_id(cid, a), b)
                        == rotate_id(cid, (a + b) % NUM_COLORS))
            for tid in range(len(TILES)):
                assert (rotate_tile_id(rotate_tile_id(tid, a), b)
                        == rotate_tile_id(tid, (a + b) % NUM_COLORS))


# ── the permutations ──────────────────────────────────────────────────────

def test_action_perm_only_moves_gem_takes():
    for k in range(NUM_COLORS):
        perm = SY.action_perm(k)
        assert perm.shape == (65,)
        assert sorted(perm.tolist()) == list(range(65))
        assert perm[30:].tolist() == list(range(30, 65))
        if k:
            assert not np.array_equal(perm[:30], np.arange(30))
    assert np.array_equal(SY.action_perm(0), np.arange(65))
    assert np.array_equal(SY.action_perm(5), SY.action_perm(0))


def test_perm_inverses_and_composition():
    for k in range(NUM_COLORS):
        inv = SY.inverse_perm(SY.action_perm(k))
        assert np.array_equal(inv, SY.action_perm(-k))
        assert np.array_equal(SY.action_perm(k)[inv], np.arange(65))
        finv = SY.inverse_perm(SY.feature_perm(k))
        assert np.array_equal(finv, SY.feature_perm(-k))
        for a in range(65):
            assert SY.rotate_action(a, k) == int(inv[a])
    for a in range(NUM_COLORS):
        for b in range(NUM_COLORS):
            composed = SY.action_perm(a)[SY.action_perm(b)]
            assert np.array_equal(composed, SY.action_perm(a + b))


def test_feature_perm_is_a_permutation_of_the_observation():
    for k in range(NUM_COLORS):
        perm = SY.feature_perm(k)
        assert perm.shape == (EN.OBS_DIM,)
        assert sorted(perm.tolist()) == list(range(EN.OBS_DIM))
    moved = np.count_nonzero(SY.feature_perm(1) != np.arange(EN.OBS_DIM))
    # every colour-major group of five moves under a non-trivial rotation
    assert moved == 5 * len(EN.COLOUR_GROUP_BASES)


# ── rotate_state itself ───────────────────────────────────────────────────

def test_rotate_state_is_a_group_action_on_states():
    rng = random.Random(17)
    s = E.new_game(4, "TEAM", "ADJACENT", rng=rng)
    for _ in range(40):
        actions = E.legal_actions(s)
        if not actions:
            break
        E.apply(s, rng.choice(actions))
    assert SY.rotate_state(s, 0).to_bytes() == s.to_bytes()
    for a in range(NUM_COLORS):
        for b in range(NUM_COLORS):
            lhs = SY.rotate_state(SY.rotate_state(s, a), b).to_bytes()
            rhs = SY.rotate_state(s, (a + b) % NUM_COLORS).to_bytes()
            assert lhs == rhs
    # the original must not be touched
    before = s.to_bytes()
    SY.rotate_state(s, 3)
    assert s.to_bytes() == before


def test_rotate_state_keeps_the_cached_discounts_consistent():
    rng = random.Random(5)
    s = E.new_game(3, rng=rng)
    for _ in range(60):
        actions = E.legal_actions(s)
        if not actions or s.is_over():
            break
        E.apply(s, rng.choice(actions))
    for k in range(NUM_COLORS):
        r = SY.rotate_state(s, k)
        for p in r.players:
            recomputed = [0] * 5
            for cid in p.cards:
                recomputed[CARDS[cid].reward] += 1
            assert p.discount == recomputed
            assert p.score == (sum(CARDS[c].points for c in p.cards)
                               + 3 * len(p.tiles))


def test_rotate_state_recolours_the_last_event():
    rng = random.Random(9)
    s = E.new_game(2, rng=rng)
    E.apply(s, 0)                                   # take colours (0, 1, 2)
    r = SY.rotate_state(s, 2)
    assert s.last_event["payload"]["selected"] == [0, 1, 2]
    assert r.last_event["payload"]["selected"] == [2, 3, 4]


# ── the state bytes the property run compares with ───────────────────────

def test_state_bytes_round_trip_every_field():
    """``GameState.to_bytes()`` is what "the same state" means below, so it has
    to be lossless and deterministic for every field of every mode."""
    rng = random.Random(23)
    sizes = []
    seen_pending = seen_result = seen_resigned = seen_blind = 0
    for mode, n, layout in MODES:
        for _ in range(12):
            s = E.new_game(n, mode, layout, rng=rng)
            while True:
                blob = s.to_bytes()
                sizes.append(len(blob))
                copy = E.GameState.from_bytes(blob)
                assert copy.to_bytes() == blob          # deterministic
                assert copy.decks == s.decks            # order included
                assert copy.board == s.board
                assert copy.deck_counts == s.deck_counts
                assert copy.gems == s.gems and copy.tiles == s.tiles
                assert copy.resigned == s.resigned
                assert copy.pending_tile_choice == s.pending_tile_choice
                assert copy.game_result == s.game_result
                assert copy.final_round_triggered_by == s.final_round_triggered_by
                assert copy.phase == s.phase and copy.turn_action == s.turn_action
                assert copy.current_player == s.current_player
                assert copy.round_start_player == s.round_start_player
                assert copy.turn_number == s.turn_number
                assert copy.mode == s.mode and copy.team_layout == s.team_layout
                assert copy.config == s.config and copy.teams == s.teams
                for a, b in zip(copy.players, s.players):
                    assert a.gems == b.gems and a.cards == b.cards
                    assert a.reserved == b.reserved
                    assert a.reserved_public == b.reserved_public
                    assert a.tiles == b.tiles and a.score == b.score
                    assert a.discount == b.discount and a.team_id == b.team_id
                assert (E.legal_mask(copy) == E.legal_mask(s))
                seen_pending += s.pending_tile_choice is not None
                seen_result += s.game_result is not None
                seen_resigned += bool(s.resigned)
                seen_blind += any(False in p.reserved_public
                                  for p in s.players)
                if s.is_over():
                    break
                actions = E.legal_actions(s)
                if not actions or rng.random() < 0.01:
                    E.resign(s, s.current_player)
                    continue
                E.apply(s, rng.choice(actions))
    assert seen_pending and seen_result and seen_resigned and seen_blind
    assert max(sizes) < 400                              # compact records


def test_state_bytes_keep_a_result_whose_reason_is_none():
    """A ``game_result`` with a ``None`` reason must survive the round trip.

    Format v1 encoded that reason as code 0 — the same byte that means "no
    game_result at all" — so ``from_bytes`` dropped the whole result.
    """
    s = E.new_game(4, "TEAM", "ADJACENT", rng=random.Random(5))
    s.phase = E.PHASE_GAME_OVER
    s.game_result = {"reason": None, "forfeitingTeamId": 1,
                     "winningTeamIds": [0]}
    blob = s.to_bytes()
    copy = E.GameState.from_bytes(blob)
    assert copy.game_result == s.game_result
    assert copy.to_bytes() == blob                   # and it is stable

    # A reason string the table does not know shares that code: the string is
    # not stored, but the result itself still comes back.
    s.game_result = {"reason": "ABANDONED"}
    assert E.GameState.from_bytes(s.to_bytes()).game_result == {"reason": None}

    # "no result at all" is still its own encoding.
    s.game_result = None
    assert E.GameState.from_bytes(s.to_bytes()).game_result is None


def test_state_bytes_reject_a_foreign_version():
    s = E.new_game(2, rng=random.Random(1))
    blob = bytearray(s.to_bytes())
    blob[0] = 99
    with pytest.raises(ValueError) as excinfo:
        E.GameState.from_bytes(bytes(blob))
    assert "99" in str(excinfo.value)


# ── the property run ──────────────────────────────────────────────────────

def _walk_states(rng, budget):
    """Random legal play through every mode, yielding each position seen."""
    i = 0
    while budget > 0:
        mode, n, layout = MODES[i % len(MODES)]
        i += 1
        s = E.new_game(n, mode, layout, rng=rng)
        while budget > 0:
            yield s
            budget -= 1
            if s.is_over():
                break
            actions = E.legal_actions(s)
            if not actions or rng.random() < 0.004:
                E.resign(s, s.current_player)       # stuck seat -> resign
                continue
            E.apply(s, rng.choice(actions))


def _check_rotations(num_states, seed, chunk=256):
    rng = random.Random(seed)
    perms = [SY.action_perm(k) for k in range(NUM_COLORS)]
    fperms = [SY.feature_perm(k) for k in range(NUM_COLORS)]
    checked = {"states": 0, "applied": 0, "terminal": 0, "encoded": 0}
    obs_states, obs_seats, obs_expect = [], [], []

    def flush():
        if not obs_states:
            return
        got = EN.encode_batch(obs_states, obs_seats)
        for row, expected in zip(got, obs_expect):
            assert np.array_equal(row, expected)
        checked["encoded"] += len(obs_states)
        obs_states.clear()
        obs_seats.clear()
        obs_expect.clear()

    for state in _walk_states(rng, num_states):
        checked["states"] += 1
        base_mask = np.array(E.legal_mask(state), dtype=np.bool_)
        legal = np.flatnonzero(base_mask)
        seat = rng.randrange(state.num_players)
        base_obs = EN.encode(state, seat)
        over = state.is_over()
        base_z = V.terminal_values(state) if over else None
        action = int(rng.choice(legal)) if len(legal) else -1
        if action >= 0:
            after = state.clone()
            E.apply(after, action)
            after_bytes = [SY.rotate_state(after, k).to_bytes()
                           for k in range(NUM_COLORS)]

        for k in range(NUM_COLORS):
            rotated = SY.rotate_state(state, k)
            assert rotated.to_bytes() == SY.rotate_state(state, k).to_bytes()

            mask = np.array(E.legal_mask(rotated), dtype=np.bool_)
            assert np.array_equal(mask, base_mask[perms[k]])
            assert E.is_stuck(rotated) == E.is_stuck(state)

            obs_states.append(rotated)
            obs_seats.append(seat)
            obs_expect.append(base_obs[fperms[k]])

            if over:
                assert np.array_equal(V.terminal_values(rotated), base_z)
                assert np.array_equal(V.standings_values(rotated),
                                      V.standings_values(state))
                checked["terminal"] += 1
            if action >= 0:
                moved = SY.rotate_action(action, k)
                assert mask[moved]
                nxt = rotated.clone()
                E.apply(nxt, moved)
                assert nxt.to_bytes() == after_bytes[k]
                checked["applied"] += 1
        if len(obs_states) >= chunk:
            flush()
    flush()
    return checked


def test_rotation_equivariance():
    stats = _check_rotations(DEFAULT_STATES, seed=2026)
    assert stats["states"] == DEFAULT_STATES
    assert stats["encoded"] == DEFAULT_STATES * NUM_COLORS
    assert stats["applied"] > 0 and stats["terminal"] > 0


@pytest.mark.skipif(os.environ.get("SPLENDOR_SYM_FULL") != "1",
                    reason="set SPLENDOR_SYM_FULL=1 for the 100k-state G1 run")
def test_rotation_equivariance_full(capsys):
    stats = _check_rotations(FULL_STATES, seed=7)
    with capsys.disabled():
        print(f"\n  C5 equivariance over {stats['states']:,} states: "
              f"{stats['encoded']:,} encodes, {stats['applied']:,} applies, "
              f"{stats['terminal']:,} terminal checks")
    assert stats["states"] == FULL_STATES
