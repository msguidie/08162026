"""Observation encoder (docs/AI_DESIGN.md §1.3) and the G1 throughput gate."""

import random
import time

import numpy as np
import pytest

from splendor_ai.rules import engine as E
from splendor_ai.rules import view as VIEW
from splendor_ai.rules.actions import RESERVE_BOARD_START, RESERVE_DECK_START
from splendor_ai.rules.cards import (CARDS, CARD_POINTS, CARD_REWARD,
                                     CARD_TIER0, NUM_CARDS)
from splendor_ai import encode as EN

MODES = [("INDIVIDUAL", 2, None), ("INDIVIDUAL", 3, None),
         ("INDIVIDUAL", 4, None), ("ONE_V_TWO", 3, None),
         ("TEAM", 4, "ADJACENT"), ("TEAM", 4, "OPPOSITE")]


def _play(state, rng, plies, resign_p=0.0):
    for _ in range(plies):
        if state.is_over():
            break
        actions = E.legal_actions(state)
        if not actions or rng.random() < resign_p:
            E.resign(state, state.current_player)
            continue
        E.apply(state, rng.choice(actions))
    return state


def _random_states(count, seed=1, max_plies=90, resign_p=0.003):
    """``(state, seat)`` pairs from random legal play in every mode."""
    rng = random.Random(seed)
    out = []
    i = 0
    while len(out) < count:
        mode, n, layout = MODES[i % len(MODES)]
        i += 1
        s = E.new_game(n, mode, layout, rng=rng)
        _play(s, rng, rng.randint(0, max_plies), resign_p)
        for seat in range(n):
            out.append((s, seat))
            if len(out) == count:
                break
    return out


def _with_blind_reserves(seed=3):
    """A 4p game where seats 1 and 2 hold deck-reserved (hidden) cards."""
    rng = random.Random(seed)
    s = E.new_game(4, rng=rng)
    while s.current_player != 1:
        E.apply(s, E.legal_actions(s)[0])
    E.apply(s, RESERVE_DECK_START)              # seat 1, tier 1, blind
    while s.current_player != 2:
        E.apply(s, E.legal_actions(s)[0])
    E.apply(s, RESERVE_DECK_START + 2)          # seat 2, tier 3, blind
    while s.current_player != 3:
        E.apply(s, E.legal_actions(s)[0])
    E.apply(s, RESERVE_BOARD_START)             # seat 3, face up (public)
    return s


# ── layout ────────────────────────────────────────────────────────────────

def test_layout_constants_add_up():
    assert EN.OBS_VERSION == 1
    assert EN.OBS_DIM == 860
    assert EN.BOARD_OFF == 0
    assert EN.OWN_RESERVED_OFF == 12 * EN.CARD_FEATURES == 276
    assert EN.OTHER_RESERVED_OFF == 276 + 3 * EN.CARD_FEATURES == 345
    assert EN.PLAYER_OFF == 345 + 9 * EN.OTHER_CARD_FEATURES == 570
    assert EN.TILE_OFF == 570 + 4 * EN.PLAYER_FEATURES == 682
    assert EN.DECK_OFF == 682 + 5 * EN.TILE_FEATURES == 772
    assert EN.GLOBAL_OFF == 772 + EN.DECK_FEATURES == 820
    assert EN.OBS_DIM == EN.GLOBAL_OFF + EN.GLOBAL_FEATURES
    assert (EN.CARD_FEATURES, EN.OTHER_CARD_FEATURES, EN.PLAYER_FEATURES,
            EN.TILE_FEATURES, EN.DECK_FEATURES, EN.GLOBAL_FEATURES) == \
        (23, 25, 28, 18, 48, 40)


def test_colour_groups_are_disjoint_and_in_range():
    seen = set()
    for base in EN.COLOUR_GROUP_BASES:
        block = set(range(base, base + 5))
        assert not (block & seen)
        assert base + 5 <= EN.OBS_DIM
        seen |= block


# ── value ranges ──────────────────────────────────────────────────────────

def test_every_feature_is_finite_and_in_range():
    for state, seat in _random_states(400, seed=11):
        obs = EN.encode(state, seat)
        assert obs.dtype == np.float32 and obs.shape == (EN.OBS_DIM,)
        assert np.isfinite(obs).all()
        assert obs.max() <= 1.0 and obs.min() >= -1.0


def test_a_reused_buffer_is_cleared_first():
    states = _random_states(60, seed=12)
    buf = np.full(EN.OBS_DIM, 7.0, dtype=np.float32)
    for state, seat in states:
        fresh = EN.encode(state, seat)
        again = EN.encode(state, seat, buf)
        assert again is buf
        assert np.array_equal(fresh, buf)
        buf[:] = -3.0


# ── the information set ───────────────────────────────────────────────────

def test_deck_order_never_reaches_the_observation():
    rng = random.Random(4)
    for state, seat in _random_states(120, seed=13):
        before = EN.encode(state, seat)
        shuffled = state.clone()
        for tier in range(3):
            rng.shuffle(shuffled.decks[tier])
        assert shuffled.deck_counts == state.deck_counts
        assert np.array_equal(EN.encode(shuffled, seat), before)


def test_another_seats_blind_reserve_is_invisible():
    state = _with_blind_reserves()
    for seat in (0, 3):
        before = EN.encode(state, seat)
        swapped = state.clone()
        changed = 0
        for i, p in enumerate(swapped.players):
            if i == seat:
                continue
            for slot, cid in enumerate(p.reserved):
                if p.reserved_public[slot]:
                    continue
                tier = CARD_TIER0[cid]
                deck = swapped.decks[tier]
                # swap the hidden card with another unseen card of its tier
                other = deck[0]
                deck[0] = cid
                p.reserved[slot] = other
                changed += 1
        assert changed == 2
        assert np.array_equal(EN.encode(swapped, seat), before)


def test_my_own_blind_reserve_is_visible_to_me():
    state = _with_blind_reserves()
    seat = 1
    before = EN.encode(state, seat)
    swapped = state.clone()
    me = swapped.players[seat]
    hidden = me.reserved[0]
    assert me.reserved_public[0] is False
    tier = CARD_TIER0[hidden]
    deck = swapped.decks[tier]
    me.reserved[0], deck[0] = deck[0], hidden
    assert not np.array_equal(EN.encode(swapped, seat), before)
    # ... and the other seats still cannot tell
    for other in (0, 2, 3):
        assert np.array_equal(EN.encode(swapped, other),
                              EN.encode(state, other))


def test_hidden_card_block_holds_only_tier_present_deck_reserved():
    state = _with_blind_reserves()
    seat = 0
    obs = EN.encode(state, seat)
    # seat 1 is relative block j = 1, its first reserved slot
    base = EN.OTHER_RESERVED_OFF + 0 * 3 * EN.OTHER_CARD_FEATURES
    block = obs[base:base + EN.OTHER_CARD_FEATURES]
    tier0 = CARD_TIER0[state.players[1].reserved[0]]
    expected = np.zeros(EN.OTHER_CARD_FEATURES, dtype=np.float32)
    expected[11 + tier0] = 1.0      # tier one-hot
    expected[22] = 1.0              # present
    expected[24] = 1.0              # deck_reserved
    assert np.array_equal(block, expected)

    # seat 3's face-up reserve is public: known, not deck_reserved, real cost
    base = EN.OTHER_RESERVED_OFF + 2 * 3 * EN.OTHER_CARD_FEATURES
    block = obs[base:base + EN.OTHER_CARD_FEATURES]
    card = state.players[3].reserved[0]
    assert block[23] == 1.0 and block[24] == 0.0 and block[22] == 1.0
    assert np.allclose(block[0:5], np.array(CARDS[card].cost) / 7.0)
    assert block[5 + CARD_REWARD[card]] == 1.0


def test_a_pending_choice_of_another_seat_is_not_shown():
    rng = random.Random(6)
    s = E.new_game(3, rng=rng)
    s.turn_action = "BUY"
    s.pending_tile_choice = list(s.tiles[:2])
    acting = s.current_player
    other = (acting + 1) % 3
    assert EN.encode(s, acting)[EN.GLOBAL_OFF + 24] == 1.0
    assert EN.encode(s, other)[EN.GLOBAL_OFF + 24] == 0.0
    assert VIEW.public_view(s, other)["pendingTileChoice"] is None


# ── block contents ────────────────────────────────────────────────────────

def test_board_slots_follow_the_action_indices():
    rng = random.Random(8)
    s = _play(E.new_game(4, rng=rng), rng, 25)
    obs = EN.encode(s, s.current_player)
    for tier in range(3):
        for slot in range(4):
            base = (tier * 4 + slot) * EN.CARD_FEATURES
            block = obs[base:base + EN.CARD_FEATURES]
            row = s.board[tier]
            if slot >= len(row):
                assert not block.any()
                continue
            card = row[slot]
            assert block[22] == 1.0
            assert np.allclose(block[0:5], np.array(CARDS[card].cost) / 7.0)
            assert block[5 + CARD_REWARD[card]] == 1.0
            assert block[10] == CARD_POINTS[card] / 5.0
            assert block[11 + CARD_TIER0[card]] == 1.0
            assert block[20] == float(E.can_afford(s.players[s.current_player],
                                                   card))


def test_card_shortfall_and_affordability():
    rng = random.Random(21)
    s = _play(E.new_game(2, rng=rng), rng, 40)
    seat = s.current_player
    me = s.players[seat]
    obs = EN.encode(s, seat)
    card = s.board[0][0]
    block = obs[0:EN.CARD_FEATURES]
    cost = CARDS[card].cost
    shortfall = [max(0, cost[c] - me.discount[c] - me.gems[c]) for c in range(5)]
    assert np.allclose(block[14:19], np.array(shortfall) / 7.0)
    assert block[19] == min(sum(shortfall) / 5.0, 1.0)
    assert block[20] == float(sum(shortfall) <= me.gems[5])


def test_player_blocks_are_seat_relative_and_padded():
    rng = random.Random(31)
    s = _play(E.new_game(3, "ONE_V_TWO", rng=rng), rng, 30)
    for seat in range(3):
        obs = EN.encode(s, seat)
        for j in range(4):
            base = EN.PLAYER_OFF + j * EN.PLAYER_FEATURES
            block = obs[base:base + EN.PLAYER_FEATURES]
            if j >= s.num_players:
                assert not block.any()
                continue
            p = s.players[(seat + j) % s.num_players]
            assert block[16] == 1.0                    # present
            assert block[17] == (1.0 if j == 0 else 0.0)
            assert block[20 + j] == 1.0                # seat-offset one-hot
            assert block[11] == min(p.score / 15.0, 1.0)
            assert np.allclose(block[6:11],
                               np.minimum(np.array(p.discount) / 7.0, 1.0))
            assert block[19] == (1.0 if p.team_id == 0 else 0.0)  # solo role
            assert block[25:28].sum() == 0.0           # documented padding


def test_tile_blocks_and_qualification():
    rng = random.Random(41)
    s = _play(E.new_game(4, rng=rng), rng, 60)
    seat = s.current_player
    me = s.players[seat]
    obs = EN.encode(s, seat)
    for i in range(5):
        base = EN.TILE_OFF + i * EN.TILE_FEATURES
        block = obs[base:base + EN.TILE_FEATURES]
        if i >= len(s.tiles):
            assert not block.any()
            continue
        tile = s.tiles[i]
        req = np.array(E.TILE_REQ[tile], dtype=np.float32)
        assert np.allclose(block[0:5], req / 4.0)
        gap = np.maximum(req - np.array(me.discount), 0.0)
        assert np.allclose(block[5:10], gap / 4.0)
        assert np.isclose(block[10], gap.sum() / 12.0)
        assert block[11] == 1.0
        assert block[12] == float(E.qualifies_for_tile(me, tile))
        assert block[16:18].sum() == 0.0


def test_deck_composition_counts_exactly_the_unseen_cards():
    rng = random.Random(51)
    s = _play(E.new_game(4, rng=rng), rng, 70)
    for seat in range(4):
        obs = EN.encode(s, seat)
        seen = set()
        for row in s.board:
            seen |= set(row)
        for i, p in enumerate(s.players):
            seen |= set(p.cards)
            if i == seat:
                seen |= set(p.reserved)
            else:
                seen |= {cid for slot, cid in enumerate(p.reserved)
                         if p.reserved_public[slot]}
        expect = np.zeros(45, dtype=np.float64)
        for cid in range(NUM_CARDS):
            if cid in seen:
                continue
            points = CARD_POINTS[cid]
            bucket = 0 if points == 0 else (1 if points <= 2 else 2)
            expect[CARD_TIER0[cid] * 15 + bucket * 5 + CARD_REWARD[cid]] += 1
        assert np.allclose(obs[EN.DECK_OFF:EN.DECK_OFF + 45], expect / 8.0)
        assert np.allclose(obs[EN.DECK_OFF + 45:EN.GLOBAL_OFF],
                           np.array(s.deck_counts) / np.array([36.0, 26.0, 16.0]))


def test_global_block_mode_seat_and_final_round():
    rng = random.Random(61)
    s = _play(E.new_game(4, "TEAM", "OPPOSITE", rng=rng), rng, 30)
    s.final_round_triggered_by = 2
    for seat in range(4):
        g = EN.encode(s, seat)[EN.GLOBAL_OFF:]
        assert g[6:9].tolist() == [0.0, 1.0, 0.0]        # TEAM
        assert g[9:11].tolist() == [0.0, 1.0]            # OPPOSITE
        assert g[11:14].tolist() == [0.0, 0.0, 1.0]      # 4 players
        assert g[15] == 1.0 and g[21] == 1.0             # final round, revocable
        assert g[16 + (2 - seat) % 4] == 1.0
        assert g[25 + seat] == 1.0
        assert g[29:].sum() == 0.0                       # documented padding


# ── batch path ────────────────────────────────────────────────────────────

def test_encode_batch_matches_encode_bitwise():
    pairs = _random_states(600, seed=71)
    states = [s for s, _ in pairs]
    seats = [q for _, q in pairs]
    stacked = np.stack([EN.encode(s, q) for s, q in pairs])
    assert np.array_equal(EN.encode_batch(states, seats), stacked)
    # ... including the short-batch path
    assert np.array_equal(EN.encode_batch(states[:3], seats[:3]), stacked[:3])
    out = np.full((len(states), EN.OBS_DIM), 5.0, dtype=np.float32)
    assert EN.encode_batch(states, seats, out) is out
    assert np.array_equal(out, stacked)


def test_encode_batch_rejects_bad_shapes():
    pairs = _random_states(10, seed=72)
    states = [s for s, _ in pairs]
    seats = [q for _, q in pairs]
    with pytest.raises(ValueError):
        EN.encode_batch(states, seats[:5])
    with pytest.raises(ValueError):
        EN.encode_batch(states, seats, np.zeros((3, EN.OBS_DIM), np.float32))


def test_finished_games_still_encode():
    rng = random.Random(81)
    for mode, n, layout in MODES:
        s = E.new_game(n, mode, layout, rng=rng)
        while not s.is_over():
            actions = E.legal_actions(s)
            if not actions:
                E.resign(s, s.current_player)
                continue
            E.apply(s, rng.choice(actions))
        for seat in range(n):
            obs = EN.encode(s, seat)
            assert np.isfinite(obs).all()
            assert obs[EN.GLOBAL_OFF + 20] == 0.0     # no plies left to run


def test_encoding_survives_the_record_round_trip():
    """The learner stores ``GameState.to_bytes()`` and re-encodes later, so a
    rebuilt state has to encode to the very same vector."""
    for state, seat in _random_states(300, seed=82):
        rebuilt = E.GameState.from_bytes(state.to_bytes())
        assert np.array_equal(EN.encode(rebuilt, seat), EN.encode(state, seat))


# ── throughput (gate G1) ──────────────────────────────────────────────────

#: The design target is 100k encodes/s/core; the assertion is far below it so
#: a busy box does not fail the build, but both numbers are printed.
MIN_SINGLE_PER_SECOND = 3_000
MIN_BATCH_PER_SECOND = 15_000


def test_encode_throughput(capsys):
    pairs = _random_states(512, seed=91)
    states = [s for s, _ in pairs]
    seats = [q for _, q in pairs]
    buf = np.zeros(EN.OBS_DIM, dtype=np.float32)
    for i in range(200):
        EN.encode(states[i], seats[i], buf)
    n = 4000
    t0 = time.perf_counter()
    for i in range(n):
        EN.encode(states[i % 512], seats[i % 512], buf)
    single = n / (time.perf_counter() - t0)

    out = np.zeros((512, EN.OBS_DIM), dtype=np.float32)
    EN.encode_batch(states, seats, out)
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        EN.encode_batch(states, seats, out)
    batch = reps * 512 / (time.perf_counter() - t0)

    with capsys.disabled():
        print(f"\n  encode: {single:,.0f}/s single, {batch:,.0f}/s batched "
              f"(B=512) — target 100,000/s/core")
    assert single > MIN_SINGLE_PER_SECOND
    assert batch > MIN_BATCH_PER_SECOND
