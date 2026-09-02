"""Determinization invariants (``docs/AI_DESIGN.md`` §1.6).

The contract: a determinized universe must be (a) reproducible from its RNG,
(b) indistinguishable from the real position for every seat that is allowed to
look, and (c) a legal position — every card exactly once, deck counts exact.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest

from splendor_ai.rules import engine as E, view as V
from splendor_ai.rules.cards import CARD_TIER0, NUM_CARDS
from splendor_ai.search.determinize import (
    determinize, hidden_slots, universe_rng, unseen_pool,
)

RESERVE_DECK = 42


def _game_with_hidden_reserves(n=4, mode="INDIVIDUAL", layout=None, seed=5):
    """Every seat blind-reserves once, so each has one card only it can see."""
    s = E.new_game(n, mode, layout, rng=random.Random(seed))
    for i in range(n):
        a = RESERVE_DECK + (i % 3)
        assert E.legal_mask(s)[a]
        E.apply(s, a)
    return s


def _all_placed(state):
    """Every card id the state accounts for, with duplicates kept."""
    out = []
    for row in state.board:
        out += list(row)
    for deck in state.decks:
        out += list(deck)
    for p in state.players:
        out += list(p.cards) + list(p.reserved)
    return out


def _public_facts(view):
    """A seat's view with everything it is NOT allowed to know removed.

    ``public_view`` shows a seat its own reserved cards in full, so comparing
    raw views across a determinization would compare hidden information.  This
    keeps only what is public to *everyone*: a non-public reserve collapses to
    its tier.
    """
    v = copy.deepcopy(view)
    for p in v["players"]:
        for slot, r in enumerate(p["reserved"]):
            if not r.get("public"):
                p["reserved"][slot] = {"tier": r["tier"], "public": False}
    v.pop("affordable", None)          # depends on the viewer's own reserves
    return v


# ── the unseen pool ───────────────────────────────────────────────────────

def test_unseen_pool_is_the_complement_of_what_the_seat_can_place():
    s = _game_with_hidden_reserves()
    for seat in range(s.num_players):
        pool = unseen_pool(s, seat)
        flat = [c for tier in pool for c in tier]
        assert len(flat) == len(set(flat))

        placed = set()
        for row in s.board:
            placed |= set(row)
        for i, p in enumerate(s.players):
            placed |= set(p.cards)
            for slot, cid in enumerate(p.reserved):
                if i == seat or p.reserved_public[slot]:
                    placed.add(cid)
        assert set(flat) == set(range(NUM_CARDS)) - placed
        for tier in range(3):
            assert all(CARD_TIER0[c] == tier for c in pool[tier])


def test_unseen_pool_keeps_other_seats_deck_reserves_and_drops_my_own():
    s = _game_with_hidden_reserves()
    seat = 0
    pool = {c for tier in unseen_pool(s, seat) for c in tier}
    for cid in s.players[seat].reserved:
        assert cid not in pool                       # I know my own cards
    for i in range(1, s.num_players):
        p = s.players[i]
        for slot, cid in enumerate(p.reserved):
            if p.reserved_public[slot]:
                assert cid not in pool               # everybody saw it leave
            else:
                assert cid in pool                   # only the tier is public


def test_unseen_pool_size_matches_decks_plus_hidden_slots():
    s = _game_with_hidden_reserves()
    for seat in range(s.num_players):
        pool = unseen_pool(s, seat)
        hidden = hidden_slots(s, seat)
        for tier in range(3):
            n_hidden = sum(1 for _, _, t in hidden if t == tier)
            assert len(pool[tier]) == s.deck_counts[tier] + n_hidden


# ── determinize ───────────────────────────────────────────────────────────

def test_determinize_is_deterministic_given_the_rng():
    s = _game_with_hidden_reserves()
    a = determinize(s, 0, universe_rng(1234, 3))
    b = determinize(s, 0, universe_rng(1234, 3))
    assert a.decks == b.decks
    assert [p.reserved for p in a.players] == [p.reserved for p in b.players]
    c = determinize(s, 0, universe_rng(1234, 4))
    assert c.decks != a.decks                        # a different universe


def test_universe_rng_universes_differ_but_repeat_exactly():
    s = _game_with_hidden_reserves()
    seen = {}
    for u in range(6):
        for _ in range(2):
            d = determinize(s, 0, universe_rng(99, u))
            key = (tuple(tuple(x) for x in d.decks),
                   tuple(tuple(p.reserved) for p in d.players))
            seen.setdefault(u, key)
            assert seen[u] == key
    assert len(set(seen.values())) == 6


def test_determinize_preserves_the_acting_seats_whole_view():
    s = _game_with_hidden_reserves()
    for seat in range(s.num_players):
        d = determinize(s, seat, universe_rng(7, seat))
        assert V.public_view(d, seat) == V.public_view(s, seat)


def test_determinize_preserves_every_seats_public_facts():
    s = _game_with_hidden_reserves()
    d = determinize(s, 1, universe_rng(11, 0))
    for viewer in range(s.num_players):
        assert (_public_facts(V.public_view(d, viewer))
                == _public_facts(V.public_view(s, viewer)))


def test_determinize_keeps_my_reserves_and_replaces_only_hidden_ones():
    s = _game_with_hidden_reserves()
    seat = 2
    d = determinize(s, seat, universe_rng(3, 1))
    assert d.players[seat].reserved == s.players[seat].reserved
    changed = 0
    for i, (p, q) in enumerate(zip(s.players, d.players)):
        assert p.reserved_public == q.reserved_public
        for slot, (before, after) in enumerate(zip(p.reserved, q.reserved)):
            if i == seat or p.reserved_public[slot]:
                assert before == after
            else:
                assert CARD_TIER0[before] == CARD_TIER0[after]   # tier is public
                changed += before != after
    assert changed > 0


def test_determinize_is_a_legal_position():
    s = _game_with_hidden_reserves()
    for u in range(6):
        d = determinize(s, 0, universe_rng(21, u))
        placed = _all_placed(d)
        assert len(placed) == len(set(placed)) == NUM_CARDS
        assert d.deck_counts == s.deck_counts
        assert [len(x) for x in d.decks] == s.deck_counts
        for tier in range(3):
            assert all(CARD_TIER0[c] == tier for c in d.decks[tier])
        # everything else is untouched
        assert d.board == s.board and d.gems == s.gems and d.tiles == s.tiles
        assert d.current_player == s.current_player
        assert [p.cards for p in d.players] == [p.cards for p in s.players]


def test_determinize_samples_the_whole_pool():
    """Every unseen tier-1 card must be reachable for a hidden slot."""
    s = E.new_game(2, rng=random.Random(4))
    E.apply(s, RESERVE_DECK)                      # seat 0 blind-reserves tier 1
    seat = 1
    pool = set(unseen_pool(s, seat)[0])
    drawn = set()
    rng = np.random.default_rng(0)
    for _ in range(400):
        d = determinize(s, seat, rng)
        drawn.add(d.players[0].reserved[0])
    assert drawn <= pool
    assert len(drawn) > 0.5 * len(pool)


def test_determinize_after_a_resign_discards_the_surplus():
    """INDIVIDUAL resign discards a tableau, so the pool exceeds the decks."""
    s = E.new_game(3, rng=random.Random(9))
    for _ in range(12):
        E.apply(s, E.legal_actions(s)[0])
    E.resign(s, 2)
    assert s.phase == E.PHASE_PLAYING
    pool = unseen_pool(s, 0)
    assert sum(len(t) for t in pool) >= sum(s.deck_counts)
    d = determinize(s, 0, universe_rng(5, 0))
    assert [len(x) for x in d.decks] == s.deck_counts
    placed = _all_placed(d)
    assert len(placed) == len(set(placed))


@pytest.mark.parametrize("mode,n,layout", [
    ("INDIVIDUAL", 2, None), ("INDIVIDUAL", 3, None), ("INDIVIDUAL", 4, None),
    ("ONE_V_TWO", 3, None), ("TEAM", 4, "ADJACENT"), ("TEAM", 4, "OPPOSITE"),
])
def test_determinize_over_random_play(mode, n, layout):
    """Fuzz: every mode, mid-game states, all seats."""
    rng = np.random.default_rng(hash((mode, n, layout)) % (1 << 31))
    s = E.new_game(n, mode, layout, rng=random.Random(17))
    for ply in range(40):
        if s.phase != E.PHASE_PLAYING or E.is_stuck(s):
            break
        legal = E.legal_actions(s)
        E.apply(s, legal[int(rng.integers(len(legal)))])
        seat = s.current_player
        d = determinize(s, seat, universe_rng(31, ply % 6))
        assert V.public_view(d, seat) == V.public_view(s, seat)
        assert [len(x) for x in d.decks] == s.deck_counts
        placed = _all_placed(d)
        assert len(placed) == len(set(placed))
        assert E.legal_mask(d) == E.legal_mask(s)     # same choices for me
