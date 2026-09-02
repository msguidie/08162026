"""Per-simulation determinization of the hidden information (PIMC).

What is actually hidden in this variant (``docs/AI_DESIGN.md`` §1.6 and
``rules/view.py``) is small and fully enumerable:

* the **order** of the three face-down decks (their *composition* is public
  deduction — everything that is not on the board, in a tableau or publicly
  reserved is still in a deck or in somebody's blind reserve);
* the identity of another seat's **deck-reserved** cards (at most three per
  seat).  Only the tier is public (``RESERVE_FROM_DECK`` announces it), which
  the engine records as ``reserved_public[slot] is False``.

Board-reserved cards are *not* secret: everybody watched the card leave the
market, so ``reserved_public[slot] is True`` and the card id stays put.

:func:`determinize` samples one consistent "universe" from a seat's
information set.  It is a pure function of ``(state, seat, rng)``, so a fixed
universe seed (see :func:`universe_rng`) reproduces the same universe for every
simulation that uses it — cestpasphoto's PC-PIMC seeding, which is what makes
an open-loop tree consistent across revisits of the same action path.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..rules.cards import CARDS_BY_TIER, CARD_TIER0
from ..rules.engine import GameState

__all__ = ["unseen_pool", "determinize", "universe_rng", "hidden_slots"]


def universe_rng(base_seed: int, universe_index: int) -> np.random.Generator:
    """The deterministic RNG of one determinization universe.

    ``universe_rng(s, k)`` always yields the same generator, so simulation
    ``i`` using universe ``i % K`` always sees the same hidden state.
    """
    return np.random.default_rng(
        [int(base_seed) & ((1 << 63) - 1), int(universe_index)])


def hidden_slots(state: GameState, seat: int) -> List[tuple]:
    """``(player_index, slot, tier0)`` for every reserve hidden from ``seat``."""
    out = []
    for i, p in enumerate(state.players):
        if i == seat:
            continue
        for slot, cid in enumerate(p.reserved):
            if not p.reserved_public[slot]:
                out.append((i, slot, CARD_TIER0[cid]))
    return out


def unseen_pool(state: GameState, seat: int) -> List[List[int]]:
    """Cards seat ``seat`` cannot place, split by tier (index 0..2).

    ``all cards − board − every tableau − every publicly known reserve``.
    "Publicly known" means the viewer's own reserves (it knows all three) plus
    other seats' ``reserved_public`` ones.  Other seats' **deck**-reserved
    cards deliberately stay in the pool: from ``seat``'s point of view they are
    still indistinguishable from the deck.

    The result is in ascending card-id order — a canonical order, so the
    sampling below depends only on ``rng``.
    """
    seen = set()
    for row in state.board:
        seen.update(row)
    for i, p in enumerate(state.players):
        seen.update(p.cards)
        if i == seat:
            seen.update(p.reserved)
        else:
            pub = p.reserved_public
            for slot, cid in enumerate(p.reserved):
                if pub[slot]:
                    seen.add(cid)
    return [[cid for cid in CARDS_BY_TIER[t] if cid not in seen]
            for t in range(3)]


def determinize(state: GameState, seat: int,
                rng: np.random.Generator) -> GameState:
    """One sample from ``seat``'s information set.

    * every OTHER seat's blind reserve is replaced by a uniformly random unseen
      card **of the announced tier**, drawn without replacement;
    * the remainder is shuffled into the three decks, respecting
      ``state.deck_counts`` exactly;
    * the acting seat's own reserves, the board, tokens, tableaus, tiles and
      all turn plumbing are untouched — the public view of the position is
      preserved (``tests/test_determinize.py``).

    Deterministic for a given ``rng`` (a ``numpy.random.Generator``).

    A resigned seat in INDIVIDUAL mode has its cards discarded by the engine,
    so those ids are unrecoverable from the state alone and re-enter the pool;
    the deck-count clamp below drops the surplus at random, which keeps every
    public count exact.
    """
    s = state.clone()
    pools = unseen_pool(state, seat)

    for i, p in enumerate(s.players):
        if i == seat:
            continue
        pub = p.reserved_public
        for slot, cid in enumerate(p.reserved):
            if pub[slot]:
                continue
            pool = pools[CARD_TIER0[cid]]
            if not pool:                                   # pragma: no cover
                raise ValueError(
                    f"determinize: no unseen tier-{CARD_TIER0[cid] + 1} card "
                    f"left for seat {i} slot {slot}")
            p.reserved[slot] = pool.pop(int(rng.integers(len(pool))))

    decks = []
    for t in range(3):
        pool = pools[t]
        need = state.deck_counts[t]
        if len(pool) < need:                               # pragma: no cover
            raise ValueError(
                f"determinize: tier {t + 1} needs {need} deck cards but only "
                f"{len(pool)} are unseen — inconsistent state")
        order = rng.permutation(len(pool))
        decks.append([pool[k] for k in order[:need]])
    s.decks = decks
    s.deck_counts = [len(decks[0]), len(decks[1]), len(decks[2])]
    return s
