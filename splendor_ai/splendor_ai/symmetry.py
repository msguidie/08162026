"""The C5 colour symmetry — ``docs/AI_DESIGN.md`` §1.4.

The five gem colours of this variant are interchangeable: ``addCycle`` emits
every card template in all five colour rotations and both bonus-tile families
are 5-cycles, so the tables are **closed** under ``c -> (c + k) % 5``
(``tests/test_symmetry.py::test_card_and_tile_tables_are_closed`` proves it
from the tables rather than trusting the generator).  Relabelling the colours
of a position therefore yields another legal position with the same value, and
the group acts on the action space and on the observation as a plain index
permutation.  Training augments every recorded position with its five
rotations and the deployment worker averages the net over them.

Conventions — all three permutations are **gathers**, ``x_rotated =
x_original[perm]``:

``legal_mask(rotate_state(s, k)) == legal_mask(s)[action_perm(k)]``
    and the same for policy targets and visit counts.
``encode(rotate_state(s, k), seat) == encode(s, seat)[feature_perm(k)]``
    exactly (both are ``float32`` and the encoder only moves the features
    around).
``apply(rotate_state(s, k), rotate_action(a, k))`` == ``rotate_state(apply(s, a), k)``
    note the *inverse*: ``action_perm(k)[i]`` answers "which action of the
    original state does action ``i`` of the rotated state correspond to",
    while :func:`rotate_action` maps an action of the original state forward
    into the rotated one.  ``rotate_action(a, k) == action_perm(-k)[a] ==
    inverse_perm(action_perm(k))[a]``.

Only the 30 gem-take actions move; reserve/buy/tile actions are addressed by
board slot and are invariant.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .encode import COLOUR_GROUP_BASES, OBS_DIM
from .rules.actions import (NUM_ACTIONS, NUM_TAKE_ACTIONS, TAKE_PATTERNS,
                            take_index)
from .rules.cards import (NUM_COLORS, ROTATED_CARD_ID, ROTATED_TILE_ID)
from .rules.engine import GameState

NUM_ROTATIONS = NUM_COLORS      # the group is C5

#: ``_GEM_SRC[k][c]`` = the colour whose count ends up in slot ``c`` after the
#: rotation, i.e. ``(c - k) % 5``.  Gold (index 5) never moves.
_GEM_SRC: Tuple[Tuple[int, ...], ...] = tuple(
    tuple((c - k) % NUM_COLORS for c in range(NUM_COLORS)) + (NUM_COLORS,)
    for k in range(NUM_ROTATIONS)
)

_ACTION_PERM = np.tile(np.arange(NUM_ACTIONS, dtype=np.int32),
                       (NUM_ROTATIONS, 1))
for _k in range(NUM_ROTATIONS):
    for _i in range(NUM_TAKE_ACTIONS):
        _ACTION_PERM[_k, _i] = take_index(
            tuple((c - _k) % NUM_COLORS for c in TAKE_PATTERNS[_i]))
_ACTION_PERM.flags.writeable = False

_FEATURE_PERM = np.tile(np.arange(OBS_DIM, dtype=np.int32),
                        (NUM_ROTATIONS, 1))
for _k in range(NUM_ROTATIONS):
    for _base in COLOUR_GROUP_BASES:
        for _c in range(NUM_COLORS):
            _FEATURE_PERM[_k, _base + _c] = _base + (_c - _k) % NUM_COLORS
_FEATURE_PERM.flags.writeable = False
del _k, _i, _base, _c


def action_perm(k: int) -> np.ndarray:
    """``int32[65]`` gather that maps quantities of the original state onto
    the rotated one: ``legal_mask(rotate_state(s, k)) ==
    legal_mask(s)[action_perm(k)]``."""
    return _ACTION_PERM[k % NUM_ROTATIONS]


def feature_perm(k: int) -> np.ndarray:
    """``int32[OBS_DIM]`` gather with ``encode(rotate_state(s, k), seat) ==
    encode(s, seat)[feature_perm(k)]``."""
    return _FEATURE_PERM[k % NUM_ROTATIONS]


def inverse_perm(perm: np.ndarray) -> np.ndarray:
    """The permutation ``q`` with ``x[perm][q] == x``."""
    out = np.empty_like(perm)
    out[perm] = np.arange(len(perm), dtype=perm.dtype)
    return out


def rotate_action(action: int, k: int) -> int:
    """The action of ``rotate_state(s, k)`` that plays ``action`` of ``s``."""
    return int(_ACTION_PERM[(-k) % NUM_ROTATIONS][action])


def rotate_event(event: Optional[Dict[str, Any]],
                 k: int) -> Optional[Dict[str, Any]]:
    """Recolour an ``apply`` result so that a rotated state keeps a consistent
    ``last_event``.  Unknown keys are passed through untouched."""
    if event is None or not k % NUM_ROTATIONS:
        return event
    k %= NUM_ROTATIONS
    cards = ROTATED_CARD_ID[k]
    tiles = ROTATED_TILE_ID[k]
    src = _GEM_SRC[k]
    out = dict(event)
    payload = event.get("payload")
    if payload is not None:
        p = dict(payload)
        if "selected" in p:
            p["selected"] = [(c + k) % NUM_COLORS for c in p["selected"]]
        if p.get("cardId") is not None:
            cid = cards[p["cardId"]]
            p["cardId"] = cid
            if "reward" in p:
                p["reward"] = (p["reward"] + k) % NUM_COLORS
        if p.get("tileId") is not None:
            p["tileId"] = tiles[p["tileId"]]
        if "gemsReturned" in p:
            g = p["gemsReturned"]
            p["gemsReturned"] = [g[i] for i in src]
        out["payload"] = p
    claimed = event.get("tileClaimed")
    if claimed is not None:
        claimed = dict(claimed)
        claimed["tileId"] = tiles[claimed["tileId"]]
        out["tileClaimed"] = claimed
    return out


def rotate_state(state: GameState, k: int) -> GameState:
    """A new :class:`GameState` with every colour relabelled ``c -> (c+k)%5``.

    Gems, per-player discounts (the engine's cached ``getDiscount``), card ids
    on the board, in the decks, in every tableau and every reserve, the noble
    tiles and a pending noble choice all move together; slot order, deck
    order, ``reserved_public``, scores, the turn plumbing and ``game_result``
    are untouched, so the rotated position is legal, has the same value and
    reaches the same successors under :func:`rotate_action`.
    """
    k %= NUM_ROTATIONS
    s = state.clone()
    if k == 0:
        return s
    cards = ROTATED_CARD_ID[k]
    tiles = ROTATED_TILE_ID[k]
    src = _GEM_SRC[k]
    g = state.gems
    s.gems = [g[src[0]], g[src[1]], g[src[2]], g[src[3]], g[src[4]], g[5]]
    board = state.board
    s.board = [[cards[c] for c in board[0]],
               [cards[c] for c in board[1]],
               [cards[c] for c in board[2]]]
    decks = state.decks
    s.decks = [[cards[c] for c in decks[0]],
               [cards[c] for c in decks[1]],
               [cards[c] for c in decks[2]]]
    s.tiles = [tiles[t] for t in state.tiles]
    for new, old in zip(s.players, state.players):
        og = old.gems
        new.gems = [og[src[0]], og[src[1]], og[src[2]], og[src[3]], og[src[4]],
                    og[5]]
        od = old.discount
        new.discount = [od[src[0]], od[src[1]], od[src[2]], od[src[3]],
                        od[src[4]]]
        new.cards = [cards[c] for c in old.cards]
        new.reserved = [cards[c] for c in old.reserved]
        new.tiles = [tiles[t] for t in old.tiles]
    if state.pending_tile_choice is not None:
        s.pending_tile_choice = [tiles[t] for t in state.pending_tile_choice]
    s.last_event = rotate_event(state.last_event, k)
    if state.tile_claimed is not None:
        claimed = dict(state.tile_claimed)
        claimed["tileId"] = tiles[claimed["tileId"]]
        s.tile_claimed = claimed
    return s


def rotate_obs(obs: np.ndarray, k: int) -> np.ndarray:
    """Apply :func:`feature_perm` to one observation or a batch of them."""
    return obs[..., _FEATURE_PERM[k % NUM_ROTATIONS]]


def rotate_policy(policy: np.ndarray, k: int) -> np.ndarray:
    """Apply :func:`action_perm` to a 65-wide policy/visit/mask vector."""
    return policy[..., _ACTION_PERM[k % NUM_ROTATIONS]]


def closed_rotations() -> Tuple[int, ...]:
    """The subgroup of ``0..4`` the card *and* tile tables are closed under.

    Checked against the tables themselves — a card must map to a card with the
    same tier and points whose cost is the rotated cost, and likewise for the
    tiles.  Expected to be the full ``(0, 1, 2, 3, 4)``; anything smaller makes
    augmentation and the root ensemble unsound and the tests fail loudly.
    """
    from .rules.cards import CARDS, TILES
    ok = []
    for k in range(NUM_ROTATIONS):
        good = True
        for c in CARDS:
            r = CARDS[ROTATED_CARD_ID[k][c.id]]
            if (r.tier != c.tier or r.points != c.points
                    or r.reward != (c.reward + k) % NUM_COLORS
                    or r.cost != tuple(c.cost[(j - k) % NUM_COLORS]
                                       for j in range(NUM_COLORS))):
                good = False
                break
        if good:
            for t in TILES:
                r = TILES[ROTATED_TILE_ID[k][t.id]]
                if (r.points != t.points
                        or r.requirement != tuple(t.requirement[(j - k) % NUM_COLORS]
                                                  for j in range(NUM_COLORS))):
                    good = False
                    break
        if good:
            ok.append(k)
    return tuple(ok)


__all__ = ["NUM_ROTATIONS", "rotate_state", "rotate_event", "action_perm",
           "feature_perm", "inverse_perm", "rotate_action", "rotate_obs",
           "rotate_policy", "closed_rotations"]
