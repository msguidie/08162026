"""The 65-way discrete action space (see ``docs/PLAN.md`` §4.3).

Index table — fixed forever, do not renumber
============================================

======  =====================================================================
 index   meaning
======  =====================================================================
  0-9    TAKE 3 distinct colours.  All ``C(5,3)`` triples in lexicographic
         order: (0,1,2) (0,1,3) (0,1,4) (0,2,3) (0,2,4) (0,3,4) (1,2,3)
         (1,2,4) (1,3,4) (2,3,4)
 10-19   TAKE 2 distinct colours.  All ``C(5,2)`` pairs in lexicographic
         order: (0,1) (0,2) (0,3) (0,4) (1,2) (1,3) (1,4) (2,3) (2,4) (3,4)
 20-24   TAKE 1 gem of colour ``index - 20`` (a *forced short take*)
 25-29   TAKE 2 gems of the same colour ``index - 25``
 30-41   RESERVE a face-up board card: ``30 + tier0*4 + slot``
         (``tier0`` 0..2, ``slot`` = position in that board row, left→right)
 42-44   RESERVE from the top of deck ``index - 42`` (tier0 0..2)
 45-56   BUY a face-up board card: ``45 + tier0*4 + slot``
 57-59   BUY own reserved card in slot ``index - 57`` (0..2)
 60-64   CHOOSE a noble tile: ``state.tiles[index - 60]`` — legal only while a
         multi-noble choice is pending
======  =====================================================================

Resign and timeout are **not** part of the policy action space (the server has
no PASS/RESIGN game action; they arrive out of band).  They get the sentinel
indices :data:`ACTION_RESIGN` and :data:`ACTION_TIMEOUT` so that replay codes
can still be decoded into a single integer.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Tuple

GEM_NAMES = ("Indigo", "Jade", "Amber", "Rose", "Violet")

# ── layout constants ──────────────────────────────────────────────────────

TAKE3_START = 0
TAKE2_START = 10
TAKE1_START = 20
TAKE2SAME_START = 25
RESERVE_BOARD_START = 30
RESERVE_DECK_START = 42
BUY_BOARD_START = 45
BUY_RESERVED_START = 57
CHOOSE_TILE_START = 60

NUM_TAKE_ACTIONS = 30
NUM_ACTIONS = 65

MAX_BOARD_SLOTS = 4
MAX_RESERVED = 3
MAX_TILE_CHOICES = 5        # revealedTiles = n + 1 <= 5

ACTION_RESIGN = -1
ACTION_TIMEOUT = -2

# ── take patterns ─────────────────────────────────────────────────────────

TRIPLES: Tuple[Tuple[int, int, int], ...] = tuple(combinations(range(5), 3))
PAIRS: Tuple[Tuple[int, int], ...] = tuple(combinations(range(5), 2))

#: ``TAKE_PATTERNS[i]`` is the canonical (sorted) colour list of take action
#: ``i``.  The server validates a colour list *in the given order*, but the
#: validation is provably order-independent for every accepted multiset
#: (see ``tests/test_takes.py::test_take_order_independence``), so the sorted
#: order is a faithful canonical representative.
TAKE_PATTERNS: Tuple[Tuple[int, ...], ...] = (
    TRIPLES + PAIRS + tuple((c,) for c in range(5))
    + tuple((c, c) for c in range(5))
)
assert len(TAKE_PATTERNS) == NUM_TAKE_ACTIONS

#: multiset (sorted tuple) → take action index
TAKE_INDEX: Dict[Tuple[int, ...], int] = {
    p: i for i, p in enumerate(TAKE_PATTERNS)
}


def take_index(colors) -> int:
    """Map any colour ordering to its canonical take action index."""
    key = tuple(sorted(colors))
    try:
        return TAKE_INDEX[key]
    except KeyError:                                       # pragma: no cover
        raise ValueError(f"not a representable gem take: {list(colors)}") from None


def reserve_board_action(tier0: int, slot: int) -> int:
    return RESERVE_BOARD_START + tier0 * MAX_BOARD_SLOTS + slot


def reserve_deck_action(tier0: int) -> int:
    return RESERVE_DECK_START + tier0


def buy_board_action(tier0: int, slot: int) -> int:
    return BUY_BOARD_START + tier0 * MAX_BOARD_SLOTS + slot


def buy_reserved_action(slot: int) -> int:
    return BUY_RESERVED_START + slot


def choose_tile_action(tile_slot: int) -> int:
    return CHOOSE_TILE_START + tile_slot


# ── decoding ──────────────────────────────────────────────────────────────

KIND_TAKE = "TAKE"
KIND_RESERVE_BOARD = "RESERVE_BOARD"
KIND_RESERVE_DECK = "RESERVE_DECK"
KIND_BUY_BOARD = "BUY_BOARD"
KIND_BUY_RESERVED = "BUY_RESERVED"
KIND_CHOOSE_TILE = "CHOOSE_TILE"

#: ``ACTION_TABLE[i] = (kind, a, b)`` — a decoded form of every action index.
#: For takes ``a`` is the colour tuple; for board actions ``a`` is tier0 and
#: ``b`` the slot; for deck reserves ``a`` is tier0; for reserved buys and tile
#: choices ``a`` is the slot.
ACTION_TABLE: Tuple[Tuple[str, object, int], ...] = tuple(
    [(KIND_TAKE, TAKE_PATTERNS[i], -1) for i in range(NUM_TAKE_ACTIONS)]
    + [(KIND_RESERVE_BOARD, t, s) for t in range(3) for s in range(MAX_BOARD_SLOTS)]
    + [(KIND_RESERVE_DECK, t, -1) for t in range(3)]
    + [(KIND_BUY_BOARD, t, s) for t in range(3) for s in range(MAX_BOARD_SLOTS)]
    + [(KIND_BUY_RESERVED, s, -1) for s in range(MAX_RESERVED)]
    + [(KIND_CHOOSE_TILE, s, -1) for s in range(MAX_TILE_CHOICES)]
)
assert len(ACTION_TABLE) == NUM_ACTIONS

# Fast per-index decode arrays used by the engine hot path.
_KIND: Tuple[str, ...] = tuple(k for k, _a, _b in ACTION_TABLE)
_ARG_A: Tuple[object, ...] = tuple(a for _k, a, _b in ACTION_TABLE)
_ARG_B: Tuple[int, ...] = tuple(b for _k, _a, b in ACTION_TABLE)


def action_name(index: int) -> str:
    """Human readable label, for logs and mismatch reports."""
    if index == ACTION_RESIGN:
        return "RESIGN"
    if index == ACTION_TIMEOUT:
        return "TIMEOUT"
    kind, a, b = ACTION_TABLE[index]
    if kind == KIND_TAKE:
        return "TAKE(" + ",".join(GEM_NAMES[c] for c in a) + ")"
    if kind == KIND_RESERVE_BOARD:
        return f"RESERVE_BOARD(tier={a + 1},slot={b})"
    if kind == KIND_RESERVE_DECK:
        return f"RESERVE_DECK(tier={a + 1})"
    if kind == KIND_BUY_BOARD:
        return f"BUY_BOARD(tier={a + 1},slot={b})"
    if kind == KIND_BUY_RESERVED:
        return f"BUY_RESERVED(slot={a})"
    return f"CHOOSE_TILE(slot={a})"


ACTION_NAMES: Tuple[str, ...] = tuple(action_name(i) for i in range(NUM_ACTIONS))


def index_table_markdown() -> str:
    """The documented action index table, rendered as Markdown."""
    rows = [
        ("0-9", "take 3 distinct colours",
         " ".join("".join(str(c) for c in t) for t in TRIPLES)),
        ("10-19", "take 2 distinct colours",
         " ".join("".join(str(c) for c in p) for p in PAIRS)),
        ("20-24", "take 1 gem (forced short take)", "colour = index - 20"),
        ("25-29", "take 2 of the same colour", "colour = index - 25"),
        ("30-41", "reserve a face-up board card", "30 + tier0*4 + slot"),
        ("42-44", "reserve from deck top", "tier0 = index - 42"),
        ("45-56", "buy a face-up board card", "45 + tier0*4 + slot"),
        ("57-59", "buy own reserved card", "slot = index - 57"),
        ("60-64", "choose a noble tile", "state.tiles[index - 60]"),
    ]
    out = ["| index | action | encoding |", "| --- | --- | --- |"]
    out += [f"| {a} | {b} | `{c}` |" for a, b, c in rows]
    return "\n".join(out)


__all__ = [
    "GEM_NAMES", "NUM_ACTIONS", "NUM_TAKE_ACTIONS", "TAKE_PATTERNS",
    "TAKE_INDEX", "TRIPLES", "PAIRS", "ACTION_TABLE", "ACTION_NAMES",
    "ACTION_RESIGN", "ACTION_TIMEOUT", "TAKE3_START", "TAKE2_START",
    "TAKE1_START", "TAKE2SAME_START", "RESERVE_BOARD_START",
    "RESERVE_DECK_START", "BUY_BOARD_START", "BUY_RESERVED_START",
    "CHOOSE_TILE_START", "MAX_BOARD_SLOTS", "MAX_RESERVED",
    "MAX_TILE_CHOICES", "take_index", "action_name", "index_table_markdown",
    "reserve_board_action", "reserve_deck_action", "buy_board_action",
    "buy_reserved_action", "choose_tile_action",
    "KIND_TAKE", "KIND_RESERVE_BOARD", "KIND_RESERVE_DECK", "KIND_BUY_BOARD",
    "KIND_BUY_RESERVED", "KIND_CHOOSE_TILE",
]
