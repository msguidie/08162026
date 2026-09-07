"""Helpers for building hand-made positions for the rule tests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from splendor_ai.rules.cards import CARDS, CARDS_BY_TIER, CARD_REWARD
from splendor_ai.tests.oracle import Position

#: tier-1 cards worth 0 points — seven per reward colour, ideal for building a
#: tableau with an exact discount vector and no score side effects.
T1_ZERO: List[int] = [c.id for c in CARDS if c.tier == 1 and c.points == 0]
#: tier-1 cards worth 1 point (ids 35..39), for nudging a score by exactly one.
T1_ONE: List[int] = [c.id for c in CARDS if c.tier == 1 and c.points == 1]


def discount_cards(discount: Sequence[int]) -> List[int]:
    """Card ids whose tableau gives exactly ``discount`` and zero points."""
    out: List[int] = []
    for color, count in enumerate(discount):
        pool = [cid for cid in T1_ZERO if CARD_REWARD[cid] == color]
        assert count <= len(pool), f"only {len(pool)} zero-point cards of colour {color}"
        out += pool[:count]
    return out


def position(n: int = 2, mode: str = "INDIVIDUAL",
             layout: Optional[str] = None,
             tiles: Sequence[int] = (),
             players: Optional[Sequence[Dict[str, Any]]] = None,
             gems: Optional[Sequence[int]] = None,
             current: int = 0,
             round_start: Optional[int] = None,
             board: Optional[Sequence[Sequence[int]]] = None,
             decks: Optional[Sequence[Sequence[int]]] = None,
             **extra: Any) -> Dict[str, Any]:
    """Build a probe_state.js request body.

    Cards not explicitly placed are dealt from the unused remainder so a
    tableau card can never also sit on the board.
    """
    plist: List[Dict[str, Any]] = [dict(p) for p in (players or [{} for _ in range(n)])]
    used = set()
    for p in plist:
        used |= set(p.get("cards", [])) | set(p.get("reserved", []))

    if board is None:
        board_rows: List[List[int]] = []
        for t in range(3):
            pool = [cid for cid in CARDS_BY_TIER[t] if cid not in used][:4]
            board_rows.append(pool)
            used |= set(pool)
    else:
        board_rows = [list(r) for r in board]
        for row in board_rows:
            used |= set(row)

    if decks is None:
        deck_rows = [[cid for cid in CARDS_BY_TIER[t] if cid not in used]
                     for t in range(3)]
    else:
        deck_rows = [list(r) for r in decks]

    state: Dict[str, Any] = {
        "board": board_rows, "decks": deck_rows, "tiles": list(tiles),
        "players": plist, "current": current,
        "roundStart": current if round_start is None else round_start,
    }
    if gems is not None:
        state["gems"] = list(gems)
    state.update(extra)
    return {"mode": mode, "layout": layout, "n": n, "state": state}


def run(spec: Dict[str, Any], ops: Sequence[Sequence[Any]]) -> Position:
    """Probe the position, mirror every op in Python and assert agreement."""
    return Position({**spec, "ops": [list(o) for o in ops]}).run_all()


__all__ = ["T1_ZERO", "T1_ONE", "discount_cards", "position", "run"]
