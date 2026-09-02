"""Card and bonus-tile tables for the custom Splendor variant.

Bit-exact mirror of the generator at the top of ``server/gameLogic.js``.  The
ids produced here (cards ``0..89``, tiles ``0..9``) are the ids used by the
wire protocol and by ``docs/REPLAY_FORMAT.md``, so they must never drift.

The Node source is::

    let nextId = 0;
    function addCycle(tier, points, template) {
      for (let i = 0; i < 5; i++) {
        const cost = [0, 0, 0, 0, 0];
        for (let j = 0; j < 5; j++) cost[j] = template[(j - i + 5) % 5];
        ALL_CARDS.push({ id: nextId++, tier, reward: i, points, cost });
      }
    }

``self_test()`` (also exposed as ``python -m splendor_ai.rules.cards``) diffs
the tables against a live dump from ``server/gameLogic.js``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import List, NamedTuple, Tuple

NUM_COLORS = 5          # 0..4 are the gem colours; index 5 is gold/wild
GOLD = 5
NUM_GEM_SLOTS = 6


class Card(NamedTuple):
    id: int
    tier: int                       # 1..3 (as in gameLogic.js, NOT 0-based)
    reward: int                     # 0..4 — the colour bonus the card grants
    points: int
    cost: Tuple[int, int, int, int, int]


class Tile(NamedTuple):
    id: int
    points: int                     # always 3 in this variant
    requirement: Tuple[int, int, int, int, int]


# ── card table ────────────────────────────────────────────────────────────

_CARD_CYCLES: List[Tuple[int, int, Tuple[int, ...]]] = [
    # TIER 1: 40 cards
    (1, 0, (1, 1, 0, 1, 1)),
    (1, 0, (1, 2, 1, 0, 1)),
    (1, 0, (0, 2, 1, 0, 2)),
    (1, 0, (3, 0, 0, 1, 0)),
    (1, 0, (0, 0, 2, 2, 0)),
    (1, 0, (0, 0, 0, 0, 3)),
    (1, 0, (0, 2, 1, 1, 1)),
    (1, 1, (0, 4, 0, 0, 0)),
    # TIER 2: 30 cards
    (2, 3, (6, 0, 0, 0, 0)),
    (2, 1, (0, 2, 0, 3, 3)),
    (2, 2, (0, 0, 0, 0, 5)),
    (2, 2, (0, 4, 0, 1, 3)),
    (2, 2, (0, 1, 4, 2, 0)),
    (2, 3, (0, 0, 0, 0, 6)),
    # TIER 3: 20 cards
    (3, 3, (0, 3, 3, 3, 5)),
    (3, 4, (0, 0, 0, 7, 0)),
    (3, 4, (0, 0, 0, 3, 6)),
    (3, 5, (0, 3, 0, 0, 7)),
]


def _build_cards() -> List[Card]:
    cards: List[Card] = []
    next_id = 0
    for tier, points, template in _CARD_CYCLES:
        for i in range(5):
            cost = tuple(template[(j - i + 5) % 5] for j in range(5))
            cards.append(Card(next_id, tier, i, points, cost))
            next_id += 1
    return cards


def _build_tiles() -> List[Tile]:
    tiles: List[Tile] = []
    tile_id = 0
    for i in range(5):
        req = [0, 0, 0, 0, 0]
        req[i] = 4
        req[(i + 1) % 5] = 4
        tiles.append(Tile(tile_id, 3, tuple(req)))
        tile_id += 1
    for i in range(5):
        req = [0, 0, 0, 0, 0]
        req[i] = 3
        req[(i + 2) % 5] = 3
        req[(i + 4) % 5] = 3
        tiles.append(Tile(tile_id, 3, tuple(req)))
        tile_id += 1
    return tiles


CARDS: Tuple[Card, ...] = tuple(_build_cards())
TILES: Tuple[Tile, ...] = tuple(_build_tiles())

NUM_CARDS = len(CARDS)          # 90
NUM_TILES = len(TILES)          # 10

# Flat lookup tables — the engine hot path indexes these by card id and never
# touches the NamedTuples (attribute access is measurably slower).
CARD_TIER: Tuple[int, ...] = tuple(c.tier for c in CARDS)            # 1..3
CARD_TIER0: Tuple[int, ...] = tuple(c.tier - 1 for c in CARDS)       # 0..2
CARD_REWARD: Tuple[int, ...] = tuple(c.reward for c in CARDS)
CARD_POINTS: Tuple[int, ...] = tuple(c.points for c in CARDS)
CARD_COST: Tuple[Tuple[int, ...], ...] = tuple(c.cost for c in CARDS)
#: Non-zero cost entries only, as ``((colour, amount), ...)`` — cards cost at
#: most four colours and often only one, so the affordability check iterates
#: ~2.4 entries instead of 5.
CARD_COST_NZ: Tuple[Tuple[Tuple[int, int], ...], ...] = tuple(
    tuple((i, v) for i, v in enumerate(c.cost) if v) for c in CARDS
)
TILE_REQ: Tuple[Tuple[int, ...], ...] = tuple(t.requirement for t in TILES)
TILE_POINTS: Tuple[int, ...] = tuple(t.points for t in TILES)

# Card ids grouped by tier, in id order — the pool each tier is shuffled from.
CARDS_BY_TIER: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(c.id for c in CARDS if c.tier == t) for t in (1, 2, 3)
)


# ── cross-language self test ──────────────────────────────────────────────

_NODE_DUMP = r"""
let g = null;
try { g = require(PATH); } catch (e) { g = null; }
let cards = g && g.ALL_CARDS;
let tiles = g && g.ALL_BONUS_TILES;
if (!cards || !tiles) {
  // Fallback: the export may not exist yet on this checkout.  Re-run the exact
  // generator from the top of gameLogic.js (kept byte-identical below).
  cards = []; tiles = [];
  let nextId = 0;
  const addCycle = (tier, points, template) => {
    for (let i = 0; i < 5; i++) {
      const cost = [0, 0, 0, 0, 0];
      for (let j = 0; j < 5; j++) cost[j] = template[(j - i + 5) % 5];
      cards.push({ id: nextId++, tier, reward: i, points, cost });
    }
  };
  const src = require('fs').readFileSync(PATH, 'utf8');
  const body = src.slice(0, src.indexOf('// ── Helpers ──'));
  for (const m of body.matchAll(/addCycle\((\d+),\s*(\d+),\s*\[([^\]]*)\]\)/g)) {
    addCycle(Number(m[1]), Number(m[2]), m[3].split(',').map(Number));
  }
  let tileId = 0;
  for (let i = 0; i < 5; i++) {
    const req = [0, 0, 0, 0, 0];
    req[i] = 4; req[(i + 1) % 5] = 4;
    tiles.push({ id: tileId++, points: 3, requirement: req });
  }
  for (let i = 0; i < 5; i++) {
    const req = [0, 0, 0, 0, 0];
    req[i] = 3; req[(i + 2) % 5] = 3; req[(i + 4) % 5] = 3;
    tiles.push({ id: tileId++, points: 3, requirement: req });
  }
}
process.stdout.write(JSON.stringify({ cards, tiles }));
"""


def default_game_logic_path() -> str:
    """Absolute path to ``server/gameLogic.js`` relative to this package."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(repo, "server", "gameLogic.js")


def node_dump(game_logic_path: str | None = None) -> dict:
    """Return ``{'cards': [...], 'tiles': [...]}`` straight out of Node."""
    path = game_logic_path or default_game_logic_path()
    script = "const PATH = " + json.dumps(path) + ";\n" + _NODE_DUMP
    out = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def python_dump() -> dict:
    """The same structure, from the Python tables."""
    return {
        "cards": [
            {"id": c.id, "tier": c.tier, "reward": c.reward,
             "points": c.points, "cost": list(c.cost)}
            for c in CARDS
        ],
        "tiles": [
            {"id": t.id, "points": t.points, "requirement": list(t.requirement)}
            for t in TILES
        ],
    }


def self_test(game_logic_path: str | None = None) -> List[str]:
    """Compare the Python tables against Node.  Returns a list of diffs."""
    js = node_dump(game_logic_path)
    py = python_dump()
    diffs: List[str] = []
    for key in ("cards", "tiles"):
        a, b = py[key], js[key]
        if len(a) != len(b):
            diffs.append(f"{key}: length {len(a)} (py) != {len(b)} (node)")
            continue
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                diffs.append(f"{key}[{i}]: py={x} node={y}")
    return diffs


def _main(argv: List[str]) -> int:
    if "--dump" in argv:
        json.dump(python_dump(), sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    diffs = self_test()
    if diffs:
        print(f"MISMATCH ({len(diffs)}):")
        for d in diffs[:40]:
            print("  " + d)
        return 1
    print(f"OK: {NUM_CARDS} cards and {NUM_TILES} tiles identical to "
          f"server/gameLogic.js")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))
