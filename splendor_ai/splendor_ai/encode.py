"""Observation encoder — the binding layout of ``docs/AI_DESIGN.md`` §1.3.

``encode(state, seat)`` turns a :class:`~splendor_ai.rules.engine.GameState`
into a fixed ``float32`` vector of :data:`OBS_DIM` features, **reading only the
information set of** ``seat`` (``rules/view.public_view``): never
``state.decks`` (only ``deck_counts``), never the identity of another seat's
deck-reserved card (only its tier), never a pending noble choice that belongs
to another seat.  Cards are *content addressed* — cost / reward / points /
tier and the derived affordability features — so there is no card-id
embedding and the C5 colour symmetry of §1.4 acts on the observation as a
plain index permutation (:func:`splendor_ai.symmetry.feature_perm`).

Everything is seat relative: player block ``j`` is seat ``(seat + j) % n`` and
blocks ``j >= n`` are all zero with ``present = 0``.  Every feature is finite
and lies in ``[-1, 1]``; ratios that could exceed the range are clipped (the
clipped cases are noted below).

Layout
======

======  ====  ==========================================================
offset  size  block
======  ====  ==========================================================
     0   276  12 board card slots x 23  (slot = tier0*4 + slot, i.e. the
              order of actions 30-41 / 45-56)
   276    69  my 3 reserved card slots x 23
   345   225  the 3 other seats x 3 reserved slots x 25
   570   112  4 player blocks x 28
   682    90  5 noble-tile slots x 18 (slot = the order of actions 60-64)
   772    48  public deck composition
   820    40  global block
======  ====  ==========================================================
                                                     total = 860 = OBS_DIM

Card block (23) — all shortfalls are computed against **my** discounts and
tokens, so the same 23 features describe a board card, one of my reserves and
a publicly reserved card of an opponent:

======  ==========================================================
 index   feature
======  ==========================================================
   0-4   cost / 7                                    (colour group)
   5-9   reward one-hot                              (colour group)
    10   points / 5
 11-13   tier one-hot
 14-18   shortfall after my discounts and tokens / 7 (colour group)
    19   gold needed / 5                          (clipped to 1)
    20   affordable now
    21   turns_to_buy = max(ceil(sum(shortfall)/3), max(shortfall)) / 6
                                                   (clipped to 1)
    22   present
======  ==========================================================

The other seats' 25-wide blocks add ``23 = known`` and ``24 = deck_reserved``.
A card another seat took blind off a deck is unknown: its block is zero except
for the tier one-hot, ``present`` and ``deck_reserved``.

Player block (28), block ``j`` = seat ``(seat + j) % n``:

======  ==========================================================
   0-5   gems / supply max (0-4 colour group, 5 = gold / 5)
  6-10   discount / 7                                (colour group)
    11   score / 15                               (clipped to 1)
    12   cards / 20                               (clipped to 1)
    13   reserved / 3
    14   tiles / 3                                (clipped to 1)
    15   resigned
    16   present
    17   is_self
    18   is_teammate
    19   is_solo_role (the ONE_V_TWO solo seat)
 20-23   seat-offset one-hot (j)
    24   excess over that seat's own side threshold / 15  (clipped)
 25-27   reserved for future — always zero
======  ==========================================================

Tile block (18), slot ``i`` = ``state.tiles[i]``:

======  ==========================================================
   0-4   requirement / 4                             (colour group)
  5-9    my per-colour card shortfall / 4            (colour group)
    10   my total shortfall / 12
    11   present
    12   qualifies now
 13-15   per-other-seat total shortfall / 12 (seat-relative j = 1..3)
 16-17   reserved for future — always zero
======  ==========================================================

Public deck composition (48): ``tier*15 + bucket*5 + colour`` holds the number
of **unseen** cards of that tier whose reward is ``colour`` and whose points
fall in bucket ``{0, 1-2, 3+}``, divided by 8; the last three entries are
``deck_counts[t]`` over the initial deck size (36 / 26 / 16).  "Unseen" = not
on the board, not in any tableau, not publicly reserved and not one of my own
reserves — another seat's deck-reserved card stays in the pool, exactly as the
information set requires.

Global block (40):

======  ==========================================================
   0-5   supply / max (0-4 colour group, 5 = gold / 5)
   6-8   mode one-hot (INDIVIDUAL, TEAM, ONE_V_TWO)
  9-10   team layout one-hot (ADJACENT, OPPOSITE)
 11-13   num_players one-hot (2, 3, 4)
    14   turn_number / 100                        (clipped to 1)
    15   final-round flag
 16-19   final-round triggered-by seat-offset one-hot
    20   plies until the turn returns to the round leader / 4
    21   revocable final round (TEAM)
    22   my side's threshold progress
    23   the other side's threshold progress
    24   pending-tile-choice flag (only ever set for the acting seat)
 25-28   my absolute seat one-hot
 29-39   padding — always zero
======  ==========================================================

Deviations from §1.3, all additive padding so that the documented block sizes
hold exactly: the player block lists 25 features and is 28 wide (3 zeros), the
tile block lists 17 and is 18 wide (2 zeros), the global block lists 29 and is
40 wide (11 zeros).  "per-other-seat min total shortfall" is implemented as
the per-other-seat total shortfall in seat-relative order.
"""

from __future__ import annotations

from array import array
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .rules.actions import MAX_BOARD_SLOTS, MAX_RESERVED, MAX_TILE_CHOICES
from .rules.cards import (CARDS, CARD_COST, CARD_POINTS, CARD_REWARD,
                          CARD_TIER0, NUM_CARDS, NUM_TILES, TILE_REQ)
from .rules.engine import (GameState, MODE_INDIVIDUAL, MODE_ONE_V_TWO,
                           MODE_TEAM, PHASE_PLAYING, get_next_active_player)

OBS_VERSION = 1

# ── block geometry ────────────────────────────────────────────────────────

MAX_SEATS = 4
NUM_COLORS = 5
BOARD_SLOTS = 3 * MAX_BOARD_SLOTS               # 12
OTHER_SEATS = MAX_SEATS - 1                     # 3

CARD_FEATURES = 23
OTHER_CARD_FEATURES = 25                        # + known + deck_reserved
PLAYER_FEATURES = 28
TILE_FEATURES = 18
DECK_FEATURES = 48
GLOBAL_FEATURES = 40

BOARD_OFF = 0
OWN_RESERVED_OFF = BOARD_OFF + BOARD_SLOTS * CARD_FEATURES              # 276
OTHER_RESERVED_OFF = OWN_RESERVED_OFF + MAX_RESERVED * CARD_FEATURES    # 345
PLAYER_OFF = (OTHER_RESERVED_OFF
              + OTHER_SEATS * MAX_RESERVED * OTHER_CARD_FEATURES)       # 570
TILE_OFF = PLAYER_OFF + MAX_SEATS * PLAYER_FEATURES                     # 682
DECK_OFF = TILE_OFF + MAX_TILE_CHOICES * TILE_FEATURES                  # 772
GLOBAL_OFF = DECK_OFF + DECK_FEATURES                                   # 820
OBS_DIM = GLOBAL_OFF + GLOBAL_FEATURES                                  # 860

#: Card slots evaluated in one shot: 12 board + 3 own reserved + 9 opponent.
_CARD_SLOTS = BOARD_SLOTS + MAX_RESERVED + OTHER_SEATS * MAX_RESERVED   # 24
_OWN_SLOT0 = BOARD_SLOTS
_OTHER_SLOT0 = BOARD_SLOTS + MAX_RESERVED

#: Sentinel rows appended to the card tables: ``EMPTY_CARD`` for an empty slot
#: and ``HIDDEN_CARD0 + tier0`` for a card another seat took blind off a deck
#: (all we may know is its tier).
EMPTY_CARD = NUM_CARDS                      # 90
HIDDEN_CARD0 = NUM_CARDS + 1                # 91..93
EMPTY_TILE = NUM_TILES                      # 10

#: Offsets of every contiguous 5-slot colour-major feature group.  The C5
#: symmetry permutes exactly these (``symmetry.feature_perm`` is built from
#: this list); every other feature is colour invariant.
_GROUPS: List[int] = []
for _slot in range(BOARD_SLOTS + MAX_RESERVED):
    _b = _slot * CARD_FEATURES
    _GROUPS += [_b, _b + 5, _b + 14]
for _slot in range(OTHER_SEATS * MAX_RESERVED):
    _b = OTHER_RESERVED_OFF + _slot * OTHER_CARD_FEATURES
    _GROUPS += [_b, _b + 5, _b + 14]
for _j in range(MAX_SEATS):
    _b = PLAYER_OFF + _j * PLAYER_FEATURES
    _GROUPS += [_b, _b + 6]
for _i in range(MAX_TILE_CHOICES):
    _b = TILE_OFF + _i * TILE_FEATURES
    _GROUPS += [_b, _b + 5]
for _t in range(3):
    for _bucket in range(3):
        _GROUPS.append(DECK_OFF + _t * 15 + _bucket * 5)
_GROUPS.append(GLOBAL_OFF)
COLOUR_GROUP_BASES: Tuple[int, ...] = tuple(_GROUPS)
del _slot, _b, _j, _i, _t, _bucket, _GROUPS

# ── static tables ─────────────────────────────────────────────────────────

#: ``(94, 5)`` raw card costs; the four sentinel rows cost nothing.
_CARD_COST_EXT = np.zeros((NUM_CARDS + 4, NUM_COLORS), dtype=np.float32)
_CARD_COST_EXT[:NUM_CARDS] = np.array(CARD_COST, dtype=np.float32)

#: ``(94, 25)`` colour-invariant part of a card block: cost/7, reward one-hot,
#: points/5, tier one-hot, the (dynamic) columns 14-21 left at zero, present,
#: known and deck_reserved.  Row ``EMPTY_CARD`` is all zeros; the three
#: ``HIDDEN_CARD0`` rows carry only the tier one-hot, present and
#: deck_reserved, exactly as §1.3 requires for an unknown card.
_CARD_STATIC = np.zeros((NUM_CARDS + 4, OTHER_CARD_FEATURES), dtype=np.float32)
_CARD_STATIC[:NUM_CARDS, 0:5] = _CARD_COST_EXT[:NUM_CARDS] / 7.0
_CARD_STATIC[np.arange(NUM_CARDS), 5 + np.array(CARD_REWARD)] = 1.0
_CARD_STATIC[:NUM_CARDS, 10] = np.array(CARD_POINTS, dtype=np.float32) / 5.0
_CARD_STATIC[np.arange(NUM_CARDS), 11 + np.array(CARD_TIER0)] = 1.0
_CARD_STATIC[:NUM_CARDS, 22] = 1.0
for _t in range(3):
    _CARD_STATIC[HIDDEN_CARD0 + _t, 11 + _t] = 1.0
    _CARD_STATIC[HIDDEN_CARD0 + _t, 22] = 1.0      # present
    _CARD_STATIC[HIDDEN_CARD0 + _t, 24] = 1.0      # deck_reserved
del _t

#: ``(11, 5)`` raw tile requirements plus an all-zero sentinel row.
_TILE_REQ_EXT = np.zeros((NUM_TILES + 1, NUM_COLORS), dtype=np.float32)
_TILE_REQ_EXT[:NUM_TILES] = np.array(TILE_REQ, dtype=np.float32)

#: Deck-composition group of every card: ``tier*15 + bucket*5 + colour``.
_POINT_BUCKET = np.array([0 if c.points == 0 else (1 if c.points <= 2 else 2)
                          for c in CARDS])
_DECK_GROUP = (np.array([(c.tier - 1) * 15 for c in CARDS])
               + _POINT_BUCKET * 5 + np.array(CARD_REWARD))
#: How many cards of each of the 45 groups exist in total.
_GROUP_TOTALS = np.bincount(_DECK_GROUP, minlength=45).astype(np.float32)
#: Same, extended with a 46th "not a card" group so that the sentinel rows of
#: a card-slot array (empty slots and other seats' blind reserves) can be
#: counted and thrown away in one :func:`numpy.bincount`.
_DECK_GROUP_EXT = np.concatenate([_DECK_GROUP, [45, 45, 45, 45]])
#: Deck size after the initial deal — 40/30/20 cards minus the 4 face up.
_INITIAL_DECK = np.array([36.0, 26.0, 16.0], dtype=np.float32)
_DECK_SCALE = 1.0 / _INITIAL_DECK

_MODE_INDEX = {MODE_INDIVIDUAL: 0, MODE_TEAM: 1, MODE_ONE_V_TWO: 2}
_LAYOUT_INDEX = {None: 0, "ADJACENT": 1, "OPPOSITE": 2}

#: Win thresholds per side, used by the "excess" and "progress" features.
_THRESHOLD_INDIVIDUAL = 15.0
_THRESHOLD_TEAM = 30.0
_THRESHOLD_SOLO = 15.0
_THRESHOLD_DUO = 34.0

_PAD_CARD: Tuple[Tuple[int, ...], ...] = tuple(
    (EMPTY_CARD,) * i for i in range(MAX_BOARD_SLOTS + 1))
_PAD_TILE: Tuple[Tuple[int, ...], ...] = tuple(
    (EMPTY_TILE,) * i for i in range(MAX_TILE_CHOICES + 1))
_ZERO_PVEC = (0,) * 11
_ZERO_PSCAL = (0,) * 6
_HIDDEN_OF_TIER0 = tuple(HIDDEN_CARD0 + CARD_TIER0[c] for c in range(NUM_CARDS))

#: Below this batch size the per-state path beats the vectorised one.
_BATCH_MIN = 8


# ── shared helpers ────────────────────────────────────────────────────────

def _side_totals(state: GameState) -> Tuple[List[float], List[float]]:
    """Per-seat ``(side total, side threshold)`` — what the "excess" and
    "threshold progress" features are measured against."""
    players = state.players
    mode = state.mode
    if mode == MODE_INDIVIDUAL:
        return ([float(p.score) for p in players],
                [_THRESHOLD_INDIVIDUAL] * len(players))
    team_total = [0.0, 0.0]
    for p in players:
        if p.team_id is not None:
            team_total[p.team_id] += p.score
    thr = ((_THRESHOLD_SOLO, _THRESHOLD_DUO) if mode == MODE_ONE_V_TWO
           else (_THRESHOLD_TEAM, _THRESHOLD_TEAM))
    totals = [team_total[p.team_id] if p.team_id is not None else 0.0
              for p in players]
    thresholds = [thr[p.team_id] if p.team_id is not None
                  else _THRESHOLD_INDIVIDUAL for p in players]
    return totals, thresholds


def _plies_to_round_leader(state: GameState) -> int:
    """Turns (1..active seats) until the round leader acts again."""
    n = state.num_players
    leader = state.round_start_player
    resigned = state.resigned
    if leader is None or leader in resigned:
        leader = next((i for i in range(n) if i not in resigned), 0)
    p = state.current_player
    for step in range(1, n + 1):
        p = get_next_active_player(state, p)
        if p == leader:
            return step
    return n


def _card_slot_ids(state: GameState, seat: int, sink: List[int]) -> None:
    """Append the 24 card-slot ids (board, my reserves, their reserves) of the
    information set of ``seat``, padded with the sentinel rows."""
    n = state.num_players
    players = state.players
    for row in state.board:
        sink.extend(row)
        sink.extend(_PAD_CARD[MAX_BOARD_SLOTS - len(row)])
    reserved = players[seat].reserved
    sink.extend(reserved)
    sink.extend(_PAD_CARD[MAX_RESERVED - len(reserved)])
    for j in range(1, MAX_SEATS):
        if j >= n:
            sink.extend(_PAD_CARD[MAX_RESERVED])
            continue
        p = players[(seat + j) % n]
        public = p.reserved_public
        for s, cid in enumerate(p.reserved):
            sink.append(cid if public[s] else _HIDDEN_OF_TIER0[cid])
        sink.extend(_PAD_CARD[MAX_RESERVED - len(p.reserved)])


def _tableau_cards(state: GameState, sink: List[int]) -> None:
    """Append every card in a tableau.  Together with the 24 card slots (the
    board, my reserves and the publicly reserved cards of the other seats)
    this is exactly the set of cards ``seat`` has seen, so the unseen pool is
    ``all cards - (slots + tableaus)`` and another seat's blind reserve — a
    sentinel in the slot array — correctly stays in the pool.""" 
    for p in state.players:
        sink.extend(p.cards)


# ── single-state encoder ──────────────────────────────────────────────────

def encode(state: GameState, seat: int,
           out: Optional[np.ndarray] = None) -> np.ndarray:
    """Encode ``state`` from the point of view of ``seat``.

    ``out`` (``float32``, :data:`OBS_DIM` entries — a row of a batch works)
    is filled in place and returned; a fresh array is allocated when it is
    ``None``.
    """
    if out is None:
        out = np.zeros(OBS_DIM, dtype=np.float32)
    else:
        out[:] = 0.0

    n = state.num_players
    players = state.players
    me = players[seat]
    cfg = state.config
    tpc = float(cfg["tokensPerColor"])
    wild = float(cfg["wildTokens"])
    mode = state.mode
    d = me.discount
    g = me.gems

    # ---- card slots ------------------------------------------------------
    ids: List[int] = []
    _card_slot_ids(state, seat, ids)
    idx = np.array(ids)
    blk = _CARD_STATIC[idx]                                  # (24, 25) copy
    short = _CARD_COST_EXT[idx] - [d[0] + g[0], d[1] + g[1], d[2] + g[2],
                                   d[3] + g[3], d[4] + g[4]]
    np.maximum(short, 0.0, out=short)
    total_short = short.sum(1)
    max_short = short.max(1)
    real = idx < EMPTY_CARD
    short *= 1.0 / 7.0
    blk[:, 14:19] = short
    blk[:, 19] = np.minimum(total_short * 0.2, 1.0)
    blk[:, 20] = np.less_equal(total_short, g[5]) & real
    blk[:, 21] = np.minimum(
        np.maximum(np.ceil(total_short * (1.0 / 3.0)), max_short) * (1.0 / 6.0),
        1.0)
    blk[_OTHER_SLOT0:, 23] = real[_OTHER_SLOT0:]
    out[BOARD_OFF:OWN_RESERVED_OFF] = blk[:_OWN_SLOT0, :CARD_FEATURES].reshape(-1)
    out[OWN_RESERVED_OFF:OTHER_RESERVED_OFF] = \
        blk[_OWN_SLOT0:_OTHER_SLOT0, :CARD_FEATURES].reshape(-1)
    out[OTHER_RESERVED_OFF:PLAYER_OFF] = blk[_OTHER_SLOT0:].reshape(-1)

    # ---- player blocks ---------------------------------------------------
    totals, thresholds = _side_totals(state)
    my_team = me.team_id
    resigned = state.resigned
    flat: List[float] = []
    for j in range(n):
        who = (seat + j) % n
        p = players[who]
        pg = p.gems
        pd = p.discount
        excess = (totals[who] - thresholds[who]) / 15.0
        if excess > 1.0:
            excess = 1.0
        elif excess < -1.0:
            excess = -1.0
        flat += [
            pg[0] / tpc, pg[1] / tpc, pg[2] / tpc, pg[3] / tpc, pg[4] / tpc,
            pg[5] / wild,
            min(pd[0] / 7.0, 1.0), min(pd[1] / 7.0, 1.0), min(pd[2] / 7.0, 1.0),
            min(pd[3] / 7.0, 1.0), min(pd[4] / 7.0, 1.0),
            min(p.score / 15.0, 1.0),
            min(len(p.cards) / 20.0, 1.0),
            len(p.reserved) / 3.0,
            min(len(p.tiles) / 3.0, 1.0),
            1.0 if who in resigned else 0.0,
            1.0,                                          # present
            1.0 if j == 0 else 0.0,                       # is_self
            1.0 if (j != 0 and my_team is not None
                    and p.team_id == my_team) else 0.0,   # is_teammate
            1.0 if (mode == MODE_ONE_V_TWO and p.team_id == 0) else 0.0,
            1.0 if j == 0 else 0.0, 1.0 if j == 1 else 0.0,
            1.0 if j == 2 else 0.0, 1.0 if j == 3 else 0.0,
            excess,
            0.0, 0.0, 0.0,
        ]
    out[PLAYER_OFF:PLAYER_OFF + n * PLAYER_FEATURES] = flat

    # ---- noble tiles -----------------------------------------------------
    tiles = state.tiles
    m = len(tiles) if len(tiles) < MAX_TILE_CHOICES else MAX_TILE_CHOICES
    if m:
        tile_idx = np.array(tiles[:m])
        req = _TILE_REQ_EXT[tile_idx]
        mine = np.maximum(req - [d[0], d[1], d[2], d[3], d[4]], 0.0)
        mine_total = mine.sum(1)
        tb = np.zeros((MAX_TILE_CHOICES, TILE_FEATURES), dtype=np.float32)
        tb[:m, 0:5] = req * 0.25
        tb[:m, 5:10] = mine * 0.25
        tb[:m, 10] = mine_total * (1.0 / 12.0)
        tb[:m, 11] = 1.0
        tb[:m, 12] = mine_total == 0.0
        if n > 1:
            others = np.array([players[(seat + j) % n].discount
                               for j in range(1, n)], dtype=np.float32)
            gap = np.maximum(req[None, :, :] - others[:, None, :], 0.0)
            tb[:m, 13:12 + n] = gap.sum(2).T * (1.0 / 12.0)
        out[TILE_OFF:DECK_OFF] = tb.reshape(-1)

    # ---- public deck composition ----------------------------------------
    tableaus: List[int] = []
    _tableau_cards(state, tableaus)
    counts = (np.bincount(_DECK_GROUP_EXT[idx], minlength=46)
              + np.bincount(_DECK_GROUP[np.array(tableaus, dtype=np.intp)],
                            minlength=46))
    out[DECK_OFF:DECK_OFF + 45] = (_GROUP_TOTALS - counts[:45]) * 0.125
    dc = state.deck_counts
    out[DECK_OFF + 45:GLOBAL_OFF] = [dc[0] / 36.0, dc[1] / 26.0, dc[2] / 16.0]

    # ---- global ----------------------------------------------------------
    sg = state.gems
    frt = state.final_round_triggered_by
    frt_off = -1 if frt is None else (frt - seat) % n
    layout = state.team_layout
    my_progress = min(totals[seat] / thresholds[seat], 1.0)
    if mode == MODE_INDIVIDUAL:
        best_other = max((players[i].score for i in range(n) if i != seat),
                         default=0)
        other_progress = min(best_other / _THRESHOLD_INDIVIDUAL, 1.0)
    else:
        other_team = 1 - (my_team if my_team is not None else 0)
        other_total, other_thr = 0.0, _THRESHOLD_TEAM
        for i in range(n):
            if players[i].team_id == other_team:
                other_total, other_thr = totals[i], thresholds[i]
                break
        other_progress = min(other_total / other_thr, 1.0)
    out[GLOBAL_OFF:OBS_DIM] = [
        sg[0] / tpc, sg[1] / tpc, sg[2] / tpc, sg[3] / tpc, sg[4] / tpc,
        sg[5] / wild,
        1.0 if mode == MODE_INDIVIDUAL else 0.0,
        1.0 if mode == MODE_TEAM else 0.0,
        1.0 if mode == MODE_ONE_V_TWO else 0.0,
        1.0 if layout == "ADJACENT" else 0.0,
        1.0 if layout == "OPPOSITE" else 0.0,
        1.0 if n == 2 else 0.0, 1.0 if n == 3 else 0.0, 1.0 if n == 4 else 0.0,
        min(state.turn_number / 100.0, 1.0),
        1.0 if frt is not None else 0.0,
        1.0 if frt_off == 0 else 0.0, 1.0 if frt_off == 1 else 0.0,
        1.0 if frt_off == 2 else 0.0, 1.0 if frt_off == 3 else 0.0,
        (min(_plies_to_round_leader(state) * 0.25, 1.0)
         if state.phase == PHASE_PLAYING else 0.0),
        1.0 if (mode == MODE_TEAM and frt is not None) else 0.0,
        my_progress,
        other_progress,
        1.0 if (state.pending_tile_choice
                and state.current_player == seat) else 0.0,
        1.0 if seat == 0 else 0.0, 1.0 if seat == 1 else 0.0,
        1.0 if seat == 2 else 0.0, 1.0 if seat == 3 else 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    return out


# ── batch encoder ─────────────────────────────────────────────────────────

def encode_batch(states: Sequence[GameState], seats: Sequence[int],
                 out: Optional[np.ndarray] = None) -> np.ndarray:
    """Encode ``(state, seat)`` pairs into a ``(B, OBS_DIM)`` ``float32`` array.

    This is the hot path of the inference server and of the learner (which
    re-encodes stored positions on the fly), so the per-state Python work is
    reduced to gathering flat lists of ids and counters and every feature is
    then computed once for the whole batch.  ``encode(state, seat)`` is
    exercised against this on random positions by ``tests/test_encode.py``.
    """
    b = len(states)
    if b != len(seats):
        raise ValueError(f"{b} states but {len(seats)} seats")
    if out is None:
        out = np.zeros((b, OBS_DIM), dtype=np.float32)
    elif out.shape != (b, OBS_DIM):
        raise ValueError(f"out has shape {out.shape}, expected {(b, OBS_DIM)}")
    if b < _BATCH_MIN:
        for i in range(b):
            encode(states[i], seats[i], out[i])
        return out

    # ``array('h')`` buffers instead of Python lists: extending them costs
    # about a third of a list append per element and ``np.frombuffer`` then
    # wraps the result without converting anything.
    card_ids = array("h")
    tile_ids = array("h")
    pvec = array("h")               # per seat: gems[6] + discount[5]
    pscal = array("h")              # per seat: score, cards, reserved, tiles,
    #                                 resigned, team_id + 1 (0 = none)
    gvec = array("h")               # 19 per state, see the unpacking below
    tableaus = array("h")
    tableau_len = []

    for i in range(b):
        state = states[i]
        seat = seats[i]
        n = state.num_players
        players = state.players
        cfg = state.config
        resigned = state.resigned
        _card_slot_ids(state, seat, card_ids)
        tiles = state.tiles
        m = len(tiles) if len(tiles) < MAX_TILE_CHOICES else MAX_TILE_CHOICES
        tile_ids.extend(tiles[:m])
        tile_ids.extend(_PAD_TILE[MAX_TILE_CHOICES - m])
        for j in range(MAX_SEATS):
            if j >= n:
                pvec.extend(_ZERO_PVEC)
                pscal.extend(_ZERO_PSCAL)
                continue
            who = (seat + j) % n
            p = players[who]
            pvec.extend(p.gems)
            pvec.extend(p.discount)
            pscal.extend((p.score, len(p.cards), len(p.reserved), len(p.tiles),
                          1 if who in resigned else 0,
                          0 if p.team_id is None else p.team_id + 1))
        before = len(tableaus)
        _tableau_cards(state, tableaus)
        tableau_len.append(len(tableaus) - before)
        frt = state.final_round_triggered_by
        dc = state.deck_counts
        sg = state.gems
        gvec.extend((n, _MODE_INDEX[state.mode],
                     _LAYOUT_INDEX[state.team_layout],
                     cfg["tokensPerColor"], cfg["wildTokens"],
                     state.turn_number,
                     -1 if frt is None else (frt - seat) % n,
                     (_plies_to_round_leader(state)
                      if state.phase == PHASE_PLAYING else 0),
                     1 if (state.pending_tile_choice
                           and state.current_player == seat) else 0,
                     seat, dc[0], dc[1], dc[2],
                     sg[0], sg[1], sg[2], sg[3], sg[4], sg[5]))

    rows = np.arange(b)
    cid = np.frombuffer(card_ids, dtype=np.int16).reshape(b, _CARD_SLOTS)
    pv = np.frombuffer(pvec, dtype=np.int16).reshape(
        b, MAX_SEATS, 11).astype(np.float32)
    ps = np.frombuffer(pscal, dtype=np.int16).reshape(
        b, MAX_SEATS, 6).astype(np.float32)
    gv = np.frombuffer(gvec, dtype=np.int16).reshape(b, 19).astype(np.float32)
    n_players = gv[:, 0]
    mode = gv[:, 1]
    tpc = gv[:, 3, None]
    wild = gv[:, 4, None]
    my_gems = pv[:, 0, 0:6]
    my_disc = pv[:, 0, 6:11]
    present = np.arange(MAX_SEATS)[None, :] < n_players[:, None]     # (b, 4)

    # ---- card slots ------------------------------------------------------
    blk = _CARD_STATIC[cid]                                  # (b, 24, 25)
    short = _CARD_COST_EXT[cid] - (my_disc + my_gems[:, 0:5])[:, None, :]
    np.maximum(short, 0.0, out=short)
    total_short = short.sum(2)
    max_short = short.max(2)
    real = cid < EMPTY_CARD
    short *= 1.0 / 7.0
    blk[:, :, 14:19] = short
    blk[:, :, 19] = np.minimum(total_short * 0.2, 1.0)
    blk[:, :, 20] = np.less_equal(total_short, my_gems[:, 5, None]) & real
    blk[:, :, 21] = np.minimum(
        np.maximum(np.ceil(total_short * (1.0 / 3.0)), max_short) * (1.0 / 6.0),
        1.0)
    blk[:, _OTHER_SLOT0:, 23] = real[:, _OTHER_SLOT0:]
    out[:, BOARD_OFF:OWN_RESERVED_OFF] = \
        blk[:, :_OWN_SLOT0, :CARD_FEATURES].reshape(b, -1)
    out[:, OWN_RESERVED_OFF:OTHER_RESERVED_OFF] = \
        blk[:, _OWN_SLOT0:_OTHER_SLOT0, :CARD_FEATURES].reshape(b, -1)
    out[:, OTHER_RESERVED_OFF:PLAYER_OFF] = blk[:, _OTHER_SLOT0:].reshape(b, -1)

    # ---- player blocks ---------------------------------------------------
    pb = np.zeros((b, MAX_SEATS, PLAYER_FEATURES), dtype=np.float32)
    pb[:, :, 0:5] = pv[:, :, 0:5] / tpc[:, :, None]
    pb[:, :, 5] = pv[:, :, 5] / wild
    pb[:, :, 6:11] = np.minimum(pv[:, :, 6:11] * (1.0 / 7.0), 1.0)
    score = ps[:, :, 0]
    pb[:, :, 11] = np.minimum(score * (1.0 / 15.0), 1.0)
    pb[:, :, 12] = np.minimum(ps[:, :, 1] * 0.05, 1.0)
    pb[:, :, 13] = ps[:, :, 2] * (1.0 / 3.0)
    pb[:, :, 14] = np.minimum(ps[:, :, 3] * (1.0 / 3.0), 1.0)
    pb[:, :, 15] = ps[:, :, 4]
    pb[:, :, 16] = 1.0
    pb[:, 0, 17] = 1.0
    team = ps[:, :, 5]                             # 0 = none, 1 = t0, 2 = t1
    my_team = team[:, 0:1]
    mate = (team == my_team) & (my_team > 0)
    mate[:, 0] = False
    pb[:, :, 18] = mate
    is_ovt = mode[:, None] == 2
    pb[:, :, 19] = (team == 1) & is_ovt
    for _j in range(MAX_SEATS):
        pb[:, _j, 20 + _j] = 1.0
    total0 = (score * (team == 1)).sum(1)[:, None]
    total1 = (score * (team == 2)).sum(1)[:, None]
    is_ind = mode[:, None] == 0
    side_total = np.where(is_ind, score,
                          np.where(team == 1, total0,
                                   np.where(team == 2, total1, 0.0)))
    side_thr = np.where(is_ind, _THRESHOLD_INDIVIDUAL,
                        np.where(is_ovt,
                                 np.where(team == 1, _THRESHOLD_SOLO,
                                          _THRESHOLD_DUO),
                                 _THRESHOLD_TEAM))
    pb[:, :, 24] = np.clip((side_total - side_thr) * (1.0 / 15.0), -1.0, 1.0)
    pb *= present[:, :, None]
    out[:, PLAYER_OFF:TILE_OFF] = pb.reshape(b, -1)

    # ---- noble tiles -----------------------------------------------------
    tid = np.frombuffer(tile_ids, dtype=np.int16).reshape(b, MAX_TILE_CHOICES)
    req = _TILE_REQ_EXT[tid]                                 # (b, 5, 5)
    tb = np.zeros((b, MAX_TILE_CHOICES, TILE_FEATURES), dtype=np.float32)
    tb[:, :, 0:5] = req * 0.25
    mine = np.maximum(req - my_disc[:, None, :], 0.0)
    mine_total = mine.sum(2)
    tb[:, :, 5:10] = mine * 0.25
    tb[:, :, 10] = mine_total * (1.0 / 12.0)
    tb[:, :, 11] = 1.0
    tb[:, :, 12] = mine_total == 0.0
    gap = np.maximum(req[:, None, :, :] - pv[:, 1:MAX_SEATS, None, 6:11], 0.0)
    tb[:, :, 13:16] = (gap.sum(3) * (1.0 / 12.0)).transpose(0, 2, 1) \
        * present[:, None, 1:MAX_SEATS]
    tb *= (tid < EMPTY_TILE)[:, :, None]
    out[:, TILE_OFF:DECK_OFF] = tb.reshape(b, -1)

    # ---- public deck composition ----------------------------------------
    offsets = rows * 46
    slot_groups = _DECK_GROUP_EXT[cid] + offsets[:, None]
    tab_groups = (_DECK_GROUP[np.frombuffer(tableaus, dtype=np.int16)]
                  + np.repeat(offsets, tableau_len))
    counts = (np.bincount(slot_groups.reshape(-1), minlength=b * 46)
              + np.bincount(tab_groups, minlength=b * 46)).reshape(b, 46)
    out[:, DECK_OFF:DECK_OFF + 45] = (_GROUP_TOTALS - counts[:, :45]) * 0.125
    out[:, DECK_OFF + 45:GLOBAL_OFF] = gv[:, 10:13] * _DECK_SCALE

    # ---- global ----------------------------------------------------------
    gb = np.zeros((b, GLOBAL_FEATURES), dtype=np.float32)
    gb[:, 0:5] = gv[:, 13:18] / tpc
    gb[:, 5] = gv[:, 18] / gv[:, 4]
    gb[rows, 6 + mode.astype(np.intp)] = 1.0
    layout = gv[:, 2].astype(np.intp)
    has_layout = layout > 0
    gb[rows[has_layout], 8 + layout[has_layout]] = 1.0
    gb[rows, 9 + n_players.astype(np.intp)] = 1.0
    gb[:, 14] = np.minimum(gv[:, 5] * 0.01, 1.0)
    frt = gv[:, 6]
    in_final = frt >= 0.0
    gb[:, 15] = in_final
    frt_i = frt.astype(np.intp)
    gb[rows[in_final], 16 + frt_i[in_final]] = 1.0
    gb[:, 20] = np.minimum(gv[:, 7] * 0.25, 1.0)
    gb[:, 21] = in_final & (mode == 1)
    my_total = side_total[:, 0]
    my_thr = side_thr[:, 0]
    gb[:, 22] = np.minimum(my_total / my_thr, 1.0)
    other_score = (score[:, 1:MAX_SEATS] * present[:, 1:MAX_SEATS]).max(1)
    other_total = np.where(my_team[:, 0] == 1, total1[:, 0], total0[:, 0])
    other_thr = np.where(is_ovt[:, 0],
                         np.where(my_team[:, 0] == 1, _THRESHOLD_DUO,
                                  _THRESHOLD_SOLO),
                         _THRESHOLD_TEAM)
    gb[:, 23] = np.where(is_ind[:, 0],
                         np.minimum(other_score * (1.0 / 15.0), 1.0),
                         np.minimum(other_total / other_thr, 1.0))
    gb[:, 24] = gv[:, 8]
    gb[rows, 25 + gv[:, 9].astype(np.intp)] = 1.0
    out[:, GLOBAL_OFF:OBS_DIM] = gb
    return out


__all__ = ["OBS_VERSION", "OBS_DIM", "encode", "encode_batch",
           "COLOUR_GROUP_BASES", "CARD_FEATURES", "OTHER_CARD_FEATURES",
           "PLAYER_FEATURES", "TILE_FEATURES", "DECK_FEATURES",
           "GLOBAL_FEATURES", "BOARD_OFF", "OWN_RESERVED_OFF",
           "OTHER_RESERVED_OFF", "PLAYER_OFF", "TILE_OFF", "DECK_OFF",
           "GLOBAL_OFF", "MAX_SEATS", "EMPTY_CARD", "HIDDEN_CARD0",
           "EMPTY_TILE"]
