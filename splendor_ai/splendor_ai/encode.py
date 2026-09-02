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

from typing import List, Optional, Sequence

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

#: Number of card slots the encoder evaluates in one shot: 12 board + 3 own
#: reserved + 9 opponent reserved.
_CARD_SLOTS = BOARD_SLOTS + MAX_RESERVED + OTHER_SEATS * MAX_RESERVED   # 24
_OWN_SLOT0 = BOARD_SLOTS
_OTHER_SLOT0 = BOARD_SLOTS + MAX_RESERVED

#: Offsets of every contiguous 5-slot colour-major feature group.  The C5
#: symmetry permutes exactly these (``symmetry.feature_perm`` is built from
#: this list); every other feature is colour invariant.
_COLOUR_GROUP_BASES: List[int] = []
for _slot in range(BOARD_SLOTS + MAX_RESERVED):
    _b = _slot * CARD_FEATURES
    _COLOUR_GROUP_BASES += [_b, _b + 5, _b + 14]
for _slot in range(OTHER_SEATS * MAX_RESERVED):
    _b = OTHER_RESERVED_OFF + _slot * OTHER_CARD_FEATURES
    _COLOUR_GROUP_BASES += [_b, _b + 5, _b + 14]
for _j in range(MAX_SEATS):
    _b = PLAYER_OFF + _j * PLAYER_FEATURES
    _COLOUR_GROUP_BASES += [_b, _b + 6]
for _i in range(MAX_TILE_CHOICES):
    _b = TILE_OFF + _i * TILE_FEATURES
    _COLOUR_GROUP_BASES += [_b, _b + 5]
for _t in range(3):
    for _bucket in range(3):
        _COLOUR_GROUP_BASES.append(DECK_OFF + _t * 15 + _bucket * 5)
_COLOUR_GROUP_BASES.append(GLOBAL_OFF)
COLOUR_GROUP_BASES = tuple(_COLOUR_GROUP_BASES)
del _slot, _b, _j, _i, _t, _bucket, _COLOUR_GROUP_BASES

# ── static tables ─────────────────────────────────────────────────────────

#: ``(90, 5)`` raw card costs.
_CARD_COST = np.array(CARD_COST, dtype=np.float32)
#: ``(90, 14)`` cost/7, reward one-hot, points/5, tier one-hot.
_CARD_STATIC = np.zeros((NUM_CARDS, 14), dtype=np.float32)
_CARD_STATIC[:, 0:5] = _CARD_COST / 7.0
_CARD_STATIC[np.arange(NUM_CARDS), 5 + np.array(CARD_REWARD)] = 1.0
_CARD_STATIC[:, 10] = np.array(CARD_POINTS, dtype=np.float32) / 5.0
_CARD_STATIC[np.arange(NUM_CARDS), 11 + np.array(CARD_TIER0)] = 1.0

#: ``(10, 5)`` raw tile requirements.
_TILE_REQ = np.array(TILE_REQ, dtype=np.float32)
_TILE_REQ_SCALED = _TILE_REQ / 4.0

#: Deck-composition group of every card: ``tier*15 + bucket*5 + colour``.
_POINT_BUCKET = np.array([0 if c.points == 0 else (1 if c.points <= 2 else 2)
                          for c in CARDS])
_DECK_GROUP = np.array([(c.tier - 1) * 15 for c in CARDS]) \
    + _POINT_BUCKET * 5 + np.array(CARD_REWARD)
#: Total number of cards in each of the 45 groups (the "all cards" baseline).
_GROUP_TOTALS = np.bincount(_DECK_GROUP, minlength=45).astype(np.float32)
#: Deck size after the initial deal — 40/30/20 cards minus the 4 face up.
_INITIAL_DECK = np.array([36.0, 26.0, 16.0], dtype=np.float32)

_MODE_INDEX = {MODE_INDIVIDUAL: 0, MODE_TEAM: 1, MODE_ONE_V_TWO: 2}

#: Win thresholds per side, used for the "excess" and "progress" features.
_THRESHOLD_INDIVIDUAL = 15.0
_THRESHOLD_TEAM = 30.0
_THRESHOLD_SOLO = 15.0
_THRESHOLD_DUO = 34.0

_ZEROS_CARD_BLOCK = np.zeros((_CARD_SLOTS, OTHER_CARD_FEATURES), dtype=np.float32)
_ZEROS_TILE_BLOCK = np.zeros((MAX_TILE_CHOICES, TILE_FEATURES), dtype=np.float32)


# ── helpers ───────────────────────────────────────────────────────────────

def _side_totals(state: GameState) -> tuple:
    """``(per-seat side total, per-seat side threshold, my/other progress)``
    inputs — the aggregates the player and global blocks share."""
    players = state.players
    mode = state.mode
    if mode == MODE_INDIVIDUAL:
        totals = [float(p.score) for p in players]
        thresholds = [_THRESHOLD_INDIVIDUAL] * len(players)
        return totals, thresholds
    t = [0.0, 0.0]
    for p in players:
        if p.team_id is not None:
            t[p.team_id] += p.score
    if mode == MODE_ONE_V_TWO:
        thr = (_THRESHOLD_SOLO, _THRESHOLD_DUO)
    else:
        thr = (_THRESHOLD_TEAM, _THRESHOLD_TEAM)
    totals = [t[p.team_id] if p.team_id is not None else 0.0 for p in players]
    thresholds = [thr[p.team_id] if p.team_id is not None else
                  _THRESHOLD_INDIVIDUAL for p in players]
    return totals, thresholds


def _plies_to_round_leader(state: GameState) -> int:
    """How many turns (1..active seats) until the round leader acts again."""
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


# ── the encoder ───────────────────────────────────────────────────────────

def encode(state: GameState, seat: int,
           out: Optional[np.ndarray] = None) -> np.ndarray:
    """Encode ``state`` from the point of view of ``seat``.

    ``out`` (a ``float32`` array of :data:`OBS_DIM` entries, or a writable row
    of a batch) is filled in place and returned; a fresh array is allocated
    when it is ``None``.
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
    gold = g[5]

    # ---- card slots (board, my reserves, their reserves) -----------------
    ids = [0] * _CARD_SLOTS
    valid = [False] * _CARD_SLOTS
    board = state.board
    for t in range(3):
        row = board[t]
        base = t * MAX_BOARD_SLOTS
        for s in range(len(row)):
            ids[base + s] = row[s]
            valid[base + s] = True
    reserved = me.reserved
    for s in range(len(reserved)):
        ids[_OWN_SLOT0 + s] = reserved[s]
        valid[_OWN_SLOT0 + s] = True
    hidden: List[tuple] = []
    known: List[float] = [0.0] * (OTHER_SEATS * MAX_RESERVED)
    for j in range(1, n):
        p = players[(seat + j) % n]
        base = _OTHER_SLOT0 + (j - 1) * MAX_RESERVED
        pub = p.reserved_public
        for s, cid in enumerate(p.reserved):
            if pub[s]:
                ids[base + s] = cid
                valid[base + s] = True
                known[(j - 1) * MAX_RESERVED + s] = 1.0
            else:
                hidden.append((base + s, CARD_TIER0[cid]))

    idx = np.array(ids)
    live = np.array(valid)
    short = _CARD_COST[idx] - [d[0] + g[0], d[1] + g[1], d[2] + g[2],
                               d[3] + g[3], d[4] + g[4]]
    np.maximum(short, 0.0, out=short)
    total_short = short.sum(1)
    max_short = short.max(1)

    blk = np.empty((_CARD_SLOTS, OTHER_CARD_FEATURES), dtype=np.float32)
    blk[:, 0:14] = _CARD_STATIC[idx]
    short *= 1.0 / 7.0
    blk[:, 14:19] = short
    blk[:, 19] = np.minimum(total_short * 0.2, 1.0)
    blk[:, 20] = total_short <= gold
    blk[:, 21] = np.minimum(
        np.maximum(np.ceil(total_short * (1.0 / 3.0)), max_short) * (1.0 / 6.0),
        1.0)
    blk[:, 22] = 1.0
    blk[:, 23:25] = 0.0
    blk *= live[:, None]
    blk[_OTHER_SLOT0:, 23] = known
    for slot, tier0 in hidden:
        blk[slot, 11 + tier0] = 1.0
        blk[slot, 22] = 1.0
        blk[slot, 24] = 1.0

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
            1.0,                                        # present
            1.0 if j == 0 else 0.0,                     # is_self
            1.0 if (j != 0 and my_team is not None
                    and p.team_id == my_team) else 0.0,  # is_teammate
            1.0 if (mode == MODE_ONE_V_TWO and p.team_id == 0) else 0.0,
            1.0 if j == 0 else 0.0, 1.0 if j == 1 else 0.0,
            1.0 if j == 2 else 0.0, 1.0 if j == 3 else 0.0,
            excess,
            0.0, 0.0, 0.0,
        ]
    out[PLAYER_OFF:PLAYER_OFF + n * PLAYER_FEATURES] = flat

    # ---- noble tiles -----------------------------------------------------
    tiles = state.tiles
    m = len(tiles)
    if m > MAX_TILE_CHOICES:
        m = MAX_TILE_CHOICES
    if m:
        tile_idx = np.array(tiles[:m])
        req = _TILE_REQ[tile_idx]
        mine = np.maximum(req - [d[0], d[1], d[2], d[3], d[4]], 0.0)
        mine_total = mine.sum(1)
        tb = _ZEROS_TILE_BLOCK.copy()
        tb[:m, 0:5] = _TILE_REQ_SCALED[tile_idx]
        tb[:m, 5:10] = mine * 0.25
        tb[:m, 10] = mine_total * (1.0 / 12.0)
        tb[:m, 11] = 1.0
        tb[:m, 12] = mine_total == 0.0
        if n > 1:
            others = np.array([players[(seat + j) % n].discount
                               for j in range(1, n)], dtype=np.float32)
            gap = np.maximum(req[None, :, :] - others[:, None, :], 0.0)
            tb[:m, 13:13 + n - 1] = gap.sum(2).T * (1.0 / 12.0)
        out[TILE_OFF:DECK_OFF] = tb.reshape(-1)

    # ---- public deck composition ----------------------------------------
    seen: List[int] = []
    seen += board[0]
    seen += board[1]
    seen += board[2]
    for i, p in enumerate(players):
        seen += p.cards
        if i == seat:
            seen += p.reserved
        else:
            pub = p.reserved_public
            for s, cid in enumerate(p.reserved):
                if pub[s]:
                    seen.append(cid)
    counts = np.bincount(_DECK_GROUP[np.array(seen)], minlength=45)
    out[DECK_OFF:DECK_OFF + 45] = (_GROUP_TOTALS - counts) * 0.125
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
        other = 1 - (my_team if my_team is not None else 0)
        other_total = 0.0
        other_thr = _THRESHOLD_TEAM
        for i in range(n):
            if players[i].team_id == other:
                other_total = totals[i]
                other_thr = thresholds[i]
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
        (min(_plies_to_round_leader(state) / 4.0, 1.0)
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


def encode_batch(states: Sequence[GameState], seats: Sequence[int],
                 out: Optional[np.ndarray] = None) -> np.ndarray:
    """Encode a batch of ``(state, seat)`` pairs into a ``(B, OBS_DIM)`` array.

    ``out`` is filled in place when given (it must be C-contiguous ``float32``
    of the right shape) — the inference server and the learner keep one pinned
    buffer per batch slot and never allocate in the hot loop.
    """
    b = len(states)
    if b != len(seats):
        raise ValueError(f"{b} states but {len(seats)} seats")
    if out is None:
        out = np.zeros((b, OBS_DIM), dtype=np.float32)
    elif out.shape != (b, OBS_DIM):
        raise ValueError(f"out has shape {out.shape}, expected {(b, OBS_DIM)}")
    for i in range(b):
        encode(states[i], seats[i], out[i])
    return out


__all__ = ["OBS_VERSION", "OBS_DIM", "encode", "encode_batch",
           "COLOUR_GROUP_BASES", "CARD_FEATURES", "OTHER_CARD_FEATURES",
           "PLAYER_FEATURES", "TILE_FEATURES", "DECK_FEATURES",
           "GLOBAL_FEATURES", "BOARD_OFF", "OWN_RESERVED_OFF",
           "OTHER_RESERVED_OFF", "PLAYER_OFF", "TILE_OFF", "DECK_OFF",
           "GLOBAL_OFF", "MAX_SEATS"]
