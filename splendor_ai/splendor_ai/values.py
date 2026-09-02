"""Terminal value vectors — ``docs/AI_DESIGN.md`` §1.2.

One mechanism for every mode: a value **vector** with one entry per seat, in
ABSOLUTE seat order, entries ``>= num_players`` left at zero.

* ``INDIVIDUAL`` — rank linear over all seats, ``z = 1 - 2*(rank-1)/(n-1)``
  (``+-1`` in 2p).  The ranking is the server's:
  :func:`splendor_ai.rules.engine.rating_changes` sorts by score descending and
  then by *fewer* cards, and ties share the mean of the positions they occupy
  (so the vector always sums to zero).  A seat that resigned is ranked behind
  every seat still in the game — the engine has already zeroed its score, its
  cards and its nobles.
* ``ONE_V_TWO`` / ``TEAM`` — ``+1`` to every seat of the winning side, ``-1``
  to the other side, ``0`` to everyone when the sides tie (``winningTeamIds``
  holds both ids) or when nobody qualified (it is empty).  A ``FORFEIT``
  result gives the forfeiting side ``-1`` and the other side ``+1``.

:func:`seat_relative` turns the absolute vector into the network target
(index 0 = the acting seat) and :func:`z_valid_mask` marks the entries a value
head may be trained on.  :func:`standings_values` scores a *truncated* game
(``max_plies``) with the same ranking applied to the current standings; the
self-play loop keeps those targets at a reduced weight
(:data:`TRUNCATION_Z_WEIGHT`).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .rules.engine import (GameState, MODE_INDIVIDUAL, MODE_ONE_V_TWO,
                           PHASE_GAME_OVER, resolve_one_vs_two_winners)

#: Width of every value vector: the largest table this variant supports.
MAX_SEATS = 4

#: Weight the learner puts on a value target that came from
#: :func:`standings_values` instead of a real terminal (§1.2).
TRUNCATION_Z_WEIGHT = 0.3

#: Per-side score thresholds (``engine.qualifying_team_ids``).  The 1v2 sides
#: have *different* ones, so only progress towards them is comparable; both
#: TEAM sides share one, which is what makes the TEAM comparison symmetric.
_SOLO_THRESHOLD = 15
_DUO_THRESHOLD = 34
_TEAM_THRESHOLD = 30


def _rank_values(state: GameState, keys: Sequence[tuple]) -> np.ndarray:
    """Rank-linear values for ``keys`` (smaller sorts first), mean rank on
    ties.  ``z = 1 - 2*(rank-1)/(n-1)`` so the vector sums to zero."""
    n = state.num_players
    z = np.zeros(MAX_SEATS, dtype=np.float32)
    if n < 2:
        return z
    order = sorted(range(n), key=lambda i: keys[i])
    scale = 2.0 / (n - 1)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and keys[order[j + 1]] == keys[order[i]]:
            j += 1
        mean_rank = (i + j) * 0.5 + 1.0
        value = 1.0 - scale * (mean_rank - 1.0)
        for t in range(i, j + 1):
            z[order[t]] = value
        i = j + 1
    return z


def _individual_keys(state: GameState) -> List[tuple]:
    """Sort key per seat: resigned last, then score desc, then fewer cards."""
    resigned = state.resigned
    return [(1 if i in resigned else 0, -p.score, len(p.cards))
            for i, p in enumerate(state.players)]


def _team_side_values(state: GameState, winners: Sequence[int]) -> np.ndarray:
    """``+1`` for the seats of the single winning team, ``-1`` for the rest;
    all zero when ``winners`` does not name exactly one team."""
    z = np.zeros(MAX_SEATS, dtype=np.float32)
    if len(winners) != 1:
        return z
    winner = winners[0]
    for i, p in enumerate(state.players):
        z[i] = 1.0 if p.team_id == winner else -1.0
    return z


def terminal_values(state: GameState) -> np.ndarray:
    """The ``float32[4]`` outcome of a finished game, in absolute seat order.

    Raises ``ValueError`` unless ``state.phase == 'GAME_OVER'``.
    """
    if state.phase != PHASE_GAME_OVER:
        raise ValueError(
            "terminal_values() needs a finished game, "
            f"phase is {state.phase!r}")
    if state.mode == MODE_INDIVIDUAL:
        return _rank_values(state, _individual_keys(state))
    result = state.game_result
    if result is None:
        # Not reachable through the engine (both team modes always attach a
        # gameResult when they end); fall back to the standings so a
        # hand-built state still gets a sane value.
        return standings_values(state)
    if result.get("reason") == "FORFEIT":
        forfeiting = result.get("forfeitingTeamId")
        if forfeiting is None:
            return _team_side_values(state, result.get("winningTeamIds") or [])
        z = np.zeros(MAX_SEATS, dtype=np.float32)
        for i, p in enumerate(state.players):
            z[i] = -1.0 if p.team_id == forfeiting else 1.0
        return z
    return _team_side_values(state, result.get("winningTeamIds") or [])


def standings_values(state: GameState) -> np.ndarray:
    """Value vector for a game cut short at ``max_plies``.

    The same ranking as :func:`terminal_values`, applied to the standings as
    they are.  ``game_result`` is ignored, so this is meaningful on an
    unfinished position.

    * INDIVIDUAL — seats by (score desc, cards asc), resigned seats last.
    * TEAM — by team total, card count breaking the tie.  Both sides need the
      same 30 points, so their progress ratios share a denominator and
      comparing the totals directly is already threshold-normalised (and
      therefore symmetric: swapping the two sides negates the vector).
    * ONE_V_TWO — once a side has qualified the real rule can call the game,
      so :func:`~splendor_ai.rules.engine.resolve_one_vs_two_winners` (excess
      over each side's own threshold) decides it, exactly as the engine will
      at the end of the round.  Before that, the sides are compared by how far
      each has come towards its own threshold, graded in ``[-1, 1]``: the two
      thresholds differ by 19 points, so a plain difference of excesses is
      dominated by that offset and would score almost every early position as
      a solo win.
    """
    mode = state.mode
    if mode == MODE_INDIVIDUAL:
        return _rank_values(state, _individual_keys(state))

    totals = [0, 0]
    cards = [0, 0]
    for p in state.players:
        if p.team_id is not None:
            totals[p.team_id] += p.score
            cards[p.team_id] += len(p.cards)
    z = np.zeros(MAX_SEATS, dtype=np.float32)

    if mode == MODE_ONE_V_TWO:
        # A side has already qualified → the engine's own rule decides.  It
        # returns [] while neither side qualifies and both ids on an exact
        # excess tie (which _team_side_values scores as a draw).
        winners = resolve_one_vs_two_winners(state)
        if winners:
            return _team_side_values(state, winners)
        solo_prog = totals[0] / _SOLO_THRESHOLD
        duo_prog = totals[1] / _DUO_THRESHOLD
        v_solo = float(np.clip(2.0 * (solo_prog - duo_prog), -1.0, 1.0))
        for i, p in enumerate(state.players):
            if p.team_id == 0:
                z[i] = v_solo
            elif p.team_id == 1:
                z[i] = -v_solo
        return z

    # TEAM: both sides need the same 30 points, so dividing by the shared
    # threshold cannot change the order or the ties — the comparison is
    # symmetric, and swapping the sides just negates the vector.
    prog = (totals[0] / _TEAM_THRESHOLD, totals[1] / _TEAM_THRESHOLD)
    if prog[0] != prog[1]:
        lead = 1 if prog[0] > prog[1] else -1
    elif cards[0] != cards[1]:
        lead = 1 if cards[0] < cards[1] else -1
    else:
        lead = 0
    if lead:
        for i, p in enumerate(state.players):
            if p.team_id is not None:
                z[i] = float(lead) if p.team_id == 0 else float(-lead)
    return z


def seat_relative(z, seat: int, n: int = MAX_SEATS) -> np.ndarray:
    """Rotate an absolute value vector so that index 0 is ``seat``.

    §1.2 writes this as ``np.roll(z, -seat)``, which is exact for a full table
    of four.  With fewer seats the roll has to stay inside the first ``n``
    entries, otherwise a real value would land in a padding slot that
    :func:`z_valid_mask` masks off — so pass ``n`` for 2p and 3p games (the
    default keeps the documented ``np.roll`` behaviour).
    """
    z = np.asarray(z, dtype=np.float32)
    out = np.array(z, dtype=np.float32, copy=True)
    if n > 0:
        out[:n] = np.roll(z[:n], -seat)
    return out


def z_valid_mask(n: int) -> np.ndarray:
    """``float32[4]`` marking the value-head entries a game with ``n`` seats
    actually supervises."""
    mask = np.zeros(MAX_SEATS, dtype=np.float32)
    mask[:n] = 1.0
    return mask


__all__ = ["MAX_SEATS", "TRUNCATION_Z_WEIGHT", "terminal_values",
           "standings_values", "seat_relative", "z_valid_mask"]
