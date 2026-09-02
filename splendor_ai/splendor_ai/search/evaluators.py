"""Leaf evaluators and the NN-free heuristics they share.

**Contract** (``AI_DESIGN`` §1.6, and see ``search/__init__.py``)::

    priors, values = evaluator.evaluate(obs, mask)
    #  obs   : batch of B observations produced by encode_fn(state, seat)
    #  mask  : bool[B, 65] legal-action masks
    #  priors: float32[B, 65]  (masked + renormalised again by the tree)
    #  values: float32[B, 4]   RELATIVE TO THE LEAF'S ACTING SEAT
    #          index j is absolute seat (j + leaf_seat) % 4, so index 0 is
    #          always the seat to move at that leaf.

``obs`` is whatever the caller's ``encode_fn`` produces.  Numeric encoders
(``splendor_ai/encode.py`` later, :class:`ZeroEncoder` in tests) yield
``float32[OBS_DIM]`` rows and the batch is a ``float32[B, OBS_DIM]`` array;
the NN-free evaluators here need the position itself, so they are paired with
:func:`state_encoder`, whose "observation" is a :class:`LeafRef`.
"""

from __future__ import annotations

from typing import Any, List, NamedTuple, Optional, Protocol, Sequence, Tuple

import numpy as np

from ..rules import engine as E
from ..rules.actions import (
    BUY_BOARD_START, BUY_RESERVED_START, CHOOSE_TILE_START, MAX_BOARD_SLOTS,
    NUM_ACTIONS, NUM_TAKE_ACTIONS, RESERVE_BOARD_START, RESERVE_DECK_START,
    TAKE_PATTERNS,
)
from ..rules.cards import CARD_COST, CARD_COST_NZ, CARD_POINTS
from .mcts import seat_relative, standings_values, terminal_values

__all__ = [
    "Evaluator", "UniformEvaluator", "RolloutEvaluator", "GreedyValueEvaluator",
    "ZeroEncoder", "state_encoder", "LeafRef", "greedy_action", "rollout_values",
]

_COST_SUM: Tuple[int, ...] = tuple(sum(c) for c in CARD_COST)


class LeafRef(NamedTuple):
    """The "observation" produced by :func:`state_encoder`."""

    state: E.GameState
    seat: int


def state_encoder(state: E.GameState, seat: int) -> LeafRef:
    """encode_fn for the NN-free evaluators: hand them the position itself."""
    return LeafRef(state, seat)


class ZeroEncoder:
    """Trivial numeric encoder for tests: ``float32[dim]`` of zeros."""

    def __init__(self, dim: int = 8):
        self.dim = int(dim)
        self._zeros = np.zeros(self.dim, dtype=np.float32)

    def __call__(self, state: E.GameState, seat: int) -> np.ndarray:
        return self._zeros.copy()


class Evaluator(Protocol):
    def evaluate(self, obs: Any, mask: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:      # pragma: no cover
        ...


def _uniform_priors(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 1:
        m = m[None]
    s = m.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return m / s


# ── the 1-ply greedy heuristic (shared by GreedyBot and the rollouts) ─────

def _shortfall(player: E.PlayerState, card_id: int, out: List[int]) -> int:
    """Per-colour tokens still missing for ``card_id`` (gold not counted)."""
    out[0] = out[1] = out[2] = out[3] = out[4] = 0
    d = player.discount
    g = player.gems
    total = 0
    for i, amount in CARD_COST_NZ[card_id]:
        need = amount - d[i] - g[i]
        if need > 0:
            out[i] = need
            total += need
    return total


def greedy_action(state: E.GameState, mask: Optional[Sequence[bool]] = None
                  ) -> Optional[int]:
    """One-ply greedy move (``AI_DESIGN`` §1.7), or ``None`` when stuck.

    Order: pending noble choice → buy the affordable card with the most points
    (ties: a reserved card first, then the cheaper printed cost) → take gems
    that shrink the cheapest attractive card's shortfall the most → reserve the
    best board card → whatever single action is left.  The returned action is
    always in ``mask``.
    """
    if mask is None:
        mask = E.legal_mask(state)
    player = state.players[state.current_player]

    # 1. A pending multi-noble choice is the only thing the engine accepts.
    for a in range(CHOOSE_TILE_START, NUM_ACTIONS):
        if mask[a]:
            return a

    # 2. Buy: most points, reserved first, then the cheaper card.
    best_buy = None
    best_key = None
    reserved = player.reserved
    for slot in range(len(reserved)):
        a = BUY_RESERVED_START + slot
        if a < CHOOSE_TILE_START and mask[a]:
            cid = reserved[slot]
            key = (CARD_POINTS[cid], 1, -_COST_SUM[cid])
            if best_key is None or key > best_key:
                best_key, best_buy = key, a
    board = state.board
    for t in range(3):
        row = board[t]
        base = BUY_BOARD_START + t * MAX_BOARD_SLOTS
        for s in range(min(len(row), MAX_BOARD_SLOTS)):
            a = base + s
            if mask[a]:
                cid = row[s]
                key = (CARD_POINTS[cid], 0, -_COST_SUM[cid])
                if best_key is None or key > best_key:
                    best_key, best_buy = key, a
    if best_buy is not None:
        return best_buy

    # 3. Take gems.  Target = the attractive card closest to affordable.
    if any(mask[a] for a in range(NUM_TAKE_ACTIONS)):
        need = [0, 0, 0, 0, 0]
        want = [0, 0, 0, 0, 0]          # how many cards want each colour
        target_need = None
        target_key = None
        gold = player.gems[5]
        tmp = [0, 0, 0, 0, 0]
        candidates = [(cid, True) for cid in reserved]
        for t in range(3):
            for cid in board[t][:MAX_BOARD_SLOTS]:
                candidates.append((cid, False))
        for cid, is_res in candidates:
            total = _shortfall(player, cid, tmp)
            for c in range(5):
                if tmp[c]:
                    want[c] += 1
            eff = total - gold
            if eff < 0:
                eff = 0
            key = (eff, -CARD_POINTS[cid], _COST_SUM[cid])
            if target_key is None or key < target_key:
                target_key = key
                target_need = tmp[:]
        if target_need is None:
            target_need = need

        best_take = None
        best_score = None
        for a in range(NUM_TAKE_ACTIONS):
            if not mask[a]:
                continue
            pattern = TAKE_PATTERNS[a]
            gain = 0
            spread = 0
            taken = [0, 0, 0, 0, 0]
            for c in pattern:
                taken[c] += 1
                if taken[c] <= target_need[c]:
                    gain += 1
                spread += want[c]
            score = (gain, spread, len(pattern))
            if best_score is None or score > best_score:
                best_score, best_take = score, a
        if best_take is not None:
            return best_take

    # 4. Reserve the best board card (most points, then cheapest).
    best_res = None
    best_key = None
    for t in range(3):
        row = board[t]
        base = RESERVE_BOARD_START + t * MAX_BOARD_SLOTS
        for s in range(min(len(row), MAX_BOARD_SLOTS)):
            a = base + s
            if mask[a]:
                cid = row[s]
                key = (CARD_POINTS[cid], -_COST_SUM[cid])
                if best_key is None or key > best_key:
                    best_key, best_res = key, a
    if best_res is not None:
        return best_res

    # 5. Anything left (a blind deck reserve), else stuck.
    for a in range(NUM_ACTIONS):
        if mask[a]:
            return a
    return None


def rollout_values(state: E.GameState, policy: str = "greedy",
                   max_plies: int = 60,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Play ``state`` out and return the value vector in ABSOLUTE seat order.

    Stuck seats resign (the variant has no pass); a rollout that runs past
    ``max_plies`` is scored by :func:`~.mcts.standings_values`.
    """
    s = state.clone()
    greedy = policy == "greedy"
    plies = 0
    while plies < max_plies and s.phase == E.PHASE_PLAYING:
        mask = E.legal_mask(s)
        if greedy:
            a = greedy_action(s, mask)
        else:
            legal = [i for i, v in enumerate(mask) if v]
            a = (legal[int(rng.integers(len(legal)))] if legal else None)
        if a is None:
            E.resign(s, s.current_player)
        else:
            E.apply(s, a)
        plies += 1
    return terminal_values(s) if s.is_over() else standings_values(s)


# ── evaluators ────────────────────────────────────────────────────────────

class UniformEvaluator:
    """Uniform priors over the legal actions, zero values.  The cheapest one."""

    name = "uniform"

    def evaluate(self, obs: Any, mask: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        priors = _uniform_priors(mask)
        return priors, np.zeros((priors.shape[0], 4), dtype=np.float32)


class RolloutEvaluator:
    """NN-free value: play the leaf out with a cheap policy (§1.6 anchor bot).

    Pair with :func:`state_encoder` — ``obs`` must be a sequence of
    :class:`LeafRef` (or ``(state, seat)`` pairs).
    """

    def __init__(self, policy: str = "greedy", max_plies: int = 60,
                 rng: Optional[np.random.Generator] = None):
        if policy not in ("greedy", "random"):
            raise ValueError("policy must be 'greedy' or 'random'")
        self.policy = policy
        self.max_plies = int(max_plies)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.name = f"rollout:{policy}"

    def evaluate(self, obs: Any, mask: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        priors = _uniform_priors(mask)
        n = priors.shape[0]
        values = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            state, seat = obs[i]
            z = rollout_values(state, self.policy, self.max_plies, self.rng)
            values[i] = seat_relative(z, seat)
        return priors, values


class GreedyValueEvaluator:
    """Static 1-ply heuristic value + a buy-biased prior.  No rollout.

    "Engine strength" is points plus a discounted count of the tableau (a card
    is worth roughly half a point of future tempo) plus a small token term;
    seats are compared to the best opponent (INDIVIDUAL) or side progress
    towards the mode's own threshold (TEAM / ONE_V_TWO).
    """

    name = "greedy-value"

    def __init__(self, card_weight: float = 0.45, gem_weight: float = 0.05,
                 scale: float = 4.0, buy_bias: float = 3.0,
                 reserve_bias: float = 0.5):
        self.card_weight = card_weight
        self.gem_weight = gem_weight
        self.scale = scale
        self.buy_bias = buy_bias
        self.reserve_bias = reserve_bias

    # -- value ---------------------------------------------------------
    def _power(self, state: E.GameState) -> List[float]:
        out = []
        for i, p in enumerate(state.players):
            if i in state.resigned:
                out.append(-1e6)
                continue
            out.append(p.score + self.card_weight * len(p.cards)
                       + self.gem_weight * p.total_gems())
        return out

    def value(self, state: E.GameState) -> np.ndarray:
        """Value vector for a position, ABSOLUTE seat order."""
        if state.is_over():
            return terminal_values(state)
        z = np.zeros(4, dtype=np.float32)
        power = self._power(state)
        n = state.num_players
        if state.mode == E.MODE_INDIVIDUAL:
            for i in range(n):
                others = [power[j] for j in range(n) if j != i]
                z[i] = np.tanh((power[i] - max(others)) / self.scale)
            return z
        thresholds = {0: 15.0, 1: 34.0} if state.mode == E.MODE_ONE_V_TWO \
            else {0: 30.0, 1: 30.0}
        side = {0: 0.0, 1: 0.0}
        for i, p in enumerate(state.players):
            if p.team_id in side and i not in state.resigned:
                side[p.team_id] += power[i]
        prog = {t: side[t] / thresholds[t] for t in (0, 1)}
        diff = np.tanh((prog[0] - prog[1]) * 2.0)
        for i, p in enumerate(state.players):
            z[i] = diff if p.team_id == 0 else -diff
        return z

    # -- prior ---------------------------------------------------------
    def _prior(self, mask: np.ndarray) -> np.ndarray:
        w = np.asarray(mask, dtype=np.float32).copy()
        w[BUY_BOARD_START:CHOOSE_TILE_START] *= self.buy_bias
        w[RESERVE_BOARD_START:BUY_BOARD_START] *= self.reserve_bias
        s = w.sum()
        if s <= 0:
            return _uniform_priors(mask)[0]
        return w / s

    def evaluate(self, obs: Any, mask: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 1:
            m = m[None]
        n = m.shape[0]
        priors = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
        values = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            state, seat = obs[i]
            priors[i] = self._prior(m[i])
            values[i] = seat_relative(self.value(state), seat)
        return priors, values
