"""Bots and a self-contained game driver (``docs/AI_DESIGN.md`` §1.7).

Every bot implements::

    action = bot.act(state, seat, rng)      # -> int in 0..64, or None if stuck

``rng`` is a ``numpy.random.Generator`` (``random.Random`` also works for the
non-search bots).  A bot must never return an action outside
``engine.legal_mask(state)``; the only legitimate ``None`` is a stuck seat,
which the caller turns into ``engine.resign`` — the variant has no pass.

:func:`play_game` drives a whole game: stuck seats resign, a pending noble
choice is just the same seat acting again (``CHOOSE_TILE`` is a same-player
sub-decision), and a game that outlives ``max_plies`` is truncated and scored
by current standings.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from .rules import engine as E
from .rules.actions import NUM_ACTIONS
from .search.evaluators import (
    GreedyValueEvaluator, RolloutEvaluator, UniformEvaluator, greedy_action,
    state_encoder,
)
from .search.mcts import (
    MCTS, SearchConfig, SearchResult, run_search, standings_values,
    terminal_values,
)

__all__ = [
    "Bot", "RandomBot", "GreedyBot", "MctsBot", "play_game", "legal_actions_of",
]


class Bot(Protocol):
    name: str

    def act(self, state: E.GameState, seat: int, rng) -> Optional[int]:
        ...                                                # pragma: no cover


def _randint(rng, n: int) -> int:
    """Works with numpy Generators and ``random.Random`` alike."""
    integers = getattr(rng, "integers", None)
    if integers is not None:
        return int(integers(n))
    return int(rng.randrange(n))


def legal_actions_of(state: E.GameState) -> List[int]:
    mask = E.legal_mask(state)
    return [i for i, v in enumerate(mask) if v]


class RandomBot:
    """Uniform over the legal actions."""

    def __init__(self, name: str = "random"):
        self.name = name

    def act(self, state: E.GameState, seat: int, rng) -> Optional[int]:
        mask = E.legal_mask(state)
        legal = [i for i, v in enumerate(mask) if v]
        if not legal:
            return None
        return legal[_randint(rng, len(legal))]


class GreedyBot:
    """One-ply heuristic (see :func:`~.search.evaluators.greedy_action`)."""

    def __init__(self, name: str = "greedy"):
        self.name = name

    def act(self, state: E.GameState, seat: int, rng=None) -> Optional[int]:
        mask = E.legal_mask(state)
        a = greedy_action(state, mask)
        if a is None:
            return None
        if not mask[a]:                                    # pragma: no cover
            raise AssertionError(f"GreedyBot produced illegal action {a}")
        return a


class MctsBot:
    """PUCT/Gumbel search with any evaluator (§1.6).

    ``encode_fn`` must match the evaluator: :func:`~.search.evaluators
    .state_encoder` for the NN-free ones, the real encoder for a net.
    """

    def __init__(self, cfg: SearchConfig, evaluator=None, encode_fn=None,
                 name: Optional[str] = None):
        self.cfg = cfg
        self.evaluator = evaluator if evaluator is not None else RolloutEvaluator()
        self.encode_fn = encode_fn if encode_fn is not None else state_encoder
        self.name = name or f"mcts{cfg.sims}:{getattr(self.evaluator, 'name', '?')}"
        self.last_result: Optional[SearchResult] = None

    def act(self, state: E.GameState, seat: int, rng) -> Optional[int]:
        if E.is_stuck(state):
            return None
        if not hasattr(rng, "integers"):                   # pragma: no cover
            rng = np.random.default_rng(rng.randrange(1 << 62))
        res = run_search(state, seat, self.evaluator, self.encode_fn,
                         self.cfg, rng)
        self.last_result = res
        return int(res.action)

    def search(self, state: E.GameState, seat: int, rng) -> SearchResult:
        return run_search(state, seat, self.evaluator, self.encode_fn,
                          self.cfg, rng)


# ── game driver ───────────────────────────────────────────────────────────

def play_game(bots, mode: str = "INDIVIDUAL", num_players: int = 2,
              layout: Optional[str] = None, seed: int = 0,
              max_plies: int = 400) -> Dict[str, Any]:
    """Play one game and return the outcome.

    ``bots`` is one bot per seat (a single bot is used for every seat).
    ``seed`` seeds both the deal and the bots, so a pair of games that differ
    only in the seating is exactly paired.

    Returns ``{values, reason, plies, winners, scores, cards, resigned,
    stuck_resigns, truncated, actions, mode, num_players, layout, seed}``
    where ``values`` is the §1.2 value vector in ABSOLUTE seat order (current
    standings if the game was truncated).
    """
    if isinstance(bots, (list, tuple)):
        seats = list(bots)
        if len(seats) == 1:
            seats = seats * num_players
    else:
        seats = [bots] * num_players
    if len(seats) != num_players:
        raise ValueError(f"need {num_players} bots, got {len(seats)}")

    state = E.new_game(num_players, mode, layout, rng=random.Random(seed))
    rng = np.random.default_rng(seed)

    plies = 0
    stuck_resigns = 0
    actions: List[int] = []
    while state.phase == E.PHASE_PLAYING and plies < max_plies:
        seat = state.current_player
        if E.is_stuck(state):
            E.resign(state, seat)
            stuck_resigns += 1
            plies += 1
            continue
        action = seats[seat].act(state, seat, rng)
        if action is None:                                 # bot conceded
            E.resign(state, seat)
            stuck_resigns += 1
            plies += 1
            continue
        if not E.legal_mask(state)[action]:
            raise ValueError(
                f"bot {getattr(seats[seat], 'name', seat)} returned illegal "
                f"action {action} for seat {seat}")
        E.apply(state, action)
        actions.append(action)
        plies += 1

    truncated = state.phase == E.PHASE_PLAYING
    values = standings_values(state) if truncated else terminal_values(state)
    n = num_players
    best = float(np.max(values[:n]))
    winners = [i for i in range(n) if float(values[i]) == best and best > 0]
    if truncated:
        reason = "TRUNCATED"
    elif state.game_result is not None:
        reason = state.game_result.get("reason", "SCORE")
    elif state.resigned:
        reason = "FORFEIT"
    else:
        reason = "SCORE"

    return {
        "mode": mode, "num_players": num_players, "layout": layout,
        "seed": seed, "plies": plies, "reason": reason,
        "truncated": truncated,
        "values": [float(v) for v in values],
        "winners": winners,
        "winning_team_ids": (list(state.game_result["winningTeamIds"])
                             if state.game_result and
                             "winningTeamIds" in state.game_result else None),
        "scores": [p.score for p in state.players],
        "cards": [len(p.cards) for p in state.players],
        "resigned": list(state.resigned),
        "stuck_resigns": stuck_resigns,
        "actions": actions,
        "names": [getattr(b, "name", "?") for b in seats],
    }
