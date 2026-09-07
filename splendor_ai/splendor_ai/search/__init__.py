"""Search: PIMC determinization + open-loop PUCT/Gumbel MCTS.

``docs/AI_DESIGN.md`` §1.6 is the binding contract for everything in here.

Value-vector conventions (used by every module below)
-----------------------------------------------------
* a value vector is ``float32[4]`` — one entry per seat, entries ``>= n`` are 0;
* **absolute** order means index ``i`` is seat ``i``;
* **relative to seat s** means index ``j`` is absolute seat ``(j + s) % n``
  for an ``n``-seat game, with the padding entries left in place — §1.2 writes
  it as ``np.roll(z, -s)``, exact for a full table of four, and
  :func:`seat_relative` / :func:`seat_absolute` (identical to
  ``splendor_ai.values.seat_relative``) do the ``n``-aware version;
* :class:`Evaluator` implementations return values **relative to the leaf's
  acting seat** (index 0 = the seat to move at that leaf); the tree converts
  them to absolute order on backup, so backup is a plain accumulation.
"""

from .determinize import determinize, hidden_slots, unseen_pool, universe_rng
from .mcts import (
    MCTS, Leaf, SearchConfig, SearchResult, run_search, seat_absolute,
    seat_relative, standings_values, terminal_values,
)
from .evaluators import (
    Evaluator, GreedyValueEvaluator, LeafRef, RolloutEvaluator,
    UniformEvaluator, UniformRolloutEvaluator, ZeroEncoder, greedy_action,
    rollout_values, state_encoder,
)
from .scheduler import Scheduler, SearchSlot

__all__ = [
    "determinize", "unseen_pool", "hidden_slots", "universe_rng",
    "MCTS", "Leaf", "SearchConfig", "SearchResult", "run_search",
    "terminal_values", "standings_values", "seat_relative", "seat_absolute",
    "Evaluator", "UniformEvaluator", "RolloutEvaluator",
    "UniformRolloutEvaluator", "GreedyValueEvaluator", "ZeroEncoder",
    "state_encoder", "LeafRef", "greedy_action", "rollout_values",
    "Scheduler", "SearchSlot",
]
