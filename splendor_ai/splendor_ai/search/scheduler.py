"""Lockstep leaf batching across G independent trees (``AI_DESIGN`` §1.6).

Each step gathers at most one leaf per tree, issues a **single**
``evaluator.evaluate(obs[B], mask[B])`` call for the whole batch and backs the
results up.  Because every tree contributes at most one pending leaf, no
virtual loss is needed.  Simulations that ended at a terminal (or the depth
cap) inside ``select_leaf`` need no evaluation and simply do not join the batch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..rules.engine import GameState
from .mcts import MCTS, Leaf, SearchConfig, SearchResult

__all__ = ["SearchSlot", "Scheduler"]


@dataclass
class SearchSlot:
    """One tree plus the root it is searching."""

    tree: MCTS
    state: GameState
    seat: int
    sims: Optional[int] = None       # defaults to ``tree.cfg.sims``
    done: int = 0

    @property
    def remaining(self) -> int:
        budget = self.tree.cfg.sims if self.sims is None else self.sims
        return max(0, budget - self.done)

    def result(self) -> SearchResult:
        return self.tree.result()


class Scheduler:
    """Step G trees in lockstep, one batched evaluator call per step."""

    def __init__(self, trees: Sequence[Any], evaluator, encode_fn):
        self.slots: List[SearchSlot] = [
            s if isinstance(s, SearchSlot) else SearchSlot(*s) for s in trees
        ]
        self.evaluator = evaluator
        self.encode_fn = encode_fn
        self.stats: Dict[str, Any] = {
            "steps": 0, "sims": 0, "evaluated": 0, "batches": 0,
            "batch_total": 0, "seconds": 0.0,
        }

    # -- one lockstep round ---------------------------------------------
    def step(self) -> int:
        """Advance every unfinished tree by one simulation.  Returns the
        number of leaves that actually needed the evaluator."""
        pending: List[tuple] = []
        for slot in self.slots:
            if slot.remaining <= 0:
                continue
            leaf = slot.tree.select_leaf(slot.state, slot.seat)
            slot.done += 1
            self.stats["sims"] += 1
            if leaf is not None:
                leaf.obs = self.encode_fn(leaf.state, leaf.seat)
                pending.append((slot, leaf))
        self.stats["steps"] += 1
        if not pending:
            return 0

        obs = [leaf.obs for _, leaf in pending]
        batch = np.stack(obs) if isinstance(obs[0], np.ndarray) else obs
        masks = np.stack([leaf.mask for _, leaf in pending])
        priors, values = self.evaluator.evaluate(batch, masks)
        for i, (slot, leaf) in enumerate(pending):
            slot.tree.backup(leaf.token, priors[i], values[i])
        self.stats["evaluated"] += len(pending)
        self.stats["batches"] += 1
        self.stats["batch_total"] += len(pending)
        return len(pending)

    # -- drive to completion ---------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Step until every slot has spent its simulation budget."""
        t0 = time.perf_counter()
        while any(slot.remaining > 0 for slot in self.slots):
            self.step()
        self.stats["seconds"] += time.perf_counter() - t0
        return self.report()

    def results(self) -> List[SearchResult]:
        return [slot.result() for slot in self.slots]

    def report(self) -> Dict[str, Any]:
        st = dict(self.stats)
        secs = st["seconds"]
        st["trees"] = len(self.slots)
        st["sims_per_s"] = (st["sims"] / secs) if secs > 0 else float("inf")
        st["evals_per_s"] = (st["evaluated"] / secs) if secs > 0 else float("inf")
        st["mean_batch"] = (st["batch_total"] / st["batches"]
                            if st["batches"] else 0.0)
        return st
