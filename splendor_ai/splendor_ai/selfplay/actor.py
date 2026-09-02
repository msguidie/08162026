"""Self-play actor process (``docs/AI_DESIGN.md`` §1.8).

One actor runs ``games_per_actor`` games **in lockstep**: every game holds its
own :class:`~splendor_ai.search.mcts.MCTS`, one simulation is taken from each
tree per round, all the resulting leaves are encoded with a single
``encode_batch`` and scored with a single evaluator call, and the values are
backed up.  That is the whole reason the actor exists as a batching unit:
``encode`` costs ~85 us alone and ~19 us inside a batch, and one 24-row forward
costs about what one 1-row forward costs.

Per move (playout cap randomization, KataGo / judges.md):

* with probability ``pcr_full_prob`` the move gets ``search_full`` — Dirichlet
  noise at the root, forced playouts, policy-target pruning — and **is
  recorded**;
* otherwise it gets ``search_fast`` — no noise, fewer sims — and is **not
  recorded**.

Other contracts honoured here:

* moves are sampled from the visit distribution while
  ``state.turn_number < temperature_plies`` and are the argmax afterwards
  (the temperature lives inside :class:`MCTS`);
* a seat with no legal action resigns (the variant has no pass);
* a game that reaches ``max_plies`` is truncated and scored from the current
  standings with value weight 0.3 — a distinct, labelled outcome;
* ``mixed_game_frac`` of games give 1..n-1 seats to a frozen historical
  checkpoint or to the greedy anchor; **only current-net seats are recorded**;
* finished games are augmented over the C5 colour group and shipped as one raw
  buffer per game.

Failure policy: an actor never swallows an exception.  It prints the traceback
with its id and re-raises, the process dies, and the orchestrator restarts it
and counts the restart.
"""

from __future__ import annotations

import os
import random
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..encode import OBS_DIM, encode_batch
from ..model import NetConfig, SplendorNet, load_checkpoint
from ..rules import engine as E
from ..rules.actions import NUM_ACTIONS
from ..search.evaluators import greedy_action
from ..search.mcts import MCTS, SearchConfig
from ..values import standings_values, terminal_values
from . import configure_process
from .config import MODE_SPECS, RunConfig, normalise_mixture
from .inference import LocalEvaluator, RemoteEvaluator
from .sample import augment_many, empty, finish_game_records, make_record, \
    records_to_bytes

__all__ = ["GameSlot", "Actor", "actor_main"]

_ANCHOR = "greedy"
_CURRENT = "net"


@dataclass
class GameSlot:
    """One of the ``G`` games an actor plays at once."""

    index: int
    game_id: int = -1
    state: Optional[E.GameState] = None
    mode_name: str = "ind2"
    #: per seat: ``"net"``, ``"greedy"`` or a checkpoint path
    controllers: List[str] = field(default_factory=list)
    records: List[np.ndarray] = field(default_factory=list)
    plies: int = 0
    stuck_seats: List[bool] = field(default_factory=list)
    tree: Optional[MCTS] = None
    seat: int = 0
    sims: int = 0
    sims_done: int = 0
    recorded: bool = False
    eval_key: str = _CURRENT
    root_argmax: int = -1
    mixed: bool = False

    @property
    def searching(self) -> bool:
        return self.tree is not None and self.sims_done < self.sims


class Actor:
    """The self-play loop for one process."""

    def __init__(self, cfg: RunConfig, actor_id: int, out_queue,
                 stats_queue=None, stop_event=None,
                 request_q=None, response_q=None) -> None:
        self.cfg = cfg
        self.actor_id = int(actor_id)
        self.out_queue = out_queue
        self.stats_queue = stats_queue
        self.stop_event = stop_event
        self.rng = np.random.default_rng(cfg.seed * 7919 + actor_id)
        self.py_rng = random.Random(cfg.seed * 104729 + actor_id)

        sp = cfg.selfplay
        self.G = int(sp.games_per_actor)
        self.slots = [GameSlot(index=i) for i in range(self.G)]
        self._obs = np.zeros((max(self.G, 8), OBS_DIM), dtype=np.float32)
        self._mask = np.zeros((max(self.G, 8), NUM_ACTIONS), dtype=bool)

        self.model = SplendorNet(cfg.net)
        if cfg.inference.mode == "server":
            if request_q is None or response_q is None:
                raise ValueError("server inference needs request/response queues")
            self.evaluator: Any = RemoteEvaluator(actor_id, request_q, response_q)
        else:
            self.evaluator = LocalEvaluator(
                self.model, cfg.latest_weights, device="cpu",
                refresh_s=sp.weight_refresh_s)
            self.evaluator.refresh(force=True)
        #: checkpoint path -> NetEvaluator, small LRU (§1.8 opponent mixing)
        self._historical: Dict[str, Any] = {}
        self._historical_order: List[str] = []

        self.games_started = 0
        self.games_finished = 0
        self.moves = 0
        self.sims = 0
        self.records_made = 0
        self.stuck_resigns = 0
        self.truncations = 0
        self.disagree = 0
        self.disagree_total = 0
        self.mode_plies: Dict[str, List[int]] = {}
        self.t0 = time.perf_counter()
        self._last_stats = self.t0
        self._last_refresh = 0.0
        self._pending_games = 0
        self._pending_records: List[np.ndarray] = []

    # -- setup helpers ---------------------------------------------------
    def _sample_mode(self) -> str:
        phase = self.cfg.phase_for(self.games_started * self.cfg.selfplay.actors)
        names, weights = normalise_mixture(phase.mixture)
        return str(self.rng.choice(names, p=weights))

    def _search_cfg(self, full: bool) -> SearchConfig:
        cfg = self.cfg
        base = cfg.search_full if full else cfg.search_fast
        phase = cfg.phase_for(self.games_started * cfg.selfplay.actors)
        sims = phase.sims_full if full else phase.sims_fast
        if sims is None:
            return base
        import dataclasses

        return dataclasses.replace(base, sims=int(sims))

    def _pick_controllers(self, n: int) -> Tuple[List[str], bool]:
        """Seat -> controller for a new game (§1.8 opponent mixing)."""
        sp = self.cfg.selfplay
        controllers = [_CURRENT] * n
        if sp.mixed_game_frac <= 0 or self.rng.random() >= sp.mixed_game_frac:
            return controllers, False
        pool = self._historical_pool()
        kinds, weights = [], []
        for kind, w in sp.opponent_weights.items():
            if float(w) <= 0:
                continue
            if kind == "historical" and not pool:
                continue
            kinds.append(kind)
            weights.append(float(w))
        if not kinds:
            return controllers, False
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()
        n_opp = 1 if n <= 2 else int(self.rng.integers(1, min(2, n - 1) + 1))
        seats = self.rng.choice(n, size=n_opp, replace=False)
        for seat in seats:
            kind = str(self.rng.choice(kinds, p=weights))
            if kind == "latest":
                continue
            if kind == "anchor":
                controllers[int(seat)] = _ANCHOR
            else:
                controllers[int(seat)] = str(self.rng.choice(pool))
        return controllers, any(c != _CURRENT for c in controllers)

    def _historical_pool(self) -> List[str]:
        directory = self.cfg.checkpoints_dir
        size = self.cfg.selfplay.historical_pool_size
        if size <= 0:
            return []
        try:
            names = sorted(f for f in os.listdir(directory)
                           if f.startswith("gen_") and f.endswith(".pt"))
        except OSError:
            return []
        return [os.path.join(directory, f) for f in names[-size:]]

    def _historical_evaluator(self, path: str):
        ev = self._historical.get(path)
        if ev is not None:
            return ev
        from ..model import NetEvaluator

        model, _ckpt = load_checkpoint(path, map_location="cpu")
        ev = NetEvaluator(model.eval(), "cpu")
        self._historical[path] = ev
        self._historical_order.append(path)
        while len(self._historical_order) > max(1, self.cfg.selfplay.historical_cache):
            drop = self._historical_order.pop(0)
            self._historical.pop(drop, None)
        return ev

    def _evaluator_for(self, key: str):
        return self.evaluator if key == _CURRENT else self._historical_evaluator(key)

    # -- game lifecycle --------------------------------------------------
    def _start_game(self, slot: GameSlot) -> None:
        mode_name = self._sample_mode()
        n, mode, layout = MODE_SPECS[mode_name]
        state = E.new_game(n, mode, layout,
                           rng=random.Random(self.py_rng.getrandbits(63)))
        thr = self.cfg.selfplay.win_threshold
        if thr is not None:
            # Smoke-only shortcut (§4 "reduced-threshold smoke matrix"): a
            # per-state engine config override, never used for deployment.
            state.config = dict(state.config)
            state.config["winThreshold"] = int(thr)
        controllers, mixed = self._pick_controllers(n)
        slot.game_id = self.actor_id * 1_000_000_000 + self.games_started
        slot.state = state
        slot.mode_name = mode_name
        slot.controllers = controllers
        slot.mixed = mixed
        slot.records = []
        slot.plies = 0
        slot.stuck_seats = [False] * 4
        slot.tree = None
        self.games_started += 1

    def _finish_game(self, slot: GameSlot, truncated: bool) -> None:
        state = slot.state
        if truncated:
            z = standings_values(state)
            weight = float(self.cfg.selfplay.truncation_z_weight)
            self.truncations += 1
        else:
            z = terminal_values(state)
            weight = 1.0
        scores = [p.score for p in state.players] + [0] * (4 - state.num_players)
        stuck = list(slot.stuck_seats)
        finish_game_records(slot.records, z, weight, scores, stuck, slot.plies)
        self.mode_plies.setdefault(slot.mode_name, []).append(slot.plies)
        self.games_finished += 1
        if slot.records:
            packed = augment_many(slot.records,
                                 self.cfg.selfplay.augment_rotations)
            self.records_made += len(packed)
            self._pending_records.append(packed)
        self._pending_games += 1
        if self._pending_games >= max(1, self.cfg.selfplay.ship_every_games):
            self._ship()
        self._start_game(slot)

    def _ship(self) -> None:
        if self._pending_records:
            buf = np.concatenate(self._pending_records)
            payload = {"type": "records", "actor": self.actor_id,
                       "games": self._pending_games, "n": len(buf),
                       "buf": records_to_bytes(buf)}
        else:
            payload = {"type": "records", "actor": self.actor_id,
                       "games": self._pending_games, "n": 0, "buf": b""}
        self.out_queue.put(payload)
        self._pending_records = []
        self._pending_games = 0

    # -- one move --------------------------------------------------------
    def _open_move(self, slot: GameSlot) -> None:
        """Advance ``slot`` to a position where a tree is waiting, or act
        directly for the non-search controllers."""
        while True:
            state = slot.state
            if state.phase != E.PHASE_PLAYING:
                self._finish_game(slot, truncated=False)
                continue
            if slot.plies >= self.cfg.selfplay.max_plies:
                self._finish_game(slot, truncated=True)
                continue
            seat = state.current_player
            if E.is_stuck(state):
                E.resign(state, seat)
                slot.stuck_seats[seat] = True
                self.stuck_resigns += 1
                slot.plies += 1
                continue
            controller = slot.controllers[seat]
            if controller == _ANCHOR:
                action = greedy_action(state, E.legal_mask(state))
                if action is None:                          # pragma: no cover
                    E.resign(state, seat)
                    slot.stuck_seats[seat] = True
                    self.stuck_resigns += 1
                else:
                    E.apply(state, action)
                slot.plies += 1
                self.moves += 1
                continue
            full = (controller == _CURRENT
                    and self.rng.random() < self.cfg.selfplay.pcr_full_prob)
            search_cfg = self._search_cfg(full)
            import dataclasses

            # PCR: noise only on the recorded full searches, never on the
            # cheap ones (KataGo; judges.md "SEARCH BUDGETS").
            search_cfg = dataclasses.replace(
                search_cfg,
                noise=bool(full),
                temperature=self.cfg.selfplay.temperature,
                temperature_plies=self.cfg.selfplay.temperature_plies)
            slot.tree = MCTS(search_cfg,
                             np.random.default_rng(self.rng.integers(1 << 62)))
            slot.seat = seat
            slot.sims = int(search_cfg.sims)
            slot.sims_done = 0
            slot.recorded = bool(full and controller == _CURRENT)
            slot.eval_key = _CURRENT if controller == _CURRENT else controller
            slot.root_argmax = -1
            return

    def _root_priors(self, slots: Sequence[GameSlot]) -> None:
        """One batched forward over the roots of the recorded moves, to log
        the search-vs-policy argmax disagreement (§1.8 instrumentation)."""
        items = [s for s in slots if s.recorded and s.eval_key == _CURRENT]
        if not items:
            return
        b = len(items)
        if b > self._obs.shape[0]:
            self._obs = np.zeros((b, OBS_DIM), dtype=np.float32)
            self._mask = np.zeros((b, NUM_ACTIONS), dtype=bool)
        encode_batch([s.state for s in items], [s.seat for s in items],
                     out=self._obs[:b])
        for i, s in enumerate(items):
            self._mask[i] = np.asarray(E.legal_mask(s.state), dtype=bool)
        if not self._mask[:b].any(axis=1).all():            # pragma: no cover
            raise AssertionError("actor: root mask with no legal action")
        priors, _values = self.evaluator.evaluate(self._obs[:b], self._mask[:b])
        for i, s in enumerate(items):
            s.root_argmax = int(np.argmax(priors[i]))

    def _run_searches(self) -> None:
        """Lockstep simulation rounds until every open tree has spent its
        budget.  One ``encode_batch`` + one evaluator call per evaluator per
        round."""
        while True:
            pending: Dict[str, List[Tuple[GameSlot, Any]]] = {}
            active = False
            for slot in self.slots:
                if not slot.searching:
                    continue
                active = True
                leaf = slot.tree.select_leaf(slot.state, slot.seat)
                slot.sims_done += 1
                self.sims += 1
                if leaf is not None:
                    pending.setdefault(slot.eval_key, []).append((slot, leaf))
            if not active:
                return
            for key, items in pending.items():
                b = len(items)
                if b > self._obs.shape[0]:
                    self._obs = np.zeros((b, OBS_DIM), dtype=np.float32)
                    self._mask = np.zeros((b, NUM_ACTIONS), dtype=bool)
                encode_batch([leaf.state for _s, leaf in items],
                             [leaf.seat for _s, leaf in items],
                             out=self._obs[:b])
                for i, (_s, leaf) in enumerate(items):
                    self._mask[i] = leaf.mask
                if not self._mask[:b].any(axis=1).all():    # pragma: no cover
                    raise AssertionError("actor: leaf mask with no legal action")
                priors, values = self._evaluator_for(key).evaluate(
                    self._obs[:b], self._mask[:b])
                for i, (slot, leaf) in enumerate(items):
                    slot.tree.backup(leaf.token, priors[i], values[i])

    def _close_moves(self) -> None:
        for slot in self.slots:
            if slot.tree is None:
                continue
            result = slot.tree.result()
            state = slot.state
            mask = np.asarray(E.legal_mask(state), dtype=bool)
            if slot.recorded:
                target = np.asarray(result.policy_target, dtype=np.float32)
                rec = make_record(
                    state, slot.seat, target, mask, slot.mode_name,
                    slot.game_id, slot.plies,
                    generation=self._weight_generation(),
                    root_value=(result.root_value
                                if self.cfg.selfplay.store_root_value else None))
                slot.records.append(rec)
                if slot.root_argmax >= 0:
                    self.disagree_total += 1
                    if int(np.argmax(result.visits)) != slot.root_argmax:
                        self.disagree += 1
            action = int(result.action)
            if not mask[action]:                            # pragma: no cover
                raise AssertionError(
                    f"actor {self.actor_id}: search returned illegal action "
                    f"{action} for seat {slot.seat}")
            E.apply(state, action)
            slot.plies += 1
            self.moves += 1
            slot.tree = None

    def _weight_generation(self) -> int:
        watcher = getattr(self.evaluator, "watcher", None)
        return int(getattr(watcher, "generation", 0)) if watcher else 0

    # -- stats -----------------------------------------------------------
    def _maybe_stats(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_stats < self.cfg.selfplay.stats_every_s:
            return
        dt = max(1e-9, now - self._last_stats)
        elapsed = max(1e-9, now - self.t0)
        payload = {
            "type": "stats", "actor": self.actor_id,
            "elapsed_s": elapsed,
            "sims": self.sims, "moves": self.moves,
            "games": self.games_finished, "records": self.records_made,
            "sims_per_s": self.sims / elapsed,
            "moves_per_s": self.moves / elapsed,
            "games_per_s": self.games_finished / elapsed,
            "stuck_resigns": self.stuck_resigns,
            "truncations": self.truncations,
            "stuck_rate": self.stuck_resigns / max(1, self.moves),
            "truncation_rate": self.truncations / max(1, self.games_finished),
            "disagreement": self.disagree / max(1, self.disagree_total),
            "eval_calls": int(getattr(self.evaluator, "calls", 0)),
            "eval_rows": int(getattr(self.evaluator, "rows", 0)),
            "weight_reloads": int(getattr(getattr(self.evaluator, "watcher", None),
                                          "reloads", 0)),
            "weight_generation": self._weight_generation(),
            "mode_plies": {k: float(np.mean(v)) for k, v in self.mode_plies.items()},
            "mode_games": {k: len(v) for k, v in self.mode_plies.items()},
            "window_s": dt,
        }
        if self.stats_queue is not None:
            self.stats_queue.put(payload)
        self._last_stats = now
        self.mode_plies = {k: v[-64:] for k, v in self.mode_plies.items()}

    def _maybe_refresh(self) -> None:
        now = time.perf_counter()
        if now - self._last_refresh < self.cfg.selfplay.weight_refresh_s:
            return
        self._last_refresh = now
        self.evaluator.refresh()

    # -- main loop -------------------------------------------------------
    def run(self, max_waves: Optional[int] = None) -> None:
        for slot in self.slots:
            self._start_game(slot)
        waves = 0
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                break
            if max_waves is not None and waves >= max_waves:
                break
            self._maybe_refresh()
            for slot in self.slots:
                self._open_move(slot)
            self._root_priors(self.slots)
            self._run_searches()
            self._close_moves()
            self._maybe_stats()
            waves += 1
        self._ship()
        self._maybe_stats(force=True)


def actor_main(cfg: RunConfig, actor_id: int, out_queue, stats_queue=None,
               stop_event=None, request_q=None, response_q=None,
               max_waves: Optional[int] = None) -> None:
    """Process entry point.  Logs and re-raises — never swallows."""
    configure_process(cfg.torch_threads, seed=cfg.seed * 31 + actor_id)
    try:
        actor = Actor(cfg, actor_id, out_queue, stats_queue, stop_event,
                      request_q=request_q, response_q=response_q)
        actor.run(max_waves=max_waves)
    except Exception:
        print(f"[actor {actor_id}] died:\n{traceback.format_exc()}", flush=True)
        raise
