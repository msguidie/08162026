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

Two things an actor must NOT compute from its own state:

* **the curriculum phase.**  It is a function of RUN-GLOBAL finished games.
  An actor's own counter restarts at 0 on every resume and every restart, so
  deriving the phase from it rewound the whole node to phase 0 (2p at reduced
  simulations) after every link of the PBS chain.  The trainer passes
  ``games_offset`` at spawn and republishes ``run_dir/progress.json``; the
  actor re-reads it on its weight-refresh cadence.
* **its RNG seeds and game ids.**  Seeded from ``(seed, actor_id)`` alone,
  every resume replayed the same deals in the same order and reissued the same
  game ids -- so a resumed run trained on duplicates of the data it already
  had, under ids that collided with it.  The trainer passes a per-launch
  ``instance`` nonce (a counter persisted in ``trainer_state.pt`` and bumped on
  every restore, plus a spawn sequence) which is mixed into both, along with
  the actor's own pid.

Failure policy: an actor never swallows an exception.  It prints the traceback
with its id and re-raises, the process dies, and the orchestrator restarts it
and counts the restart.  SIGINT/SIGTERM are the exception: they set a flag, the
wave in flight finishes, its records ship, and the process exits 0.
"""

from __future__ import annotations

import dataclasses
import json
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

__all__ = ["GameSlot", "Actor", "actor_main", "make_game_id",
           "read_progress"]

#: Game id layout, 62 bits so it is always a positive int64::
#:
#:     [ instance : 20 ][ actor_id : 10 ][ game index : 32 ]
#:
#: ``instance`` is the per-launch nonce, which is what makes ids issued after a
#: resume (or after an actor restart) disjoint from the ones already in the
#: replay buffer.  Records are grouped by game id downstream, so a collision
#: silently merges two different games.
_GAME_ID_INSTANCE_BITS = 20
_GAME_ID_ACTOR_BITS = 10
_GAME_ID_INDEX_BITS = 32


def make_game_id(instance: int, actor_id: int, index: int) -> int:
    """Unique-per-launch game id (see the bit layout above)."""
    return (((int(instance) & ((1 << _GAME_ID_INSTANCE_BITS) - 1))
             << (_GAME_ID_ACTOR_BITS + _GAME_ID_INDEX_BITS))
            | ((int(actor_id) & ((1 << _GAME_ID_ACTOR_BITS) - 1))
               << _GAME_ID_INDEX_BITS)
            | (int(index) & ((1 << _GAME_ID_INDEX_BITS) - 1)))


def read_progress(path: str) -> Optional[Dict[str, Any]]:
    """``run_dir/progress.json``, or None if it is absent or half-written.

    Written by the learner with a temp file and an atomic rename, so a partial
    read should be impossible -- but a shared filesystem can still hand out an
    empty file, and a curriculum phase is not worth crashing an actor over.
    """
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None

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
                 request_q=None, response_q=None, instance: int = 0,
                 games_offset: int = 0) -> None:
        self.cfg = cfg
        self.actor_id = int(actor_id)
        self.out_queue = out_queue
        self.stats_queue = stats_queue
        self.stop_event = stop_event
        #: per-launch nonce from the orchestrator (resume counter + spawn seq)
        self.instance = int(instance)
        # ...mixed with this process's pid, so two actors that somehow share an
        # instance (a state file restored twice by hand, say) still deal
        # different games.  Self-play data is deliberately NOT reproducible
        # across launches: reproducing it is exactly the bug.
        self.nonce = (self.instance * 1_000_003 + os.getpid()) & 0x7FFF_FFFF
        self.rng = np.random.default_rng(
            [cfg.seed, self.actor_id, self.nonce])
        self.py_rng = random.Random(
            f"{cfg.seed}|{self.actor_id}|{self.nonce}")
        #: Run-global finished games, for the curriculum.  ``games_offset`` is
        #: the trainer's restored count at spawn; it is refreshed from
        #: ``progress.json`` and only ever moves forward.
        self._games_base = int(games_offset)
        self._games_at_sync = 0
        self._last_progress = 0.0
        self._stop_flag = False

        sp = cfg.selfplay
        self.G = int(sp.games_per_actor)
        self.slots = [GameSlot(index=i) for i in range(self.G)]
        self._obs = np.zeros((max(self.G, 8), OBS_DIM), dtype=np.float32)
        self._mask = np.zeros((max(self.G, 8), NUM_ACTIONS), dtype=bool)

        self.model = SplendorNet(cfg.net)
        if cfg.inference.mode == "server":
            if request_q is None or response_q is None:
                raise ValueError("server inference needs request/response queues")
            # Request ids start in this launch's own block: a restarted actor
            # inherits its predecessor's response queue and must never mistake
            # a reply meant for the dead process for its own.
            self.evaluator: Any = RemoteEvaluator(
                actor_id, request_q, response_q,
                start_id=RemoteEvaluator.start_id_for(self.nonce))
        else:
            self.evaluator = LocalEvaluator(
                self.model, cfg.latest_weights, device="cpu",
                refresh_s=sp.weight_refresh_s)
            self.evaluator.refresh(force=True)
        #: checkpoint path -> NetEvaluator, small LRU (§1.8 opponent mixing)
        self._historical: Dict[str, Any] = {}
        self._historical_order: List[str] = []          # LRU, oldest first
        self._pool_cache: List[str] = []
        self._pool_cache_t = -1e18
        self._pool_failed: set = set()
        self.historical_loads = 0

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
        self.mode_games: Dict[str, int] = {}
        self.mode_truncations: Dict[str, int] = {}
        self.t0 = time.perf_counter()
        self._last_stats = self.t0
        self._last_refresh = 0.0
        self._pending_games = 0
        self._pending_records: List[np.ndarray] = []

    # -- setup helpers ---------------------------------------------------
    def global_games(self) -> int:
        """Run-global finished games, as far as this actor knows.

        ``base`` is the last figure read from the trainer (at spawn, then from
        ``progress.json``); the extrapolation covers the interval since, on the
        assumption that the other actors are producing at the same rate.  It is
        an estimate on purpose -- the phase boundaries are at 50k/300k/1M games,
        so being a few hundred games stale is irrelevant, while being a whole
        resume stale (the bug this replaces) put the node back in phase 0.
        """
        produced = max(0, self.games_started - self._games_at_sync)
        return int(self._games_base + produced * max(1, self.cfg.selfplay.actors))

    def _phase(self):
        return self.cfg.phase_for(self.global_games())

    def _sync_progress(self, force: bool = False) -> bool:
        """Re-read ``run_dir/progress.json`` (cheap, and only every N s)."""
        now = time.perf_counter()
        if not force and now - self._last_progress < \
                self.cfg.selfplay.progress_refresh_s:
            return False
        self._last_progress = now
        data = read_progress(self.cfg.progress_path)
        if not data:
            return False
        try:
            games = int(data.get("games_done", 0))
        except (TypeError, ValueError):                     # pragma: no cover
            return False
        # Never move the curriculum backwards: progress.json is written on a
        # timer and can lag this actor's own extrapolation.
        self._games_base = max(games, self.global_games())
        self._games_at_sync = self.games_started
        return True

    def _sample_mode(self) -> str:
        names, weights = normalise_mixture(self._phase().mixture)
        return str(self.rng.choice(names, p=weights))

    def _search_cfg(self, full: bool) -> SearchConfig:
        cfg = self.cfg
        base = cfg.search_full if full else cfg.search_fast
        phase = self._phase()
        sims = phase.sims_full if full else phase.sims_fast
        if sims is None:
            return base
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
                # Reuse a checkpoint this process already holds whenever we
                # can: a 12.6M-parameter net costs ~200 ms to load and runs on
                # the actor's own core, so a cache miss stalls every one of
                # this actor's `games_per_actor` games for the load.  With a
                # pool of 16 and a cache of 3 an unbiased draw missed on ~80%
                # of mixed games; `historical_reuse_prob` makes the miss rate
                # a knob instead of an accident, and the games that do share a
                # checkpoint are then batched together in `_run_searches`.
                path = self._pick_historical(pool)
                if path is None:                    # unloadable: fall back
                    controllers[int(seat)] = _ANCHOR
                else:
                    controllers[int(seat)] = path
        return controllers, any(c != _CURRENT for c in controllers)

    def _pick_historical(self, pool: List[str]) -> Optional[str]:
        """A historical checkpoint for one seat, loaded and ready.

        Returns ``None`` when nothing in the pool could be loaded (a
        checkpoint pruned by the retention policy between the listing and the
        load, say) so the caller can fall back to the anchor rather than
        killing the actor over an opponent choice.
        """
        cached = [p for p in self._historical_order if p in pool]
        prefer_cached = bool(cached) and (
            self.rng.random() < float(self.cfg.selfplay.historical_reuse_prob))
        order = ([str(self.rng.choice(cached))] if prefer_cached
                 else [str(self.rng.choice(pool))])
        # One fallback attempt from the cache, then give up on this seat.
        if not prefer_cached and cached:
            order.append(str(self.rng.choice(cached)))
        for path in order:
            if self._historical_evaluator(path) is not None:
                return path
        return None

    def _historical_pool(self) -> List[str]:
        """The last N generation checkpoints, from a cached listing.

        The listing changes once per generation (minutes) and this is called
        once per mixed game, so it is cached for
        ``selfplay.historical_pool_refresh_s``: an ``os.listdir`` of the
        directory the learner is writing into, from 56 actors, is pure
        overhead on a shared filesystem.
        """
        size = self.cfg.selfplay.historical_pool_size
        if size <= 0:
            return []
        now = time.perf_counter()
        if now - self._pool_cache_t < self.cfg.selfplay.historical_pool_refresh_s:
            return self._pool_cache
        self._pool_cache_t = now
        try:
            names = sorted(f for f in os.listdir(self.cfg.checkpoints_dir)
                           if f.startswith("gen_") and f.endswith(".pt"))
        except OSError:
            self._pool_cache = []
            return self._pool_cache
        pool = [os.path.join(self.cfg.checkpoints_dir, f) for f in names[-size:]]
        # A checkpoint that reappears (it cannot, but be explicit) gets a
        # second chance; one that is gone stops being remembered as broken.
        self._pool_failed &= set(pool)
        self._pool_cache = [p for p in pool if p not in self._pool_failed]
        return self._pool_cache

    def _historical_evaluator(self, path: str):
        """The NetEvaluator for ``path``, loading it if need be (LRU).

        Returns ``None`` if the checkpoint cannot be read.  That is not
        hypothetical: the trainer prunes old ``gen_XXXX.pt`` files, so a path
        from a cached listing can disappear under an actor, and losing an
        opponent is not a reason to lose the actor's games.
        """
        ev = self._historical.get(path)
        if ev is not None:
            # Touch: this is an LRU, and the whole point is that a checkpoint
            # in use by live games is not the one evicted.
            if self._historical_order[-1] != path:
                self._historical_order.remove(path)
                self._historical_order.append(path)
            return ev
        from ..model import NetEvaluator

        try:
            model, _ckpt = load_checkpoint(path, map_location="cpu")
        except Exception as exc:
            # Every failure mode, on purpose.  A checkpoint half-written by the
            # learner raises UnpicklingError, a pruned one OSError, a truncated
            # one RuntimeError, one from another encoder RuntimeError again --
            # and not one of them is a reason to kill an actor and its games
            # over the choice of an OPPONENT.  (The encoder gate that does
            # matter is on weights/latest.pt, where WeightWatcher enforces it.)
            if path not in self._pool_failed:
                from .inference import is_version_gate

                extra = (" -- the opponent pool is from another encoder"
                         if is_version_gate(exc) else "")
                print(f"[actor {self.actor_id}] historical opponent "
                      f"{os.path.basename(path)} unavailable "
                      f"({type(exc).__name__}: {exc}){extra}; dropping it from "
                      f"the pool", flush=True)
            self._pool_failed.add(path)
            self._pool_cache = [p for p in self._pool_cache if p != path]
            return None
        ev = NetEvaluator(model.eval(), "cpu")
        self.historical_loads += 1
        self._historical[path] = ev
        self._historical_order.append(path)
        while len(self._historical_order) > max(1, self.cfg.selfplay.historical_cache):
            drop = self._historical_order.pop(0)
            self._historical.pop(drop, None)
        return ev

    def _evaluator_for(self, key: str):
        if key == _CURRENT:
            return self.evaluator
        ev = self._historical_evaluator(key)
        # The checkpoint went away mid-game (pruned, evicted and then
        # unreadable).  The seat is not recorded either way, so finishing the
        # game against the current net beats killing the actor.
        return ev if ev is not None else self.evaluator

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
        slot.game_id = make_game_id(self.instance, self.actor_id,
                                    self.games_started)
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
        self.mode_games[slot.mode_name] = self.mode_games.get(slot.mode_name, 0) + 1
        if truncated:
            self.mode_truncations[slot.mode_name] = \
                self.mode_truncations.get(slot.mode_name, 0) + 1
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
        """Generation of the weights that produced this record's search.

        ``inproc`` reads it off the local weight watcher; ``server`` reads it
        off the last response (the server stamps every reply with the
        generation it served it from).  Before that, every record written in
        server mode carried generation 0 and the replay window's provenance
        was a lie.
        """
        return int(getattr(self.evaluator, "generation", 0) or 0)

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
            "eval_stale": int(getattr(self.evaluator, "stale", 0)),
            "weight_reloads": int(getattr(getattr(self.evaluator, "watcher", None),
                                          "reloads", 0)),
            "weight_generation": self._weight_generation(),
            "historical_loads": self.historical_loads,
            "global_games": self.global_games(),
            "phase_sims_full": self._phase().sims_full,
            "mode_plies": {k: float(np.mean(v)) for k, v in self.mode_plies.items()},
            # Lifetime counts, not "how many are in the plies window": the
            # trainer sums these across actors for the per-mode game counts.
            "mode_games": dict(self.mode_games),
            "mode_truncations": dict(self.mode_truncations),
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
        # Same cadence: the run-global game counter the curriculum reads.
        self._sync_progress()

    # -- main loop -------------------------------------------------------
    def stop(self) -> None:
        """Ask the loop to finish the wave in flight and ship (signal-safe)."""
        self._stop_flag = True

    def stopping(self) -> bool:
        return bool(self._stop_flag) or bool(
            self.stop_event is not None and self.stop_event.is_set())

    def run(self, max_waves: Optional[int] = None) -> None:
        self._sync_progress(force=True)
        for slot in self.slots:
            self._start_game(slot)
        waves = 0
        while True:
            # Checked between waves, never inside one: a wave that is
            # abandoned mid-search loses every game it has finished but not
            # yet shipped, which is precisely what `timeout --signal=INT` used
            # to cost this run.
            if self.stopping():
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
               max_waves: Optional[int] = None, instance: int = 0,
               games_offset: int = 0) -> None:
    """Process entry point.  Logs and re-raises — never swallows.

    SIGINT/SIGTERM are not an error: ``timeout --signal=INT`` signals the whole
    process group, so the actor sets its stop flag, finishes the wave in
    flight, ships its finished games and exits 0.
    """
    from .inference import install_stop_handlers

    configure_process(cfg.torch_threads,
                      seed=(cfg.seed * 31 + actor_id) * 1_000_003 + instance)
    actor = None
    try:
        actor = Actor(cfg, actor_id, out_queue, stats_queue, stop_event,
                      request_q=request_q, response_q=response_q,
                      instance=instance, games_offset=games_offset)

        def _stop(signum):
            print(f"[actor {actor_id}] signal {signum}: finishing the wave",
                  flush=True)
            actor.stop()

        install_stop_handlers(_stop)
        actor.run(max_waves=max_waves)
    except Exception:
        print(f"[actor {actor_id}] died:\n{traceback.format_exc()}", flush=True)
        raise
