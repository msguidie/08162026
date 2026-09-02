"""Move selection with a fallback ladder (``docs/AI_DESIGN.md`` §1.9).

``MoveAgent.decide(payload)`` is the whole brain of the worker: it hydrates the
request, picks a move under a wall-clock budget and hands back the wire action
plus everything the move log wants to know.  The ladder is

===== ========================================================================
 level  policy
===== ========================================================================
search  net + PIMC MCTS, anytime under ``TIME_BUDGET_MS`` / ``HARD_BUDGET_MS``
policy  net policy argmax over the legal actions (no search)
greedy  :class:`splendor_ai.bots.GreedyBot` — the 1-ply heuristic, no net
 none   ``{"type": "NONE"}``: genuinely stuck, the server resigns the seat
===== ========================================================================

Each rung is tried in order and a rung that raises, times out or produces an
action that fails re-validation falls through to the next one, so a missing
checkpoint, a CUDA hiccup or a torch import error degrades the *strength* of
the bot and never its availability.  With no checkpoint at all the worker
starts on ``greedy`` and says so loudly — that is the intended way to test the
wiring before the first model exists.

Two safety nets sit on top of the ladder:

* **1-ply stuck filter** — an action that would leave *our own* next turn with
  no legal move at all (10 tokens, 3 reserved, nothing affordable: the variant
  has no pass, so the server would have to resign us) is skipped unless every
  candidate is like that.
* **re-validation** — the action finally chosen is checked against
  ``legal_mask`` on a *freshly hydrated* copy of the payload, and the wire
  message is built from that fresh state, so nothing the search did to its own
  copy can leak into what we send.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..rules import engine as E
from ..rules.actions import CHOOSE_TILE_START, NUM_ACTIONS
from .adapter import HydrationError, hydrate, payload_mode_key, to_wire
from .config import WorkerConfig

__all__ = ["MoveAgent", "Decision", "ModelHandle", "resolve_device",
           "self_stuck_after", "LEVELS", "SKEW_FLOOR_MS", "MIN_BUDGET_MS"]

#: Ladder rungs, strongest first.
LEVELS = ("search", "policy", "greedy", "none")

#: ``deadlineMs`` is an absolute epoch stamp taken on the SERVER's clock, and
#: the two machines are not synchronised (a home worker and a Render dyno can
#: sit seconds apart).  A request that has only just arrived cannot really have
#: less than this long to run — the server's own budget is 15 s — so a smaller
#: "remaining" is read as clock skew and the local budget is used instead.
SKEW_FLOOR_MS = 500.0

#: Floor for a genuinely short budget (the request queued behind another move).
MIN_BUDGET_MS = 50.0

_NONE_ACTION = {"type": "NONE"}


@dataclass
class Decision:
    """What the worker will send, and why."""

    action: Dict[str, Any]
    level: str
    action_index: int = -1
    sims: int = 0
    root_value: Optional[List[float]] = None
    policy: Optional[float] = None
    ms: float = 0.0
    mode: str = ""
    seat: int = -1
    kind: str = "MOVE"
    notes: List[str] = field(default_factory=list)

    @property
    def info(self) -> Dict[str, Any]:
        """The optional ``info`` block of ``ai_move_response``."""
        out: Dict[str, Any] = {"ms": round(self.ms, 1), "level": self.level}
        if self.sims:
            out["sims"] = int(self.sims)
        if self.root_value is not None:
            out["value"] = round(float(self.root_value[self.seat])
                                 if 0 <= self.seat < len(self.root_value)
                                 else float(self.root_value[0]), 4)
        if self.policy is not None:
            out["policy"] = round(float(self.policy), 4)
        return out


# ── device / model handling ───────────────────────────────────────────────

def resolve_device(spec: str = "auto") -> str:
    """``auto`` → ``cuda`` when torch sees one, otherwise ``cpu``."""
    spec = (spec or "auto").strip().lower()
    if spec not in ("auto", ""):
        return spec
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class ModelHandle:
    """A loaded checkpoint plus the evaluator built on it."""

    path: Path
    evaluator: Any
    encode_fn: Any
    device: str
    generation: int = 0
    step: int = 0


class _RootEnsemble:
    """C5 colour-symmetry ensemble for a single (root) evaluation.

    ``arm()`` makes the *next* ``evaluate`` call average the network over the
    five colour rotations (§1.4: the card and tile tables are closed under
    them, so all five are the same position with relabelled colours); every
    later call passes straight through.  In a single-tree search the first
    evaluated leaf is always the root, which is where the extra accuracy pays
    for itself.
    """

    def __init__(self, base: Any) -> None:
        self.base = base
        self._armed = False
        from ..symmetry import action_perm, feature_perm, inverse_perm
        self._feat = [feature_perm(k) for k in range(5)]
        self._act = [action_perm(k) for k in range(5)]
        self._inv = [inverse_perm(action_perm(k)) for k in range(5)]

    def arm(self) -> None:
        self._armed = True

    def evaluate(self, obs, mask):
        if not self._armed:
            return self.base.evaluate(obs, mask)
        self._armed = False
        obs = np.asarray(obs, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        priors = np.zeros((obs.shape[0], NUM_ACTIONS), dtype=np.float64)
        values = None
        for k in range(5):
            p, v = self.base.evaluate(obs[:, self._feat[k]],
                                      mask[:, self._act[k]])
            priors += np.asarray(p, dtype=np.float64)[:, self._inv[k]]
            values = np.asarray(v, dtype=np.float64) if values is None \
                else values + np.asarray(v, dtype=np.float64)
        priors *= mask
        total = priors.sum(axis=1, keepdims=True)
        priors = np.divide(priors, np.maximum(total, 1e-12))
        return (priors.astype(np.float32), (values / 5.0).astype(np.float32))


# ── the 1-ply stuck filter ────────────────────────────────────────────────

def self_stuck_after(state: E.GameState, seat: int, action: int) -> bool:
    """Would playing ``action`` leave ``seat`` with no legal move next turn?

    The check is deliberately 1-ply and *self*-referential: it asks whether the
    position we hand back to ourselves is one the engine has no accepted action
    for (the ``10 tokens + 3 reserved + nothing affordable`` trap).  Opponent
    replies can only add cards to the board and tokens to the bank, so a
    position that is stuck here is stuck for real unless somebody helps us.
    """
    try:
        after = state.clone()
        E.apply(after, action)
    except Exception:                                     # pragma: no cover
        return False
    if after.phase != E.PHASE_PLAYING or seat in after.resigned:
        return False
    probe = after.clone()
    probe.current_player = seat
    probe.turn_action = None
    probe.pending_tile_choice = None
    return not any(E.legal_mask(probe))


# ── the agent ─────────────────────────────────────────────────────────────

class MoveAgent:
    """Chooses one move per ``ai_move_request``.

    Thread-confined: the worker runs every request on the same thread, so the
    torch model, the tree and the RNG need no locking.
    """

    def __init__(self, cfg: WorkerConfig, log=None) -> None:
        self.cfg = cfg
        self.log = log or (lambda level, message, **kw: None)
        self.device = resolve_device(cfg.device)
        self._models: Dict[str, Optional[ModelHandle]] = {}
        self._torch_ready: Optional[bool] = None
        self._greedy = None
        self._rng = np.random.default_rng(cfg.seed or None)
        self._warned_no_model = False
        self._warned_clock_skew = False

    # -- torch / models ---------------------------------------------------
    def _ensure_torch(self) -> bool:
        if self._torch_ready is not None:
            return self._torch_ready
        try:
            import torch
            if self.device == "cpu":
                # One move at a time on one thread: extra threads only add
                # scheduling jitter to a 1500 ms budget.
                torch.set_num_threads(1)
            self._torch_ready = True
            self.log("info", f"torch {torch.__version__} on {self.device}")
        except Exception as err:                          # pragma: no cover
            self._torch_ready = False
            self.log("warn", f"torch is unavailable ({err}) — the worker will "
                             f"play on the greedy ladder")
        return self._torch_ready

    def model_for(self, key: str) -> Optional[ModelHandle]:
        """``MODEL_DIR/<key>.pt`` → ``MODEL_DIR/shared.pt`` → ``None``."""
        if key in self._models:
            return self._models[key]
        handle: Optional[ModelHandle] = None
        if self._ensure_torch():
            for path in self.cfg.checkpoint_candidates(key):
                if not path.is_file():
                    continue
                try:
                    handle = self._load(path)
                    break
                except Exception as err:
                    self.log("error", f"checkpoint {path} failed to load: {err}")
        if handle is None and not self._warned_no_model:
            self._warned_no_model = True
            self.log("warn",
                     f"no usable checkpoint under {self.cfg.model_path_dir} "
                     f"(looked for {key}.pt then shared.pt) — playing on the "
                     f"GREEDY ladder; drop a trained .pt there to enable search")
        self._models[key] = handle
        return handle

    def _load(self, path: Path) -> ModelHandle:
        import torch
        from ..encode import encode
        from ..model import NetEvaluator, load_checkpoint
        model, ckpt = load_checkpoint(str(path), map_location=self.device)
        model.eval()
        dtype = torch.float16 if self.device.startswith("cuda") else None
        evaluator = NetEvaluator(model, device=self.device,
                                 autocast_dtype=dtype)
        handle = ModelHandle(path=path, evaluator=evaluator, encode_fn=encode,
                             device=self.device,
                             generation=int(ckpt.get("generation", 0) or 0),
                             step=int(ckpt.get("step", 0) or 0))
        self.log("info", f"loaded {path.name} (generation {handle.generation}, "
                         f"step {handle.step}) on {self.device}")
        return handle

    def warmup_keys(self) -> Tuple[str, ...]:
        """Model keys this worker can be asked for, given ``cfg.modes``.

        ``mode_key()`` maps a request to ``ind2|ind3|ind4|ovt|team``; a worker
        that only advertises INDIVIDUAL will never be asked for ``team``, so
        only the reachable keys are warmed.
        """
        modes = {str(m).upper() for m in self.cfg.modes}
        keys: List[str] = []
        if "INDIVIDUAL" in modes:
            keys += ["ind2", "ind3", "ind4"]
        if "ONE_V_TWO" in modes:
            keys.append("ovt")
        if "TEAM" in modes:
            keys.append("team")
        return tuple(keys or ["ind2"])

    @staticmethod
    def _warmup_position(key: str) -> E.GameState:
        """A short, legal position of the shape ``key`` describes."""
        if key == "ovt":
            state = E.new_game(3, E.MODE_ONE_V_TWO, rng=random.Random(0))
        elif key == "team":
            state = E.new_game(4, E.MODE_TEAM, "ADJACENT", rng=random.Random(0))
        else:
            players = {"ind3": 3, "ind4": 4}.get(key, 2)
            state = E.new_game(players, rng=random.Random(0))
        for _ in range(6):
            actions = E.legal_actions(state)
            if not actions:
                break
            E.apply(state, actions[0])
        return state

    def warmup(self) -> None:
        """Pay the cold-start cost up front, not on somebody's first move.

        Loading torch, importing the search, building the CUDA context and
        JIT-warming the first forward pass together cost a second or two; the
        first real request would otherwise blow through its soft budget.

        Every mode this worker serves is warmed, not just ``ind2``: with
        per-mode checkpoints (``MODEL_DIR/ovt.pt`` …) each one is a separate
        file, a separate torch module and a separate cold start, so warming
        one of them left the first request of every *other* mode paying it.
        Distinct keys that resolve to the same file (the usual
        ``shared.pt`` deployment) are searched once.
        """
        started = time.monotonic()
        warmed: Dict[str, str] = {}                 # checkpoint path -> key
        for key in self.warmup_keys():
            handle = self.model_for(key)
            if handle is None:
                continue
            path = str(handle.path)
            if path in warmed:
                self.log("debug", f"warm-up: {key} shares "
                                  f"{Path(path).name} with {warmed[path]}")
                continue
            warmed[path] = key
            try:
                state = self._warmup_position(key)
                decision = Decision(action=_NONE_ACTION, level="warmup", seat=0)
                soft = time.monotonic() + 2.0
                self._search(state, state.current_player, handle, soft,
                             soft + 1.0, decision)
                self.log("info", f"warm-up {key} ({Path(path).name}) in "
                                 f"{(time.monotonic() - started) * 1000:.0f} ms "
                                 f"({decision.sims} simulations)")
            except Exception as err:                      # pragma: no cover
                self.log("warn", f"model warm-up for {key} failed: {err}")

    # -- budgets ----------------------------------------------------------
    def _budgets(self, payload: Mapping[str, Any],
                 started: float) -> Tuple[float, float]:
        """``(soft, hard)`` deadlines as ``time.monotonic`` stamps.

        ``payload['deadlineMs']`` is epoch milliseconds on the SERVER's clock;
        ours may be minutes away from it.  The deadline can therefore only
        *shorten* the local budget, never lengthen it, and a "remaining" that
        is implausibly small for a request we have only just picked up
        (< :data:`SKEW_FLOOR_MS`) is treated as skew and ignored — otherwise a
        worker whose clock runs a few seconds fast would play every move on a
        50 ms budget, i.e. on the greedy rung, for ever.
        """
        cfg = self.cfg
        local_ms = float(max(cfg.hard_budget_ms, cfg.time_budget_ms))
        hard_ms = local_ms
        deadline = payload.get("deadlineMs")
        if isinstance(deadline, (int, float)) and deadline > 0:
            remaining = max(0.0, float(deadline) - time.time() * 1000.0)
            waited_ms = max(0.0, (time.monotonic() - started) * 1000.0)
            if remaining < SKEW_FLOOR_MS and waited_ms < SKEW_FLOOR_MS:
                self._note_clock_skew(float(deadline), remaining)
            else:
                hard_ms = max(MIN_BUDGET_MS,
                              min(local_ms, remaining - cfg.deadline_margin_ms))
        soft_ms = min(float(cfg.time_budget_ms), hard_ms)
        return started + soft_ms / 1000.0, started + hard_ms / 1000.0

    def _note_clock_skew(self, deadline_ms: float, remaining_ms: float) -> None:
        """Log the first skewed deadline; stay silent afterwards."""
        if self._warned_clock_skew:
            return
        self._warned_clock_skew = True
        self.log("warn",
                 f"server deadline {deadline_ms:.0f} leaves only "
                 f"{remaining_ms:.0f} ms on this machine's clock for a request "
                 f"that just arrived — treating it as clock skew and using the "
                 f"local budget ({self.cfg.time_budget_ms}/"
                 f"{self.cfg.hard_budget_ms} ms). Sync the worker's clock "
                 f"(NTP) if moves start arriving late.")

    # -- the ladder -------------------------------------------------------
    def decide(self, payload: Mapping[str, Any]) -> Decision:
        started = time.monotonic()
        notes: List[str] = []
        try:
            state, seat = hydrate(payload, validate=True)
        except HydrationError as err:
            notes.append(f"validation failed: {err}")
            self.log("error", f"hydration rejected the payload ({err}) — "
                              f"retrying without consistency checks")
            state, seat = hydrate(payload, validate=False)
            notes.append("hydrated without validation")

        kind = "TILE" if str(payload.get("kind")) == "TILE" else "MOVE"
        key = payload_mode_key(payload)
        decision = Decision(action=_NONE_ACTION, level="none", mode=key,
                            seat=seat, kind=kind, notes=notes)

        mask = E.legal_mask(state)
        legal = [i for i in range(NUM_ACTIONS) if mask[i]]
        if kind == "TILE":
            legal = [i for i in legal if i >= CHOOSE_TILE_START]
            orphan = self._orphan_tile(payload, state, legal)
            if orphan is not None:
                decision.action = orphan
                decision.level = "greedy"
                decision.notes.append("orphaned tile choice (turnAction is not "
                                      "BUY): answered from pendingTileChoice")
                decision.ms = (time.monotonic() - started) * 1000.0
                return decision
        else:
            legal = [i for i in legal if i < CHOOSE_TILE_START]

        if not legal:
            decision.notes.append("no legal action — the seat is stuck")
            decision.ms = (time.monotonic() - started) * 1000.0
            return decision

        soft, hard = self._budgets(payload, started)
        # A fresh, independent copy of the position: everything the ladder
        # proposes is re-validated against this one and the wire message is
        # built from it.
        fresh, _ = hydrate(payload, validate=False)
        fresh_mask = E.legal_mask(fresh)

        for level, candidates in self._ladder(state, seat, kind, key, legal,
                                              soft, hard, decision):
            chosen = self._first_playable(fresh, fresh_mask, candidates, seat,
                                          kind, decision)
            if chosen is None:
                continue
            index, wire = chosen
            decision.action = wire
            decision.action_index = index
            decision.level = level
            decision.ms = (time.monotonic() - started) * 1000.0
            return decision

        decision.notes.append("every ladder rung failed")
        decision.ms = (time.monotonic() - started) * 1000.0
        return decision

    # -- ladder rungs -----------------------------------------------------
    def _ladder(self, state, seat, kind, key, legal, soft, hard, decision):
        """Yield ``(level, ranked candidate action indices)`` lazily."""
        handle = self.model_for(key)
        if handle is not None:
            yield "search", self._search(state, seat, handle, soft, hard,
                                         decision)
            yield "policy", self._policy(state, seat, handle, legal, decision)
        yield "greedy", self._greedy_candidates(state, seat, legal, decision)

    def _search(self, state, seat, handle, soft, hard,
                decision: Decision) -> List[int]:
        """Anytime PIMC MCTS: simulate in chunks, stop when the clock says so."""
        try:
            from ..search.mcts import MCTS, SearchConfig
        except Exception as err:                          # pragma: no cover
            decision.notes.append(f"search unavailable: {err}")
            return []
        cfg = self.cfg
        search_cfg = SearchConfig(
            sims=cfg.search_sims,
            universes=cfg.universes,
            noise=False,
            temperature=0.0,          # deployment: always the argmax
            temperature_plies=0,
        )
        evaluator = handle.evaluator
        ensemble = None
        if cfg.root_ensemble:
            try:
                ensemble = _RootEnsemble(evaluator)
                ensemble.arm()
                evaluator = ensemble
            except Exception as err:                      # pragma: no cover
                decision.notes.append(f"root ensemble disabled: {err}")
                evaluator = handle.evaluator
        try:
            tree = MCTS(search_cfg, self._rng)
            encode_fn = handle.encode_fn
            chunk = max(1, cfg.sim_chunk)
            done = 0
            while done < search_cfg.sims:
                # Always run one chunk: `result()` needs an expanded root, and
                # a budget that has already expired must still produce a move.
                if done and time.monotonic() >= soft:
                    break
                for _ in range(chunk):
                    if done >= search_cfg.sims:
                        break
                    leaf = tree.select_leaf(state, seat)
                    done += 1
                    if leaf is None:
                        continue
                    obs = encode_fn(leaf.state, leaf.seat)
                    priors, values = evaluator.evaluate(obs[None],
                                                        leaf.mask[None])
                    tree.backup(leaf.token, priors[0], values[0])
                if time.monotonic() >= hard:
                    decision.notes.append("hard budget hit")
                    break
            result = tree.result()
            decision.sims = int(result.stats.get("sims_run", tree.sims_done))
            # Entries >= num_players are padding the value head never learns
            # (the loss masks them); zero them so the log means something.
            decision.root_value = [float(v) if i < state.num_players else 0.0
                                   for i, v in enumerate(result.root_value)]
            visits = np.asarray(result.visits, dtype=np.int64)
            order = [int(a) for a in np.argsort(-visits) if visits[a] > 0]
            best = int(result.action)
            if best in order:
                order.remove(best)
            return [best] + order
        except Exception as err:
            decision.notes.append(f"search failed: {err}")
            self.log("error", f"search failed: {err!r}")
            return []

    def _policy(self, state, seat, handle, legal,
                decision: Decision) -> List[int]:
        """Network policy argmax over the legal actions (no search)."""
        try:
            obs = handle.encode_fn(state, seat)
            mask = np.asarray(E.legal_mask(state), dtype=bool)
            evaluator = handle.evaluator
            if self.cfg.root_ensemble:
                ensemble = _RootEnsemble(evaluator)
                ensemble.arm()
                evaluator = ensemble
            priors, values = evaluator.evaluate(obs[None], mask[None])
            p = np.asarray(priors[0], dtype=np.float64)
            if decision.root_value is None:
                absolute = _absolute(values[0], seat, state.num_players)
                decision.root_value = [float(v) if i < state.num_players else 0.0
                                       for i, v in enumerate(absolute)]
            order = [int(a) for a in np.argsort(-p) if mask[a]]
            if order:
                decision.policy = float(p[order[0]])
            return order
        except Exception as err:
            decision.notes.append(f"policy failed: {err}")
            self.log("error", f"policy evaluation failed: {err!r}")
            return []

    def _greedy_candidates(self, state, seat, legal,
                           decision: Decision) -> List[int]:
        """``GreedyBot`` first, then the remaining legal actions as a floor."""
        order: List[int] = []
        try:
            if self._greedy is None:
                from ..bots import GreedyBot
                self._greedy = GreedyBot()
            action = self._greedy.act(state, seat, self._rng)
            if action is not None:
                order.append(int(action))
        except Exception as err:
            decision.notes.append(f"greedy failed: {err}")
            self.log("error", f"greedy bot failed: {err!r}")
        order.extend(a for a in legal if a not in order)
        return order

    # -- shared plumbing --------------------------------------------------
    def _first_playable(self, fresh, fresh_mask, candidates, seat, kind,
                        decision: Decision):
        """First candidate that is legal on the fresh state and not self-stuck.

        The stuck filter is applied in two passes so that a position where
        *every* move traps us still produces a move rather than a resignation.
        """
        usable = [a for a in candidates
                  if 0 <= a < NUM_ACTIONS and fresh_mask[a]
                  and (a >= CHOOSE_TILE_START) == (kind == "TILE")]
        if not usable:
            return None
        safe = [a for a in usable if not self_stuck_after(fresh, seat, a)]
        if not safe:
            decision.notes.append("every candidate self-traps; playing anyway")
            safe = usable
        elif len(safe) < len(usable):
            decision.notes.append(
                f"stuck filter dropped {len(usable) - len(safe)} action(s)")
        for action in safe:
            try:
                return action, to_wire(fresh, action, kind)
            except HydrationError as err:                 # pragma: no cover
                decision.notes.append(f"action {action} unmappable: {err}")
        return None

    def _orphan_tile(self, payload, state, legal) -> Optional[Dict[str, Any]]:
        """The server's orphaned-noble corner (engine.py's subtlety #2).

        A gem take or a reserve that qualifies two or more nobles leaves
        ``_pendingTileChoice`` set while ``turnAction`` stays ``null``, so the
        bridge asks for a ``TILE`` move that ``processAction`` will refuse.
        Nothing we send can be accepted; answer with the tile the seat
        actually qualifies for so the server's own fallback lands on the same
        move instead of a resignation.
        """
        if legal:
            return None
        pending = payload.get("pendingTileChoice") or []
        player = state.players[state.current_player]
        for tile_id in pending:
            try:
                if E.qualifies_for_tile(player, int(tile_id)):
                    return {"type": "CHOOSE_TILE", "tileId": int(tile_id)}
            except Exception:                             # pragma: no cover
                continue
        return None


def _absolute(values, seat: int, num_players: int):
    """Evaluator values are relative to the acting seat; roll them back."""
    from ..search.mcts import seat_absolute
    return seat_absolute(np.asarray(values, dtype=np.float32)[:4], seat,
                         num_players)
