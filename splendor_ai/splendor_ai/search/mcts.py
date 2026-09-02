"""Open-loop PUCT / Gumbel MCTS over determinized universes (``AI_DESIGN`` §1.6).

Shape of the algorithm
----------------------
* **Open loop.**  A node is identified by the action path that reaches it, not
  by a state hash.  The state is re-derived every simulation by applying the
  path to a freshly determinized root, so hidden information is re-sampled
  without ever keying the tree on it.
* **PIMC universes.**  Simulation ``i`` uses universe ``i % cfg.universes``;
  each universe has a fixed seed, so a given action path always sees the same
  hidden state within a universe (cestpasphoto's PC-PIMC).  The legal set at a
  node can still differ between universes (a refilled board slot is a different
  card), so selection always intersects the node's priors with the mask of the
  *current* determinization; mass left over from now-illegal actions is handed
  to actions that were not legal when the node was expanded.
* **Per-seat value vectors, max^n backup.**  Every edge stores ``W[4]`` in
  ABSOLUTE seat order and backup is a plain accumulation.  A node maximises the
  entry of the seat that moves there: ``Q(a) = W[a][acting_seat] / N(a)``.
  Because the vector is absolute, the ``same_player`` flag (CHOOSE_TILE is a
  same-seat sub-decision) never has to flip a sign — it is recorded per edge
  for diagnostics and for consumers that want it.
* **Stuck leaves.**  The variant has no pass: a seat with no legal action
  resigns (``engine.resign``) and the simulation continues from the resulting
  state, which may be terminal or another seat's turn.
* **Terminal leaves** back up :func:`terminal_values` and need no evaluator
  call — ``select_leaf`` returns ``None`` for them.

Evaluator contract: ``values`` handed to :meth:`MCTS.backup` are RELATIVE to
the leaf's acting seat — index ``j`` is absolute seat ``(j + leaf_seat) % n``
and entries ``>= n`` are padding (``splendor_ai.values.seat_relative``).  The
tree rolls them back into absolute seat order.  ``priors`` are a 65-vector;
they are masked and renormalised here.

``terminal_values`` / ``standings_values`` delegate to ``splendor_ai/values.py``
as soon as that module exists, and fall back to the local §1.2 implementation
until it does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..rules import engine as E
from ..rules.actions import NUM_ACTIONS, RESERVE_DECK_START
from .determinize import determinize, universe_rng

__all__ = [
    "SearchConfig", "SearchResult", "Leaf", "MCTS", "run_search",
    "terminal_values", "standings_values", "seat_relative", "seat_absolute",
]

_DECK_RESERVE = (RESERVE_DECK_START, RESERVE_DECK_START + 1,
                 RESERVE_DECK_START + 2)          # 42, 43, 44
_ZERO4 = np.zeros(4, dtype=np.float32)


# ── value vectors (§1.2) ──────────────────────────────────────────────────

def seat_relative(z, seat: int, n: int = 4) -> np.ndarray:
    """Absolute → relative-to-``seat`` (index 0 = ``seat``).

    §1.2 writes this as ``np.roll(z, -seat)``, which is exact for a full table
    of four; with fewer seats the rotation has to stay inside the first ``n``
    entries so that a real value never lands in a padding slot (the same
    mod-``n`` seat ordering the observation encoder uses for its player
    blocks).  Identical to ``splendor_ai.values.seat_relative``.
    """
    return _rotate(z, -int(seat), n)


def seat_absolute(z, seat: int, n: int = 4) -> np.ndarray:
    """Relative-to-``seat`` → absolute seat order.  Inverse of the above."""
    return _rotate(z, int(seat), n)


def _rotate(z, shift: int, n: int) -> np.ndarray:
    """``np.roll(z[:n], shift)`` in place over a 4-vector, padding untouched.

    Hand-rolled because this runs once per backup and ``np.roll`` on four
    floats costs an order of magnitude more than the loop.
    """
    values = np.asarray(z, dtype=np.float32).tolist()
    if n <= 0 or shift % n == 0:
        return np.array(values, dtype=np.float32)
    out = list(values)
    shift %= n
    for j in range(n):
        out[(j + shift) % n] = values[j]
    return np.array(out, dtype=np.float32)


def _rank_values(keys: Sequence[tuple], n: int) -> np.ndarray:
    """Rank-linear value vector from per-seat sort keys (ties share the mean).

    ``z = 1 - 2*(rank-1)/(n-1)``; ``keys`` sort ascending (best first).
    """
    z = np.zeros(4, dtype=np.float32)
    if n <= 1:
        return z
    order = sorted(range(n), key=lambda i: keys[i])
    i = 0
    while i < n:
        j = i
        while j + 1 < n and keys[order[j + 1]] == keys[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            z[order[k]] = 1.0 - 2.0 * (mean_rank - 1.0) / (n - 1)
        i = j + 1
    return z


def _individual_keys(state: E.GameState) -> List[tuple]:
    """(resigned last, score desc, cards asc) — the server's ranking."""
    resigned = state.resigned
    keys = []
    for i, p in enumerate(state.players):
        if i in resigned:
            keys.append((1, 0, 0))
        else:
            keys.append((0, -p.score, len(p.cards)))
    return keys


def _team_side_values(state: E.GameState, side_of_seat, better) -> np.ndarray:
    z = np.zeros(4, dtype=np.float32)
    for i, p in enumerate(state.players):
        s = side_of_seat(i, p)
        z[i] = 1.0 if better == s else (0.0 if better is None else -1.0)
    return z


def _terminal_values(state: E.GameState) -> np.ndarray:
    """``AI_DESIGN`` §1.2 terminal value vector, ABSOLUTE seat order.

    Used until ``splendor_ai/values.py`` lands (see :func:`terminal_values`).
    """
    if state.mode == E.MODE_INDIVIDUAL:
        return _rank_values(_individual_keys(state), state.num_players)

    gr = state.game_result or {}
    if gr.get("reason") == "FORFEIT":
        forfeiting = gr.get("forfeitingTeamId")
        return _team_side_values(state, lambda i, p: p.team_id,
                                 1 - forfeiting if forfeiting in (0, 1) else None)
    winners = gr.get("winningTeamIds")
    if not winners or len(winners) != 1:
        # A tie (two winning teams) or nobody qualified: 0 for everybody.
        return np.zeros(4, dtype=np.float32)
    return _team_side_values(state, lambda i, p: p.team_id, winners[0])


_STANDINGS_VALUES = None


def _resolve(name: str, fallback):
    """``splendor_ai.values.<name>`` if that module exists, else ``fallback``."""
    try:                           # pragma: no cover - depends on repo state
        from .. import values as _values
        return getattr(_values, name, fallback)
    except Exception:
        return fallback


_TERMINAL_VALUES = None


def _resolve_terminal_values():
    """Prefer ``splendor_ai.values.terminal_values`` once that module exists."""
    global _TERMINAL_VALUES
    if _TERMINAL_VALUES is None:
        _TERMINAL_VALUES = _resolve("terminal_values", _terminal_values)
    return _TERMINAL_VALUES


def reset_terminal_values() -> None:
    """Drop the cached resolutions (tests / after ``values.py`` is added)."""
    global _TERMINAL_VALUES, _STANDINGS_VALUES
    _TERMINAL_VALUES = None
    _STANDINGS_VALUES = None


def terminal_values(state: E.GameState) -> np.ndarray:
    """Value vector of a finished game, ABSOLUTE seat order, ``float32[4]``."""
    return np.asarray(_resolve_terminal_values()(state), dtype=np.float32)


#: Threshold each side must reach, by mode and team id (``AI_DESIGN`` §1.2 and
#: ``engine.qualifying_team_ids``).
_SIDE_THRESHOLD = {(E.MODE_ONE_V_TWO, 0): 15.0, (E.MODE_ONE_V_TWO, 1): 34.0,
                   (E.MODE_TEAM, 0): 30.0, (E.MODE_TEAM, 1): 30.0}


def _standings_values(state: E.GameState) -> np.ndarray:
    """Value vector of an UNFINISHED game scored by current standings.

    Used for ``max_plies`` truncation (§1.2, "score by current standings via
    the same ranking").  INDIVIDUAL reuses the exact server ranking; the team
    modes compare each side's progress towards its own threshold, which is the
    natural generalisation of the ranking to asymmetric win conditions.
    """
    if state.mode == E.MODE_INDIVIDUAL:
        return _rank_values(_individual_keys(state), state.num_players)
    totals = {0: 0, 1: 0}
    for i, p in enumerate(state.players):
        if p.team_id in totals:
            totals[p.team_id] += p.score
    prog = {t: totals[t] / _SIDE_THRESHOLD[(state.mode, t)] for t in (0, 1)}
    if prog[0] == prog[1]:
        best = None
    else:
        best = 0 if prog[0] > prog[1] else 1
    return _team_side_values(state, lambda i, p: p.team_id, best)


def standings_values(state: E.GameState) -> np.ndarray:
    """Value vector of an unfinished game, ABSOLUTE seat order (§1.2)."""
    global _STANDINGS_VALUES
    if _STANDINGS_VALUES is None:
        _STANDINGS_VALUES = _resolve("standings_values", _standings_values)
    if state.is_over():
        return terminal_values(state)
    return np.asarray(_STANDINGS_VALUES(state), dtype=np.float32)


# ── configuration ─────────────────────────────────────────────────────────

@dataclass
class SearchConfig:
    """All search knobs.  Defaults are the §1.6 contract values."""

    sims: int = 400
    c_puct: float = 1.5
    fpu_reduction: float = 0.25
    #: Dirichlet ``alpha = dirichlet_alpha_scale / num_legal``.
    dirichlet_alpha_scale: float = 10.0
    noise_eps: float = 0.25
    noise: bool = False
    #: KataGo forced playouts at the root: force while ``N < sqrt(k*P*N_root)``.
    forced_playouts_k: float = 2.0
    prune_policy_target: bool = True
    #: Number of determinization universes cycled by simulation index.
    universes: int = 6
    root: str = "puct"                     # 'puct' | 'gumbel'
    gumbel_m: int = 16
    #: Anti-clairvoyance penalty ceilings per tier for RESERVE_FROM_DECK at the
    #: root, or ``None`` to disable.
    deck_reserve_penalty: Optional[Tuple[float, float, float]] = (0.02, 0.06, 0.12)
    temperature: float = 1.0
    #: Sample the move while ``state.turn_number < temperature_plies``.
    temperature_plies: int = 12
    max_depth: int = 200
    #: Gumbel sigma: ``(c_visit + max_N) * c_scale * q`` (mctx defaults).
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0


@dataclass
class SearchResult:
    visits: np.ndarray                     # int32[65]
    policy_target: np.ndarray              # float32[65], sums to 1
    root_value: np.ndarray                 # float32[4], ABSOLUTE seat order
    action: int
    stats: Dict[str, Any]


@dataclass
class Leaf:
    """A position waiting for an evaluator call."""

    state: E.GameState
    seat: int                              # seat to act at the leaf
    mask: np.ndarray                       # bool[65]
    token: int
    depth: int = 0
    obs: Any = None                        # filled by the caller via encode_fn


# ── tree ──────────────────────────────────────────────────────────────────

class _Node:
    """Flat per-action arrays; children keyed by action index."""

    __slots__ = ("P", "N", "W", "children", "same_player", "n_total",
                 "n_visits", "w_self", "expanded", "acting_seat")

    def __init__(self) -> None:
        self.P: Optional[List[float]] = None
        self.N = [0] * NUM_ACTIONS
        self.W = [0.0] * (NUM_ACTIONS * 4)        # absolute seat order
        self.children: Dict[int, "_Node"] = {}
        self.same_player = [False] * NUM_ACTIONS
        self.n_total = 0                          # sum of edge visits
        self.n_visits = 0                         # backups through this node
        self.w_self = [0.0, 0.0, 0.0, 0.0]
        self.expanded = False
        self.acting_seat = -1


class MCTS:
    """One search tree.  Batch across trees with :class:`~.scheduler.Scheduler`."""

    def __init__(self, cfg: SearchConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.root = _Node()
        self.base_seed = int(rng.integers(1 << 62))
        self.root_state: Optional[E.GameState] = None
        self.root_seat: int = -1
        self.num_players: int = 4
        self.root_mask: List[bool] = []
        self.root_legal: List[int] = []
        self.root_turn: int = 0
        self.sims_started = 0
        self.sims_done = 0
        self.nodes = 1
        self._pending: Dict[int, tuple] = {}
        #: universe index -> its determinized root (cloned per simulation);
        #: ``determinize`` is a pure function of (root, seat, universe seed).
        self._universes: Dict[int, E.GameState] = {}
        self._next_token = 1
        self._noise_applied = False
        self.stats: Dict[str, Any] = {
            "evaluated": 0, "terminal": 0, "stuck": 0, "depth_truncated": 0,
            "illegal_forced": 0, "depth_hist": {},
        }
        # Gumbel root bookkeeping
        self._g_ready = False
        self._g_cands: List[int] = []
        self._g_logit: Dict[int, float] = {}
        self._g_gumbel: Dict[int, float] = {}
        self._g_queue: List[int] = []
        self._g_cursor = 0
        self._g_budget = 0
        self._g_phases = 1

    # -- root ------------------------------------------------------------
    def _ensure_root(self, root_state: E.GameState, seat: int) -> None:
        if self.root_state is None:
            if root_state.phase != E.PHASE_PLAYING:
                raise ValueError("MCTS: root state is already terminal")
            self.root_state = root_state
            self.root_seat = int(seat)
            self.num_players = root_state.num_players
            self.root_mask = E.legal_mask(root_state)
            self.root_legal = [i for i, v in enumerate(self.root_mask) if v]
            if not self.root_legal:
                raise ValueError(
                    "MCTS: root seat has no legal action (stuck) — the caller "
                    "must resign instead of searching")
            self.root_turn = root_state.turn_number
        elif seat != self.root_seat:
            raise ValueError(
                f"MCTS: root seat changed {self.root_seat} -> {seat}")
        elif root_state is not self.root_state:
            # The determinized universes are cached against this exact root.
            raise ValueError(
                "MCTS: the root position changed between simulations — build "
                "a new tree per root")

    # -- selection -------------------------------------------------------
    def _priors(self, node: _Node, legal: List[int]) -> List[float]:
        """Node priors restricted to the currently legal actions.

        Priors were fixed when the node was expanded, under that universe's
        mask.  Under another universe some of those actions can be gone and
        some new ones can appear; the mass freed by the vanished actions is
        spread over the new ones (and a floor keeps them reachable).
        """
        P = node.P
        ps = [P[a] for a in legal]
        total = 0.0
        zeros = 0
        for v in ps:
            total += v
            if v <= 0.0:
                zeros += 1
        if zeros:
            fill = (1.0 - total) / zeros
            if fill <= 0.0:
                fill = 1.0 / len(legal)
            ps = [v if v > 0.0 else fill for v in ps]
            total = sum(ps)
        if total <= 0.0:
            return [1.0 / len(legal)] * len(legal)
        if abs(total - 1.0) > 1e-9:
            inv = 1.0 / total
            ps = [v * inv for v in ps]
        return ps

    def _select_action(self, node: _Node, legal: List[int], acting: int,
                       is_root: bool) -> int:
        cfg = self.cfg
        ps = self._priors(node, legal)
        n_total = node.n_total
        sqrt_n = math.sqrt(n_total) if n_total > 0 else 1.0
        parent_q = (node.w_self[acting] / node.n_visits) if node.n_visits else 0.0
        fpu = parent_q - cfg.fpu_reduction
        c_puct = cfg.c_puct
        N = node.N
        W = node.W

        forced_k = cfg.forced_playouts_k if (is_root and cfg.root == "puct") else 0.0
        # anti-clairvoyance: one uniform draw in [0, p_tier] per tier, drawn
        # lazily so a root without deck reserves costs nothing
        caps = cfg.deck_reserve_penalty if is_root else None
        penalty = None

        best_a = -1
        best_score = -1e30
        best_forced_a = -1
        best_forced_score = -1e30
        for i, a in enumerate(legal):
            n = N[a]
            q = (W[a * 4 + acting] / n) if n else fpu
            score = q + c_puct * ps[i] * sqrt_n / (1 + n)
            if caps is not None and a in _DECK_RESERVE:
                if penalty is None:
                    penalty = self.rng.random(3) * np.asarray(
                        caps, dtype=np.float64)
                score -= float(penalty[a - RESERVE_DECK_START])
            if forced_k > 0.0 and n_total > 0:
                if n < math.sqrt(forced_k * ps[i] * n_total):
                    if score > best_forced_score:
                        best_forced_score = score
                        best_forced_a = a
            if score > best_score:
                best_score = score
                best_a = a
        if best_forced_a >= 0:
            return best_forced_a
        return best_a

    # -- simulation ------------------------------------------------------
    def select_leaf(self, root_state: E.GameState, seat: int) -> Optional[Leaf]:
        """Run one simulation down to a leaf.

        Returns ``None`` when the simulation finished without needing the
        evaluator (terminal leaf, or the depth cap); the backup has already
        happened in that case.  Otherwise the caller must fill ``leaf.obs``
        (``encode_fn(leaf.state, leaf.seat)``) and call :meth:`backup`.
        """
        self._ensure_root(root_state, seat)
        cfg = self.cfg
        universe = self.sims_started % max(1, cfg.universes)
        self.sims_started += 1
        world = self._universes.get(universe)
        if world is None:
            world = determinize(root_state, seat,
                                universe_rng(self.base_seed, universe))
            self._universes[universe] = world
        state = world.clone()

        node = self.root
        path: List[Tuple[_Node, int]] = []
        depth = 0
        forced_root = None
        if cfg.root == "gumbel" and self.root.expanded:
            forced_root = self._gumbel_next()

        while True:
            # A seat with no legal action resigns (the variant has no pass) and
            # the simulation continues from whatever that leaves behind.
            mask = None
            if state.phase == E.PHASE_PLAYING:
                mask = E.legal_mask(state)
                guard = 0
                while not any(mask):
                    self.stats["stuck"] += 1
                    E.resign(state, state.current_player)
                    guard += 1
                    if state.phase != E.PHASE_PLAYING:
                        break
                    if guard > state.num_players:      # pragma: no cover
                        break
                    mask = E.legal_mask(state)

            if state.phase != E.PHASE_PLAYING:
                self._backup_values(path, node, terminal_values(state))
                self.stats["terminal"] += 1
                self._note_depth(depth)
                self.sims_done += 1
                return None

            if depth >= cfg.max_depth or not any(mask):
                self._backup_values(path, node, _ZERO4)
                self.stats["depth_truncated"] += 1
                self._note_depth(depth)
                self.sims_done += 1
                return None

            if not node.expanded:
                token = self._next_token
                self._next_token += 1
                self._pending[token] = (path, node, state.current_player,
                                        mask, depth)
                self._note_depth(depth)
                return Leaf(state=state, seat=state.current_player,
                            mask=np.asarray(mask, dtype=bool), token=token,
                            depth=depth)

            legal = [i for i, v in enumerate(mask) if v]
            if depth == 0 and forced_root is not None:
                a = forced_root if mask[forced_root] else \
                    self._select_action(node, legal, state.current_player, True)
                if not mask[forced_root]:
                    self.stats["illegal_forced"] += 1
            else:
                a = self._select_action(node, legal, state.current_player,
                                        depth == 0)

            prev_seat = state.current_player
            E.apply(state, a)
            child = node.children.get(a)
            if child is None:
                child = _Node()
                node.children[a] = child
                self.nodes += 1
            node.same_player[a] = (state.phase == E.PHASE_PLAYING
                                   and state.current_player == prev_seat)
            path.append((node, a))
            node = child
            depth += 1

    def backup(self, token: int, priors, values) -> None:
        """Expand the pending leaf with ``priors`` and back up ``values``.

        ``values`` is relative to the leaf's acting seat (index 0 = that seat).
        """
        try:
            path, node, leaf_seat, mask, depth = self._pending.pop(token)
        except KeyError:                                   # pragma: no cover
            raise KeyError(f"MCTS.backup: unknown or reused token {token}")

        p = np.asarray(priors, dtype=np.float64).reshape(-1)[:NUM_ACTIONS]
        m = np.asarray(mask, dtype=bool)
        p = np.where(m, np.maximum(p, 0.0), 0.0)
        total = float(p.sum())
        if total <= 0.0:
            p = m.astype(np.float64)
            total = float(p.sum())
        p = p / total
        node.P = p.tolist()
        node.expanded = True
        node.acting_seat = leaf_seat

        if node is self.root:
            if self.cfg.noise and not self._noise_applied:
                self._apply_root_noise()
            self._noise_applied = True

        v_abs = seat_absolute(
            np.asarray(values, dtype=np.float32).reshape(-1)[:4],
            leaf_seat, self.num_players)
        self._backup_values(path, node, v_abs)
        self.stats["evaluated"] += 1
        self.sims_done += 1

    def _apply_root_noise(self) -> None:
        legal = self.root_legal
        if len(legal) < 2:
            return
        alpha = self.cfg.dirichlet_alpha_scale / len(legal)
        noise = self.rng.dirichlet([alpha] * len(legal))
        eps = self.cfg.noise_eps
        P = self.root.P
        for i, a in enumerate(legal):
            P[a] = (1.0 - eps) * P[a] + eps * float(noise[i])

    def _backup_values(self, path, leaf_node: _Node, v) -> None:
        v0, v1, v2, v3 = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
        w = leaf_node.w_self
        w[0] += v0
        w[1] += v1
        w[2] += v2
        w[3] += v3
        leaf_node.n_visits += 1
        for node, a in path:
            node.N[a] += 1
            node.n_total += 1
            node.n_visits += 1
            base = a * 4
            W = node.W
            W[base] += v0
            W[base + 1] += v1
            W[base + 2] += v2
            W[base + 3] += v3
            w = node.w_self
            w[0] += v0
            w[1] += v1
            w[2] += v2
            w[3] += v3

    def _note_depth(self, depth: int) -> None:
        h = self.stats["depth_hist"]
        h[depth] = h.get(depth, 0) + 1

    # -- Gumbel root (sequential halving over Gumbel top-m) ---------------
    def _gumbel_init(self) -> None:
        legal = self.root_legal
        P = self.root.P
        logits = np.log(np.maximum(np.array([P[a] for a in legal]), 1e-12))
        g = self.rng.gumbel(size=len(legal))
        self._g_logit = {a: float(logits[i]) for i, a in enumerate(legal)}
        self._g_gumbel = {a: float(g[i]) for i, a in enumerate(legal)}
        m = max(1, min(self.cfg.gumbel_m, len(legal)))
        order = np.argsort(-(logits + g))[:m]
        self._g_cands = [legal[int(i)] for i in order]
        self._g_budget = max(len(self._g_cands),
                             self.cfg.sims - self.sims_started + 1)
        self._g_phases = max(1, int(math.ceil(math.log2(max(m, 2)))))
        self._g_ready = True
        self._g_new_phase()

    def _g_new_phase(self) -> None:
        k = len(self._g_cands)
        if k <= 1:
            per = max(1, self._g_budget)
        else:
            per = max(1, self._g_budget // (self._g_phases * k))
        self._g_queue = [a for _ in range(per) for a in self._g_cands]
        self._g_cursor = 0
        self._g_budget = max(0, self._g_budget - per * k)
        self._g_phases = max(1, self._g_phases - 1)

    def _root_q(self, a: int) -> Optional[float]:
        n = self.root.N[a]
        if not n:
            return None
        return self.root.W[a * 4 + self.root_seat] / n

    def _v_mix(self) -> float:
        """mctx's mixed value: root value blended with the visited children."""
        root = self.root
        v_root = (root.w_self[self.root_seat] / root.n_visits
                  if root.n_visits else 0.0)
        sum_n = 0
        wq = 0.0
        wp = 0.0
        P = root.P
        for a in self.root_legal:
            n = root.N[a]
            if n:
                sum_n += n
                wp += P[a]
                wq += P[a] * (root.W[a * 4 + self.root_seat] / n)
        if sum_n == 0 or wp <= 0.0:
            return v_root
        return (v_root + sum_n * (wq / wp)) / (1.0 + sum_n)

    def _sigma(self, q: float) -> float:
        max_n = max((self.root.N[a] for a in self.root_legal), default=0)
        return (self.cfg.gumbel_c_visit + max_n) * self.cfg.gumbel_c_scale * q

    def _g_score(self, a: int) -> float:
        q = self._root_q(a)
        if q is None:
            q = self._v_mix()
        return self._g_logit[a] + self._g_gumbel[a] + self._sigma(q)

    def _gumbel_next(self) -> int:
        if not self._g_ready:
            self._gumbel_init()
        if self._g_cursor >= len(self._g_queue):
            if len(self._g_cands) > 1:
                ranked = sorted(self._g_cands, key=self._g_score, reverse=True)
                self._g_cands = ranked[:max(1, len(ranked) // 2)]
            self._g_new_phase()
        a = self._g_queue[self._g_cursor]
        self._g_cursor += 1
        return a

    def _gumbel_policy(self) -> Tuple[np.ndarray, int]:
        """Improved policy ``softmax(logits + sigma(completedQ))`` and action.

        mctx returns ``argmax(g + logits + sigma(q))`` — the Gumbel sample —
        which is an *exploration* choice: after sequential halving two
        candidates usually survive and the Gumbel noise picks between them.
        For play we return the noise-free improved-policy argmax over the
        surviving candidates instead; :meth:`result` still samples the improved
        policy itself during the temperature plies, which is where the
        exploration belongs.
        """
        target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        legal = self.root_legal
        if not self._g_ready:
            visits = np.array(self.root.N, dtype=np.float64)
            visits[~np.asarray(self.root_mask, dtype=bool)] = 0.0
            if visits.sum() <= 0:
                target[legal] = 1.0 / len(legal)
                return target, int(legal[0])
            target[:] = (visits / visits.sum()).astype(np.float32)
            return target, int(np.argmax(visits))
        v_mix = self._v_mix()
        scores = []
        for a in legal:
            q = self._root_q(a)
            scores.append(self._g_logit[a] + self._sigma(v_mix if q is None else q))
        s = np.array(scores, dtype=np.float64)
        s -= s.max()
        e = np.exp(s)
        target[legal] = (e / e.sum()).astype(np.float32)
        improved = dict(zip(legal, scores))
        best = max(self._g_cands, key=lambda a: improved[a])
        return target, int(best)

    # -- result ----------------------------------------------------------
    def _policy_target(self) -> np.ndarray:
        """Visit distribution with KataGo forced-playout pruning applied."""
        root = self.root
        legal = self.root_legal
        visits = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for a in legal:
            visits[a] = root.N[a]
        total = visits.sum()
        target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if total <= 0:
            target[legal] = 1.0 / len(legal)
            return target
        pruned = visits.copy()
        if self.cfg.prune_policy_target and self.cfg.forced_playouts_k > 0:
            best = int(np.argmax(visits))
            k = self.cfg.forced_playouts_k
            for a in legal:
                if a == best or visits[a] <= 0:
                    continue
                n_forced = math.sqrt(k * root.P[a] * total)
                v = visits[a] - n_forced
                pruned[a] = v if v > 1.0 else 0.0
        if pruned.sum() <= 0:
            pruned = visits
        target[:] = (pruned / pruned.sum()).astype(np.float32)
        return target

    def _pick_action(self, target: np.ndarray) -> int:
        cfg = self.cfg
        visits = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for a in self.root_legal:
            visits[a] = self.root.N[a]
        if visits.sum() <= 0:
            visits[:] = target
        if (cfg.temperature > 0.0 and self.root_turn < cfg.temperature_plies
                and visits.sum() > 0):
            w = np.power(visits, 1.0 / cfg.temperature)
            s = w.sum()
            if s > 0:
                return int(self.rng.choice(NUM_ACTIONS, p=w / s))
        return int(np.argmax(visits))

    def result(self) -> SearchResult:
        root = self.root
        visits = np.zeros(NUM_ACTIONS, dtype=np.int32)
        for a in self.root_legal:
            visits[a] = root.N[a]
        if self.cfg.root == "gumbel":
            target, action = self._gumbel_policy()
            if (self.cfg.temperature > 0.0
                    and self.root_turn < self.cfg.temperature_plies
                    and target.sum() > 0):
                action = int(self.rng.choice(NUM_ACTIONS,
                                             p=(target / target.sum()).astype(np.float64)))
        else:
            target = self._policy_target()
            action = self._pick_action(target)
        root_value = (np.array(root.w_self, dtype=np.float32) / root.n_visits
                      if root.n_visits else np.zeros(4, dtype=np.float32))
        stats = dict(self.stats)
        stats.update({
            "sims_requested": self.cfg.sims,
            "sims_run": self.sims_done,
            "nodes": self.nodes,
            "root_visits": int(visits.sum()),
            "max_depth": max(stats["depth_hist"]) if stats["depth_hist"] else 0,
            "universes": self.cfg.universes,
            "root": self.cfg.root,
        })
        stats["depth_hist"] = dict(stats["depth_hist"])
        return SearchResult(visits=visits, policy_target=target,
                            root_value=root_value, action=int(action),
                            stats=stats)


# ── single-tree convenience driver ────────────────────────────────────────

def run_search(state: E.GameState, seat: int, evaluator, encode_fn,
               cfg: SearchConfig, rng: np.random.Generator) -> SearchResult:
    """Run ``cfg.sims`` simulations on one tree and return the result."""
    tree = MCTS(cfg, rng)
    for _ in range(cfg.sims):
        leaf = tree.select_leaf(state, seat)
        if leaf is None:
            continue
        leaf.obs = encode_fn(leaf.state, leaf.seat)
        obs = leaf.obs
        batch = (obs[None] if isinstance(obs, np.ndarray) else [obs])
        priors, values = evaluator.evaluate(batch, leaf.mask[None])
        tree.backup(leaf.token, priors[0], values[0])
    return tree.result()
