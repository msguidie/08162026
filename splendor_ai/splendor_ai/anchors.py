"""The fixed anchor ladder and the bot registry — ``docs/AI_DESIGN.md`` §1.7, §4.

Every strength number this project publishes is measured against *pinned*
anchors, never against the previous generation: the research judges' single
loudest finding is that a self-referential Elo chain (chlligence's 51.8M-step
run) cannot distinguish progress from drift.  So the ladder below is frozen
forever:

======================  ====================================================
``random``              uniform over the legal actions
``greedy``              the 1-ply heuristic (``search.evaluators.greedy_action``)
``mcts40``              NN-free PUCT, 40 sims, heuristic priors + greedy-rollout
                        leaf values
``mcts160``             the same search at 160 sims
``mcts640``             the same search at 640 sims
======================  ====================================================

The three MCTS rungs share one :data:`ANCHOR_SEARCH` config and one evaluator
(:data:`ANCHOR_PRIORS` priors over greedy-rollout values) and differ *only* in
``sims``, which makes them an absolute, monotone strength curve: if a
checkpoint's Elo against ``mcts160`` rises while its Elo against ``mcts640``
does not, that is a real, interpretable signal rather than pool noise.  Do not
"improve" these settings — a changed anchor invalidates every historical Elo.

The ladder must also be monotone in itself, and with *uniform* leaf priors it
was not: at 40 simulations PUCT spends its whole budget on the ~40-way take
block and never expands a buy, so ``mcts40`` scored 0.13 against ``greedy``
over 20 paired 2p games — a rung *below* the rung under it, which makes any
Elo fitted through it meaningless.  The anchors therefore use
``RolloutEvaluator(..., priors='heuristic')``: the same NN-free greedy rollout
for the value, with the buy block weighted up in the prior
(``search.evaluators.heuristic_priors``).  Measured over 20 paired ind2 games
that lifts ``mcts40`` to 0.71 against ``greedy`` and leaves the ordering
``greedy < mcts40 < mcts160 < mcts640``.

Learned bots are built from a checkpoint:

* :func:`net_policy` — the raw policy head, ``argmax`` over the legal actions,
  no search.  This is the ablation partner that isolates how much strength
  comes from search (§4: "the same-weights search ablation").
* :func:`net_search` — the same weights inside the §1.6 MCTS.

Both take an optional C5 root ensemble (``ensemble=True``): the network is
evaluated on all five colour rotations of the position and the priors are
rotated back and averaged (``symmetry.feature_perm`` / ``action_perm``), which
is what the deployment worker does (§1.9).

Everything is addressed by a **spec string** so a bot can cross a process
boundary as plain text (the arena's worker pool) and so the CLI is copyable::

    random  greedy  mcts40  mcts160  mcts640  mcts400
    net:runs/nscc/weights/latest.pt
    net:runs/nscc/weights/latest.pt:c5
    search:runs/nscc/weights/latest.pt:400
    gen40=search:runs/nscc/checkpoints/gen_0040.pt:400:c5

``make_bot(spec, device)`` builds one; ``make_factory(spec, device)`` returns a
picklable callable that builds it later, inside a worker process.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .bots import Bot, GreedyBot, MctsBot, RandomBot
from .rules import engine as E
from .search.evaluators import RolloutEvaluator, state_encoder
from .search.mcts import SearchConfig

__all__ = [
    "ANCHOR_LADDER", "ANCHOR_SIMS", "ANCHOR_SEARCH", "ANCHOR_ROLLOUT_PLIES",
    "anchor_search_config", "make_anchor", "anchor_ladder",
    "net_policy", "net_search", "make_bot", "make_factory", "BotFactory",
    "NetPolicyBot", "C5Evaluator", "load_net", "parse_spec", "spec_label",
]

#: The five rungs, weakest first.  Frozen: see the module docstring.
ANCHOR_LADDER: Tuple[str, ...] = ("random", "greedy", "mcts40", "mcts160",
                                  "mcts640")

#: Simulation budget of each MCTS rung.
ANCHOR_SIMS: Dict[str, int] = {"mcts40": 40, "mcts160": 160, "mcts640": 640}

#: Plies a leaf rollout plays before it is scored by current standings.
ANCHOR_ROLLOUT_PLIES = 60

#: Leaf priors of the MCTS rungs.  ``'heuristic'`` = buy-biased
#: (:func:`~.search.evaluators.heuristic_priors`); ``'uniform'`` is what the
#: ladder used before and is kept only for A/B experiments.
ANCHOR_PRIORS = "heuristic"

#: The frozen search settings shared by every MCTS anchor.  Evaluation-mode
#: PUCT: no Dirichlet noise, no forced playouts (both are training devices
#: that deliberately distort the visit distribution), argmax move selection.
ANCHOR_SEARCH: Dict[str, Any] = {
    "c_puct": 1.5,
    "fpu_reduction": 0.25,
    "noise": False,
    "forced_playouts_k": 0.0,
    "prune_policy_target": False,
    "universes": 6,
    "root": "puct",
    "temperature": 0.0,
    "temperature_plies": 0,
    "deck_reserve_penalty": (0.02, 0.06, 0.12),
}

_MCTS_SPEC = re.compile(r"^mcts(\d+)$")
_FLAGS = {"c5", "ens", "ensemble"}


def anchor_search_config(sims: int, **overrides: Any) -> SearchConfig:
    """The frozen :class:`SearchConfig` of the anchor ladder at ``sims``.

    ``overrides`` exist for experiments (an A/B of ``root='gumbel'``, say) and
    for the learned search bots, which reuse the same evaluation-mode settings
    with a bigger universe count.  An *anchor* never passes any.
    """
    return SearchConfig(sims=int(sims), **{**ANCHOR_SEARCH, **overrides})


# ── the NN-free ladder ────────────────────────────────────────────────────

def make_anchor(name: str) -> Bot:
    """Build one rung of the ladder by name (``random`` … ``mcts640``).

    The MCTS rungs are seeded by the caller: :func:`~.bots.play_game` hands
    every ``act`` the game's own generator, so an anchor's whole game is
    reproducible from the game seed alone (the greedy rollout policy draws no
    random numbers at all, so only the determinization universes consume the
    stream).
    """
    if name == "random":
        return RandomBot()
    if name == "greedy":
        return GreedyBot()
    m = _MCTS_SPEC.match(name)
    if m:
        sims = int(m.group(1))
        if sims <= 0:
            raise ValueError(f"anchor {name!r} needs a positive sim count")
        return MctsBot(
            anchor_search_config(sims),
            RolloutEvaluator("greedy", max_plies=ANCHOR_ROLLOUT_PLIES,
                             priors=ANCHOR_PRIORS),
            state_encoder,
            name=name,
        )
    raise ValueError(
        f"unknown anchor {name!r}; the ladder is {list(ANCHOR_LADDER)} "
        f"(any 'mcts<N>' also works)")


def anchor_ladder(names: Sequence[str] = ANCHOR_LADDER) -> Dict[str, "BotFactory"]:
    """``{name: factory}`` for the ladder — ready for :func:`arena.run_matches`."""
    return {name: make_factory(name) for name in names}


# ── learned bots ──────────────────────────────────────────────────────────

_NET_CACHE: Dict[Tuple[str, str], Any] = {}


def load_net(path: str, device: str = "cpu"):
    """``(model, ckpt)`` from ``path``, cached per process.

    A process-wide cache matters in the arena: an arena worker plays hundreds
    of games with the same checkpoint and must not re-read it every game.  The
    model is left in ``eval()`` mode and is never mutated afterwards, so
    sharing one instance between bots is safe.
    """
    from .model import load_checkpoint            # torch import stays lazy

    key = (os.path.abspath(str(path)), str(device))
    hit = _NET_CACHE.get(key)
    if hit is None:
        model, ckpt = load_checkpoint(str(path), map_location=str(device))
        model.eval()
        hit = _NET_CACHE[key] = (model, ckpt)
    return hit


def _net_encode(state: E.GameState, seat: int) -> np.ndarray:
    """``encode_fn`` for the learned bots (the real §1.3 observation)."""
    from .encode import encode
    return encode(state, seat)


class C5Evaluator:
    """Colour-symmetry ensemble around any batched evaluator (§1.4, §1.9).

    ``evaluate`` runs the wrapped evaluator on all five colour rotations of
    the batch and averages the priors after rotating them back.  Rotation is a
    pure index permutation on both the observation
    (``encode(rotate_state(s,k)) == encode(s)[feature_perm(k)]``) and the
    action vector (``legal_mask(rotate_state(s,k)) == mask[action_perm(k)]``),
    so no state is re-encoded — it costs 5× the network time and nothing else.
    Values are colour-invariant and are averaged directly.

    Note this ensembles **every** leaf, a superset of the worker's root-only
    ensemble; it is off by default because in search the extra network time is
    usually better spent on more simulations.
    """

    name = "c5"

    def __init__(self, inner: Any) -> None:
        from .symmetry import NUM_ROTATIONS, action_perm, feature_perm, inverse_perm

        self.inner = inner
        self._k = NUM_ROTATIONS
        self._fperm = [feature_perm(k) for k in range(self._k)]
        self._aperm = [action_perm(k) for k in range(self._k)]
        self._ainv = [inverse_perm(action_perm(k)) for k in range(self._k)]

    def evaluate(self, obs: np.ndarray, mask: np.ndarray):
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=np.bool_)
        priors_sum: Optional[np.ndarray] = None
        values_sum: Optional[np.ndarray] = None
        for k in range(self._k):
            p, v = self.inner.evaluate(obs[:, self._fperm[k]],
                                       mask[:, self._aperm[k]])
            p = np.asarray(p, dtype=np.float32)[:, self._ainv[k]]
            v = np.asarray(v, dtype=np.float32)
            priors_sum = p if priors_sum is None else priors_sum + p
            values_sum = v if values_sum is None else values_sum + v
        priors = priors_sum / float(self._k)
        total = priors.sum(axis=1, keepdims=True)
        priors = np.divide(priors, np.maximum(total, 1e-12),
                           out=np.zeros_like(priors), where=total > 0)
        return priors.astype(np.float32), (values_sum / float(self._k)
                                           ).astype(np.float32)


def _make_evaluator(ckpt: Any, device: str, ensemble: bool):
    """``NetEvaluator`` (optionally C5-wrapped) plus the model it wraps."""
    from .model import NetEvaluator, SplendorNet

    if isinstance(ckpt, SplendorNet):
        model, meta = ckpt, {}
    else:
        model, meta = load_net(str(ckpt), device)
    evaluator = NetEvaluator(model, device)
    return (C5Evaluator(evaluator) if ensemble else evaluator), model, meta


class NetPolicyBot:
    """Policy-head ``argmax``, no search (§1.7 ``NetBot``).

    The mask is applied inside :class:`~.model.SplendorNet.forward`, so the
    argmax over the returned priors is always legal; a seat with no legal
    action at all returns ``None`` and the driver resigns it.
    """

    def __init__(self, ckpt: Any, device: str = "cpu", ensemble: bool = False,
                 name: Optional[str] = None) -> None:
        self.evaluator, self.model, self.meta = _make_evaluator(
            ckpt, device, ensemble)
        self.device = device
        self.ensemble = bool(ensemble)
        self.name = name or "net"
        self.last_priors: Optional[np.ndarray] = None

    def act(self, state: E.GameState, seat: int, rng=None) -> Optional[int]:
        mask = np.asarray(E.legal_mask(state), dtype=np.bool_)
        if not mask.any():
            return None
        obs = _net_encode(state, seat)
        priors, _values = self.evaluator.evaluate(obs[None], mask[None])
        priors = np.asarray(priors, dtype=np.float32)[0] * mask
        self.last_priors = priors
        if not np.isfinite(priors).any() or priors.max() <= 0.0:
            # A degenerate (untrained / all-zero) policy still has to move.
            legal = np.flatnonzero(mask)
            return int(legal[0])
        return int(np.argmax(priors))


def net_policy(ckpt: Any, device: str = "cpu", ensemble: bool = False,
               name: Optional[str] = None) -> NetPolicyBot:
    """Policy-argmax bot from a checkpoint path (or a live ``SplendorNet``)."""
    return NetPolicyBot(ckpt, device=device, ensemble=ensemble, name=name)


def net_search(ckpt: Any, sims: int = 400, device: str = "cpu",
               ensemble: bool = False, name: Optional[str] = None,
               **overrides: Any) -> MctsBot:
    """The §1.6 search driven by a checkpoint's network.

    Same evaluation-mode settings as the anchors (no noise, no forced
    playouts, argmax) with more determinization universes, because a learned
    evaluator is cheap enough per leaf that hidden-information variance, not
    node count, dominates.
    """
    evaluator, _model, _meta = _make_evaluator(ckpt, device, ensemble)
    cfg = anchor_search_config(sims, **{"universes": 16, **overrides})
    return MctsBot(cfg, evaluator, _net_encode,
                   name=name or f"search{sims}")


# ── spec strings ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BotSpec:
    """A parsed spec string."""

    label: str                     # display name in the report
    kind: str                      # random | greedy | mcts | net | search
    path: Optional[str] = None
    sims: int = 0
    ensemble: bool = False
    raw: str = ""


def parse_spec(spec: str) -> BotSpec:
    """Parse ``[label=]kind[:path][:sims][:c5]``.

    The path is rejoined from the middle tokens, so a Windows path
    (``search:C:\\models\\latest.pt:400``) survives the colon split.
    """
    raw = str(spec).strip()
    if not raw:
        raise ValueError("empty bot spec")
    label = ""
    head = raw.split(":", 1)[0]
    if "=" in head:                                  # label=... (before any ':')
        label, _, raw = raw.partition("=")
        label = label.strip()
        raw = raw.strip()
    tokens = raw.split(":")
    kind = tokens[0].strip().lower()
    rest = [t.strip() for t in tokens[1:]]

    ensemble = False
    while rest and rest[-1].lower() in _FLAGS:
        ensemble = True
        rest.pop()

    if kind in ("random", "greedy"):
        if rest:
            raise ValueError(f"{kind!r} takes no arguments: {spec!r}")
        return BotSpec(label or kind, kind, raw=raw)

    m = _MCTS_SPEC.match(kind)
    if m:
        if rest:
            raise ValueError(f"{kind!r} takes no arguments: {spec!r}")
        return BotSpec(label or kind, "mcts", sims=int(m.group(1)), raw=raw)

    if kind in ("net", "search"):
        sims = 0
        if kind == "search" and len(rest) >= 2 and rest[-1].isdigit():
            sims = int(rest.pop())
        elif kind == "search":
            sims = 400
        if not rest or not rest[0]:
            raise ValueError(f"{kind!r} needs a checkpoint path: {spec!r}")
        path = ":".join(rest)
        return BotSpec(label or spec_label(kind, path, sims, ensemble),
                       kind, path=path, sims=sims, ensemble=ensemble, raw=raw)

    raise ValueError(
        f"unknown bot spec {spec!r}; expected one of random | greedy | "
        f"mcts<N> | net:<ckpt>[:c5] | search:<ckpt>:<sims>[:c5]")


def spec_label(kind: str, path: str, sims: int, ensemble: bool) -> str:
    """A short, stable report name for a checkpoint bot.

    ``runs/nscc/checkpoints/gen_0040.pt`` → ``search400:gen_0040``; a
    ``weights/latest.pt`` keeps the run directory instead of the useless
    basename → ``net:nscc``.
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    if stem in ("latest", "best", "model", "shared"):
        parent = os.path.basename(os.path.dirname(os.path.dirname(path)))
        stem = parent or stem
    head = "net" if kind == "net" else f"search{sims}"
    return f"{head}:{stem}" + ("+c5" if ensemble else "")


def make_bot(spec: str, device: str = "cpu") -> Bot:
    """Build the bot named by a spec string (see the module docstring)."""
    parsed = parse_spec(spec)
    if parsed.kind in ("random", "greedy"):
        bot = make_anchor(parsed.kind)
    elif parsed.kind == "mcts":
        bot = make_anchor(f"mcts{parsed.sims}")
    elif parsed.kind == "net":
        bot = net_policy(parsed.path, device=device, ensemble=parsed.ensemble,
                         name=parsed.label)
    elif parsed.kind == "search":
        bot = net_search(parsed.path, sims=parsed.sims, device=device,
                         ensemble=parsed.ensemble, name=parsed.label)
    else:                                                  # pragma: no cover
        raise ValueError(f"unhandled spec kind {parsed.kind!r}")
    bot.name = parsed.label
    return bot


@dataclass(frozen=True)
class BotFactory:
    """Picklable ``() -> Bot``.

    The arena's worker processes receive factories, not bots: a torch module
    is expensive (and on CUDA, illegal) to pickle across a process boundary,
    while a spec string costs nothing and rebuilds the bot on the far side.
    """

    spec: str
    device: str = "cpu"
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            object.__setattr__(self, "label", parse_spec(self.spec).label)

    @property
    def name(self) -> str:
        return self.label

    def __call__(self) -> Bot:
        bot = make_bot(self.spec, self.device)
        bot.name = self.label
        return bot


def make_factory(spec: str, device: str = "cpu",
                 label: str = "") -> BotFactory:
    """A picklable factory for ``spec`` (validated eagerly, built lazily)."""
    parsed = parse_spec(spec)
    return BotFactory(spec=parsed.raw, device=device,
                      label=label or parsed.label)
