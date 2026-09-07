"""Run configuration for the self-play trainer (``docs/AI_DESIGN.md`` §1.8).

Everything the trainer needs is a tree of dataclasses rooted at
:class:`RunConfig`.  The tree is loaded from YAML (``configs/*.yaml``) and any
leaf can be overridden on the command line::

    python -m splendor_ai.selfplay.train --config configs/smoke_cpu.yaml \
        --set selfplay.actors=4 --set learner.batch=512 \
        --set 'selfplay.mode_mixture={ind2: 1.0}'

The value of ``--set`` is parsed with ``yaml.safe_load``, so ``true``, ``12``,
``0.25``, ``[1,2]`` and ``{a: 1}`` all do what they look like, and anything
that fails to parse stays a string.  Unknown keys raise — a typo in a config
must never silently train something else.

Two conventions worth knowing:

* **Mode names.**  ``ind2 / ind3 / ind4 / ovt / team_adj / team_opp`` expand to
  ``(num_players, mode, team_layout)`` through :data:`MODE_SPECS`.  A mixture is
  a dict of name -> weight; weights are normalised, zero-weight entries drop out.
* **Curriculum.**  ``selfplay.phases`` is a list of
  ``{until_games: int|null, mixture: {...}, sims_full: int|null, sims_fast: int|null}``.
  The first phase whose ``until_games`` exceeds the number of finished games
  wins; ``until_games: null`` means "from here on".  An empty list falls back to
  ``selfplay.mode_mixture`` for the whole run.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..model import NetConfig
from ..rules.engine import MODE_INDIVIDUAL, MODE_ONE_V_TWO, MODE_TEAM
from ..search.mcts import SearchConfig

__all__ = [
    "MODE_SPECS", "SelfPlayConfig", "ReplayConfig", "LearnerConfig",
    "InferenceConfig", "EvalConfig", "RunConfig", "load_config",
    "apply_overrides", "config_to_dict", "dump_config", "PPO_NOT_READY",
]

#: ``name -> (num_players, engine mode, team layout)`` for every mixture key.
MODE_SPECS: Dict[str, Tuple[int, str, Optional[str]]] = {
    "ind2": (2, MODE_INDIVIDUAL, None),
    "ind3": (3, MODE_INDIVIDUAL, None),
    "ind4": (4, MODE_INDIVIDUAL, None),
    "ovt": (3, MODE_ONE_V_TWO, None),
    "team_adj": (4, MODE_TEAM, "ADJACENT"),
    "team_opp": (4, MODE_TEAM, "OPPOSITE"),
}


@dataclass
class PhaseConfig:
    """One curriculum phase (§1.8 "Curriculum")."""

    #: Finished-game count at which this phase ends; ``None`` = never.
    until_games: Optional[int] = None
    mixture: Dict[str, float] = field(default_factory=lambda: {"ind2": 1.0})
    #: Optional per-phase search budget override (the warm-up phase runs
    #: cheaper searches so the net gets off random quickly).
    sims_full: Optional[int] = None
    sims_fast: Optional[int] = None


@dataclass
class SelfPlayConfig:
    """Actors, playout-cap randomization, curriculum and opponent mixing."""

    actors: int = 2
    #: Concurrent games per actor process, searched in lockstep.
    games_per_actor: int = 8
    #: Playout Cap Randomization: probability a move gets the full, recorded,
    #: noisy search.  The rest get ``sims_fast`` with no noise and are dropped.
    pcr_full_prob: float = 0.25
    #: Plies (``state.turn_number``) during which the move is sampled from the
    #: visit distribution instead of being the argmax.
    temperature_plies: int = 12
    temperature: float = 1.0
    max_plies: int = 400
    #: Value-target weight for a game cut short at ``max_plies`` (§1.2).
    truncation_z_weight: float = 0.3
    mode_mixture: Dict[str, float] = field(default_factory=lambda: {"ind2": 1.0})
    phases: List[PhaseConfig] = field(default_factory=list)
    #: INDIVIDUAL-only engine override for short smoke games.  ``None`` keeps
    #: the real threshold (15).  NEVER set this for a deployment run.
    win_threshold: Optional[int] = None
    #: C5 colour augmentation on write: 1 = off, 5 = all rotations (§1.4).
    augment_rotations: int = 5
    #: Fraction of games where 1..2 seats are played by somebody other than the
    #: current net; only current-net seats are recorded.
    mixed_game_frac: float = 0.10
    #: Sampling weights for a non-current seat in a mixed game.
    opponent_weights: Dict[str, float] = field(
        default_factory=lambda: {"latest": 0.0, "historical": 0.6, "anchor": 0.4})
    #: How many past generation checkpoints stay eligible (0 = pool disabled).
    historical_pool_size: int = 8
    #: Historical nets kept warm per actor process (an LRU).
    historical_cache: int = 2
    #: Seconds an actor may reuse its cached listing of the opponent pool.
    #: ``os.listdir`` of a directory the learner writes to, once per mixed
    #: game, is pure overhead: the pool only changes once per generation.
    historical_pool_refresh_s: float = 60.0
    #: Probability that a historical seat reuses a checkpoint this actor has
    #: already loaded.  A 12.6M-parameter net costs ~200 ms to load; at the
    #: production pool size (16) and cache size (3) an unbiased draw would
    #: reload on nearly every mixed game and stall a whole wave of games.
    historical_reuse_prob: float = 0.85
    #: Seconds between stats emissions from each actor.
    stats_every_s: float = 20.0
    #: Seconds between ``weights/latest.pt`` freshness checks in an actor.
    weight_refresh_s: float = 10.0
    #: Records are shipped in chunks of this many games.
    ship_every_games: int = 1
    #: Store the root value from search alongside the outcome so the learner
    #: can blend them (``learner.value_blend``).
    store_root_value: bool = True
    #: Seconds an actor may go without re-reading ``run_dir/progress.json``
    #: (the run-global game counter the curriculum phase is a function of).
    #: Read on the weight-refresh cadence; this only bounds it from above.
    progress_refresh_s: float = 30.0
    #: Seconds the orchestrator gives actors (and inference servers) to finish
    #: the wave in flight and ship its records after a stop signal.  A wave is
    #: `games_per_actor` searches deep, so this has to exceed one wave's wall
    #: clock or `timeout --signal=INT` costs the run its last games.
    stop_grace_s: float = 30.0
    #: Per-actor restart budget: more than `restart_budget` restarts inside
    #: `restart_window_s` aborts the run instead of restarting for ever.
    restart_budget: int = 8
    restart_window_s: float = 900.0
    #: Restart backoff: `restart_backoff_s * 2**(n-1)`, capped.  A crash loop
    #: that restarts instantly burns a core and floods the log.
    restart_backoff_s: float = 2.0
    restart_backoff_max_s: float = 120.0


@dataclass
class ReplayConfig:
    """Rolling generational window (§1.8, judges.md "REPLAY BUFFER")."""

    window_start: int = 4
    window_end: int = 20
    #: Generations over which the window ramps ``window_start -> window_end``.
    window_ramp_generations: int = 40
    #: Hard cap on stored (augmented) samples; oldest generations go first.
    max_samples: int = 20_000_000
    #: Learner will not step until this many samples exist.
    min_samples: int = 4096
    #: Persist the buffer inside the periodic checkpoint (``replay.npz``).
    checkpoint: bool = True
    #: Write ``replay.npz`` from a background thread.  The save is ~41 s at the
    #: production window and the main loop owns the learner, so a synchronous
    #: save stalls training for that long every `checkpoint_every_s`.
    checkpoint_async: bool = True


@dataclass
class LearnerConfig:
    """AdamW + warmup/cosine, replay-ratio throttle, checkpointing."""

    #: ``az`` = AlphaZero targets from search; ``ppo`` = the fallback learner.
    #: ``ppo`` additionally needs ``ppo_experimental: true`` — the module is
    #: only partially implemented and would otherwise silently train
    #: off-policy on AlphaZero targets (see :func:`validate`).
    algorithm: str = "az"
    #: Acknowledge that ``algorithm: ppo`` is an unfinished experiment.
    ppo_experimental: bool = False
    device: str = "cpu"
    batch: int = 4096
    lr: float = 2e-4
    lr_final: float = 2e-5
    warmup_steps: int = 2000
    #: Horizon of the cosine decay in optimizer steps.
    cosine_steps: int = 400_000
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    #: Target sample reuse: the learner throttles so that
    #: ``steps*batch <= replay_ratio * samples_produced``.
    replay_ratio: float = 4.0
    #: Publish ``weights/latest.pt`` every N optimizer steps.
    publish_every: int = 50
    #: Full resumable state every N seconds.
    checkpoint_every_s: float = 600.0
    #: bf16 autocast (CUDA only; a no-op on CPU).
    bf16: bool = True
    #: Value target = (1-blend)*outcome + blend*root value from search.
    value_blend: float = 0.0
    #: Seconds the learner sleeps when the replay ratio is saturated.
    idle_sleep_s: float = 0.05
    #: Log a metrics line every N steps.
    log_every: int = 25
    #: Prepare the next batch on a background thread while the current step
    #: runs (numpy/torch release the GIL; batch prep is ~272 ms per 4096).
    prefetch: bool = True
    #: Reserved.  DDP is **not wired**: the orchestrator is single-rank (it
    #: would spawn a full set of actors, servers and evaluators on every rank
    #: and every rank would publish over the others' ``weights/latest.pt``).
    #: ``WORLD_SIZE > 1`` is refused by :class:`~.learner.Learner`.
    ddp: bool = False
    #: Prune ``checkpoints/gen_XXXX.pt``: keep the opponent pool plus the
    #: milestone generations, delete the rest (~125 GB over a long chain).
    checkpoint_retention: bool = True


@dataclass
class InferenceConfig:
    """Where the leaf evaluations happen."""

    #: ``inproc`` = a CPU evaluator inside each actor (the smoke layout);
    #: ``server`` = dedicated processes, one per device.
    mode: str = "inproc"
    devices: List[str] = field(default_factory=lambda: ["cpu"])
    max_batch: int = 1024
    max_wait_ms: float = 1.0
    #: Seconds between weight-file freshness checks in a server.
    reload_every_s: float = 30.0
    #: Queue depth per server (requests, not samples).
    queue_size: int = 256


@dataclass
class EvalConfig:
    """Periodic strength check against fixed anchors (§1.7, §2 G3/G5)."""

    enabled: bool = True
    every_generations: int = 1
    #: Paired (seat-swapped) games per opponent; 1 pair = 2 games.
    pairs: int = 12
    #: Pairs for the (much more expensive) search bot.
    search_pairs: int = 6
    sims_eval: int = 48
    opponents: List[str] = field(default_factory=lambda: ["random", "greedy"])
    #: Run the search bot evaluation at all.
    search_bot: bool = True
    #: Evaluation runs in its own process so it never blocks the learner.
    async_process: bool = True
    #: Wall-clock budget for the ONE final evaluation `shutdown` runs after
    #: the last checkpoint.  ``0`` = skip it when the run was stopped by a
    #: signal (the PBS chain's `timeout --signal=INT` case: the job has
    #: minutes left, and a production final eval takes tens of them).  A
    #: positive value runs it in a child process and kills it at the budget.
    final_eval_seconds: float = 0.0


@dataclass
class RunConfig:
    """The whole run."""

    run_dir: str = "runs/smoke"
    seed: int = 0
    net: NetConfig = field(default_factory=lambda: NetConfig(width=128, blocks=2))
    search_full: SearchConfig = field(default_factory=SearchConfig)
    search_fast: SearchConfig = field(default_factory=SearchConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    #: Finished games that close a generation.
    games_per_generation: int = 600
    #: Stop conditions (0/None = no limit); whichever hits first ends the run.
    max_generations: int = 0
    max_games: int = 0
    max_steps: int = 0
    max_seconds: float = 0.0
    #: Torch threads per process.  1 everywhere on CPU: a 4-thread forward of a
    #: tiny net is pathological (measured), and every actor is its own process.
    torch_threads: int = 1
    #: Extra threads the learner alone may use (CPU runs only).
    learner_threads: int = 1

    # -- derived paths ---------------------------------------------------
    @property
    def weights_dir(self) -> str:
        return os.path.join(self.run_dir, "weights")

    @property
    def latest_weights(self) -> str:
        return os.path.join(self.weights_dir, "latest.pt")

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.run_dir, "checkpoints")

    @property
    def state_path(self) -> str:
        return os.path.join(self.run_dir, "trainer_state.pt")

    @property
    def replay_path(self) -> str:
        return os.path.join(self.run_dir, "replay.npz")

    @property
    def metrics_path(self) -> str:
        return os.path.join(self.run_dir, "metrics.jsonl")

    def make_dirs(self) -> None:
        for d in (self.run_dir, self.weights_dir, self.checkpoints_dir):
            os.makedirs(d, exist_ok=True)

    # -- curriculum ------------------------------------------------------
    @property
    def progress_path(self) -> str:
        """``run_dir/progress.json`` — the run-global counters actors read.

        The curriculum is a function of RUN-GLOBAL progress, so it cannot be
        computed from an actor's own game counter: that counter restarts at 0
        on every resume and on every actor restart, which used to rewind the
        whole node to phase 0 (2p at reduced sims) after each link of the PBS
        chain.  The learner publishes ``{games_done, generation, instance}``
        here atomically; actors re-read it on their weight-refresh cadence.
        """
        return os.path.join(self.run_dir, "progress.json")

    def phase_for(self, games_done: int) -> PhaseConfig:
        """The curriculum phase in force after ``games_done`` RUN-GLOBAL
        finished games (see :attr:`progress_path`)."""
        if not self.selfplay.phases:
            return PhaseConfig(until_games=None,
                               mixture=dict(self.selfplay.mode_mixture))
        for phase in self.selfplay.phases:
            if phase.until_games is None or games_done < phase.until_games:
                return phase
        return self.selfplay.phases[-1]


# ── dataclass <-> dict ────────────────────────────────────────────────────

def _dataclass_from(cls, data: Any, path: str):
    """Build ``cls`` from a plain dict, refusing unknown keys."""
    if data is None:
        return cls()
    if isinstance(data, cls):
        return data
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping for {cls.__name__}, "
                         f"got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {unknown} for {cls.__name__}; "
            f"known keys are {sorted(known)}")
    kwargs = {}
    for name, value in data.items():
        f = known[name]
        sub = f"{path}.{name}" if path else name
        if is_dataclass(f.type) and isinstance(f.type, type):
            kwargs[name] = _dataclass_from(f.type, value, sub)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _coerce(cfg: RunConfig) -> RunConfig:
    """Post-process a freshly built RunConfig: nested dataclasses and tuples."""
    cfg.net = _dataclass_from(NetConfig, cfg.net, "net")
    cfg.search_full = _dataclass_from(SearchConfig, cfg.search_full, "search_full")
    cfg.search_fast = _dataclass_from(SearchConfig, cfg.search_fast, "search_fast")
    cfg.selfplay = _dataclass_from(SelfPlayConfig, cfg.selfplay, "selfplay")
    cfg.replay = _dataclass_from(ReplayConfig, cfg.replay, "replay")
    cfg.learner = _dataclass_from(LearnerConfig, cfg.learner, "learner")
    cfg.inference = _dataclass_from(InferenceConfig, cfg.inference, "inference")
    cfg.eval = _dataclass_from(EvalConfig, cfg.eval, "eval")
    cfg.selfplay.phases = [_dataclass_from(PhaseConfig, p, "selfplay.phases")
                           for p in (cfg.selfplay.phases or [])]
    for search in (cfg.search_full, cfg.search_fast):
        if isinstance(search.deck_reserve_penalty, list):
            search.deck_reserve_penalty = tuple(search.deck_reserve_penalty)
    validate(cfg)
    return cfg


def validate(cfg: RunConfig) -> None:
    """Loud, early failure for the mistakes that would otherwise show up as a
    silently wrong training run."""
    sp = cfg.selfplay
    if sp.actors < 1 or sp.games_per_actor < 1:
        raise ValueError("selfplay.actors and selfplay.games_per_actor must be >= 1")
    if not 0.0 <= sp.pcr_full_prob <= 1.0:
        raise ValueError("selfplay.pcr_full_prob must be in [0, 1]")
    if not 1 <= sp.augment_rotations <= 5:
        raise ValueError("selfplay.augment_rotations must be in 1..5 (the C5 group)")
    mixtures = [sp.mode_mixture] + [p.mixture for p in sp.phases]
    for mixture in mixtures:
        unknown = sorted(set(mixture) - set(MODE_SPECS))
        if unknown:
            raise ValueError(f"unknown mode(s) {unknown}; known: {sorted(MODE_SPECS)}")
        if sum(float(v) for v in mixture.values()) <= 0:
            raise ValueError(f"mode mixture {mixture} has no positive weight")
    if sp.win_threshold is not None:
        bad = [m for m in mixtures
               for name, w in m.items()
               if float(w) > 0 and MODE_SPECS[name][1] != MODE_INDIVIDUAL]
        if bad:
            raise ValueError(
                "selfplay.win_threshold is an INDIVIDUAL-only smoke shortcut, "
                "but the mixture contains team modes")
    if cfg.inference.mode not in ("inproc", "server"):
        raise ValueError("inference.mode must be 'inproc' or 'server'")
    if cfg.learner.algorithm not in ("az", "ppo"):
        raise ValueError("learner.algorithm must be 'az' or 'ppo'")
    if cfg.learner.algorithm == "ppo" and not cfg.learner.ppo_experimental:
        raise ValueError(PPO_NOT_READY)
    if cfg.replay.window_start < 1 or cfg.replay.window_end < cfg.replay.window_start:
        raise ValueError("replay window must satisfy 1 <= window_start <= window_end")
    _warn_forced_playouts(cfg)


#: Why ``learner.algorithm: ppo`` refuses to start without an explicit
#: acknowledgement.  The PPO module is real but *unfinished*, and the pieces
#: that are missing are exactly the ones that make the difference between PPO
#: and "gradient ascent on somebody else's data": the actors still run MCTS and
#: write AlphaZero targets, so the learner would train off-policy on a policy
#: it never sampled from, with a ratio that starts at 1 by construction.
PPO_NOT_READY = (
    "learner.algorithm: ppo is not finished and would silently train "
    "off-policy on AlphaZero search targets.  Unimplemented (see the TODO at "
    "the top of splendor_ai/selfplay/ppo_learner.py):\n"
    "  1. actor.py has no search-free branch — it still builds an MCTS and "
    "records the search visit distribution, so no action was ever sampled "
    "from the policy being updated and no log_prob or value estimate is "
    "stored; PPOLearner.train_step recovers the 'action' as the argmax of the "
    "search target and recomputes logp_old from the CURRENT weights, which "
    "makes the importance ratio identically 1 on the first epoch;\n"
    "  2. the replay buffer is a generational window, not an on-policy "
    "rollout: a PPO iteration must consume the last rollout_games and drop "
    "them;\n"
    "  3. records carry no logp/value columns (RECORD_DTYPE has spare bytes "
    "for them);\n"
    "  4. advantages are normalised globally rather than per seat, and the "
    "margin-scaled terminal reward has never been A/B'd against plain +-1.\n"
    "Set learner.ppo_experimental: true to run it anyway (a diagnostic, never "
    "a production run)."
)


#: Forced playouts cost about ``sqrt(k * sims * num_legal)`` simulations at the
#: root.  Below this many simulations that is a large fraction of the whole
#: budget, the visit distribution flattens, and the policy target becomes noise
#: (measured in the G3 smoke run; see ``configs/smoke_cpu.yaml``).
FORCED_PLAYOUT_MIN_SIMS = 200


def _warn_forced_playouts(cfg: "RunConfig") -> None:
    k = cfg.search_full.forced_playouts_k
    sims = cfg.search_full.sims
    if k > 0 and sims < FORCED_PLAYOUT_MIN_SIMS:
        import warnings

        warnings.warn(
            f"search_full: forced playouts (k={k}) with only {sims} sims — "
            f"forcing sqrt(k*P*N) visits on every root action spends roughly "
            f"sqrt({k}*{sims}*num_legal) simulations, which at this budget "
            f"flattens the visit distribution and turns the policy target into "
            f"noise.  Set search_full.forced_playouts_k=0 (and "
            f"prune_policy_target=false) below ~{FORCED_PLAYOUT_MIN_SIMS} sims.",
            RuntimeWarning, stacklevel=2)


def config_to_dict(cfg: Any) -> Any:
    """Recursively turn a config tree into plain YAML-able data."""
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return {f.name: config_to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, dict):
        return {k: config_to_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [config_to_dict(v) for v in cfg]
    return cfg


def dump_config(cfg: RunConfig, path: str) -> str:
    """Write the fully-resolved config next to the run (atomic)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)
    os.replace(tmp, path)
    return path


def from_dict(data: Dict[str, Any]) -> RunConfig:
    cfg = _dataclass_from(RunConfig, data or {}, "")
    return _coerce(cfg)


#: Nested sections of :class:`RunConfig`.  The tree cannot be walked through
#: ``dataclasses.fields(...).type`` because ``from __future__ import
#: annotations`` turns every annotation into a string, so the map is explicit
#: (``_coerce`` builds the same set).
_SECTIONS: Dict[str, type] = {
    "net": NetConfig,
    "search_full": SearchConfig,
    "search_fast": SearchConfig,
    "selfplay": SelfPlayConfig,
    "replay": ReplayConfig,
    "learner": LearnerConfig,
    "inference": InferenceConfig,
    "eval": EvalConfig,
}


def _check_override_key(key: str, item: str) -> None:
    """Refuse a ``--set`` path RunConfig does not have.

    ``--set`` writes into a plain dict, so an unknown *top-level* key would
    otherwise only fail later inside :func:`from_dict` with no mention of the
    flag that introduced it (and a free-form section such as
    ``selfplay.mode_mixture.ind2`` must still be allowed to invent keys).
    Only the first two levels are schema-checked; below them anything goes.
    """
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError(f"--set expects key=value, got {item!r}")
    known = {f.name for f in fields(RunConfig)}
    if parts[0] not in known:
        raise ValueError(
            f"--set {item!r}: RunConfig has no field {parts[0]!r}; "
            f"known keys are {sorted(known)}")
    section = _SECTIONS.get(parts[0])
    if section is None:
        if len(parts) > 1:
            raise ValueError(
                f"--set {item!r}: {parts[0]!r} is a single value, not a "
                f"section — it has no {'.'.join(parts[1:])!r} below it")
        return
    if len(parts) < 2:
        return
    sub = {f.name for f in fields(section)}
    if parts[1] not in sub:
        raise ValueError(
            f"--set {item!r}: {section.__name__} ({parts[0]}) has no field "
            f"{parts[1]!r}; known keys are {sorted(sub)}")


def _parse_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(data: Dict[str, Any], overrides) -> Dict[str, Any]:
    """Apply ``key.path=value`` strings onto a raw config dict."""
    data = copy.deepcopy(data)
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        _check_override_key(key.strip(), item)
        node = data
        parts = key.strip().split(".")
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = _parse_value(raw.strip())
    return data


def load_config(path: Optional[str] = None, overrides=None) -> RunConfig:
    """Load YAML (if given), apply ``--set`` overrides, validate."""
    data: Dict[str, Any] = {}
    if path:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top level must be a mapping")
    data = apply_overrides(data, overrides)
    return from_dict(data)


def normalise_mixture(mixture: Dict[str, float]) -> Tuple[List[str], List[float]]:
    """``{name: weight}`` -> parallel lists with the weights summing to 1."""
    names = [k for k, v in mixture.items() if float(v) > 0]
    weights = [float(mixture[k]) for k in names]
    total = sum(weights)
    return names, [w / total for w in weights]
