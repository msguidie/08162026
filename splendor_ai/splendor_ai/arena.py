"""Paired-seed, seat-rotated arena with a joint Bradley–Terry fit.

``docs/AI_DESIGN.md`` §1.7 and the evaluation sections of
``docs/research/judges.md`` are the contract.  Four design rules come straight
from the judges' post-mortem of the profiled projects, and every one of them
is load-bearing:

1. **Paired seeds.**  Both games of a pair use the *same* deck seed and differ
   only in the seating, which cancels deal variance instead of adding an
   independent noisy sample.  Here every seating of a *seed group* shares one
   seed, and the seed depends only on ``(base_seed, mode, group)`` — so two
   different pairings, and two different arena runs, see the *same* deals.
2. **Full seat rotation / Latin squares.**  2p swaps.  For 3p/4p a pairing is
   seated alternately (``A B A``) and the group plays every cyclic rotation
   plus, for odd tables, the mirrored composition, which makes each bot's seat
   occupancy exactly balanced.  Tables of ``n`` *distinct* bots play a cyclic
   Latin square (each bot in each seat exactly once).  Never cestpasphoto's
   biased 1-vs-(n-1) arrangement.
3. **Team roles rotate.**  In TEAM and ONE_V_TWO a side is always homogeneous
   (so "who won" stays attributable) and the group plays both side
   assignments plus the within-side seat orders.  ONE_V_TWO results are broken
   out by role: an agent can be exploitable as the solo seat while looking
   fine on average.
4. **One joint fit, pinned anchors.**  Ratings come from a single
   Bradley–Terry (BayesElo-style) fit over the *whole* win/loss matrix with
   ``random`` pinned at 0 Elo — never a chain of pairwise deltas, and never a
   self-referential ladder against the previous generation.

Outcomes keep ``STALE`` (a stuck seat had to resign — the variant has no pass)
and ``TRUNCATED`` (``max_plies``) as their own buckets, as Rinascimento's
``WinStaleLostStatsFull`` does; a deadlock rate that drifts upward is a
first-class alarm, not a rounding error.

Command line::

    python -m splendor_ai.arena --bots random greedy mcts40 \\
        --modes ind2 ovt team --games 100 --out reports/arena.md

which writes ``reports/arena.md`` and ``reports/arena.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

import numpy as np

from . import anchors as anchors_mod
from .bots import play_game
from .rules import engine as E

__all__ = [
    "ModeSpec", "MODES", "DEFAULT_MODES", "parse_mode", "parse_modes",
    "seat_arrangements", "pair_compositions", "build_schedule",
    "run_matches", "ArenaResults", "fit_bradley_terry", "Ratings",
    "build_report", "render_markdown", "write_reports", "main",
]

#: Elo points per natural log unit of Bradley–Terry strength.
ELO_SCALE = 400.0 / math.log(10.0)

#: Default truncation cap, matching ``bots.play_game`` and §1.2.
MAX_PLIES = 400


# ── modes ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModeSpec:
    """One table configuration: the engine mode, the seat count, the layout.

    ``key`` is the short name used everywhere else in the stack — it is the
    same key the deployment worker uses to pick a checkpoint
    (``worker/config.py::mode_key``), so ``ind2`` here and ``ind2.pt`` there
    mean the same thing.
    """

    key: str
    mode: str
    num_players: int
    layout: Optional[str] = None

    @property
    def label(self) -> str:
        base = f"{self.mode} {self.num_players}p"
        return f"{base} {self.layout}" if self.layout else base

    @property
    def sides(self) -> Optional[Tuple[int, ...]]:
        """Seat → side id, or ``None`` in INDIVIDUAL (every seat its own side)."""
        ids = E.default_team_ids(self.mode, self.layout, self.num_players)
        return None if ids is None else tuple(int(i) for i in ids)

    @property
    def roles(self) -> Tuple[str, ...]:
        """Seat → role name.  Only ONE_V_TWO has asymmetric named roles."""
        if self.mode == E.MODE_ONE_V_TWO:
            return tuple("solo" if s == 0 else "duo"
                         for s in range(self.num_players))
        return tuple(f"seat{s}" for s in range(self.num_players))

    @property
    def has_named_roles(self) -> bool:
        return self.mode == E.MODE_ONE_V_TWO

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "mode": self.mode,
                "num_players": self.num_players, "layout": self.layout,
                "label": self.label}


#: Every table configuration the stack supports.  The first five are the
#: contract's five modes; ``team_opp`` is the OPPOSITE seating of 2v2, worth
#: running occasionally because it changes the turn order inside a team.
MODES: Dict[str, ModeSpec] = {
    "ind2": ModeSpec("ind2", E.MODE_INDIVIDUAL, 2),
    "ind3": ModeSpec("ind3", E.MODE_INDIVIDUAL, 3),
    "ind4": ModeSpec("ind4", E.MODE_INDIVIDUAL, 4),
    "ovt": ModeSpec("ovt", E.MODE_ONE_V_TWO, 3),
    "team": ModeSpec("team", E.MODE_TEAM, 4, "ADJACENT"),
    "team_opp": ModeSpec("team_opp", E.MODE_TEAM, 4, "OPPOSITE"),
}

DEFAULT_MODES: Tuple[str, ...] = ("ind2", "ind3", "ind4", "ovt", "team")


def parse_mode(spec: Any) -> ModeSpec:
    """``'ind2'``, ``'INDIVIDUAL:3'``, ``'TEAM:4:OPPOSITE'`` or a ModeSpec."""
    if isinstance(spec, ModeSpec):
        return spec
    text = str(spec).strip()
    if text in MODES:
        return MODES[text]
    if text.lower() in MODES:
        return MODES[text.lower()]
    parts = [p for p in text.replace("/", ":").split(":") if p]
    if len(parts) >= 2:
        mode = parts[0].upper()
        try:
            n = int(parts[1])
        except ValueError:
            raise ValueError(f"bad mode spec {spec!r}") from None
        layout = parts[2].upper() if len(parts) > 2 else (
            "ADJACENT" if mode == E.MODE_TEAM else None)
        key = f"{mode.lower()}{n}" + (f"_{layout.lower()}" if layout else "")
        return ModeSpec(key, mode, n, layout)
    raise ValueError(
        f"unknown mode {spec!r}; known keys: {sorted(MODES)} "
        f"(or 'MODE:num_players[:layout]')")


def parse_modes(specs: Optional[Iterable[Any]] = None) -> List[ModeSpec]:
    if specs is None:
        specs = DEFAULT_MODES
    if isinstance(specs, (str, ModeSpec)):
        specs = [specs]
    return [parse_mode(s) for s in specs]


# ── seatings ──────────────────────────────────────────────────────────────

def _dedupe(seq: Iterable[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
    """Order-preserving dedupe (homogeneous sides collapse many rotations)."""
    seen: Dict[Tuple[str, ...], None] = {}
    for item in seq:
        seen.setdefault(tuple(item), None)
    return list(seen)


def seat_arrangements(mode: ModeSpec,
                      base: Sequence[str]) -> List[Tuple[str, ...]]:
    """Every seating of one table composition, as ``tuple(name_per_seat)``.

    INDIVIDUAL: all ``n`` cyclic rotations of ``base`` — with ``n`` distinct
    bots that is a cyclic Latin square (each bot in each seat exactly once).

    TEAM / ONE_V_TWO: the two side assignments crossed with the rotations
    *inside* each side, so every bot plays every role of both sides.  A plain
    cyclic rotation would be wrong here — rotating ``A A B B`` by one seat
    produces the mixed teams ``B A | A B``, which is a different (and, for
    rating purposes, uninformative) matchup.

    Duplicates are dropped, so a homogeneous side yields the two side
    assignments and nothing else.
    """
    n = mode.num_players
    base = tuple(base)
    if len(base) != n:
        raise ValueError(f"{mode.key} seats {n} bots, got {len(base)}")
    sides = mode.sides
    if sides is None:
        return _dedupe(tuple(base[(i + r) % n] for i in range(n))
                       for r in range(n))

    groups = [[seat for seat in range(n) if sides[seat] == side]
              for side in sorted(set(sides))]
    if len(groups) != 2:                                   # pragma: no cover
        raise ValueError(f"{mode.key} does not have exactly two sides")
    out: List[Tuple[str, ...]] = []
    shifts = max(len(g) for g in groups)
    for swap in (False, True):
        for shift in range(shifts):
            assign: List[Optional[str]] = [None] * n
            for side, dst in enumerate(groups):
                src = groups[1 - side] if swap else groups[side]
                for j, seat in enumerate(dst):
                    assign[seat] = base[src[(j + shift) % len(src)]]
            out.append(tuple(a for a in assign if a is not None))
    return _dedupe(out)


def pair_compositions(mode: ModeSpec, a: str,
                      b: str) -> List[Tuple[str, ...]]:
    """Base compositions for the pairing ``{a, b}`` in ``mode``.

    INDIVIDUAL seats the two bots alternately (``A B``, ``A B A``,
    ``A B A B``).  For an odd table the mirrored composition is added as well,
    so neither bot permanently holds the majority of the seats: over the
    6 seatings of a 3p group each bot sits in each seat exactly three times.

    TEAM / ONE_V_TWO give side 0 to ``a`` and side 1 to ``b``;
    :func:`seat_arrangements` then plays the swap as well.
    """
    n = mode.num_players
    sides = mode.sides
    if sides is None:
        comps = [tuple((a, b)[i % 2] for i in range(n))]
        if n % 2:
            comps.append(tuple((b, a)[i % 2] for i in range(n)))
        return comps
    return [tuple(a if sides[seat] == 0 else b for seat in range(n))]


@dataclass(frozen=True)
class Table:
    """One composition to be played, plus the seatings it rotates through."""

    mode: str
    index: int
    kind: str                              # 'pair' | 'mixed'
    base: Tuple[str, ...]
    arrangements: Tuple[Tuple[str, ...], ...]
    members: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "index": self.index, "kind": self.kind,
                "base": list(self.base), "members": list(self.members),
                "arrangements": [list(a) for a in self.arrangements]}


def build_tables(mode: ModeSpec, names: Sequence[str], mixed: bool = True,
                 max_mixed_tables: int = 4) -> List[Table]:
    """Every table played in ``mode``: all pairings, then the mixed tables.

    Mixed tables (``n`` distinct bots at one table, Latin-square seated) only
    exist for INDIVIDUAL modes with ``n >= 3``: in the team modes a mixed
    table would mean mixed sides, and a win by a side of two different bots
    cannot be attributed to either of them.
    """
    tables: List[Table] = []
    for a, b in itertools.combinations(names, 2):
        for base in pair_compositions(mode, a, b):
            tables.append(Table(
                mode=mode.key, index=len(tables), kind="pair", base=base,
                arrangements=tuple(seat_arrangements(mode, base)),
                members=(a, b)))
    n = mode.num_players
    if mixed and mode.sides is None and n >= 3 and len(names) >= n:
        combos = list(itertools.combinations(names, n))[:max(0, max_mixed_tables)]
        for combo in combos:
            tables.append(Table(
                mode=mode.key, index=len(tables), kind="mixed", base=combo,
                arrangements=tuple(seat_arrangements(mode, combo)),
                members=tuple(combo)))
    return tables


# ── schedule ──────────────────────────────────────────────────────────────

def group_seed(base_seed: int, mode_key: str, group: int) -> int:
    """A 62-bit deal seed from ``(base_seed, mode, group)``.

    Deliberately independent of the pairing: every table in a mode plays the
    *same* sequence of deals (common random numbers), which is the paired-seed
    idea extended across pairings and across arena runs.  ``hashlib`` rather
    than ``hash()`` because the latter is salted per process.
    """
    digest = hashlib.blake2b(
        f"{int(base_seed)}|{mode_key}|{int(group)}".encode("utf-8"),
        digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 62) - 1)


def build_schedule(names: Sequence[str], modes: Sequence[ModeSpec],
                   games_per_pairing: int = 100, seed: int = 0,
                   mixed: bool = True, max_mixed_tables: int = 4
                   ) -> Tuple[List[Dict[str, Any]], List[Table]]:
    """``(jobs, tables)`` — the full, deterministic game list.

    Each table is played for ``ceil(games_per_pairing / len(arrangements))``
    seed groups, so the realised game count per table is
    ``groups * len(arrangements)`` — at least ``games_per_pairing``, rounded up
    to a whole number of complete rotations (a partial rotation would
    reintroduce exactly the seat bias the rotation removes).
    """
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate bot names: {names}")
    if len(names) < 2:
        raise ValueError("need at least two bots to run an arena")
    jobs: List[Dict[str, Any]] = []
    tables: List[Table] = []
    for mode in modes:
        for table in build_tables(mode, names, mixed, max_mixed_tables):
            table = Table(mode=table.mode, index=len(tables), kind=table.kind,
                          base=table.base, arrangements=table.arrangements,
                          members=table.members)
            tables.append(table)
            per_group = len(table.arrangements)
            groups = max(1, math.ceil(games_per_pairing / per_group))
            for group in range(groups):
                gseed = group_seed(seed, mode.key, group)
                for arrangement in table.arrangements:
                    jobs.append({
                        "i": len(jobs), "mode": mode.key, "table": table.index,
                        "kind": table.kind, "group": group, "seed": gseed,
                        "seats": list(arrangement),
                    })
    return jobs, tables


# ── worker side ───────────────────────────────────────────────────────────

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

_STATE: Dict[str, Any] = {}


def _pin_threads() -> None:
    """One BLAS/OMP thread per process.

    Arena parallelism is over *games*; a torch or numpy op that also spawns
    threads would oversubscribe the node and make the wall clock (and the
    throughput numbers in the report) meaningless.
    """
    for var in _THREAD_VARS:
        os.environ[var] = "1"
    torch = sys.modules.get("torch")
    if torch is not None:                                  # already imported
        try:
            torch.set_num_threads(1)
        except Exception:                                  # pragma: no cover
            pass


def _init_worker(factories: Mapping[str, Any], mode_specs: Dict[str, Any],
                 max_plies: int) -> None:
    _pin_threads()
    _STATE["factories"] = dict(factories)
    _STATE["modes"] = {k: ModeSpec(**{kk: vv for kk, vv in v.items()
                                      if kk in ("key", "mode", "num_players",
                                                "layout")})
                       for k, v in mode_specs.items()}
    _STATE["max_plies"] = int(max_plies)
    _STATE["bots"] = {}


def _bot(name: str):
    """The worker's bot for ``name``, built once and reused.

    Reuse is safe because no bot in the stack carries state across games (the
    MCTS tree is rebuilt per move and the rollout policy is deterministic),
    and it is necessary because building a net bot means reading a checkpoint
    off disk.
    """
    cache = _STATE["bots"]
    bot = cache.get(name)
    if bot is None:
        factory = _STATE["factories"][name]
        bot = factory() if callable(factory) else factory
        try:
            bot.name = name
        except Exception:                                  # pragma: no cover
            pass
        cache[name] = bot
    return bot


def _play_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Play one scheduled game and return a compact, JSON-safe record."""
    mode = _STATE["modes"][job["mode"]]
    seats = list(job["seats"])
    bots = [_bot(name) for name in seats]
    start = time.perf_counter()
    out = play_game(bots, mode.mode, mode.num_players, mode.layout,
                    seed=int(job["seed"]), max_plies=_STATE["max_plies"])
    n = mode.num_players
    return {
        "i": job["i"], "mode": job["mode"], "table": job["table"],
        "kind": job["kind"], "group": job["group"], "seed": job["seed"],
        "seats": seats,
        "values": [float(v) for v in out["values"][:n]],
        "scores": [int(s) for s in out["scores"]],
        "plies": int(out["plies"]),
        "reason": str(out["reason"]),
        "truncated": bool(out["truncated"]),
        "stuck": int(out["stuck_resigns"]),
        "resigned": [int(r) for r in out["resigned"]],
        "seconds": time.perf_counter() - start,
    }


# ── running ───────────────────────────────────────────────────────────────

def _normalise_bots(bots: Mapping[str, Any], device: str
                    ) -> Dict[str, Any]:
    """Accept factories, bots or plain spec strings as the values."""
    out: Dict[str, Any] = {}
    for name, value in bots.items():
        if isinstance(value, str):
            value = anchors_mod.make_factory(value, device=device, label=name)
        out[str(name)] = value
    return out


def run_matches(bots: Mapping[str, Any], modes: Sequence[Any],
                games_per_pairing: int = 100, seed: int = 0, workers: int = 1,
                max_plies: int = MAX_PLIES, mixed: bool = True,
                max_mixed_tables: int = 4, progress: Any = None,
                mp_context: str = "spawn") -> "ArenaResults":
    """Play the whole schedule and return the raw results.

    ``bots`` maps a report name to a **factory** (``() -> Bot``) — a bot is
    built inside the worker that will use it, because a torch module is
    expensive to pickle and a CUDA one cannot be pickled at all.  A spec
    string (``'mcts160'``, ``'search:runs/x/weights/latest.pt:400'``) is
    accepted as shorthand and turned into ``anchors.make_factory``; a ready
    bot object works too when ``workers == 1``.

    ``workers > 1`` uses a ``multiprocessing`` pool whose children have their
    BLAS/OMP thread counts pinned to 1.  Results are reassembled in schedule
    order, so the report is bit-identical whatever the worker count.
    """
    factories = _normalise_bots(dict(bots), device="cpu")
    names = list(factories)
    mode_specs = parse_modes(modes)
    jobs, tables = build_schedule(names, mode_specs, games_per_pairing, seed,
                                  mixed, max_mixed_tables)
    mode_payload = {m.key: m.to_dict() for m in mode_specs}
    started = time.time()
    records: List[Optional[Dict[str, Any]]] = [None] * len(jobs)
    done = 0

    def note(record: Dict[str, Any]) -> None:
        nonlocal done
        records[record["i"]] = record
        done += 1
        if progress is not None:
            progress(done, len(jobs), record)

    if workers and workers > 1 and len(jobs) > 1:
        import multiprocessing as mp

        previous = {var: os.environ.get(var) for var in _THREAD_VARS}
        for var in _THREAD_VARS:                    # children inherit these
            os.environ[var] = "1"
        try:
            ctx = mp.get_context(mp_context)
            chunk = max(1, min(16, len(jobs) // (workers * 4) or 1))
            with ctx.Pool(processes=int(workers), initializer=_init_worker,
                          initargs=(factories, mode_payload,
                                    max_plies)) as pool:
                for record in pool.imap_unordered(_play_job, jobs,
                                                  chunksize=chunk):
                    note(record)
        finally:
            for var, value in previous.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
    else:
        _init_worker(factories, mode_payload, max_plies)
        for job in jobs:
            note(_play_job(job))

    games = [r for r in records if r is not None]
    if len(games) != len(jobs):                            # pragma: no cover
        raise RuntimeError(f"{len(jobs) - len(games)} games did not come back")
    config = {
        "bots": {name: getattr(f, "spec", getattr(f, "name", str(f)))
                 for name, f in factories.items()},
        "modes": [m.to_dict() for m in mode_specs],
        "games_per_pairing": int(games_per_pairing),
        "seed": int(seed), "workers": int(workers),
        "max_plies": int(max_plies), "mixed": bool(mixed),
        "max_mixed_tables": int(max_mixed_tables),
        "scheduled_games": len(jobs),
        "started": started, "elapsed": time.time() - started,
    }
    return ArenaResults(bots=names, modes=mode_specs, games=games,
                        tables=tables, config=config)


# ── results ───────────────────────────────────────────────────────────────

def game_values(record: Mapping[str, Any]) -> Dict[str, float]:
    """Per-bot value: the mean of §1.2 terminal values over the seats it held.

    A bot that holds two seats of a 4p table (the alternating pairing seating)
    is judged on how both of them did, which is the only reading under which
    "A beat B" means the same thing at every table size.
    """
    totals: Dict[str, List[float]] = {}
    for seat, name in enumerate(record["seats"]):
        totals.setdefault(name, []).append(float(record["values"][seat]))
    return {name: sum(v) / len(v) for name, v in totals.items()}


def table_credit(record: Mapping[str, Any]) -> Dict[str, float]:
    """Win credit per bot at this table: 1 top, 0.5 all-tie, 0 otherwise."""
    scores = game_values(record)
    best = max(scores.values())
    winners = [n for n, v in scores.items() if v >= best - 1e-9]
    if len(winners) == len(scores):
        return {n: 0.5 for n in scores}
    return {n: (1.0 if n in winners else 0.0) for n in scores}


def seat_credit(values: Sequence[float]) -> List[float]:
    """Win credit per *seat* — the standing seat-asymmetry detector."""
    vals = [float(v) for v in values]
    best = max(vals)
    winners = [i for i, v in enumerate(vals) if v >= best - 1e-9]
    if len(winners) == len(vals):
        return [0.5] * len(vals)
    return [1.0 if i in winners else 0.0 for i in range(len(vals))]


def pairwise_from_game(record: Mapping[str, Any]
                       ) -> List[Tuple[Tuple[str, str], float]]:
    """``[((a, b), score_of_a)]`` for every distinct pair at the table.

    ``score_of_a`` is 1 / 0.5 / 0 — a draw is half a win, which is exactly how
    the Bradley–Terry likelihood below consumes it.  Pairs are canonically
    ordered so counts from different seatings accumulate together.
    """
    scores = game_values(record)
    out = []
    for a, b in itertools.combinations(sorted(scores), 2):
        diff = scores[a] - scores[b]
        out.append(((a, b), 1.0 if diff > 1e-9 else
                    (0.0 if diff < -1e-9 else 0.5)))
    return out


@dataclass
class ArenaResults:
    """Everything :func:`run_matches` produced, plus the aggregations."""

    bots: List[str]
    modes: List[ModeSpec]
    games: List[Dict[str, Any]]
    tables: List[Table] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    # -- selection -------------------------------------------------------
    def mode_keys(self) -> List[str]:
        return [m.key for m in self.modes]

    def mode(self, key: str) -> ModeSpec:
        for m in self.modes:
            if m.key == key:
                return m
        raise KeyError(key)                                # pragma: no cover

    def select(self, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        if mode is None:
            return list(self.games)
        return [g for g in self.games if g["mode"] == mode]

    # -- pairwise --------------------------------------------------------
    def pair_counts(self, mode: Optional[str] = None
                    ) -> Dict[Tuple[str, str], List[float]]:
        """``{(a, b): [wins_a, draws, wins_b]}`` (``a < b``)."""
        counts: Dict[Tuple[str, str], List[float]] = {}
        for record in self.select(mode):
            for pair, score in pairwise_from_game(record):
                slot = counts.setdefault(pair, [0.0, 0.0, 0.0])
                slot[0 if score > 0.75 else (1 if score > 0.25 else 2)] += 1
        return counts

    def units(self, mode: Optional[str] = None
              ) -> List[Dict[Tuple[str, str], List[float]]]:
        """Bootstrap units: one per *seed group*, not per game.

        Games inside a group share a deal and differ only in the seating, so
        they are not independent; resampling whole groups is what makes the
        confidence intervals honest.
        """
        by_group: Dict[Tuple[str, int, int], Dict[Tuple[str, str],
                                                  List[float]]] = {}
        for record in self.select(mode):
            key = (record["mode"], record["table"], record["group"])
            bucket = by_group.setdefault(key, {})
            for pair, score in pairwise_from_game(record):
                slot = bucket.setdefault(pair, [0.0, 0.0, 0.0])
                slot[0 if score > 0.75 else (1 if score > 0.25 else 2)] += 1
        return list(by_group.values())

    # -- per-bot ---------------------------------------------------------
    def bot_stats(self, mode: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Games, win rate, outcome buckets, seat/role splits, per bot."""
        spec = self.mode(mode) if mode else None
        n_seats = spec.num_players if spec else 4
        stats: Dict[str, Dict[str, Any]] = {
            name: {
                "games": 0, "wins": 0.0, "pair_points": 0.0,
                "pair_games": 0, "plies": 0, "stuck_games": 0,
                "truncated": 0, "forfeit": 0, "points": 0.0, "seats": 0,
                "seat_games": [0] * n_seats, "seat_wins": [0.0] * n_seats,
                "roles": {}, "credits_by_unit": {},
            } for name in self.bots
        }
        for record in self.select(mode):
            mspec = spec or self.mode(record["mode"])
            roles = mspec.roles
            credit = table_credit(record)
            seats_credit = seat_credit(record["values"])
            unit = (record["mode"], record["table"], record["group"])
            for name, value in credit.items():
                s = stats[name]
                s["games"] += 1
                s["wins"] += value
                s["plies"] += int(record["plies"])
                s["stuck_games"] += 1 if record["stuck"] else 0
                s["truncated"] += 1 if record["truncated"] else 0
                s["forfeit"] += 1 if record["reason"] == "FORFEIT" else 0
                s["credits_by_unit"].setdefault(unit, []).append(value)
            for seat, name in enumerate(record["seats"]):
                s = stats[name]
                s["seats"] += 1
                s["points"] += float(record["scores"][seat])
                if seat < len(s["seat_games"]):
                    s["seat_games"][seat] += 1
                    s["seat_wins"][seat] += seats_credit[seat]
                role = s["roles"].setdefault(roles[seat],
                                             {"games": 0, "wins": 0.0})
                role["games"] += 1
                role["wins"] += seats_credit[seat]
            for pair, score in pairwise_from_game(record):
                stats[pair[0]]["pair_points"] += score
                stats[pair[0]]["pair_games"] += 1
                stats[pair[1]]["pair_points"] += 1.0 - score
                stats[pair[1]]["pair_games"] += 1
        for name, s in stats.items():
            games = max(1, s["games"])
            s["win_rate"] = s["wins"] / games
            s["score_rate"] = (s["pair_points"] / s["pair_games"]
                               if s["pair_games"] else float("nan"))
            s["mean_plies"] = s["plies"] / games
            s["stuck_rate"] = s["stuck_games"] / games
            s["trunc_rate"] = s["truncated"] / games
            s["forfeit_rate"] = s["forfeit"] / games
            s["mean_points"] = s["points"] / max(1, s["seats"])
        return stats

    def mode_stats(self, mode: str) -> Dict[str, Any]:
        """Outcome buckets and the seat table for one mode."""
        spec = self.mode(mode)
        n = spec.num_players
        records = self.select(mode)
        reasons: Dict[str, int] = {}
        seat_wins = [0.0] * n
        plies = 0
        stuck = trunc = 0
        for record in records:
            reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
            for seat, credit in enumerate(seat_credit(record["values"])):
                seat_wins[seat] += credit
            plies += int(record["plies"])
            stuck += 1 if record["stuck"] else 0
            trunc += 1 if record["truncated"] else 0
        games = max(1, len(records))
        return {
            "mode": spec.to_dict(), "games": len(records), "reasons": reasons,
            "seat_win_share": [w / games for w in seat_wins],
            "mean_plies": plies / games,
            "stale_rate": stuck / games, "trunc_rate": trunc / games,
        }

    def to_dict(self, include_games: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "bots": list(self.bots),
            "modes": [m.to_dict() for m in self.modes],
            "config": dict(self.config),
            "tables": [t.to_dict() for t in self.tables],
            "num_games": len(self.games),
        }
        if include_games:
            out["games"] = self.games
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArenaResults":
        modes = [ModeSpec(key=m["key"], mode=m["mode"],
                          num_players=m["num_players"], layout=m.get("layout"))
                 for m in data["modes"]]
        return cls(bots=list(data["bots"]), modes=modes,
                   games=list(data.get("games", [])),
                   tables=[Table(mode=t["mode"], index=t["index"],
                                 kind=t["kind"], base=tuple(t["base"]),
                                 arrangements=tuple(tuple(a) for a in
                                                    t["arrangements"]),
                                 members=tuple(t["members"]))
                           for t in data.get("tables", [])],
                   config=dict(data.get("config", {})))


# ── Bradley–Terry ─────────────────────────────────────────────────────────

@dataclass
class Ratings:
    """An Elo-scale Bradley–Terry fit with bootstrap confidence intervals."""

    names: List[str]
    rating: Dict[str, float]
    lo: Dict[str, float] = field(default_factory=dict)
    hi: Dict[str, float] = field(default_factory=dict)
    games: Dict[str, int] = field(default_factory=dict)
    score: Dict[str, float] = field(default_factory=dict)
    strength: Dict[str, float] = field(default_factory=dict)
    anchor: str = ""
    anchor_rating: float = 0.0
    scale: float = ELO_SCALE
    iterations: int = 0
    converged: bool = True
    disconnected: List[List[str]] = field(default_factory=list)
    bootstrap: int = 0

    def sorted_names(self) -> List[str]:
        return sorted(self.names,
                      key=lambda n: (-self.rating.get(n, float("-inf")), n))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor": self.anchor, "anchor_rating": self.anchor_rating,
            "scale": self.scale, "iterations": self.iterations,
            "converged": self.converged, "bootstrap": self.bootstrap,
            "disconnected": self.disconnected,
            "ratings": {
                n: {"elo": self.rating.get(n), "lo": self.lo.get(n),
                    "hi": self.hi.get(n), "games": self.games.get(n, 0),
                    "score": self.score.get(n),
                    "strength": self.strength.get(n)}
                for n in self.sorted_names()},
        }


def _counts_to_matrices(counts: Mapping[Tuple[str, str], Sequence[float]],
                        names: Sequence[str], prior: float
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """``(wins, games)`` matrices with draws split and a virtual-draw prior.

    ``prior`` adds that many virtual *drawn* games to every pair that actually
    met.  Without it a bot that never lost has infinite strength (BayesElo
    solves this the same way); with the default 0.5 the shrinkage is a few Elo
    at 100 games and invisible at 1,000.
    """
    index = {name: i for i, name in enumerate(names)}
    k = len(names)
    wins = np.zeros((k, k), dtype=np.float64)
    games = np.zeros((k, k), dtype=np.float64)
    for (a, b), (wa, draws, wb) in counts.items():
        if a not in index or b not in index:
            continue
        i, j = index[a], index[b]
        total = float(wa) + float(draws) + float(wb)
        if total <= 0:
            continue
        wins[i, j] += float(wa) + 0.5 * float(draws)
        wins[j, i] += float(wb) + 0.5 * float(draws)
        games[i, j] += total
        games[j, i] += total
    if prior > 0:
        played = games > 0
        wins[played] += prior / 2.0
        games[played] += prior
    return wins, games


def _components(games: np.ndarray) -> List[List[int]]:
    """Connected components of the "has played" graph."""
    k = games.shape[0]
    seen = [False] * k
    out: List[List[int]] = []
    for start in range(k):
        if seen[start]:
            continue
        stack, comp = [start], []
        seen[start] = True
        while stack:
            node = stack.pop()
            comp.append(node)
            for other in np.flatnonzero(games[node] > 0):
                if not seen[other]:
                    seen[other] = True
                    stack.append(int(other))
        out.append(sorted(comp))
    return out


def _mm_fit(wins: np.ndarray, games: np.ndarray, max_iter: int,
            tol: float) -> Tuple[np.ndarray, int, bool]:
    """Zermelo / Hunter MM iteration for the Bradley–Terry likelihood.

    ``p_i <- W_i / sum_j n_ij / (p_i + p_j)``, renormalised to a geometric
    mean of one each round.  It is the standard minorise–maximise update: it
    never decreases the likelihood and converges from any positive start.
    """
    k = wins.shape[0]
    total_wins = wins.sum(axis=1)
    active = games.sum(axis=1) > 0
    p = np.ones(k, dtype=np.float64)
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        denom = (games / np.add.outer(p, p)).sum(axis=1)
        new = np.where(active & (denom > 0), total_wins / np.maximum(denom, 1e-300), p)
        new = np.maximum(new, 1e-300)
        if active.any():
            new[active] /= float(np.exp(np.mean(np.log(new[active]))))
        delta = float(np.max(np.abs(np.log(new[active]) - np.log(p[active]))
                             )) if active.any() else 0.0
        p = new
        if delta < tol:
            converged = True
            break
    p[~active] = float("nan")
    return p, iteration, converged


def _to_elo(p: np.ndarray, games: np.ndarray, names: Sequence[str],
            anchor: str, anchor_rating: float, scale: float
            ) -> Tuple[np.ndarray, List[List[str]]]:
    """Log-strengths to Elo, pinned on the anchor, per connected component."""
    with np.errstate(divide="ignore", invalid="ignore"):
        elo = scale * np.log(p)
    index = {name: i for i, name in enumerate(names)}
    disconnected: List[List[str]] = []
    for comp in _components(games):
        finite = [i for i in comp if np.isfinite(elo[i])]
        if not finite:
            continue
        if anchor in index and index[anchor] in comp:
            shift = anchor_rating - elo[index[anchor]]
        else:
            shift = anchor_rating - float(np.mean(elo[finite]))
            if len(names) > len(comp):
                disconnected.append([names[i] for i in comp])
        elo[finite] += shift
    return elo, disconnected


def _pairs_and_units(results: Any, mode: Optional[str]
                     ) -> Tuple[Dict[Tuple[str, str], List[float]],
                                Optional[List[Dict[Tuple[str, str],
                                                   List[float]]]],
                                List[str]]:
    """Normalise whatever the caller passed into ``(counts, units, names)``.

    Accepts an :class:`ArenaResults`, a list of game records, a
    ``{(a, b): (wins_a, draws, wins_b)}`` mapping or a nested
    ``{a: {b: (w, d, l)}}`` mapping.
    """
    if isinstance(results, ArenaResults):
        counts = results.pair_counts(mode)
        units = results.units(mode)
        names = list(results.bots)
        return counts, units, names
    if isinstance(results, (list, tuple)):
        wrapped = ArenaResults(bots=[], modes=[], games=list(results))
        names_set: List[str] = []
        for record in results:
            for name in record["seats"]:
                if name not in names_set:
                    names_set.append(name)
        wrapped.bots = names_set
        counts = wrapped.pair_counts(mode)
        return counts, wrapped.units(mode), names_set
    if isinstance(results, Mapping):
        counts = {}
        names_set = []
        items: List[Tuple[Tuple[str, str], Sequence[float]]] = []
        for key, value in results.items():
            if isinstance(value, Mapping) and not isinstance(key, tuple):
                for other, wdl in value.items():
                    items.append(((str(key), str(other)), wdl))
            else:
                a, b = key
                items.append(((str(a), str(b)), value))
        for (a, b), wdl in items:
            wdl = list(wdl) if not isinstance(wdl, Mapping) else [
                wdl.get("win", 0), wdl.get("draw", 0), wdl.get("loss", 0)]
            if len(wdl) == 2:                              # (wins_a, wins_b)
                wdl = [wdl[0], 0, wdl[1]]
            for name in (a, b):
                if name not in names_set:
                    names_set.append(name)
            key2 = (a, b) if a <= b else (b, a)
            slot = counts.setdefault(key2, [0.0, 0.0, 0.0])
            if key2 == (a, b):
                slot[0] += float(wdl[0]); slot[1] += float(wdl[1])
                slot[2] += float(wdl[2])
            else:
                slot[2] += float(wdl[0]); slot[1] += float(wdl[1])
                slot[0] += float(wdl[2])
        return counts, None, sorted(names_set)
    raise TypeError(f"cannot read results of type {type(results)!r}")


def fit_bradley_terry(results: Any, anchor: str = "random",
                      anchor_rating: float = 0.0, scale: float = ELO_SCALE,
                      prior: float = 0.5, bootstrap: int = 0,
                      alpha: float = 0.05, seed: int = 0,
                      mode: Optional[str] = None, max_iter: int = 10000,
                      tol: float = 1e-11) -> Ratings:
    """Joint Bradley–Terry / BayesElo fit over the whole result matrix.

    One fit, not a chain of pairwise deltas: every game constrains every
    rating through the likelihood, so an agent that never met ``random``
    directly is still placed on the same scale via the anchors it did meet.
    Draws count as half a win.  ``anchor`` is pinned at ``anchor_rating``
    (the model has exactly one degree of freedom, so pinning one player fixes
    the whole scale); ``scale = 400/ln 10`` makes the numbers ordinary Elo,
    i.e. a 400-point gap is a 10:1 expected score.

    ``bootstrap > 0`` resamples **seed groups** with replacement and refits,
    returning percentile confidence intervals — groups, because the games
    inside one are the same deal played from rotated seats and are therefore
    dependent.  With aggregate counts (no game records) it falls back to a
    multinomial resample of each pairing's win/draw/loss counts.
    """
    counts, units, names = _pairs_and_units(results, mode)
    if not names:
        raise ValueError("no players in the results")
    wins, games = _counts_to_matrices(counts, names, prior)
    p, iterations, converged = _mm_fit(wins, games, max_iter, tol)
    elo, disconnected = _to_elo(p, games, names, anchor, anchor_rating, scale)

    raw_games = _counts_to_matrices(counts, names, 0.0)[1]
    played = raw_games.sum(axis=1)
    raw_wins = _counts_to_matrices(counts, names, 0.0)[0].sum(axis=1)
    out = Ratings(
        names=list(names),
        rating={n: float(elo[i]) for i, n in enumerate(names)},
        games={n: int(played[i]) for i, n in enumerate(names)},
        score={n: (float(raw_wins[i] / played[i]) if played[i] else float("nan"))
               for i, n in enumerate(names)},
        strength={n: float(p[i]) for i, n in enumerate(names)},
        anchor=anchor, anchor_rating=float(anchor_rating), scale=float(scale),
        iterations=iterations, converged=converged, disconnected=disconnected,
        bootstrap=int(bootstrap))

    if bootstrap and bootstrap > 0:
        rng = np.random.default_rng(seed)
        samples = np.full((bootstrap, len(names)), np.nan)
        for b in range(int(bootstrap)):
            if units:
                pick = rng.integers(0, len(units), size=len(units))
                resampled: Dict[Tuple[str, str], List[float]] = {}
                for idx in pick:
                    for pair, wdl in units[int(idx)].items():
                        slot = resampled.setdefault(pair, [0.0, 0.0, 0.0])
                        slot[0] += wdl[0]; slot[1] += wdl[1]; slot[2] += wdl[2]
            else:
                resampled = {}
                for pair, wdl in counts.items():
                    total = int(round(sum(wdl)))
                    if total <= 0:
                        continue
                    probs = np.array(wdl, dtype=np.float64) / sum(wdl)
                    resampled[pair] = [float(x) for x in
                                       rng.multinomial(total, probs)]
            bw, bg = _counts_to_matrices(resampled, names, prior)
            bp, _it, _ok = _mm_fit(bw, bg, max_iter, tol)
            belo, _dc = _to_elo(bp, bg, names, anchor, anchor_rating, scale)
            samples[b] = belo
        lo = np.nanpercentile(samples, 100 * alpha / 2.0, axis=0)
        hi = np.nanpercentile(samples, 100 * (1 - alpha / 2.0), axis=0)
        out.lo = {n: float(lo[i]) for i, n in enumerate(names)}
        out.hi = {n: float(hi[i]) for i, n in enumerate(names)}
    return out


def bootstrap_ci(values_by_unit: Mapping[Any, Sequence[float]],
                 bootstrap: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    """Percentile CI for a mean, resampling whole seed groups."""
    units = [list(v) for v in values_by_unit.values() if v]
    if not units or bootstrap <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    sums = np.array([sum(u) for u in units], dtype=np.float64)
    sizes = np.array([len(u) for u in units], dtype=np.float64)
    draws = rng.integers(0, len(units), size=(int(bootstrap), len(units)))
    means = sums[draws].sum(axis=1) / np.maximum(sizes[draws].sum(axis=1), 1e-9)
    return (float(np.percentile(means, 100 * alpha / 2.0)),
            float(np.percentile(means, 100 * (1 - alpha / 2.0))))


# ── report ────────────────────────────────────────────────────────────────

def build_report(results: ArenaResults, anchor: str = "random",
                 anchor_rating: float = 0.0, bootstrap: int = 500,
                 seed: int = 0, per_mode: bool = True) -> Dict[str, Any]:
    """Fit the ratings and assemble the JSON payload the report renders."""
    if anchor not in results.bots:
        anchor = results.bots[0]
    overall = fit_bradley_terry(results, anchor=anchor,
                                anchor_rating=anchor_rating,
                                bootstrap=bootstrap, seed=seed)
    modes: Dict[str, Any] = {}
    for spec in results.modes:
        stats = results.bot_stats(spec.key)
        for name, bucket in stats.items():
            lo, hi = bootstrap_ci(bucket.pop("credits_by_unit"),
                                  bootstrap=bootstrap, seed=seed + 1)
            bucket["win_rate_lo"], bucket["win_rate_hi"] = lo, hi
        entry: Dict[str, Any] = {
            "spec": spec.to_dict(),
            "summary": results.mode_stats(spec.key),
            "bots": stats,
            "pairs": {f"{a}|{b}": wdl
                      for (a, b), wdl in results.pair_counts(spec.key).items()},
        }
        if per_mode:
            try:
                entry["ratings"] = fit_bradley_terry(
                    results, anchor=anchor, anchor_rating=anchor_rating,
                    bootstrap=bootstrap, seed=seed + 2,
                    mode=spec.key).to_dict()
            except Exception as exc:                       # pragma: no cover
                entry["ratings"] = {"error": str(exc)}
        modes[spec.key] = entry
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " UTC",
        "config": results.config,
        "bots": list(results.bots),
        "num_games": len(results.games),
        "ratings": overall.to_dict(),
        "modes": modes,
        "anchor": anchor,
    }


def _fmt(value: Optional[float], digits: int = 1, dash: str = "—") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return dash
    return f"{value:.{digits}f}"


def _pct(value: Optional[float], digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{100.0 * value:.{digits}f}%"


def render_markdown(report: Mapping[str, Any]) -> str:
    """The human-readable report (README §"Evaluation & export" reads it)."""
    cfg = report.get("config", {})
    lines: List[str] = []
    add = lines.append
    add("# Splendor arena report")
    add("")
    add(f"*Generated {report.get('generated', '?')} — "
        f"{report.get('num_games', 0)} games in "
        f"{_fmt(cfg.get('elapsed'), 1)} s.*")
    add("")
    add("| setting | value |")
    add("| --- | --- |")
    add("| bots | " + ", ".join(f"`{n}` = `{s}`"
                                for n, s in cfg.get("bots", {}).items()) + " |")
    add("| modes | " + ", ".join(m["key"] for m in cfg.get("modes", [])) + " |")
    add(f"| games per pairing (target) | {cfg.get('games_per_pairing', '?')} |")
    add(f"| games played | {report.get('num_games', 0)} |")
    add(f"| seed | {cfg.get('seed', '?')} |")
    add(f"| truncation cap | {cfg.get('max_plies', '?')} plies |")
    add(f"| workers | {cfg.get('workers', '?')} |")
    add("")

    ratings = report.get("ratings", {})
    add(f"## Elo — joint Bradley–Terry fit (anchor `{ratings.get('anchor')}` "
        f"= {_fmt(ratings.get('anchor_rating'), 0)})")
    add("")
    add("One fit over the whole win/loss matrix, all modes pooled; draws count "
        "as half a win. CIs are 95% percentile bootstrap over *seed groups* "
        f"({ratings.get('bootstrap', 0)} resamples).")
    add("")
    add("| bot | Elo | 95% CI | games | score |")
    add("| --- | ---: | :---: | ---: | ---: |")
    for name, row in ratings.get("ratings", {}).items():
        ci = (f"{_fmt(row.get('lo'), 0)} … {_fmt(row.get('hi'), 0)}"
              if row.get("lo") is not None else "—")
        add(f"| `{name}` | {_fmt(row.get('elo'), 0)} | {ci} | "
            f"{row.get('games', 0)} | {_pct(row.get('score'))} |")
    add("")
    if ratings.get("disconnected"):
        add(f"> **Warning:** disconnected comparison graph "
            f"{ratings['disconnected']} — those ratings are centred on their "
            f"own group and are not comparable with the anchor.")
        add("")
    if not ratings.get("converged", True):
        add("> **Warning:** the Bradley–Terry iteration hit its cap without "
            "converging.")
        add("")

    modes = report.get("modes", {})
    if len(modes) > 1:
        add("## Elo per mode")
        add("")
        keys = list(modes)
        add("| bot | " + " | ".join(f"`{k}`" for k in keys) + " |")
        add("| --- | " + " | ".join("---:" for _ in keys) + " |")
        for name in ratings.get("ratings", {}):
            cells = []
            for key in keys:
                row = modes[key].get("ratings", {}).get("ratings", {}).get(name)
                cells.append(_fmt(row.get("elo"), 0) if row else "—")
            add(f"| `{name}` | " + " | ".join(cells) + " |")
        add("")

    for key, entry in modes.items():
        spec = entry["spec"]
        summary = entry["summary"]
        add(f"## Mode `{key}` — {spec['label']}")
        add("")
        add(f"{summary['games']} games, mean "
            f"{_fmt(summary['mean_plies'])} plies; STALE (a seat had to "
            f"resign) {_pct(summary['stale_rate'])}, truncated "
            f"{_pct(summary['trunc_rate'])}; outcomes "
            + ", ".join(f"{k} {v}" for k, v in sorted(summary["reasons"].items()))
            + ".")
        add("")
        n = spec["num_players"]
        seat_cols = " | ".join(f"seat {i}" for i in range(n))
        add(f"| bot | games | win% | 95% CI | score% | mean plies | mean pts | "
            f"STALE% | trunc% | {seat_cols} |")
        add("| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | "
            + " | ".join("---:" for _ in range(n)) + " |")
        order = sorted(entry["bots"],
                       key=lambda b: -entry["bots"][b].get("win_rate", 0.0))
        for name in order:
            row = entry["bots"][name]
            if not row.get("games"):
                continue
            seat_cells = []
            for i in range(n):
                gsum = row["seat_games"][i] if i < len(row["seat_games"]) else 0
                wsum = row["seat_wins"][i] if i < len(row["seat_wins"]) else 0.0
                seat_cells.append(f"{_fmt(wsum, 1)}/{gsum}" if gsum else "—")
            ci = (f"{_pct(row.get('win_rate_lo'), 0)}…"
                  f"{_pct(row.get('win_rate_hi'), 0)}"
                  if row.get("win_rate_lo") == row.get("win_rate_lo") else "—")
            add(f"| `{name}` | {row['games']} | {_pct(row['win_rate'])} | {ci} "
                f"| {_pct(row['score_rate'])} | {_fmt(row['mean_plies'])} | "
                f"{_fmt(row['mean_points'])} | {_pct(row['stuck_rate'], 1)} | "
                f"{_pct(row['trunc_rate'], 1)} | " + " | ".join(seat_cells)
                + " |")
        add("")
        note = ("" if spec["mode"] == E.MODE_INDIVIDUAL else
                " — every seat of the winning side counts as a win here, so "
                "the shares sum to more than 100%")
        add("Seat win share (all bots pooled — the standing seat-asymmetry "
            f"detector{note}): "
            + ", ".join(f"seat {i} {_pct(v)}"
                        for i, v in enumerate(summary["seat_win_share"]))
            + ".")
        add("")
        if entry["spec"]["mode"] == E.MODE_ONE_V_TWO:
            add("Role split (1v2 is asymmetric — an agent can be exploitable "
                "as the solo seat and fine on average):")
            add("")
            add("| bot | solo games | solo win% | duo games | duo win% |")
            add("| --- | ---: | ---: | ---: | ---: |")
            for name in order:
                roles = entry["bots"][name].get("roles", {})
                solo = roles.get("solo", {"games": 0, "wins": 0.0})
                duo = roles.get("duo", {"games": 0, "wins": 0.0})
                add(f"| `{name}` | {solo['games']} | "
                    f"{_pct(solo['wins'] / solo['games']) if solo['games'] else '—'} "
                    f"| {duo['games']} | "
                    f"{_pct(duo['wins'] / duo['games']) if duo['games'] else '—'} |")
            add("")
        pairs = entry.get("pairs", {})
        if pairs:
            add("Head-to-head (row's score % against column, draws as half):")
            add("")
            names = [n for n in order if entry["bots"][n].get("games")]
            add("| | " + " | ".join(f"`{n}`" for n in names) + " |")
            add("| --- | " + " | ".join("---:" for _ in names) + " |")
            for a in names:
                cells = []
                for b in names:
                    if a == b:
                        cells.append("·")
                        continue
                    key_ab = f"{a}|{b}" if a <= b else f"{b}|{a}"
                    wdl = pairs.get(key_ab)
                    if not wdl:
                        cells.append("—")
                        continue
                    wins, draws, losses = wdl
                    total = wins + draws + losses
                    if a > b:
                        wins, losses = losses, wins
                    cells.append(
                        f"{_pct((wins + 0.5 * draws) / total, 0)} "
                        f"({int(wins)}-{int(draws)}-{int(losses)})"
                        if total else "—")
                add(f"| `{a}` | " + " | ".join(cells) + " |")
            add("")
    add("---")
    add("")
    add("How to read this: ratings are Elo on a pinned scale — `random` is 0 "
        "by construction, +400 means a 10:1 expected score against that "
        "opponent. Gates live in `docs/AI_DESIGN.md` §2; the anchors are "
        "frozen in `splendor_ai/anchors.py` and must never be retuned.")
    add("")
    return "\n".join(lines)


def write_reports(results: ArenaResults, out_md: str,
                  out_json: Optional[str] = None, anchor: str = "random",
                  bootstrap: int = 500, seed: int = 0,
                  include_games: bool = True) -> Tuple[str, str]:
    """Write the Markdown report and its JSON twin; return both paths."""
    report = build_report(results, anchor=anchor, bootstrap=bootstrap,
                          seed=seed)
    out_md = str(out_md)
    if out_json is None:
        stem = out_md[:-3] if out_md.endswith(".md") else out_md
        out_json = stem + ".json"
    for path in (out_md, out_json):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    payload = dict(report)
    payload["results"] = results.to_dict(include_games=include_games)
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=False, default=float)
    return out_md, out_json


# ── CLI ───────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m splendor_ai.arena",
        description="Paired-seed, seat-rotated arena with a Bradley-Terry Elo "
                    "fit against the pinned anchor ladder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python -m splendor_ai.arena --bots random greedy mcts40 --games 20 \\
      --modes ind2 --out reports/smoke.md
  python -m splendor_ai.arena --bots random greedy mcts40 mcts160 mcts640 \\
      search:runs/nscc/weights/latest.pt:400 \\
      --modes ind2 ind3 ind4 ovt team --games 400 --workers 32 \\
      --out reports/gen0040.md
bot specs: random | greedy | mcts<N> | net:<ckpt>[:c5] |
           search:<ckpt>:<sims>[:c5] | label=<spec>""")
    parser.add_argument("--bots", nargs="+", required=True,
                        help="bot specs (see below); each may be 'label=spec'")
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES),
                        help=f"mode keys {sorted(MODES)} or MODE:n[:layout]")
    parser.add_argument("--games", type=int, default=100,
                        help="target games per pairing per mode (rounded up "
                             "to whole seat rotations)")
    parser.add_argument("--seed", type=int, default=0, help="deal seed base")
    parser.add_argument("--workers", type=int, default=1,
                        help="worker processes (1 = in-process)")
    parser.add_argument("--device", default="cpu",
                        help="torch device for net/search bots")
    parser.add_argument("--max-plies", type=int, default=MAX_PLIES,
                        help="truncation cap per game")
    parser.add_argument("--anchor", default="random",
                        help="rating anchor pinned to --anchor-rating")
    parser.add_argument("--anchor-rating", type=float, default=0.0)
    parser.add_argument("--bootstrap", type=int, default=500,
                        help="bootstrap resamples for the CIs (0 = none)")
    parser.add_argument("--no-mixed", action="store_true",
                        help="skip Latin-square tables of n distinct bots")
    parser.add_argument("--max-mixed-tables", type=int, default=4)
    parser.add_argument("--out", default="reports/arena.md",
                        help="markdown report path (JSON goes beside it)")
    parser.add_argument("--json", default=None, help="explicit JSON path")
    parser.add_argument("--no-games", action="store_true",
                        help="omit per-game records from the JSON")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    factories: Dict[str, Any] = {}
    for spec in args.bots:
        factory = anchors_mod.make_factory(spec, device=args.device)
        if factory.label in factories:
            raise SystemExit(f"duplicate bot name {factory.label!r} — give one "
                             f"of them a label: 'name={spec}'")
        factories[factory.label] = factory
    modes = parse_modes(args.modes)

    last = [0.0]

    def progress(done: int, total: int, _record: Dict[str, Any]) -> None:
        now = time.time()
        if done == total or now - last[0] > 2.0:
            last[0] = now
            print(f"\r  {done}/{total} games", end="", file=sys.stderr,
                  flush=True)

    if not args.quiet:
        print(f"arena: {len(factories)} bots × {len(modes)} modes, "
              f"{args.games} games/pairing, workers={args.workers}",
              file=sys.stderr)
    results = run_matches(factories, modes, games_per_pairing=args.games,
                          seed=args.seed, workers=args.workers,
                          max_plies=args.max_plies, mixed=not args.no_mixed,
                          max_mixed_tables=args.max_mixed_tables,
                          progress=None if args.quiet else progress)
    if not args.quiet:
        print("", file=sys.stderr)
    md, js = write_reports(results, args.out, args.json, anchor=args.anchor,
                           bootstrap=args.bootstrap, seed=args.seed,
                           include_games=not args.no_games)
    ratings = fit_bradley_terry(results, anchor=args.anchor,
                                anchor_rating=args.anchor_rating)
    if not args.quiet:
        print(f"{len(results.games)} games in "
              f"{results.config['elapsed']:.1f} s → {md}, {js}",
              file=sys.stderr)
        for name in ratings.sorted_names():
            print(f"  {name:<28} {ratings.rating[name]:8.1f} Elo "
                  f"({ratings.games[name]} games)", file=sys.stderr)
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
