#!/usr/bin/env python3
"""Cross-language checker: Python engine vs. the Node oracle.

Consumes the JSONL produced by ``gen_trajectories.js`` (or a directory of
them) and, for every step of every game:

  (a) asserts the Python legal action set equals the set the Node server
      actually accepted (both directions, after mapping compact codes to
      action indices — gem takes compare as colour multisets because the
      index *is* the canonical multiset),
  (b) applies the action through :func:`splendor_ai.rules.engine.apply`,
  (c) asserts FULL post-state equality with the Node snapshot: board ids per
      tier in order, deck counts and deck tops, gem supply, revealed tiles,
      per-player gems / cards / reserved / tiles / score, currentPlayerIndex,
      roundStartPlayer, turnNumber, phase, finalRoundTriggeredBy,
      resignedPlayers, gameResult, pendingTileChoice and turnAction, plus the
      event payload fields (selected / gemsReturned / goldTaken / tier /
      cardId / tileClaimed).

At the end of each game it additionally replays the stored replay-format JSON
(``docs/REPLAY_FORMAT.md`` §1) from scratch and requires the same final state,
so genuine replay files are proven consumable by the same code path.

Usage:
    python -m splendor_ai.validation.replay_check FILE [FILE...]
    python replay_check.py data/*.jsonl.gz [--max-games N] [--bench 200000]
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from splendor_ai.rules import engine as E                       # noqa: E402
from splendor_ai.rules.actions import (                          # noqa: E402
    NUM_ACTIONS, action_name,
)


# ── snapshots ─────────────────────────────────────────────────────────────

def py_snap(s: E.GameState) -> Dict[str, Any]:
    """Exactly the shape ``snap()`` produces in gen_trajectories.js."""
    return {
        "b": [list(s.board[0]), list(s.board[1]), list(s.board[2])],
        "dc": list(s.deck_counts),
        "dt": [(s.decks[t][-1] if s.decks[t] else -1) for t in range(3)],
        "g": list(s.gems),
        "tl": list(s.tiles),
        "p": [[list(p.gems), list(p.cards), list(p.reserved), list(p.tiles),
               p.score] for p in s.players],
        "cp": s.current_player,
        "rs": s.round_start_player,
        "tn": s.turn_number,
        "ph": s.phase,
        "fr": s.final_round_triggered_by,
        "rg": list(s.resigned),
        "gr": s.game_result,
        "pt": s.pending_tile_choice,
        "ta": s.turn_action,
    }


_SNAP_LABELS = {
    "b": "board card ids per tier", "dc": "deck counts", "dt": "deck tops",
    "g": "gem supply", "tl": "revealed tiles",
    "p": "players [gems, cards, reserved, tiles, score]",
    "cp": "currentPlayerIndex", "rs": "roundStartPlayer", "tn": "turnNumber",
    "ph": "phase", "fr": "finalRoundTriggeredBy", "rg": "resignedPlayers",
    "gr": "gameResult", "pt": "pendingTileChoice", "ta": "turnAction",
}


def snap_diff(py: Dict[str, Any], node: Dict[str, Any]) -> List[str]:
    out = []
    for key in _SNAP_LABELS:
        a, b = py.get(key), node.get(key)
        if a != b:
            out.append(f"{key} ({_SNAP_LABELS[key]}): python={a!r} node={b!r}")
    return out


# ── event normalisation ───────────────────────────────────────────────────

_TAKE_TYPES = {"SELECT_GEM", "TAKE_GEMS_CONFIRMED"}


def py_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(ev.get("payload") or {})
    t = ev["type"]
    out["type"] = "TAKE" if t in _TAKE_TYPES else t
    out["tileClaimed"] = ev.get("tileClaimed")
    # Gem takes compare as colour MULTISETS: the incremental SELECT_GEM path
    # records the order the human clicked, which the server does not constrain.
    if "selected" in out:
        out["selected"] = sorted(out["selected"])
    return out


def node_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(ev)
    t = out["type"]
    out["type"] = "TAKE" if t in _TAKE_TYPES else t
    out.setdefault("tileClaimed", None)
    if "selected" in out:
        out["selected"] = sorted(out["selected"])
    return out


def event_diff(py: Dict[str, Any], node: Dict[str, Any]) -> List[str]:
    """Node is the oracle: every field it reports must match.  Python may
    carry extra fields (e.g. ``cardId`` on a deck reserve, which the server
    does not report but the post-state verifies anyway)."""
    out = []
    for key, want in node.items():
        got = py.get(key, "<missing>")
        if got != want:
            out.append(f"event.{key}: python={got!r} node={want!r}")
    return out


# ── IO ────────────────────────────────────────────────────────────────────

def open_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if path.endswith(".gz"):
        fh: Any = io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    else:
        fh = open(path, "r", encoding="utf-8")
    with fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def expand_inputs(paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
                    out.append(os.path.join(p, name))
        else:
            out.append(p)
    return out


# ── failure reporting ─────────────────────────────────────────────────────

class Mismatch(Exception):
    def __init__(self, path: str, game_id: str, step: int, kind: str,
                 details: List[str], context: str = ""):
        self.path, self.game_id, self.step = path, game_id, step
        self.kind, self.details, self.context = kind, details, context
        super().__init__(self.render())

    def render(self) -> str:
        lines = [
            "",
            "=" * 78,
            f"MISMATCH ({self.kind})",
            f"  file : {self.path}",
            f"  game : {self.game_id}",
            f"  step : {self.step}",
        ]
        if self.context:
            lines += ["  ---- context ----"]
            lines += ["  " + ln for ln in self.context.splitlines()]
        lines += ["  ---- differences ----"]
        lines += ["  * " + d for d in self.details]
        lines += ["=" * 78]
        return "\n".join(lines)


def describe(state: E.GameState) -> str:
    p = state.players[state.current_player]
    return "\n".join([
        f"mode={state.mode} layout={state.team_layout} n={state.num_players}",
        f"turn={state.turn_number} current={state.current_player} "
        f"roundStart={state.round_start_player} phase={state.phase}",
        f"final_round_triggered_by={state.final_round_triggered_by} "
        f"resigned={state.resigned} turn_action={state.turn_action} "
        f"pending={state.pending_tile_choice}",
        f"supply={state.gems} tiles={state.tiles} board={state.board}",
        f"deck_counts={state.deck_counts}",
        f"actor gems={p.gems} (total {p.total_gems()}) discount={p.discount} "
        f"cards={len(p.cards)} reserved={p.reserved} score={p.score}",
    ])


# ── the check ─────────────────────────────────────────────────────────────

class Stats:
    def __init__(self) -> None:
        self.games = 0
        self.steps = 0
        self.legal_checks = 0
        self.legal_entries = 0
        self.resigns = 0
        self.timeouts = 0
        self.stuck = 0
        self.tile_choices = 0
        self.auto_tiles = 0
        self.orphan_pending = 0
        self.endings: Dict[str, int] = {}
        self.replay_checks = 0
        self.engine_seconds = 0.0
        self.engine_calls = 0

    def merge(self, other: "Stats") -> None:
        for k, v in vars(other).items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    getattr(self, k)[kk] = getattr(self, k).get(kk, 0) + vv
            else:
                setattr(self, k, getattr(self, k) + v)


def check_file(path: str, stats: Stats, max_games: Optional[int] = None,
               progress: bool = True) -> None:
    state: Optional[E.GameState] = None
    game: Optional[Dict[str, Any]] = None
    replay_actions: List[List[Any]] = []
    t_last = time.time()

    for rec in open_jsonl(path):
        kind = rec.get("k")

        if kind == "game":
            if max_games is not None and stats.games >= max_games:
                return
            game = rec
            replay_actions = []
            state = E.new_game(
                rec["n"], rec["mode"], rec.get("layout"),
                team_ids=rec.get("teams"), setup={**rec["setup"], "first": rec["first"]},
            )
            continue

        if kind == "summary":
            continue

        if kind == "end":
            if state is None or game is None:
                continue
            gid = rec["id"]
            diffs = snap_diff(py_snap(state), rec["s"])
            if diffs:
                raise Mismatch(path, gid, rec["steps"], "final state", diffs,
                               describe(state))
            fd = [list(state.decks[t]) for t in range(3)]
            if fd != rec["finalDecks"]:
                raise Mismatch(path, gid, rec["steps"], "final decks",
                               [f"python={fd} node={rec['finalDecks']}"])
            rat = E.rating_changes(state)
            if rat != rec["rating"]:
                raise Mismatch(path, gid, rec["steps"], "ratingChanges",
                               [f"python={rat} node={rec['rating']}"])
            res = rec["result"]
            if res.get("winners") is not None:
                win = E.individual_winners(state)
                if win != res["winners"]:
                    raise Mismatch(path, gid, rec["steps"], "winners",
                                   [f"python={win} node={res['winners']}"])
            wt = (state.game_result or {}).get("winningTeamIds")
            if wt != res.get("winningTeamIds"):
                raise Mismatch(path, gid, rec["steps"], "winningTeamIds",
                               [f"python={wt} node={res.get('winningTeamIds')}"])

            # Replay-format round trip: rebuild the whole game from the stored
            # compact actions and require an identical final state.
            rp = rec["replay"]
            rstate = E.replay(rp)
            rdiffs = snap_diff(py_snap(rstate), rec["s"])
            if rdiffs:
                raise Mismatch(path, gid, rec["steps"],
                               "replay-format reconstruction", rdiffs)
            stats.replay_checks += 1

            reason = ((state.game_result or {}).get("reason")
                      if state.phase == E.PHASE_GAME_OVER else None)
            if state.phase != E.PHASE_GAME_OVER:
                key = "TRUNCATED"
            else:
                key = reason or "SCORE"
            stats.endings[key] = stats.endings.get(key, 0) + 1
            stats.games += 1
            state = None
            if progress and stats.games % 250 == 0:
                now = time.time()
                sys.stderr.write(
                    f"\r  {os.path.basename(path)}: {stats.games} games "
                    f"{stats.steps} steps  ({250 / max(now - t_last, 1e-9):.0f} games/s)   ")
                sys.stderr.flush()
                t_last = now
            continue

        if kind != "step":
            continue
        assert state is not None and game is not None
        gid = game["id"]
        i = rec["i"]
        via = rec["via"]
        code = rec["a"]
        actor = rec["actor"]

        if state.pending_tile_choice is not None and state.turn_action is None:
            stats.orphan_pending += 1

        # (a) legality
        if rec["legal"] is not None:
            if state.current_player != actor:
                raise Mismatch(path, gid, i, "acting seat",
                               [f"python current={state.current_player} node actor={actor}"],
                               describe(state))
            t0 = time.perf_counter()
            py_legal = set(E.legal_actions(state))
            stats.engine_seconds += time.perf_counter() - t0
            stats.engine_calls += 1
            node_legal = set()
            for c in rec["legal"]:
                try:
                    node_legal.add(E.from_replay_code(state, c))
                except E.IllegalAction as exc:
                    raise Mismatch(path, gid, i, "undecodable accepted action",
                                   [f"{c!r}: {exc}"], describe(state)) from None
            if py_legal != node_legal:
                only_py = sorted(py_legal - node_legal)
                only_node = sorted(node_legal - py_legal)
                details = []
                if only_py:
                    details.append("legal in python, REJECTED by node: "
                                   + ", ".join(f"{a}={action_name(a)}" for a in only_py))
                if only_node:
                    details.append("accepted by node, illegal in python: "
                                   + ", ".join(f"{a}={action_name(a)}" for a in only_node))
                raise Mismatch(path, gid, i, "legal action set", details,
                               describe(state))
            stats.legal_checks += 1
            stats.legal_entries += len(py_legal)
            if via == "stuck-resign":
                if py_legal:
                    raise Mismatch(path, gid, i, "stuck detection",
                                   [f"node found no action but python has "
                                    f"{sorted(py_legal)}"], describe(state))
                if not E.is_stuck(state):
                    raise Mismatch(path, gid, i, "is_stuck",
                                   ["is_stuck(state) is False on a stuck state"],
                                   describe(state))
                stats.stuck += 1

        # (b) apply
        before = describe(state)
        letter = code[1]
        if letter == "X":
            ev = E.resign(state, actor)
            stats.resigns += 1
        elif letter == "T":
            ev = E.timeout(state, actor)
            stats.timeouts += 1
        else:
            t0 = time.perf_counter()
            idx = E.from_replay_code(state, code)
            try:
                ev = E.apply(state, idx)
            except E.IllegalAction as exc:
                raise Mismatch(path, gid, i, "apply rejected a Node action",
                               [f"{code!r} -> {action_name(idx)}: {exc}"],
                               before) from None
            stats.engine_seconds += time.perf_counter() - t0
            stats.engine_calls += 1
            if letter == "N":
                stats.tile_choices += 1
        replay_actions.append(code)
        stats.steps += 1
        if ev.get("tileClaimed"):
            stats.auto_tiles += 1

        # (c) full state equality
        diffs = snap_diff(py_snap(state), rec["s"])
        if diffs:
            raise Mismatch(path, gid, i, "post-action state", diffs,
                           before + f"\naction: {code!r} (via {via})")

        # For the confirmed (non-incremental) path the order must match too:
        # the generator sends the canonical sorted order, which is exactly what
        # the Python action index encodes.
        if via == "confirm" and "selected" in rec["ev"]:
            got = (ev.get("payload") or {}).get("selected")
            if got != rec["ev"]["selected"]:
                raise Mismatch(path, gid, i, "selected order",
                               [f"python={got!r} node={rec['ev']['selected']!r}"],
                               before)

        ediffs = event_diff(py_event(ev), node_event(rec["ev"]))
        if ediffs:
            raise Mismatch(path, gid, i, "action result payload", ediffs,
                           before + f"\naction: {code!r} (via {via})")

    if progress:
        sys.stderr.write("\r" + " " * 78 + "\r")


# ── benchmark ─────────────────────────────────────────────────────────────

def bench(n_calls: int = 200000, seed: int = 12345) -> Dict[str, float]:
    """Measure ``legal_mask`` + ``apply`` throughput on one core."""
    import random
    rng = random.Random(seed)
    configs = [(2, "INDIVIDUAL", None), (3, "INDIVIDUAL", None),
               (4, "INDIVIDUAL", None), (3, "ONE_V_TWO", None),
               (4, "TEAM", "ADJACENT"), (4, "TEAM", "OPPOSITE")]
    ci = 0
    n, mode, layout = configs[0]
    state = E.new_game(n, mode, layout, rng=rng)
    done = 0
    t0 = time.perf_counter()
    while done < n_calls:
        mask = E.legal_mask(state)
        acts = [k for k in range(NUM_ACTIONS) if mask[k]]
        if not acts or state.phase != E.PHASE_PLAYING:
            ci += 1
            n, mode, layout = configs[ci % len(configs)]
            state = E.new_game(n, mode, layout, rng=rng)
            continue
        E.apply(state, acts[int(rng.random() * len(acts))])
        done += 1
    dt = time.perf_counter() - t0
    return {"calls": done, "seconds": dt, "per_second": done / dt}


# ── main ──────────────────────────────────────────────────────────────────

def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="JSONL(.gz) files or directories")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--bench", type=int, default=0,
                    metavar="N", help="also benchmark N apply+legal_mask calls")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    stats = Stats()
    files = expand_inputs(args.inputs)
    t0 = time.time()
    for path in files:
        if not args.quiet:
            print(f"checking {path} ...", flush=True)
        check_file(path, stats, args.max_games, progress=not args.quiet)
    wall = time.time() - t0

    print()
    print("=" * 62)
    print("CROSS-LANGUAGE VALIDATION RESULT")
    print("=" * 62)
    print(f"  files                 {len(files)}")
    print(f"  games                 {stats.games}")
    print(f"  steps                 {stats.steps}")
    print(f"  legal-set comparisons {stats.legal_checks} "
          f"({stats.legal_entries} action entries)")
    print(f"  replay-file rebuilds  {stats.replay_checks}")
    print(f"  resigns / timeouts    {stats.resigns} / {stats.timeouts}")
    print(f"  stuck positions       {stats.stuck}")
    print(f"  noble choices / auto  {stats.tile_choices} / {stats.auto_tiles}")
    print(f"  orphaned noble states {stats.orphan_pending}")
    print(f"  endings               {stats.endings}")
    print(f"  MISMATCHES            0")
    if stats.engine_calls:
        print(f"  engine time           {stats.engine_seconds:.2f}s over "
              f"{stats.engine_calls} calls "
              f"({stats.engine_calls / max(stats.engine_seconds, 1e-9):,.0f}/s)")
    print(f"  wall clock            {wall:.1f}s")

    if args.bench:
        b = bench(args.bench)
        print(f"  BENCH apply+legal_mask {b['per_second']:,.0f} steps/s "
              f"({b['calls']} calls in {b['seconds']:.2f}s, single core)")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Mismatch as exc:                      # pragma: no cover
        print(exc.render())
        raise SystemExit(1)
