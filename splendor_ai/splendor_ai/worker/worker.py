"""Worker entry point — ``python -m splendor_ai.worker.worker``.

::

    # normal operation: dial the server in .env and answer moves for ever
    python -m splendor_ai.worker.worker

    # offline self-test: hydrate one payload, print the move, exit
    python -m splendor_ai.worker.worker --once
    python -m splendor_ai.worker.worker --once request.json

``--once`` is the "is my install sane?" button: no socket, no server, no
secret.  With a path it replays a saved ``ai_move_request`` payload (capture
one by copying a request out of the server log or by dumping it from the
browser dev tools); without one it deals a fresh position and builds the
payload the way ``aiBridge.buildObservation`` would.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..rules import engine as E
from ..rules.cards import (CARD_COST, CARD_POINTS, CARD_REWARD, CARD_TIER,
                           TILE_POINTS, TILE_REQ)
from .agent import MoveAgent
from .config import WorkerConfig, load_config
from .client import Logger, WorkerClient

__all__ = ["main", "build_parser", "synthetic_request", "run_once"]

_JS_MODE = {E.MODE_INDIVIDUAL: "INDIVIDUAL", E.MODE_TEAM: "TEAM",
            E.MODE_ONE_V_TWO: "ONE_V_TWO"}


# ── offline self-test ─────────────────────────────────────────────────────

def _card(cid: int) -> Dict[str, Any]:
    return {"id": cid, "tier": CARD_TIER[cid], "reward": CARD_REWARD[cid],
            "points": CARD_POINTS[cid], "cost": list(CARD_COST[cid])}


def _tile(tid: int) -> Dict[str, Any]:
    return {"id": tid, "points": TILE_POINTS[tid],
            "requirement": list(TILE_REQ[tid])}


def synthetic_request(state: E.GameState, seat: Optional[int] = None,
                      request_id: str = "self-test") -> Dict[str, Any]:
    """An ``ai_move_request`` for ``state``, shaped like the real bridge one.

    Mirrors ``aiBridge.buildObservation``: a ``clientViewForPlayer`` with the
    other seats' reserved cards collapsed to ``{id, tier, hidden, known}``.
    """
    seat = state.current_player if seat is None else seat
    known = {cid for p in state.players
             for cid, public in zip(p.reserved, p.reserved_public) if public}
    players: List[Dict[str, Any]] = []
    for i, p in enumerate(state.players):
        entry: Dict[str, Any] = {
            "username": p.username, "gems": list(p.gems),
            "cards": [_card(c) for c in p.cards],
            "bonusTiles": [_tile(t) for t in p.tiles],
            "score": p.score, "avatarSeed": p.avatar_seed,
        }
        if p.team_id is not None:
            entry["teamId"] = p.team_id
        if i == seat:
            entry["reserved"] = [_card(c) for c in p.reserved]
        else:
            entry["reserved"] = [
                {"id": c if c in known else -1, "tier": CARD_TIER[c],
                 "hidden": True, "known": c in known} for c in p.reserved]
        players.append(entry)
    view = {
        "phase": state.phase,
        "board": [[_card(c) for c in row] for row in state.board],
        "deckCounts": list(state.deck_counts),
        "gems": list(state.gems),
        "bonusTiles": [_tile(t) for t in state.tiles],
        "players": players,
        "currentPlayerIndex": state.current_player,
        "roundStartPlayer": state.round_start_player,
        "turnAction": ({"type": state.turn_action} if state.turn_action
                       else None),
        "finalRoundTriggeredBy": state.final_round_triggered_by,
        "turnNumber": state.turn_number,
        "numPlayers": state.num_players,
        "config": dict(state.config),
        "resignedPlayers": list(state.resigned),
        "gameMode": _JS_MODE[state.mode],
        "teamLayout": state.team_layout,
        "teams": [dict(t) for t in state.teams],
        "gameResult": (dict(state.game_result) if state.game_result else None),
        "timeControl": None,
    }
    pending = list(state.pending_tile_choice or []) or None
    if pending:
        view["_pendingTileChoice"] = pending
    return {
        "requestId": request_id, "roomId": "self-test", "playerIndex": seat,
        "kind": "TILE" if pending else "MOVE",
        "deadlineMs": 0, "state": view,
        "knownReserved": sorted(known), "pendingTileChoice": pending,
    }


def _demo_state(seed: int = 0, plies: int = 20,
                num_players: int = 2) -> E.GameState:
    """A position a few random plies into a fresh game."""
    rng = random.Random(seed)
    state = E.new_game(num_players, rng=rng)
    for _ in range(plies):
        if state.phase != E.PHASE_PLAYING:
            break
        legal = E.legal_actions(state)
        if not legal:
            E.resign(state, state.current_player)
            continue
        E.apply(state, legal[rng.randrange(len(legal))])
    return state


def run_once(cfg: WorkerConfig, path: Optional[str], log: Logger) -> int:
    """Hydrate one payload, print the chosen action, return an exit code."""
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and "request" in payload:
            payload = payload["request"]
        log("info", f"loaded fixture {path}")
    else:
        payload = synthetic_request(_demo_state())
        log("info", "no fixture given — using a freshly dealt 2p position")

    agent = MoveAgent(cfg, log=log)
    decision = agent.decide(payload)
    print(json.dumps({
        "requestId": payload.get("requestId"),
        "seat": decision.seat,
        "kind": decision.kind,
        "mode": decision.mode,
        "level": decision.level,
        "actionIndex": decision.action_index,
        "action": decision.action,
        "sims": decision.sims,
        "ms": round(decision.ms, 1),
        "rootValue": (None if decision.root_value is None
                      else [round(float(v), 4) for v in decision.root_value]),
        "notes": decision.notes,
    }, indent=2))
    return 0 if decision.action.get("type") != "NONE" else 1


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="splendor-worker",
        description="Splendor AI deployment worker (docs/AI_BRIDGE.md §1)")
    parser.add_argument("--env", metavar="PATH",
                        help="explicit .env file (default: ./.env, then "
                             "splendor_ai/.env)")
    parser.add_argument("--once", nargs="?", const="", metavar="FIXTURE",
                        help="offline self-test: hydrate one saved payload "
                             "(or a freshly dealt position), print the move")
    parser.add_argument("--print-config", action="store_true",
                        help="show the resolved configuration and exit")
    parser.add_argument("--server", metavar="URL", help="override SERVER_URL")
    parser.add_argument("--name", metavar="NAME", help="override WORKER_NAME")
    parser.add_argument("--model-dir", metavar="DIR", help="override MODEL_DIR")
    parser.add_argument("--device", metavar="DEV", help="override DEVICE")
    parser.add_argument("--log-dir", metavar="DIR", help="override LOG_DIR")
    parser.add_argument("--log-level", metavar="LEVEL",
                        help="DEBUG | INFO | WARN | ERROR")
    return parser


def _overrides(args: argparse.Namespace) -> Dict[str, str]:
    pairs = {
        "SERVER_URL": args.server, "WORKER_NAME": args.name,
        "MODEL_DIR": args.model_dir, "DEVICE": args.device,
        "LOG_DIR": args.log_dir, "LOG_LEVEL": args.log_level,
    }
    return {k: str(v) for k, v in pairs.items() if v}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    import os
    overrides = _overrides(args)
    if overrides:
        os.environ.update(overrides)
    cfg = load_config(env_file=args.env)
    log = Logger(cfg.log_level)

    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2, default=str))
        return 0

    if args.once is not None:
        return run_once(cfg, args.once or None, log)

    if not cfg.secret:
        log("error", "AI_WORKER_SECRET is empty — the server will refuse the "
                     "registration.  Copy .env.example to .env and fill it in.")
        return 2

    log("info", f"model dir {cfg.model_path_dir}  device {cfg.device}  "
                f"budget {cfg.time_budget_ms}/{cfg.hard_budget_ms} ms  "
                f"sims<={cfg.search_sims}  K={cfg.universes}")
    if cfg.env_file:
        log("info", f"configuration from {cfg.env_file}")

    agent = MoveAgent(cfg, log=log)
    agent.warmup()
    client = WorkerClient(cfg, agent, log=log)
    log("info", f"move log: {cfg.moves_log}")

    def shutdown(signum: int, _frame: Any) -> None:
        log("info", f"signal {signum} — disconnecting "
                    f"({client.answered} moves answered)")
        client.stop()

    for name in ("SIGINT", "SIGTERM"):
        handler = getattr(signal, name, None)
        if handler is not None:
            try:
                signal.signal(handler, shutdown)
            except ValueError:                             # pragma: no cover
                pass                                       # not the main thread

    try:
        client.run_forever()
    except KeyboardInterrupt:
        log("info", "interrupted")
        client.stop()
    log("info", f"stopped after {client.answered} moves "
                f"({client.rejected} refused by the server)")
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    sys.exit(main())
