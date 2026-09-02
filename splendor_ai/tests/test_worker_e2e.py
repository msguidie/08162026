"""End-to-end: real Node server + real Python worker + scripted humans.

This is the deployment gate of ``docs/AI_DESIGN.md`` §1.9 / G6 in miniature.
Nothing is mocked: ``server/index.js`` runs as a child process with
``AI_WORKER_SECRET`` set, ``splendor_ai.worker.worker`` connects to it over a
real socket.io websocket, and ``splendor_ai/worker/dev/humanDriver.mjs`` plays
the human seats with random legal moves while the bot seats are driven by the
server through the AI bridge.

Both worker configurations are exercised:

* **greedy ladder** — no checkpoint anywhere, which is how the user first
  tests the wiring;
* **search ladder** — a freshly initialised smoke-config network saved as
  ``shared.pt``, so the net + MCTS path really runs (with a random net at
  small sims, which says nothing about strength and everything about wiring).

Assertions per configuration: every bot move was answered by *the worker*
(``moves.jsonl`` MOVE lines per room == the driver's bot-turn count), the
server never fell back for a worker fault, no action was rejected, every game
reached ``GAME_OVER`` and the p99 move latency stayed inside the budget.

Runtime is a couple of minutes.  ``SPLENDOR_E2E_GAMES`` (default 3) sets the
games per scenario; the full 500-game G6 gate is a README recipe for the
user's own machine, not something to run in CI.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

#: Marked ``slow`` (a few minutes) but NOT skipped: this is the deployment
#: gate.  The project has no pytest.ini, so the marker is unregistered and
#: pytest says so once at collection time; ``-m "not slow"`` still works.
pytestmark = pytest.mark.slow

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent                      # splendor_ai/
REPO = PROJECT.parent                      # repository root
SERVER_ENTRY = REPO / "server" / "index.js"
DRIVER = PROJECT / "splendor_ai" / "worker" / "dev" / "humanDriver.mjs"

SECRET = "test"
GAMES = int(os.environ.get("SPLENDOR_E2E_GAMES", "3"))
SCENARIOS = ("ind2", "ovt-solo", "ovt-duo", "team-2v2", "ind3")

# Small budgets: the point is the wiring, not the strength.  The latency
# assertion is made against these numbers, so keep them in sync.
SEARCH_SIMS = 64
TIME_BUDGET_MS = 150
HARD_BUDGET_MS = 400
#: Allowance over HARD_BUDGET_MS for hydration, the ladder and the log write.
LATENCY_SLACK_MS = 250

#: Fallback reasons that mean the WORKER misbehaved (docs/AI_BRIDGE.md §2).
WORKER_FAULT = re.compile(
    r"no worker connected|did not answer|unusable action|rejected by the rules"
    r" engine|worker disconnected")
#: The one benign fallback the current bridge produces — see the module note
#: in the report: a BUY that qualifies two nobles leaves the turn "unfinished"
#: by design and `runSequence` mistakes that for a stall.
BENIGN_STALL = re.compile(r"sequence left the turn unfinished")


def _require_node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    if not (REPO / "node_modules" / "socket.io-client").is_dir():
        pytest.skip("node_modules/socket.io-client is missing (npm install)")
    if not SERVER_ENTRY.is_file():
        pytest.skip(f"{SERVER_ENTRY} is missing")
    return node


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _get_json(url: str, timeout: float = 3.0) -> Optional[Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _wait(predicate, what: str, timeout: float = 40.0,
          process: Optional[subprocess.Popen] = None,
          log: Optional[Path] = None) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        if process is not None and process.poll() is not None:
            tail = log.read_text(errors="replace")[-3000:] if log else ""
            raise AssertionError(
                f"{what}: the process exited with {process.returncode}\n{tail}")
        time.sleep(0.1)
    tail = log.read_text(errors="replace")[-3000:] if log else ""
    raise AssertionError(f"timed out waiting for {what}\n{tail}")


class Stack:
    """The Node server under test."""

    def __init__(self, node: str, log_dir: Path) -> None:
        self.node = node
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = log_dir / "server.log"
        env = dict(os.environ)
        env.update(PORT=str(self.port), AI_WORKER_SECRET=SECRET,
                   AI_MOVE_DELAY_MS="10")
        for noisy in ("REPLAY_GITHUB_TOKEN", "REPLAY_GITHUB_REPO",
                      "RENDER_EXTERNAL_URL"):
            env.pop(noisy, None)
        self.handle = self.log.open("wb")
        self.process = subprocess.Popen(
            [node, str(SERVER_ENTRY)], env=env, cwd=str(REPO),
            stdout=self.handle, stderr=subprocess.STDOUT)
        _wait(lambda: _get_json(f"{self.url}/health") is not None,
              "the server to listen", process=self.process, log=self.log)

    def offset(self) -> int:
        return self.log.stat().st_size

    def log_since(self, offset: int) -> List[str]:
        with self.log.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            return [line.rstrip("\n") for line in handle if line.strip()]

    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:                  # pragma: no cover
            self.process.kill()
        self.handle.close()


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    node = _require_node()
    server = Stack(node, tmp_path_factory.mktemp("server"))
    try:
        yield server
    finally:
        server.stop()


@contextmanager
def worker(stack: Stack, log_dir: Path, model_dir: Optional[Path],
           name: str):
    """Run ``splendor_ai.worker.worker`` against ``stack`` for the block."""
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        SERVER_URL=stack.url,
        AI_WORKER_SECRET=SECRET,
        MODEL_DIR=str(model_dir) if model_dir else str(log_dir / "no-models"),
        LOG_DIR=str(log_dir),
        DEVICE="cpu",
        SEARCH_SIMS=str(SEARCH_SIMS),
        TIME_BUDGET_MS=str(TIME_BUDGET_MS),
        HARD_BUDGET_MS=str(HARD_BUDGET_MS),
        UNIVERSES="4",
        WORKER_NAME=name,
        LOG_LEVEL="INFO",
        PYTHONPATH=str(REPO),
        SPLENDOR_WORKER_ENV=str(log_dir / "absent.env"),   # ignore any real .env
    )
    stdout = (log_dir / "worker.log").open("wb")
    process = subprocess.Popen(
        [sys.executable, "-m", "splendor_ai.worker.worker"],
        env=env, cwd=str(REPO), stdout=stdout, stderr=subprocess.STDOUT)
    try:
        _wait(lambda: (_get_json(f"{stack.url}/api/ai/status") or {})
              .get("name") == name,
              f"worker {name} to register", process=process,
              log=log_dir / "worker.log")
        yield process
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:                  # pragma: no cover
            process.kill()
        stdout.close()
        _wait(lambda: not (_get_json(f"{stack.url}/api/ai/status") or {})
              .get("available"), "the server to notice the worker left",
              timeout=20)


def drive(stack: Stack, scenario: str, games: int, tag: str) -> List[Dict]:
    """Run the Node human driver and return its per-game summaries."""
    result = subprocess.run(
        [shutil.which("node"), str(DRIVER), "--url", stack.url,
         "--scenario", scenario, "--games", str(games), "--tag", tag,
         "--seed", str(abs(hash((scenario, tag))) % 100000 + 1)],
        cwd=str(REPO), capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise AssertionError(
            f"humanDriver {scenario} failed:\n{result.stdout[-4000:]}\n"
            f"{result.stderr[-2000:]}")
    return [json.loads(line[len("##GAME##"):])
            for line in result.stdout.splitlines()
            if line.startswith("##GAME##")]


def read_moves(log_dir: Path) -> List[Dict]:
    path = log_dir / "moves.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_run(stack: Stack, log_dir: Path, games: List[Dict],
              offset: int, expect_levels: set) -> Dict[str, Any]:
    """The shared assertion block for one worker configuration."""
    moves = read_moves(log_dir)
    by_room: Dict[str, List[Dict]] = {}
    for move in moves:
        by_room.setdefault(move.get("roomId"), []).append(move)

    total_bot_turns = 0
    for game in games:
        assert game["phase"] == "GAME_OVER", game
        assert game["botTurns"] > 0, game
        room_moves = by_room.get(game["roomId"], [])
        answered = [m for m in room_moves if m.get("kind") == "MOVE"]
        assert len(answered) == game["botTurns"], (
            f"{game['scenario']} {game['roomId']}: the worker answered "
            f"{len(answered)} moves but the bots took {game['botTurns']} turns")
        # A genuinely stuck seat (10 tokens, 3 reserved, nothing affordable — this
        # variant has no pass) is answered with NONE by design; a random smoke net
        # walks into that state occasionally, so accept it when the log says so.
        def _level_ok(m):
            if m["level"] in expect_levels:
                return True
            return m["level"] == "none" and "stuck" in str(m.get("notes", "")).lower()
        assert all(_level_ok(m) for m in room_moves), \
            sorted({m["level"] for m in room_moves})
        assert all((m.get("actionIndex") is not None and m.get("actionIndex", -1) >= 0)
                   or (m["level"] == "none" and _level_ok(m)) for m in room_moves)
        total_bot_turns += game["botTurns"]

    # The server must never have needed its own policy for a worker fault.
    lines = stack.log_since(offset)
    faults = [line for line in lines if WORKER_FAULT.search(line)]
    assert not faults, "the server fell back for a worker fault:\n" + \
        "\n".join(faults)
    stalls = [line for line in lines if "fallback for" in line]
    assert all(BENIGN_STALL.search(line) for line in stalls), \
        "unexpected fallback:\n" + "\n".join(stalls)
    assert not [line for line in lines if "action failed for" in line], \
        "an action the worker sent was rejected by the rules engine"

    worker_log = (log_dir / "worker.log").read_text(errors="replace")
    assert "refused by the server" not in worker_log, worker_log[-2000:]

    latencies = sorted(m["ms"] for m in moves)
    percentile = latencies[min(len(latencies) - 1,
                               int(len(latencies) * 0.99))] if latencies else 0
    return {
        "games": len(games),
        "botTurns": total_bot_turns,
        "moves": len(moves),
        "p50": statistics.median(latencies) if latencies else 0,
        "p99": percentile,
        "max": latencies[-1] if latencies else 0,
        "sims": statistics.median([m.get("sims", 0) for m in moves]) if moves
        else 0,
        "benignStalls": len(stalls),
    }


def _run_all(stack: Stack, log_dir: Path, tag: str,
             expect_levels: set) -> Dict[str, Any]:
    offset = stack.offset()
    games: List[Dict] = []
    for scenario in SCENARIOS:
        played = drive(stack, scenario, GAMES, tag)
        assert len(played) == GAMES
        games.extend(played)
    stats = check_run(stack, log_dir, games, offset, expect_levels)
    per_scenario = {s: sum(1 for g in games if g["scenario"] == s)
                    for s in SCENARIOS}
    print(f"\n  [{tag}] {stats['games']} games {per_scenario}, "
          f"{stats['botTurns']} bot turns, {stats['moves']} worker moves, "
          f"latency p50 {stats['p50']:.1f} ms / p99 {stats['p99']:.1f} ms / "
          f"max {stats['max']:.1f} ms, median {stats['sims']:.0f} sims, "
          f"{stats['benignStalls']} bridge noble-stalls")
    return stats


def test_greedy_ladder_plays_every_mode(stack, tmp_path_factory):
    """No checkpoint anywhere: the worker still answers every bot turn."""
    log_dir = tmp_path_factory.mktemp("greedy")
    with worker(stack, log_dir, model_dir=None, name="pytest-greedy"):
        stats = _run_all(stack, log_dir, "greedy", {"greedy"})
    assert stats["moves"] >= len(SCENARIOS) * GAMES
    assert stats["p99"] < HARD_BUDGET_MS + LATENCY_SLACK_MS, stats


def test_search_ladder_plays_every_mode(stack, tmp_path_factory):
    """With a (random) smoke checkpoint the net + MCTS path is the one used."""
    torch = pytest.importorskip("torch")
    from splendor_ai.model import SMOKE_CONFIG, SplendorNet, save_checkpoint

    model_dir = tmp_path_factory.mktemp("models")
    torch.manual_seed(0)
    save_checkpoint(str(model_dir / "shared.pt"), SplendorNet(SMOKE_CONFIG),
                    {"generation": 0, "meta": {"smoke": True}})

    log_dir = tmp_path_factory.mktemp("search")
    with worker(stack, log_dir, model_dir=model_dir, name="pytest-search"):
        stats = _run_all(stack, log_dir, "search", {"search"})
    assert stats["sims"] > 0, "the search never ran a simulation"
    assert stats["p99"] < HARD_BUDGET_MS + LATENCY_SLACK_MS, stats


def test_reconnects_when_the_server_goes_away(stack, tmp_path_factory):
    """Render free tier sleeps: the worker must keep dialling, not exit."""
    node = _require_node()
    log_dir = tmp_path_factory.mktemp("reconnect")
    dead = Stack(node, tmp_path_factory.mktemp("dead-server"))
    url, port = dead.url, dead.port
    with worker(dead, log_dir, model_dir=None, name="pytest-reconnect") as proc:
        dead.stop()
        time.sleep(2.0)
        assert proc.poll() is None, "the worker exited when the server died"
        # Bring a server back on the same port and expect a re-registration.
        revived = Stack.__new__(Stack)
        revived.node = node
        revived.port = port
        revived.url = url
        revived.log = tmp_path_factory.mktemp("revived") / "server.log"
        env = dict(os.environ)
        env.update(PORT=str(port), AI_WORKER_SECRET=SECRET,
                   AI_MOVE_DELAY_MS="10")
        revived.handle = revived.log.open("wb")
        revived.process = subprocess.Popen(
            [node, str(SERVER_ENTRY)], env=env, cwd=str(REPO),
            stdout=revived.handle, stderr=subprocess.STDOUT)
        try:
            _wait(lambda: (_get_json(f"{url}/api/ai/status") or {})
                  .get("name") == "pytest-reconnect",
                  "the worker to re-register after the server came back",
                  timeout=90)
        finally:
            revived.stop()
