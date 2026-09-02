"""socket.io client and the per-move JSONL log (``docs/AI_BRIDGE.md`` §1).

The worker is a socket.io *client*: it dials out to the Render server, so a
home machine behind NAT needs no port forwarding, no tunnel and no inbound
firewall rule.  One worker is active per server; a fresh registration replaces
the previous one.

Threading model
---------------
``python-socketio`` delivers events on its own reader thread.  Anything slow
there would stall the connection (including the engine.io heartbeat), so
``ai_move_request`` is handed to a single-slot executor and answered from that
thread; the socket thread returns immediately.  One slot, not a pool: the
whole point is that one move at a time gets the GPU, and the server's 15 s
deadline leaves ample room for a second room to queue behind a 2.5 s move.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .agent import Decision, MoveAgent
from .config import WorkerConfig

__all__ = ["MoveLog", "WorkerClient", "Logger"]


class Logger:
    """Tiny levelled console logger (no logging config to fight with)."""

    _ORDER = {"debug": 10, "info": 20, "warn": 30, "error": 40}

    def __init__(self, level: str = "INFO", stream=None) -> None:
        self.threshold = self._ORDER.get(str(level).lower(), 20)
        self.stream = stream

    def __call__(self, level: str, message: str, **_: Any) -> None:
        if self._ORDER.get(level, 20) < self.threshold:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] [{level}] {message}"
        print(line, file=self.stream, flush=True)


class MoveLog:
    """Append-only JSONL move log (``LOG_DIR/moves.jsonl``).

    One line per answered request: enough to reconstruct latency percentiles,
    the fallback-level histogram and the exact action that was sent, without
    ever holding the file open across a move.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._count = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:                                    # pragma: no cover
            pass

    @property
    def count(self) -> int:
        return self._count

    def write(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"), default=str)
        with self._lock:
            self._count += 1
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as err:                         # pragma: no cover
                print(f"[warn] cannot write {self.path}: {err}", flush=True)

    @staticmethod
    def record(request: Mapping[str, Any], decision: Decision,
               queue_ms: float, extra: Optional[Mapping[str, Any]] = None
               ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "requestId": request.get("requestId"),
            "roomId": request.get("roomId"),
            "seat": decision.seat,
            "mode": decision.mode,
            "kind": decision.kind,
            "level": decision.level,
            "sims": int(decision.sims),
            "ms": round(decision.ms, 2),
            "queueMs": round(queue_ms, 2),
            "action": decision.action,
            "actionIndex": decision.action_index,
        }
        if decision.root_value is not None:
            out["rootValue"] = [round(float(v), 4) for v in decision.root_value]
            if 0 <= decision.seat < len(decision.root_value):
                out["value"] = round(float(decision.root_value[decision.seat]), 4)
        if decision.policy is not None:
            out["policy"] = round(float(decision.policy), 4)
        if decision.notes:
            out["notes"] = list(decision.notes)
        if extra:
            out.update(extra)
        return out


class WorkerClient:
    """Registers with the server and answers ``ai_move_request`` forever."""

    def __init__(self, cfg: WorkerConfig, agent: MoveAgent,
                 log: Optional[Callable[..., None]] = None,
                 move_log: Optional[MoveLog] = None) -> None:
        import socketio                                    # lazy: --once needs no socket
        self.cfg = cfg
        self.agent = agent
        self.log = log or Logger(cfg.log_level)
        self.moves = move_log if move_log is not None else MoveLog(cfg.moves_log)
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=0,                      # forever
            reconnection_delay=max(0.1, cfg.reconnect_delay_ms / 1000.0),
            reconnection_delay_max=max(1.0, cfg.reconnect_delay_max_ms / 1000.0),
            randomization_factor=0.5,
            logger=False,
            engineio_logger=False,
        )
        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="splendor-move")
        self._cancelled: set = set()
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self.registered = threading.Event()
        self.answered = 0
        self.rejected = 0
        self._install()

    # -- socket handlers --------------------------------------------------
    def _install(self) -> None:
        sio = self.sio

        @sio.event
        def connect() -> None:                             # noqa: D401
            self.log("info", f"connected to {self.cfg.server_url}")
            self._register()

        @sio.event
        def connect_error(data: Any) -> None:
            self.log("warn", f"connection failed: {data}")

        @sio.event
        def disconnect(*_: Any) -> None:
            self.registered.clear()
            if not self._stopping.is_set():
                self.log("warn", "disconnected — reconnecting")

        @sio.on("ai_move_request")
        def _on_request(data: Any = None) -> None:
            self._submit(data or {})

        @sio.on("ai_move_cancel")
        def _on_cancel(data: Any = None) -> None:
            request_id = (data or {}).get("requestId")
            if request_id is None:
                return
            with self._lock:
                self._cancelled.add(request_id)
            self.log("debug", f"cancelled {request_id}")

    def _register(self) -> None:
        payload = {
            "secret": self.cfg.secret,
            "name": self.cfg.worker_name,
            "version": self.cfg.version,
            "modes": list(self.cfg.modes),
        }

        def ack(response: Any = None) -> None:
            response = response or {}
            if isinstance(response, Mapping) and response.get("error"):
                self.log("error",
                         f"registration refused: {response['error']} — check "
                         f"AI_WORKER_SECRET on both sides")
                return
            self.registered.set()
            self.log("info", f"registered as {self.cfg.worker_name} "
                             f"v{self.cfg.version} "
                             f"[{', '.join(self.cfg.modes)}] — ack {response}")

        self.sio.emit("ai_worker_register", payload, callback=ack)

    # -- move handling ----------------------------------------------------
    def _submit(self, request: Mapping[str, Any]) -> None:
        received = time.monotonic()
        try:
            self._pool.submit(self._handle, request, received)
        except RuntimeError:                               # pragma: no cover
            self.log("warn", "executor is shutting down; request dropped")

    def _handle(self, request: Mapping[str, Any], received: float) -> None:
        request_id = request.get("requestId")
        queue_ms = (time.monotonic() - received) * 1000.0
        with self._lock:
            if request_id in self._cancelled:
                self._cancelled.discard(request_id)
                self.log("debug", f"{request_id} was cancelled before it ran")
                return
        try:
            decision = self.agent.decide(request)
        except Exception as err:
            # Deliberately silent on the wire: an unanswerable request is left
            # to the server's own fallback, which is strictly better than
            # sending {"type": "NONE"} and resigning a healthy seat.
            self.log("error", f"{request_id}: no move produced ({err!r}) — "
                              f"leaving it to the server fallback")
            self.moves.write({
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "requestId": request_id, "roomId": request.get("roomId"),
                "seat": request.get("playerIndex"), "level": "error",
                "error": repr(err), "queueMs": round(queue_ms, 2),
            })
            return

        with self._lock:
            if request_id in self._cancelled:
                self._cancelled.discard(request_id)
                self.log("debug", f"{request_id} cancelled while thinking")
                return

        def ack(response: Any = None) -> None:
            if isinstance(response, Mapping) and response.get("error"):
                self.rejected += 1
                self.log("warn", f"{request_id} refused by the server: "
                                 f"{response['error']}")

        try:
            self.sio.emit("ai_move_response", {
                "requestId": request_id,
                "action": decision.action,
                "info": decision.info,
            }, callback=ack)
        except Exception as err:                           # pragma: no cover
            self.log("error", f"{request_id}: emit failed ({err!r})")
            return

        self.answered += 1
        self.moves.write(MoveLog.record(request, decision, queue_ms))
        self.log("info" if decision.level != "search" else "debug",
                 f"{request_id} seat {decision.seat} {decision.kind} "
                 f"[{decision.mode}] -> {decision.action['type']} "
                 f"({decision.level}, {decision.sims} sims, "
                 f"{decision.ms:.0f} ms)")

    # -- lifecycle --------------------------------------------------------
    def run_forever(self) -> None:
        """Connect, then block until :meth:`stop` — reconnecting for ever."""
        delay = max(0.5, self.cfg.reconnect_delay_ms / 1000.0)
        max_delay = max(1.0, self.cfg.reconnect_delay_max_ms / 1000.0)
        while not self._stopping.is_set():
            try:
                self.sio.connect(self.cfg.server_url,
                                 transports=["websocket"],
                                 wait_timeout=15)
                delay = max(0.5, self.cfg.reconnect_delay_ms / 1000.0)
                self.sio.wait()
            except KeyboardInterrupt:                      # pragma: no cover
                raise
            except Exception as err:
                if self._stopping.is_set():
                    break
                self.log("warn", f"cannot reach {self.cfg.server_url} ({err}) "
                                 f"— retrying in {delay:.0f} s")
                if self._stopping.wait(delay):
                    break
                delay = min(max_delay, delay * 2)

    def stop(self) -> None:
        self._stopping.set()
        try:
            self.sio.disconnect()
        except Exception:                                  # pragma: no cover
            pass
        self._pool.shutdown(wait=False, cancel_futures=True)
