"""Batched inference servers and their client evaluator (§1.8).

Layout on the production node: the learner owns ``cuda:0`` and one
:class:`InferenceServer` process owns each of ``cuda:1..3``.  Actors send leaf
batches to a server and block on their own response queue; the server coalesces
whatever has arrived within ``max_wait_ms`` (or ``max_batch`` rows) into one
forward pass.

Transport
---------
``multiprocessing.Queue`` carrying **raw buffers**, not numpy objects:
request ``(client_id, req_id, rows, obs_bytes, mask_bytes)`` with ``obs``
float32 and ``mask`` packed bits, response ``(req_id, priors_bytes,
values_bytes, weight_generation)``.  The generation on the response is what
lets a record written in ``inference.mode=server`` be stamped with the
generation of the weights that actually produced it: the actor holds no model
in that mode, so without it every record was stamped 0.

A queue costs a pickle of two ``bytes`` objects (a
memcpy) rather than of two ndarrays; shared-memory slots would save the second
copy, but a ring of shm slots has to be reclaimed correctly when an actor dies
mid-request, and a dead actor is a routine event here (the orchestrator
restarts it).  The copy is ~90 KB per request at the production batch size and
the GPUs are ~10-20% utilised, so robustness wins.  See the README for the
measured cost and the switch-over criterion.

Safety
------
* ``obs_version`` is checked twice: :func:`splendor_ai.model.load_checkpoint`
  refuses a checkpoint from a different observation layout, and every request
  carries its ``OBS_DIM`` so an actor built against another encoder is
  rejected loudly instead of being scored against garbage.
* the server never silently swallows an exception: it answers the pending
  request with an error marker, logs the traceback and re-raises.
"""

from __future__ import annotations

import os
import queue
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..encode import OBS_DIM, OBS_VERSION
from ..model import NetConfig, SplendorNet, load_checkpoint
from ..rules.actions import NUM_ACTIONS
from . import configure_process

__all__ = ["InferenceServer", "RemoteEvaluator", "LocalEvaluator",
           "server_main", "WeightWatcher", "is_version_gate",
           "install_stop_handlers"]

_ERROR = -1


def is_version_gate(exc: BaseException) -> bool:
    """Is ``exc`` the ``obs_version`` / ``action_version`` refusal?

    :func:`splendor_ai.model.load_checkpoint` raises ``RuntimeError`` for a
    checkpoint from another encoder -- and ``torch.load`` raises
    ``RuntimeError`` for a truncated file as well, so the type alone cannot
    tell "this build cannot read these weights, ever" from "read it again in
    50 ms".  The message can: the gate always names the version it rejected.
    """
    text = str(exc)
    return "obs_version" in text or "action_version" in text


class WeightWatcher:
    """Reloads ``weights/latest.pt`` into a live model when the file changes.

    Cheap enough to call in the actor's inner loop: it stats the file and only
    touches torch when ``(mtime, size)`` moved.  ``load_checkpoint`` enforces
    the ``obs_version`` / ``action_version`` gate, so a stale checkpoint from
    another encoder stops the process instead of poisoning the run.
    """

    #: Attempts and pause for a torn read of ``weights/latest.pt``.
    retries = 5
    retry_sleep_s = 0.05

    def __init__(self, path: str, model, device: str = "cpu",
                 min_interval_s: float = 1.0) -> None:
        self.path = path
        self.model = model
        self.device = device
        self.min_interval_s = float(min_interval_s)
        self.signature: Optional[Tuple[float, int]] = None
        self.version = -1
        self.step = 0
        self.generation = 0
        self.reloads = 0
        self._last_check = 0.0

    def _signature(self) -> Optional[Tuple[float, int]]:
        try:
            st = os.stat(self.path)
        except OSError:
            return None
        return (st.st_mtime, st.st_size)

    def poll(self, force: bool = False) -> bool:
        """Reload if the file changed.  Returns True when weights were swapped."""
        now = time.monotonic()
        if not force and now - self._last_check < self.min_interval_s:
            return False
        self._last_check = now
        sig = self._signature()
        if sig is None or sig == self.signature:
            return False
        import torch

        loaded = ckpt = None
        for attempt in range(self.retries):
            try:
                loaded, ckpt = load_checkpoint(self.path, map_location=self.device)
                break
            except Exception as exc:
                # The version gate is a REAL failure and must stop the process.
                # Everything else is a torn read: the writer renames atomically
                # but a shared filesystem can still hand out a truncated file,
                # and torch raises RuntimeError for that too -- which is why
                # this cannot key off the exception type (`except RuntimeError:
                # raise` used to make the whole retry loop dead code).
                if is_version_gate(exc) or attempt == self.retries - 1:
                    raise
                time.sleep(self.retry_sleep_s)
        with torch.inference_mode():
            self.model.load_state_dict(loaded.state_dict())
        self.model.eval()
        # Re-stat: if the file moved again while we were reading it, the
        # signature of what we actually loaded is the newer one.
        self.signature = self._signature() or sig
        meta = ckpt.get("meta") or {}
        self.version = int(meta.get("version", ckpt.get("step", 0)))
        self.step = int(ckpt.get("step", 0))
        self.generation = int(ckpt.get("generation", 0))
        self.reloads += 1
        return True


class LocalEvaluator:
    """``inproc`` evaluator: the net lives inside the actor on the CPU.

    Wraps :class:`splendor_ai.model.NetEvaluator` and adds the weight watcher,
    so an actor using it picks up new weights without any inter-process
    machinery.  This is the smoke-run path.
    """

    def __init__(self, model: SplendorNet, weights_path: str,
                 device: str = "cpu", refresh_s: float = 10.0) -> None:
        from ..model import NetEvaluator

        self.model = model
        self.evaluator = NetEvaluator(model, device)
        self.watcher = WeightWatcher(weights_path, model, device,
                                     min_interval_s=refresh_s)
        self.calls = 0
        self.rows = 0

    @property
    def generation(self) -> int:
        """Generation of the weights currently loaded (record stamping)."""
        return int(self.watcher.generation)

    def refresh(self, force: bool = False) -> bool:
        return self.watcher.poll(force=force)

    def evaluate(self, obs: np.ndarray, mask: np.ndarray):
        if obs.shape[0] == 0:                               # pragma: no cover
            raise ValueError("evaluate() called with an empty batch")
        if not mask.any(axis=1).all():
            raise AssertionError("evaluate(): a row of the mask has no legal action")
        self.calls += 1
        self.rows += obs.shape[0]
        return self.evaluator.evaluate(obs, mask)


class RemoteEvaluator:
    """Client side of an :class:`InferenceServer`; same ``evaluate`` contract."""

    #: Request ids are ``start_id + 1, +2, ...``.  ``start_id`` comes from the
    #: actor's per-launch nonce, spaced this far apart, so a RESTARTED actor
    #: never issues an id its dead predecessor already used: the replacement
    #: inherits the same response queue and would otherwise accept a stale
    #: reply (identical ids, both starting at 0) as the answer to its own
    #: request -- silently scoring one game's leaves with another's priors.
    ID_STRIDE = 1 << 24

    def __init__(self, client_id: int, request_q, response_q,
                 timeout_s: float = 120.0, start_id: int = 0) -> None:
        self.client_id = int(client_id)
        self.request_q = request_q
        self.response_q = response_q
        self.timeout_s = float(timeout_s)
        self._req_id = int(start_id)
        self.start_id = int(start_id)
        self.calls = 0
        self.rows = 0
        self.wait_s = 0.0
        #: replies for a request this evaluator never made (a dead actor's)
        self.stale = 0
        #: weight generation the server used for the last response (§1.8
        #: instrumentation: records must carry the generation that produced
        #: them, and in ``server`` mode the actor holds no model to ask)
        self.generation = 0

    @classmethod
    def start_id_for(cls, nonce: int) -> int:
        """First request id for an actor launched with ``nonce``."""
        return (int(nonce) & 0xFFFF) * cls.ID_STRIDE

    def evaluate(self, obs: np.ndarray, mask: np.ndarray):
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=bool)
        if obs.ndim != 2 or obs.shape[1] != OBS_DIM:
            raise ValueError(f"obs must be [B, {OBS_DIM}], got {obs.shape}")
        if not mask.any(axis=1).all():
            raise AssertionError("evaluate(): a row of the mask has no legal action")
        self._req_id += 1
        req = self._req_id
        rows = obs.shape[0]
        self.request_q.put((self.client_id, req, rows, obs.tobytes(),
                            np.packbits(mask, axis=1).tobytes()))
        t0 = time.perf_counter()
        while True:
            got = self.response_q.get(timeout=self.timeout_s)
            rid, priors_buf, values_buf = got[0], got[1], got[2]
            generation = int(got[3]) if len(got) > 3 else 0
            if rid == _ERROR:
                raise RuntimeError(f"inference server failed: {priors_buf}")
            if rid != req:
                # A reply to somebody else's request: this actor replaced one
                # that died with a request in flight.  Drop it and keep
                # waiting for our own id.
                self.stale += 1
                continue
            self.generation = generation
            break
        self.wait_s += time.perf_counter() - t0
        self.calls += 1
        self.rows += rows
        priors = np.frombuffer(priors_buf, dtype=np.float32).reshape(rows, NUM_ACTIONS)
        values = np.frombuffer(values_buf, dtype=np.float32).reshape(rows, 4)
        return priors, values

    def refresh(self, force: bool = False) -> bool:
        """No-op: the server reloads its own weights."""
        return False


class InferenceServer:
    """The server loop.  Run it with :func:`server_main` in its own process."""

    def __init__(self, device: str, net_cfg: NetConfig, weights_path: str,
                 max_batch: int = 1024, max_wait_ms: float = 1.0,
                 reload_every_s: float = 30.0, name: str = "infer",
                 stop_grace_s: float = 30.0) -> None:
        import torch

        self.device = device
        self.name = name
        self.max_batch = int(max_batch)
        self.max_wait_s = float(max_wait_ms) / 1000.0
        #: After a stop signal the server keeps serving until its request
        #: queue goes quiet, for at most this long.  It must outlive the
        #: actors' last wave: `timeout --signal=INT` signals the whole process
        #: group, so a server that exited immediately would strand every actor
        #: mid-search on `response_q.get`, and their finished games with them.
        self.stop_grace_s = float(stop_grace_s)
        #: set by the SIGINT/SIGTERM handler in :func:`server_main`
        self.signalled = False
        self.model = SplendorNet(net_cfg).to(device).eval()
        self.watcher = WeightWatcher(weights_path, self.model, device,
                                     min_interval_s=reload_every_s)
        self.watcher.poll(force=True)
        self.use_autocast = str(device).startswith("cuda")
        self.autocast_dtype = torch.bfloat16
        self.stats = {"batches": 0, "rows": 0, "requests": 0, "seconds": 0.0,
                      "reloads": 0}

    # -- one coalesced batch ---------------------------------------------
    def _gather(self, request_q, stop_event) -> List[tuple]:
        pending: List[tuple] = []
        rows = 0
        try:
            item = request_q.get(timeout=0.1)
        except queue.Empty:
            return pending
        if item is None:
            return [None]
        pending.append(item)
        rows += item[2]
        deadline = time.perf_counter() + self.max_wait_s
        while rows < self.max_batch:
            timeout = deadline - time.perf_counter()
            if timeout <= 0:
                break
            try:
                item = request_q.get(timeout=timeout)
            except queue.Empty:
                break
            if item is None:
                pending.append(None)
                break
            pending.append(item)
            rows += item[2]
        return pending

    def run(self, request_q, response_qs: Dict[int, Any], stop_event) -> None:
        # A ``None`` on the request queue retires THIS server only (the stop
        # event is shared by every process in the run and must not be set by a
        # single server draining its queue).
        retiring = False
        deadline: Optional[float] = None
        while True:
            stopping = self.signalled or stop_event.is_set()
            if stopping and deadline is None:
                deadline = time.perf_counter() + self.stop_grace_s
                print(f"[{self.name}] stopping: serving in-flight requests for "
                      f"up to {self.stop_grace_s:.0f}s", flush=True)
            if deadline is not None and time.perf_counter() >= deadline:
                break
            batch = self._gather(request_q, stop_event)
            if not batch:
                # `_gather` blocks 100 ms on an empty queue, so an empty batch
                # after a stop means every actor has stopped asking: the wave
                # in flight finished and it is safe to go.
                if retiring or deadline is not None:
                    break
                self.watcher.poll()
                continue
            if batch[-1] is None:
                batch = batch[:-1]
                retiring = True
                if not batch:
                    break
            t0 = time.perf_counter()
            try:
                self._serve(batch, response_qs)
            except Exception as exc:                        # no silent failures
                info = f"{self.name}: {exc}\n{traceback.format_exc()}"
                for client_id, _req, _rows, _o, _m in batch:
                    q = response_qs.get(client_id)
                    if q is not None:
                        q.put((_ERROR, info, None, 0))
                print(info, flush=True)
                raise
            self.stats["seconds"] += time.perf_counter() - t0
            if self.watcher.poll():
                self.stats["reloads"] = self.watcher.reloads

    def _serve(self, batch: List[tuple], response_qs: Dict[int, Any]) -> None:
        import torch

        rows = [item[2] for item in batch]
        total = int(sum(rows))
        obs = np.empty((total, OBS_DIM), dtype=np.float32)
        mask = np.empty((total, NUM_ACTIONS), dtype=bool)
        off = 0
        for (_cid, _req, n, obs_buf, mask_buf) in batch:
            flat = np.frombuffer(obs_buf, dtype=np.float32)
            if flat.size != n * OBS_DIM:
                raise ValueError(
                    f"{self.name}: request carries {flat.size} floats for {n} "
                    f"rows; this build encodes OBS_DIM={OBS_DIM} "
                    f"(obs_version {OBS_VERSION}) — actor/server mismatch")
            obs[off:off + n] = flat.reshape(n, OBS_DIM)
            bits = np.frombuffer(mask_buf, dtype=np.uint8).reshape(n, -1)
            mask[off:off + n] = np.unpackbits(bits, axis=1)[:, :NUM_ACTIONS].astype(bool)
            off += n
        if not mask.any(axis=1).all():
            raise AssertionError(f"{self.name}: request row with no legal action")

        obs_t = torch.from_numpy(obs).to(self.device, non_blocking=True)
        mask_t = torch.from_numpy(mask).to(self.device, non_blocking=True)
        with torch.inference_mode():
            if self.use_autocast:
                with torch.autocast("cuda", dtype=self.autocast_dtype):
                    out = self.model(obs_t, mask_t)
            else:
                out = self.model(obs_t, mask_t)
            priors = torch.softmax(out["logits"].float(), dim=-1) * mask_t
            priors = priors / priors.sum(-1, keepdim=True).clamp_min(1e-12)
            values = out["value"].float()
        priors_np = priors.to("cpu").numpy().astype(np.float32, copy=False)
        values_np = values.to("cpu").numpy().astype(np.float32, copy=False)

        off = 0
        generation = int(self.watcher.generation)
        for (cid, req, n, _o, _m) in batch:
            q = response_qs.get(cid)
            if q is None:                                   # pragma: no cover
                off += n
                continue
            q.put((req,
                   np.ascontiguousarray(priors_np[off:off + n]).tobytes(),
                   np.ascontiguousarray(values_np[off:off + n]).tobytes(),
                   generation))
            off += n
        self.stats["batches"] += 1
        self.stats["requests"] += len(batch)
        self.stats["rows"] += total


def install_stop_handlers(on_signal) -> None:
    """SIGINT/SIGTERM set a flag; they do not raise.

    The PBS chain stops a job with ``timeout --signal=INT``, which signals the
    whole **process group** -- every actor and every inference server, not just
    the trainer.  Default handling would raise ``KeyboardInterrupt`` inside
    whatever the process happened to be doing (a search, a queue put, a
    ``torch`` forward), losing the wave in flight and its finished games.  With
    a flag instead, each process finishes what it is doing, ships it, and
    exits 0.
    """
    import signal

    def handler(signum, _frame):
        on_signal(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):                       # not the main thread
            pass


def server_main(device: str, net_cfg_dict: Dict[str, Any], weights_path: str,
                request_q, response_qs: Dict[int, Any], stop_event,
                max_batch: int = 1024, max_wait_ms: float = 1.0,
                reload_every_s: float = 30.0, torch_threads: int = 1,
                name: str = "infer", ready_event=None,
                stop_grace_s: float = 30.0) -> None:
    """Process entry point for one inference server."""
    configure_process(torch_threads)
    server = None
    try:
        server = InferenceServer(device, NetConfig.from_dict(dict(net_cfg_dict)),
                                 weights_path, max_batch=max_batch,
                                 max_wait_ms=max_wait_ms,
                                 reload_every_s=reload_every_s, name=name,
                                 stop_grace_s=stop_grace_s)

        def _stop(signum):
            print(f"[{name}] signal {signum}: draining", flush=True)
            server.signalled = True

        install_stop_handlers(_stop)
        if ready_event is not None:
            ready_event.set()
        server.run(request_q, response_qs, stop_event)
    except Exception:
        print(f"[{name}] died:\n{traceback.format_exc()}", flush=True)
        if ready_event is not None:
            ready_event.set()
        raise
