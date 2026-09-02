"""Self-play trainer orchestrator (``docs/AI_DESIGN.md`` §1.8).

::

    python -m splendor_ai.selfplay.train --config splendor_ai/configs/smoke_cpu.yaml
    python -m splendor_ai.selfplay.train --config .../nscc_4xa100.yaml --resume runs/nscc0
    python -m splendor_ai.selfplay.train --config ... --set learner.batch=8192
    python -m splendor_ai.selfplay.train --config ... --warm-start bootstrap.pt
    python -m splendor_ai.selfplay.train --evaluate runs/smoke/checkpoints/gen_0008.pt \
        --config splendor_ai/configs/smoke_cpu.yaml

Process layout (the ``server`` inference mode; ``inproc`` drops the middle row)::

    main process ── learner (cuda:0) ── weights/latest.pt ──┐
         │  replay buffer, metrics, generations             │ (atomic rename,
         │                                                  │  polled by mtime)
         ├── inference servers  cuda:1  cuda:2  cuda:3 <────┤
         │        ^ request queues        | response queues │
         ├── actors x N  (each: G games in lockstep, PCR) ──┘
         │        └─ records ─> record queue ─> replay buffer
         └── eval process (NetBot / SearchBot vs random & greedy anchors)

Everything the run needs afterwards lands in ``run_dir``::

    config.yaml       the fully resolved config (including --set overrides)
    metrics.jsonl     one JSON object per line: actor stats, learner steps, evals
    weights/latest.pt the published net (actors and servers poll this)
    checkpoints/      gen_XXXX.pt per generation (opponent pool + arena history)
    trainer_state.pt  model + optimizer + counters + RNG (resume)
    replay.npz        the replay buffer (resume)
    progress.json     run-global counters actors poll for the curriculum

Shutdown is explicit: SIGINT/SIGTERM set the stop event, the record queue is
drained so no actor is blocked in ``put``, children are joined (then killed if
they overstay), and the weights and the checkpoint are written **before**
anything else.  That order is not cosmetic: the PBS chain stops a link with
``timeout --signal=INT ... --kill-after=10m``, so everything after the SIGINT
runs on borrowed time and a SIGKILL lands in the middle of it.  The final
evaluation used to run first and cost the run its checkpoint -- and with it the
whole chain, because the next link resumed from the previous checkpoint.  It is
now last, and by default it is skipped entirely when the stop came from a
signal (``eval.final_eval_seconds``).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from . import configure_process, set_thread_env

set_thread_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..model import SplendorNet, count_params, load_checkpoint  # noqa: E402
from .config import RunConfig, dump_config, load_config  # noqa: E402
from .metrics import MetricWriter, eval_main, evaluate_weights  # noqa: E402
from .replay import ReplayBuffer  # noqa: E402

__all__ = ["Trainer", "main"]


#: Batch schedulers (PBS, Slurm) have no place in the config tree — a job id is
#: not a knob, it changes every run and `--set job_id=...` is rejected by
#: RunConfig on purpose.  `scripts/nscc_train.pbs` exports it here instead and
#: it is recorded in metrics.jsonl so a chained run can be traced link by link.
JOB_ID_ENV = "SPLENDOR_JOB_ID"


def job_id() -> Optional[str]:
    """The batch job id this run belongs to, if the scheduler exported one."""
    value = (os.environ.get(JOB_ID_ENV) or "").strip()
    return value or None


#: Layout of the per-launch nonce handed to every actor process.  The high bits
#: are ``run_instance`` -- a counter persisted in ``trainer_state.pt`` and bumped
#: on every restore -- and the low :data:`_SPAWN_BITS` count process spawns
#: inside one launch, so a *restarted* actor is given a nonce its predecessor
#: never had.  20 bits in total, matching the field
#: :func:`splendor_ai.selfplay.actor.make_game_id` reserves for it.
#:
#: Actors mix the nonce into their RNG seeds, their game ids and (in ``server``
#: mode) their inference request ids.  Without it a resumed run replayed the
#: same deals in the same order under game ids that collided with the records
#: already in the buffer, and a restarted actor's request ids collided with its
#: dead predecessor's.  It wraps after 1024 launches of one run directory; the
#: replay window is 20 generations deep, so a wrapped nonce can never meet the
#: data it duplicates.
_SPAWN_BITS = 10
_INSTANCE_BITS = 20

#: How often the trainer refreshes ``run_dir/progress.json``.  Actors read it on
#: their own (slower) weight-refresh cadence; this only has to be fast enough
#: that a fresh or restarted actor never waits long for the run-global game
#: count its curriculum phase is a function of.
_PROGRESS_WRITE_S = 5.0


def _final_eval_main(cfg: RunConfig, weights: str, generation: int,
                     out_q) -> None:
    """One evaluation, in a child process, so it can be killed at a budget."""
    try:
        out_q.put(evaluate_weights(cfg, weights, generation=generation))
    except Exception as exc:                                # pragma: no cover
        out_q.put({"generation": generation,
                   "error": f"{type(exc).__name__}: {exc}"})


class Trainer:
    """Owns every process in the run."""

    def __init__(self, cfg: RunConfig, resume: bool = False) -> None:
        import multiprocessing as mp

        self.cfg = cfg
        cfg.make_dirs()
        self.ctx = mp.get_context("spawn")
        self.stop_event = self.ctx.Event()
        self.record_q = self.ctx.Queue(maxsize=64)
        self.stats_q = self.ctx.Queue(maxsize=256)
        self.eval_req_q = self.ctx.Queue(maxsize=8)
        self.eval_res_q = self.ctx.Queue(maxsize=8)
        self.actors: List[Any] = []
        self.servers: List[Any] = []
        self.request_qs: List[Any] = []
        self.response_qs: Dict[int, Any] = {}
        self.eval_proc: Optional[Any] = None

        self.metrics = MetricWriter(cfg.metrics_path,
                                    tensorboard=True, run_dir=cfg.run_dir)
        self.replay = ReplayBuffer(cfg.replay.window_start, cfg.replay.window_end,
                                   cfg.replay.window_ramp_generations,
                                   cfg.replay.max_samples,
                                   rng=np.random.default_rng(cfg.seed + 17))
        if cfg.learner.algorithm == "ppo":
            from .ppo_learner import PPOLearner

            self.learner: Any = PPOLearner(cfg)
        else:
            from .learner import Learner

            self.learner = Learner(cfg)

        self.games_done = 0
        self.records_seen = 0
        self.generation = 0
        self.gen_start_games = 0
        self.actor_restarts = 0
        self.stopping = False
        self.t0 = time.monotonic()
        self.last_checkpoint = self.t0
        self.last_stat_log = self.t0
        self.actor_stats: Dict[int, Dict[str, Any]] = {}
        self.eval_history: List[Dict[str, Any]] = []
        self.pending_evals = 0

        #: Per-launch nonce block for actor RNG seeds and game ids; bumped by
        #: :meth:`restore` and persisted, so no two launches of this run deal
        #: the same games (see :data:`_SPAWN_BITS`).
        self.run_instance = 1
        self._spawn_seq = 0
        #: Wall clock spent by *earlier* links of this run.  ``max_seconds`` is
        #: a run-lifetime budget, so it has to be restored, not restarted.
        self.elapsed_base = 0.0
        #: Counters as they stood at restore.  Throughput is reported twice:
        #: for this launch (the rate an operator can act on) and for the run.
        #: Dividing a restored lifetime counter by this launch's elapsed time
        #: reported a rate that was pure fiction.
        self.base_games = 0
        self.base_records = 0
        self.base_steps = 0
        #: Set by the SIGINT/SIGTERM handler.  A signal means the scheduler is
        #: about to SIGKILL this job: the checkpoint is the only thing that
        #: matters and the final evaluation is skipped.
        self.stopped_by_signal = False
        #: Restart bookkeeping per actor: recent death times (a sliding window)
        #: and the earliest time the replacement may start (backoff).
        self._restart_times: Dict[int, List[float]] = {}
        self._restart_due: Dict[int, float] = {}
        self._stale_replies: Dict[int, int] = {}
        self._warned_window = False
        self._prefetch: Optional[Any] = None
        self._last_progress_write = 0.0

        if resume:
            self.restore()

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        cfg = self.cfg
        # Publish before any actor starts: every actor must begin from the
        # *same* weights, not from its own random initialisation.
        self.learner.publish()
        self.metrics.log("run", {
            "run_dir": cfg.run_dir,
            "params": count_params(self.learner.model),
            "obs_dim": cfg.net.obs_dim,
            "actors": cfg.selfplay.actors,
            "games_per_actor": cfg.selfplay.games_per_actor,
            "sims_full": cfg.search_full.sims,
            "sims_fast": cfg.search_fast.sims,
            "batch": cfg.learner.batch,
            "inference": cfg.inference.mode,
            "algorithm": cfg.learner.algorithm,
            "win_threshold": cfg.selfplay.win_threshold,
            "resumed_step": self.learner.step,
            **({"job_id": job_id()} if job_id() else {}),
        }, step=self.learner.step, generation=self.generation)

        # Before the first actor exists: an actor reads its curriculum phase
        # from this file, and a resumed run's actors must not spend their first
        # `progress_refresh_s` playing phase 0.
        self._write_progress()
        if cfg.inference.mode == "server":
            self._start_servers()
        for i in range(cfg.selfplay.actors):
            self._start_actor(i)
        if cfg.eval.enabled and cfg.eval.async_process:
            self._start_eval()

    # -- run-global progress ---------------------------------------------
    def _write_progress(self) -> None:
        """Publish the counters the actors' curriculum reads (progress.json).

        The curriculum phase is a function of RUN-GLOBAL finished games, and an
        actor cannot know that number: its own counter starts at 0 every time
        it is spawned.  Deriving the phase from it rewound the whole node to
        phase 0 -- 2p at reduced simulations -- on every resume and every actor
        restart, silently undoing the curriculum for the rest of the chain.
        Temp file + atomic rename, so a reader never sees half a file.
        """
        payload = {
            "games_done": int(self.games_done),
            "records_seen": int(self.records_seen),
            "generation": int(self.generation),
            "step": int(self.learner.step),
            "instance": int(self.run_instance),
            "updated_at": time.time(),
        }
        path = self.cfg.progress_path
        tmp = f"{path}.tmp{os.getpid()}"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, path)
        except OSError as exc:                              # pragma: no cover
            print(f"[train] could not write {path}: {exc}", flush=True)
        self._last_progress_write = time.monotonic()

    def _next_instance(self) -> int:
        """A fresh per-launch nonce for one actor process."""
        seq = self._spawn_seq
        self._spawn_seq += 1
        value = (self.run_instance << _SPAWN_BITS) | (seq & ((1 << _SPAWN_BITS) - 1))
        return value & ((1 << _INSTANCE_BITS) - 1)

    def _start_servers(self) -> None:
        from .inference import server_main

        cfg = self.cfg
        for i in range(cfg.selfplay.actors):
            self.response_qs[i] = self.ctx.Queue(maxsize=8)
        for d, device in enumerate(cfg.inference.devices):
            request_q = self.ctx.Queue(maxsize=cfg.inference.queue_size)
            ready = self.ctx.Event()
            proc = self.ctx.Process(
                target=server_main,
                args=(device, cfg.net.to_dict(), cfg.latest_weights, request_q,
                      self.response_qs, self.stop_event),
                kwargs=dict(max_batch=cfg.inference.max_batch,
                            max_wait_ms=cfg.inference.max_wait_ms,
                            reload_every_s=cfg.inference.reload_every_s,
                            torch_threads=cfg.torch_threads,
                            name=f"infer{d}:{device}", ready_event=ready,
                            # `timeout --signal=INT` signals the whole process
                            # group, so a server that exited on the spot would
                            # strand every actor mid-search on response_q.get
                            # and lose the wave in flight.  It drains instead.
                            stop_grace_s=cfg.selfplay.stop_grace_s),
                name=f"infer{d}", daemon=True)
            proc.start()
            self.servers.append(proc)
            self.request_qs.append(request_q)
            ready.wait(timeout=120)

    def _start_actor(self, actor_id: int) -> None:
        """Spawn (or replace) one actor process.

        Two things the replacement must not inherit from the actor it replaces:

        * **its queued replies.**  A dead actor can have had an inference
          request in flight; the server's answer is still sitting in the
          response queue.  The queue object itself cannot be recreated (the
          servers were handed the dict at spawn and would keep writing to the
          old one), so it is drained here, and
          :class:`~.inference.RemoteEvaluator` starts its request ids in a
          block derived from the new nonce and skips any id that is not its
          own.  Both halves are needed: the drain cannot catch a reply that
          arrives a moment later.
        * **its RNG stream and game ids.**  Those come from the nonce.
        """
        from .actor import actor_main

        cfg = self.cfg
        request_q = response_q = None
        if cfg.inference.mode == "server":
            request_q = self.request_qs[actor_id % len(self.request_qs)]
            response_q = self.response_qs[actor_id]
            dropped = self._drain_response_queue(actor_id)
            if dropped:
                self._stale_replies[actor_id] = \
                    self._stale_replies.get(actor_id, 0) + dropped
                print(f"[train] actor {actor_id}: dropped {dropped} stale "
                      f"inference repl{'y' if dropped == 1 else 'ies'} left by "
                      f"its predecessor", flush=True)
        instance = self._next_instance()
        proc = self.ctx.Process(
            target=actor_main,
            args=(cfg, actor_id, self.record_q, self.stats_q, self.stop_event,
                  request_q, response_q),
            kwargs=dict(instance=instance, games_offset=self.games_done),
            name=f"actor{actor_id}", daemon=True)
        proc.start()
        while len(self.actors) <= actor_id:
            self.actors.append(None)
        self.actors[actor_id] = proc

    def _drain_response_queue(self, actor_id: int) -> int:
        """Throw away whatever the dead actor never collected."""
        import queue

        q = self.response_qs.get(actor_id)
        if q is None:
            return 0
        dropped = 0
        while True:
            try:
                q.get_nowait()
            except (queue.Empty, OSError, ValueError):
                break
            dropped += 1
        return dropped

    def _start_eval(self) -> None:
        self.eval_proc = self.ctx.Process(
            target=eval_main,
            args=(self.cfg, self.eval_req_q, self.eval_res_q, self.stop_event),
            name="eval", daemon=True)
        self.eval_proc.start()

    # -- queues ----------------------------------------------------------
    def _drain_records(self, budget: int = 64) -> int:
        import queue

        added = 0
        for _ in range(budget):
            try:
                msg = self.record_q.get_nowait()
            except queue.Empty:
                break
            self.games_done += int(msg.get("games", 0))
            buf = msg.get("buf") or b""
            if buf:
                n = self.replay.add(buf)
                self.records_seen += n
                added += n
        return added

    def _drain_stats(self) -> None:
        import queue

        while True:
            try:
                msg = self.stats_q.get_nowait()
            except queue.Empty:
                return
            self.actor_stats[int(msg.get("actor", -1))] = msg

    def _drain_evals(self) -> None:
        import queue

        while True:
            try:
                result = self.eval_res_q.get_nowait()
            except queue.Empty:
                return
            self.pending_evals = max(0, self.pending_evals - 1)
            self.eval_history.append(result)
            self.metrics.log("eval", result, step=self.learner.step,
                             generation=int(result.get("generation", 0)))
            print(f"[eval] gen {result.get('generation')} "
                  f"net-vs-random {result.get('net_vs_random')} "
                  f"net-vs-greedy {result.get('net_vs_greedy')} "
                  f"search-vs-greedy {result.get('search_vs_greedy')}"
                  + (f" ERROR {result['error']}" if "error" in result else ""),
                  flush=True)

    # -- generations -----------------------------------------------------
    def _maybe_close_generation(self) -> None:
        if self.games_done - self.gen_start_games < self.cfg.games_per_generation:
            return
        size = self.replay.close_generation(self.generation)
        self.learner.generation = self.generation
        ckpt = self.learner.save_generation(self.generation)
        self._warn_if_window_truncated()
        self.metrics.log("generation", {
            "generation": self.generation,
            "games_done": self.games_done,
            "samples_in_generation": size,
            "buffer": len(self.replay),
            # `window` is the window the ramp asks for; `window_retained` is
            # what is actually held.  They differ exactly when replay
            # max_samples is choosing the window instead of the design.
            "window": self.replay.window_size(),
            "window_retained": self.replay.retained_generations(),
            "window_truncated": bool(self.replay.window_truncated()),
            "cap_dropped_samples": self.replay.cap_dropped_samples,
            "records_seen": self.records_seen,
            "steps": self.learner.step,
            "samples_consumed": self.learner.samples_consumed,
            "reuse": (self.learner.samples_consumed / max(1, self.records_seen)),
            "actor_restarts": self.actor_restarts,
            "throughput": self._throughput(),
        }, step=self.learner.step, generation=self.generation)
        if self.cfg.eval.enabled and (
                self.generation % max(1, self.cfg.eval.every_generations) == 0):
            self._request_eval(ckpt, self.generation)
        self.gen_start_games = self.games_done
        self.generation += 1
        self._write_progress()

    def _warn_if_window_truncated(self) -> None:
        """Say so, once, when ``replay.max_samples`` is cutting the window.

        Silently training on 6 generations while the config says 20 is the kind
        of thing that only ever shows up as "the run is worse than the last
        one".  Warn once (this is per generation, and a permanent condition),
        and log the retained window every time.
        """
        if not self.replay.window_truncated() or self._warned_window:
            return
        self._warned_window = True
        print(f"[train] replay.max_samples={self.replay.max_samples:,} is "
              f"truncating the window: holding "
              f"{self.replay.retained_generations()} of "
              f"{self.replay.window_size()} generations "
              f"({len(self.replay):,} samples, "
              f"{self.replay.cap_dropped_samples:,} dropped by the cap so "
              f"far).  Raise replay.max_samples, or lower replay.window_end "
              f"or games_per_generation, so the config states the window the "
              f"run is really training on.", flush=True)
        self.metrics.log("replay_window_truncated", {
            "max_samples": self.replay.max_samples,
            "window": self.replay.window_size(),
            "window_retained": self.replay.retained_generations(),
            "buffer": len(self.replay),
            "cap_dropped_samples": self.replay.cap_dropped_samples,
        }, step=self.learner.step, generation=self.generation)

    def _request_eval(self, weights: str, generation: int) -> None:
        request = {"weights": weights, "generation": generation,
                   "step": self.learner.step}
        if self.eval_proc is not None:
            try:
                self.eval_req_q.put_nowait(request)
                self.pending_evals += 1
            except Exception:                               # queue full: skip
                pass
            return
        result = evaluate_weights(self.cfg, weights, generation)  # inline
        self.eval_res_q.put(result)

    def _elapsed(self) -> float:
        """Wall clock of the whole run, across every link of the chain."""
        return self.elapsed_base + (time.monotonic() - self.t0)

    def _throughput(self) -> Dict[str, float]:
        """Rates for this launch, plus lifetime rates for the whole run.

        ``games_done``/``records_seen``/``step`` are *lifetime* counters
        restored from the checkpoint.  Dividing them by the time since this
        process started (which is what used to happen) reports a rate that
        starts at "everything the run has ever done, in one second" and decays
        from there -- worse than useless on a chained run, because it looks
        plausible.  The interval rates below subtract the counters as they
        stood at restore; the lifetime rates divide by the run's cumulative
        elapsed time instead.
        """
        elapsed = max(1e-9, time.monotonic() - self.t0)
        lifetime = max(1e-9, self._elapsed())
        totals = {"sims": 0.0, "moves": 0.0, "games": 0.0}
        for stat in self.actor_stats.values():
            for key in totals:
                totals[key] += float(stat.get(key, 0.0))
        games = self.games_done - self.base_games
        records = self.records_seen - self.base_records
        steps = self.learner.step - self.base_steps
        return {
            "elapsed_s": elapsed,
            "lifetime_s": lifetime,
            # this launch (actor sims/moves are per actor process anyway)
            "sims_per_s": totals["sims"] / elapsed,
            "moves_per_s": totals["moves"] / elapsed,
            "games_per_s": games / elapsed,
            "records_per_s": records / elapsed,
            "steps_per_s": steps / elapsed,
            "games_this_launch": float(games),
            "records_this_launch": float(records),
            "steps_this_launch": float(steps),
            # whole run
            "lifetime_games_per_s": self.games_done / lifetime,
            "lifetime_records_per_s": self.records_seen / lifetime,
            "lifetime_steps_per_s": self.learner.step / lifetime,
        }

    # -- health ----------------------------------------------------------
    def _check_actors(self) -> None:
        """Restart dead actors, with a budget and a backoff.

        An actor that dies on its first search dies again on its first search.
        Restarting it immediately and for ever burned a core, flooded the log
        with the same traceback, and left a run that looked alive while
        producing nothing.  Deaths are counted in a sliding window
        (``selfplay.restart_window_s``); more than ``selfplay.restart_budget``
        of them aborts the run with the actor's exit code in the message, and
        each restart waits ``restart_backoff_s * 2**(n-1)`` seconds first.
        """
        if self.stopping:
            return
        now = time.monotonic()
        for actor_id, proc in enumerate(self.actors):
            if proc is None or proc.is_alive():
                continue
            due = self._restart_due.get(actor_id)
            if due is None:
                due = self._note_actor_death(actor_id, proc.exitcode, now)
            if now < due:
                continue
            self._restart_due.pop(actor_id, None)
            print(f"[train] actor {actor_id}: restarting "
                  f"(total restarts {self.actor_restarts})", flush=True)
            self._start_actor(actor_id)
        for i, proc in enumerate(self.servers):
            if not proc.is_alive():
                raise RuntimeError(
                    f"inference server {i} exited with {proc.exitcode}; "
                    f"see its traceback above")

    def _note_actor_death(self, actor_id: int, code: Optional[int],
                          now: float) -> float:
        """Record one death; return the time its replacement may start."""
        sp = self.cfg.selfplay
        times = self._restart_times.setdefault(actor_id, [])
        times[:] = [t for t in times if now - t <= sp.restart_window_s]
        times.append(now)
        self.actor_restarts += 1
        budget = max(1, int(sp.restart_budget))
        delay = min(float(sp.restart_backoff_max_s),
                    float(sp.restart_backoff_s) * (2.0 ** (len(times) - 1)))
        self.metrics.log("actor_restart",
                         {"actor": actor_id, "exitcode": code,
                          "restarts": self.actor_restarts,
                          "in_window": len(times), "budget": budget,
                          "backoff_s": delay},
                         step=self.learner.step, generation=self.generation)
        if len(times) > budget:
            raise RuntimeError(
                f"actor {actor_id} died {len(times)} times in the last "
                f"{sp.restart_window_s:.0f}s (selfplay.restart_budget="
                f"{budget}), last exit code {code}.  Restarting it again "
                f"would only reproduce the traceback above, which is the real "
                f"failure -- fix that, or raise selfplay.restart_budget if "
                f"these deaths are genuinely transient.")
        print(f"[train] actor {actor_id} exited with {code} "
              f"({len(times)}/{budget} deaths in the last "
              f"{sp.restart_window_s:.0f}s); restarting in {delay:.0f}s",
              flush=True)
        self._restart_due[actor_id] = now + delay
        return now + delay

    # -- checkpoint / resume ---------------------------------------------
    def checkpoint(self, final: bool = False) -> None:
        """Write ``trainer_state.pt`` (+ ``replay.npz``) atomically.

        ``final`` forces the replay save to be synchronous: a background save
        that has not finished when the process exits leaves the *previous*
        buffer on disk (which resumes correctly, only older), and at shutdown
        there is no next tick to catch up on.
        """
        cfg = self.cfg
        state = {
            "learner": self.learner.state_dict(),
            "replay": self.replay.state_dict(),
            "games_done": self.games_done,
            "records_seen": self.records_seen,
            "generation": self.generation,
            "gen_start_games": self.gen_start_games,
            "actor_restarts": self.actor_restarts,
            # Cumulative across the chain, not "since this process started":
            # `max_seconds` is a run-lifetime budget (see `_should_stop`).
            "elapsed_s": self._elapsed(),
            "run_instance": self.run_instance,
            "numpy_rng": self.replay.rng.bit_generator.state,
            "eval_history": self.eval_history[-32:],
        }
        tmp = f"{cfg.state_path}.tmp{os.getpid()}"
        torch.save(state, tmp)
        os.replace(tmp, cfg.state_path)
        replay_mode = "skipped"
        if cfg.replay.checkpoint:
            if cfg.replay.checkpoint_async and not final:
                # ~41 s at the production window, and this thread is also the
                # learner.  The saver takes a snapshot of the existing
                # generation arrays under the lock and writes them while the
                # run carries on; the rename is atomic.
                replay_mode = ("async" if self.replay.save_async(cfg.replay_path)
                               else "busy")
                if replay_mode == "busy":
                    print("[train] replay checkpoint skipped: the previous "
                          "background save is still running", flush=True)
            else:
                self.replay.save(cfg.replay_path)
                replay_mode = "sync"
        self.last_checkpoint = time.monotonic()
        self._write_progress()
        self.metrics.log("checkpoint", {
            "path": cfg.state_path, "buffer": len(self.replay),
            "games_done": self.games_done, "steps": self.learner.step,
            "replay_save": replay_mode, "final": bool(final),
            "elapsed_s": self._elapsed(),
        }, step=self.learner.step, generation=self.generation)

    def restore(self) -> None:
        cfg = self.cfg
        if not os.path.exists(cfg.state_path):
            print(f"[train] --resume: no {cfg.state_path}, starting fresh",
                  flush=True)
            return
        state = torch.load(cfg.state_path, map_location="cpu", weights_only=False)
        self.learner.load_state_dict(state["learner"])
        self.games_done = int(state.get("games_done", 0))
        self.records_seen = int(state.get("records_seen", 0))
        self.generation = int(state.get("generation", 0))
        self.gen_start_games = int(state.get("gen_start_games", 0))
        self.actor_restarts = int(state.get("actor_restarts", 0))
        self.eval_history = list(state.get("eval_history", []))
        meta = state.get("replay", {})
        if os.path.exists(cfg.replay_path):
            self.replay.load(cfg.replay_path)
        self.replay.generation = int(meta.get("generation", self.replay.generation))
        self.replay.total_added = int(meta.get("total_added", 0))
        self.replay.total_dropped = int(meta.get("total_dropped", 0))
        rng_state = state.get("numpy_rng")
        if rng_state is not None:
            self.replay.rng.bit_generator.state = rng_state
        # A new launch of the same run: bump the nonce every actor mixes into
        # its RNG seeds and game ids, so this link does not replay the deals
        # the previous one already trained on, under ids that collide with
        # them.
        self.run_instance = int(state.get("run_instance", 0)) + 1
        # Cumulative wall clock, and the counters this launch starts from.
        self.elapsed_base = float(state.get("elapsed_s", 0.0))
        self.base_games = self.games_done
        self.base_records = self.records_seen
        self.base_steps = self.learner.step
        print(f"[train] resumed at step {self.learner.step}, generation "
              f"{self.generation}, {self.games_done} games, buffer "
              f"{len(self.replay)}, run instance {self.run_instance}, "
              f"{self.elapsed_base / 60.0:.1f} min already spent", flush=True)

    # -- stop conditions -------------------------------------------------
    def _should_stop(self) -> bool:
        cfg = self.cfg
        if self.stopping:
            return True
        # Run-lifetime budget, restored from the checkpoint: `max_seconds` is
        # how long the whole run may take, not how long each link of the PBS
        # chain may take (that is the scheduler's walltime and `TRAIN_TIMEOUT`).
        if cfg.max_seconds and self._elapsed() >= cfg.max_seconds:
            return True
        if cfg.max_generations and self.generation >= cfg.max_generations:
            return True
        if cfg.max_games and self.games_done >= cfg.max_games:
            return True
        if cfg.max_steps and self.learner.step >= cfg.max_steps:
            return True
        return False

    # -- main loop -------------------------------------------------------
    def _start_prefetch(self):
        """Background preparation of the next learner batch (opt-in).

        Batch prep -- rehydrating the positions, ``encode_batch``, densifying
        the policy, rotating the seat-major vectors -- is ~272 ms per 4096
        records, and it used to happen between optimizer steps with the GPU
        idle.  Both halves release the GIL, so a thread overlaps them.
        """
        if not self.cfg.learner.prefetch:
            return None
        from .learner import BatchPrefetcher

        cfg = self.cfg
        blend = float(cfg.learner.value_blend)

        def make(buf):
            return self.replay.batch(self.learner.local_batch,
                                     value_blend=blend, obs_out=buf)

        def ready():
            return self.learner.ready(self.records_seen, len(self.replay))

        return BatchPrefetcher(make, ready, self.learner.local_batch,
                               cfg.net.obs_dim,
                               poll_s=max(0.005, cfg.learner.idle_sleep_s))

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        self._install_signals()
        if cfg.max_seconds and self._elapsed() >= cfg.max_seconds:
            print(f"[train] max_seconds={cfg.max_seconds:.0f}s is a RUN "
                  f"budget and earlier links already spent "
                  f"{self._elapsed():.0f}s of it: nothing to do.  Raise it "
                  f"(--set max_seconds=...) or set it to 0 for no limit.",
                  flush=True)
        self.start()
        self._prefetch = self._start_prefetch()
        last_log_step = -1
        try:
            while not self._should_stop():
                self._drain_records()
                self._drain_stats()
                self._drain_evals()
                self._maybe_close_generation()
                self._check_actors()
                if time.monotonic() - self._last_progress_write >= \
                        _PROGRESS_WRITE_S:
                    self._write_progress()

                stepped = False
                if self.learner.ready(self.records_seen, len(self.replay)):
                    batch = None
                    if self._prefetch is not None:
                        # Short wait, then prepare it here: a batch is more
                        # important than the overlap that saves 272 ms of it.
                        batch = self._prefetch.get(
                            timeout=max(0.05, cfg.learner.idle_sleep_s * 4))
                    if batch is None:
                        batch = self.replay.batch(
                            self.learner.local_batch,
                            value_blend=cfg.learner.value_blend)
                    metrics = self.learner.train_step(batch)
                    stepped = True
                    if (self.learner.step % max(1, cfg.learner.publish_every)) == 0:
                        self.learner.publish()
                    if (self.learner.step % max(1, cfg.learner.log_every)) == 0 \
                            and self.learner.step != last_log_step:
                        last_log_step = self.learner.step
                        metrics.update({
                            "buffer": len(self.replay),
                            "games_done": self.games_done,
                            "records_seen": self.records_seen,
                            "reuse": self.learner.samples_consumed /
                                     max(1, self.records_seen),
                        })
                        self.metrics.log("learner", metrics,
                                         step=self.learner.step,
                                         generation=self.generation)
                if not stepped:
                    time.sleep(cfg.learner.idle_sleep_s)
                if time.monotonic() - self.last_checkpoint >= \
                        cfg.learner.checkpoint_every_s:
                    self.checkpoint()
                if time.monotonic() - self.last_stat_log >= 30.0:
                    self.last_stat_log = time.monotonic()
                    self._log_progress()
        except KeyboardInterrupt:                           # pragma: no cover
            print("[train] interrupted", flush=True)
        finally:
            summary = self.shutdown()
        return summary

    def _log_progress(self) -> None:
        tp = self._throughput()
        stats = {
            "games_done": self.games_done,
            "generation": self.generation,
            "buffer": len(self.replay),
            "steps": self.learner.step,
            "actor_restarts": self.actor_restarts,
        }
        stats.update(tp)
        agg: Dict[str, Any] = {}
        for key in ("stuck_rate", "truncation_rate", "disagreement"):
            values = [float(s.get(key, 0.0)) for s in self.actor_stats.values()]
            if values:
                agg[key] = float(np.mean(values))
        mode_plies: Dict[str, List[float]] = {}
        # Per-mode game counts and truncation rates: the actors have always
        # counted these and the trainer used to drop them on the floor, which
        # left the curriculum -- the thing that decides which modes are played
        # at all -- with no observable output whatsoever.  Counts are per actor
        # process lifetime, so they reset when an actor restarts.
        mode_games: Dict[str, int] = {}
        mode_truncations: Dict[str, int] = {}
        for s in self.actor_stats.values():
            for mode, value in (s.get("mode_plies") or {}).items():
                mode_plies.setdefault(mode, []).append(float(value))
            for mode, value in (s.get("mode_games") or {}).items():
                mode_games[mode] = mode_games.get(mode, 0) + int(value)
            for mode, value in (s.get("mode_truncations") or {}).items():
                mode_truncations[mode] = mode_truncations.get(mode, 0) + int(value)
        agg["mode_plies"] = {k: float(np.mean(v)) for k, v in mode_plies.items()}
        agg["mode_games"] = mode_games
        agg["mode_truncations"] = mode_truncations
        agg["mode_truncation_rate"] = {
            k: (mode_truncations.get(k, 0) / max(1, v))
            for k, v in mode_games.items()}
        agg["actor_global_games"] = int(max(
            [int(s.get("global_games", 0)) for s in self.actor_stats.values()]
            or [0]))
        agg["window_retained"] = self.replay.retained_generations()
        agg["window"] = self.replay.window_size()
        if self._prefetch is not None:
            agg["prefetch_batches"] = int(self._prefetch.produced)
            agg["prefetch_errors"] = int(self._prefetch.errors)
        agg["replay_saving"] = bool(self.replay.saving())
        # Provenance and health of the evaluator side, aggregated over actors:
        # the generation the records being written were produced by (0 for the
        # whole run means the server is not stamping its responses), replies
        # dropped as another actor's (should be 0 unless an actor restarted),
        # and historical checkpoint loads (the cache-miss counter).
        for key, reduce in (("weight_generation", max), ("eval_stale", sum),
                            ("historical_loads", sum), ("weight_reloads", sum)):
            values = [int(s.get(key, 0)) for s in self.actor_stats.values()]
            if values:
                agg[key] = int(reduce(values))
        stats.update(agg)
        self.metrics.log("progress", stats, step=self.learner.step,
                         generation=self.generation)
        modes = " ".join(f"{k}:{v}" for k, v in sorted(mode_games.items())) \
            or "(none yet)"
        print(f"[train] t={tp['elapsed_s']:.0f}s gen={self.generation} "
              f"games={self.games_done} buffer={len(self.replay)} "
              f"steps={self.learner.step} "
              f"sims/s={tp['sims_per_s']:.0f} moves/s={tp['moves_per_s']:.1f} "
              f"games/s={tp['games_per_s']:.2f}", flush=True)
        print(f"[train]   modes {modes} | stuck {agg.get('stuck_rate', 0.0):.3f}"
              f" trunc {agg.get('truncation_rate', 0.0):.3f}"
              f" | window {self.replay.retained_generations()}/"
              f"{self.replay.window_size()} gens", flush=True)
        self._write_progress()

    # -- shutdown --------------------------------------------------------
    def _final_eval(self) -> None:
        """The last point of the learning curve -- if there is time for it.

        Three cases:

        * stopped by a signal and ``eval.final_eval_seconds == 0`` (the
          default): **skipped**.  A signal means the scheduler is about to
          SIGKILL this job; a production evaluation is 100 paired games plus 25
          search games and takes tens of minutes.  The checkpoint is already
          written, and the per-generation evaluations are already in
          ``metrics.jsonl``.
        * ``final_eval_seconds > 0``: run it in a child process and kill it at
          that budget, so a slow evaluation cannot cost the job its exit.
        * stopped by a limit in the config (``max_games``, ``max_seconds``,
          ``--smoke``): run it inline, as before.  Nothing is waiting on this
          job and this is the number the G3 gate is read from.
        """
        cfg = self.cfg
        if not cfg.eval.enabled:
            return
        budget = float(cfg.eval.final_eval_seconds or 0.0)
        if self.stopped_by_signal and budget <= 0.0:
            print("[train] final evaluation skipped: the run was stopped by a "
                  "signal and eval.final_eval_seconds is 0.  The checkpoint is "
                  "written and resumable; set eval.final_eval_seconds to a "
                  "wall-clock budget in seconds if this run really needs a "
                  "final evaluation on the way out.", flush=True)
            self.metrics.log("eval_skipped",
                             {"reason": "stopped_by_signal",
                              "final_eval_seconds": budget},
                             step=self.learner.step, generation=self.generation)
            return
        final: Optional[Dict[str, Any]] = None
        if budget > 0.0:
            out_q = self.ctx.Queue(maxsize=1)
            proc = self.ctx.Process(target=_final_eval_main,
                                    args=(cfg, cfg.latest_weights,
                                          self.generation, out_q),
                                    name="final-eval", daemon=True)
            proc.start()
            proc.join(timeout=budget)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                print(f"[train] final evaluation exceeded its "
                      f"{budget:.0f}s budget (eval.final_eval_seconds) and was "
                      f"killed; the checkpoint is unaffected", flush=True)
                self.metrics.log("eval_skipped",
                                 {"reason": "budget_exceeded",
                                  "final_eval_seconds": budget},
                                 step=self.learner.step,
                                 generation=self.generation)
            else:
                try:
                    final = out_q.get_nowait()
                except Exception:                           # pragma: no cover
                    final = None
            try:
                out_q.close()
                out_q.cancel_join_thread()
            except Exception:                               # pragma: no cover
                pass
        else:
            try:
                final = evaluate_weights(cfg, cfg.latest_weights,
                                         generation=self.generation)
            except Exception:                               # pragma: no cover
                print(f"[train] final eval failed:\n{traceback.format_exc()}",
                      flush=True)
                return
        if not final:
            return
        final["final"] = True
        self.eval_history.append(final)
        self.metrics.log("eval", final, step=self.learner.step,
                         generation=self.generation)
        print(f"[eval] FINAL gen {self.generation} "
              f"net-vs-random {final.get('net_vs_random')} "
              f"net-vs-greedy {final.get('net_vs_greedy')} "
              f"search-vs-greedy {final.get('search_vs_greedy')}", flush=True)

    def _install_signals(self) -> None:
        def handler(signum, _frame):
            if self.stopping:
                return
            print(f"[train] signal {signum}: stopping (checkpoint first)",
                  flush=True)
            # `timeout --signal=INT ... --kill-after=10m` means a SIGKILL is
            # coming.  Remember that the stop was a signal: `shutdown` writes
            # the checkpoint before anything optional, and skips the final
            # evaluation entirely unless eval.final_eval_seconds says how long
            # it may take.
            self.stopped_by_signal = True
            self.stopping = True
            self.stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except ValueError:                              # pragma: no cover
                pass

    def shutdown(self) -> Dict[str, Any]:
        """Stop everything, **checkpoint first**, then anything optional.

        Order matters more than anything else in this method.  The PBS chain
        stops a link with ``timeout --signal=INT ... --kill-after=10m``: after
        the SIGINT the job has minutes, and then it is SIGKILLed.  The final
        evaluation used to run before the checkpoint, so a production run was
        killed in the middle of a 20-minute evaluation with nothing written --
        no ``trainer_state.pt``, no published weights, and the next link of the
        chain resumed from the previous checkpoint (or from scratch).
        """
        self.stopping = True
        self.stop_event.set()
        if self._prefetch is not None:
            self._prefetch.close()
            self._prefetch = None
        # Actors finish the wave in flight and ship it; give them room to.
        grace = max(1.0, float(self.cfg.selfplay.stop_grace_s))
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            self._drain_records()
            self._drain_stats()
            self._drain_evals()
            alive = [p for p in self.actors if p is not None and p.is_alive()]
            if not alive:
                break
            time.sleep(0.05)
        # The servers leave on their own a few seconds after the last request
        # (they wait for the request queue to stay quiet); joining them here
        # turns what would otherwise be a SIGTERM into a clean exit.
        server_deadline = time.monotonic() + min(15.0, grace)
        for proc in self.servers:
            proc.join(timeout=max(0.0, server_deadline - time.monotonic()))
        late = [i for i, p in enumerate(self.actors)
                if p is not None and p.is_alive()]
        if late:
            print(f"[train] actors {late} did not stop within {grace:.0f}s "
                  f"(selfplay.stop_grace_s); terminating them -- their "
                  f"unfinished games are lost", flush=True)
        for proc in list(self.actors) + list(self.servers) + \
                ([self.eval_proc] if self.eval_proc else []):
            if proc is None:
                continue
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=5)
        # A shipment made just before an actor exited can still be inside the
        # queue's feeder thread, and `get_nowait` reports Empty while bytes are
        # in the pipe -- so drain until several consecutive passes come back
        # with nothing rather than reading the queue once.  These are finished
        # games: dropping them here would undo the whole point of letting the
        # actors finish their wave.
        settle = time.monotonic() + 2.0
        misses = 0
        while time.monotonic() < settle and misses < 4:
            before = (self.games_done, self.records_seen)
            self._drain_records()
            self._drain_stats()
            self._drain_evals()
            if (self.games_done, self.records_seen) == before:
                misses += 1
                time.sleep(0.05)
            else:
                misses = 0
        if self.replay._pending:
            self.replay.close_generation(self.generation)

        # 1. THE CHECKPOINT.  Everything below this line is optional; nothing
        #    below it may run before it.
        try:
            self.learner.publish()
            self.checkpoint(final=True)
            print(f"[train] final checkpoint: step {self.learner.step}, "
                  f"generation {self.generation}, {self.games_done} games, "
                  f"buffer {len(self.replay)}", flush=True)
        except Exception:                                   # pragma: no cover
            print(f"[train] final checkpoint failed:\n{traceback.format_exc()}",
                  flush=True)

        # 2. The final evaluation, if there is time for it.
        self._final_eval()
        summary = {
            "games_done": self.games_done,
            "records_seen": self.records_seen,
            "generations": self.generation,
            "steps": self.learner.step,
            "actor_restarts": self.actor_restarts,
            "buffer": len(self.replay),
            "run_instance": self.run_instance,
            "stopped_by_signal": self.stopped_by_signal,
            "elapsed_s": self._elapsed(),
            "throughput": self._throughput(),
            "evals": self.eval_history,
        }
        if job_id():
            summary["job_id"] = job_id()
        self.metrics.log("summary", summary, step=self.learner.step,
                         generation=self.generation)
        self.metrics.close()
        self.replay.join_save(timeout=60.0)
        for q in [self.record_q, self.stats_q, self.eval_req_q, self.eval_res_q]:
            try:
                q.close()
                q.cancel_join_thread()
            except Exception:                               # pragma: no cover
                pass
        return summary


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m splendor_ai.selfplay.train",
        description="AlphaZero self-play trainer for the Splendor variant")
    p.add_argument("--config", default=None, help="YAML run config")
    p.add_argument("--resume", default=None, metavar="RUN_DIR",
                   help="resume the run in RUN_DIR (config.yaml is read from "
                        "there unless --config is given)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config leaf, e.g. --set learner.batch=512")
    p.add_argument("--evaluate", default=None, metavar="CHECKPOINT",
                   help="evaluate one checkpoint against the anchors and exit")
    p.add_argument("--warm-start", default=None, metavar="CHECKPOINT",
                   help="initialise the network from CHECKPOINT (e.g. the "
                        "output of splendor_ai.selfplay.bootstrap) before the "
                        "first generation; ignored when --resume finds state")
    p.add_argument("--print-config", action="store_true",
                   help="print the resolved config and exit")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config
    if args.resume and not config_path:
        candidate = os.path.join(args.resume, "config.yaml")
        if os.path.exists(candidate):
            config_path = candidate
    overrides = list(args.set)
    if args.resume:
        overrides.append(f"run_dir={args.resume}")
    try:
        cfg = load_config(config_path, overrides)
    except (ValueError, OSError) as exc:
        # A typo in --set or a missing config file is a submission mistake, not
        # a crash: say what is wrong on one line and exit non-zero (the PBS
        # chain treats any status other than 0/124 as "do not re-submit").
        print(f"[train] bad configuration: {exc}", file=sys.stderr, flush=True)
        return 2

    if args.print_config:
        from .config import config_to_dict
        import yaml

        print(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))
        return 0

    configure_process(cfg.learner_threads, seed=cfg.seed)
    if args.evaluate:
        result = evaluate_weights(cfg, args.evaluate)
        print(result)
        return 0

    cfg.make_dirs()
    dump_config(cfg, os.path.join(cfg.run_dir, "config.yaml"))
    trainer = Trainer(cfg, resume=bool(args.resume))
    if args.warm_start and trainer.learner.step == 0:
        trainer.learner.warm_start(args.warm_start)
        print(f"[train] warm started from {args.warm_start}", flush=True)
    try:
        summary = trainer.run()
    except RuntimeError as exc:
        # A supervisor decision, not a crash: the actor restart budget was
        # exhausted, or an inference server died.  `run` has already been
        # through `shutdown`, so the checkpoint is written.  Exit 3 -- the PBS
        # chain re-submits only on 0 and 124, so this stops the chain instead
        # of letting it burn a queue slot on the same failure 30 more times.
        print(f"[train] aborted: {exc}", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 3
    print(f"[train] done: {summary['games_done']} games, "
          f"{summary['generations']} generations, {summary['steps']} steps, "
          f"{summary['actor_restarts']} actor restarts", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
