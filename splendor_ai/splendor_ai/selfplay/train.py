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

Shutdown is explicit: SIGINT/SIGTERM set the stop event, the record queue is
drained so no actor is blocked in ``put``, children are joined (then killed if
they overstay), and a final checkpoint is written.
"""

from __future__ import annotations

import argparse
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
        }, step=self.learner.step, generation=self.generation)

        if cfg.inference.mode == "server":
            self._start_servers()
        for i in range(cfg.selfplay.actors):
            self._start_actor(i)
        if cfg.eval.enabled and cfg.eval.async_process:
            self._start_eval()

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
                            name=f"infer{d}:{device}", ready_event=ready),
                name=f"infer{d}", daemon=True)
            proc.start()
            self.servers.append(proc)
            self.request_qs.append(request_q)
            ready.wait(timeout=120)

    def _start_actor(self, actor_id: int) -> None:
        from .actor import actor_main

        cfg = self.cfg
        request_q = response_q = None
        if cfg.inference.mode == "server":
            request_q = self.request_qs[actor_id % len(self.request_qs)]
            response_q = self.response_qs[actor_id]
        proc = self.ctx.Process(
            target=actor_main,
            args=(cfg, actor_id, self.record_q, self.stats_q, self.stop_event,
                  request_q, response_q),
            name=f"actor{actor_id}", daemon=True)
        proc.start()
        while len(self.actors) <= actor_id:
            self.actors.append(None)
        self.actors[actor_id] = proc

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
        self.metrics.log("generation", {
            "generation": self.generation,
            "games_done": self.games_done,
            "samples_in_generation": size,
            "buffer": len(self.replay),
            "window": self.replay.window_size(),
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

    def _throughput(self) -> Dict[str, float]:
        elapsed = max(1e-9, time.monotonic() - self.t0)
        totals = {"sims": 0.0, "moves": 0.0, "games": 0.0}
        for stat in self.actor_stats.values():
            for key in totals:
                totals[key] += float(stat.get(key, 0.0))
        return {
            "elapsed_s": elapsed,
            "sims_per_s": totals["sims"] / elapsed,
            "moves_per_s": totals["moves"] / elapsed,
            "games_per_s": self.games_done / elapsed,
            "records_per_s": self.records_seen / elapsed,
            "steps_per_s": self.learner.step / elapsed,
        }

    # -- health ----------------------------------------------------------
    def _check_actors(self) -> None:
        if self.stopping:
            return
        for actor_id, proc in enumerate(self.actors):
            if proc is None or proc.is_alive():
                continue
            code = proc.exitcode
            self.actor_restarts += 1
            self.metrics.log("actor_restart",
                             {"actor": actor_id, "exitcode": code,
                              "restarts": self.actor_restarts},
                             step=self.learner.step, generation=self.generation)
            print(f"[train] actor {actor_id} exited with {code}; restarting "
                  f"(total restarts {self.actor_restarts})", flush=True)
            self._start_actor(actor_id)
        for i, proc in enumerate(self.servers):
            if not proc.is_alive():
                raise RuntimeError(
                    f"inference server {i} exited with {proc.exitcode}; "
                    f"see its traceback above")

    # -- checkpoint / resume ---------------------------------------------
    def checkpoint(self) -> None:
        cfg = self.cfg
        state = {
            "learner": self.learner.state_dict(),
            "replay": self.replay.state_dict(),
            "games_done": self.games_done,
            "records_seen": self.records_seen,
            "generation": self.generation,
            "gen_start_games": self.gen_start_games,
            "actor_restarts": self.actor_restarts,
            "elapsed_s": time.monotonic() - self.t0,
            "numpy_rng": self.replay.rng.bit_generator.state,
            "eval_history": self.eval_history[-32:],
        }
        tmp = f"{cfg.state_path}.tmp{os.getpid()}"
        torch.save(state, tmp)
        os.replace(tmp, cfg.state_path)
        if cfg.replay.checkpoint:
            self.replay.save(cfg.replay_path)
        self.last_checkpoint = time.monotonic()
        self.metrics.log("checkpoint", {
            "path": cfg.state_path, "buffer": len(self.replay),
            "games_done": self.games_done, "steps": self.learner.step,
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
        print(f"[train] resumed at step {self.learner.step}, generation "
              f"{self.generation}, {self.games_done} games, buffer "
              f"{len(self.replay)}", flush=True)

    # -- stop conditions -------------------------------------------------
    def _should_stop(self) -> bool:
        cfg = self.cfg
        if self.stopping:
            return True
        if cfg.max_seconds and (time.monotonic() - self.t0) >= cfg.max_seconds:
            return True
        if cfg.max_generations and self.generation >= cfg.max_generations:
            return True
        if cfg.max_games and self.games_done >= cfg.max_games:
            return True
        if cfg.max_steps and self.learner.step >= cfg.max_steps:
            return True
        return False

    # -- main loop -------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        self._install_signals()
        self.start()
        last_log_step = -1
        try:
            while not self._should_stop():
                self._drain_records()
                self._drain_stats()
                self._drain_evals()
                self._maybe_close_generation()
                self._check_actors()

                stepped = False
                if self.learner.ready(self.records_seen, len(self.replay)):
                    batch = self.replay.batch(self.learner.local_batch,
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
        agg = {}
        for key in ("stuck_rate", "truncation_rate", "disagreement"):
            values = [float(s.get(key, 0.0)) for s in self.actor_stats.values()]
            if values:
                agg[key] = float(np.mean(values))
        mode_plies: Dict[str, List[float]] = {}
        for s in self.actor_stats.values():
            for mode, value in (s.get("mode_plies") or {}).items():
                mode_plies.setdefault(mode, []).append(float(value))
        agg["mode_plies"] = {k: float(np.mean(v)) for k, v in mode_plies.items()}
        stats.update(agg)
        self.metrics.log("progress", stats, step=self.learner.step,
                         generation=self.generation)
        print(f"[train] t={tp['elapsed_s']:.0f}s gen={self.generation} "
              f"games={self.games_done} buffer={len(self.replay)} "
              f"steps={self.learner.step} "
              f"sims/s={tp['sims_per_s']:.0f} moves/s={tp['moves_per_s']:.1f} "
              f"games/s={tp['games_per_s']:.2f}", flush=True)

    # -- shutdown --------------------------------------------------------
    def _install_signals(self) -> None:
        def handler(signum, _frame):
            if self.stopping:
                return
            print(f"[train] signal {signum}: stopping", flush=True)
            self.stopping = True
            self.stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except ValueError:                              # pragma: no cover
                pass

    def shutdown(self) -> Dict[str, Any]:
        self.stopping = True
        self.stop_event.set()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            self._drain_records()
            self._drain_stats()
            self._drain_evals()
            alive = [p for p in self.actors if p is not None and p.is_alive()]
            if not alive:
                break
            time.sleep(0.05)
        for proc in list(self.actors) + list(self.servers) + \
                ([self.eval_proc] if self.eval_proc else []):
            if proc is None:
                continue
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=5)
        self._drain_records()
        self._drain_stats()
        self._drain_evals()
        if self.replay._pending:
            self.replay.close_generation(self.generation)
        if self.cfg.eval.enabled:
            # One last evaluation of the published weights, run inline: the
            # eval process is gone by now and the final point of the learning
            # curve is the one the G3 gate is read from.
            try:
                final = evaluate_weights(self.cfg, self.cfg.latest_weights,
                                         generation=self.generation)
                final["final"] = True
                self.eval_history.append(final)
                self.metrics.log("eval", final, step=self.learner.step,
                                 generation=self.generation)
                print(f"[eval] FINAL gen {self.generation} "
                      f"net-vs-random {final.get('net_vs_random')} "
                      f"net-vs-greedy {final.get('net_vs_greedy')} "
                      f"search-vs-greedy {final.get('search_vs_greedy')}",
                      flush=True)
            except Exception:                               # pragma: no cover
                print(f"[train] final eval failed:\n{traceback.format_exc()}",
                      flush=True)
        try:
            self.checkpoint()
            self.learner.publish()
        except Exception:                                   # pragma: no cover
            print(f"[train] final checkpoint failed:\n{traceback.format_exc()}",
                  flush=True)
        summary = {
            "games_done": self.games_done,
            "records_seen": self.records_seen,
            "generations": self.generation,
            "steps": self.learner.step,
            "actor_restarts": self.actor_restarts,
            "buffer": len(self.replay),
            "throughput": self._throughput(),
            "evals": self.eval_history,
        }
        self.metrics.log("summary", summary, step=self.learner.step,
                         generation=self.generation)
        self.metrics.close()
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
    cfg = load_config(config_path, overrides)

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
    summary = trainer.run()
    print(f"[train] done: {summary['games_done']} games, "
          f"{summary['generations']} generations, {summary['steps']} steps, "
          f"{summary['actor_restarts']} actor restarts", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
