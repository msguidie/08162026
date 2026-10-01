#!/usr/bin/env python
"""Where does self-play throughput go?  Run it on the 4-GPU node, in the env:

    cd ~/scratch/splendor
    python scripts/speed_probe.py                  # every stage, ~12 min
    python scripts/speed_probe.py --only 1,2,3     # the fast stages, ~2 min
    python scripts/speed_probe.py --seconds 10     # shorter measurements

It reproduces one actor's hot loop exactly -- the lockstep rounds of
selfplay/actor.py `_run_searches`: one select_leaf per open tree, one
encode_batch, one evaluate call, one backup per leaf; a wave ends when every
tree has spent its budget, then every game plays its move -- and swaps what
sits underneath the evaluate call:

  1  instant evaluator (uniform priors, no net, no IPC)
       -> what the pure-Python search itself can do on one core
  2  the real net alone on one GPU, batch 8..4096
       -> the GPU ceiling, and what one small call costs
  3  one actor with the net in-process on a GPU (no IPC)
  4  the production path: real InferenceServer processes on GPU 1-3 and N
     actor processes, wired through the same queues train.py builds
  5  the production path with one knob changed at a time

The long run was measured at ~6,700 sims/s for the whole node.  Stage 4
should reproduce that; stages 1-3 say what it could be; stage 5 says which
change closes the gap.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import queue
import random
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

DEFAULT_CONFIG = os.path.join(REPO, "splendor_ai", "configs", "nscc_4xa100.yaml")


# ── positions ──────────────────────────────────────────────────────────────

def make_roots(count: int, seed: int) -> list:
    """2-player positions 0-40 random plies deep (the run was in its 2p phase)."""
    from splendor_ai.rules import engine as E

    rng = random.Random(seed)
    out = []
    while len(out) < count:
        s = E.new_game(2, E.MODE_INDIVIDUAL, rng=random.Random(rng.randrange(1 << 30)))
        for _ in range(rng.randrange(0, 40)):
            legal = E.legal_actions(s) if s.phase == E.PHASE_PLAYING else []
            if not legal:
                break
            E.apply(s, rng.choice(legal))
        if s.phase == E.PHASE_PLAYING and not E.is_stuck(s):
            out.append(s)
    return out


# ── evaluators ─────────────────────────────────────────────────────────────

class InstantEvaluator:
    """Uniform priors over the legal actions and zero values: no net, no IPC."""

    def evaluate(self, obs, mask):
        m = mask.astype(np.float32)
        return m / m.sum(axis=1, keepdims=True), np.zeros((mask.shape[0], 4), np.float32)


# ── the actor's hot loop, instrumented ──────────────────────────────────────

def actor_loop(evaluator, roots: list, games: int, seconds: float,
               full_cfg, fast_cfg, pcr: float, seed: int,
               warmup_s: float = 3.0) -> Dict[str, Any]:
    """selfplay/actor.py `_run_searches` + `_close_moves`, timed per phase."""
    from splendor_ai.encode import OBS_DIM, encode_batch
    from splendor_ai.rules import engine as E
    from splendor_ai.rules.actions import NUM_ACTIONS
    from splendor_ai.search.mcts import MCTS

    rng = np.random.default_rng(seed)
    obs = np.zeros((games, OBS_DIM), np.float32)
    maskbuf = np.zeros((games, NUM_ACTIONS), bool)

    def fresh():
        return roots[int(rng.integers(len(roots)))].clone()

    states = [fresh() for _ in range(games)]
    zero = {"select": 0.0, "encode": 0.0, "eval": 0.0, "backup": 0.0, "close": 0.0}
    t = dict(zero)
    n = {"sims": 0, "rows": 0, "calls": 0, "moves": 0}
    start = time.perf_counter()
    t_measure: Optional[float] = None
    wait0 = 0.0

    while True:
        # open one tree per game: the wave
        trees = []
        for i in range(games):
            s = states[i]
            if s.phase != E.PHASE_PLAYING or E.is_stuck(s):
                states[i] = fresh()
            cfg = full_cfg if rng.random() < pcr else fast_cfg
            trees.append([MCTS(cfg, np.random.default_rng(int(rng.integers(1 << 62)))),
                          int(cfg.sims), 0])
        # lockstep rounds until every tree has spent its budget
        while True:
            now = time.perf_counter()
            if t_measure is None and now - start >= warmup_s:
                t_measure = now
                t = dict(zero)
                n = {k: 0 for k in n}
                wait0 = float(getattr(evaluator, "wait_s", 0.0))
            if t_measure is not None and now - t_measure >= seconds:
                elapsed = now - t_measure
                return _summarise(t, n, elapsed, evaluator, wait0)
            t0 = time.perf_counter()
            pending = []
            active = False
            for i, tr in enumerate(trees):
                if tr[2] >= tr[1]:
                    continue
                active = True
                leaf = tr[0].select_leaf(states[i], states[i].current_player)
                tr[2] += 1
                n["sims"] += 1
                if leaf is not None:
                    pending.append((tr[0], leaf))
            t1 = time.perf_counter()
            t["select"] += t1 - t0
            if not active:
                break
            if not pending:
                continue
            b = len(pending)
            encode_batch([lf.state for _, lf in pending],
                         [lf.seat for _, lf in pending], out=obs[:b])
            for j, (_tree, lf) in enumerate(pending):
                maskbuf[j] = lf.mask
            t2 = time.perf_counter()
            t["encode"] += t2 - t1
            priors, values = evaluator.evaluate(obs[:b], maskbuf[:b])
            t3 = time.perf_counter()
            t["eval"] += t3 - t2
            for j, (tree, lf) in enumerate(pending):
                tree.backup(lf.token, priors[j], values[j])
            t["backup"] += time.perf_counter() - t3
            n["rows"] += b
            n["calls"] += 1
        # close the wave: every game plays the move its search chose
        t4 = time.perf_counter()
        for i, tr in enumerate(trees):
            E.apply(states[i], int(tr[0].result().action))
            n["moves"] += 1
        t["close"] += time.perf_counter() - t4


def _summarise(t, n, elapsed, evaluator, wait0) -> Dict[str, Any]:
    out = {"elapsed": elapsed, **n}
    out.update({f"t_{k}": v for k, v in t.items()})
    # RemoteEvaluator.wait_s counts from process start; keep the window only
    out["wait_s"] = float(getattr(evaluator, "wait_s", 0.0)) - wait0
    return out


def _rate_line(res: Dict[str, Any]) -> str:
    e = res["elapsed"]
    busy = {k: res[f"t_{k}"] / e for k in ("select", "encode", "eval", "backup", "close")}
    rows = res["rows"] / max(1, res["calls"])
    return (f"{res['sims'] / e:>9,.0f} sims/s {res['moves'] / e:>7.1f} moves/s"
            f"  {rows:>5.1f} rows/call"
            f"  time: select {busy['select']:.0%} encode {busy['encode']:.0%}"
            f" EVAL {busy['eval']:.0%} backup {busy['backup']:.0%}"
            f" close {busy['close']:.0%}")


# ── the production path: real servers + actor processes ──────────────────────

def _server_proc(device, net_cfg_dict, weights, request_q, response_qs, stop,
                 max_wait_ms, stats_q, name, ready):
    from splendor_ai.model import NetConfig
    from splendor_ai.selfplay import configure_process
    from splendor_ai.selfplay.inference import InferenceServer

    configure_process(1)
    server = InferenceServer(device, NetConfig.from_dict(dict(net_cfg_dict)), weights,
                             max_batch=1024, max_wait_ms=max_wait_ms,
                             reload_every_s=1e9, name=name, stop_grace_s=5.0)
    ready.set()
    server.run(request_q, response_qs, stop)
    stats_q.put((name, dict(server.stats)))


def _actor_proc(actor_id, request_q, response_q, ready_q, go, result_q,
                games, seconds, full_cfg, fast_cfg, pcr):
    from splendor_ai.selfplay import configure_process
    from splendor_ai.selfplay.inference import RemoteEvaluator

    configure_process(1)
    ev = RemoteEvaluator(actor_id, request_q, response_q, timeout_s=300.0,
                         start_id=RemoteEvaluator.start_id_for(actor_id + 1))
    roots = make_roots(48, seed=1000 + actor_id)
    ready_q.put(actor_id)
    go.wait()
    try:
        res = actor_loop(ev, roots, games, seconds, full_cfg, fast_cfg, pcr,
                         seed=actor_id)
    except Exception as exc:                                  # report, don't hang
        res = {"error": f"{type(exc).__name__}: {exc}"}
    result_q.put(res)


def run_pipeline(label: str, n_actors: int, devices: List[str], games: int,
                 seconds: float, max_wait_ms: float, full_cfg, fast_cfg,
                 pcr: float, net_cfg_dict, weights: str) -> Dict[str, Any]:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")                     # what train.py uses
    stop = ctx.Event()
    go = ctx.Event()
    request_qs = [ctx.Queue(maxsize=512) for _ in devices]
    response_qs = {i: ctx.Queue(maxsize=8) for i in range(n_actors)}
    stats_q, result_q, ready_q = ctx.Queue(), ctx.Queue(), ctx.Queue()

    t_spawn = time.perf_counter()
    servers, readies = [], []
    for k, dev in enumerate(devices):
        ready = ctx.Event()
        p = ctx.Process(target=_server_proc, daemon=True,
                        args=(dev, net_cfg_dict, weights, request_qs[k], response_qs,
                              stop, max_wait_ms, stats_q, f"infer{k}", ready))
        p.start()
        servers.append(p)
        readies.append(ready)
    for r in readies:
        r.wait(300)
    actors = []
    for i in range(n_actors):
        p = ctx.Process(target=_actor_proc, daemon=True,
                        args=(i, request_qs[i % len(devices)], response_qs[i], ready_q,
                              go, result_q, games, seconds, full_cfg, fast_cfg, pcr))
        p.start()
        actors.append(p)
    for _ in range(n_actors):                          # everyone measures together
        ready_q.get(timeout=600)
    spawn_s = time.perf_counter() - t_spawn
    go.set()
    results = [result_q.get(timeout=seconds + 900) for _ in range(n_actors)]
    for p in actors:
        p.join(30)
    stop.set()
    for q in request_qs:
        q.put(None)
    server_stats = []
    for _ in servers:
        try:
            server_stats.append(stats_q.get(timeout=60)[1])
        except queue.Empty:
            pass
    for p in servers:
        p.join(10)
        if p.is_alive():
            p.terminate()

    errors = [r["error"] for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    if not ok:
        print(f"  {label}: every actor failed: {errors[:3]}", flush=True)
        return {"label": label, "sims_per_s": 0.0}
    sims = sum(r["sims"] / r["elapsed"] for r in ok)
    moves = sum(r["moves"] / r["elapsed"] for r in ok)
    rows = sum(r["rows"] for r in ok) / max(1, sum(r["calls"] for r in ok))
    wait = sum(r["wait_s"] for r in ok) / sum(r["elapsed"] for r in ok)
    sel = sum(r["t_select"] for r in ok) / sum(r["elapsed"] for r in ok)
    per_batch_ms = rows_per_batch = reqs_per_batch = 0.0
    if server_stats:
        b = sum(s["batches"] for s in server_stats)
        if b:
            per_batch_ms = 1000 * sum(s["seconds"] for s in server_stats) / b
            rows_per_batch = sum(s["rows"] for s in server_stats) / b
            reqs_per_batch = sum(s["requests"] for s in server_stats) / b
    line = (f"  {label:<34} {sims:>9,.0f} sims/s {moves:>7.1f} moves/s"
            f" | actor: {rows:>5.1f} rows/call, waiting {wait:.0%}, searching {sel:.0%}"
            f" | server: {reqs_per_batch:>4.1f} req, {rows_per_batch:>6.1f} rows,"
            f" {per_batch_ms:>5.2f} ms per batch  (spawn {spawn_s:.0f}s)")
    if errors:
        line += f"  [{len(errors)} actors failed: {errors[0]}]"
    print(line, flush=True)
    return {"label": label, "sims_per_s": sims, "moves_per_s": moves,
            "wait": wait, "rows_per_call": rows}


# ── stages ──────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="measurement window per test (after a 3 s warm-up)")
    ap.add_argument("--only", default="1,2,3,4,5", help="comma-separated stages")
    ap.add_argument("--actors", type=int, default=0,
                    help="actor processes for stages 4-5 (0 = cores minus 8)")
    args = ap.parse_args()
    stages = {int(s) for s in args.only.split(",") if s.strip()}

    import torch
    from splendor_ai.model import NetEvaluator, SplendorNet, save_checkpoint
    from splendor_ai.selfplay import configure_process
    from splendor_ai.selfplay.config import load_config

    configure_process(1)
    cfg = load_config(args.config, [])
    phase = cfg.phase_for(0)
    full_cfg = dataclasses.replace(cfg.search_full, sims=int(phase.sims_full or cfg.search_full.sims))
    fast_cfg = dataclasses.replace(cfg.search_fast, sims=int(phase.sims_fast or cfg.search_fast.sims))
    pcr = float(cfg.selfplay.pcr_full_prob)
    games = int(cfg.selfplay.games_per_actor)
    cores = os.cpu_count() or 1
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    n_actors = args.actors or max(1, cores - 8)
    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devices = list(cfg.inference.devices)
    if ngpu < 4:
        devices = [f"cuda:{i}" for i in range(1 if ngpu > 1 else 0, ngpu)] or ["cpu"]
    probe_dev = devices[0]

    print("=" * 100)
    print(f"host {os.uname().nodename}   cores {cores}   torch {torch.__version__}"
          f"   cuda {torch.cuda.is_available()}   gpus {ngpu}")
    for i in range(ngpu):
        print(f"  cuda:{i}  {torch.cuda.get_device_name(i)}")
    print(f"config {os.path.relpath(args.config, REPO)}   net {cfg.net.width}x{cfg.net.blocks}"
          f"   phase-1 search {full_cfg.sims}/{fast_cfg.sims} sims (pcr {pcr})"
          f"   universes {full_cfg.universes}   games/actor {games}")
    print(f"servers {devices}   actors for stages 4-5: {n_actors}")
    print("=" * 100, flush=True)
    summary: Dict[str, float] = {}

    if 1 in stages:
        print("\n[1] one actor, INSTANT evaluator (no net, no IPC) — the pure-Python search per core")
        roots = make_roots(48, seed=1)
        for sims in ((full_cfg.sims, fast_cfg.sims), (600, 120)):
            f = dataclasses.replace(full_cfg, sims=sims[0])
            s = dataclasses.replace(fast_cfg, sims=sims[1])
            res = actor_loop(InstantEvaluator(), roots, games, args.seconds, f, s, pcr, seed=1)
            print(f"  budget {sims[0]:>3}/{sims[1]:<3}  {_rate_line(res)}", flush=True)
            if sims[0] == full_cfg.sims:
                summary["cpu_ceiling_per_core"] = res["sims"] / res["elapsed"]

    model = None
    if 2 in stages or 3 in stages:
        model = SplendorNet(cfg.net).to(probe_dev).eval()

    if 2 in stages and probe_dev.startswith("cuda"):
        print(f"\n[2] the {cfg.net.width}x{cfg.net.blocks} net alone on {probe_dev}"
              f" (bf16, numpy in -> numpy out, the server's path)")
        ev = NetEvaluator(model, device=probe_dev, autocast_dtype=torch.bfloat16)
        from splendor_ai.encode import OBS_DIM
        from splendor_ai.rules.actions import NUM_ACTIONS
        for b in (8, 16, 32, 64, 128, 256, 1024, 4096):
            obs = np.random.rand(b, OBS_DIM).astype(np.float32)
            mask = np.ones((b, NUM_ACTIONS), bool)
            for _ in range(5):
                ev.evaluate(obs, mask)
            k = 50 if b <= 1024 else 20
            t0 = time.perf_counter()
            for _ in range(k):
                ev.evaluate(obs, mask)
            dt = (time.perf_counter() - t0) / k
            print(f"  batch {b:>5}   {dt * 1000:>7.2f} ms/call   {b / dt:>12,.0f} rows/s", flush=True)
            if b == 1024:
                summary["gpu_rows_per_s_b1024"] = b / dt

    if 3 in stages and probe_dev.startswith("cuda"):
        print(f"\n[3] one actor, the net IN-PROCESS on {probe_dev} (no IPC)")
        roots = make_roots(48, seed=2)
        ev = NetEvaluator(model, device=probe_dev, autocast_dtype=torch.bfloat16)
        res = actor_loop(ev, roots, games, args.seconds, full_cfg, fast_cfg, pcr, seed=2)
        print(f"  games {games:<4}   {_rate_line(res)}", flush=True)
        big = 128
        res = actor_loop(ev, roots, big, args.seconds, full_cfg, fast_cfg, pcr, seed=3)
        print(f"  games {big:<4}   {_rate_line(res)}", flush=True)

    if 4 in stages or 5 in stages:
        tmp = tempfile.mkdtemp(prefix="speed_probe_")
        weights = os.path.join(tmp, "latest.pt")
        save_checkpoint(weights, SplendorNet(cfg.net), {})
        net_dict = cfg.net.to_dict()
        common = dict(seconds=args.seconds, full_cfg=full_cfg, fast_cfg=fast_cfg,
                      pcr=pcr, net_cfg_dict=net_dict, weights=weights)

        if 4 in stages:
            print(f"\n[4] PRODUCTION PATH: {len(devices)} inference servers, N actors,"
                  f" {games} games each, max_wait 1 ms")
            for n in sorted({1, 8, n_actors}):
                r = run_pipeline(f"{n} actors", n, devices, games, max_wait_ms=1.0, **common)
                summary[f"prod_{n}"] = r["sims_per_s"]

        if 5 in stages:
            print(f"\n[5] one knob at a time, {n_actors} actors")
            doubled = [d for d in devices for _ in range(2)]
            avg = int(round(pcr * full_cfg.sims + (1 - pcr) * fast_cfg.sims))
            variants = [
                ("64 games per actor", dict(devices=devices, games=64, max_wait_ms=1.0)),
                ("128 games per actor", dict(devices=devices, games=128, max_wait_ms=1.0)),
                (f"2 servers per GPU ({len(doubled)})", dict(devices=doubled, games=games,
                                                             max_wait_ms=1.0)),
                ("max_wait 5 ms", dict(devices=devices, games=games, max_wait_ms=5.0)),
            ]
            for label, kw in variants:
                r = run_pipeline(label, n_actors, **kw, **common)
                summary[f"knob:{label}"] = r["sims_per_s"]
            # What-if: every tree gets the same budget, so a wave is not gated
            # by its slowest (full) search.  Needs a code change to get for real.
            eq = dict(common)
            eq["full_cfg"] = dataclasses.replace(full_cfg, sims=avg)
            eq["fast_cfg"] = dataclasses.replace(fast_cfg, sims=avg)
            r = run_pipeline(f"what-if: every tree {avg} sims", n_actors, devices, games,
                             max_wait_ms=1.0, **eq)
            summary["whatif_equal_budget"] = r["sims_per_s"]

    print("\n" + "=" * 100)
    print("SUMMARY (sims/s, whole node unless noted)")
    for k, v in summary.items():
        print(f"  {k:<44} {v:>12,.0f}")
    if "cpu_ceiling_per_core" in summary:
        ceiling = summary["cpu_ceiling_per_core"] * n_actors
        print(f"  {'CPU ceiling = stage 1 x ' + str(n_actors) + ' actors':<44} {ceiling:>12,.0f}")
    print("=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
