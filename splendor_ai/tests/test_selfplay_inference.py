"""Inference server, weight hot-reload and the obs_version gate.

The integration case is the one the design doc asks for: two real actor
processes talking to one CPU inference server over the queues, producing
records that survive the record invariants.
"""

import multiprocessing as mp
import os
import queue
import time

import numpy as np
import pytest
import torch

from splendor_ai.encode import OBS_DIM
from splendor_ai.model import (NetConfig, NetEvaluator, SplendorNet,
                               load_checkpoint, save_checkpoint)
from splendor_ai.rules import engine as E
from splendor_ai.selfplay import sample as S
from splendor_ai.selfplay.config import load_config
from splendor_ai.selfplay.inference import (LocalEvaluator, RemoteEvaluator,
                                            WeightWatcher, server_main)


def _publish(path, cfg, seed=0):
    torch.manual_seed(seed)
    model = SplendorNet(cfg)
    save_checkpoint(path, model, {"step": seed, "generation": seed,
                                  "meta": {"version": seed}})
    return model


def _random_inputs(n=5, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((n, OBS_DIM)).astype(np.float32)
    mask = rng.random((n, 65)) < 0.4
    mask[:, 0] = True                                   # never an empty row
    return obs, mask


def test_weight_watcher_reloads_and_gates_obs_version(tmp_path):
    cfg = NetConfig(width=32, blocks=1)
    path = str(tmp_path / "latest.pt")
    first = _publish(path, cfg, seed=1)
    live = SplendorNet(cfg)
    watcher = WeightWatcher(path, live, min_interval_s=0.0)
    assert watcher.poll(force=True) is True
    obs, mask = _random_inputs()
    a = NetEvaluator(first, "cpu").evaluate(obs, mask)
    b = NetEvaluator(live, "cpu").evaluate(obs, mask)
    assert np.allclose(a[0], b[0], atol=1e-6) and np.allclose(a[1], b[1], atol=1e-6)
    assert watcher.poll() is False                      # unchanged file

    time.sleep(0.01)
    second = _publish(path, cfg, seed=2)
    assert watcher.poll(force=True) is True
    c = NetEvaluator(second, "cpu").evaluate(obs, mask)
    d = NetEvaluator(live, "cpu").evaluate(obs, mask)
    assert np.allclose(c[0], d[0], atol=1e-6)
    assert watcher.reloads == 2

    # tamper with the observation version: the gate must fire, loudly
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["obs_version"] = payload["obs_version"] + 99
    torch.save(payload, path)
    with pytest.raises(RuntimeError) as exc:
        load_checkpoint(path)
    assert "obs_version" in str(exc.value)
    with pytest.raises(RuntimeError):
        watcher.poll(force=True)


def test_local_evaluator_rejects_an_empty_mask_row(tmp_path):
    cfg = NetConfig(width=32, blocks=1)
    path = str(tmp_path / "latest.pt")
    _publish(path, cfg)
    model = SplendorNet(cfg)
    evaluator = LocalEvaluator(model, path, refresh_s=0.0)
    obs, mask = _random_inputs(3)
    mask[1, :] = False
    with pytest.raises(AssertionError):
        evaluator.evaluate(obs, mask)


def test_server_matches_the_local_evaluator(tmp_path):
    ctx = mp.get_context("spawn")
    cfg = NetConfig(width=32, blocks=1)
    path = str(tmp_path / "latest.pt")
    model = _publish(path, cfg, seed=5)

    request_q = ctx.Queue()
    response_qs = {0: ctx.Queue(), 1: ctx.Queue()}
    stop = ctx.Event()
    ready = ctx.Event()
    proc = ctx.Process(target=server_main,
                       args=("cpu", cfg.to_dict(), path, request_q, response_qs,
                             stop),
                       kwargs=dict(max_batch=64, max_wait_ms=2.0,
                                   ready_event=ready, name="test-server"),
                       daemon=True)
    proc.start()
    try:
        assert ready.wait(timeout=120)
        reference = NetEvaluator(model, "cpu")
        for client in (0, 1):
            remote = RemoteEvaluator(client, request_q, response_qs[client],
                                     timeout_s=60)
            obs, mask = _random_inputs(7, seed=client)
            priors, values = remote.evaluate(obs, mask)
            want_p, want_v = reference.evaluate(obs, mask)
            assert priors.shape == (7, 65) and values.shape == (7, 4)
            assert np.allclose(priors, want_p, atol=1e-5)
            assert np.allclose(values, want_v, atol=1e-5)
            assert np.allclose(priors[~mask], 0.0)
    finally:
        stop.set()
        request_q.put(None)
        proc.join(timeout=20)
        if proc.is_alive():                                 # pragma: no cover
            proc.terminate()
            proc.join(timeout=5)


def test_two_actors_through_one_server(tmp_path):
    """The §1.8 ``server`` layout end to end on CPU: 2 actors, 1 server."""
    from splendor_ai.selfplay.actor import actor_main

    ctx = mp.get_context("spawn")
    cfg = load_config(None, [
        f"run_dir={tmp_path}/run", "net.width=32", "net.blocks=1",
        "selfplay.actors=2", "selfplay.games_per_actor=2",
        "selfplay.pcr_full_prob=1.0", "selfplay.win_threshold=5",
        "selfplay.augment_rotations=2", "selfplay.mixed_game_frac=0.0",
        "search_full.sims=8", "search_full.universes=1",
        "search_fast.sims=4", "search_fast.universes=1",
        "inference.mode=server", "inference.devices=[cpu]",
        "inference.max_batch=64", "inference.max_wait_ms=2.0",
        "selfplay.max_plies=60"])
    cfg.make_dirs()
    _publish(cfg.latest_weights, cfg.net, seed=3)

    request_q = ctx.Queue()
    response_qs = {0: ctx.Queue(), 1: ctx.Queue()}
    record_q = ctx.Queue()
    stats_q = ctx.Queue()
    stop = ctx.Event()
    ready = ctx.Event()
    server = ctx.Process(target=server_main,
                         args=("cpu", cfg.net.to_dict(), cfg.latest_weights,
                               request_q, response_qs, stop),
                         kwargs=dict(max_batch=64, max_wait_ms=2.0,
                                     ready_event=ready, name="test-server"),
                         daemon=True)
    server.start()
    actors = []
    try:
        assert ready.wait(timeout=120)
        for i in (0, 1):
            proc = ctx.Process(target=actor_main,
                               args=(cfg, i, record_q, stats_q, stop,
                                     request_q, response_qs[i]),
                               kwargs=dict(max_waves=25), daemon=True)
            proc.start()
            actors.append(proc)
        deadline = time.time() + 180
        seen = 0
        payloads = []
        while time.time() < deadline and any(p.is_alive() for p in actors):
            try:
                msg = record_q.get(timeout=1.0)
            except queue.Empty:
                continue
            payloads.append(msg)
            seen += int(msg.get("n", 0))
        while True:
            try:
                payloads.append(record_q.get_nowait())
            except queue.Empty:
                break
        for proc in actors:
            proc.join(timeout=30)
            assert proc.exitcode == 0, f"actor exited with {proc.exitcode}"
        seen = sum(int(m.get("n", 0)) for m in payloads)
        assert seen > 0, "no records came back from the actors"
        records = np.concatenate([S.records_from_bytes(m["buf"])
                                  for m in payloads if m.get("n")])
        S.check_records(records)
        assert set(np.unique(records["mode"]).tolist()) <= set(range(len(S.MODE_TAGS)))
    finally:
        stop.set()
        request_q.put(None)
        for proc in actors + [server]:
            proc.join(timeout=10)
            if proc.is_alive():                             # pragma: no cover
                proc.terminate()
                proc.join(timeout=5)
