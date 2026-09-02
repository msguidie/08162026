"""Instrumentation: the metrics log, the eval bots and the anchor arena.

Two jobs live here because they are two halves of the same question ("is the
run working?"):

1. :class:`MetricWriter` — one JSON object per line in ``run_dir/metrics.jsonl``
   plus an optional TensorBoard mirror behind a guarded import.  Every line
   carries ``t`` (seconds since the run started), ``step``, ``generation`` and a
   ``kind`` so the file can be sliced without a schema.
2. The **evaluation ladder** of §1.7 / §2: ``NetBot`` (policy argmax, no
   search) and ``SearchBot`` (net + MCTS) played against the fixed anchors
   ``RandomBot`` and ``GreedyBot`` over *paired, seat-swapped* games — both
   games of a pair use the same deck seed, so only the seating differs and the
   deal variance cancels (judges.md "EVALUATION BUDGET").

``splendor_ai/arena.py`` belongs to another stream.  If it is importable and
exposes a paired-match entry point we use it; otherwise we fall back to
``bots.play_game``, and to a local copy of its loop when the INDIVIDUAL
win-threshold override is in force (``play_game`` has no hook for the engine
config, and a smoke run trained at threshold 8 has to be *evaluated* at
threshold 8 or the number means nothing).
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..encode import encode
from ..rules import engine as E
from ..search.mcts import SearchConfig, run_search
from ..values import standings_values, terminal_values
from .config import MODE_SPECS, RunConfig

__all__ = ["MetricWriter", "NetBot", "SearchBot", "play_one", "paired_match",
           "evaluate_weights", "eval_main"]


class MetricWriter:
    """Append-only JSONL (+ optional TensorBoard) metrics sink."""

    def __init__(self, path: str, tensorboard: bool = False,
                 run_dir: Optional[str] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.t0 = time.time()
        self._fh = open(path, "a", buffering=1)
        self._tb = None
        if tensorboard:
            try:                                            # guarded on purpose
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(os.path.join(run_dir or
                                                      os.path.dirname(path), "tb"))
            except Exception as exc:                        # pragma: no cover
                print(f"[metrics] tensorboard disabled: {exc}", flush=True)

    def log(self, kind: str, data: Dict[str, Any], step: int = 0,
            generation: int = 0) -> Dict[str, Any]:
        row = {"t": round(time.time() - self.t0, 3), "kind": kind,
               "step": int(step), "generation": int(generation)}
        row.update(data)
        self._fh.write(json.dumps(row, default=_jsonable) + "\n")
        if self._tb is not None:                            # pragma: no cover
            for key, value in data.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._tb.add_scalar(f"{kind}/{key}", value, step)
        return row

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            if self._tb is not None:                        # pragma: no cover
                self._tb.close()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ── bots that need the network ────────────────────────────────────────────

class NetBot:
    """Policy argmax over the legal mask — no search (§1.7)."""

    def __init__(self, evaluator, name: str = "net") -> None:
        self.evaluator = evaluator
        self.name = name

    def act(self, state: E.GameState, seat: int, rng=None) -> Optional[int]:
        mask = np.asarray(E.legal_mask(state), dtype=bool)
        if not mask.any():
            return None
        obs = encode(state, seat)[None]
        priors, _values = self.evaluator.evaluate(obs, mask[None])
        return int(np.argmax(priors[0]))


class SearchBot:
    """Net + MCTS at ``sims`` simulations, deterministic (no noise, argmax)."""

    def __init__(self, evaluator, sims: int = 48, universes: int = 2,
                 name: Optional[str] = None) -> None:
        self.evaluator = evaluator
        self.cfg = SearchConfig(sims=int(sims), noise=False, universes=universes,
                                temperature_plies=0, prune_policy_target=False)
        self.name = name or f"search{sims}"

    def act(self, state: E.GameState, seat: int, rng=None) -> Optional[int]:
        if E.is_stuck(state):
            return None
        if rng is None or not hasattr(rng, "integers"):
            rng = np.random.default_rng(0)
        result = run_search(state, seat, self.evaluator, encode, self.cfg, rng)
        return int(result.action)


# ── game driver ───────────────────────────────────────────────────────────

def _arena_module():
    """``splendor_ai.arena`` if the other stream has landed it, else None."""
    try:
        from .. import arena                                # type: ignore
    except Exception:
        return None
    return arena


def play_one(seats: Sequence[Any], mode_name: str = "ind2", seed: int = 0,
             max_plies: int = 400, win_threshold: Optional[int] = None
             ) -> Dict[str, Any]:
    """One game; the same contract as :func:`splendor_ai.bots.play_game`.

    Delegates to ``bots.play_game`` unless an INDIVIDUAL win-threshold
    override is in force, in which case it runs the identical loop against a
    state whose engine config carries the override.
    """
    n, mode, layout = MODE_SPECS[mode_name]
    if win_threshold is None:
        from ..bots import play_game

        return play_game(list(seats), mode, n, layout, seed=seed,
                         max_plies=max_plies)

    state = E.new_game(n, mode, layout, rng=random.Random(seed))
    state.config = dict(state.config)
    state.config["winThreshold"] = int(win_threshold)
    rng = np.random.default_rng(seed)
    plies = 0
    stuck_resigns = 0
    while state.phase == E.PHASE_PLAYING and plies < max_plies:
        seat = state.current_player
        if E.is_stuck(state):
            E.resign(state, seat)
            stuck_resigns += 1
            plies += 1
            continue
        action = seats[seat].act(state, seat, rng)
        if action is None:
            E.resign(state, seat)
            stuck_resigns += 1
            plies += 1
            continue
        if not E.legal_mask(state)[action]:
            raise ValueError(
                f"bot {getattr(seats[seat], 'name', seat)} returned illegal "
                f"action {action} for seat {seat}")
        E.apply(state, action)
        plies += 1
    truncated = state.phase == E.PHASE_PLAYING
    values = standings_values(state) if truncated else terminal_values(state)
    return {
        "mode": mode, "num_players": n, "layout": layout, "seed": seed,
        "plies": plies, "truncated": truncated,
        "values": [float(v) for v in values],
        "scores": [p.score for p in state.players],
        "stuck_resigns": stuck_resigns,
        "names": [getattr(b, "name", "?") for b in seats],
    }


def paired_match(bot, opponent, mode_name: str = "ind2", pairs: int = 12,
                 seed: int = 0, max_plies: int = 400,
                 win_threshold: Optional[int] = None) -> Dict[str, float]:
    """Seat-swapped paired games; returns the win rate of ``bot``.

    A pair is two games on the *same* deck seed with the seats rotated, so the
    deal cancels.  Score: win = 1, draw (equal value) = 0.5.  For n > 2 the
    seat rotation cycles ``bot`` through every seat.
    """
    n, _mode, _layout = MODE_SPECS[mode_name]
    arena = _arena_module()
    if arena is not None and hasattr(arena, "paired_match") and win_threshold is None:
        return arena.paired_match(bot, opponent, mode=mode_name, pairs=pairs,
                                  seed=seed)                # pragma: no cover
    score = 0.0
    games = 0
    plies: List[int] = []
    truncated = 0
    for p in range(pairs):
        for rot in range(n):
            seats = [opponent] * n
            seats[rot] = bot
            result = play_one(seats, mode_name, seed=seed + p,
                              max_plies=max_plies, win_threshold=win_threshold)
            values = result["values"]
            mine = values[rot]
            best = max(values[:n])
            if mine >= best - 1e-9:
                ties = sum(1 for v in values[:n] if v >= best - 1e-9)
                score += 1.0 / ties
            games += 1
            plies.append(result["plies"])
            truncated += int(result["truncated"])
    return {"win_rate": score / max(1, games), "games": games,
            "mean_plies": float(np.mean(plies)) if plies else 0.0,
            "truncated": truncated / max(1, games)}


def evaluate_weights(cfg: RunConfig, weights_path: str,
                     generation: int = 0) -> Dict[str, Any]:
    """Full evaluation round for one checkpoint (§2 G3/G5).

    ``net_vs_random``, ``net_vs_greedy`` (policy argmax, no search) and
    ``search_vs_greedy`` (net + ``sims_eval`` simulations).
    """
    from ..bots import GreedyBot, RandomBot
    from ..model import NetEvaluator, load_checkpoint

    ev_cfg = cfg.eval
    model, ckpt = load_checkpoint(weights_path, map_location="cpu")
    evaluator = NetEvaluator(model.eval(), "cpu")
    net_bot = NetBot(evaluator)
    mode_name = _primary_mode(cfg)
    thr = cfg.selfplay.win_threshold
    out: Dict[str, Any] = {
        "generation": generation,
        "weights_step": int(ckpt.get("step", 0)),
        "mode": mode_name,
    }
    t0 = time.perf_counter()
    if "random" in ev_cfg.opponents:
        res = paired_match(net_bot, RandomBot(), mode_name, ev_cfg.pairs,
                           seed=1000 + generation,
                           max_plies=cfg.selfplay.max_plies, win_threshold=thr)
        out["net_vs_random"] = res["win_rate"]
        out["net_vs_random_games"] = res["games"]
    if "greedy" in ev_cfg.opponents:
        res = paired_match(net_bot, GreedyBot(), mode_name, ev_cfg.pairs,
                           seed=2000 + generation,
                           max_plies=cfg.selfplay.max_plies, win_threshold=thr)
        out["net_vs_greedy"] = res["win_rate"]
        out["net_vs_greedy_games"] = res["games"]
        out["net_vs_greedy_plies"] = res["mean_plies"]
    if ev_cfg.search_bot and ev_cfg.search_pairs > 0:
        search_bot = SearchBot(evaluator, sims=ev_cfg.sims_eval,
                               universes=max(1, cfg.search_full.universes))
        res = paired_match(search_bot, GreedyBot(), mode_name,
                           ev_cfg.search_pairs, seed=3000 + generation,
                           max_plies=cfg.selfplay.max_plies, win_threshold=thr)
        out["search_vs_greedy"] = res["win_rate"]
        out["search_vs_greedy_games"] = res["games"]
    out["eval_seconds"] = time.perf_counter() - t0
    return out


def _primary_mode(cfg: RunConfig) -> str:
    mixture = cfg.phase_for(0).mixture
    return max(mixture.items(), key=lambda kv: float(kv[1]))[0]


def eval_main(cfg: RunConfig, request_q, result_q, stop_event=None) -> None:
    """Process entry point: evaluate on demand so the learner never blocks."""
    import traceback

    from . import configure_process

    configure_process(cfg.torch_threads, seed=cfg.seed + 991)
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        try:
            request = request_q.get(timeout=0.5)
        except Exception:
            continue
        if request is None:
            break
        try:
            out = evaluate_weights(cfg, request["weights"],
                                   generation=int(request.get("generation", 0)))
            out["step"] = int(request.get("step", 0))
            result_q.put(out)
        except Exception as exc:
            print(f"[eval] failed: {traceback.format_exc()}", flush=True)
            result_q.put({"generation": int(request.get("generation", 0)),
                          "error": f"{type(exc).__name__}: {exc}"})
