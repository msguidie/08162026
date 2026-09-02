"""NN-free MCTS teacher -> supervised warm start (``docs/AI_DESIGN.md`` §1.8).

The very first generations of self-play are expensive precisely because the
network is random: the search has no prior worth following and the value head
has nothing to say.  A cheap fix, and the one the design doc leaves optional,
is to spend a few CPU-minutes generating targets from a *network-free* searcher
— :class:`~splendor_ai.bots.MctsBot` with the greedy rollout evaluator, which
G2 already showed beats the greedy bot ~75% of the time at 400 sims — and to
fit the network to those before the real loop starts.

    python -m splendor_ai.selfplay.bootstrap --config configs/smoke_cpu.yaml \
        --games 40 --sims 96 --steps 800 --out runs/smoke_cpu/weights/latest.pt

The records are exactly the :mod:`.sample` records the actors write (C5
augmentation included), so the warm-started buffer can be handed straight to
the learner, and ``--out weights/latest.pt`` is a valid starting point for
``train.py`` (which will publish over it once it takes its first step).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..rules import engine as E
from ..search.evaluators import RolloutEvaluator, state_encoder
from ..search.mcts import MCTS, SearchConfig
from ..values import standings_values, terminal_values
from .config import MODE_SPECS, RunConfig, load_config
from .replay import ReplayBuffer
from .sample import augment_many, finish_game_records, make_record

__all__ = ["generate_games", "pretrain", "main"]


def generate_games(cfg: RunConfig, games: int, sims: int, mode_name: str = "ind2",
                   seed: int = 0, max_plies: int = 400,
                   verbose: bool = True) -> np.ndarray:
    """Play ``games`` teacher games and return their (augmented) records."""
    n, mode, layout = MODE_SPECS[mode_name]
    evaluator = RolloutEvaluator("greedy")
    search_cfg = SearchConfig(sims=sims, noise=True, universes=2,
                              temperature_plies=cfg.selfplay.temperature_plies,
                              prune_policy_target=True)
    out: List[np.ndarray] = []
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    for game in range(games):
        state = E.new_game(n, mode, layout, rng=random.Random(seed * 1000 + game))
        if cfg.selfplay.win_threshold is not None:
            state.config = dict(state.config)
            state.config["winThreshold"] = int(cfg.selfplay.win_threshold)
        records: List[np.ndarray] = []
        stuck = [False] * 4
        plies = 0
        while state.phase == E.PHASE_PLAYING and plies < max_plies:
            seat = state.current_player
            if E.is_stuck(state):
                E.resign(state, seat)
                stuck[seat] = True
                plies += 1
                continue
            tree = MCTS(search_cfg, np.random.default_rng(rng.integers(1 << 62)))
            for _ in range(sims):
                leaf = tree.select_leaf(state, seat)
                if leaf is None:
                    continue
                obs = state_encoder(leaf.state, leaf.seat)
                priors, values = evaluator.evaluate([obs], leaf.mask[None])
                tree.backup(leaf.token, priors[0], values[0])
            result = tree.result()
            mask = np.asarray(E.legal_mask(state), dtype=bool)
            records.append(make_record(state, seat,
                                       np.asarray(result.policy_target, np.float32),
                                       mask, mode_name, game, plies,
                                       root_value=result.root_value))
            E.apply(state, int(result.action))
            plies += 1
        truncated = state.phase == E.PHASE_PLAYING
        z = standings_values(state) if truncated else terminal_values(state)
        weight = cfg.selfplay.truncation_z_weight if truncated else 1.0
        scores = [p.score for p in state.players] + [0] * (4 - n)
        finish_game_records(records, z, weight, scores, stuck, plies)
        out.append(augment_many(records, cfg.selfplay.augment_rotations))
        if verbose:
            print(f"[bootstrap] game {game + 1}/{games} plies={plies} "
                  f"records={len(records)} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return np.concatenate(out) if out else augment_many([], 1)


def pretrain(cfg: RunConfig, records: np.ndarray, steps: int,
             out_path: Optional[str] = None, log_every: int = 50) -> str:
    """Fit the network to teacher records and publish it."""
    from .learner import Learner

    buffer = ReplayBuffer(window_start=1, window_end=1,
                          max_samples=len(records) + 1)
    buffer.add(records)
    buffer.close_generation(0)
    learner = Learner(cfg)
    for step in range(steps):
        batch = buffer.batch(cfg.learner.batch, value_blend=0.0)
        metrics = learner.train_step(batch)
        if log_every and step % log_every == 0:
            print(f"[bootstrap] step {step}/{steps} "
                  f"loss={metrics['total']:.4f} policy={metrics['policy']:.4f} "
                  f"value={metrics['value_mse']:.4f} "
                  f"top1={metrics['policy_top1_agreement']:.3f}", flush=True)
    return learner.publish(out_path)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m splendor_ai.selfplay.bootstrap")
    p.add_argument("--config", default=None)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--sims", type=int, default=96)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--mode", default="ind2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="checkpoint path to write")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.set)
    cfg.make_dirs()
    records = generate_games(cfg, args.games, args.sims, args.mode, args.seed,
                             max_plies=cfg.selfplay.max_plies)
    print(f"[bootstrap] {len(records)} augmented records", flush=True)
    path = pretrain(cfg, records, args.steps, args.out or cfg.latest_weights)
    print(f"[bootstrap] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
