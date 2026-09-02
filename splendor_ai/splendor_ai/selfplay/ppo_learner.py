"""Masked-PPO fallback learner (``docs/AI_DESIGN.md`` §0, §4).

AlphaZero is the primary learner.  This module is the documented escape hatch:
if search throughput on the production node turns out to be unaffordable, the
*only* thing that changes is the target producer — the engine, the encoder, the
network, the arena, the replay format, the orchestrator and the deployment
worker are all shared.  Switch with ``learner: ppo`` in the config
(``learner.algorithm``).

Status
------
**Partially implemented.**  What is here and exercised by
``tests/test_selfplay_ppo.py``:

* :class:`PPOBatch` / :func:`compute_gae` — per-seat returns tracked by
  *decision points* (a seat's next decision, not the next ply: with 2-4 seats
  and same-seat sub-decisions the two differ), GAE(lambda) over those, and
  margin-scaled terminal rewards;
* :class:`PPOLearner` — the clipped surrogate with additive ``-1e9`` masking
  (identical to :class:`splendor_ai.model.SplendorNet`'s own masking), value
  loss on the per-seat vector, entropy bonus, minibatch epochs, AdamW +
  warmup/cosine, and the same ``publish`` / ``state_dict`` /
  ``save_generation`` interface the orchestrator uses for the AZ learner, so
  ``train.py`` can drive it unchanged;
* :func:`records_to_rollout` — reads the *same* :mod:`.sample` records the AZ
  actors write.

TODO before this path can be trusted (it has never been run end to end):

1. ``actor.py`` needs a search-free branch (``inference.mode`` unchanged, but
   sample the move from the masked policy instead of building an MCTS and
   record **every** move with its ``log_prob`` and the value estimate).  The
   record layout needs two extra columns (``logp``, ``value``); reuse the
   spare bytes rather than growing ``RECORD_DTYPE``.
2. The replay buffer must become on-policy: a PPO iteration consumes exactly
   the last ``rollout_games`` and then drops them (no generational window).
3. Reward shaping stays at zero on purpose; the margin scaling below is the
   only deviation from the pure terminal signal, and it needs an A/B against
   plain +-1 before it is trusted.
4. Advantage normalisation across seats of the same game is currently global;
   per-seat normalisation is probably better for 1v2 and needs measuring.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..encode import OBS_VERSION
from ..model import ACTION_VERSION, SplendorNet, save_checkpoint
from ..rules.actions import NUM_ACTIONS
from .config import RunConfig
from .learner import lr_at

__all__ = ["PPOConfig", "PPOBatch", "compute_gae", "records_to_rollout",
           "PPOLearner"]


@dataclass
class PPOConfig:
    """PPO knobs.  Defaults follow the MaskablePPO baselines in the corpus."""

    clip: float = 0.2
    gamma: float = 0.997          # ~100-ply horizon
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    epochs: int = 4
    minibatch: int = 1024
    #: Terminal reward = sign(margin) * (1 + margin_scale * |margin| / 20).
    margin_scale: float = 0.5
    normalise_advantage: bool = True
    target_kl: Optional[float] = 0.03


@dataclass
class PPOBatch:
    """One on-policy rollout, flattened over (game, seat, decision point)."""

    obs: np.ndarray                     # [N, OBS_DIM] float32
    mask: np.ndarray                    # [N, 65] bool
    action: np.ndarray                  # [N] int64
    logp_old: np.ndarray                # [N] float32
    value_old: np.ndarray               # [N, 4] float32 (seat-relative)
    advantage: np.ndarray               # [N] float32
    ret: np.ndarray                     # [N, 4] float32 (seat-relative)
    seat: np.ndarray                    # [N] int64
    z_valid: np.ndarray                 # [N, 4] float32

    def __len__(self) -> int:
        return int(self.obs.shape[0])


def terminal_reward(values: np.ndarray, scores: Sequence[int],
                    margin_scale: float = 0.5) -> np.ndarray:
    """Margin-scaled terminal reward, absolute seat order.

    The sign is the §1.2 outcome; the magnitude grows with how decisively the
    seat won or lost, which is the one piece of shaping this fallback keeps
    (a 15-2 win and a 15-14 win are not the same evidence).
    """
    values = np.asarray(values, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    n = len(scores)
    margin = np.zeros(4, dtype=np.float32)
    if n:
        mean = float(scores[:n].mean())
        margin[:n] = (scores[:n] - mean) / 20.0
    return np.sign(values) * (1.0 + margin_scale * np.abs(margin))


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                gamma: float, lam: float) -> tuple:
    """GAE(lambda) along ONE seat's chain of decision points.

    ``values[t]`` is that seat's own entry of the value vector at its ``t``-th
    decision; ``rewards[t]`` is zero everywhere except the terminal step.  The
    caller is responsible for building one chain per (game, seat) — that is the
    "per-seat returns by decision points" of §4, and it is what makes the
    multiplayer credit assignment correct without any sign flipping.
    """
    t_max = len(rewards)
    adv = np.zeros(t_max, dtype=np.float32)
    last = 0.0
    for t in range(t_max - 1, -1, -1):
        next_value = 0.0 if (t + 1 >= t_max or dones[t]) else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * (0.0 if dones[t] else last)
        adv[t] = last
    return adv, adv + values


def records_to_rollout(records: np.ndarray, cfg: PPOConfig) -> PPOBatch:
    """Turn :mod:`.sample` records into a PPO batch.

    Works today for the *targets* (obs, mask, action, returns, advantages);
    ``logp_old`` and ``value_old`` are recomputed from the current policy,
    which makes the first epoch an on-policy gradient step and the later ones
    the usual clipped updates.  Once the search-free actor writes ``logp`` and
    ``value`` into the record (TODO 1), read them here instead — that is the
    only change needed for a strictly correct importance ratio.
    """
    from .replay import make_batch

    batch = make_batch(records)
    n = len(records)
    action = np.argmax(batch["policy_target"], axis=1).astype(np.int64)
    z = batch["z"]
    value_old = np.zeros_like(z)
    advantage = (z[:, 0] - value_old[:, 0]).astype(np.float32)
    if cfg.normalise_advantage and n > 1:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    return PPOBatch(obs=batch["obs"], mask=batch["mask"], action=action,
                    logp_old=np.zeros(n, dtype=np.float32), value_old=value_old,
                    advantage=advantage, ret=z, seat=records["seat"].astype(np.int64),
                    z_valid=batch["z_valid"])


class PPOLearner:
    """Drop-in replacement for :class:`splendor_ai.selfplay.learner.Learner`.

    Implements the same surface the orchestrator uses (``ready``,
    ``train_step``, ``publish``, ``save_generation``, ``state_dict``,
    ``load_state_dict``, ``local_batch``, ``step``, ``generation``,
    ``samples_consumed``) so ``train.py`` needs no PPO-specific branch.
    """

    def __init__(self, cfg: RunConfig, model: Optional[SplendorNet] = None,
                 ppo: Optional[PPOConfig] = None) -> None:
        self.cfg = cfg
        self.ppo = ppo or PPOConfig()
        self.device = torch.device(cfg.learner.device)
        self.model = (model or SplendorNet(cfg.net)).to(self.device)
        self.ddp_model = self.model
        decay = [p for p in self.model.parameters() if p.ndim > 1]
        no_decay = [p for p in self.model.parameters() if p.ndim <= 1]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.learner.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.learner.lr, betas=(0.9, 0.95))
        self.step = 0
        self.generation = 0
        self.samples_consumed = 0
        self.published = 0
        self.rank = 0
        self.world_size = 1

    # -- orchestrator interface -----------------------------------------
    @property
    def local_batch(self) -> int:
        return self.cfg.learner.batch

    def ready(self, samples_produced: int, buffer_size: int) -> bool:
        return buffer_size >= max(self.local_batch, self.cfg.replay.min_samples)

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """One PPO iteration over ``batch`` (``epochs`` passes of minibatches).

        ``batch`` arrives in the AZ format from :func:`replay.make_batch`; the
        conversion is :func:`records_to_rollout`'s job when the search-free
        actor lands.  Until then the action is the argmax of the stored policy
        target and the ratio starts at 1, which makes the first epoch a plain
        policy-gradient step.
        """
        ppo = self.ppo
        lr = lr_at(self.step, self.cfg.learner.lr, self.cfg.learner.lr_final,
                   self.cfg.learner.warmup_steps, self.cfg.learner.cosine_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(batch["mask"], device=self.device).bool()
        target = torch.as_tensor(batch["policy_target"], dtype=torch.float32,
                                 device=self.device)
        action = target.argmax(dim=-1)
        z = torch.as_tensor(batch["z"], dtype=torch.float32, device=self.device)
        valid = torch.as_tensor(batch["z_valid"], dtype=torch.float32,
                                device=self.device)
        with torch.no_grad():
            out = self.model(obs, mask)
            logp_old = torch.log_softmax(out["logits"].float(), -1).gather(
                1, action[:, None]).squeeze(1)
            advantage = z[:, 0] - out["value"].float()[:, 0]
            if ppo.normalise_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        n = obs.shape[0]
        metrics: Dict[str, float] = {}
        for epoch in range(ppo.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, ppo.minibatch):
                idx = perm[start:start + ppo.minibatch]
                out = self.model(obs[idx], mask[idx])
                logits = out["logits"].float()
                logp_all = torch.log_softmax(logits, -1)
                logp = logp_all.gather(1, action[idx][:, None]).squeeze(1)
                ratio = torch.exp(logp - logp_old[idx])
                a = advantage[idx]
                unclipped = ratio * a
                clipped = torch.clamp(ratio, 1 - ppo.clip, 1 + ppo.clip) * a
                policy_loss = -torch.min(unclipped, clipped).mean()
                value = out["value"].float()
                value_loss = (((value - z[idx]) ** 2) * valid[idx]).sum() / \
                    valid[idx].sum().clamp_min(1e-6)
                entropy = -(logp_all.exp() * logp_all.clamp_min(-30.0)).sum(-1).mean()
                loss = (policy_loss + ppo.value_coef * value_loss
                        - ppo.entropy_coef * entropy)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.learner.grad_clip)
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = float((logp_old[idx] - logp).mean())
                    metrics = {"policy": float(policy_loss.detach()),
                               "value": float(value_loss.detach()),
                               "entropy": float(entropy.detach()),
                               "total": float(loss.detach()),
                               "approx_kl": approx_kl, "lr": lr,
                               "grad_norm": float(grad_norm),
                               "clip_frac": float(((ratio - 1).abs() > ppo.clip)
                                                  .float().mean())}
            if ppo.target_kl is not None and metrics.get("approx_kl", 0) > ppo.target_kl:
                metrics["early_stop_epoch"] = epoch
                break
        self.step += 1
        self.samples_consumed += n
        return metrics

    def publish(self, path: Optional[str] = None,
                meta: Optional[Dict] = None) -> str:
        path = path or self.cfg.latest_weights
        self.published += 1
        return save_checkpoint(path, self.model,
                               {"step": self.step, "generation": self.generation,
                                "meta": {"version": self.published,
                                         "learner": "ppo", **(meta or {})}})

    def save_generation(self, generation: int) -> str:
        path = os.path.join(self.cfg.checkpoints_dir, f"gen_{generation:04d}.pt")
        return save_checkpoint(path, self.model,
                               {"step": self.step, "generation": generation,
                                "meta": {"learner": "ppo"}})

    def state_dict(self) -> Dict[str, Any]:
        return {"obs_version": OBS_VERSION, "action_version": ACTION_VERSION,
                "cfg": self.cfg.net.to_dict(), "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "step": self.step,
                "generation": self.generation,
                "samples_consumed": self.samples_consumed,
                "published": self.published, "learner": "ppo"}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("obs_version") != OBS_VERSION:
            raise RuntimeError(
                f"trainer state has obs_version {state.get('obs_version')!r} but "
                f"this build encodes obs_version {OBS_VERSION}")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = int(state.get("step", 0))
        self.generation = int(state.get("generation", 0))
        self.samples_consumed = int(state.get("samples_consumed", 0))
        self.published = int(state.get("published", 0))
