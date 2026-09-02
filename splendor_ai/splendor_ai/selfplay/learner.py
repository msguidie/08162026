"""AlphaZero learner (``docs/AI_DESIGN.md`` §1.8, judges.md "LEARNER").

AdamW, 2k-step linear warmup then cosine to ``lr_final``, weight decay on the
matrices only, grad clip 1.0, bf16 autocast on CUDA, and
:func:`splendor_ai.model.compute_loss` for the four heads.  Two things beyond
the obvious:

* **Replay-ratio throttle.**  Self-play produces samples at its own pace; the
  learner refuses to step whenever it would push sample reuse past
  ``learner.replay_ratio``.  On this node the learner is idle >90% of the time
  and that is the intended operating point — spare learner capacity is not a
  reason to inflate the net (judges.md).
* **Two kinds of checkpoint.**  ``weights/latest.pt`` is the *published*
  artefact the actors and inference servers poll (written with
  :func:`splendor_ai.model.save_checkpoint`, i.e. temp file + atomic rename,
  carrying ``obs_version``/``action_version``); ``trainer_state.pt`` is the
  full resumable state (model, optimizer, schedule, counters, RNG) and
  ``checkpoints/gen_XXXX.pt`` is the per-generation history the opponent pool
  and the final arena draw from.

DDP: when ``WORLD_SIZE > 1`` the model is wrapped in ``DistributedDataParallel``
and each rank consumes ``batch // world_size`` samples, so the effective batch
is unchanged.  Untested in this sandbox (single CPU box, no NCCL); the design
doc's position is that a ~13M-parameter MLP does not need it below ~100M
parameters.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..encode import OBS_VERSION
from ..model import (ACTION_VERSION, SplendorNet, compute_loss, load_checkpoint,
                     save_checkpoint)
from .config import RunConfig

__all__ = ["Learner", "lr_at"]


def lr_at(step: int, lr: float, lr_final: float, warmup: int,
          cosine_steps: int) -> float:
    """Linear warmup then cosine decay, clamped at ``lr_final``."""
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / float(warmup)
    if cosine_steps <= 0:
        return lr
    progress = min(1.0, (step - warmup) / float(max(1, cosine_steps - warmup)))
    return lr_final + 0.5 * (lr - lr_final) * (1.0 + math.cos(math.pi * progress))


class Learner:
    """Owns the network, the optimizer and everything that persists."""

    def __init__(self, cfg: RunConfig, model: Optional[SplendorNet] = None) -> None:
        self.cfg = cfg
        lcfg = cfg.learner
        self.device = torch.device(lcfg.device)
        self.model = (model or SplendorNet(cfg.net)).to(self.device)
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.ddp_model = self._maybe_ddp()
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if param.ndim <= 1 else decay).append(param)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": lcfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lcfg.lr, betas=(0.9, 0.95), eps=1e-8)
        self.step = 0
        self.generation = 0
        self.samples_consumed = 0
        self.published = 0
        self.last_checkpoint_t = time.monotonic()
        self.use_autocast = (self.device.type == "cuda" and lcfg.bf16)

    # -- DDP -------------------------------------------------------------
    def _maybe_ddp(self):
        if self.world_size <= 1 or not self.cfg.learner.ddp:
            return self.model
        import torch.distributed as dist                    # pragma: no cover

        if not dist.is_initialized():
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            dist.init_process_group(backend=backend)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        from torch.nn.parallel import DistributedDataParallel

        return DistributedDataParallel(
            self.model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None)

    @property
    def local_batch(self) -> int:
        """Per-rank batch; ``batch`` stays the global (effective) batch."""
        return max(1, self.cfg.learner.batch // max(1, self.world_size))

    # -- throttle --------------------------------------------------------
    def ready(self, samples_produced: int, buffer_size: int) -> bool:
        """Replay-ratio + minimum-buffer gate."""
        if buffer_size < max(self.local_batch, self.cfg.replay.min_samples):
            return False
        ratio = self.cfg.learner.replay_ratio
        if ratio <= 0:
            return True
        budget = ratio * max(1, samples_produced)
        return (self.samples_consumed + self.cfg.learner.batch) <= budget

    # -- one optimizer step ----------------------------------------------
    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        cfg = self.cfg.learner
        lr = lr_at(self.step, cfg.lr, cfg.lr_final, cfg.warmup_steps,
                   cfg.cosine_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        tensors = self._to_torch(batch)
        obs, mask = tensors["obs"], tensors["mask"]
        if not bool(mask.any(dim=1).all()):
            raise AssertionError("learner: batch row with no legal action")
        target = tensors["policy_target"]
        leak = float((target * (~mask)).sum())
        if leak > 1e-3:
            raise AssertionError(
                f"learner: policy target puts {leak:.4g} mass on illegal actions")
        sums = target.sum(dim=-1)
        if not bool(((sums - 1.0).abs() < 5e-3).all()):
            raise AssertionError("learner: policy targets do not sum to 1")

        model = self.ddp_model
        model.train()
        if self.use_autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(obs, mask)
                loss, parts = compute_loss(out, tensors, self.cfg.net)
        else:
            out = model(obs, mask)
            loss, parts = compute_loss(out, tensors, self.cfg.net)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   cfg.grad_clip)
        self.optimizer.step()
        self.step += 1
        self.samples_consumed += self.cfg.learner.batch

        with torch.no_grad():
            metrics = {k: float(v) for k, v in parts.items()}
            metrics.update(self._diagnostics(out, tensors))
            metrics["lr"] = lr
            metrics["grad_norm"] = float(grad_norm)
        return metrics

    def _to_torch(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if key == "mask":
                out[key] = torch.from_numpy(np.ascontiguousarray(value)).to(
                    self.device, non_blocking=True)
            else:
                out[key] = torch.from_numpy(
                    np.ascontiguousarray(value, dtype=np.float32)).to(
                        self.device, non_blocking=True)
        return out

    @staticmethod
    def _diagnostics(out: Dict[str, torch.Tensor],
                     batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Value MSE / explained variance, policy entropy and top-1 agreement."""
        value = out["value"].float()
        z = batch["z"]
        valid = batch["z_valid"]
        weight = valid
        if "z_weight" in batch:
            weight = weight * batch["z_weight"].reshape(-1, 1)
        denom = weight.sum().clamp_min(1e-6)
        mse = float((((value - z) ** 2) * weight).sum() / denom)
        mean_z = (z * weight).sum() / denom
        var_z = float((((z - mean_z) ** 2) * weight).sum() / denom)
        logits = out["logits"].float()
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        entropy = float(-(p * logp.clamp_min(-30.0)).sum(-1).mean())
        target = batch["policy_target"]
        agree = float((logits.argmax(-1) == target.argmax(-1)).float().mean())
        target_entropy = float(
            -(target * torch.log(target.clamp_min(1e-9))).sum(-1).mean())
        return {
            "value_mse": mse,
            "value_explained_variance": (1.0 - mse / var_z) if var_z > 1e-9 else 0.0,
            "policy_entropy": entropy,
            "target_entropy": target_entropy,
            "policy_top1_agreement": agree,
        }

    # -- publishing / checkpointing --------------------------------------
    def publish(self, path: Optional[str] = None, meta: Optional[Dict] = None) -> str:
        """Atomically write ``weights/latest.pt`` for actors and servers."""
        if self.rank != 0:
            return ""
        path = path or self.cfg.latest_weights
        self.published += 1
        payload = {"step": self.step, "generation": self.generation,
                   "meta": {"version": self.published,
                            "samples": self.samples_consumed,
                            "published_at": time.time(),
                            **(meta or {})}}
        return save_checkpoint(path, self.model, payload)

    def save_generation(self, generation: int) -> str:
        """``checkpoints/gen_XXXX.pt`` — the opponent pool and arena history."""
        if self.rank != 0:
            return ""
        path = os.path.join(self.cfg.checkpoints_dir, f"gen_{generation:04d}.pt")
        return save_checkpoint(path, self.model,
                               {"step": self.step, "generation": generation,
                                "meta": {"version": self.published}})

    def state_dict(self) -> Dict[str, Any]:
        return {
            "obs_version": OBS_VERSION,
            "action_version": ACTION_VERSION,
            "cfg": self.cfg.net.to_dict(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "generation": self.generation,
            "samples_consumed": self.samples_consumed,
            "published": self.published,
            "torch_rng": torch.get_rng_state(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("obs_version") != OBS_VERSION:
            raise RuntimeError(
                f"trainer state has obs_version {state.get('obs_version')!r} but "
                f"this build encodes obs_version {OBS_VERSION}")
        if state.get("action_version") != ACTION_VERSION:
            raise RuntimeError(
                f"trainer state has action_version {state.get('action_version')!r} "
                f"but this build uses action_version {ACTION_VERSION}")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = int(state.get("step", 0))
        self.generation = int(state.get("generation", 0))
        self.samples_consumed = int(state.get("samples_consumed", 0))
        self.published = int(state.get("published", 0))
        rng = state.get("torch_rng")
        if rng is not None:
            torch.set_rng_state(rng.cpu() if hasattr(rng, "cpu") else rng)

    def warm_start(self, path: str) -> None:
        """Initialise from an existing checkpoint (e.g. :mod:`.bootstrap`)."""
        model, _ckpt = load_checkpoint(path, map_location=str(self.device))
        self.model.load_state_dict(model.state_dict())
