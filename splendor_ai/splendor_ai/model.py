"""Policy / value network — ``docs/AI_DESIGN.md`` §1.5.

One mode-conditioned network for 2p, 3p, 4p, 1v2 and 2v2.  The trunk is a
pre-LayerNorm residual MLP and there are four heads::

    obs[B, OBS_DIM] ─ LN ─ Linear(OBS_DIM, width)
                        ├─ blocks x [LN -> Linear -> GELU -> Linear] (+skip)
                        └─ LN ─┬─ policy  Linear(width, 65)  (+ -1e9 mask)
                               ├─ value   Linear(width, 4) -> tanh
                               ├─ score   Linear(width, 4)      (aux)
                               └─ stuck   Linear(width, 4)      (aux logits)

The legality mask is applied **inside** ``forward`` and additively (``-1e9``),
so every consumer — search, arena, the worker — sees logits that are already
safe to ``softmax``.  Defaults are ``width=768, blocks=10`` (~12.6M
parameters); the smoke config is ``width=128, blocks=2``.

Checkpoints carry ``obs_version`` and ``action_version``; loading a checkpoint
built against a different observation layout or action space raises
``RuntimeError`` naming the mismatch instead of silently producing garbage.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import OBS_DIM, OBS_VERSION
from .rules.actions import NUM_ACTIONS
from .values import MAX_SEATS

#: Bumped whenever the 65-way action space changes meaning (never, so far).
ACTION_VERSION = 1

#: Additive penalty put on illegal logits inside :meth:`SplendorNet.forward`.
MASK_FILL = -1e9


@dataclass
class NetConfig:
    """Architecture + loss weights.  Stored verbatim in every checkpoint."""

    width: int = 768
    blocks: int = 10
    obs_dim: int = OBS_DIM
    num_actions: int = NUM_ACTIONS
    value_seats: int = MAX_SEATS
    aux_score: bool = True
    aux_stuck: bool = True
    value_weight: float = 1.0
    score_weight: float = 0.15
    stuck_weight: float = 0.15

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise RuntimeError(
                f"checkpoint cfg has unknown NetConfig fields: {unknown}")
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SMOKE_CONFIG = NetConfig(width=128, blocks=2)


class ResidualBlock(nn.Module):
    """Pre-LN residual block: ``x + Linear(GELU(Linear(LN(x))))``."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class SplendorNet(nn.Module):
    """The trunk and the four heads of §1.5."""

    def __init__(self, cfg: Optional[NetConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or NetConfig()
        c = self.cfg
        self.input_norm = nn.LayerNorm(c.obs_dim)
        self.stem = nn.Linear(c.obs_dim, c.width)
        self.blocks = nn.ModuleList(
            [ResidualBlock(c.width) for _ in range(c.blocks)])
        self.out_norm = nn.LayerNorm(c.width)
        self.policy_head = nn.Linear(c.width, c.num_actions)
        self.value_head = nn.Linear(c.width, c.value_seats)
        self.score_head = (nn.Linear(c.width, c.value_seats)
                           if c.aux_score else None)
        self.stuck_head = (nn.Linear(c.width, c.value_seats)
                           if c.aux_stuck else None)

    def trunk(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.stem(self.input_norm(obs))
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)

    def forward(self, obs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """``obs[B, OBS_DIM]`` and a boolean ``mask[B, 65]`` of legal actions.

        The mask is applied additively so illegal actions come out at about
        ``-1e9`` and vanish under any softmax.  ``mask=None`` leaves the raw
        logits alone (for tests and for supervised warm starts).
        """
        if obs.dim() != 2:
            raise ValueError(f"obs must be [B, OBS_DIM], got {tuple(obs.shape)}")
        x = self.trunk(obs)
        logits = self.policy_head(x)
        if mask is not None:
            if mask.shape != logits.shape:
                raise ValueError(
                    f"mask {tuple(mask.shape)} does not match logits "
                    f"{tuple(logits.shape)}")
            logits = logits + torch.where(
                mask.bool(),
                logits.new_zeros(()),
                logits.new_full((), MASK_FILL))
        out = {"logits": logits, "value": torch.tanh(self.value_head(x))}
        if self.score_head is not None:
            out["score"] = self.score_head(x)
        if self.stuck_head is not None:
            out["stuck"] = self.stuck_head(x)
        return out


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters, for the model report and the config sanity check."""
    return sum(p.numel() for p in module.parameters()
               if p.requires_grad or not trainable_only)


# ── loss ──────────────────────────────────────────────────────────────────

def compute_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                 cfg: Optional[NetConfig] = None
                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """``CE(policy) + 1.0*masked MSE(value) + 0.15*MSE(score) + 0.15*BCE(stuck)``.

    ``batch`` holds ``policy_target[B, 65]`` (already zero on illegal actions
    and summing to one), ``z[B, 4]`` with ``z_valid[B, 4]`` (seats a game of
    this size actually has), an optional per-sample ``z_weight[B]`` (0.3 for a
    truncated game), and the aux targets ``score_target[B, 4]`` /
    ``stuck_target[B, 4]``.  Every term is masked with ``z_valid`` so padding
    seats never contribute a gradient.
    """
    cfg = cfg or NetConfig()
    logits = out["logits"]
    target = batch["policy_target"]
    logp = F.log_softmax(logits.float(), dim=-1)
    policy_loss = -(target * logp).sum(-1).mean()

    z = batch["z"]
    valid = batch["z_valid"]
    weight = valid
    z_weight = batch.get("z_weight")
    if z_weight is not None:
        weight = weight * z_weight.reshape(-1, 1)
    denom = weight.sum().clamp_min(1e-6)
    value = out["value"].float()
    value_loss = (((value - z) ** 2) * weight).sum() / denom

    parts = {"policy": policy_loss.detach(), "value": value_loss.detach()}
    total = policy_loss + cfg.value_weight * value_loss

    valid_denom = valid.sum().clamp_min(1e-6)
    if "score" in out and "score_target" in batch:
        score_loss = ((((out["score"].float() - batch["score_target"]) ** 2)
                       * valid).sum() / valid_denom)
        total = total + cfg.score_weight * score_loss
        parts["score"] = score_loss.detach()
    if "stuck" in out and "stuck_target" in batch:
        stuck_loss = (F.binary_cross_entropy_with_logits(
            out["stuck"].float(), batch["stuck_target"],
            reduction="none") * valid).sum() / valid_denom
        total = total + cfg.stuck_weight * stuck_loss
        parts["stuck"] = stuck_loss.detach()
    parts["total"] = total.detach()
    return total, parts


# ── checkpoints ───────────────────────────────────────────────────────────

def save_checkpoint(path: str, model: SplendorNet,
                    extra: Optional[Dict[str, Any]] = None) -> str:
    """Write ``{obs_version, action_version, cfg, state_dict, step,
    generation, meta}`` atomically (temp file + ``os.replace``) so a reader
    polling the file never sees a half-written checkpoint.

    ``extra`` may carry ``step`` / ``generation`` / ``meta``; any other key is
    folded into ``meta``.  Everything in ``meta`` must be plain data (numbers,
    strings, lists, dicts) — checkpoints are loaded with ``weights_only``.
    """
    extra = dict(extra or {})
    meta = dict(extra.pop("meta", {}) or {})
    step = extra.pop("step", 0)
    generation = extra.pop("generation", 0)
    meta.update(extra)
    payload = {
        "obs_version": OBS_VERSION,
        "action_version": ACTION_VERSION,
        "cfg": model.cfg.to_dict(),
        "state_dict": model.state_dict(),
        "step": step,
        "generation": generation,
        "meta": meta,
    }
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str, map_location: Any = "cpu"
                    ) -> Tuple[SplendorNet, Dict[str, Any]]:
    """Rebuild the network from a checkpoint.

    Raises ``RuntimeError`` naming the mismatch when the checkpoint was
    written against a different ``obs_version`` (the encoder layout) or
    ``action_version`` (the 65-way action space): an old checkpoint read with
    a new encoder would silently score garbage.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    obs_version = ckpt.get("obs_version")
    if obs_version != OBS_VERSION:
        raise RuntimeError(
            f"checkpoint {path!r} has obs_version {obs_version!r} but this "
            f"build encodes obs_version {OBS_VERSION} — the observation "
            f"layout changed, retrain or re-encode")
    action_version = ckpt.get("action_version")
    if action_version != ACTION_VERSION:
        raise RuntimeError(
            f"checkpoint {path!r} has action_version {action_version!r} but "
            f"this build uses action_version {ACTION_VERSION} — the action "
            f"space changed, retrain")
    cfg = NetConfig.from_dict(dict(ckpt["cfg"]))
    if cfg.obs_dim != OBS_DIM:
        raise RuntimeError(
            f"checkpoint {path!r} was built for obs_dim {cfg.obs_dim} but "
            f"this build encodes obs_dim {OBS_DIM}")
    model = SplendorNet(cfg)
    model.load_state_dict(ckpt["state_dict"])
    if map_location not in (None, "cpu"):
        model.to(map_location)
    return model, ckpt


# ── evaluator ─────────────────────────────────────────────────────────────

class NetEvaluator:
    """Batched ``Evaluator`` for the search (§1.6): ``evaluate(obs, mask)``.

    Batch only — the scheduler always has a batch of leaves — under
    ``torch.inference_mode``; ``bfloat16`` autocast on CUDA, fp32 on CPU.
    Priors come back normalised over the legal actions of each row (a row with
    no legal action at all returns all zeros, which the caller reads as a
    stuck seat).
    """

    def __init__(self, model: SplendorNet, device: Any = "cpu",
                 autocast_dtype: Optional[torch.dtype] = None) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.use_autocast = self.device.type == "cuda"
        self.autocast_dtype = autocast_dtype or torch.bfloat16

    @torch.inference_mode()
    def evaluate(self, obs: np.ndarray,
                 mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``obs[B, OBS_DIM] float32`` + ``mask[B, 65] bool`` ->
        ``(priors[B, 65] float32, values[B, 4] float32)``."""
        if obs.ndim != 2 or mask.ndim != 2:
            raise ValueError(
                f"evaluate() is batch only: got obs {obs.shape}, "
                f"mask {mask.shape}")
        if obs.shape[0] != mask.shape[0]:
            raise ValueError(
                f"{obs.shape[0]} observations but {mask.shape[0]} masks")
        obs_t = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32)
                                 ).to(self.device, non_blocking=True)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.bool_)
                                  ).to(self.device, non_blocking=True)
        if self.use_autocast:
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                out = self.model(obs_t, mask_t)
        else:
            out = self.model(obs_t, mask_t)
        priors = torch.softmax(out["logits"].float(), dim=-1)
        priors = priors * mask_t
        priors = priors / priors.sum(-1, keepdim=True).clamp_min(1e-12)
        values = out["value"].float()
        return (priors.to("cpu").numpy().astype(np.float32, copy=False),
                values.to("cpu").numpy().astype(np.float32, copy=False))


__all__ = ["NetConfig", "SMOKE_CONFIG", "SplendorNet", "ResidualBlock",
           "NetEvaluator", "compute_loss", "save_checkpoint",
           "load_checkpoint", "count_params", "ACTION_VERSION", "MASK_FILL"]
