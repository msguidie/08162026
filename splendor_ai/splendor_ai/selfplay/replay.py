"""Rolling generational replay buffer (``docs/AI_DESIGN.md`` §1.8).

Samples are the compact records of :mod:`.sample`; the observation is built
here, on the way into a learner batch, by re-hydrating the position and calling
:func:`splendor_ai.encode.encode_batch` (~19 us/state batched against ~85 us
one at a time).

The window is trimmed **by whole generations**, never by individual samples
(judges.md: "Rolling window over the last N generations, trimmed by generation
... ramping 4 -> 20"), and ramps linearly:

    window(g) = window_start + (window_end - window_start) * min(1, g / ramp)

A second, harder cap (``max_samples``) drops whole generations from the front
if the window would not fit in RAM.

``save``/``load`` round-trip the whole buffer through a single ``.npz`` so a
PBS job that restarts resumes with its data (and its generation boundaries)
intact.  This module is numpy-only on purpose: the tests, and the actors, never
have to import torch to touch it.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..encode import OBS_DIM, encode_batch
from ..rules.actions import NUM_ACTIONS
from ..rules.engine import GameState
from .sample import (RECORD_DTYPE, densify_policy, empty, records_from_bytes,
                     unpack_mask)

__all__ = ["ReplayBuffer", "make_batch", "seat_relative_rows"]


def seat_relative_rows(values: np.ndarray, seats: np.ndarray,
                       num_players: np.ndarray) -> np.ndarray:
    """Rotate ``[B, 4]`` absolute-seat vectors so column 0 is the acting seat.

    The mod-``n`` roll of :func:`splendor_ai.values.seat_relative`, vectorised:
    ``out[i, j] = values[i, (j + seat_i) % n_i]`` for ``j < n_i`` and 0 beyond,
    so a real value never lands in a padding slot.
    """
    b = values.shape[0]
    cols = np.arange(4)[None, :]
    n = num_players.reshape(-1, 1).astype(np.int64)
    src = (cols + seats.reshape(-1, 1).astype(np.int64)) % np.maximum(n, 1)
    out = np.take_along_axis(values.astype(np.float32), src, axis=1)
    out[cols >= n] = 0.0
    return out


def make_batch(records: np.ndarray, value_blend: float = 0.0,
               obs_out: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Decode records into the arrays :func:`splendor_ai.model.compute_loss`
    wants (plus ``obs``/``mask`` for the forward pass).

    Everything stored in absolute seat order (``z``, ``root_value``, ``score``,
    ``stuck``) is rotated to the acting seat here — the one place in the
    pipeline that is allowed to do it.
    """
    b = len(records)
    states = [GameState.from_bytes(bytes(records["state"][i][:records["nbytes"][i]]))
              for i in range(b)]
    seats = records["seat"].astype(np.int64)
    obs = encode_batch(states, seats.tolist(), out=obs_out)
    mask = unpack_mask(records["mask"])
    if not mask.any(axis=1).all():                          # pragma: no cover
        raise AssertionError("replay batch contains a position with no legal action")
    policy = densify_policy(records["policy_idx"], records["policy_prob"],
                            records["policy_n"])
    n = records["num_players"].astype(np.int64)
    z = seat_relative_rows(np.asarray(records["z"], dtype=np.float32), seats, n)
    if value_blend > 0.0:
        root = seat_relative_rows(
            np.asarray(records["root_value"], dtype=np.float32), seats, n)
        z = (1.0 - value_blend) * z + value_blend * root
    score = seat_relative_rows(np.asarray(records["score"], dtype=np.float32),
                               seats, n)
    stuck = seat_relative_rows(np.asarray(records["stuck"], dtype=np.float32),
                               seats, n)
    valid = (np.arange(4)[None, :] < n.reshape(-1, 1)).astype(np.float32)
    return {
        "obs": obs,
        "mask": mask,
        "policy_target": policy,
        "z": z.astype(np.float32),
        "z_valid": valid,
        "z_weight": np.asarray(records["z_weight"], dtype=np.float32),
        "score_target": score.astype(np.float32),
        "stuck_target": stuck.astype(np.float32),
    }


class ReplayBuffer:
    """Generational window with uniform sampling."""

    def __init__(self, window_start: int = 4, window_end: int = 20,
                 window_ramp_generations: int = 40,
                 max_samples: int = 20_000_000,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.window_start = int(window_start)
        self.window_end = int(window_end)
        self.window_ramp = max(1, int(window_ramp_generations))
        self.max_samples = int(max_samples)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        #: sealed generations, oldest first: ``(generation_id, records)``
        self.generations: List[Tuple[int, np.ndarray]] = []
        self._pending: List[np.ndarray] = []
        self._pending_n = 0
        self.generation = 0
        #: lifetime counters (kept across trims, restored from a checkpoint)
        self.total_added = 0
        self.total_dropped = 0

    # -- writing ---------------------------------------------------------
    def add(self, records) -> int:
        """Append records (an array or a raw buffer) to the open generation."""
        if isinstance(records, (bytes, bytearray, memoryview)):
            records = records_from_bytes(records)
        records = np.asarray(records, dtype=RECORD_DTYPE)
        if not len(records):
            return 0
        self._pending.append(records)
        self._pending_n += len(records)
        self.total_added += len(records)
        return len(records)

    def close_generation(self, generation: Optional[int] = None) -> int:
        """Seal the open generation and trim the window.  Returns its size."""
        gen = self.generation if generation is None else int(generation)
        if self._pending:
            block = (self._pending[0] if len(self._pending) == 1
                     else np.concatenate(self._pending))
            self.generations.append((gen, block))
        else:
            self.generations.append((gen, empty(0)))
        size = self._pending_n
        self._pending = []
        self._pending_n = 0
        self.generation = gen + 1
        self.trim()
        return size

    def window_size(self, generation: Optional[int] = None) -> int:
        """Generations kept at ``generation`` (ramped ``start -> end``)."""
        g = self.generation if generation is None else generation
        frac = min(1.0, max(0.0, g / float(self.window_ramp)))
        span = self.window_end - self.window_start
        return int(round(self.window_start + span * frac))

    def trim(self) -> int:
        """Drop whole generations that fall outside the window or the cap."""
        dropped = 0
        keep = max(1, self.window_size())
        while len(self.generations) > keep:
            dropped += len(self.generations.pop(0)[1])
        while (len(self.generations) > 1
               and self.sealed_size() > self.max_samples):
            dropped += len(self.generations.pop(0)[1])
        self.total_dropped += dropped
        return dropped

    # -- reading ---------------------------------------------------------
    def sealed_size(self) -> int:
        return int(sum(len(block) for _, block in self.generations))

    def __len__(self) -> int:
        return self.sealed_size() + self._pending_n

    @property
    def size(self) -> int:
        return len(self)

    def _blocks(self) -> List[np.ndarray]:
        blocks = [b for _, b in self.generations if len(b)]
        blocks += [b for b in self._pending if len(b)]
        return blocks

    def sample(self, batch: int) -> np.ndarray:
        """Uniform sample of ``batch`` records over the whole window.

        Uniform on purpose (judges.md: "uncertainty-agnostic uniform
        sampling"); the open generation is included so a fresh run can start
        learning before the first generation closes.
        """
        blocks = self._blocks()
        if not blocks:
            raise ValueError("replay buffer is empty")
        sizes = np.array([len(b) for b in blocks], dtype=np.int64)
        total = int(sizes.sum())
        flat = self.rng.integers(0, total, size=batch)
        edges = np.concatenate([[0], np.cumsum(sizes)])
        which = np.searchsorted(edges, flat, side="right") - 1
        out = empty(batch)
        for bi in np.unique(which):
            rows = flat[which == bi] - edges[bi]
            out[which == bi] = blocks[bi][rows]
        return out

    def batch(self, batch: int, value_blend: float = 0.0) -> Dict[str, np.ndarray]:
        return make_batch(self.sample(batch), value_blend=value_blend)

    # -- persistence -----------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "total_added": self.total_added,
            "total_dropped": self.total_dropped,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "window_ramp": self.window_ramp,
            "max_samples": self.max_samples,
        }

    def save(self, path: str) -> str:
        """Write the whole buffer to ``path`` (npz, atomic rename)."""
        blocks = []
        ids = []
        sizes = []
        for gen, block in self.generations:
            blocks.append(block)
            ids.append(gen)
            sizes.append(len(block))
        if self._pending:
            blocks.append(np.concatenate(self._pending))
            ids.append(self.generation)
            sizes.append(self._pending_n)
        data = (np.concatenate(blocks) if blocks else empty(0))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"
        meta = self.state_dict()
        meta["open_generation"] = bool(self._pending)
        np.savez(tmp,
                 records=data,
                 gen_ids=np.array(ids, dtype=np.int64),
                 gen_sizes=np.array(sizes, dtype=np.int64),
                 meta=np.array([repr(meta)], dtype=object),
                 **{f"meta_{k}": np.array(v) for k, v in meta.items()})
        os.replace(tmp + ".npz", path)
        return path

    def load(self, path: str) -> "ReplayBuffer":
        with np.load(path, allow_pickle=True) as fh:
            data = fh["records"]
            ids = fh["gen_ids"]
            sizes = fh["gen_sizes"]
            get = lambda k, d: (fh[f"meta_{k}"].item() if f"meta_{k}" in fh else d)
            self.generation = int(get("generation", 0))
            self.total_added = int(get("total_added", 0))
            self.total_dropped = int(get("total_dropped", 0))
            open_gen = bool(get("open_generation", False))
        self.generations = []
        self._pending = []
        self._pending_n = 0
        offset = 0
        for i, (gen, size) in enumerate(zip(ids.tolist(), sizes.tolist())):
            block = data[offset:offset + size]
            offset += size
            if open_gen and i == len(ids) - 1:
                if size:
                    self._pending = [block]
                    self._pending_n = size
            else:
                self.generations.append((int(gen), block))
        return self
