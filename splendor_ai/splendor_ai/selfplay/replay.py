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
intact.  Two things about the save that matter at production size (~30M
records, ~12 GB):

* it writes **one array per generation** instead of concatenating the window
  into a single array first — the concatenate alone was a second full copy of
  the buffer (2x RAM) and most of the ~41 s the save cost;
* it can run on a **background thread** (:meth:`ReplayBuffer.save_async`) so
  the main loop, which is also the learner, does not stall on it.  The
  snapshot it hands the thread is a list of the *existing* generation arrays;
  nothing in this class ever mutates an array in place, so the writer sees a
  consistent buffer even as the run keeps adding to it.  The rename is atomic
  and happens only after the temp file is fully written.

Thread safety: every mutation and every read of the block lists takes
``self._lock`` (uncontended, ~50 ns).  It is what makes the learner's
background batch prefetch safe against a generation closing underneath it.
This module is numpy-only on purpose: the tests, and the actors, never have to
import torch to touch it.
"""

from __future__ import annotations

import os
import threading
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
        #: shipments held before the open generation is compacted in place
        self.compact_every = 256
        #: sealed generations, oldest first: ``(generation_id, records)``
        self.generations: List[Tuple[int, np.ndarray]] = []
        self._pending: List[np.ndarray] = []
        self._pending_n = 0
        self.generation = 0
        #: lifetime counters (kept across trims, restored from a checkpoint)
        self.total_added = 0
        self.total_dropped = 0
        #: generations dropped by ``max_samples`` rather than by the window --
        #: i.e. how often the RAM cap, not the design, chose the window
        self.cap_drops = 0
        self.cap_dropped_samples = 0
        self._lock = threading.RLock()
        self._saver: Optional[threading.Thread] = None
        self._save_error: Optional[BaseException] = None

    # -- writing ---------------------------------------------------------
    def add(self, records) -> int:
        """Append records (an array or a raw buffer) to the open generation."""
        if isinstance(records, (bytes, bytearray, memoryview)):
            records = records_from_bytes(records)
        records = np.asarray(records, dtype=RECORD_DTYPE)
        if not len(records):
            return 0
        with self._lock:
            self._pending.append(records)
            self._pending_n += len(records)
            self.total_added += len(records)
            # Keep the number of blocks bounded: a 20k-game generation arrives
            # as thousands of small shipments and ``sample`` touches every
            # block it draws from, so compact them as they come in.  Rebinding
            # the list (never mutating a block) is what keeps a concurrent
            # reader -- the batch prefetcher, the background saver -- safe.
            if len(self._pending) > self.compact_every:
                self._pending = [np.concatenate(self._pending)]
        return len(records)

    def close_generation(self, generation: Optional[int] = None) -> int:
        """Seal the open generation and trim the window.  Returns its size."""
        with self._lock:
            gen = self.generation if generation is None else int(generation)
            if self._pending:
                block = (self._pending[0] if len(self._pending) == 1
                         else np.concatenate(self._pending))
                self.generations = self.generations + [(gen, block)]
            else:
                self.generations = self.generations + [(gen, empty(0))]
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
        """Drop whole generations that fall outside the window or the cap.

        The two reasons are counted separately: dropping because the ramped
        window moved on is the design; dropping because ``max_samples`` was
        reached means the RAM cap silently shortened the window the run
        believes it is training on, which the trainer warns about once.
        """
        with self._lock:
            dropped = 0
            keep = max(1, self.window_size())
            gens = list(self.generations)
            while len(gens) > keep:
                dropped += len(gens.pop(0)[1])
            cap_dropped = 0
            while (len(gens) > 1
                   and sum(len(b) for _, b in gens) > self.max_samples):
                block = gens.pop(0)[1]
                cap_dropped += len(block)
                self.cap_drops += 1
            self.generations = gens
            self.cap_dropped_samples += cap_dropped
            self.total_dropped += dropped + cap_dropped
            return dropped + cap_dropped

    def retained_generations(self) -> int:
        """Sealed generations actually held -- the *true* window, which is
        ``window_size()`` unless ``max_samples`` cut it shorter."""
        return len(self.generations)

    def window_truncated(self) -> bool:
        """True when ``max_samples``, not the ramp, is choosing the window."""
        return self.retained_generations() < min(self.window_size(),
                                                 self.generation)

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
        with self._lock:
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

    def batch(self, batch: int, value_blend: float = 0.0,
              obs_out: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """One training batch.  ``obs_out`` is a caller-owned observation
        buffer (the learner's prefetcher rotates over three of them) so a
        4096x1094 float32 array is not reallocated on every step."""
        return make_batch(self.sample(batch), value_blend=value_blend,
                          obs_out=obs_out)

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

    def _snapshot(self) -> Dict[str, Any]:
        """A consistent, copy-free view of the buffer for the writer.

        Only *references* to the existing blocks are taken.  Nothing in this
        class mutates a block in place (``add`` appends to a list, ``trim``
        rebinds it), so the writer can take as long as it likes while the run
        carries on filling the next generation.
        """
        with self._lock:
            blocks = [(int(gen), block) for gen, block in self.generations]
            pending = list(self._pending)
            pending_n = self._pending_n
            meta = self.state_dict()
            meta["open_generation"] = bool(pending)
            return {"blocks": blocks, "pending": pending,
                    "pending_n": pending_n, "meta": meta}

    @staticmethod
    def _write_snapshot(path: str, snap: Dict[str, Any]) -> str:
        """Write one ``.npz`` per generation array, then rename atomically.

        No ``np.concatenate`` over the whole window: at the production size
        that single call is a second full copy of the buffer (2x RAM) and most
        of the ~41 s the save used to cost.  ``np.savez`` streams each array
        into the archive one at a time instead.
        """
        arrays: Dict[str, np.ndarray] = {}
        ids: List[int] = []
        sizes: List[int] = []
        for i, (gen, block) in enumerate(snap["blocks"]):
            arrays[f"gen_{i}"] = block
            ids.append(int(gen))
            sizes.append(len(block))
        pending = snap["pending"]
        if pending:
            block = (pending[0] if len(pending) == 1 else np.concatenate(pending))
            arrays[f"gen_{len(ids)}"] = block
            ids.append(int(snap["meta"]["generation"]))
            sizes.append(len(block))
        meta = snap["meta"]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}.{threading.get_ident()}"
        np.savez(tmp,
                 gen_ids=np.array(ids, dtype=np.int64),
                 gen_sizes=np.array(sizes, dtype=np.int64),
                 blocks=np.array([len(ids)], dtype=np.int64),
                 meta=np.array([repr(meta)], dtype=object),
                 **{f"meta_{k}": np.array(v) for k, v in meta.items()},
                 **arrays)
        # The rename happens only here, after np.savez has closed the file:
        # a reader (a resume) either sees the previous buffer or this one,
        # never a half-written archive.
        os.replace(tmp + ".npz", path)
        return path

    def save(self, path: str) -> str:
        """Write the whole buffer to ``path`` (npz, atomic rename)."""
        self.join_save()
        return self._write_snapshot(path, self._snapshot())

    def save_async(self, path: str) -> bool:
        """Start a background save.  Returns False if one is still running.

        Skipping rather than queueing is deliberate: the caller checkpoints on
        a timer, and a save that has not finished by the next tick means the
        buffer is bigger than the disk is fast — writing twice as often would
        only make that worse.
        """
        with self._lock:
            if self._saver is not None and self._saver.is_alive():
                return False
            snap = self._snapshot()

            def _run() -> None:
                try:
                    self._write_snapshot(path, snap)
                except BaseException as exc:                # pragma: no cover
                    self._save_error = exc
                    print(f"[replay] background save to {path} failed: "
                          f"{type(exc).__name__}: {exc}", flush=True)

            self._saver = threading.Thread(target=_run, name="replay-save",
                                           daemon=True)
            self._saver.start()
            return True

    def join_save(self, timeout: Optional[float] = None) -> bool:
        """Wait for a background save.  Returns True when none is running."""
        saver = self._saver
        if saver is None:
            return True
        saver.join(timeout)
        return not saver.is_alive()

    def saving(self) -> bool:
        return self._saver is not None and self._saver.is_alive()

    def load(self, path: str) -> "ReplayBuffer":
        """Read a buffer written by :meth:`save` (either layout).

        ``gen_<i>`` arrays are the current, per-generation layout; a single
        concatenated ``records`` array is what older runs wrote, and is still
        read so a checkpoint from before this change resumes.
        """
        with np.load(path, allow_pickle=True) as fh:
            ids = fh["gen_ids"]
            sizes = fh["gen_sizes"]
            get = lambda k, d: (fh[f"meta_{k}"].item() if f"meta_{k}" in fh else d)
            self.generation = int(get("generation", 0))
            self.total_added = int(get("total_added", 0))
            self.total_dropped = int(get("total_dropped", 0))
            open_gen = bool(get("open_generation", False))
            legacy = fh["records"] if "records" in fh else None
            blocks: List[np.ndarray] = []
            offset = 0
            for i, size in enumerate(sizes.tolist()):
                if legacy is not None:
                    blocks.append(legacy[offset:offset + size])
                    offset += size
                else:
                    blocks.append(np.asarray(fh[f"gen_{i}"]))
        with self._lock:
            self.generations = []
            self._pending = []
            self._pending_n = 0
            for i, (gen, block) in enumerate(zip(ids.tolist(), blocks)):
                if open_gen and i == len(blocks) - 1:
                    if len(block):
                        self._pending = [block]
                        self._pending_n = len(block)
                else:
                    self.generations.append((int(gen), block))
        return self
