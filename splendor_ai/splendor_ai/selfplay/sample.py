"""The training record and its numpy packing (``docs/AI_DESIGN.md`` §1.8).

One record is one *recorded* decision: the compact position, who was to move,
the sparse search policy target, the per-seat value vector and the auxiliary
targets.  The observation is **not** stored — the learner re-encodes with
:func:`splendor_ai.encode.encode_batch` on the fly, which is what makes it
possible to change the encoder without throwing the buffer away
(``docs/research/judges.md``, "REPLAY BUFFER").

Layout (``RECORD_DTYPE``, ~414 bytes packed)::

    state        u1[256] + nbytes u2   GameState.to_bytes()  (max seen: 196)
    mask         u1[9]                 np.packbits of the 65 legal flags
    policy_idx   u1[32]  policy_prob f2[32]  policy_n u1
    z            f2[4]   z_weight f2   root_value f2[4]
    score        f2[4]   stuck u1[4]
    seat u1, num_players u1, mode u1, rot u1
    generation i4, game_id i8, ply i2, plies i2

Conventions that the rest of the system depends on:

* ``z``, ``root_value``, ``score`` and ``stuck`` are in **ABSOLUTE seat order**
  (entries ``>= num_players`` are zero).  They are rotated to the acting seat
  exactly once, in :func:`splendor_ai.selfplay.replay.seat_relative_rows` (the
  vectorised :func:`splendor_ai.values.seat_relative`), on the way into a
  learner batch; nothing else in the pipeline may rotate them.
* the policy target sums to 1 and is zero on every illegal action — both are
  asserted on write by :func:`pack_policy` and re-checked in the learner.
* ``rot`` records which C5 colour rotation produced the record (0 = the game as
  played) purely for diagnostics; a rotated record is a first-class sample.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..rules.actions import NUM_ACTIONS
from ..rules.engine import GameState
from ..symmetry import NUM_ROTATIONS, action_perm, rotate_action, rotate_state

__all__ = [
    "STATE_BYTES", "POLICY_TOPK", "MASK_BYTES", "RECORD_DTYPE", "MODE_TAGS",
    "empty", "pack_policy", "densify_policy", "pack_mask", "unpack_mask",
    "make_record", "augment", "augment_many", "records_to_bytes",
    "records_from_bytes", "check_records",
]

#: Fixed slot for ``GameState.to_bytes()``.  The engine's worst case is ~206
#: bytes (all 90 card ids plus the header); 196 is the largest ever observed.
STATE_BYTES = 256
#: Sparse policy width.  A 600-sim search rarely spreads over more than ~30
#: actions after forced-playout pruning; anything past the top-K is folded back
#: in by renormalising, and :func:`pack_policy` records how much was dropped.
POLICY_TOPK = 32
MASK_BYTES = (NUM_ACTIONS + 7) // 8            # 9

#: Mixture keys, in the order stored in the ``mode`` byte.
MODE_TAGS: Tuple[str, ...] = ("ind2", "ind3", "ind4", "ovt", "team_adj",
                              "team_opp")
_MODE_INDEX = {name: i for i, name in enumerate(MODE_TAGS)}

RECORD_DTYPE = np.dtype([
    ("state", np.uint8, (STATE_BYTES,)),
    ("nbytes", np.uint16),
    ("mask", np.uint8, (MASK_BYTES,)),
    ("policy_idx", np.uint8, (POLICY_TOPK,)),
    ("policy_prob", np.float16, (POLICY_TOPK,)),
    ("policy_n", np.uint8),
    ("z", np.float16, (4,)),
    ("z_weight", np.float16),
    ("root_value", np.float16, (4,)),
    ("score", np.float16, (4,)),
    ("stuck", np.uint8, (4,)),
    ("seat", np.uint8),
    ("num_players", np.uint8),
    ("mode", np.uint8),
    ("rot", np.uint8),
    ("generation", np.int32),
    ("game_id", np.int64),
    ("ply", np.int16),
    ("plies", np.int16),
])

_ACTION_PERMS = np.stack([action_perm(k) for k in range(NUM_ROTATIONS)])
#: ``_ACTION_INV[k][j]`` = index of action ``j`` after rotating the state by k,
#: i.e. exactly :func:`splendor_ai.symmetry.rotate_action` in table form.
_ACTION_INV = np.stack([
    np.asarray([rotate_action(a, k) for a in range(NUM_ACTIONS)], dtype=np.int32)
    for k in range(NUM_ROTATIONS)
])


def mode_tag(name: str) -> int:
    try:
        return _MODE_INDEX[name]
    except KeyError:                                        # pragma: no cover
        raise KeyError(f"unknown mode tag {name!r}; known: {MODE_TAGS}")


def mode_name(tag: int) -> str:
    return MODE_TAGS[int(tag)]


def empty(n: int) -> np.ndarray:
    """``n`` zeroed records."""
    return np.zeros(n, dtype=RECORD_DTYPE)


# ── policy / mask packing ─────────────────────────────────────────────────

def pack_policy(policy: np.ndarray, mask: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Dense 65-vector -> ``(idx[uint8], prob[float16], n)``.

    Raises if the target does not sum to 1 or puts mass on an illegal action —
    a silently mis-normalised target is the kind of bug that only shows up as a
    plateau days later.
    """
    p = np.asarray(policy, dtype=np.float64).reshape(-1)
    if p.shape[0] != NUM_ACTIONS:
        raise ValueError(f"policy target must have {NUM_ACTIONS} entries, "
                         f"got {p.shape[0]}")
    if np.any(p < -1e-6):
        raise ValueError("policy target has negative mass")
    total = float(p.sum())
    if not abs(total - 1.0) < 1e-3:
        raise ValueError(f"policy target sums to {total!r}, expected 1")
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        if float(p[~m].sum()) > 1e-6:
            raise ValueError(
                "policy target puts mass on illegal actions: "
                f"{np.flatnonzero(p * ~m)[:8].tolist()}")
    nz = np.flatnonzero(p > 0)
    if nz.size > POLICY_TOPK:
        nz = nz[np.argsort(p[nz])[::-1][:POLICY_TOPK]]
        nz.sort()
    idx = np.zeros(POLICY_TOPK, dtype=np.uint8)
    prob = np.zeros(POLICY_TOPK, dtype=np.float16)
    n = int(nz.size)
    if n:
        kept = p[nz]
        kept = kept / kept.sum()
        idx[:n] = nz.astype(np.uint8)
        prob[:n] = kept.astype(np.float16)
    return idx, prob, n


def densify_policy(idx: np.ndarray, prob: np.ndarray, n,
                   out: Optional[np.ndarray] = None) -> np.ndarray:
    """Inverse of :func:`pack_policy` for one record or a whole batch.

    ``idx``/``prob`` may be ``[K]`` or ``[B, K]``; the result is ``[65]`` or
    ``[B, 65]`` float32 renormalised to exactly 1 (float16 storage loses a few
    ulps).
    """
    idx = np.asarray(idx)
    prob = np.asarray(prob, dtype=np.float32)
    if idx.ndim == 1:
        dense = np.zeros(NUM_ACTIONS, dtype=np.float32) if out is None else out
        dense[:] = 0.0
        k = int(n)
        if k:
            np.add.at(dense, idx[:k].astype(np.int64), prob[:k])
            s = dense.sum()
            if s > 0:
                dense /= s
        return dense
    b, _ = idx.shape
    dense = (np.zeros((b, NUM_ACTIONS), dtype=np.float32) if out is None else out)
    dense[:] = 0.0
    counts = np.asarray(n).reshape(-1)
    rows = np.repeat(np.arange(b), counts)
    cols = np.concatenate([idx[i, :counts[i]] for i in range(b)]).astype(np.int64) \
        if rows.size else np.zeros(0, dtype=np.int64)
    vals = np.concatenate([prob[i, :counts[i]] for i in range(b)]) \
        if rows.size else np.zeros(0, dtype=np.float32)
    if rows.size:
        np.add.at(dense, (rows, cols), vals)
    s = dense.sum(axis=1, keepdims=True)
    np.divide(dense, np.maximum(s, 1e-12), out=dense)
    return dense


def pack_mask(mask: Sequence[bool]) -> np.ndarray:
    return np.packbits(np.asarray(mask, dtype=bool))


def unpack_mask(packed: np.ndarray) -> np.ndarray:
    """``[9]`` or ``[B, 9]`` packed bits -> ``[65]`` / ``[B, 65]`` bool."""
    packed = np.asarray(packed, dtype=np.uint8)
    bits = np.unpackbits(packed, axis=-1)
    return bits[..., :NUM_ACTIONS].astype(bool)


# ── building ──────────────────────────────────────────────────────────────

def make_record(state: GameState, seat: int, policy: np.ndarray,
                mask: Sequence[bool], mode: str, game_id: int, ply: int,
                generation: int = 0, root_value=None) -> np.ndarray:
    """One record for a position whose outcome is not known yet.

    ``z``/``score``/``stuck``/``plies`` are filled in by
    :func:`finish_game_records` once the game ends.
    """
    raw = state.to_bytes()
    if len(raw) > STATE_BYTES:                              # pragma: no cover
        raise ValueError(f"state serialises to {len(raw)} bytes, "
                         f"STATE_BYTES is {STATE_BYTES}")
    rec = empty(1)
    rec["state"][0, :len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    rec["nbytes"][0] = len(raw)
    rec["mask"][0] = pack_mask(mask)
    idx, prob, n = pack_policy(policy, mask)
    rec["policy_idx"][0] = idx
    rec["policy_prob"][0] = prob
    rec["policy_n"][0] = n
    rec["seat"][0] = seat
    rec["num_players"][0] = state.num_players
    rec["mode"][0] = mode_tag(mode)
    rec["generation"][0] = generation
    rec["game_id"][0] = game_id
    rec["ply"][0] = ply
    rec["z_weight"][0] = 1.0
    if root_value is not None:
        rec["root_value"][0] = np.asarray(root_value, dtype=np.float16)[:4]
    return rec[0]


def finish_game_records(records: List[np.ndarray], z: np.ndarray,
                        z_weight: float, scores: Sequence[int],
                        stuck: Sequence[bool], plies: int) -> None:
    """Stamp the game outcome onto every record of that game (in place)."""
    z16 = np.asarray(z, dtype=np.float16)[:4]
    score16 = (np.asarray(scores, dtype=np.float32) / 20.0).astype(np.float16)
    stuck8 = np.asarray(stuck, dtype=np.uint8)
    for rec in records:
        rec["z"] = z16
        rec["z_weight"] = np.float16(z_weight)
        rec["score"][:len(score16)] = score16
        rec["stuck"][:len(stuck8)] = stuck8
        rec["plies"] = plies


# ── C5 colour augmentation (§1.4) ─────────────────────────────────────────

def augment(record: np.ndarray, k: int) -> np.ndarray:
    """A copy of ``record`` with every colour relabelled ``c -> (c+k)%5``.

    The state is re-hydrated, rotated with
    :func:`splendor_ai.symmetry.rotate_state` and re-serialised; the sparse
    policy indices and the packed legal mask move through
    :func:`splendor_ai.symmetry.rotate_action` / ``action_perm``.  Values,
    seats and aux targets are colour-blind and are copied unchanged.
    """
    k %= NUM_ROTATIONS
    out = record.copy()
    if k == 0:
        return out
    state = GameState.from_bytes(bytes(record["state"][:record["nbytes"]]))
    raw = rotate_state(state, k).to_bytes()
    out["state"][:] = 0
    out["state"][:len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    out["nbytes"] = len(raw)
    n = int(record["policy_n"])
    if n:
        idx = record["policy_idx"][:n].astype(np.int32)
        out["policy_idx"][:n] = _ACTION_INV[k][idx].astype(np.uint8)
    mask = unpack_mask(record["mask"])
    out["mask"][:] = pack_mask(mask[_ACTION_PERMS[k]])
    out["rot"] = k
    return out


def augment_many(records: Sequence[np.ndarray], rotations: int) -> np.ndarray:
    """``records`` plus ``rotations - 1`` extra colour rotations of each.

    ``rotations == 1`` is the identity (augmentation off).  The rotations used
    are ``0, 1, ... rotations-1`` of the C5 group.
    """
    if not len(records):
        return empty(0)
    base = np.array(records, dtype=RECORD_DTYPE)
    if rotations <= 1:
        return base
    chunks = [base]
    for k in range(1, min(rotations, NUM_ROTATIONS)):
        rotated = empty(len(base))
        for i in range(len(base)):
            rotated[i] = augment(base[i], k)
        chunks.append(rotated)
    return np.concatenate(chunks)


# ── transport ─────────────────────────────────────────────────────────────

def records_to_bytes(records: np.ndarray) -> bytes:
    """Raw buffer for a ``multiprocessing.Queue`` (no pickling of objects)."""
    return np.ascontiguousarray(records, dtype=RECORD_DTYPE).tobytes()


def records_from_bytes(buf) -> np.ndarray:
    return np.frombuffer(buf, dtype=RECORD_DTYPE)


# ── invariants ────────────────────────────────────────────────────────────

def check_records(records: np.ndarray, strict: bool = True) -> Dict[str, float]:
    """Assert the invariants the learner relies on; returns a small report.

    * the policy target sums to 1 (within float16 tolerance) and is zero on
      illegal actions,
    * the stored mask has at least one legal action,
    * ``z`` entries beyond ``num_players`` are zero.
    """
    n = len(records)
    if n == 0:
        return {"n": 0}
    dense = densify_policy(records["policy_idx"], records["policy_prob"],
                           records["policy_n"])
    mask = unpack_mask(records["mask"])
    sums = dense.sum(axis=1)
    leak = (dense * ~mask).sum(axis=1)
    empty_mask = ~mask.any(axis=1)
    seats = records["num_players"].astype(np.int64)
    pad = np.zeros(n, dtype=bool)
    for i in range(n):
        pad[i] = bool(np.any(np.asarray(records["z"][i])[seats[i]:] != 0))
    report = {
        "n": float(n),
        "policy_sum_err": float(np.max(np.abs(sums - 1.0))),
        "mask_leak": float(np.max(leak)),
        "empty_masks": float(empty_mask.sum()),
        "z_padding_nonzero": float(pad.sum()),
    }
    if strict:
        if report["policy_sum_err"] > 5e-3:
            raise AssertionError(f"policy targets do not sum to 1: {report}")
        if report["mask_leak"] > 1e-3:
            raise AssertionError(f"policy target leaks onto illegal actions: {report}")
        if report["empty_masks"]:
            raise AssertionError(f"record with an empty legal mask: {report}")
        if report["z_padding_nonzero"]:
            raise AssertionError(f"value vector has non-zero padding seats: {report}")
    return report
