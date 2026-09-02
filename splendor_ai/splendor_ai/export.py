"""Package a trained checkpoint for the deployment worker — §1.9, §2 (G6).

The worker on the Windows box reads ``MODEL_DIR/<mode>.pt`` first and
``MODEL_DIR/shared.pt`` as the fallback (``worker/config.py::
checkpoint_candidates``), and loads it with :func:`splendor_ai.model
.load_checkpoint`, which **refuses** a checkpoint whose ``obs_version`` or
``action_version`` does not match the build.  So the bundle this module writes
is exactly a ``save_checkpoint`` payload plus provenance:

``shared.pt``
    the model, re-saved through :func:`~.model.save_checkpoint` so the file is
    guaranteed to carry the running build's versions and a plain-data ``meta``
    (checkpoints are read with ``weights_only=True``; anything exotic in
    ``meta`` would make the worker unable to load its own model).
``<mode>.pt``
    optional byte-identical copies for ``ind2 ind3 ind4 ovt team`` — the hook
    for per-mode specialists.  Copying a shared net into all five is a
    deliberate no-op that lets the worker keep one code path.
``manifest.json``
    what is in the bundle and where it came from: source checkpoint + SHA-256,
    generation/step, network config and parameter count, the observation and
    action versions, the Elo table if an arena report was handed in, and the
    SHA-256 of every file written.
``shared.ts``
    a TorchScript **trace** for a future C++/mobile consumer.  It is a
    convenience, not the contract: the worker loads the ``.pt``.  Tracing is
    guarded — if it fails, or if the traced module disagrees with eager, the
    file is skipped and the reason is recorded in the manifest instead of
    breaking the export.

Command line::

    python -m splendor_ai.export --ckpt runs/nscc/weights/latest.pt \\
        --out dist/model --modes ind2 ind3 ind4 ovt team \\
        --elo reports/arena.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .encode import OBS_DIM, OBS_VERSION
from .model import (ACTION_VERSION, NetConfig, SplendorNet, count_params,
                    load_checkpoint, save_checkpoint)
from .rules.actions import NUM_ACTIONS

__all__ = [
    "BUNDLE_VERSION", "MODEL_MODES", "TS_OUTPUTS", "export_bundle",
    "TracedNet", "trace_network", "read_elo_table", "sha256_file", "main",
]

#: Bumped when the *bundle layout* changes (not the weights, not the encoder).
BUNDLE_VERSION = 1

#: Per-mode checkpoint names the worker looks for (``worker/config.py``).
MODEL_MODES: Tuple[str, ...] = ("ind2", "ind3", "ind4", "ovt", "team")

#: Output order of the TorchScript module — a traced module cannot return the
#: dict :meth:`SplendorNet.forward` returns, so the bundle documents the tuple.
TS_OUTPUTS: Tuple[str, ...] = ("logits", "value", "score", "stuck")


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# ── TorchScript ───────────────────────────────────────────────────────────

class TracedNet(torch.nn.Module):
    """Tuple-returning wrapper so :func:`torch.jit.trace` can see the outputs.

    Order is :data:`TS_OUTPUTS`.  Heads the config disabled come back as empty
    tensors rather than disappearing, so the arity is fixed for the consumer.
    """

    def __init__(self, model: SplendorNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        out = self.model(obs, mask)
        empty = obs.new_zeros((obs.shape[0], 0))
        return (out["logits"], out["value"], out.get("score", empty),
                out.get("stuck", empty))


def trace_network(model: SplendorNet, path: str, batch: int = 4,
                  tolerance: float = 1e-4, seed: int = 0
                  ) -> Dict[str, Any]:
    """Trace ``model``, verify it against eager, write ``path``.

    Returns a status dict for the manifest; never raises — a TorchScript
    failure must not cost you the bundle you actually deploy.
    """
    status: Dict[str, Any] = {"file": None, "outputs": list(TS_OUTPUTS),
                              "ok": False, "reason": None, "max_abs_diff": None}
    try:
        rng = np.random.default_rng(seed)
        model = model.to("cpu").eval()
        obs = torch.from_numpy(
            rng.standard_normal((batch, model.cfg.obs_dim)).astype(np.float32))
        mask_np = rng.random((batch, model.cfg.num_actions)) < 0.4
        mask_np[:, 0] = True                       # never a fully illegal row
        mask = torch.from_numpy(mask_np)
        wrapper = TracedNet(model).eval()
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traced = torch.jit.trace(wrapper, (obs, mask), strict=False)
            traced = torch.jit.freeze(traced)
            eager = wrapper(obs, mask)
            got = traced(obs, mask)
        diff = max(float((a - b).abs().max()) if a.numel() else 0.0
                   for a, b in zip(eager, got))
        status["max_abs_diff"] = diff
        if not (diff <= tolerance):
            status["reason"] = (f"traced output differs from eager by {diff:g} "
                                f"(> {tolerance:g}) — not written")
            return status
        traced.save(path)
        status.update(file=os.path.basename(path), ok=True)
    except Exception as exc:                               # pragma: no cover
        status["reason"] = f"{type(exc).__name__}: {exc}"
    return status


# ── Elo table ─────────────────────────────────────────────────────────────

def read_elo_table(path: str) -> Dict[str, Any]:
    """Pull the rating table out of an arena JSON report (``arena.py``).

    Only plain numbers survive into the checkpoint's ``meta`` — the worker
    loads with ``weights_only=True`` and the operator wants to be able to ask
    a deployed file "how strong was this, measured how".
    """
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    ratings = report.get("ratings", {}) or {}
    table = {
        name: {k: (float(v) if isinstance(v, (int, float)) else v)
               for k, v in row.items() if k in ("elo", "lo", "hi", "games",
                                                "score")}
        for name, row in (ratings.get("ratings", {}) or {}).items()
    }
    per_mode = {}
    for key, entry in (report.get("modes", {}) or {}).items():
        rows = (entry.get("ratings", {}) or {}).get("ratings", {})
        per_mode[key] = {name: row.get("elo") for name, row in rows.items()}
    return {
        "source": os.path.basename(path),
        "generated": report.get("generated"),
        "anchor": ratings.get("anchor"),
        "anchor_rating": ratings.get("anchor_rating"),
        "num_games": report.get("num_games"),
        "ratings": table,
        "per_mode_elo": per_mode,
    }


# ── the bundle ────────────────────────────────────────────────────────────

def export_bundle(ckpt_path: str, out_dir: str,
                  modes: Optional[Sequence[str]] = None,
                  torchscript: bool = True,
                  elo: Optional[Any] = None,
                  notes: Optional[str] = None,
                  overwrite: bool = True) -> Dict[str, Any]:
    """Write a worker-loadable bundle for ``ckpt_path`` into ``out_dir``.

    ``modes`` are the per-mode copies to make (``None``/``[]`` → only
    ``shared.pt``; ``'all'`` → all five worker keys).  ``elo`` may be a path to
    an arena JSON report or an already-parsed mapping.  Returns the manifest.

    The checkpoint is *re-saved* rather than copied: that runs it through
    :func:`~.model.load_checkpoint`'s version gate on the way in (so a stale
    ``obs_version`` is caught here, on a workstation, instead of on the
    deployment box at 2 a.m.) and through ``save_checkpoint``'s atomic write on
    the way out.
    """
    ckpt_path = str(ckpt_path)
    out_dir = str(out_dir)
    model, ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(modes, str):
        modes = list(MODEL_MODES) if modes == "all" else [modes]
    mode_keys = [m for m in (modes or []) if m]
    unknown = [m for m in mode_keys if m not in MODEL_MODES]
    if unknown:
        raise ValueError(
            f"unknown mode key(s) {unknown}; the worker looks for "
            f"{list(MODEL_MODES)} (worker/config.py::MODEL_MODES)")

    elo_table: Optional[Dict[str, Any]] = None
    if isinstance(elo, str):
        elo_table = read_elo_table(elo)
    elif isinstance(elo, Mapping):
        elo_table = dict(elo)
    elif elo is None:
        stored = (ckpt.get("meta") or {}).get("elo")
        elo_table = dict(stored) if isinstance(stored, Mapping) else None

    source_meta = dict(ckpt.get("meta") or {})
    generation = int(ckpt.get("generation", 0) or 0)
    step = int(ckpt.get("step", 0) or 0)
    meta: Dict[str, Any] = dict(source_meta)
    meta.update({
        "bundle_version": BUNDLE_VERSION,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_checkpoint": os.path.abspath(ckpt_path),
        "source_sha256": sha256_file(ckpt_path),
        # ``str`` matters: ``TorchVersion`` is not a ``weights_only``-safe
        # global, and the worker loads every checkpoint with weights_only=True.
        "torch_version": str(torch.__version__),
        "modes": list(mode_keys),
    })
    if notes:
        meta["notes"] = str(notes)
    if elo_table is not None:
        meta["elo"] = elo_table

    shared = os.path.join(out_dir, "shared.pt")
    if os.path.exists(shared) and not overwrite:
        raise FileExistsError(shared)
    save_checkpoint(shared, model, {"step": step, "generation": generation,
                                    "meta": meta})

    files = ["shared.pt"]
    for key in mode_keys:
        target = os.path.join(out_dir, f"{key}.pt")
        shutil.copyfile(shared, target)
        files.append(f"{key}.pt")

    ts_status: Dict[str, Any] = {"file": None, "ok": False,
                                 "reason": "disabled"}
    if torchscript:
        ts_status = trace_network(model, os.path.join(out_dir, "shared.ts"))
        if ts_status.get("ok"):
            files.append("shared.ts")
        else:
            warnings.warn(f"TorchScript export skipped: {ts_status['reason']}",
                          RuntimeWarning, stacklevel=2)

    manifest: Dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "created": meta["exported_at"],
        "source_checkpoint": meta["source_checkpoint"],
        "source_sha256": meta["source_sha256"],
        "obs_version": OBS_VERSION,
        "obs_dim": OBS_DIM,
        "action_version": ACTION_VERSION,
        "num_actions": NUM_ACTIONS,
        "generation": generation,
        "step": step,
        "net": model.cfg.to_dict(),
        "parameters": count_params(model),
        "torch_version": str(torch.__version__),
        "loader": "splendor_ai.model.load_checkpoint",
        "worker": {
            "model_dir_layout": ["<mode>.pt", "shared.pt"],
            "mode_keys": list(MODEL_MODES),
            "shipped_mode_files": list(mode_keys),
        },
        "torchscript": ts_status,
        "elo": elo_table,
        "notes": notes,
        "files": [],
    }
    for name in files:
        path = os.path.join(out_dir, name)
        manifest["files"].append({
            "name": name, "bytes": os.path.getsize(path),
            "sha256": sha256_file(path),
        })
    manifest_path = os.path.join(out_dir, "manifest.json")
    tmp = f"{manifest_path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    os.replace(tmp, manifest_path)
    manifest["manifest_path"] = manifest_path
    manifest["out_dir"] = os.path.abspath(out_dir)
    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m splendor_ai.export",
        description="Package a checkpoint into a worker-loadable bundle "
                    "(shared.pt [+ per-mode copies] + manifest.json "
                    "[+ shared.ts]).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""example:
  python -m splendor_ai.export --ckpt runs/nscc/weights/latest.pt \\
      --out dist/model --modes all --elo reports/arena.json
then copy dist/model/shared.pt to the worker's MODEL_DIR.""")
    parser.add_argument("--ckpt", required=True, help="checkpoint to export")
    parser.add_argument("--out", required=True, help="bundle directory")
    parser.add_argument("--modes", nargs="*", default=[],
                        help=f"per-mode copies: {list(MODEL_MODES)} or 'all'")
    parser.add_argument("--elo", default=None,
                        help="arena JSON report whose Elo table goes into the "
                             "bundle metadata")
    parser.add_argument("--notes", default=None,
                        help="free-text note stored in the manifest and meta")
    parser.add_argument("--no-torchscript", action="store_true",
                        help="skip the TorchScript trace")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    modes: Any = args.modes
    if modes and len(modes) == 1 and modes[0] == "all":
        modes = "all"
    manifest = export_bundle(args.ckpt, args.out, modes=modes,
                             torchscript=not args.no_torchscript,
                             elo=args.elo, notes=args.notes)
    if not args.quiet:
        print(f"exported generation {manifest['generation']} "
              f"(step {manifest['step']}, {manifest['parameters']:,} params) "
              f"→ {manifest['out_dir']}")
        for entry in manifest["files"]:
            print(f"  {entry['name']:<14} {entry['bytes']:>12,} B  "
                  f"{entry['sha256'][:12]}")
        if not manifest["torchscript"]["ok"]:
            print(f"  (no TorchScript: {manifest['torchscript']['reason']})")
        print("copy shared.pt to the worker's MODEL_DIR "
              "(see README 'Evaluation & export').")
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
