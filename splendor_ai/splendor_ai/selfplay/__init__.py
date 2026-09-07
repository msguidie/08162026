"""Self-play trainer for the Splendor variant (``docs/AI_DESIGN.md`` §1.8).

Importing this package is the first thing every trainer process does, so the
thread-environment fix lives here: ``OMP/MKL/OPENBLAS_NUM_THREADS=1`` must be
in the environment **before** numpy or torch is imported, or 56 actor
processes each spin up a 64-thread BLAS pool on a 64-core node and the whole
pipeline runs at a fraction of its speed with no error message anywhere
(judges.md, "MANDATORY ops detail").  Python guarantees that
``splendor_ai.selfplay.__init__`` runs before any of its submodules, and every
submodule imports numpy/torch only after that, so a plain
``from splendor_ai.selfplay import actor`` is already safe.

Modules
-------
``config``   dataclasses + YAML + ``--set`` overrides
``sample``   the training record, its numpy packing and C5 augmentation
``replay``   generational rolling window, uniform sampling, npz checkpoint
``actor``    self-play process: G games in lockstep, PCR, opponent mixing
``inference``batched inference server + ``RemoteEvaluator`` client
``learner``  AdamW/warmup-cosine AlphaZero learner, publishes ``weights/latest.pt``
``ppo_learner`` the fallback learner (``learner.algorithm: ppo``)
``metrics``  JSONL/TensorBoard writer, eval bots and the anchor arena
``train``    orchestrator CLI
``bootstrap``NN-free MCTS teacher -> supervised warm start
"""

from __future__ import annotations

import os

#: Every one of these has to be 1 in an actor: the model is tiny and each
#: process is already a unit of parallelism.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def set_thread_env(threads: int = 1) -> None:
    """``setdefault`` the BLAS thread limits.  Called at import."""
    for var in THREAD_ENV_VARS:
        os.environ.setdefault(var, str(threads))


set_thread_env()


def configure_process(torch_threads: int = 1, seed: int = None) -> None:
    """Per-process torch setup: single-threaded by default (measured: a
    4-thread forward of the 128x2 smoke net is *slower* than 1 thread), plus
    an optional seed."""
    import torch

    torch.set_num_threads(max(1, int(torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:                                    # already started
        pass
    if seed is not None:
        torch.manual_seed(int(seed))


__all__ = ["THREAD_ENV_VARS", "set_thread_env", "configure_process"]
