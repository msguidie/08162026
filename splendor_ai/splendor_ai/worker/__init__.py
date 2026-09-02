"""Deployment worker — ``docs/AI_BRIDGE.md`` §1, ``docs/AI_DESIGN.md`` §1.9.

A socket.io **client** that runs on the user's own GPU box, dials out to the
Render server and answers ``ai_move_request`` with a move chosen by
net + MCTS.  Nothing here is imported by the training stack.

::

    python -m splendor_ai.worker.worker            # serve moves
    python -m splendor_ai.worker.worker --once     # offline self-test

Layout
------
======================= ===================================================
 module                  contents
======================= ===================================================
 :mod:`~.config`         ``.env`` / environment → :class:`~.config.WorkerConfig`
 :mod:`~.adapter`        payload → ``GameState`` (:func:`~.adapter.hydrate`),
                         action index → wire action (:func:`~.adapter.to_wire`)
 :mod:`~.agent`          :class:`~.agent.MoveAgent` and the fallback ladder
 :mod:`~.client`         socket.io plumbing and the JSONL move log
 :mod:`~.worker`         the CLI entry point
======================= ===================================================

The submodules are exposed lazily so that ``import splendor_ai.worker`` does
not drag torch or python-socketio into a process that only wants the adapter.
"""

from typing import Any

__all__ = ["adapter", "agent", "client", "config", "worker",
           "hydrate", "to_wire", "HydrationError", "MoveAgent",
           "WorkerConfig", "load_config", "main"]

_LAZY = {
    "hydrate": "adapter", "to_wire": "adapter", "HydrationError": "adapter",
    "MoveAgent": "agent",
    "WorkerConfig": "config", "load_config": "config",
    "main": "worker",
}


def __getattr__(name: str) -> Any:
    from importlib import import_module
    if name in ("adapter", "agent", "client", "config", "worker"):
        return import_module(f".{name}", __name__)
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
