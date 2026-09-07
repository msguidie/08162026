"""Bit-exact Python port of ``server/gameLogic.js``.

Submodules: :mod:`cards`, :mod:`actions`, :mod:`engine`, :mod:`view`.

The common names are re-exported lazily (PEP 562) so that importing the
package is cheap and ``python -m splendor_ai.rules.cards`` runs the card-table
self test without re-executing an already-imported module.
"""

from typing import Any

_EXPORTS = {
    # module -> names
    "cards": ("CARDS", "TILES", "Card", "Tile", "NUM_CARDS", "NUM_TILES"),
    "actions": ("NUM_ACTIONS", "ACTION_RESIGN", "ACTION_TIMEOUT",
                "TAKE_PATTERNS", "ACTION_TABLE", "action_name"),
    "engine": ("GameState", "PlayerState", "IllegalAction", "new_game",
               "legal_mask", "legal_mask_np", "legal_actions", "is_stuck",
               "apply", "step", "resign", "timeout", "to_protocol",
               "to_replay_code", "from_replay_code", "replay",
               "replay_actions", "rating_changes", "individual_winners",
               "team_stats", "qualifying_team_ids", "resolve_team_winners",
               "resolve_one_vs_two_winners"),
    "view": ("public_view",),
}

_NAME_TO_MODULE = {name: mod for mod, names in _EXPORTS.items() for name in names}

__all__ = sorted(_NAME_TO_MODULE) + sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    from importlib import import_module
    if name in _EXPORTS:
        return import_module(f".{name}", __name__)
    module = _NAME_TO_MODULE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return __all__
