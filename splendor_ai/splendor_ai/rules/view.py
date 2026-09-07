"""Per-seat information sets — what one player is allowed to know.

The server hides other seats' reserved cards from the wire
(``clientViewForPlayer``), but the *board* half of that information is not
actually secret: everybody watched the card leave the market.  Only a card
taken blind off a deck top is hidden, and even then its tier is public
(``RESERVE_FROM_DECK`` announces ``tier``).  The engine therefore tracks a
``reserved_public`` flag per reserved card, and this module turns that into the
observation the encoder will consume.

Also hidden: the identity and order of the face-down decks.  Only
``deck_counts`` is public.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .cards import CARD_POINTS, CARD_REWARD, CARD_TIER, CARD_COST
from .engine import GameState, PlayerState, can_afford, qualified_tiles

HIDDEN_CARD = -1


def card_info(card_id: int) -> Dict[str, Any]:
    return {
        "id": card_id,
        "tier": CARD_TIER[card_id],
        "reward": CARD_REWARD[card_id],
        "points": CARD_POINTS[card_id],
        "cost": list(CARD_COST[card_id]),
    }


def _reserved_view(player: PlayerState, full: bool) -> List[Dict[str, Any]]:
    """``full`` for the viewer's own seat; otherwise board-reserved cards stay
    visible and deck-reserved cards collapse to ``{id: -1, tier: <tier>}``."""
    out = []
    for i, cid in enumerate(player.reserved):
        if full or player.reserved_public[i]:
            info = card_info(cid)
            info["hidden"] = False
            info["public"] = player.reserved_public[i]
            out.append(info)
        else:
            # Deck-reserved: only the tier is public knowledge.
            out.append({"id": HIDDEN_CARD, "tier": CARD_TIER[cid],
                        "reward": -1, "points": -1,
                        "cost": [-1, -1, -1, -1, -1],
                        "hidden": True, "public": False})
    return out


def public_view(state: GameState, player: int) -> Dict[str, Any]:
    """The information set of seat ``player``.

    Everything here is derivable from what that seat has legitimately seen.
    """
    me = state.players[player]
    return {
        "viewer": player,
        "phase": state.phase,
        "mode": state.mode,
        "teamLayout": state.team_layout,
        "numPlayers": state.num_players,
        "config": dict(state.config),

        "board": [list(row) for row in state.board],
        "deckCounts": list(state.deck_counts),
        "gems": list(state.gems),
        "tiles": list(state.tiles),

        "currentPlayer": state.current_player,
        "roundStartPlayer": state.round_start_player,
        "turnNumber": state.turn_number,
        "finalRoundTriggeredBy": state.final_round_triggered_by,
        "resigned": list(state.resigned),
        "gameResult": (dict(state.game_result)
                       if state.game_result is not None else None),
        # Only the acting seat is ever told about a pending noble choice.
        "pendingTileChoice": (list(state.pending_tile_choice)
                              if state.pending_tile_choice is not None
                              and state.current_player == player else None),
        "turnAction": state.turn_action,

        "players": [
            {
                "seat": i,
                "isSelf": i == player,
                "teamId": p.team_id,
                "isTeammate": (p.team_id is not None
                               and p.team_id == me.team_id and i != player),
                "gems": list(p.gems),
                "discount": list(p.discount),
                "cards": list(p.cards),          # tableau is always public
                "cardCount": len(p.cards),
                "reserved": _reserved_view(p, full=(i == player)),
                "reservedCount": len(p.reserved),
                "tiles": list(p.tiles),
                "score": p.score,
                "resigned": i in state.resigned,
            }
            for i, p in enumerate(state.players)
        ],

        # Convenience derived facts the encoder wants anyway.
        "affordable": {
            "board": [[can_afford(me, cid) for cid in row] for row in state.board],
            "reserved": [can_afford(me, cid) for cid in me.reserved],
        },
        "qualifiedTiles": qualified_tiles(state, me),
    }


__all__ = ["public_view", "card_info", "HIDDEN_CARD"]
