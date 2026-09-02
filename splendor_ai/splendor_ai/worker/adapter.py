"""Server payload ⇄ engine ``GameState`` adapter (``docs/AI_BRIDGE.md`` §1).

The bridge sends the worker a *client view* of the position — the very object
``clientViewForPlayer`` builds for a browser, with other seats' reserved cards
collapsed to ``{id, tier, hidden, known}`` — plus ``knownReserved`` (every card
id that was reserved face-up off the board, i.e. public knowledge) and
``pendingTileChoice``.  The decks are never sent: only ``deckCounts`` is.

:func:`hydrate` turns that back into a :class:`~splendor_ai.rules.engine.
GameState` the search can play on, and :func:`to_wire` turns a 65-way action
index back into the worker action dict the bridge understands.

What hydration has to invent, and why it is safe
------------------------------------------------
The engine wants *concrete* card ids everywhere, but two things are genuinely
hidden from the acting seat:

``decks``
    Only the count is public.  We fill each tier with unseen cards of that
    tier in ascending id order so that ``len(decks[t]) == deckCounts[t]``.
    Nothing downstream reads the deck *order*: the encoder uses only
    ``deck_counts`` and the public unseen pool (``encode.py`` §1.3), and the
    search re-samples the decks from the unseen pool on every simulation
    (``search/determinize.py``).  The fill only has to make
    ``RESERVE_FROM_DECK`` legal and board refills possible on the root state.

another seat's blind reserve
    ``{id: -1, known: false}``.  The engine has no "unknown card" value, so
    each such slot gets a deterministic unseen card **of the announced tier**
    and ``reserved_public = False``.  Every consumer that could be misled by
    the invented identity ignores it: the encoder emits a tier-only sentinel
    for a non-public reserve, and :func:`~splendor_ai.search.determinize
    .determinize` overwrites the slot with a fresh random unseen card of that
    tier for every universe.  The placeholder is excluded from the deck fill
    so no card id is ever in two places at once.

Own reserved cards arrive as full card objects with no "was it public?" flag,
so their ``reserved_public`` is recovered from ``knownReserved`` — which is
exactly what that list is for.  It matters: a search leaf where an *opponent*
is to move encodes our reserves through the other-seat block, and a card we
took blind off a deck must stay hidden there.

Consistency validation (Splendor-Zero style)
--------------------------------------------
``hydrate(payload, validate=True)`` refuses a payload that cannot describe a
real position, rather than quietly searching a corrupt one.  See
:data:`HYDRATION_INVARIANTS` for the list; every failure raises
:class:`HydrationError` naming the field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..rules import engine as E
from ..rules.actions import CHOOSE_TILE_START, NUM_ACTIONS
from ..rules.cards import (
    CARD_COST, CARD_POINTS, CARD_REWARD, CARD_TIER, CARD_TIER0, CARDS_BY_TIER,
    NUM_CARDS, NUM_TILES, TILE_POINTS, TILE_REQ,
)

__all__ = [
    "HydrationError", "hydrate", "to_wire", "wire_action_kind",
    "payload_mode_key", "HYDRATION_INVARIANTS",
]


class HydrationError(ValueError):
    """The payload does not describe a position the engine can represent."""


#: Documented, in the order :func:`hydrate` checks them.
HYDRATION_INVARIANTS: Tuple[str, ...] = (
    "the payload carries a state, players, board and deckCounts",
    "numPlayers == len(players) and 2 <= numPlayers <= 4",
    "playerIndex is a seat and equals currentPlayerIndex",
    "every card object matches the canonical table (tier/reward/points/cost)",
    "every tile object matches the canonical table (points/requirement)",
    "no card id appears twice (board, tableaus, known reserves)",
    "no tile id appears twice (board tiles and claimed tiles)",
    "each player's score == sum(card points) + sum(tile points)",
    "each player's derived discount matches a discount field when one is sent",
    "tokens are conserved: bank + hands == tokensPerColor x5 + wildTokens",
    "every tier has at least deckCounts[t] unseen cards left to fill it with",
    "len(decks[t]) == deckCounts[t] after the fill",
    "reserved counts stay within maxReserved and board rows within cardsPerRow",
)

_MODE_MAP = {
    "INDIVIDUAL": E.MODE_INDIVIDUAL,
    "TEAM": E.MODE_TEAM,
    "ONE_V_TWO": E.MODE_ONE_V_TWO,
}


# ── small readers ─────────────────────────────────────────────────────────

def _require(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload or payload[key] is None:
        raise HydrationError(f"payload is missing {key!r}")
    return payload[key]


def _as_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HydrationError(f"{what} is not a number: {value!r}")
    out = int(value)
    if out != value:
        raise HydrationError(f"{what} is not an integer: {value!r}")
    return out


def _card_id(entry: Any, what: str) -> int:
    """Card id out of an int or a ``{id: ...}`` object (``-1`` = hidden)."""
    if isinstance(entry, Mapping):
        entry = entry.get("id", -1)
    cid = _as_int(entry, what)
    if cid < -1 or cid >= NUM_CARDS:
        raise HydrationError(f"{what}: card id {cid} out of range")
    return cid


def _tile_id(entry: Any, what: str) -> int:
    if isinstance(entry, Mapping):
        entry = entry.get("id", -1)
    tid = _as_int(entry, what)
    if tid < 0 or tid >= NUM_TILES:
        raise HydrationError(f"{what}: tile id {tid} out of range")
    return tid


def _check_card_object(entry: Any, cid: int, what: str) -> None:
    """A card object must agree with the canonical table for its id."""
    if not isinstance(entry, Mapping) or cid < 0:
        return
    if "tier" in entry and entry["tier"] not in (0, None):
        if _as_int(entry["tier"], f"{what}.tier") != CARD_TIER[cid]:
            raise HydrationError(
                f"{what}: card {cid} says tier {entry['tier']}, the table says "
                f"{CARD_TIER[cid]}")
    if entry.get("hidden"):
        return                       # {id, tier, hidden, known} — nothing else
    if "reward" in entry and entry["reward"] is not None:
        if _as_int(entry["reward"], f"{what}.reward") != CARD_REWARD[cid]:
            raise HydrationError(
                f"{what}: card {cid} says reward {entry['reward']}, the table "
                f"says {CARD_REWARD[cid]}")
    if "points" in entry and entry["points"] is not None:
        if _as_int(entry["points"], f"{what}.points") != CARD_POINTS[cid]:
            raise HydrationError(
                f"{what}: card {cid} says points {entry['points']}, the table "
                f"says {CARD_POINTS[cid]}")
    cost = entry.get("cost")
    if isinstance(cost, Sequence) and not isinstance(cost, (str, bytes)):
        got = tuple(_as_int(c, f"{what}.cost") for c in cost)
        if got != tuple(CARD_COST[cid]):
            raise HydrationError(
                f"{what}: card {cid} says cost {list(got)}, the table says "
                f"{list(CARD_COST[cid])}")


def _check_tile_object(entry: Any, tid: int, what: str) -> None:
    if not isinstance(entry, Mapping):
        return
    if "points" in entry and entry["points"] is not None:
        if _as_int(entry["points"], f"{what}.points") != TILE_POINTS[tid]:
            raise HydrationError(
                f"{what}: tile {tid} says points {entry['points']}, the table "
                f"says {TILE_POINTS[tid]}")
    req = entry.get("requirement")
    if isinstance(req, Sequence) and not isinstance(req, (str, bytes)):
        got = tuple(_as_int(r, f"{what}.requirement") for r in req)
        if got != tuple(TILE_REQ[tid]):
            raise HydrationError(
                f"{what}: tile {tid} says requirement {list(got)}, the table "
                f"says {list(TILE_REQ[tid])}")


def _gems(entry: Any, what: str, width: int = 6) -> List[int]:
    if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)):
        raise HydrationError(f"{what} is not a list: {entry!r}")
    out = [_as_int(v, f"{what}[{i}]") for i, v in enumerate(entry)]
    if len(out) != width:
        raise HydrationError(f"{what} has {len(out)} entries, expected {width}")
    if any(v < 0 for v in out):
        raise HydrationError(f"{what} has a negative entry: {out}")
    return out


def payload_mode_key(payload: Mapping[str, Any]) -> str:
    """``ind2|ind3|ind4|ovt|team`` for a request payload (checkpoint lookup)."""
    from .config import mode_key
    state = payload.get("state") if "state" in payload else payload
    state = state if isinstance(state, Mapping) else {}
    players = state.get("players")
    n = state.get("numPlayers")
    if n is None and isinstance(players, Sequence):
        n = len(players)
    return mode_key(state.get("gameMode"), n)


# ── hydration ─────────────────────────────────────────────────────────────

def hydrate(payload: Mapping[str, Any],
            validate: bool = True) -> Tuple[E.GameState, int]:
    """Rebuild ``(GameState, seat)`` from an ``ai_move_request`` payload.

    ``payload`` is the whole request (``{requestId, playerIndex, kind, state,
    knownReserved, pendingTileChoice, ...}``); a bare ``state`` object is also
    accepted, in which case the seat is ``state.currentPlayerIndex``.

    With ``validate=False`` only the checks needed to *build* a legal state are
    run (ranges, deck sizes); the derived-consistency ones — score, token
    conservation, card-table agreement — are skipped.  The worker uses that as
    the last rung of its ladder so a server-side inconsistency degrades the
    move quality instead of taking the bot off the board.
    """
    if not isinstance(payload, Mapping):
        raise HydrationError(f"payload is not an object: {type(payload)!r}")
    view = payload.get("state") if "state" in payload else payload
    if not isinstance(view, Mapping):
        raise HydrationError("payload.state is not an object")

    players_raw = _require(view, "players")
    if not isinstance(players_raw, Sequence) or not players_raw:
        raise HydrationError("payload.state.players is empty")
    num_players = _as_int(view.get("numPlayers", len(players_raw)), "numPlayers")
    if num_players != len(players_raw):
        raise HydrationError(
            f"numPlayers {num_players} != len(players) {len(players_raw)}")
    if not 2 <= num_players <= 4:
        raise HydrationError(f"numPlayers {num_players} outside 2..4")

    current = _as_int(_require(view, "currentPlayerIndex"), "currentPlayerIndex")
    seat = _as_int(payload.get("playerIndex", current), "playerIndex")
    if not 0 <= seat < num_players:
        raise HydrationError(f"playerIndex {seat} is not a seat")
    if validate and seat != current:
        raise HydrationError(
            f"playerIndex {seat} != currentPlayerIndex {current} — the request "
            f"does not belong to the seat to move")

    mode_raw = str(view.get("gameMode") or E.MODE_INDIVIDUAL).upper()
    mode = _MODE_MAP.get(mode_raw)
    if mode is None:
        raise HydrationError(f"unknown gameMode {mode_raw!r}")
    layout = view.get("teamLayout")
    layout = str(layout).upper() if layout else None
    if mode == E.MODE_TEAM:
        layout = "OPPOSITE" if layout == "OPPOSITE" else "ADJACENT"
    else:
        layout = None

    state = E.GameState()
    state.num_players = num_players
    state.mode = mode
    state.team_layout = layout
    state.config = _config_of(view, num_players, validate)
    cfg = state.config

    # -- board ----------------------------------------------------------
    board_raw = _require(view, "board")
    if not isinstance(board_raw, Sequence) or len(board_raw) != 3:
        raise HydrationError("payload.state.board must have three rows")
    board: List[List[int]] = []
    for t, row in enumerate(board_raw):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise HydrationError(f"board[{t}] is not a list")
        ids: List[int] = []
        for slot, entry in enumerate(row):
            what = f"board[{t}][{slot}]"
            cid = _card_id(entry, what)
            if cid < 0:
                raise HydrationError(f"{what}: a board card is never hidden")
            if CARD_TIER0[cid] != t:
                raise HydrationError(
                    f"{what}: card {cid} is tier {CARD_TIER[cid]} but sits in "
                    f"row {t + 1}")
            if validate:
                _check_card_object(entry, cid, what)
            ids.append(cid)
        if validate and len(ids) > cfg["cardsPerRow"]:
            raise HydrationError(
                f"board[{t}] holds {len(ids)} cards, cardsPerRow is "
                f"{cfg['cardsPerRow']}")
        board.append(ids)
    state.board = board

    # -- bank, revealed tiles -------------------------------------------
    state.gems = _gems(_require(view, "gems"), "gems")
    tiles_raw = view.get("bonusTiles", view.get("tiles", []))
    state.tiles = [_tile_id(t, f"bonusTiles[{i}]")
                   for i, t in enumerate(tiles_raw or [])]
    if validate:
        for i, entry in enumerate(tiles_raw or []):
            _check_tile_object(entry, state.tiles[i], f"bonusTiles[{i}]")

    # -- players ---------------------------------------------------------
    known_reserved = {
        _as_int(c, "knownReserved") for c in (payload.get("knownReserved") or [])
        if isinstance(c, (int, float)) and not isinstance(c, bool)
    }
    teams_raw = view.get("teams") or []
    team_of_seat = _team_ids(teams_raw, players_raw, mode, layout, num_players)

    hidden_slots: List[Tuple[int, int, int]] = []   # (seat, slot, tier0)
    seen: Dict[int, str] = {}                       # card id → where

    def _claim(cid: int, what: str) -> None:
        if not validate:
            return
        previous = seen.get(cid)
        if previous is not None:
            raise HydrationError(
                f"card {cid} appears twice: {previous} and {what}")
        seen[cid] = what

    for i, raw in enumerate(players_raw):
        if not isinstance(raw, Mapping):
            raise HydrationError(f"players[{i}] is not an object")
        p = E.PlayerState(str(raw.get("username", f"p{i}")),
                          team_of_seat[i], i)
        p.gems = _gems(raw.get("gems", [0] * 6), f"players[{i}].gems")

        for j, entry in enumerate(raw.get("cards") or []):
            what = f"players[{i}].cards[{j}]"
            cid = _card_id(entry, what)
            if cid < 0:
                raise HydrationError(f"{what}: a tableau card is never hidden")
            if validate:
                _check_card_object(entry, cid, what)
            _claim(cid, what)
            p.cards.append(cid)
            p.discount[CARD_REWARD[cid]] += 1

        for j, entry in enumerate(raw.get("reserved") or []):
            what = f"players[{i}].reserved[{j}]"
            cid = _card_id(entry, what)
            known = True
            if isinstance(entry, Mapping) and entry.get("hidden"):
                # Another seat's reserve: `known` (or a real id) means the card
                # was taken face-up off the board, so everyone saw it.
                known = bool(entry.get("known", cid >= 0)) and cid >= 0
            elif isinstance(entry, Mapping) and "public" in entry and i != seat:
                known = bool(entry["public"]) and cid >= 0     # rules/view.py
            elif i != seat:
                known = cid >= 0
            if cid >= 0 and i != seat:
                known = known or cid in known_reserved
            if not known or cid < 0:
                tier0 = _hidden_tier0(entry, what)
                hidden_slots.append((i, len(p.reserved), tier0))
                p.reserved.append(-1)          # placeholder, filled in below
                p.reserved_public.append(False)
                continue
            if validate:
                _check_card_object(entry, cid, what)
            _claim(cid, what)
            p.reserved.append(cid)
            # Own reserves carry no public flag on the wire — knownReserved is
            # the only record of how the card was taken.
            p.reserved_public.append(cid in known_reserved if i == seat else True)
        if validate and len(p.reserved) > cfg["maxReserved"]:
            raise HydrationError(
                f"players[{i}] holds {len(p.reserved)} reserved cards, "
                f"maxReserved is {cfg['maxReserved']}")

        for j, entry in enumerate(raw.get("bonusTiles") or raw.get("tiles") or []):
            what = f"players[{i}].bonusTiles[{j}]"
            tid = _tile_id(entry, what)
            if validate:
                _check_tile_object(entry, tid, what)
            p.tiles.append(tid)

        p.score = _as_int(raw.get("score", 0), f"players[{i}].score")
        if validate:
            derived = (sum(CARD_POINTS[c] for c in p.cards)
                       + sum(TILE_POINTS[t] for t in p.tiles))
            if derived != p.score:
                raise HydrationError(
                    f"players[{i}].score is {p.score} but its cards and tiles "
                    f"are worth {derived}")
            sent = raw.get("discount")
            if sent is not None:
                got = [_as_int(v, f"players[{i}].discount") for v in sent]
                if got != p.discount:
                    raise HydrationError(
                        f"players[{i}].discount is {got} but its tableau gives "
                        f"{p.discount}")
        state.players.append(p)

    if validate:
        tile_seen: Dict[int, str] = {}
        for i, tid in enumerate(state.tiles):
            tile_seen[tid] = f"bonusTiles[{i}]"
        for i, p in enumerate(state.players):
            for tid in p.tiles:
                if tid in tile_seen:
                    raise HydrationError(
                        f"tile {tid} appears twice: {tile_seen[tid]} and "
                        f"players[{i}].bonusTiles")
                tile_seen[tid] = f"players[{i}].bonusTiles"

    for row in state.board:
        for cid in row:
            _claim(cid, "board")

    # -- decks and the blind reserves ------------------------------------
    deck_counts = [_as_int(v, "deckCounts")
                   for v in _require(view, "deckCounts")]
    if len(deck_counts) != 3 or any(v < 0 for v in deck_counts):
        raise HydrationError(f"deckCounts {deck_counts} is not three counts")
    _fill_hidden_and_decks(state, seat, hidden_slots, deck_counts)

    # -- turn plumbing ---------------------------------------------------
    state.current_player = current
    rsp = view.get("roundStartPlayer")
    state.round_start_player = (current if rsp is None
                                else _as_int(rsp, "roundStartPlayer"))
    state.turn_number = _as_int(view.get("turnNumber", 0), "turnNumber")
    frt = view.get("finalRoundTriggeredBy")
    state.final_round_triggered_by = (None if frt is None
                                      else _as_int(frt, "finalRoundTriggeredBy"))
    state.resigned = [_as_int(v, "resignedPlayers")
                      for v in (view.get("resignedPlayers")
                                or view.get("resigned") or [])]
    state.phase = (E.PHASE_GAME_OVER if str(view.get("phase")) == "GAME_OVER"
                   else E.PHASE_PLAYING)
    result = view.get("gameResult")
    state.game_result = dict(result) if isinstance(result, Mapping) else None
    state.turn_action = _turn_action(view.get("turnAction"), validate)
    pending = payload.get("pendingTileChoice")
    if pending is None:
        pending = view.get("_pendingTileChoice")
    if pending:
        state.pending_tile_choice = [_tile_id(t, "pendingTileChoice")
                                     for t in pending]
    state.teams = (
        [{"id": tid,
          "playerIndices": [i for i, p in enumerate(state.players)
                            if p.team_id == tid]}
         for tid in (0, 1)]
        if mode != E.MODE_INDIVIDUAL else [])

    if validate:
        _check_tokens(state)
    return state, seat


def _config_of(view: Mapping[str, Any], num_players: int,
               validate: bool) -> Dict[str, int]:
    """The payload's config, defaulted from ``make_config`` per key."""
    default = E.make_config(num_players)
    raw = view.get("config")
    if not isinstance(raw, Mapping):
        return default
    out = dict(default)
    for key, value in raw.items():
        if key in out and isinstance(value, (int, float)) \
                and not isinstance(value, bool):
            out[key] = int(value)
    if validate and out["maxReserved"] != default["maxReserved"]:
        raise HydrationError(
            f"config.maxReserved is {out['maxReserved']}, this variant uses "
            f"{default['maxReserved']}")
    return out


def _team_ids(teams_raw: Any, players_raw: Sequence[Any], mode: str,
              layout: Optional[str], num_players: int) -> List[Optional[int]]:
    """Seat → team id, from ``teams``, then ``player.teamId``, then the map."""
    out: List[Optional[int]] = [None] * num_players
    if mode == E.MODE_INDIVIDUAL:
        return out
    if isinstance(teams_raw, Sequence):
        for team in teams_raw:
            if not isinstance(team, Mapping):
                continue
            tid = team.get("id")
            for index in team.get("playerIndices") or []:
                if isinstance(index, int) and 0 <= index < num_players:
                    out[index] = int(tid) if tid is not None else None
    for i, raw in enumerate(players_raw):
        if isinstance(raw, Mapping) and raw.get("teamId") is not None:
            out[i] = int(raw["teamId"])
    if any(v is None for v in out):
        default = E.default_team_ids(mode, layout, num_players)
        if default is not None:
            for i in range(num_players):
                if out[i] is None and i < len(default):
                    out[i] = int(default[i])
    return out


def _turn_action(raw: Any, validate: bool = True) -> Optional[str]:
    """``{type: 'BUY'}`` → ``'BUY'``.

    A bot request only ever sees ``null`` (between turns) or ``BUY`` (a noble
    choice is pending): the bridge applies a whole turn action atomically and
    ``maybeAct`` is re-entrancy guarded, so no half-finished ``TAKE_GEMS`` /
    ``RESERVE`` can be observed on a bot seat.  Anything else would not be
    representable — the engine models only those two — so it is rejected.
    """
    if raw is None:
        return None
    kind = raw.get("type") if isinstance(raw, Mapping) else raw
    if kind is None:
        return None
    if kind == E.TA_BUY:
        return E.TA_BUY
    if not validate:
        return None
    raise HydrationError(
        f"turnAction {kind!r} is a half-finished human turn; a bot seat can "
        f"only be asked to move with turnAction null or BUY")


def _hidden_tier0(entry: Any, what: str) -> int:
    """0-based tier of a blind reserve (``RESERVE_FROM_DECK`` announces it)."""
    tier = entry.get("tier") if isinstance(entry, Mapping) else None
    if tier is None:
        raise HydrationError(f"{what}: a hidden reserve must announce its tier")
    tier = _as_int(tier, f"{what}.tier")
    if tier in (1, 2, 3):
        return tier - 1
    raise HydrationError(
        f"{what}: tier {tier} is not 1..3 — a hidden reserve must come from "
        f"aiBridge.buildObservation, which keeps the real tier")


def _fill_hidden_and_decks(state: E.GameState, seat: int,
                           hidden_slots: Sequence[Tuple[int, int, int]],
                           deck_counts: Sequence[int]) -> None:
    """Give every blind reserve a concrete unseen card and fill the decks.

    Deterministic: the unseen pool of a tier is walked in ascending card id,
    blind reserves are taken from the *end* and the deck fill from the start,
    so the two never collide and the same payload always hydrates identically.
    """
    seen = set()
    for row in state.board:
        seen.update(row)
    for p in state.players:
        seen.update(p.cards)
        seen.update(cid for cid in p.reserved if cid >= 0)
    pools = [[cid for cid in CARDS_BY_TIER[t] if cid not in seen]
             for t in range(3)]

    for player_index, slot, tier0 in hidden_slots:
        pool = pools[tier0]
        if not pool:
            raise HydrationError(
                f"players[{player_index}].reserved[{slot}]: no unseen tier-"
                f"{tier0 + 1} card is left to stand in for the blind reserve")
        state.players[player_index].reserved[slot] = pool.pop()

    decks: List[List[int]] = []
    for t in range(3):
        need = int(deck_counts[t])
        pool = pools[t]
        if len(pool) < need:
            raise HydrationError(
                f"deckCounts[{t}] is {need} but only {len(pool)} tier-{t + 1} "
                f"cards are unseen — the payload is inconsistent")
        # Surplus is normal: an INDIVIDUAL resign discards a tableau, and those
        # ids become unplaceable.  Keep a deterministic prefix.
        decks.append(pool[:need])
    state.decks = decks
    state.deck_counts = [len(decks[0]), len(decks[1]), len(decks[2])]
    if state.deck_counts != [int(v) for v in deck_counts]:      # pragma: no cover
        raise HydrationError(
            f"deck fill produced {state.deck_counts}, payload says "
            f"{list(deck_counts)}")


def _check_tokens(state: E.GameState) -> None:
    """Tokens are conserved by every branch of ``gameLogic.js``."""
    cfg = state.config
    expect = [cfg["tokensPerColor"]] * 5 + [cfg["wildTokens"]]
    total = list(state.gems)
    for p in state.players:
        for i in range(6):
            total[i] += p.gems[i]
    if total != expect:
        raise HydrationError(
            f"token conservation failed: bank + hands = {total}, the setup "
            f"deals {expect}")


# ── engine action → wire action ───────────────────────────────────────────

def wire_action_kind(action_index: int) -> str:
    """``"MOVE"`` or ``"TILE"`` — which request kind an action answers."""
    return "TILE" if action_index >= CHOOSE_TILE_START else "MOVE"


def to_wire(state: E.GameState, action_index: int,
            kind: str = "MOVE") -> Dict[str, Any]:
    """The ``docs/AI_BRIDGE.md`` §1 worker action for a 65-way action index.

    Built on :func:`splendor_ai.rules.engine.to_protocol` so the mapping can
    never drift from the engine's own idea of what an index means; the bridge
    re-inserts the ``ENTER_RESERVE`` message the client sends before a reserve.
    """
    if not isinstance(action_index, int) or isinstance(action_index, bool):
        raise HydrationError(f"action index {action_index!r} is not an int")
    if not 0 <= action_index < NUM_ACTIONS:
        raise HydrationError(f"action index {action_index} outside 0..64")
    if kind == "TILE" and action_index < CHOOSE_TILE_START:
        raise HydrationError(
            f"a TILE request only accepts CHOOSE_TILE (60..64), got "
            f"{action_index}")
    if kind != "TILE" and action_index >= CHOOSE_TILE_START:
        raise HydrationError(
            f"CHOOSE_TILE ({action_index}) is only legal for a TILE request")

    protocol = E.to_protocol(state, action_index)
    last = protocol[-1]                       # ENTER_RESERVE is the prefix
    kind_ = last["type"]
    if kind_ == "TAKE_GEMS_CONFIRMED":
        return {"type": "TAKE_GEMS", "colors": list(last["colors"])}
    if kind_ == "RESERVE_CARD":
        return {"type": "RESERVE_CARD", "cardId": int(last["cardId"])}
    if kind_ == "RESERVE_FROM_DECK":
        return {"type": "RESERVE_FROM_DECK", "tier": int(last["tier"])}
    if kind_ == "BUY_CARD":
        return {"type": "BUY_CARD", "cardId": int(last["cardId"]),
                "source": last["source"]}
    if kind_ == "CHOOSE_TILE":
        return {"type": "CHOOSE_TILE", "tileId": int(last["tileId"])}
    raise HydrationError(f"unmapped protocol action {last!r}")  # pragma: no cover
