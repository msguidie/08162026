"""Unit tests for ``splendor_ai/worker/adapter.py`` (and the agent's ladder).

The oracle is a Python transcription of the three JS functions the bridge
composes to build a request payload — ``clientView`` and
``clientViewForPlayer`` in ``server/gameLogic.js`` and ``buildObservation`` in
``server/aiBridge.js``.  Keeping the mimic *here* (rather than importing
something from the worker) is deliberate: the test must be able to disagree
with the code under test.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

from splendor_ai.rules import engine as E
from splendor_ai.rules.actions import (
    ACTION_TABLE, CHOOSE_TILE_START, KIND_BUY_BOARD, KIND_BUY_RESERVED,
    KIND_CHOOSE_TILE, KIND_RESERVE_BOARD, KIND_RESERVE_DECK, KIND_TAKE,
    NUM_ACTIONS, RESERVE_DECK_START,
)
from splendor_ai.rules.cards import (
    CARD_COST, CARD_POINTS, CARD_REWARD, CARD_TIER, CARDS_BY_TIER,
    TILE_POINTS, TILE_REQ,
)
from splendor_ai.worker.adapter import (
    HydrationError, hydrate, payload_mode_key, to_wire,
)

# ── the mimic of server/aiBridge.js buildObservation ──────────────────────

_JS_MODE = {E.MODE_INDIVIDUAL: "INDIVIDUAL", E.MODE_TEAM: "TEAM",
            E.MODE_ONE_V_TWO: "ONE_V_TWO"}


def js_card(card_id: int) -> Dict[str, Any]:
    """``ALL_CARDS[id]`` as ``gameLogic.js`` builds it."""
    return {"id": card_id, "tier": CARD_TIER[card_id],
            "reward": CARD_REWARD[card_id], "points": CARD_POINTS[card_id],
            "cost": list(CARD_COST[card_id])}


def js_tile(tile_id: int) -> Dict[str, Any]:
    return {"id": tile_id, "points": TILE_POINTS[tile_id],
            "requirement": list(TILE_REQ[tile_id])}


def build_observation(state: E.GameState, seat: int,
                      known_reserved: Optional[Sequence[int]] = None,
                      request_id: str = "ai-1",
                      room_id: str = "room-1") -> Dict[str, Any]:
    """``clientView`` → ``clientViewForPlayer`` → ``buildObservation``.

    ``knownReserved`` in the server is the set of card ids that were reserved
    **from the board** (``onActionResult`` records every ``RESERVE_CARD``);
    the engine keeps the same fact per slot in ``reserved_public``.
    """
    if known_reserved is None:
        known_reserved = [cid for p in state.players
                          for cid, public in zip(p.reserved, p.reserved_public)
                          if public]
    known = set(known_reserved)

    players: List[Dict[str, Any]] = []
    for i, p in enumerate(state.players):
        entry: Dict[str, Any] = {
            "username": p.username,
            "gems": list(p.gems),
            "cards": [js_card(c) for c in p.cards],
            "bonusTiles": [js_tile(t) for t in p.tiles],
            "score": p.score,
            "avatarSeed": p.avatar_seed,
        }
        if p.team_id is not None:
            entry["teamId"] = p.team_id
        if i == seat:
            # clientViewForPlayer keeps the viewer's own reserves intact.
            entry["reserved"] = [js_card(c) for c in p.reserved]
        else:
            # buildObservation replaces the "all zeros" hidden card of
            # clientViewForPlayer with {id, tier, hidden, known}.
            entry["reserved"] = [
                {"id": c if c in known else -1, "tier": CARD_TIER[c],
                 "hidden": True, "known": c in known}
                for c in p.reserved]
        players.append(entry)

    view: Dict[str, Any] = {
        "phase": state.phase,
        "board": [[js_card(c) for c in row] for row in state.board],
        "deckCounts": list(state.deck_counts),
        "gems": list(state.gems),
        "bonusTiles": [js_tile(t) for t in state.tiles],
        "players": players,
        "currentPlayerIndex": state.current_player,
        "roundStartPlayer": state.round_start_player,
        "turnAction": ({"type": state.turn_action} if state.turn_action
                       else None),
        "finalRoundTriggeredBy": state.final_round_triggered_by,
        "turnNumber": state.turn_number,
        "numPlayers": state.num_players,
        "config": dict(state.config),
        "resignedPlayers": list(state.resigned),
        "gameMode": _JS_MODE[state.mode],
        "teamLayout": state.team_layout,
        "teams": [dict(t) for t in state.teams],
        "gameResult": (dict(state.game_result) if state.game_result else None),
        "timeControl": None,
    }
    pending = list(state.pending_tile_choice or []) or None
    if pending:
        view["_pendingTileChoice"] = pending          # clientView spreads it

    return {
        "requestId": request_id,
        "roomId": room_id,
        "playerIndex": seat,
        "kind": "TILE" if pending else "MOVE",
        "deadlineMs": 0,
        "state": view,
        "knownReserved": sorted(known),
        "pendingTileChoice": pending,
    }


def public_signature(state: E.GameState, seat: int) -> Dict[str, Any]:
    """Everything ``seat`` is allowed to know — the round-trip invariant.

    Another seat's blind reserve is represented by ``None``: its identity is
    *not* part of the information set, so hydration is free to invent one.
    """
    return {
        "board": [list(row) for row in state.board],
        "deck_counts": list(state.deck_counts),
        "gems": list(state.gems),
        "tiles": list(state.tiles),
        "current_player": state.current_player,
        "round_start_player": state.round_start_player,
        "turn_number": state.turn_number,
        "num_players": state.num_players,
        "mode": state.mode,
        "team_layout": state.team_layout,
        "phase": state.phase,
        "resigned": list(state.resigned),
        "final_round_triggered_by": state.final_round_triggered_by,
        "turn_action": state.turn_action,
        "pending_tile_choice": state.pending_tile_choice,
        "config": dict(state.config),
        "teams": [dict(t) for t in state.teams],
        "players": [{
            "gems": list(p.gems),
            "cards": list(p.cards),
            "tiles": list(p.tiles),
            "score": p.score,
            "discount": list(p.discount),
            "team_id": p.team_id,
            "reserved_public": list(p.reserved_public),
            "reserved": (list(p.reserved) if i == seat else
                         [c if public else None
                          for c, public in zip(p.reserved, p.reserved_public)]),
        } for i, p in enumerate(state.players)],
    }


CONFIGS = [
    (2, E.MODE_INDIVIDUAL, None),
    (3, E.MODE_INDIVIDUAL, None),
    (4, E.MODE_INDIVIDUAL, None),
    (3, E.MODE_ONE_V_TWO, None),
    (4, E.MODE_TEAM, "ADJACENT"),
    (4, E.MODE_TEAM, "OPPOSITE"),
]


def walk_games(seeds=range(3), max_plies=400):
    """Yield ``(state, seat)`` for every position of a few random games."""
    for num_players, mode, layout in CONFIGS:
        for seed in seeds:
            rng = random.Random(seed)
            state = E.new_game(num_players, mode, layout, rng=rng)
            for _ in range(max_plies):
                if state.phase != E.PHASE_PLAYING:
                    break
                legal = E.legal_actions(state)
                if not legal:
                    E.resign(state, state.current_player)
                    continue
                yield state, state.current_player
                E.apply(state, legal[rng.randrange(len(legal))])


# ── round trip ────────────────────────────────────────────────────────────

def test_hydrate_round_trips_the_public_view():
    """Hydration reproduces the information set exactly, in every mode."""
    positions = 0
    tile_positions = 0
    hidden_positions = 0
    resigned_positions = 0
    for state, seat in walk_games(seeds=range(6)):
        payload = build_observation(state, seat)
        rebuilt, rebuilt_seat = hydrate(payload)
        assert rebuilt_seat == seat
        assert public_signature(rebuilt, seat) == public_signature(state, seat)
        positions += 1
        if state.pending_tile_choice:
            tile_positions += 1
        if any(not all(p.reserved_public) for i, p in enumerate(state.players)
               if i != seat):
            hidden_positions += 1
        if state.resigned:
            resigned_positions += 1
    assert positions > 3000, positions
    assert tile_positions > 0, "no multi-noble position was exercised"
    assert hidden_positions > 0, "no blind reserve was exercised"
    assert resigned_positions > 0, "no post-resign position was exercised"


def test_legal_masks_are_identical():
    """The mask is the contract: a hydrated state must offer the same moves."""
    checked = 0
    for state, seat in walk_games(seeds=range(2)):
        rebuilt, _ = hydrate(build_observation(state, seat))
        assert E.legal_mask(rebuilt) == E.legal_mask(state)
        assert E.is_stuck(rebuilt) == E.is_stuck(state)
        assert any(E.legal_mask(rebuilt)) or E.is_stuck(state)
        checked += 1
    assert checked > 1000


def test_deck_counts_and_no_duplicate_cards():
    for state, seat in walk_games(seeds=range(2)):
        rebuilt, _ = hydrate(build_observation(state, seat))
        assert rebuilt.deck_counts == state.deck_counts
        assert [len(d) for d in rebuilt.decks] == state.deck_counts
        placed: List[int] = []
        for row in rebuilt.board:
            placed.extend(row)
        for deck in rebuilt.decks:
            placed.extend(deck)
        for p in rebuilt.players:
            placed.extend(p.cards)
            placed.extend(p.reserved)
        assert len(placed) == len(set(placed)), "a card id was placed twice"


# ── hidden information ────────────────────────────────────────────────────

def _state_with_blind_reserve():
    """Seat 1 reserves blind from tier 3; seat 0 reserves from the board."""
    state = E.new_game(2, rng=random.Random(7))
    E.apply(state, RESERVE_DECK_START + 2)              # seat 0, blind
    E.apply(state, RESERVE_DECK_START + 1)              # seat 1, blind
    E.apply(state, 30)                                  # seat 0, board tier 1
    return state


def test_blind_reserve_becomes_a_deterministic_unseen_placeholder():
    state = _state_with_blind_reserve()
    seat = state.current_player                          # seat 1
    payload = build_observation(state, seat)

    other = payload["state"]["players"][0]["reserved"]
    assert [slot["hidden"] for slot in other] == [True, True]
    assert [slot["known"] for slot in other] == [False, True]
    assert other[0]["id"] == -1 and other[0]["tier"] == 3
    assert other[1]["id"] >= 0

    rebuilt, _ = hydrate(payload)
    invented, public_card = rebuilt.players[0].reserved
    assert rebuilt.players[0].reserved_public == [False, True]
    assert public_card == state.players[0].reserved[1]
    assert invented != state.players[0].reserved[0], (
        "the real identity must not leak through the payload")
    assert CARD_TIER[invented] == 3
    # …and it is a card the seat has genuinely not seen.
    seen = {c for row in state.board for c in row}
    seen |= {c for p in state.players for c in p.cards}
    seen |= set(rebuilt.players[seat].reserved)
    assert invented not in seen

    # Deterministic: the same payload hydrates to the same placeholder.
    again, _ = hydrate(payload)
    assert again.players[0].reserved == rebuilt.players[0].reserved


def test_determinize_resamples_the_placeholder():
    """The invented id is a scaffold; the search replaces it every universe."""
    from splendor_ai.search.determinize import determinize, universe_rng

    state = _state_with_blind_reserve()
    seat = state.current_player
    rebuilt, _ = hydrate(build_observation(state, seat))
    drawn = {determinize(rebuilt, seat, universe_rng(11, k)).players[0].reserved[0]
             for k in range(24)}
    assert len(drawn) > 1, "the hidden slot never changed across universes"
    assert all(CARD_TIER[c] == 3 for c in drawn)
    assert all(len(d) == c for d, c in
               zip(determinize(rebuilt, seat, universe_rng(11, 0)).decks,
                   state.deck_counts))


def test_own_blind_reserve_stays_private_from_the_other_side():
    """``knownReserved`` is what recovers our own ``reserved_public`` flags.

    Nothing on the wire says whether *our* reserved card came off the board or
    off a deck, but a search leaf where an opponent moves encodes our reserves
    through the other-seat block — where a blind card must stay a tier-only
    sentinel.  ``knownReserved`` is the only record of the difference.
    """
    state = E.new_game(2, rng=random.Random(7))
    first = state.current_player
    E.apply(state, 0)                                    # opponent takes 3 gems
    E.apply(state, RESERVE_DECK_START + 1)               # we reserve blind
    E.apply(state, 6)                                    # opponent takes 3 more
    seat = state.current_player
    assert seat != first and not state.players[first].reserved
    assert state.players[seat].reserved_public == [False]

    payload = build_observation(state, seat)
    assert payload["knownReserved"] == []
    rebuilt, _ = hydrate(payload)
    assert rebuilt.players[seat].reserved == state.players[seat].reserved
    assert rebuilt.players[seat].reserved_public == [False]

    # The opponent's view of the hydrated position is bit-identical.
    from splendor_ai.encode import encode
    assert np.array_equal(encode(rebuilt, first), encode(state, first))

    # Had the card been taken off the board, the same view would differ.
    leaked = rebuilt.clone()
    leaked.players[seat].reserved_public = [True]
    assert not np.array_equal(encode(leaked, first), encode(state, first))


def test_known_reserved_marks_a_board_reserve_public():
    state = E.new_game(2, rng=random.Random(3))
    other = state.current_player
    E.apply(state, 30)                                   # reserve board tier 1
    seat = state.current_player
    payload = build_observation(state, seat)
    assert payload["state"]["players"][other]["reserved"][0]["known"] is True

    rebuilt, _ = hydrate(payload)
    assert rebuilt.players[other].reserved == state.players[other].reserved
    assert rebuilt.players[other].reserved_public == [True]

    # Drop the card from knownReserved and the bridge hides it: hydration must
    # then treat the slot as blind.
    payload["knownReserved"] = []
    payload["state"]["players"][other]["reserved"] = [
        {"id": -1, "tier": 1, "hidden": True, "known": False}]
    hidden, _ = hydrate(payload)
    assert hidden.players[other].reserved_public == [False]
    assert CARD_TIER[hidden.players[other].reserved[0]] == 1


# ── consistency validation ────────────────────────────────────────────────

def _fresh_payload(seed: int = 5, plies: int = 24):
    rng = random.Random(seed)
    state = E.new_game(3, rng=rng)
    for _ in range(plies):
        legal = E.legal_actions(state)
        E.apply(state, legal[rng.randrange(len(legal))])
    return state, build_observation(state, state.current_player)


def test_corrupted_score_is_rejected():
    _, payload = _fresh_payload()
    payload["state"]["players"][0]["score"] += 3
    with pytest.raises(HydrationError, match="score"):
        hydrate(payload)
    hydrate(payload, validate=False)                     # best-effort still works


def test_corrupted_card_table_is_rejected():
    _, payload = _fresh_payload()
    payload["state"]["board"][1][0]["cost"] = [9, 9, 9, 9, 9]
    with pytest.raises(HydrationError, match="cost"):
        hydrate(payload)
    _, payload = _fresh_payload()
    payload["state"]["board"][0][1]["reward"] = (
        payload["state"]["board"][0][1]["reward"] + 1) % 5
    with pytest.raises(HydrationError, match="reward"):
        hydrate(payload)


def test_token_conservation_is_checked():
    _, payload = _fresh_payload()
    payload["state"]["gems"][2] += 1
    with pytest.raises(HydrationError, match="token conservation"):
        hydrate(payload)


def test_duplicate_card_is_rejected():
    _, payload = _fresh_payload()
    duplicate = payload["state"]["board"][0][0]
    payload["state"]["players"][0]["cards"].append(dict(duplicate))
    payload["state"]["players"][0]["score"] += duplicate["points"]
    with pytest.raises(HydrationError, match="appears twice"):
        hydrate(payload)


def test_impossible_deck_count_is_rejected():
    _, payload = _fresh_payload()
    payload["state"]["deckCounts"][0] = 40
    with pytest.raises(HydrationError, match="only .* are unseen"):
        hydrate(payload)


def test_stale_request_seat_is_rejected():
    _, payload = _fresh_payload()
    payload["playerIndex"] = (payload["playerIndex"] + 1) % 3
    with pytest.raises(HydrationError, match="currentPlayerIndex"):
        hydrate(payload)


def test_half_finished_human_turn_is_rejected():
    _, payload = _fresh_payload()
    payload["state"]["turnAction"] = {"type": "RESERVE", "goldTaken": True}
    with pytest.raises(HydrationError, match="half-finished"):
        hydrate(payload)
    rebuilt, _ = hydrate(payload, validate=False)
    assert rebuilt.turn_action is None


def test_a_valid_payload_needs_no_leniency():
    """Whatever ``validate`` says, a real payload hydrates the same way."""
    for state, seat in walk_games(seeds=range(1), max_plies=60):
        payload = build_observation(state, seat)
        strict, _ = hydrate(payload, validate=True)
        loose, _ = hydrate(payload, validate=False)
        assert public_signature(strict, seat) == public_signature(loose, seat)


# ── the wire format ───────────────────────────────────────────────────────

def translate_worker_action(action: Dict[str, Any], kind: str = "MOVE"):
    """``aiBridge.js translateWorkerAction`` — the server's own table."""
    kind_of = action.get("type")
    colors = action.get("colors")
    if kind_of == "TAKE_GEMS":
        if kind == "TILE" or not (isinstance(colors, list)
                                  and 1 <= len(colors) <= 3
                                  and all(isinstance(c, int) and 0 <= c <= 4
                                          for c in colors)):
            return None
        return [{"type": "TAKE_GEMS_CONFIRMED", "colors": list(colors)}]
    if kind_of == "RESERVE_CARD":
        if kind == "TILE" or not isinstance(action.get("cardId"), int):
            return None
        return [{"type": "ENTER_RESERVE"},
                {"type": "RESERVE_CARD", "cardId": action["cardId"]}]
    if kind_of == "RESERVE_FROM_DECK":
        if kind == "TILE" or action.get("tier") not in (1, 2, 3):
            return None
        return [{"type": "ENTER_RESERVE"},
                {"type": "RESERVE_FROM_DECK", "tier": action["tier"]}]
    if kind_of == "BUY_CARD":
        if kind == "TILE" or not isinstance(action.get("cardId"), int):
            return None
        if action.get("source") not in ("board", "reserved"):
            return None
        return [{"type": "BUY_CARD", "cardId": action["cardId"],
                 "source": action["source"]}]
    if kind_of == "CHOOSE_TILE":
        if not isinstance(action.get("tileId"), int):
            return None
        return [{"type": "CHOOSE_TILE", "tileId": action["tileId"]}]
    return None


_TILE_STATE: List[Any] = []


def tile_choice_position():
    """The first position of a scripted game where two nobles qualify at once.

    Multi-noble choices are rare (16 in the 16k-game validation run), so the
    search is over fixed seeds and the result is cached: deterministic, and it
    fails loudly rather than silently skipping the ``CHOOSE_TILE`` coverage.
    """
    if _TILE_STATE:
        return _TILE_STATE[0]
    for num_players, mode, layout in CONFIGS:
        for seed in range(60):
            rng = random.Random(seed)
            state = E.new_game(num_players, mode, layout, rng=rng)
            for _ in range(400):
                if state.phase != E.PHASE_PLAYING:
                    break
                if state.pending_tile_choice and state.turn_action == E.TA_BUY:
                    _TILE_STATE.append(state)
                    return state
                legal = E.legal_actions(state)
                if not legal:
                    E.resign(state, state.current_player)
                    continue
                E.apply(state, legal[rng.randrange(len(legal))])
    raise AssertionError("no multi-noble position found in the scripted games")


def test_to_wire_matches_the_servers_translation_table():
    """Every action kind survives worker → bridge → ``processAction`` intact.

    The bridge turns our action back into protocol messages; those must be the
    ones ``engine.to_protocol`` (validated bit-for-bit against the server in
    ``tests/test_replay.py``) produces for the same index.
    """
    seen_kinds = set()
    for state, seat in walk_games(seeds=range(2), max_plies=120):
        rebuilt, _ = hydrate(build_observation(state, seat))
        kind = "TILE" if state.pending_tile_choice and state.turn_action else "MOVE"
        for action in E.legal_actions(state):
            wire = to_wire(rebuilt, action, kind)
            assert translate_worker_action(wire, kind) == \
                E.to_protocol(state, action), (action, wire)
            seen_kinds.add(ACTION_TABLE[action][0])

    tile_state = tile_choice_position()
    tile_seat = tile_state.current_player
    tile_rebuilt, _ = hydrate(build_observation(tile_state, tile_seat))
    legal_tiles = E.legal_actions(tile_state)
    assert legal_tiles and all(a >= CHOOSE_TILE_START for a in legal_tiles)
    for action in legal_tiles:
        wire = to_wire(tile_rebuilt, action, "TILE")
        assert wire["tileId"] in tile_state.pending_tile_choice
        assert translate_worker_action(wire, "TILE") == \
            E.to_protocol(tile_state, action)
        seen_kinds.add(ACTION_TABLE[action][0])

    assert seen_kinds == {KIND_TAKE, KIND_RESERVE_BOARD, KIND_RESERVE_DECK,
                          KIND_BUY_BOARD, KIND_BUY_RESERVED, KIND_CHOOSE_TILE}


def test_to_wire_enforces_the_request_kind():
    state, payload = _fresh_payload()
    rebuilt, _ = hydrate(payload)
    with pytest.raises(HydrationError, match="TILE request"):
        to_wire(rebuilt, 0, "TILE")
    with pytest.raises(HydrationError, match="only legal for a TILE"):
        to_wire(rebuilt, CHOOSE_TILE_START, "MOVE")
    with pytest.raises(HydrationError):
        to_wire(rebuilt, NUM_ACTIONS, "MOVE")


def test_mode_key_covers_every_mode():
    for (num_players, mode, layout), expected in zip(
            CONFIGS, ["ind2", "ind3", "ind4", "ovt", "team", "team"]):
        state = E.new_game(num_players, mode, layout, rng=random.Random(0))
        payload = build_observation(state, state.current_player)
        assert payload_mode_key(payload) == expected


# ── the agent's ladder and safety nets ────────────────────────────────────

def _agent(**env):
    from splendor_ai.worker.agent import MoveAgent
    from splendor_ai.worker.config import load_config
    settings = {"MODEL_DIR": "/nonexistent-model-dir", "DEVICE": "cpu"}
    settings.update({k: str(v) for k, v in env.items()})
    return MoveAgent(load_config(env=settings, use_dotenv=False))


def test_greedy_ladder_runs_without_any_checkpoint():
    """No model at all → the worker still answers, on the greedy rung."""
    agent = _agent()
    levels = set()
    for state, seat in walk_games(seeds=range(1), max_plies=80):
        decision = agent.decide(build_observation(state, seat))
        levels.add(decision.level)
        assert decision.action["type"] != "NONE"
        assert E.legal_mask(state)[decision.action_index]
        assert to_wire(state, decision.action_index, decision.kind) == \
            decision.action
    assert levels == {"greedy"}


def test_a_stuck_seat_answers_none():
    """The variant has no pass: the honest answer is NONE and the server
    resigns the seat."""
    state = E.new_game(2, rng=random.Random(0))
    # Hand-build the classic trap: 10 tokens, 3 reserved, nothing affordable.
    me = state.players[state.current_player]
    me.gems = [2, 2, 2, 2, 2, 0]
    me.reserved = [state.decks[2][0], state.decks[2][1], state.decks[2][2]]
    me.reserved_public = [False, False, False]
    state.decks[2] = state.decks[2][3:]
    state.deck_counts[2] = len(state.decks[2])
    state.gems = [g - 2 for g in state.gems[:5]] + [state.gems[5]]
    state.board = [[], [], []]
    state.decks = [[], [], state.decks[2]]
    state.deck_counts = [0, 0, len(state.decks[2])]
    assert E.is_stuck(state)

    payload = build_observation(state, state.current_player)
    decision = _agent().decide(payload)
    assert decision.action == {"type": "NONE"}
    assert decision.level == "none"


def _mixed_trap_position():
    """8 tokens, 3 reserved, an empty board and empty decks.

    One reserved card is a single Indigo short, so exactly the takes that
    include Indigo keep a move available next turn; every other take reaches
    the 10-token cap with nothing affordable, no reserve slot and no card to
    buy — the position the variant has no pass for.
    """
    state = E.new_game(2, rng=random.Random(2))
    seat = state.current_player
    me = state.players[seat]
    target = 0                                # cost [1, 1, 0, 1, 1], reward 0
    assert list(CARD_COST[target]) == [1, 1, 0, 1, 1]
    expensive = [c for c in CARDS_BY_TIER[2] if sum(CARD_COST[c]) >= 12][:2]
    me.cards = []
    me.discount = [0, 0, 0, 0, 0]
    me.score = 0
    me.reserved = [target] + expensive
    me.reserved_public = [True, True, True]
    me.gems = [0, 2, 4, 1, 1, 0]              # 8 tokens, one Indigo short
    state.players[1 - seat].gems = [0, 0, 0, 0, 0, 0]
    state.players[1 - seat].cards = []
    state.players[1 - seat].reserved = []
    state.players[1 - seat].reserved_public = []
    state.players[1 - seat].score = 0
    state.players[1 - seat].discount = [0, 0, 0, 0, 0]
    state.board = [[], [], []]
    state.decks = [[], [], []]
    state.deck_counts = [0, 0, 0]
    tokens = state.config["tokensPerColor"]
    state.gems = [tokens - g for g in me.gems[:5]] + [state.config["wildTokens"]]
    return state, seat


def test_the_stuck_filter_avoids_walking_into_the_trap():
    from splendor_ai.worker.agent import self_stuck_after

    state, seat = _mixed_trap_position()
    mask = E.legal_mask(state)
    legal = [a for a in range(NUM_ACTIONS) if mask[a]]
    traps = [a for a in legal if self_stuck_after(state, seat, a)]
    safe = [a for a in legal if a not in traps]
    assert traps and safe, (legal, traps)

    decision = _agent().decide(build_observation(state, seat))
    assert decision.action_index in safe, (decision.action, decision.notes)
    assert any("stuck filter" in note for note in decision.notes), decision.notes


def test_the_stuck_filter_gives_up_gracefully_when_every_move_traps():
    """A position where *all* moves trap must still produce a move."""
    from splendor_ai.worker.agent import self_stuck_after

    state, seat = _mixed_trap_position()
    state.players[seat].gems = [0, 2, 3, 2, 1, 0]        # nothing rescues us
    state.players[seat].reserved[0] = [c for c in CARDS_BY_TIER[2]
                                       if sum(CARD_COST[c]) >= 12][2]
    tokens = state.config["tokensPerColor"]
    state.gems = [tokens - g for g in state.players[seat].gems[:5]] \
        + [state.config["wildTokens"]]
    mask = E.legal_mask(state)
    legal = [a for a in range(NUM_ACTIONS) if mask[a]]
    assert legal and all(self_stuck_after(state, seat, a) for a in legal)

    decision = _agent().decide(build_observation(state, seat))
    assert decision.action_index in legal
    assert decision.action["type"] != "NONE"
    assert any("self-trap" in note for note in decision.notes), decision.notes


def test_search_rung_is_used_when_a_checkpoint_exists(tmp_path):
    torch = pytest.importorskip("torch")
    from splendor_ai.model import SMOKE_CONFIG, SplendorNet, save_checkpoint

    torch.manual_seed(0)
    save_checkpoint(str(tmp_path / "shared.pt"), SplendorNet(SMOKE_CONFIG))
    agent = _agent(MODEL_DIR=str(tmp_path), SEARCH_SIMS=16, TIME_BUDGET_MS=200,
                   HARD_BUDGET_MS=500, UNIVERSES=2, ROOT_ENSEMBLE=1)

    state = E.new_game(2, rng=random.Random(4))
    for _ in range(10):
        E.apply(state, E.legal_actions(state)[0])
    decision = agent.decide(build_observation(state, state.current_player))
    assert decision.level == "search"
    assert 0 < decision.sims <= 16
    assert decision.root_value is not None and len(decision.root_value) == 4
    assert E.legal_mask(state)[decision.action_index]
    assert decision.info["level"] == "search" and "value" in decision.info


def test_the_deadline_shrinks_the_budget(tmp_path):
    """A deadline closer than HARD_BUDGET_MS wins."""
    import time
    pytest.importorskip("torch")
    from splendor_ai.model import SMOKE_CONFIG, SplendorNet, save_checkpoint

    save_checkpoint(str(tmp_path / "shared.pt"), SplendorNet(SMOKE_CONFIG))
    agent = _agent(MODEL_DIR=str(tmp_path), SEARCH_SIMS=100000,
                   TIME_BUDGET_MS=60000, HARD_BUDGET_MS=60000,
                   UNIVERSES=2, DEADLINE_MARGIN_MS=50)
    state = E.new_game(2, rng=random.Random(4))
    payload = build_observation(state, state.current_player)
    payload["deadlineMs"] = time.time() * 1000.0 + 400
    started = time.monotonic()
    decision = agent.decide(payload)
    elapsed = (time.monotonic() - started) * 1000.0
    assert decision.level == "search"
    assert elapsed < 2000, elapsed
