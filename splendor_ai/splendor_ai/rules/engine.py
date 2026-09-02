"""Bit-exact Python port of the rules in ``server/gameLogic.js``.

Every branch below mirrors a specific branch of the Node implementation; the
comments name the JS function so a reviewer can diff the two side by side.
The hot path (``legal_mask`` / ``apply``) uses only plain Python ints and
lists — numpy is deliberately absent from this module.

Subtleties that a naive port gets wrong (all reproduced here):

* ``advanceTurn`` runs after **every** action, so a noble left over from an
  earlier multi-noble choice is auto-claimed by the next gem take / reserve.
* ``CHOOSE_TILE`` requires ``state.turnAction.type === 'BUY'``.  After a gem
  take or a reserve ``turnAction`` is ``null``, so if that action leaves two or
  more qualifying nobles the pending choice is *orphaned*: no tile choice is
  accepted, and the ordinary actions stay legal (the turn does not advance).
* ``payForCard`` computes discounts **before** the bought card is added.
* ``ENTER_RESERVE`` grants gold only when ``gold > 0`` *and* the player holds
  fewer than 10 tokens; the gold is taken before the card is chosen.
* Board refills append to the **end** of the row, so slots shift left.
* ``finishTurn``'s GAME_OVER branches ``return`` early and therefore leave
  ``_pendingTileChoice`` and ``turnNumber`` untouched.
* TEAM final rounds are revocable, ONE_V_TWO final rounds are not, and
  INDIVIDUAL game over leaves ``gameResult === null``.
* ``getTeamStats`` uses ``-Infinity`` for a team with fewer than two members,
  and ``-Infinity >= -Infinity`` is true.
* ``calculateRatingChanges`` treats an **empty** ``winningTeamIds`` array as
  truthy (JS), so a 1v2 that ends with nobody qualifying pays 3 to everyone.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cards import (
    CARD_COST, CARD_COST_NZ, CARD_POINTS, CARD_REWARD, CARD_TIER,
    CARDS_BY_TIER, TILE_POINTS, TILE_REQ,
)
from .actions import (
    ACTION_RESIGN, ACTION_TIMEOUT, BUY_BOARD_START, BUY_RESERVED_START,
    CHOOSE_TILE_START, MAX_BOARD_SLOTS, MAX_RESERVED, NUM_ACTIONS,
    NUM_TAKE_ACTIONS, PAIRS, RESERVE_BOARD_START, RESERVE_DECK_START,
    TAKE1_START, TAKE2SAME_START, TAKE2_START, TAKE3_START, TAKE_PATTERNS,
    TRIPLES, take_index,
)

NEG_INF = float("-inf")

MODE_INDIVIDUAL = "INDIVIDUAL"
MODE_TEAM = "TEAM"
MODE_ONE_V_TWO = "ONE_V_TWO"

PHASE_PLAYING = "PLAYING"
PHASE_GAME_OVER = "GAME_OVER"

TA_TAKE_GEMS = "TAKE_GEMS"
TA_RESERVE = "RESERVE"
TA_BUY = "BUY"


# ── config (createInitialGameState) ───────────────────────────────────────

def make_config(num_players: int) -> Dict[str, int]:
    return {
        "tokensPerColor": 4 if num_players <= 2 else (5 if num_players == 3 else 7),
        "wildTokens": 5,
        "revealedTiles": num_players + 1,
        "cardsPerRow": 4,
        "maxTokensInHand": 10,
        "maxReserved": 3,
        "winThreshold": 15,
        "take2MinStack": 4,
    }


# ── player ────────────────────────────────────────────────────────────────

class PlayerState:
    """One seat.  ``discount`` is a cache of ``getDiscount(player)``."""

    __slots__ = ("gems", "cards", "reserved", "reserved_public", "tiles",
                 "score", "discount", "team_id", "username", "avatar_seed")

    def __init__(self, username: str = "", team_id: Optional[int] = None,
                 avatar_seed: int = 0):
        self.gems: List[int] = [0, 0, 0, 0, 0, 0]
        self.cards: List[int] = []
        self.reserved: List[int] = []
        # Parallel to ``reserved``: True when the card was reserved face-up
        # from the board (public knowledge), False when taken from a deck.
        self.reserved_public: List[bool] = []
        self.tiles: List[int] = []
        self.score: int = 0
        self.discount: List[int] = [0, 0, 0, 0, 0]
        self.team_id: Optional[int] = team_id
        self.username: str = username
        self.avatar_seed: int = avatar_seed

    def clone(self) -> "PlayerState":
        p = PlayerState.__new__(PlayerState)
        p.gems = self.gems[:]
        p.cards = self.cards[:]
        p.reserved = self.reserved[:]
        p.reserved_public = self.reserved_public[:]
        p.tiles = self.tiles[:]
        p.score = self.score
        p.discount = self.discount[:]
        p.team_id = self.team_id
        p.username = self.username
        p.avatar_seed = self.avatar_seed
        return p

    def total_gems(self) -> int:            # totalGems()
        g = self.gems
        return g[0] + g[1] + g[2] + g[3] + g[4] + g[5]

    def __repr__(self) -> str:              # pragma: no cover
        return (f"<Player {self.username!r} score={self.score} "
                f"gems={self.gems} cards={len(self.cards)} "
                f"reserved={self.reserved} tiles={self.tiles}>")


# ── state ─────────────────────────────────────────────────────────────────

class GameState:
    """Mutable game state; ``clone()`` is a cheap structural copy."""

    __slots__ = ("board", "decks", "deck_counts", "gems", "tiles", "players",
                 "current_player", "round_start_player", "turn_number",
                 "num_players", "mode", "team_layout", "teams", "resigned",
                 "final_round_triggered_by", "phase", "game_result",
                 "pending_tile_choice", "turn_action", "config", "last_event",
                 "tile_claimed")

    def __init__(self) -> None:
        self.board: List[List[int]] = [[], [], []]
        self.decks: List[List[int]] = [[], [], []]
        self.deck_counts: List[int] = [0, 0, 0]
        self.gems: List[int] = [0, 0, 0, 0, 0, 0]
        self.tiles: List[int] = []
        self.players: List[PlayerState] = []
        self.current_player: int = 0
        self.round_start_player: int = 0
        self.turn_number: int = 0
        self.num_players: int = 0
        self.mode: str = MODE_INDIVIDUAL
        self.team_layout: Optional[str] = None
        self.teams: List[Dict[str, Any]] = []
        self.resigned: List[int] = []
        self.final_round_triggered_by: Optional[int] = None
        self.phase: str = PHASE_PLAYING
        self.game_result: Optional[Dict[str, Any]] = None
        self.pending_tile_choice: Optional[List[int]] = None
        # ``state.turnAction`` — only ``None`` or ``'BUY'`` is ever observable
        # between complete turn actions, because ``apply`` performs a whole
        # turn action atomically.
        self.turn_action: Optional[str] = None
        self.config: Dict[str, int] = {}
        # Mirrors the server's ActionResult + the consumed ``_tileClaimed``.
        self.last_event: Optional[Dict[str, Any]] = None
        # ``state._tileClaimed`` before ``broadcastProcessedAction`` deletes it.
        self.tile_claimed: Optional[Dict[str, int]] = None

    # -- copying -----------------------------------------------------------
    def clone(self) -> "GameState":
        s = GameState.__new__(GameState)
        b = self.board
        s.board = [b[0][:], b[1][:], b[2][:]]
        d = self.decks
        s.decks = [d[0][:], d[1][:], d[2][:]]
        s.deck_counts = self.deck_counts[:]
        s.gems = self.gems[:]
        s.tiles = self.tiles[:]
        s.players = [p.clone() for p in self.players]
        s.current_player = self.current_player
        s.round_start_player = self.round_start_player
        s.turn_number = self.turn_number
        s.num_players = self.num_players
        s.mode = self.mode
        s.team_layout = self.team_layout
        s.teams = self.teams
        s.resigned = self.resigned[:]
        s.final_round_triggered_by = self.final_round_triggered_by
        s.phase = self.phase
        s.game_result = (dict(self.game_result)
                         if self.game_result is not None else None)
        s.pending_tile_choice = (self.pending_tile_choice[:]
                                 if self.pending_tile_choice is not None else None)
        s.turn_action = self.turn_action
        s.config = self.config
        s.last_event = self.last_event
        s.tile_claimed = self.tile_claimed
        return s

    # -- compact serialization ---------------------------------------------
    #
    # ``to_bytes()`` / ``from_bytes()`` are the replay-record format of
    # ``docs/AI_DESIGN.md`` §1.8: deterministic (no dict iteration, no
    # padding, no timestamps), compact (~150-250 bytes for a mid-game 4p
    # position) and lossless for every field that can influence the rest of
    # the game — deck ORDER included.
    #
    # ======  ===============================================================
    #  bytes   contents
    # ======  ===============================================================
    #   0      format version (:data:`GameState.BYTES_VERSION`)
    #   1      num_players
    #   2      mode(2 bits) | layout(2)<<2 | phase(1)<<4 | turn_action(2)<<5
    #   3      current_player
    #   4      round_start_player (255 = None)
    #   5-6    turn_number, big endian uint16
    #   7      final_round_triggered_by + 1 (0 = None)
    #   8+     resigned: count, then the seats in resignation order
    #   +6     gems[6]
    #   +      tiles: count, then tile ids
    #   +      pending_tile_choice: 255 = None, else count + tile ids
    #   +      game_result: 0 = None, else reason code (1 SCORE / 2 FORFEIT /
    #          3 other), forfeitingTeamId (255 = None), winningTeamIds
    #          (255 = absent, else count + ids)
    #   +      board:  3 x (count + card ids), tier 1..3
    #   +      decks:  3 x (count + card ids) in SERVER order (pop() = last)
    #   +      players: n x (gems[6], discount[5], score uint16,
    #          team_id (255 = None), cards (count + ids), reserved (count +
    #          ids + a public-flag bitmask byte), tiles (count + ids))
    # ======  ===============================================================
    #
    # Derived state is recomputed on load rather than stored: ``deck_counts``
    # (lengths of ``decks``), ``config`` (:func:`make_config`) and ``teams``
    # (from the per-player ``team_id``).  ``last_event`` / ``tile_claimed``
    # are transient action-result payloads consumed by the caller of
    # ``apply``; they are not part of the position and are reset to ``None``.
    # Cosmetic ``username`` / ``avatar_seed`` are likewise not stored.

    BYTES_VERSION = 1

    _MODE_CODES = (MODE_INDIVIDUAL, MODE_TEAM, MODE_ONE_V_TWO)
    _LAYOUT_CODES = (None, "ADJACENT", "OPPOSITE")
    _TA_CODES = (None, TA_BUY, TA_TAKE_GEMS, TA_RESERVE)
    _REASON_CODES = (None, "SCORE", "FORFEIT")

    def to_bytes(self) -> bytes:
        """Serialize the position; see the table above."""
        b = bytearray()
        ap = b.append
        ap(GameState.BYTES_VERSION)
        ap(self.num_players)
        ap(GameState._MODE_CODES.index(self.mode)
           | (GameState._LAYOUT_CODES.index(self.team_layout) << 2)
           | ((1 if self.phase == PHASE_GAME_OVER else 0) << 4)
           | (GameState._TA_CODES.index(self.turn_action) << 5))
        ap(self.current_player)
        ap(255 if self.round_start_player is None else self.round_start_player)
        tn = self.turn_number
        ap((tn >> 8) & 0xFF)
        ap(tn & 0xFF)
        frt = self.final_round_triggered_by
        ap(0 if frt is None else frt + 1)
        ap(len(self.resigned))
        b.extend(self.resigned)
        b.extend(self.gems)
        ap(len(self.tiles))
        b.extend(self.tiles)
        pending = self.pending_tile_choice
        if pending is None:
            ap(255)
        else:
            ap(len(pending))
            b.extend(pending)
        gr = self.game_result
        if gr is None:
            ap(0)
        else:
            reason = gr.get("reason")
            ap(GameState._REASON_CODES.index(reason)
               if reason in GameState._REASON_CODES else 3)
            forfeiting = gr.get("forfeitingTeamId")
            ap(255 if forfeiting is None else forfeiting)
            winning = gr.get("winningTeamIds")
            if winning is None:
                ap(255)
            else:
                ap(len(winning))
                b.extend(winning)
        for t in range(3):
            row = self.board[t]
            ap(len(row))
            b.extend(row)
        for t in range(3):
            deck = self.decks[t]
            ap(len(deck))
            b.extend(deck)
        for p in self.players:
            b.extend(p.gems)
            b.extend(p.discount)
            score = p.score
            ap((score >> 8) & 0xFF)
            ap(score & 0xFF)
            ap(255 if p.team_id is None else p.team_id)
            ap(len(p.cards))
            b.extend(p.cards)
            ap(len(p.reserved))
            b.extend(p.reserved)
            flags = 0
            for i, public in enumerate(p.reserved_public):
                if public:
                    flags |= 1 << i
            ap(flags)
            ap(len(p.tiles))
            b.extend(p.tiles)
        return bytes(b)

    @staticmethod
    def from_bytes(data) -> "GameState":
        """Inverse of :meth:`to_bytes` (accepts ``bytes``/``memoryview``)."""
        buf = data if isinstance(data, (bytes, bytearray)) else bytes(data)
        version = buf[0]
        if version != GameState.BYTES_VERSION:
            raise ValueError(
                f"GameState.from_bytes: unsupported format version {version} "
                f"(this build writes {GameState.BYTES_VERSION})")
        s = GameState()
        n = buf[1]
        s.num_players = n
        packed = buf[2]
        s.mode = GameState._MODE_CODES[packed & 3]
        s.team_layout = GameState._LAYOUT_CODES[(packed >> 2) & 3]
        s.phase = PHASE_GAME_OVER if (packed >> 4) & 1 else PHASE_PLAYING
        s.turn_action = GameState._TA_CODES[(packed >> 5) & 3]
        s.current_player = buf[3]
        rsp = buf[4]
        s.round_start_player = None if rsp == 255 else rsp
        s.turn_number = (buf[5] << 8) | buf[6]
        frt = buf[7]
        s.final_round_triggered_by = None if frt == 0 else frt - 1
        i = 8
        count = buf[i]
        i += 1
        s.resigned = list(buf[i:i + count])
        i += count
        s.gems = list(buf[i:i + 6])
        i += 6
        count = buf[i]
        i += 1
        s.tiles = list(buf[i:i + count])
        i += count
        count = buf[i]
        i += 1
        if count == 255:
            s.pending_tile_choice = None
        else:
            s.pending_tile_choice = list(buf[i:i + count])
            i += count
        reason_code = buf[i]
        i += 1
        if reason_code == 0:
            s.game_result = None
        else:
            result: Dict[str, Any] = {
                "reason": (GameState._REASON_CODES[reason_code]
                           if reason_code < 3 else None)}
            forfeiting = buf[i]
            i += 1
            if forfeiting != 255:
                result["forfeitingTeamId"] = forfeiting
            count = buf[i]
            i += 1
            if count != 255:
                result["winningTeamIds"] = list(buf[i:i + count])
                i += count
            s.game_result = result
        board: List[List[int]] = []
        for _ in range(3):
            count = buf[i]
            i += 1
            board.append(list(buf[i:i + count]))
            i += count
        s.board = board
        decks: List[List[int]] = []
        for _ in range(3):
            count = buf[i]
            i += 1
            decks.append(list(buf[i:i + count]))
            i += count
        s.decks = decks
        s.deck_counts = [len(decks[0]), len(decks[1]), len(decks[2])]
        players: List[PlayerState] = []
        for seat in range(n):
            p = PlayerState(f"p{seat}", None, seat)
            p.gems = list(buf[i:i + 6])
            i += 6
            p.discount = list(buf[i:i + 5])
            i += 5
            p.score = (buf[i] << 8) | buf[i + 1]
            i += 2
            team_id = buf[i]
            i += 1
            p.team_id = None if team_id == 255 else team_id
            count = buf[i]
            i += 1
            p.cards = list(buf[i:i + count])
            i += count
            count = buf[i]
            i += 1
            p.reserved = list(buf[i:i + count])
            i += count
            flags = buf[i]
            i += 1
            p.reserved_public = [bool(flags >> j & 1) for j in range(count)]
            p.tiles = list(buf[i + 1:i + 1 + buf[i]])
            i += 1 + buf[i]
            players.append(p)
        s.players = players
        s.config = make_config(n)
        if s.mode != MODE_INDIVIDUAL:
            s.teams = [
                {"id": tid,
                 "playerIndices": [j for j, p in enumerate(players)
                                   if p.team_id == tid]}
                for tid in (0, 1)
            ]
        else:
            s.teams = []
        return s

    # -- convenience -------------------------------------------------------
    @property
    def current(self) -> PlayerState:
        return self.players[self.current_player]

    def active_count(self) -> int:
        return self.num_players - len(self.resigned)

    def is_over(self) -> bool:
        return self.phase != PHASE_PLAYING

    def __repr__(self) -> str:              # pragma: no cover
        return (f"<GameState {self.mode} n={self.num_players} "
                f"turn={self.turn_number} cur={self.current_player} "
                f"phase={self.phase}>")


# ── setup ─────────────────────────────────────────────────────────────────

def _shuffle(seq: Sequence[int], rng) -> List[int]:
    """Fisher-Yates, same direction as ``shuffle()`` in gameLogic.js."""
    a = list(seq)
    for i in range(len(a) - 1, 0, -1):
        j = int(rng.random() * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a


#: Seat → teamId maps, mirroring ``startGame()``'s ``seatOrder`` in
#: ``server/index.js``.
TEAM_SEAT_MAP = {
    (MODE_ONE_V_TWO, None): (0, 1, 1),          # [[0,0],[1,0],[1,1]]
    (MODE_TEAM, "ADJACENT"): (0, 0, 1, 1),      # [[0,0],[0,1],[1,0],[1,1]]
    (MODE_TEAM, "OPPOSITE"): (0, 1, 0, 1),      # [[0,0],[1,0],[0,1],[1,1]]
}


def default_team_ids(mode: str, team_layout: Optional[str],
                     num_players: int) -> Optional[Tuple[int, ...]]:
    if mode == MODE_INDIVIDUAL:
        return None
    if mode == MODE_ONE_V_TWO:
        return TEAM_SEAT_MAP[(MODE_ONE_V_TWO, None)]
    return TEAM_SEAT_MAP[(MODE_TEAM, team_layout or "ADJACENT")]


def new_game(num_players: int,
             mode: str = MODE_INDIVIDUAL,
             team_layout: Optional[str] = None,
             team_ids: Optional[Sequence[int]] = None,
             first_player: Optional[int] = None,
             rng=None,
             setup: Optional[Dict[str, Any]] = None,
             usernames: Optional[Sequence[str]] = None) -> GameState:
    """Port of ``createInitialGameState`` (+ the replay-engine overwrite).

    With ``setup={'board': [[...]]*3, 'decks': [[...]]*3, 'tiles': [...],
    'first': int}`` the shuffle is bypassed entirely and a recorded game is
    reproduced exactly (``decks`` are in SERVER order — ``pop()`` takes the
    LAST element).
    """
    if mode not in (MODE_INDIVIDUAL, MODE_TEAM, MODE_ONE_V_TWO):
        mode = MODE_INDIVIDUAL
    if mode == MODE_TEAM:
        team_layout = "OPPOSITE" if team_layout == "OPPOSITE" else "ADJACENT"
    else:
        team_layout = None

    s = GameState()
    s.num_players = num_players
    s.mode = mode
    s.team_layout = team_layout
    s.config = make_config(num_players)
    cfg = s.config

    if team_ids is None:
        team_ids = default_team_ids(mode, team_layout, num_players)
    if usernames is None:
        usernames = [f"p{i}" for i in range(num_players)]

    s.players = [
        PlayerState(usernames[i],
                    None if team_ids is None else int(team_ids[i]), i)
        for i in range(num_players)
    ]

    if setup is not None:
        s.board = [list(setup["board"][t]) for t in range(3)]
        s.decks = [list(setup["decks"][t]) for t in range(3)]
        s.tiles = list(setup["tiles"])
        first = setup.get("first", first_player)
    else:
        if rng is None:
            rng = random.Random()
        piles = [_shuffle(CARDS_BY_TIER[t], rng) for t in range(3)]
        s.board = [piles[t][:4] for t in range(3)]
        s.decks = [piles[t][4:] for t in range(3)]
        s.tiles = _shuffle(range(10), rng)[:cfg["revealedTiles"]]
        if mode == MODE_ONE_V_TWO:
            first = 0
        elif first_player is not None and 0 <= first_player < num_players:
            first = first_player
        else:
            first = int(rng.random() * num_players)

    if first is None:
        first = 0
    if mode == MODE_ONE_V_TWO and setup is None:
        first = 0
    s.deck_counts = [len(s.decks[0]), len(s.decks[1]), len(s.decks[2])]

    tpc = cfg["tokensPerColor"]
    s.gems = [tpc, tpc, tpc, tpc, tpc, cfg["wildTokens"]]
    s.current_player = first
    s.round_start_player = first
    s.turn_number = 0
    s.phase = PHASE_PLAYING
    s.resigned = []
    s.final_round_triggered_by = None
    s.game_result = None
    s.pending_tile_choice = None
    s.turn_action = None

    if mode != MODE_INDIVIDUAL:
        s.teams = [
            {"id": tid,
             "playerIndices": [i for i, p in enumerate(s.players)
                               if p.team_id == tid]}
            for tid in (0, 1)
        ]
    else:
        s.teams = []
    return s


# ── rule primitives ───────────────────────────────────────────────────────

def can_afford(player: PlayerState, card_id: int) -> bool:
    """``canAfford`` — gold makes up any per-colour shortfall.

    Equivalent to the JS loop over all five colours: a colour the player is
    already over-discounted on contributes ``max(0, cost - discount) == 0`` and
    is skipped, so iterating the non-zero cost entries and bailing out as soon
    as the gold runs short gives the identical answer.
    """
    d = player.discount
    g = player.gems
    gold = g[5]
    for i, amount in CARD_COST_NZ[card_id]:
        need = amount - d[i]
        if need > 0:
            have = g[i]
            if have < need:
                gold -= need - have
                if gold < 0:
                    return False
    return True


def pay_for_card(player: PlayerState, card_id: int) -> List[int]:
    """``payForCard`` — mutates the player, returns the 6-slot ``paid`` array.

    Colour tokens are always spent before gold, colour by colour, using the
    discount computed *before* the card joins the player's tableau.
    """
    cost = CARD_COST[card_id]
    d = player.discount
    g = player.gems
    paid = [0, 0, 0, 0, 0, 0]
    gold_used = 0
    for i in range(5):
        need = cost[i] - d[i]
        if need < 0:
            need = 0
        have = g[i]
        from_color = have if have < need else need
        paid[i] = from_color
        g[i] = have - from_color
        gold_used += need - from_color
    paid[5] = gold_used
    g[5] -= gold_used
    return paid


def qualifies_for_tile(player: PlayerState, tile_id: int) -> bool:
    d = player.discount
    req = TILE_REQ[tile_id]
    return (d[0] >= req[0] and d[1] >= req[1] and d[2] >= req[2]
            and d[3] >= req[3] and d[4] >= req[4])


def qualified_tiles(state: GameState, player: PlayerState) -> List[int]:
    d = player.discount
    out = []
    for tid in state.tiles:
        req = TILE_REQ[tid]
        if (d[0] >= req[0] and d[1] >= req[1] and d[2] >= req[2]
                and d[3] >= req[3] and d[4] >= req[4]):
            out.append(tid)
    return out


def can_select_gem(color: int, selected: Sequence[int], supply: Sequence[int],
                   player_gem_count: int, config: Dict[str, int]) -> bool:
    """Literal port of ``canSelectGem`` — used by tests and the reference
    implementation of gem-take legality."""
    if supply[color] <= 0:
        return False
    n = len(selected)
    if player_gem_count + n >= config["maxTokensInHand"]:
        return False
    if n == 0:
        return True
    if n == 1:
        if selected[0] == color:
            return supply[color] >= config["take2MinStack"] - 1
        return True
    if n == 2:
        if selected[0] == selected[1]:
            return False
        if color in selected:
            return False
        return True
    return False


def is_gem_take_complete(selected: Sequence[int], supply: Sequence[int],
                         player_gem_count: int,
                         config: Dict[str, int]) -> bool:
    """Literal port of ``isGemTakeComplete``."""
    max_can_hold = config["maxTokensInHand"] - player_gem_count
    n = len(selected)
    if n >= max_can_hold:
        return True
    if n == 2 and selected[0] == selected[1]:
        return True
    if n == 3:
        return True
    if n == 2 and selected[0] != selected[1]:
        used = set(selected)
        has_available = any(c not in used and supply[c] > 0 for c in range(5))
        if not has_available:
            return True
    if n == 1:
        color = selected[0]
        can_take_same = supply[color] >= config["take2MinStack"] - 1
        has_other = any(c != color and supply[c] > 0 for c in range(5))
        if not can_take_same and not has_other:
            return True
    return False


def gem_take_accepted(colors: Sequence[int], supply: Sequence[int],
                      player_gem_count: int,
                      config: Dict[str, int]) -> bool:
    """Reference implementation of the ``TAKE_GEMS_CONFIRMED`` validator.

    Feeds ``colors`` through ``canSelectGem`` in the given order exactly like
    the server does, then requires ``isGemTakeComplete``.  ``legal_mask`` uses
    a closed-form equivalent for speed; ``tests/test_takes.py`` proves the two
    agree on every reachable (supply, gem count) combination and that the
    result is independent of the order of ``colors``.
    """
    if not (1 <= len(colors) <= 3):
        return False
    selected: List[int] = []
    for color in colors:
        if not isinstance(color, int) or color < 0 or color > 4:
            return False
        adjusted = list(supply)
        for picked in selected:
            adjusted[picked] -= 1
        if not can_select_gem(color, selected, adjusted, player_gem_count, config):
            return False
        selected.append(color)
    final = list(supply)
    for picked in selected:
        final[picked] -= 1
    return is_gem_take_complete(selected, final, player_gem_count, config)


# ── team scoring (getTeamStats / getQualifyingTeamIds / resolve*) ─────────

def _team_totals(state: GameState):
    """``(total0, second0, total1, second1)`` without allocating dicts.

    ``secondScore`` is ``scores[1]`` of the descending score list when a team
    has exactly two members, and ``-Infinity`` otherwise — matching
    ``getTeamStats`` (and JS's ``-Infinity >= -Infinity === true``).
    """
    t0 = t1 = 0
    hi0 = se0 = hi1 = se1 = NEG_INF
    n0 = n1 = 0
    for p in state.players:
        s = p.score
        if p.team_id == 0:
            t0 += s
            n0 += 1
            if s > hi0:
                se0 = hi0
                hi0 = s
            elif s > se0:
                se0 = s
        elif p.team_id == 1:
            t1 += s
            n1 += 1
            if s > hi1:
                se1 = hi1
                hi1 = s
            elif s > se1:
                se1 = s
    if n0 != 2:
        se0 = NEG_INF
    if n1 != 2:
        se1 = NEG_INF
    return t0, se0, t1, se1


def team_stats(state: GameState) -> List[Dict[str, Any]]:
    if state.mode == MODE_INDIVIDUAL:
        return []
    out = []
    for team_id in (0, 1):
        members = [p for p in state.players if p.team_id == team_id]
        scores = sorted((p.score for p in members), reverse=True)
        out.append({
            "teamId": team_id,
            "total": sum(scores),
            "secondScore": scores[1] if len(scores) == 2 else NEG_INF,
            "cardCount": sum(len(p.cards) for p in members),
        })
    return out


def qualifying_team_ids(state: GameState) -> List[int]:
    if state.mode == MODE_INDIVIDUAL:
        return []
    t0, se0, t1, se1 = _team_totals(state)
    if state.mode == MODE_ONE_V_TWO:
        out = []
        if t0 >= 15:
            out.append(0)
        if t1 >= 34:
            out.append(1)
        return out
    out = []
    if t0 > 30 and se0 >= se1:
        out.append(0)
    if t1 > 30 and se1 >= se0:
        out.append(1)
    return out


def resolve_team_winners(state: GameState,
                         qualifying: Sequence[int]) -> List[int]:
    qualifying = list(qualifying)
    if len(qualifying) <= 1:
        return qualifying
    candidates = [t for t in team_stats(state) if t["teamId"] in qualifying]
    candidates.sort(key=lambda t: (-t["total"], t["cardCount"]))
    best = candidates[0]
    return [t["teamId"] for t in candidates
            if t["total"] == best["total"] and t["cardCount"] == best["cardCount"]]


def resolve_one_vs_two_winners(state: GameState) -> List[int]:
    stats = team_stats(state)
    solo = next((t for t in stats if t["teamId"] == 0), None)
    duo = next((t for t in stats if t["teamId"] == 1), None)
    if solo is None or duo is None:
        return []
    solo_ok = solo["total"] >= 15
    duo_ok = duo["total"] >= 34
    if solo_ok and not duo_ok:
        return [0]
    if duo_ok and not solo_ok:
        return [1]
    if not solo_ok and not duo_ok:
        return []
    solo_excess = solo["total"] - 15
    duo_excess = duo["total"] - 34
    if solo_excess == duo_excess:
        return [0, 1]
    return [0 if solo_excess > duo_excess else 1]


def rating_changes(state: GameState) -> List[int]:
    """Port of ``calculateRatingChanges(state.players, state)``."""
    players = state.players
    if state.mode in (MODE_TEAM, MODE_ONE_V_TWO) and state.game_result is not None:
        winners = state.game_result.get("winningTeamIds")
        # JS: `state.gameResult?.winningTeamIds` — an empty array is truthy,
        # so [] takes this branch and pays 3 to everyone.
        if winners is not None:
            if len(winners) != 1:
                return [3] * len(players)
            return [5 if p.team_id == winners[0] else 0 for p in players]

    ranked = [(i, p.score, len(p.cards)) for i, p in enumerate(players)]
    ranked.sort(key=lambda r: (-r[1], r[2]))
    changes = [0] * len(players)
    rank = 0
    for i, row in enumerate(ranked):
        if i > 0 and (row[1] != ranked[i - 1][1] or row[2] != ranked[i - 1][2]):
            rank = i
        changes[row[0]] = 5 if rank == 0 else (3 if rank == 1 else
                                               (1 if rank == 2 else 0))
    return changes


def individual_winners(state: GameState) -> List[int]:
    """Seats sharing the top rank (same ordering as ``calculateRatingChanges``:
    higher score first, fewer cards breaks the tie)."""
    ranked = [(i, p.score, len(p.cards)) for i, p in enumerate(state.players)]
    ranked.sort(key=lambda r: (-r[1], r[2]))
    best = ranked[0]
    return sorted(r[0] for r in ranked
                  if r[1] == best[1] and r[2] == best[2])


# ── turn plumbing (getNextActivePlayer / advanceTurn / finishTurn) ────────

def get_next_active_player(state: GameState, current_idx: int) -> int:
    resigned = state.resigned
    n = state.num_players
    nxt = (current_idx + 1) % n
    attempts = 0
    while nxt in resigned and attempts < n:
        nxt = (nxt + 1) % n
        attempts += 1
    return nxt


def _first_active_player(state: GameState) -> int:
    for i in range(state.num_players):
        if i not in state.resigned:
            return i
    return 0


def advance_turn(state: GameState) -> None:
    """Port of ``advanceTurn`` — runs after every completed action."""
    player = state.players[state.current_player]
    qualified = qualified_tiles(state, player)
    n = len(qualified)
    if n == 1:
        tile = qualified[0]
        state.tiles = [t for t in state.tiles if t != tile]
        player.tiles.append(tile)
        player.score += TILE_POINTS[tile]
        state.tile_claimed = {"tileId": tile, "playerIndex": state.current_player}
        finish_turn(state)
    elif n > 1:
        # Player must choose — the turn does NOT advance.  Note the server does
        # this regardless of whether turnAction is 'BUY', but CHOOSE_TILE is
        # only accepted while turnAction === 'BUY'.
        state.pending_tile_choice = qualified
    else:
        finish_turn(state)


def finish_turn(state: GameState) -> None:
    """Port of ``finishTurn``."""
    cur_idx = state.current_player
    current_player = state.players[cur_idx]

    if state.final_round_triggered_by is None:
        mode = state.mode
        if mode == MODE_TEAM or mode == MODE_ONE_V_TWO:
            if qualifying_team_ids(state):
                state.final_round_triggered_by = cur_idx
        elif current_player.score >= state.config["winThreshold"]:
            state.final_round_triggered_by = cur_idx

    next_player = get_next_active_player(state, cur_idx)

    if state.final_round_triggered_by is not None:
        round_leader = state.round_start_player
        if round_leader is None or round_leader in state.resigned:
            round_leader = _first_active_player(state)
        if next_player == round_leader:
            if state.mode == MODE_TEAM:
                qualifying = qualifying_team_ids(state)
                if qualifying:
                    state.phase = PHASE_GAME_OVER
                    state.game_result = {
                        "reason": "SCORE",
                        "winningTeamIds": resolve_team_winners(state, qualifying),
                    }
                    state.turn_action = None
                    return
                # Revocable: the condition lapsed while the round played out.
                state.final_round_triggered_by = None
            elif state.mode == MODE_ONE_V_TWO:
                state.phase = PHASE_GAME_OVER
                state.game_result = {
                    "reason": "SCORE",
                    "winningTeamIds": resolve_one_vs_two_winners(state),
                }
                state.turn_action = None
                return
            else:
                state.phase = PHASE_GAME_OVER
                state.turn_action = None
                return

    state.current_player = next_player
    state.turn_action = None
    state.pending_tile_choice = None
    state.turn_number += 1


# ── legality ──────────────────────────────────────────────────────────────
#
# Closed forms for the 30 take patterns.  Derived from canSelectGem +
# isGemTakeComplete with ``P`` = the acting player's token count and ``S`` the
# board supply of the five colours (gold plays no part in a take):
#
#   triple (a,b,c) : P <= 7 and S[a],S[b],S[c] > 0
#   pair   (a,b)   : P <= 8 and S[a],S[b] > 0 and
#                    (P == 8 or no other colour has supply)
#   single (a)     : S[a] > 0 and (P == 9 or (S[a] <= 3 and no other supply))
#   double (a,a)   : P <= 8 and S[a] >= 4
#
# ``tests/test_takes.py`` brute-forces these against ``gem_take_accepted``.

_TRIPLES = TRIPLES
_PAIRS = PAIRS


def _take_legal_into(mask: List[bool], gems: List[int], p_gems: int) -> None:
    s0, s1, s2, s3, s4 = gems[0], gems[1], gems[2], gems[3], gems[4]
    supply = (s0, s1, s2, s3, s4)
    nonzero = (s0 > 0) + (s1 > 0) + (s2 > 0) + (s3 > 0) + (s4 > 0)

    if p_gems <= 7:
        i = TAKE3_START
        for a, b, c in _TRIPLES:
            if supply[a] > 0 and supply[b] > 0 and supply[c] > 0:
                mask[i] = True
            i += 1

    if p_gems <= 8:
        # 2-different: only when the take cannot continue — either the hand cap
        # stops it (P == 8) or no third colour is left in the supply.
        i = TAKE2_START
        if p_gems == 8:
            for a, b in _PAIRS:
                if supply[a] > 0 and supply[b] > 0:
                    mask[i] = True
                i += 1
        elif nonzero <= 2:
            for a, b in _PAIRS:
                if supply[a] > 0 and supply[b] > 0:
                    mask[i] = True
                i += 1
        # 2-same needs a stack of >= take2MinStack (4) before taking.
        i = TAKE2SAME_START
        for c in range(5):
            if supply[c] >= 4:
                mask[i] = True
            i += 1

    if p_gems <= 9:
        i = TAKE1_START
        if p_gems == 9:
            for c in range(5):
                if supply[c] > 0:
                    mask[i] = True
                i += 1
        elif nonzero == 1:
            for c in range(5):
                if 0 < supply[c] <= 3:
                    mask[i] = True
                i += 1


def take_legal_mask(gems: Sequence[int], player_gem_count: int) -> List[bool]:
    """The 30 take bits alone — used by tests to brute-force the closed forms
    against :func:`gem_take_accepted`."""
    mask = [False] * NUM_TAKE_ACTIONS
    _take_legal_into(mask, list(gems), player_gem_count)
    return mask


_FALSE65 = [False] * NUM_ACTIONS


def legal_mask(state: GameState) -> List[bool]:
    """The exact set of actions the Node server would accept as a complete
    turn action, as a 65-long list of bools."""
    mask = _FALSE65[:]
    if state.phase != PHASE_PLAYING:
        return mask

    player = state.players[state.current_player]

    if state.turn_action == TA_BUY:
        # A multi-noble choice is pending and CHOOSE_TILE is the only accepted
        # action (every other handler bails on a non-null turnAction).
        pending = state.pending_tile_choice
        if pending:
            tiles = state.tiles
            for i in range(len(tiles)):
                if i >= 5:
                    break
                tid = tiles[i]
                if tid in pending and qualifies_for_tile(player, tid):
                    mask[CHOOSE_TILE_START + i] = True
        return mask

    # turn_action is None here.  Note that ``pending_tile_choice`` may still be
    # set (orphaned by a non-BUY action that qualified >= 2 nobles); CHOOSE_TILE
    # stays illegal in that case and ordinary actions remain available.

    _take_legal_into(mask, state.gems, player.total_gems())

    board = state.board
    decks = state.decks
    reserved = player.reserved

    # A board row never exceeds cardsPerRow and a hand never exceeds
    # maxReserved, but the caps keep an index from ever spilling into the next
    # action block if a caller hands the engine a doctored state.
    if len(reserved) < state.config["maxReserved"]:
        for t in range(3):
            row = board[t]
            n = len(row)
            if n > MAX_BOARD_SLOTS:
                n = MAX_BOARD_SLOTS
            base = RESERVE_BOARD_START + t * MAX_BOARD_SLOTS
            for s in range(n):
                mask[base + s] = True
            if decks[t]:
                mask[RESERVE_DECK_START + t] = True

    for t in range(3):
        row = board[t]
        n = len(row)
        if n > MAX_BOARD_SLOTS:
            n = MAX_BOARD_SLOTS
        base = BUY_BOARD_START + t * MAX_BOARD_SLOTS
        for s in range(n):
            if can_afford(player, row[s]):
                mask[base + s] = True

    n = len(reserved)
    if n > MAX_RESERVED:
        n = MAX_RESERVED
    for s in range(n):
        if can_afford(player, reserved[s]):
            mask[BUY_RESERVED_START + s] = True

    return mask


def legal_actions(state: GameState) -> List[int]:
    mask = legal_mask(state)
    return [i for i in range(NUM_ACTIONS) if mask[i]]


def is_stuck(state: GameState) -> bool:
    """True when the acting seat has no accepted action at all.

    Reachable in this variant: 10 tokens (cannot take), 3 reserved cards
    (cannot reserve), and nothing affordable.  The server has no PASS.
    """
    if state.phase != PHASE_PLAYING:
        return False
    return not any(legal_mask(state))


def legal_mask_np(state: GameState):
    """numpy flavour of :func:`legal_mask` (allocates — not for the hot loop)."""
    import numpy as np
    return np.fromiter(legal_mask(state), dtype=np.bool_, count=NUM_ACTIONS)


# ── apply ─────────────────────────────────────────────────────────────────

class IllegalAction(Exception):
    pass


def apply(state: GameState, action_index: int) -> Dict[str, Any]:
    """Apply one complete turn action in place; returns ``state.last_event``.

    Mirrors ``processAction`` + ``advanceTurn`` / ``finishTurn`` and then the
    server's ``broadcastProcessedAction`` step that moves ``_tileClaimed`` onto
    the action result.
    """
    if state.phase != PHASE_PLAYING:
        raise IllegalAction("Game is over")

    idx = state.current_player
    player = state.players[idx]
    state.tile_claimed = None

    if action_index < NUM_TAKE_ACTIONS:
        # TAKE_GEMS_CONFIRMED
        if state.turn_action is not None:
            raise IllegalAction("Finish your current action first")
        colors = TAKE_PATTERNS[action_index]
        if not gem_take_accepted(colors, state.gems, player.total_gems(),
                                 state.config):
            raise IllegalAction(f"Cannot take {colors}")
        pg = player.gems
        sg = state.gems
        for c in colors:
            pg[c] += 1
            sg[c] -= 1
        event = {"type": "TAKE_GEMS_CONFIRMED", "actingPlayer": idx,
                 "payload": {"selected": list(colors)}}
        state.turn_action = None
        advance_turn(state)

    elif action_index < RESERVE_DECK_START:
        # ENTER_RESERVE + RESERVE_CARD
        if state.turn_action is not None:
            raise IllegalAction("Finish or cancel your current action first")
        if len(player.reserved) >= state.config["maxReserved"]:
            raise IllegalAction("Reserve full")
        off = action_index - RESERVE_BOARD_START
        t, slot = off >> 2, off & 3
        row = state.board[t]
        if slot >= len(row):
            raise IllegalAction("Card not on board")
        gold_taken = _take_reserve_gold(state, player)
        card = row.pop(slot)
        player.reserved.append(card)
        player.reserved_public.append(True)
        if state.decks[t]:
            row.append(state.decks[t].pop())
        state.deck_counts = [len(state.decks[0]), len(state.decks[1]),
                             len(state.decks[2])]
        event = {"type": "RESERVE_CARD", "actingPlayer": idx,
                 "payload": {"cardId": card, "tier": CARD_TIER[card],
                             "fromDeck": False, "goldTaken": gold_taken}}
        state.turn_action = None
        advance_turn(state)

    elif action_index < BUY_BOARD_START:
        # ENTER_RESERVE + RESERVE_FROM_DECK
        if state.turn_action is not None:
            raise IllegalAction("Finish or cancel your current action first")
        if len(player.reserved) >= state.config["maxReserved"]:
            raise IllegalAction("Reserve full")
        t = action_index - RESERVE_DECK_START
        if not state.decks[t]:
            raise IllegalAction("Invalid or empty deck")
        gold_taken = _take_reserve_gold(state, player)
        card = state.decks[t].pop()
        player.reserved.append(card)
        player.reserved_public.append(False)
        state.deck_counts = [len(state.decks[0]), len(state.decks[1]),
                             len(state.decks[2])]
        event = {"type": "RESERVE_FROM_DECK", "actingPlayer": idx,
                 "payload": {"tier": t + 1, "fromDeck": True,
                             "cardId": card, "goldTaken": gold_taken}}
        state.turn_action = None
        advance_turn(state)

    elif action_index < BUY_RESERVED_START:
        # BUY_CARD source='board'
        if state.turn_action is not None:
            raise IllegalAction("Finish or cancel your current action first")
        off = action_index - BUY_BOARD_START
        t, slot = off >> 2, off & 3
        row = state.board[t]
        if slot >= len(row):
            raise IllegalAction("Card not found")
        card = row[slot]
        if not can_afford(player, card):
            raise IllegalAction("Cannot afford")
        gems_returned = pay_for_card(player, card)
        row.pop(slot)
        if state.decks[t]:
            row.append(state.decks[t].pop())
        state.deck_counts = [len(state.decks[0]), len(state.decks[1]),
                             len(state.decks[2])]
        _gain_card(state, player, card, gems_returned)
        event = {"type": "BUY_CARD", "actingPlayer": idx,
                 "payload": {"cardId": card, "source": "board",
                             "reward": CARD_REWARD[card],
                             "points": CARD_POINTS[card],
                             "gemsReturned": gems_returned}}
        state.turn_action = TA_BUY
        advance_turn(state)

    elif action_index < CHOOSE_TILE_START:
        # BUY_CARD source='reserved'
        if state.turn_action is not None:
            raise IllegalAction("Finish or cancel your current action first")
        slot = action_index - BUY_RESERVED_START
        if slot >= len(player.reserved):
            raise IllegalAction("Card not found")
        card = player.reserved[slot]
        if not can_afford(player, card):
            raise IllegalAction("Cannot afford")
        gems_returned = pay_for_card(player, card)
        player.reserved.pop(slot)
        player.reserved_public.pop(slot)
        _gain_card(state, player, card, gems_returned)
        event = {"type": "BUY_CARD", "actingPlayer": idx,
                 "payload": {"cardId": card, "source": "reserved",
                             "reward": CARD_REWARD[card],
                             "points": CARD_POINTS[card],
                             "gemsReturned": gems_returned}}
        state.turn_action = TA_BUY
        advance_turn(state)

    elif action_index < NUM_ACTIONS:
        # CHOOSE_TILE
        slot = action_index - CHOOSE_TILE_START
        pending = state.pending_tile_choice
        if state.turn_action != TA_BUY or not pending:
            raise IllegalAction("No matching noble choice is pending")
        if slot >= len(state.tiles):
            raise IllegalAction("Tile not found")
        tile = state.tiles[slot]
        if tile not in pending:
            raise IllegalAction("No matching noble choice is pending")
        if not qualifies_for_tile(player, tile):
            raise IllegalAction("Not qualified")
        state.tiles.pop(slot)
        player.tiles.append(tile)
        player.score += TILE_POINTS[tile]
        event = {"type": "CHOOSE_TILE", "actingPlayer": idx,
                 "payload": {"tileId": tile, "playerIndex": idx}}
        # NOTE: finishTurn, not advanceTurn — a second qualifying noble is NOT
        # claimed now; it waits for the next action's advanceTurn.
        finish_turn(state)
    else:
        raise IllegalAction(f"action index out of range: {action_index}")

    # broadcastProcessedAction(): move _tileClaimed onto the result.
    event["tileClaimed"] = state.tile_claimed
    state.tile_claimed = None
    state.last_event = event
    return event


def _take_reserve_gold(state: GameState, player: PlayerState) -> bool:
    """``ENTER_RESERVE``'s gold rule: only when gold remains AND the player is
    below the 10-token cap.  There is no discarding in this variant."""
    if state.gems[5] > 0 and player.total_gems() < state.config["maxTokensInHand"]:
        player.gems[5] += 1
        state.gems[5] -= 1
        return True
    return False


def _gain_card(state: GameState, player: PlayerState, card: int,
               gems_returned: List[int]) -> None:
    player.cards.append(card)
    player.discount[CARD_REWARD[card]] += 1
    player.score += CARD_POINTS[card]
    sg = state.gems
    for i in range(6):
        sg[i] += gems_returned[i]


# ── resign / timeout ──────────────────────────────────────────────────────

def resign(state: GameState, player_index: int) -> Dict[str, Any]:
    """Port of ``processResign`` (+ the RESIGN broadcast event)."""
    _process_resign(state, player_index)
    event = {"type": "RESIGN", "actingPlayer": player_index,
             "payload": {"resignedPlayerIndex": player_index},
             "tileClaimed": state.tile_claimed}
    state.tile_claimed = None
    state.last_event = event
    return event


def timeout(state: GameState, player_index: Optional[int] = None) -> Dict[str, Any]:
    """Port of ``eliminateTimedOutPlayer``: ``processResign`` on the *current*
    seat, then the redundant active-count check the server repeats."""
    if player_index is None:
        player_index = state.current_player
    _process_resign(state, player_index)
    if state.num_players - len(state.resigned) < 2:
        state.phase = PHASE_GAME_OVER
    event = {"type": "TIMEOUT", "actingPlayer": player_index,
             "payload": {"timedOutPlayerIndex": player_index},
             "tileClaimed": state.tile_claimed}
    state.tile_claimed = None
    state.last_event = event
    return event


def _process_resign(state: GameState, player_index: int) -> None:
    if player_index in state.resigned:
        return
    state.resigned.append(player_index)

    player = state.players[player_index]
    if state.mode in (MODE_TEAM, MODE_ONE_V_TWO):
        forfeiting = player.team_id
        state.phase = PHASE_GAME_OVER
        state.turn_action = None
        state.pending_tile_choice = None
        state.final_round_triggered_by = None
        state.game_result = {
            "reason": "FORFEIT",
            "forfeitingTeamId": forfeiting,
            "winningTeamIds": [1 if forfeiting == 0 else 0],
        }
        return

    for i in range(6):
        state.gems[i] += player.gems[i]
        player.gems[i] = 0
    player.cards = []
    player.reserved = []
    player.reserved_public = []
    player.tiles = []
    player.score = 0
    player.discount = [0, 0, 0, 0, 0]

    if state.round_start_player == player_index:
        state.round_start_player = get_next_active_player(state, player_index)

    if state.num_players - len(state.resigned) < 2:
        state.phase = PHASE_GAME_OVER
        state.turn_action = None
    elif state.current_player == player_index:
        state.current_player = get_next_active_player(state, player_index)
        state.turn_action = None
        state.pending_tile_choice = None


# ── protocol / replay bridges ─────────────────────────────────────────────

def to_protocol(state: GameState, action_index: int) -> List[Dict[str, Any]]:
    """The socket.io ``game_action`` message(s) equivalent to ``action_index``.

    Reserves are two messages, exactly as the client sends them.
    """
    if action_index < NUM_TAKE_ACTIONS:
        return [{"type": "TAKE_GEMS_CONFIRMED",
                 "colors": list(TAKE_PATTERNS[action_index])}]
    if action_index < RESERVE_DECK_START:
        off = action_index - RESERVE_BOARD_START
        t, slot = off >> 2, off & 3
        return [{"type": "ENTER_RESERVE"},
                {"type": "RESERVE_CARD", "cardId": state.board[t][slot]}]
    if action_index < BUY_BOARD_START:
        t = action_index - RESERVE_DECK_START
        return [{"type": "ENTER_RESERVE"},
                {"type": "RESERVE_FROM_DECK", "tier": t + 1}]
    if action_index < BUY_RESERVED_START:
        off = action_index - BUY_BOARD_START
        t, slot = off >> 2, off & 3
        return [{"type": "BUY_CARD", "cardId": state.board[t][slot],
                 "source": "board"}]
    if action_index < CHOOSE_TILE_START:
        slot = action_index - BUY_RESERVED_START
        return [{"type": "BUY_CARD",
                 "cardId": state.players[state.current_player].reserved[slot],
                 "source": "reserved"}]
    slot = action_index - CHOOSE_TILE_START
    return [{"type": "CHOOSE_TILE", "tileId": state.tiles[slot]}]


def to_replay_code(state: GameState, action_index: int) -> List[Any]:
    """The compact replay entry (``docs/REPLAY_FORMAT.md`` §1) for an action."""
    p = state.current_player
    if action_index == ACTION_RESIGN:
        return [p, "X"]
    if action_index == ACTION_TIMEOUT:
        return [p, "T"]
    if action_index < NUM_TAKE_ACTIONS:
        return [p, "G", list(TAKE_PATTERNS[action_index])]
    if action_index < RESERVE_DECK_START:
        off = action_index - RESERVE_BOARD_START
        return [p, "R", state.board[off >> 2][off & 3]]
    if action_index < BUY_BOARD_START:
        return [p, "RD", action_index - RESERVE_DECK_START + 1]
    if action_index < BUY_RESERVED_START:
        off = action_index - BUY_BOARD_START
        return [p, "B", state.board[off >> 2][off & 3], "b"]
    if action_index < CHOOSE_TILE_START:
        slot = action_index - BUY_RESERVED_START
        return [p, "B", state.players[p].reserved[slot], "r"]
    return [p, "N", state.tiles[action_index - CHOOSE_TILE_START]]


def from_replay_code(state: GameState, code: Sequence[Any]) -> int:
    """Map a compact replay action to an action index.

    Accepts both ``[playerIndex, 'B', 12, 'r']`` and ``['B', 12, 'r']``.
    ``X`` / ``T`` map to :data:`ACTION_RESIGN` / :data:`ACTION_TIMEOUT`.
    For ``G`` the colour list may be in any order the human clicked; it is
    canonicalised to the matching take index.
    """
    code = list(code)
    if code and isinstance(code[0], int):
        code = code[1:]
    kind = code[0]

    if kind == "G":
        return take_index(code[1])
    if kind == "R":
        card_id = code[1]
        for t in range(3):
            row = state.board[t]
            for s in range(len(row)):
                if row[s] == card_id:
                    return RESERVE_BOARD_START + t * MAX_BOARD_SLOTS + s
        raise IllegalAction(f"replay: card {card_id} not on board")
    if kind == "RD":
        return RESERVE_DECK_START + int(code[1]) - 1
    if kind == "B":
        card_id = code[1]
        source = code[2] if len(code) > 2 else "b"
        if source == "b":
            for t in range(3):
                row = state.board[t]
                for s in range(len(row)):
                    if row[s] == card_id:
                        return BUY_BOARD_START + t * MAX_BOARD_SLOTS + s
            raise IllegalAction(f"replay: card {card_id} not on board")
        reserved = state.players[state.current_player].reserved
        for s in range(len(reserved)):
            if reserved[s] == card_id:
                return BUY_RESERVED_START + s
        raise IllegalAction(f"replay: card {card_id} not reserved")
    if kind == "N":
        tile_id = code[1]
        for i in range(len(state.tiles)):
            if state.tiles[i] == tile_id:
                return CHOOSE_TILE_START + i
        raise IllegalAction(f"replay: tile {tile_id} not revealed")
    if kind == "X":
        return ACTION_RESIGN
    if kind == "T":
        return ACTION_TIMEOUT
    raise IllegalAction(f"replay: unknown action code {kind!r}")


def step(state: GameState, action_index: int) -> Dict[str, Any]:
    """``apply`` that also understands the resign/timeout sentinels."""
    if action_index == ACTION_RESIGN:
        return resign(state, state.current_player)
    if action_index == ACTION_TIMEOUT:
        return timeout(state, state.current_player)
    return apply(state, action_index)


def replay_actions(state: GameState,
                   actions: Sequence[Sequence[Any]]) -> GameState:
    """Feed a list of compact replay actions into an existing state."""
    for code in actions:
        idx = from_replay_code(state, code)
        if idx == ACTION_RESIGN or idx == ACTION_TIMEOUT:
            # The seat prefix is part of the stored format; fall back to the
            # acting seat if a caller passes the bare code.
            seat = code[0] if isinstance(code[0], int) else state.current_player
            if idx == ACTION_RESIGN:
                resign(state, seat)
            else:
                timeout(state, seat)
        else:
            apply(state, idx)
    return state


def replay(replay_json: Dict[str, Any]) -> GameState:
    """Replay a stored replay file (``docs/REPLAY_FORMAT.md`` §1) end to end.

    Accepts the JSON exactly as ``GET /api/replays/:id/raw`` serves it and
    returns the final :class:`GameState`.
    """
    rp = replay_json
    players = rp.get("players") or []
    team_ids = None
    if players and players[0].get("team") is not None:
        team_ids = [p["team"] for p in players]
    usernames = [p.get("u", f"p{i}") for i, p in enumerate(players)] or None
    setup = dict(rp["setup"])
    setup["first"] = rp.get("first", 0)
    state = new_game(rp["n"], rp.get("mode", MODE_INDIVIDUAL),
                     rp.get("layout"), team_ids=team_ids, setup=setup,
                     usernames=usernames)
    return replay_actions(state, rp.get("actions", []))


__all__ = [
    "GameState", "PlayerState", "IllegalAction", "new_game", "make_config",
    "legal_mask", "legal_mask_np", "legal_actions", "is_stuck", "apply",
    "take_legal_mask",
    "step", "resign", "timeout", "to_protocol", "to_replay_code",
    "from_replay_code", "replay", "replay_actions", "rating_changes", "individual_winners",
    "team_stats", "qualifying_team_ids", "resolve_team_winners",
    "resolve_one_vs_two_winners", "can_afford", "pay_for_card",
    "qualifies_for_tile", "qualified_tiles", "can_select_gem",
    "is_gem_take_complete", "gem_take_accepted", "advance_turn",
    "finish_turn", "get_next_active_player", "default_team_ids",
    "TEAM_SEAT_MAP", "MODE_INDIVIDUAL", "MODE_TEAM", "MODE_ONE_V_TWO",
    "PHASE_PLAYING", "PHASE_GAME_OVER",
]
