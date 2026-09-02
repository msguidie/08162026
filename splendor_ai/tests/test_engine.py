"""Rule-by-rule tests, every one cross-checked against server/gameLogic.js.

Each test builds an exact position inside the Node engine via
``validation/probe_state.js``, asks it what it accepts and what an action does,
and requires the Python engine to answer identically.
"""

import pytest

from splendor_ai.rules import engine as E
from splendor_ai.rules.actions import (
    BUY_BOARD_START, CHOOSE_TILE_START, RESERVE_BOARD_START, RESERVE_DECK_START,
)
from splendor_ai.rules.cards import CARDS_BY_TIER, CARD_COST, CARD_REWARD
from splendor_ai.tests.oracle import requires_node
from splendor_ai.tests.positions import discount_cards, position, run, T1_ONE

pytestmark = requires_node

FULL_GEMS = [7, 7, 7, 7, 7, 5]


# ── reserve ───────────────────────────────────────────────────────────────

def test_reserve_takes_gold_when_available_and_under_cap():
    p = run(position(gems=[7, 7, 7, 7, 7, 5],
                     players=[{"gems": [0, 0, 0, 0, 0, 0]}, {}]),
            [["apply", ["R", CARDS_BY_TIER[0][0]]]])
    me = p.state.players[0]
    assert me.gems[5] == 1
    assert p.state.gems[5] == 4
    assert me.reserved_public == [True]


def test_reserve_gives_no_gold_when_the_bank_is_empty():
    p = run(position(gems=[7, 7, 7, 7, 7, 0],
                     players=[{"gems": [0, 0, 0, 0, 0, 0]}, {}]),
            [["apply", ["R", CARDS_BY_TIER[0][0]]]])
    assert p.state.players[0].gems[5] == 0


def test_reserve_gives_no_gold_at_the_ten_token_cap_but_is_still_legal():
    """Ten tokens blocks every gem take, yet a reserve is still accepted — it
    just does not hand out the gold (there is no discarding in this variant)."""
    spec = position(gems=[7, 7, 7, 7, 7, 5],
                    players=[{"gems": [2, 2, 2, 2, 2, 0]}, {}])
    p = run(spec, [["probe"], ["apply", ["R", CARDS_BY_TIER[0][0]]]])
    assert p.state.players[0].gems == [2, 2, 2, 2, 2, 0]
    assert p.state.gems[5] == 5


def test_reserve_at_nine_tokens_takes_gold_and_reaches_the_cap():
    p = run(position(gems=[7, 7, 7, 7, 7, 5],
                     players=[{"gems": [2, 2, 2, 2, 1, 0]}, {}]),
            [["apply", ["R", CARDS_BY_TIER[0][0]]]])
    assert p.state.players[0].total_gems() == 10


def test_reserve_full_blocks_all_reserves():
    reserved = list(CARDS_BY_TIER[2][:3])
    spec = position(players=[{"reserved": reserved,
                              "gems": [0, 0, 0, 0, 0, 0]}, {}])
    legal = run(spec, [["probe"]]).py_legal()
    assert not any(RESERVE_BOARD_START <= a < BUY_BOARD_START for a in legal)


def test_reserve_from_deck_pops_the_last_card_and_stays_hidden():
    tier1 = list(CARDS_BY_TIER[0])
    board = [tier1[:4], list(CARDS_BY_TIER[1][:4]), list(CARDS_BY_TIER[2][:4])]
    decks = [tier1[4:8], list(CARDS_BY_TIER[1][4:8]), list(CARDS_BY_TIER[2][4:8])]
    spec = position(board=board, decks=decks,
                    players=[{"gems": [0, 0, 0, 0, 0, 0]}, {}])
    p = run(spec, [["apply", ["RD", 1]]])
    me = p.state.players[0]
    assert me.reserved == [decks[0][-1]]          # pop() takes the LAST entry
    assert me.reserved_public == [False]
    assert p.state.deck_counts == [3, 4, 4]
    assert p.state.board[0] == board[0]           # a deck reserve does not refill


def test_empty_deck_blocks_the_deck_reserve():
    spec = position(board=[list(CARDS_BY_TIER[0][:4]),
                           list(CARDS_BY_TIER[1][:4]),
                           list(CARDS_BY_TIER[2][:4])],
                    decks=[[], [], []],
                    players=[{"gems": [0, 0, 0, 0, 0, 0]}, {}])
    legal = run(spec, [["probe"]]).py_legal()
    assert not any(RESERVE_DECK_START <= a < BUY_BOARD_START for a in legal)


def test_board_reserve_refills_at_the_end_of_the_row():
    tier1 = list(CARDS_BY_TIER[0])
    board = [tier1[:4], list(CARDS_BY_TIER[1][:4]), list(CARDS_BY_TIER[2][:4])]
    decks = [tier1[4:9], list(CARDS_BY_TIER[1][4:]), list(CARDS_BY_TIER[2][4:])]
    spec = position(board=board, decks=decks,
                    players=[{"gems": [0, 0, 0, 0, 0, 0]}, {}])
    p = run(spec, [["apply", ["R", board[0][1]]]])
    # slot 1 removed, the refill is APPENDED — the row shifts left.
    assert p.state.board[0] == [board[0][0], board[0][2], board[0][3], decks[0][-1]]


# ── buying ────────────────────────────────────────────────────────────────

def _cheapest_single_colour_card():
    """A tier-1 card costing 3 of exactly one colour (addCycle(1,0,[0,0,0,0,3]))."""
    for cid in CARDS_BY_TIER[0]:
        cost = CARD_COST[cid]
        if sum(cost) == 3 and max(cost) == 3:
            return cid, cost.index(3)
    raise AssertionError


def test_buy_spends_colour_before_gold_and_returns_gems_to_the_bank():
    cid, color = _cheapest_single_colour_card()
    hand = [0, 0, 0, 0, 0, 0]
    hand[color] = 1
    hand[5] = 3
    board = [[cid] + [c for c in CARDS_BY_TIER[0] if c != cid][:3],
             list(CARDS_BY_TIER[1][:4]), list(CARDS_BY_TIER[2][:4])]
    spec = position(board=board, gems=[0, 0, 0, 0, 0, 0],
                    players=[{"gems": hand}, {}])
    p = run(spec, [["apply", ["B", cid, "b"]]])
    me = p.state.players[0]
    assert me.gems[color] == 0 and me.gems[5] == 1       # 1 colour + 2 gold
    assert p.state.gems[color] == 1 and p.state.gems[5] == 2
    assert me.discount[CARD_REWARD[cid]] == 1


def test_discount_is_computed_before_the_bought_card_joins_the_tableau():
    """A card that rewards the very colour it costs must NOT discount itself."""
    cid, color = _cheapest_single_colour_card()
    # give one existing card of the reward colour so a discount already exists
    have = [c for c in CARDS_BY_TIER[0]
            if CARD_REWARD[c] == CARD_REWARD[cid] and c != cid][:1]
    hand = [0, 0, 0, 0, 0, 0]
    hand[color] = 3
    board = [[cid] + [c for c in CARDS_BY_TIER[0]
                      if c != cid and c not in have][:3],
             list(CARDS_BY_TIER[1][:4]), list(CARDS_BY_TIER[2][:4])]
    spec = position(board=board, gems=[0, 0, 0, 0, 0, 0],
                    players=[{"gems": hand, "cards": have}, {}])
    p = run(spec, [["apply", ["B", cid, "b"]]])
    expected = 3 - (1 if CARD_REWARD[cid] == color else 0)
    assert p.state.gems[color] == expected


def test_cannot_afford_is_not_offered():
    cid, color = _cheapest_single_colour_card()
    board = [[cid] + [c for c in CARDS_BY_TIER[0] if c != cid][:3],
             list(CARDS_BY_TIER[1][:4]), list(CARDS_BY_TIER[2][:4])]
    hand = [0, 0, 0, 0, 0, 0]
    hand[color] = 2               # one short, no gold
    spec = position(board=board, gems=[0, 0, 0, 0, 0, 0],
                    players=[{"gems": hand}, {}])
    legal = run(spec, [["probe"]]).py_legal()
    assert BUY_BOARD_START not in legal


def test_buy_from_reserve_leaves_the_board_alone():
    cid, color = _cheapest_single_colour_card()
    hand = [0, 0, 0, 0, 0, 0]
    hand[color] = 3
    board = [[c for c in CARDS_BY_TIER[0] if c != cid][:4],
             list(CARDS_BY_TIER[1][:4]), list(CARDS_BY_TIER[2][:4])]
    spec = position(board=board, gems=[0, 0, 0, 0, 0, 0],
                    players=[{"gems": hand, "reserved": [cid]}, {}])
    p = run(spec, [["apply", ["B", cid, "r"]]])
    assert p.state.board[0] == board[0]
    assert p.state.players[0].reserved == []
    assert p.state.players[0].reserved_public == []


def test_buy_board_refills_at_the_end_of_the_row():
    cid, color = _cheapest_single_colour_card()
    rest = [c for c in CARDS_BY_TIER[0] if c != cid]
    board = [[rest[0], cid, rest[1], rest[2]],
             list(CARDS_BY_TIER[1][:4]), list(CARDS_BY_TIER[2][:4])]
    decks = [rest[3:8], list(CARDS_BY_TIER[1][4:]), list(CARDS_BY_TIER[2][4:])]
    hand = [0, 0, 0, 0, 0, 0]
    hand[color] = 3
    spec = position(board=board, decks=decks, gems=[0] * 6,
                    players=[{"gems": hand}, {}])
    p = run(spec, [["apply", ["B", cid, "b"]]])
    assert p.state.board[0] == [rest[0], rest[1], rest[2], decks[0][-1]]


# ── nobles ────────────────────────────────────────────────────────────────

def test_single_qualifying_noble_is_auto_claimed_after_any_action():
    cards = discount_cards([4, 4, 0, 0, 0])          # qualifies tile 0 only
    spec = position(tiles=[0], gems=FULL_GEMS,
                    players=[{"cards": cards, "gems": [0] * 6}, {}])
    p = run(spec, [["apply", ["G", [0, 1, 2]]]])
    assert p.state.players[0].tiles == [0]
    assert p.state.players[0].score == 3
    assert p.state.tiles == []
    assert p.state.current_player == 1                # the turn advanced
    assert p.state.last_event["tileClaimed"] == {"tileId": 0, "playerIndex": 0}


def test_two_qualifying_nobles_require_a_choice_and_freeze_the_turn():
    cid, color = _cheapest_single_colour_card()
    cards = discount_cards([4, 4, 4, 0, 0])          # qualifies tiles 0 and 1
    rest = [c for c in CARDS_BY_TIER[0] if c not in cards and c != cid]
    board = [[cid] + rest[:3], list(CARDS_BY_TIER[1][:4]), list(CARDS_BY_TIER[2][:4])]
    hand = [0] * 6
    hand[color] = 3
    spec = position(tiles=[0, 1], board=board, gems=FULL_GEMS,
                    players=[{"cards": cards, "gems": hand}, {}])
    p = run(spec, [["apply", ["B", cid, "b"]], ["probe"], ["apply", ["N", 1]]])
    assert p.state.players[0].tiles == [1]
    assert p.state.tiles == [0]
    assert p.state.current_player == 1


def test_only_tile_choices_are_legal_while_a_buy_choice_is_pending():
    cid, color = _cheapest_single_colour_card()
    cards = discount_cards([4, 4, 4, 0, 0])
    rest = [c for c in CARDS_BY_TIER[0] if c not in cards and c != cid]
    board = [[cid] + rest[:3], list(CARDS_BY_TIER[1][:4]), list(CARDS_BY_TIER[2][:4])]
    hand = [0] * 6
    hand[color] = 3
    spec = position(tiles=[0, 1], board=board, gems=FULL_GEMS,
                    players=[{"cards": cards, "gems": hand}, {}])
    p = run(spec, [["apply", ["B", cid, "b"]], ["probe"]])
    legal = p.py_legal()
    assert legal == {CHOOSE_TILE_START, CHOOSE_TILE_START + 1}
    assert p.state.turn_action == "BUY"


def test_second_noble_is_auto_claimed_on_the_next_turn():
    cid, color = _cheapest_single_colour_card()
    cards = discount_cards([4, 4, 4, 0, 0])
    rest = [c for c in CARDS_BY_TIER[0] if c not in cards and c != cid]
    board = [[cid] + rest[:3], list(CARDS_BY_TIER[1][:4]), list(CARDS_BY_TIER[2][:4])]
    hand = [0] * 6
    hand[color] = 3
    spec = position(tiles=[0, 1], board=board, gems=FULL_GEMS,
                    players=[{"cards": cards, "gems": hand}, {"gems": [0] * 6}])
    p = run(spec, [
        ["apply", ["B", cid, "b"]],     # -> pending choice of tiles 0 and 1
        ["apply", ["N", 1]],            # choose one; tile 0 is left behind
        ["apply", ["G", [0, 1, 2]]],    # seat 1 acts
        ["apply", ["G", [0, 1, 2]]],    # seat 0 acts -> auto-claims tile 0
    ])
    assert p.state.players[0].tiles == [1, 0]
    assert p.state.players[0].score == 6
    assert p.state.tiles == []


def test_orphaned_noble_choice_after_a_non_buy_action():
    """A gem take that qualifies TWO nobles sets ``_pendingTileChoice`` while
    ``turnAction`` stays null.  CHOOSE_TILE requires ``turnAction === 'BUY'``,
    so the choice is unreachable — the seat keeps its ordinary actions and the
    turn does not advance.  A faithful port must reproduce this quirk."""
    cards = discount_cards([4, 4, 4, 0, 0])          # tiles 0 and 1 qualify
    spec = position(tiles=[0, 1], gems=FULL_GEMS,
                    players=[{"cards": cards, "gems": [0] * 6}, {}])
    p = run(spec, [["apply", ["G", [0, 1, 2]]], ["probe"], ["apply", ["N", 0]]])
    assert p.state.pending_tile_choice == [0, 1]
    assert p.state.turn_action is None
    assert p.state.current_player == 0               # turn did NOT advance
    assert p.state.players[0].tiles == []            # the choice was rejected
    legal = p.py_legal()
    assert all(a < CHOOSE_TILE_START for a in legal)
    assert legal, "the seat must still have ordinary actions"


def test_three_qualifying_nobles_leave_two_after_the_choice():
    cid, color = _cheapest_single_colour_card()
    cards = discount_cards([4, 4, 4, 3, 0])          # tiles 0, 1, 6 and 8
    rest = [c for c in CARDS_BY_TIER[0] if c not in cards and c != cid]
    board = [[cid] + rest[:3], list(CARDS_BY_TIER[1][:4]), list(CARDS_BY_TIER[2][:4])]
    hand = [0] * 6
    hand[color] = 3
    spec = position(tiles=[0, 1, 6, 8], board=board, gems=FULL_GEMS,
                    players=[{"cards": cards, "gems": hand}, {"gems": [0] * 6}])
    p = run(spec, [["apply", ["B", cid, "b"]], ["probe"], ["apply", ["N", 6]],
                   ["apply", ["G", [0, 1, 2]]],
                   ["apply", ["G", [0, 1, 2]]], ["probe"]])
    assert p.state.players[0].tiles == [6]
    assert p.state.pending_tile_choice is not None
    assert len(p.state.pending_tile_choice) >= 2
    assert p.state.turn_action is None


# ── final round ───────────────────────────────────────────────────────────

def test_individual_final_round_ends_when_play_returns_to_the_leader():
    spec = position(n=3, gems=FULL_GEMS, current=0, round_start=0,
                    players=[{"gems": [0] * 6, "score": 15},
                             {"gems": [0] * 6}, {"gems": [0] * 6}])
    p = run(spec, [["apply", ["G", [0, 1, 2]]], ["probe"],
                   ["apply", ["G", [0, 1, 2]]],
                   ["apply", ["G", [0, 1, 2]]]])
    assert p.state.phase == "GAME_OVER"
    assert p.state.final_round_triggered_by == 0
    assert p.state.game_result is None               # INDIVIDUAL leaves it null


def test_individual_final_round_leader_shifts_after_a_resign():
    spec = position(n=3, gems=FULL_GEMS, current=0, round_start=0,
                    players=[{"gems": [1, 1, 1, 0, 0, 0]},
                             {"gems": [0] * 6, "score": 15},
                             {"gems": [0] * 6}])
    p = run(spec, [["resign", 0], ["probe"],
                   ["apply", ["G", [0, 1, 2]]],
                   ["apply", ["G", [0, 1, 2]]]])
    assert p.state.resigned == [0]
    assert p.state.round_start_player == 1           # leader moved off seat 0
    assert p.state.phase == "GAME_OVER"


def test_team_final_round_is_revocable():
    """2v2 qualification (`total > 30 && secondScore >= theirs`) is re-checked
    when play returns to the leader; if it lapsed, the game simply continues."""
    a, b = T1_ONE[0], T1_ONE[1]
    cost_a, cost_b = CARD_COST[a], CARD_COST[b]
    hand_a = [0] * 6
    hand_a[cost_a.index(4)] = 4
    hand_b = [0] * 6
    hand_b[cost_b.index(4)] = 4
    rest = [c for c in CARDS_BY_TIER[0] if c not in (a, b)]
    board = [[a, b, rest[0], rest[1]],
             list(CARDS_BY_TIER[1][:4]), list(CARDS_BY_TIER[2][:4])]
    spec = position(n=4, mode="TEAM", layout="ADJACENT", board=board,
                    gems=FULL_GEMS, current=0, round_start=0,
                    players=[{"gems": [0] * 6, "score": 31},
                             {"gems": [0] * 6, "score": 0},
                             {"gems": hand_a, "score": 0},
                             {"gems": hand_b, "score": 0}])
    p = run(spec, [["apply", ["G", [0, 1, 2]]],      # seat 0 triggers
                   ["apply", ["G", [0, 1, 2]]],      # seat 1
                   ["apply", ["B", a, "b"]],         # seat 2 scores 1
                   ["apply", ["B", b, "b"]],         # seat 3 scores 1
                   ["probe"]])
    assert p.state.phase == "PLAYING"
    assert p.state.final_round_triggered_by is None  # revoked
    assert p.state.current_player == 0


def test_team_final_round_ends_the_game_when_it_holds():
    spec = position(n=4, mode="TEAM", layout="OPPOSITE", gems=FULL_GEMS,
                    current=0, round_start=0,
                    players=[{"gems": [0] * 6, "score": 20},
                             {"gems": [0] * 6, "score": 1},
                             {"gems": [0] * 6, "score": 12},
                             {"gems": [0] * 6, "score": 1}])
    p = run(spec, [["apply", ["G", [0, 1, 2]]], ["apply", ["G", [0, 1, 2]]],
                   ["apply", ["G", [0, 1, 2]]], ["apply", ["G", [0, 1, 2]]]])
    assert p.state.phase == "GAME_OVER"
    assert p.state.game_result["reason"] == "SCORE"
    # OPPOSITE seats: team0 = {0, 2} = 32, team1 = {1, 3} = 2
    assert p.state.game_result["winningTeamIds"] == [0]
    assert E.rating_changes(p.state) == [5, 0, 5, 0]


def test_one_v_two_final_round_is_irrevocable_and_ties_on_excess():
    """Solo needs 15, the duo needs 34; equal excess is a draw and the final
    round cannot be revoked the way a 2v2 one can."""
    spec = position(n=3, mode="ONE_V_TWO", gems=FULL_GEMS,
                    current=0, round_start=0,
                    players=[{"gems": [0] * 6, "score": 20},
                             {"gems": [0] * 6, "score": 20},
                             {"gems": [0] * 6, "score": 19}])
    p = run(spec, [["apply", ["G", [0, 1, 2]]], ["probe"],
                   ["apply", ["G", [0, 1, 2]]],
                   ["apply", ["G", [0, 1, 2]]]])
    assert p.state.phase == "GAME_OVER"
    assert p.state.game_result["winningTeamIds"] == [0, 1]     # 5 == 5 excess
    assert E.rating_changes(p.state) == [3, 3, 3]


def test_one_v_two_solo_only_qualification():
    spec = position(n=3, mode="ONE_V_TWO", gems=FULL_GEMS,
                    current=0, round_start=0,
                    players=[{"gems": [0] * 6, "score": 16},
                             {"gems": [0] * 6, "score": 5},
                             {"gems": [0] * 6, "score": 5}])
    p = run(spec, [["apply", ["G", [0, 1, 2]]], ["apply", ["G", [0, 1, 2]]],
                   ["apply", ["G", [0, 1, 2]]]])
    assert p.state.game_result["winningTeamIds"] == [0]
    assert E.rating_changes(p.state) == [5, 0, 0]


# ── resign / timeout ──────────────────────────────────────────────────────

def test_individual_resign_returns_gems_and_zeroes_everything():
    cards = discount_cards([2, 1, 0, 0, 0])
    spec = position(n=3, gems=[1, 1, 1, 1, 1, 1], current=1,
                    players=[{"gems": [0] * 6},
                             {"gems": [1, 2, 0, 0, 0, 1], "cards": cards,
                              "reserved": list(CARDS_BY_TIER[2][:2]),
                              "tiles": [0]},
                             {"gems": [0] * 6}])
    p = run(spec, [["resign", 1], ["probe"]])
    me = p.state.players[1]
    assert me.gems == [0] * 6 and me.cards == [] and me.reserved == []
    assert me.tiles == [] and me.score == 0 and me.discount == [0] * 5
    assert p.state.gems == [2, 3, 1, 1, 1, 2]        # tokens went back
    assert p.state.current_player == 2               # the turn moved on
    assert p.state.phase == "PLAYING"


def test_individual_resign_ends_the_game_below_two_active_seats():
    spec = position(n=2, gems=FULL_GEMS,
                    players=[{"gems": [0] * 6}, {"gems": [0] * 6}])
    p = run(spec, [["resign", 1]])
    assert p.state.phase == "GAME_OVER"


def test_team_resign_is_an_instant_forfeit():
    spec = position(n=4, mode="TEAM", layout="ADJACENT", gems=FULL_GEMS,
                    players=[{"gems": [0] * 6} for _ in range(4)])
    p = run(spec, [["resign", 2]])
    assert p.state.phase == "GAME_OVER"
    assert p.state.game_result == {
        "reason": "FORFEIT", "forfeitingTeamId": 1, "winningTeamIds": [0]}
    assert E.rating_changes(p.state) == [5, 5, 0, 0]


def test_timeout_matches_the_server_elimination_path():
    spec = position(n=3, gems=FULL_GEMS, current=2,
                    players=[{"gems": [0] * 6}, {"gems": [0] * 6},
                             {"gems": [1, 1, 0, 0, 0, 0]}])
    p = run(spec, [["timeout", 2], ["probe"]])
    assert p.state.resigned == [2]
    assert p.state.current_player == 0
    assert p.state.last_event["type"] == "TIMEOUT"


# ── stuck ─────────────────────────────────────────────────────────────────

def test_stuck_position_has_no_legal_action():
    """10 tokens (no take), 3 reserved (no reserve) and nothing affordable —
    the server has no PASS, so the seat is genuinely stuck."""
    expensive = [c for c in CARDS_BY_TIER[2] if max(CARD_COST[c]) >= 6][:7]
    spec = position(board=[expensive[3:7], [], []],
                    decks=[[], [], []],
                    gems=[7, 7, 7, 7, 7, 5],
                    players=[{"gems": [2, 2, 2, 2, 2, 0],
                              "reserved": expensive[:3]}, {"gems": [0] * 6}])
    p = run(spec, [["probe"]])
    assert p.py_legal() == set()
    assert E.is_stuck(p.state)


def test_not_stuck_when_a_reserve_is_still_possible():
    expensive = [c for c in CARDS_BY_TIER[2] if max(CARD_COST[c]) >= 6][:6]
    spec = position(board=[expensive[2:6], [], []], decks=[[], [], []],
                    gems=[7, 7, 7, 7, 7, 5],
                    players=[{"gems": [2, 2, 2, 2, 2, 0],
                              "reserved": expensive[:2]}, {"gems": [0] * 6}])
    p = run(spec, [["probe"]])
    assert p.py_legal()
    assert not E.is_stuck(p.state)
