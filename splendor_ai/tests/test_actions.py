"""The action space: index table, protocol bridge and replay-code bridge."""

import itertools
import random

import pytest

from splendor_ai.rules import engine as E
from splendor_ai.rules import actions as A


def test_index_table_layout():
    assert A.NUM_ACTIONS == 65
    assert A.NUM_TAKE_ACTIONS == 30
    assert (A.TAKE3_START, A.TAKE2_START, A.TAKE1_START, A.TAKE2SAME_START) == (0, 10, 20, 25)
    assert (A.RESERVE_BOARD_START, A.RESERVE_DECK_START) == (30, 42)
    assert (A.BUY_BOARD_START, A.BUY_RESERVED_START, A.CHOOSE_TILE_START) == (45, 57, 60)


def test_take_patterns_are_the_documented_ones():
    assert A.TAKE_PATTERNS[:10] == tuple(itertools.combinations(range(5), 3))
    assert A.TAKE_PATTERNS[10:20] == tuple(itertools.combinations(range(5), 2))
    assert A.TAKE_PATTERNS[20:25] == tuple((c,) for c in range(5))
    assert A.TAKE_PATTERNS[25:30] == tuple((c, c) for c in range(5))
    assert len(set(A.TAKE_PATTERNS)) == 30


def test_take_index_canonicalises_any_order():
    for pat in A.TAKE_PATTERNS:
        for perm in itertools.permutations(pat):
            assert A.take_index(list(perm)) == A.TAKE_INDEX[pat]
    with pytest.raises(ValueError):
        A.take_index([0, 0, 1])          # not representable


def test_board_action_arithmetic():
    for t in range(3):
        for s in range(4):
            assert A.reserve_board_action(t, s) == 30 + t * 4 + s
            assert A.buy_board_action(t, s) == 45 + t * 4 + s
    assert [A.reserve_deck_action(t) for t in range(3)] == [42, 43, 44]
    assert [A.buy_reserved_action(s) for s in range(3)] == [57, 58, 59]
    assert [A.choose_tile_action(s) for s in range(5)] == [60, 61, 62, 63, 64]


def test_action_names_are_unique():
    assert len(set(A.ACTION_NAMES)) == A.NUM_ACTIONS


def _random_states(n=60, seed=5):
    rng = random.Random(seed)
    configs = [(2, "INDIVIDUAL", None), (3, "INDIVIDUAL", None),
               (4, "INDIVIDUAL", None), (3, "ONE_V_TWO", None),
               (4, "TEAM", "ADJACENT"), (4, "TEAM", "OPPOSITE")]
    out = []
    for i in range(n):
        np_, mode, layout = configs[i % len(configs)]
        s = E.new_game(np_, mode, layout, rng=rng)
        for _ in range(rng.randrange(0, 60)):
            acts = E.legal_actions(s)
            if not acts or s.is_over():
                break
            E.apply(s, acts[rng.randrange(len(acts))])
        if not s.is_over() and E.legal_actions(s):
            out.append(s)
    return out


def test_protocol_and_replay_round_trip():
    for s in _random_states():
        for idx in E.legal_actions(s):
            msgs = E.to_protocol(s, idx)
            assert msgs and all("type" in m for m in msgs)
            code = E.to_replay_code(s, idx)
            assert code[0] == s.current_player
            assert E.from_replay_code(s, code) == idx
            # the seat prefix is optional
            assert E.from_replay_code(s, code[1:]) == idx


def test_protocol_shapes():
    s = E.new_game(2, rng=random.Random(3))
    assert E.to_protocol(s, 0) == [{"type": "TAKE_GEMS_CONFIRMED", "colors": [0, 1, 2]}]
    msgs = E.to_protocol(s, A.RESERVE_BOARD_START)
    assert msgs[0] == {"type": "ENTER_RESERVE"}
    assert msgs[1]["type"] == "RESERVE_CARD" and msgs[1]["cardId"] == s.board[0][0]
    msgs = E.to_protocol(s, A.RESERVE_DECK_START + 2)
    assert msgs[1] == {"type": "RESERVE_FROM_DECK", "tier": 3}
    msgs = E.to_protocol(s, A.BUY_BOARD_START + 4)
    assert msgs == [{"type": "BUY_CARD", "cardId": s.board[1][0], "source": "board"}]


def test_replay_codes_for_gem_takes_accept_any_click_order():
    s = E.new_game(4, rng=random.Random(9))
    assert E.from_replay_code(s, [0, "G", [2, 0, 1]]) == A.take_index([0, 1, 2])
    assert E.from_replay_code(s, [0, "G", [3, 3]]) == A.TAKE2SAME_START + 3


def test_resign_and_timeout_sentinels():
    s = E.new_game(3, rng=random.Random(2))
    assert E.from_replay_code(s, [1, "X"]) == A.ACTION_RESIGN
    assert E.from_replay_code(s, [2, "T"]) == A.ACTION_TIMEOUT
    assert A.action_name(A.ACTION_RESIGN) == "RESIGN"
    assert A.action_name(A.ACTION_TIMEOUT) == "TIMEOUT"


def test_legal_mask_and_legal_actions_agree():
    for s in _random_states(seed=17):
        mask = E.legal_mask(s)
        assert len(mask) == A.NUM_ACTIONS
        assert [i for i, v in enumerate(mask) if v] == E.legal_actions(s)
        import numpy as np
        assert np.array_equal(E.legal_mask_np(s), np.array(mask, dtype=bool))


def test_apply_rejects_illegal_actions():
    for s in _random_states(n=20, seed=23):
        legal = set(E.legal_actions(s))
        for idx in range(A.NUM_ACTIONS):
            if idx in legal:
                continue
            with pytest.raises(E.IllegalAction):
                E.apply(s.clone(), idx)


def test_clone_is_independent():
    s = E.new_game(4, "TEAM", "ADJACENT", rng=random.Random(1))
    c = s.clone()
    E.apply(s, E.legal_actions(s)[0])
    assert c.gems != s.gems or c.board != s.board or c.current_player != s.current_player
    assert c.turn_number == 0
    for a, b in zip(c.players, s.players):
        assert a.gems is not b.gems and a.cards is not b.cards
