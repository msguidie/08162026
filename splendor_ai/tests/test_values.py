"""Terminal value vectors (docs/AI_DESIGN.md §1.2).

Every end state below is produced by the engine itself — ``finish_turn`` /
``resign`` / a full random game — so the values are read off the same
``game_result`` the server would broadcast.
"""

import random

import numpy as np
import pytest

from splendor_ai.rules import engine as E
from splendor_ai import values as V


# ── helpers ───────────────────────────────────────────────────────────────

def _game(n, mode="INDIVIDUAL", layout=None, scores=None, cards=None, seed=0):
    """A fresh game with hand-set scores / tableau sizes."""
    s = E.new_game(n, mode, layout, rng=random.Random(seed))
    for i, p in enumerate(s.players):
        if scores is not None:
            p.score = scores[i]
        if cards is not None:
            p.cards = list(range(cards[i]))
    return s


def _finish(state):
    """End the game through the engine's own ``finishTurn`` branch."""
    if state.final_round_triggered_by is None:
        state.final_round_triggered_by = state.current_player
    state.round_start_player = E.get_next_active_player(
        state, state.current_player)
    E.finish_turn(state)
    return state


def _random_finished_game(rng, mode, n, layout=None, resign_p=0.004):
    s = E.new_game(n, mode, layout, rng=rng)
    while not s.is_over():
        actions = E.legal_actions(s)
        if not actions or rng.random() < resign_p:
            E.resign(s, s.current_player)       # stuck seat -> resign (§1.2)
            continue
        E.apply(s, rng.choice(actions))
    return s


# ── individual ranking ────────────────────────────────────────────────────

def test_two_player_win_and_loss():
    s = _finish(_game(2, scores=[15, 9], cards=[8, 7]))
    assert s.phase == "GAME_OVER" and s.game_result is None
    assert V.terminal_values(s).tolist() == [1.0, -1.0, 0.0, 0.0]


def test_two_player_tie_is_zero():
    s = _finish(_game(2, scores=[15, 15], cards=[8, 8]))
    assert V.terminal_values(s).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_fewer_cards_breaks_a_score_tie_exactly_like_the_server():
    s = _finish(_game(2, scores=[15, 15], cards=[9, 8]))
    z = V.terminal_values(s)
    assert z.tolist() == [-1.0, 1.0, 0.0, 0.0]
    assert E.individual_winners(s) == [1]


def test_four_player_rank_linear():
    s = _finish(_game(4, scores=[16, 12, 8, 4], cards=[9, 9, 9, 9]))
    z = V.terminal_values(s)
    assert np.allclose(z, [1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0])
    assert abs(float(z.sum())) < 1e-6


def test_ties_share_the_mean_rank():
    # two seats tie for first: ranks 1.5, 1.5, 3, 4
    s = _finish(_game(4, scores=[16, 16, 8, 4], cards=[9, 9, 9, 9]))
    z = V.terminal_values(s)
    assert np.allclose(z, [2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, -1.0])
    assert abs(float(z.sum())) < 1e-6

    # everybody ties -> everybody gets the mean value 0
    s = _finish(_game(3, scores=[7, 7, 7], cards=[5, 5, 5]))
    assert V.terminal_values(s).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_resigned_seats_rank_last_even_with_the_better_key():
    s = _game(3, scores=[5, 4, 0], cards=[4, 4, 2])
    E.resign(s, 1)                      # zeroes seat 1's score and cards
    assert s.players[1].score == 0 and s.players[1].cards == []
    _finish(s)
    z = V.terminal_values(s)
    # (0 score, 0 cards) would out-rank seat 2's (0 score, 2 cards) on the raw
    # server key; a resigned seat is ranked behind every live seat instead.
    assert z.tolist() == [1.0, -1.0, 0.0, 0.0]


def test_individual_resign_below_two_seats_ends_the_game():
    s = _game(2, scores=[6, 11], cards=[5, 6])
    E.resign(s, 0)
    assert s.phase == "GAME_OVER"
    assert V.terminal_values(s).tolist() == [-1.0, 1.0, 0.0, 0.0]


# ── team modes ────────────────────────────────────────────────────────────

def test_team_win_by_score():
    # ADJACENT: seats 0,1 = team 0; seats 2,3 = team 1.
    s = _finish(_game(4, "TEAM", "ADJACENT", scores=[16, 16, 5, 5],
                      cards=[9, 9, 4, 4]))
    assert s.game_result["reason"] == "SCORE"
    assert s.game_result["winningTeamIds"] == [0]
    assert V.terminal_values(s).tolist() == [1.0, 1.0, -1.0, -1.0]


def test_team_draw_pays_zero_to_everyone():
    # Both teams qualify with the same total and the same card count, so
    # resolve_team_winners returns both ids.
    s = _finish(_game(4, "TEAM", "OPPOSITE", scores=[16, 16, 16, 16],
                      cards=[9, 9, 9, 9]))
    assert sorted(s.game_result["winningTeamIds"]) == [0, 1]
    assert V.terminal_values(s).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_team_forfeit_gives_the_resigning_side_minus_one():
    s = _game(4, "TEAM", "ADJACENT", scores=[10, 10, 10, 10])
    E.resign(s, 2)
    assert s.phase == "GAME_OVER"
    assert s.game_result["reason"] == "FORFEIT"
    assert s.game_result["forfeitingTeamId"] == 1
    assert V.terminal_values(s).tolist() == [1.0, 1.0, -1.0, -1.0]


def test_one_v_two_solo_wins():
    s = _finish(_game(3, "ONE_V_TWO", scores=[20, 8, 8]))
    assert s.game_result["winningTeamIds"] == [0]
    assert V.terminal_values(s).tolist() == [1.0, -1.0, -1.0, 0.0]


def test_one_v_two_duo_wins():
    s = _finish(_game(3, "ONE_V_TWO", scores=[15, 20, 20]))
    assert s.game_result["winningTeamIds"] == [1]
    assert V.terminal_values(s).tolist() == [-1.0, 1.0, 1.0, 0.0]


def test_one_v_two_equal_excess_is_a_draw():
    # solo 15 - 15 == 0 and duo 34 - 34 == 0: both sides qualify equally.
    s = _finish(_game(3, "ONE_V_TWO", scores=[15, 17, 17]))
    assert sorted(s.game_result["winningTeamIds"]) == [0, 1]
    assert V.terminal_values(s).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_one_v_two_nobody_qualifies_is_a_draw():
    s = _game(3, "ONE_V_TWO", scores=[9, 9, 9])
    _finish(s)                                   # forced final round
    assert s.game_result["winningTeamIds"] == []
    assert V.terminal_values(s).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_one_v_two_forfeit_both_directions():
    s = _game(3, "ONE_V_TWO", scores=[10, 10, 10])
    E.resign(s, 0)
    assert V.terminal_values(s).tolist() == [-1.0, 1.0, 1.0, 0.0]

    s = _game(3, "ONE_V_TWO", scores=[10, 10, 10])
    E.timeout(s, 2)
    assert s.game_result["reason"] == "FORFEIT"
    assert V.terminal_values(s).tolist() == [1.0, -1.0, -1.0, 0.0]


def test_unfinished_games_are_rejected():
    s = _game(2)
    with pytest.raises(ValueError):
        V.terminal_values(s)


# ── seat-relative view and masks ──────────────────────────────────────────

def test_seat_relative_matches_np_roll_for_four_seats():
    z = np.array([1.0, 0.25, -0.25, -1.0], dtype=np.float32)
    for seat in range(4):
        assert np.array_equal(V.seat_relative(z, seat, 4), np.roll(z, -seat))
        assert np.array_equal(V.seat_relative(z, seat), np.roll(z, -seat))


def test_seat_relative_keeps_short_tables_inside_the_valid_mask():
    z = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32)
    rel = V.seat_relative(z, 1, 2)
    assert rel.tolist() == [-1.0, 1.0, 0.0, 0.0]
    mask = V.z_valid_mask(2)
    assert mask.tolist() == [1.0, 1.0, 0.0, 0.0]
    # every real value stays inside the masked window
    assert float((rel * (1.0 - mask)).sum()) == 0.0

    z3 = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32)
    assert V.seat_relative(z3, 2, 3).tolist() == [-1.0, 1.0, 0.0, 0.0]


def test_z_valid_mask():
    assert V.z_valid_mask(2).tolist() == [1, 1, 0, 0]
    assert V.z_valid_mask(3).tolist() == [1, 1, 1, 0]
    assert V.z_valid_mask(4).tolist() == [1, 1, 1, 1]
    assert V.z_valid_mask(4).dtype == np.float32


# ── truncation ────────────────────────────────────────────────────────────

def test_standings_values_rank_a_truncated_individual_game():
    s = _game(3, scores=[9, 4, 9], cards=[6, 3, 5])
    z = V.standings_values(s)
    # seat 2 leads on the card tiebreak, then seat 0, then seat 1
    assert np.allclose(z[:3], [0.0, -1.0, 1.0])
    assert abs(float(z.sum())) < 1e-6


def test_standings_values_for_team_modes():
    s = _game(4, "TEAM", "ADJACENT", scores=[9, 9, 4, 4], cards=[6, 6, 3, 3])
    assert V.standings_values(s).tolist() == [1.0, 1.0, -1.0, -1.0]

    tie = _game(4, "TEAM", "OPPOSITE", scores=[7, 7, 7, 7], cards=[4, 4, 4, 4])
    assert V.standings_values(tie).tolist() == [0.0, 0.0, 0.0, 0.0]

    # cards break a total tie, fewer is better
    cards_tie = _game(4, "TEAM", "ADJACENT", scores=[7, 7, 7, 7],
                      cards=[5, 5, 4, 4])
    assert V.standings_values(cards_tie).tolist() == [-1.0, -1.0, 1.0, 1.0]


def test_standings_values_for_team_are_symmetric():
    """Both TEAM sides need the same 30 points, so swapping the two sides has
    to negate the vector — no side may be favoured by the comparison."""
    for scores, cards in ([9, 9, 4, 4], [6, 6, 3, 3]), ([2, 5, 9, 1], [4, 4, 4, 4]):
        a = V.standings_values(_game(4, "TEAM", "ADJACENT",
                                     scores=scores, cards=cards))
        swapped = [scores[2], scores[3], scores[0], scores[1]]
        swapped_cards = [cards[2], cards[3], cards[0], cards[1]]
        b = V.standings_values(_game(4, "TEAM", "ADJACENT",
                                     scores=swapped, cards=swapped_cards))
        assert np.allclose(a, -b)


# ── 1v2 truncation: progress towards two different thresholds ─────────────

def test_standings_values_1v2_equal_progress_is_neutral():
    """15 and 34 are the two thresholds, so 6 solo points and 14 duo points are
    the same fraction of the way home; the old `(solo-15)-(duo-34)` margin
    scored this +1 for the solo because of the constant 19-point offset."""
    s = _game(3, "ONE_V_TWO", scores=[6, 7, 7])          # 0.400 vs 0.412
    z = V.standings_values(s)
    assert abs(float(z[0])) < 0.05
    assert np.allclose(z[1:3], -z[0]) and z[3] == 0.0


def test_standings_values_1v2_solo_progress_wins():
    s = _game(3, "ONE_V_TWO", scores=[10, 5, 5])          # 10/15 vs 10/34
    z = V.standings_values(s)
    assert z[0] > 0.0 and np.allclose(z[0], 2 * (10 / 15 - 10 / 34))
    assert np.allclose(z[1:3], -z[0]) and z[3] == 0.0


def test_standings_values_1v2_duo_progress_wins():
    s = _game(3, "ONE_V_TWO", scores=[5, 15, 15])         # 5/15 vs 30/34
    z = V.standings_values(s)
    assert z[0] < 0.0 and z[1] > 0.0 and z[2] > 0.0
    assert float(z.min()) >= -1.0 and float(z.max()) <= 1.0


def test_standings_values_1v2_use_the_real_rule_once_a_side_qualifies():
    """Past a threshold the engine's own resolver decides the game, so the
    standings say exactly what the end of the round will."""
    both_tied = _game(3, "ONE_V_TWO", scores=[15, 17, 17])   # excess 0 vs 0
    assert V.standings_values(both_tied).tolist() == [0.0, 0.0, 0.0, 0.0]

    solo_by_one = _game(3, "ONE_V_TWO", scores=[16, 17, 17])  # excess 1 vs 0
    assert V.standings_values(solo_by_one).tolist() == [1.0, -1.0, -1.0, 0.0]

    duo_only = _game(3, "ONE_V_TWO", scores=[9, 20, 20])      # duo qualified
    assert V.standings_values(duo_only).tolist() == [-1.0, 1.0, 1.0, 0.0]

    for state in (both_tied, solo_by_one, duo_only):
        assert (V.standings_values(state).tolist()
                == V._team_side_values(
                    state, E.resolve_one_vs_two_winners(state)).tolist())


def test_truncation_weight_is_the_documented_one():
    assert V.TRUNCATION_Z_WEIGHT == 0.3


# ── whole random games, every mode ────────────────────────────────────────

MODES = [("INDIVIDUAL", 2, None), ("INDIVIDUAL", 3, None),
         ("INDIVIDUAL", 4, None), ("ONE_V_TWO", 3, None),
         ("TEAM", 4, "ADJACENT"), ("TEAM", 4, "OPPOSITE")]


@pytest.mark.parametrize("mode,n,layout", MODES)
def test_random_games_produce_sane_values(mode, n, layout):
    rng = random.Random(hash((mode, n, layout)) & 0xFFFF)
    reasons = set()
    for _ in range(40):
        s = _random_finished_game(rng, mode, n, layout)
        z = V.terminal_values(s)
        assert z.dtype == np.float32 and z.shape == (4,)
        assert np.isfinite(z).all() and np.abs(z).max() <= 1.0
        assert float(np.abs(z[n:]).sum()) == 0.0
        if mode == "INDIVIDUAL":
            assert abs(float(z.sum())) < 1e-5      # rank-linear is zero sum
            top = float(z[:n].max())
            best = sorted(i for i in range(n) if z[i] == top)
            live_winners = E.individual_winners(s)
            if not any(i in s.resigned for i in live_winners):
                assert best == live_winners
        else:
            assert set(np.abs(z[:n]).tolist()) <= {0.0, 1.0}
            reasons.add(s.game_result["reason"])
        if mode == "TEAM":
            assert abs(float(z.sum())) < 1e-5      # 2 + 2 seats stay zero sum
    if mode != "INDIVIDUAL":
        assert reasons                              # SCORE and/or FORFEIT seen
