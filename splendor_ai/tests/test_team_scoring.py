"""Team statistics, qualification thresholds, winners and rating deltas.

Random play reaches most of these, but the ties and the JS truthiness quirks
deserve to be pinned down explicitly against the Node implementation.
"""

import math

import pytest

from splendor_ai.rules import engine as E
from splendor_ai.tests.oracle import requires_node
from splendor_ai.tests.positions import position
from splendor_ai.tests.oracle import Position

pytestmark = requires_node

FULL_GEMS = [7, 7, 7, 7, 7, 5]


def _probe(mode, layout, scores, **extra):
    spec = position(n=len(scores), mode=mode, layout=layout, gems=FULL_GEMS,
                    players=[{"gems": [0] * 6, "score": s} for s in scores],
                    **extra)
    return Position({**spec, "ops": [["probe"]]})


def _node_stats(p):
    """Node's getTeamStats with JSON's null standing in for -Infinity."""
    out = []
    for t in p.results[0]["teamStats"]:
        t = dict(t)
        if t["secondScore"] is None:
            t["secondScore"] = -math.inf
        out.append(t)
    return out


@pytest.mark.parametrize("mode,layout,scores", [
    ("TEAM", "ADJACENT", [10, 5, 8, 9]),
    ("TEAM", "OPPOSITE", [10, 5, 8, 9]),
    ("TEAM", "ADJACENT", [31, 0, 0, 0]),
    ("TEAM", "ADJACENT", [20, 12, 20, 12]),
    ("TEAM", "OPPOSITE", [16, 16, 16, 16]),
    ("ONE_V_TWO", None, [15, 20, 14]),
    ("ONE_V_TWO", None, [0, 0, 0]),
    ("ONE_V_TWO", None, [40, 20, 20]),
])
def test_team_stats_and_qualification_match_node(mode, layout, scores):
    p = _probe(mode, layout, scores)
    assert E.team_stats(p.state) == _node_stats(p)
    assert E.qualifying_team_ids(p.state) == p.results[0]["qualifyingTeamIds"]
    assert E.rating_changes(p.state) == p.results[0]["rating"]


def test_one_v_two_solo_second_score_is_negative_infinity():
    p = _probe("ONE_V_TWO", None, [10, 5, 5])
    stats = E.team_stats(p.state)
    assert stats[0]["secondScore"] == -math.inf
    assert stats[1]["secondScore"] == 5
    assert stats == _node_stats(p)


@pytest.mark.parametrize("scores,expect", [
    # ADJACENT seats: team0 = seats {0,1}, team1 = seats {2,3}
    ([31, 5, 31, 5], [0, 1]),          # identical totals and card counts -> draw
    ([40, 5, 31, 5], [0]),
    ([31, 5, 40, 5], [1]),
    ([31, 0, 31, 1], [1]),             # team1's second score beats team0's
])
def test_team_winner_resolution_matches_node(scores, expect):
    p = _probe("TEAM", "ADJACENT", scores)
    qual = E.qualifying_team_ids(p.state)
    assert qual == p.results[0]["qualifyingTeamIds"]
    assert E.resolve_team_winners(p.state, qual) == expect


@pytest.mark.parametrize("scores,expect", [
    ([15, 20, 14], [0, 1]),            # both exactly at the bar -> excess 0 == 0
    ([14, 20, 14], [1]),               # solo short, duo exactly at 34
    ([14, 20, 13], []),                # nobody qualifies
    ([20, 20, 19], [0, 1]),            # equal excess (5 vs 5)
    ([21, 20, 19], [0]),
    ([15, 25, 20], [1]),
])
def test_one_v_two_winner_resolution_matches_node(scores, expect):
    p = _probe("ONE_V_TWO", None, scores)
    assert E.resolve_one_vs_two_winners(p.state) == expect


def test_empty_winning_team_ids_is_truthy_in_js_and_pays_three():
    """`state.gameResult?.winningTeamIds` is truthy for `[]` in JavaScript, so
    a team game that ends with no qualifying team pays 3 to everyone rather
    than falling through to the individual ranking."""
    spec = position(n=3, mode="ONE_V_TWO", gems=FULL_GEMS,
                    players=[{"gems": [0] * 6, "score": 9},
                             {"gems": [0] * 6, "score": 3},
                             {"gems": [0] * 6, "score": 1}],
                    phase="GAME_OVER",
                    gameResult={"reason": "SCORE", "winningTeamIds": []})
    p = Position({**spec, "ops": [["probe"]]})
    assert p.results[0]["rating"] == [3, 3, 3]
    assert E.rating_changes(p.state) == [3, 3, 3]


@pytest.mark.parametrize("scores,cards", [
    ([9, 9, 3, 1], [2, 2, 1, 1]),      # a true tie at the top (score AND cards)
    ([9, 9, 3, 1], [2, 3, 1, 1]),      # fewer cards wins the tie
    ([5, 5, 5, 5], [1, 1, 1, 1]),      # four-way tie -> everyone rank 0
    ([7, 3, 3, 0], [4, 2, 2, 9]),
    ([0, 0, 0, 0], [0, 1, 2, 3]),
])
def test_individual_rating_ladder_matches_node(scores, cards):
    pool = list(range(40))
    players, idx = [], 0
    for score, count in zip(scores, cards):
        players.append({"gems": [0] * 6, "score": score,
                        "cards": pool[idx:idx + count]})
        idx += count
    spec = position(n=4, gems=FULL_GEMS, players=players)
    pos = Position({**spec, "ops": [["probe"]]})
    assert E.rating_changes(pos.state) == pos.results[0]["rating"]
    assert E.individual_winners(pos.state) == sorted(
        i for i, r in enumerate(pos.results[0]["rating"]) if r == 5)
