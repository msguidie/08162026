"""Bots, the game driver, and gate G2 (``docs/AI_DESIGN.md`` §2).

G2 has two halves:

* ``GreedyBot`` beats ``RandomBot`` >= 95% in every mode — runs by default
  (~3 s for 100 games per mode);
* NN-free ``MctsBot`` @400 sims beats ``GreedyBot`` >= 75% in 2p over >= 100
  paired, seat-swapped games — that is ~30 minutes of CPU, so it only runs
  with ``SPLENDOR_G2_MCTS=1``.  A cheap smoke version runs always.
"""

from __future__ import annotations

import os
import random
import time

import numpy as np
import pytest

from splendor_ai.bots import GreedyBot, MctsBot, RandomBot, play_game
from splendor_ai.rules import engine as E
from splendor_ai.rules.actions import (
    BUY_BOARD_START, BUY_RESERVED_START, CHOOSE_TILE_START, NUM_TAKE_ACTIONS,
    RESERVE_BOARD_START, TAKE_PATTERNS,
)
from splendor_ai.rules.cards import CARDS, CARD_POINTS, CARD_REWARD
from splendor_ai.search.evaluators import (
    RolloutEvaluator, UniformEvaluator, ZeroEncoder, greedy_action,
    state_encoder,
)
from splendor_ai.search.mcts import SearchConfig, run_search
from splendor_ai.tests.test_mcts import T1_ZERO, build, stuck_position

MODES = [("INDIVIDUAL", 2, None), ("INDIVIDUAL", 3, None),
         ("INDIVIDUAL", 4, None), ("ONE_V_TWO", 3, None),
         ("TEAM", 4, "ADJACENT")]

SLOW_MCTS = os.environ.get("SPLENDOR_G2_MCTS") == "1"


def discount_cards(discount, exclude=()):
    """Zero-point tier-1 cards giving exactly ``discount`` (and no points)."""
    out = []
    for colour, count in enumerate(discount):
        pool = [c for c in T1_ZERO
                if CARD_REWARD[c] == colour and c not in exclude]
        assert count <= len(pool)
        out += pool[:count]
    return out


# ── bot contracts ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,n,layout", MODES)
def test_bots_only_ever_play_legal_actions(mode, n, layout):
    """Fuzz every bot over random play in every mode."""
    rng = np.random.default_rng(0)
    bots = [RandomBot(), GreedyBot()]
    for seed in range(6):
        s = E.new_game(n, mode, layout, rng=random.Random(seed))
        for _ in range(120):
            if s.phase != E.PHASE_PLAYING:
                break
            mask = E.legal_mask(s)
            stuck = not any(mask)
            for bot in bots:
                a = bot.act(s, s.current_player, rng)
                if stuck:
                    assert a is None
                else:
                    assert a is not None and mask[a], f"{bot.name} played {a}"
            if stuck:
                E.resign(s, s.current_player)
            else:
                legal = [i for i, v in enumerate(mask) if v]
                E.apply(s, legal[int(rng.integers(len(legal)))])


def test_greedy_buys_the_most_points_then_reserved_then_cheapest():
    board = [[0, 1, 2, 3], [40, 41, 42, 43], [75, 76, 77, 78]]
    rich = discount_cards([6, 6, 6, 6, 6], exclude=set(board[0]))
    # 4 points on the board beats a 1-point reserved card.
    s = build(2, board=board, players=[
        {"cards": rich, "gems": [2, 2, 2, 2, 2, 0], "reserved": [35]}, {}])
    assert CARD_POINTS[s.board[2][0]] == 4
    assert greedy_action(s) == BUY_BOARD_START + 2 * 4 + 0

    # Equal points (1 vs 1): the reserved copy wins the tie.
    s2 = build(2, board=[[36], [], []], players=[
        {"cards": discount_cards([4, 4, 4, 4, 4], exclude={35, 36}),
         "reserved": [35]}, {}])
    assert CARD_POINTS[35] == CARD_POINTS[36] == 1
    assert greedy_action(s2) == BUY_RESERVED_START + 0

    # Equal points on the board: the cheaper printed cost wins.
    s3 = build(2, board=[[0, 25], [], []], players=[
        {"cards": discount_cards([4, 4, 4, 4, 4], exclude={0, 25}),
         "gems": [1, 1, 1, 1, 1, 0]}, {}])
    assert CARD_POINTS[0] == CARD_POINTS[25] == 0
    assert sum(CARDS[25].cost) < sum(CARDS[0].cost)
    assert greedy_action(s3) == BUY_BOARD_START + 1


def test_greedy_takes_gems_that_shrink_the_cheapest_cards_shortfall():
    """Only one board card is nearly affordable; the take must serve it."""
    board = [[7], [], []]                 # card 7: cost (1,2,1,1,0)-ish
    s = build(2, board=board, gems=[4, 4, 4, 4, 4, 5],
              players=[{"gems": [0, 0, 0, 0, 0, 0]}, {}])
    cost = CARDS[board[0][0]].cost
    need = [c for c in range(5) if cost[c] > 0]
    a = greedy_action(s)
    assert a < NUM_TAKE_ACTIONS
    taken = TAKE_PATTERNS[a]
    assert len(taken) == 3
    assert set(taken) <= set(need), (taken, cost)


def test_greedy_reserves_when_no_take_and_no_buy_is_possible():
    """Ten tokens (no take) and nothing affordable -> reserve the best card."""
    s = build(2, board=[[], [], [70, 71, 72, 73]], gems=[0, 0, 0, 0, 0, 0],
              players=[{"gems": [4, 4, 2, 0, 0, 0]}, {}])
    mask = E.legal_mask(s)
    assert not any(mask[:NUM_TAKE_ACTIONS])
    a = greedy_action(s, mask)
    assert RESERVE_BOARD_START <= a < RESERVE_BOARD_START + 12
    slot = a - (RESERVE_BOARD_START + 2 * 4)
    assert CARD_POINTS[s.board[2][slot]] == max(CARD_POINTS[c]
                                                for c in s.board[2])


def test_greedy_resolves_a_pending_noble_choice():
    """A buy that qualifies two tiles leaves a same-seat CHOOSE_TILE decision."""
    board_card = 0                                   # reward 0, cheap
    cards = discount_cards([3, 4, 3, 0, 3], exclude={board_card})
    s = build(2, board=[[board_card, 1, 2, 3], [], []], tiles=[0, 5],
              gems=[4, 4, 4, 4, 4, 5],
              players=[{"cards": cards, "gems": [1, 1, 1, 1, 1, 0]}, {}])
    buy = BUY_BOARD_START + 0
    assert E.legal_mask(s)[buy]
    E.apply(s, buy)
    assert s.pending_tile_choice and len(s.pending_tile_choice) == 2
    assert s.current_player == 0                     # same seat acts again
    mask = E.legal_mask(s)
    assert [a for a in range(65) if mask[a]] == [CHOOSE_TILE_START,
                                                 CHOOSE_TILE_START + 1]
    a = greedy_action(s, mask)
    assert a == CHOOSE_TILE_START                    # first pending tile
    E.apply(s, a)
    assert s.current_player == 1


def test_greedy_and_mcts_return_none_when_stuck():
    s = stuck_position()
    E.apply(s, E.legal_actions(s)[0])
    assert E.is_stuck(s)
    rng = np.random.default_rng(0)
    assert GreedyBot().act(s, s.current_player, rng) is None
    assert RandomBot().act(s, s.current_player, rng) is None
    bot = MctsBot(SearchConfig(sims=8), UniformEvaluator(), ZeroEncoder(4))
    assert bot.act(s, s.current_player, rng) is None


# ── the driver ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,n,layout", MODES)
def test_play_game_produces_a_consistent_outcome(mode, n, layout):
    out = play_game([GreedyBot()] + [RandomBot()] * (n - 1), mode, n, layout,
                    seed=7)
    assert out["plies"] > 0
    assert len(out["values"]) == 4
    assert all(v == 0.0 for v in out["values"][n:])
    assert out["reason"] in ("SCORE", "FORFEIT", "TRUNCATED")
    assert not out["truncated"]
    if mode == "INDIVIDUAL":
        assert max(out["values"][:n]) == pytest.approx(1.0)
    else:
        assert set(out["values"][:n]) <= {1.0, -1.0, 0.0}
    for w in out["winners"]:
        assert out["values"][w] == max(out["values"][:n])


def test_play_game_is_reproducible_and_paired_by_seed():
    a = play_game([GreedyBot(), RandomBot()], seed=3)
    b = play_game([GreedyBot(), RandomBot()], seed=3)
    assert a["actions"] == b["actions"] and a["values"] == b["values"]
    c = play_game([RandomBot(), GreedyBot()], seed=3)     # same deal, swapped
    assert c["actions"] != a["actions"]


def test_play_game_truncates_and_scores_by_standings():
    out = play_game([GreedyBot(), GreedyBot()], seed=5, max_plies=8)
    assert out["truncated"] and out["reason"] == "TRUNCATED"
    assert out["plies"] == 8
    assert sorted(out["values"][:2]) in ([-1.0, 1.0], [0.0, 0.0])


def test_play_game_resigns_stuck_seats():
    """Random play in 4p reaches genuinely stuck seats; they must resign."""
    total_stuck = 0
    for seed in range(40):
        out = play_game([RandomBot()] * 4, "INDIVIDUAL", 4, seed=seed)
        total_stuck += out["stuck_resigns"]
        assert out["reason"] in ("SCORE", "FORFEIT")
    assert total_stuck > 0


def test_play_game_rejects_an_illegal_bot():
    class Cheater:
        name = "cheater"

        def act(self, state, seat, rng):
            mask = E.legal_mask(state)
            return next(i for i in range(65) if not mask[i])

    with pytest.raises(ValueError, match="illegal"):
        play_game([Cheater(), RandomBot()], seed=1)


# ── G2: greedy vs random ──────────────────────────────────────────────────

def side_win_rate(mode, n, layout, greedy_seats_fn, games=100, seed0=1000,
                  max_plies=400):
    """Win rate of the greedy side; a tie counts as half a win."""
    wins = 0.0
    for g in range(games):
        gs = greedy_seats_fn(g)
        bots = [GreedyBot() if i in gs else RandomBot() for i in range(n)]
        out = play_game(bots, mode, n, layout, seed=seed0 + g,
                        max_plies=max_plies)
        v = out["values"]
        mine = max(v[i] for i in gs)
        theirs = max(v[i] for i in range(n) if i not in gs)
        wins += 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
    return wins / games


@pytest.mark.parametrize("mode,n,layout,seats", [
    ("INDIVIDUAL", 2, None, lambda g: {g % 2}),
    ("INDIVIDUAL", 3, None, lambda g: {g % 3}),
    ("INDIVIDUAL", 4, None, lambda g: {g % 4}),
    ("ONE_V_TWO", 3, None, lambda g: {0} if g % 2 else {1, 2}),
    ("TEAM", 4, "ADJACENT", lambda g: {0, 1} if g % 2 else {2, 3}),
])
def test_g2_greedy_beats_random_in_every_mode(mode, n, layout, seats):
    rate = side_win_rate(mode, n, layout, seats, games=100)
    assert rate >= 0.95, f"{mode} {n}p: greedy won {rate:.2%} vs random"


# ── G2: NN-free search vs greedy ──────────────────────────────────────────

def mcts_vs_greedy(pairs, sims, max_plies=400, root="puct"):
    """Paired seat-swapped 2p games; returns the search bot's win rate."""
    cfg = SearchConfig(sims=sims, temperature_plies=0, root=root)
    wins = 0.0
    games = 0
    for g in range(pairs):
        for seat in (0, 1):
            bot = MctsBot(cfg, RolloutEvaluator("greedy"), state_encoder)
            bots = [bot, GreedyBot()] if seat == 0 else [GreedyBot(), bot]
            out = play_game(bots, "INDIVIDUAL", 2, None, seed=5000 + g,
                            max_plies=max_plies)
            v = out["values"]
            wins += (1.0 if v[seat] > v[1 - seat]
                     else 0.5 if v[seat] == v[1 - seat] else 0.0)
            games += 1
    return wins / games


def test_mcts_bot_plays_whole_games():
    """Smoke: the search bot drives real games (PUCT and Gumbel roots)."""
    for root in ("puct", "gumbel"):
        cfg = SearchConfig(sims=40, root=root, gumbel_m=8, temperature_plies=0)
        bot = MctsBot(cfg, RolloutEvaluator("greedy", max_plies=30),
                      state_encoder)
        out = play_game([bot, GreedyBot()], seed=11, max_plies=120)
        assert out["plies"] > 4
        assert bot.last_result is not None
        assert bot.last_result.stats["sims_run"] == cfg.sims


@pytest.mark.skipif(not SLOW_MCTS,
                    reason="G2 search gate: set SPLENDOR_G2_MCTS=1 (~30 min)")
def test_g2_mcts400_beats_greedy_2p():
    rate = mcts_vs_greedy(pairs=100, sims=400)
    assert rate >= 0.75, f"MCTS@400 won {rate:.2%} vs greedy (need 75%)"


def test_search_throughput_is_sane():
    """Rough sims/s floor — catches an accidental O(n^2) in the tree loop."""
    s = E.new_game(2, rng=random.Random(1))
    cfg = SearchConfig(sims=200, temperature_plies=0)
    t0 = time.perf_counter()
    run_search(s, 0, UniformEvaluator(), ZeroEncoder(4), cfg,
               np.random.default_rng(0))
    uniform = cfg.sims / (time.perf_counter() - t0)
    t0 = time.perf_counter()
    run_search(s, 0, RolloutEvaluator("greedy"), state_encoder, cfg,
               np.random.default_rng(0))
    rollout = cfg.sims / (time.perf_counter() - t0)
    print(f"\nsims/s: uniform={uniform:,.0f} rollout(greedy)={rollout:,.0f}")
    assert uniform > 500
    assert rollout > 100
