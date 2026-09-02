"""Search tests (``docs/AI_DESIGN.md`` §1.6, gate G2).

Positions are built by hand so the right answer is known before the search
runs; nothing here depends on the Node oracle.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from splendor_ai.rules import engine as E
from splendor_ai.rules.actions import (
    BUY_BOARD_START, CHOOSE_TILE_START, NUM_ACTIONS, RESERVE_DECK_START,
)
from splendor_ai.rules.cards import CARDS, CARDS_BY_TIER, CARD_REWARD
from splendor_ai.search.evaluators import (
    GreedyValueEvaluator, RolloutEvaluator, UniformEvaluator, ZeroEncoder,
    state_encoder,
)
from splendor_ai.search.mcts import (
    MCTS, SearchConfig, run_search, seat_absolute, seat_relative,
    standings_values, terminal_values, _terminal_values,
)
from splendor_ai.search.scheduler import Scheduler, SearchSlot

T1_ZERO = [c.id for c in CARDS if c.tier == 1 and c.points == 0]


# ── position construction ─────────────────────────────────────────────────

def build(n=2, mode="INDIVIDUAL", layout=None, board=None, tiles=(),
          gems=None, current=0, players=()):
    """A hand-made position.  ``players[i]`` may set cards/reserved/gems/score.

    ``reserved`` entries are ``card_id`` (public) or ``(card_id, False)`` for a
    blind deck reserve.  Anything not placed is dealt to the board/decks.
    """
    used = set()
    specs = [dict(p) for p in players] + [{} for _ in range(n - len(players))]
    for p in specs:
        used |= set(p.get("cards", ()))
        for r in p.get("reserved", ()):
            used.add(r[0] if isinstance(r, tuple) else r)
    if board is None:
        board = []
        for t in range(3):
            row = [c for c in CARDS_BY_TIER[t] if c not in used][:4]
            board.append(row)
            used |= set(row)
    else:
        board = [list(r) for r in board]
        for row in board:
            used |= set(row)
    decks = [[c for c in CARDS_BY_TIER[t] if c not in used] for t in range(3)]

    s = E.new_game(n, mode, layout,
                   setup={"board": board, "decks": decks,
                          "tiles": list(tiles), "first": current})
    if gems is not None:
        s.gems = list(gems)
    for i, spec in enumerate(specs):
        p = s.players[i]
        p.cards = list(spec.get("cards", ()))
        p.discount = [0, 0, 0, 0, 0]
        for cid in p.cards:
            p.discount[CARD_REWARD[cid]] += 1
        p.reserved, p.reserved_public = [], []
        for r in spec.get("reserved", ()):
            cid, pub = r if isinstance(r, tuple) else (r, True)
            p.reserved.append(cid)
            p.reserved_public.append(pub)
        p.gems = list(spec.get("gems", (0, 0, 0, 0, 0, 0)))
        p.score = spec.get("score", 0)
        p.tiles = list(spec.get("tiles", ()))
    return s


def rotate_seats(state, shift):
    """Relabel the seating: seat ``i`` becomes seat ``(i + shift) % n``."""
    n = state.num_players
    t = state.clone()
    t.players = [state.players[(i - shift) % n].clone() for i in range(n)]
    t.current_player = (state.current_player + shift) % n
    t.round_start_player = (state.round_start_player + shift) % n
    t.resigned = [(i + shift) % n for i in state.resigned]
    if state.final_round_triggered_by is not None:
        t.final_round_triggered_by = (state.final_round_triggered_by + shift) % n
    return t


def tactical_win_position():
    """2p, seat 0 to move, score 12, and exactly one move that wins the game.

    Seat 0's tableau is four rose (colour 3) cards and it holds three rose
    tokens; every board card costs a colour it has none of, EXCEPT the tier-3
    card 75 (cost 7 rose, 4 points), which its 4 discounts + 3 tokens cover
    exactly.  Buying it takes seat 0 to 16 >= 15, which triggers the final
    round; seat 1 (score 0, no cards, no tokens) gets one move and the game
    ends 16-0.  Every other legal move leaves the game running.
    """
    rose = [c for c in T1_ZERO if CARD_REWARD[c] == 3][:4]
    board = [[25, 26, 27, 28],        # tier 1: 3 tokens of colours 4/0/1/2
             [40, 41, 42, 44],        # tier 2: 6 tokens of colours 0/1/2/4
             [75, 76, 77, 78]]        # tier 3: 7 tokens of colours 3/4/0/1
    s = build(2, board=board, gems=[4, 4, 4, 1, 4, 5], current=0,
              players=[{"cards": rose, "gems": [0, 0, 0, 3, 0, 0], "score": 12},
                       {}])
    target = BUY_BOARD_START + 2 * 4 + 0            # buy board tier 3 slot 0
    return s, target


def stuck_position():
    """2p, seat 0 to move; whatever it plays, seat 1 has no legal action.

    Seat 1 holds ten tokens (cannot take), three unaffordable reserved cards
    (cannot reserve) and cannot afford anything on the board — the real stuck
    state the validation run hit 4,051 times.  Its forced resign leaves fewer
    than two active seats, so the game ends immediately in seat 0's favour.
    """
    s = build(2, board=[[], [], [75, 76, 77, 78]],
              gems=[0, 0, 2, 4, 4, 0], current=0,
              players=[{"gems": [0, 0, 0, 0, 0, 0]},
                       {"gems": [4, 4, 2, 0, 0, 0],
                        "reserved": [70, 71, 72]}])
    s.decks = [[], [], s.decks[2]]
    s.deck_counts = [0, 0, len(s.decks[2])]
    return s


# ── terminal values (§1.2) ────────────────────────────────────────────────

def test_terminal_values_individual_rank_linear():
    s = build(4, players=[{"score": 10, "cards": T1_ZERO[:2]},
                          {"score": 7}, {"score": 3}, {"score": 1}])
    s.phase = E.PHASE_GAME_OVER
    z = _terminal_values(s)
    assert list(z) == pytest.approx([1.0, 1 / 3, -1 / 3, -1.0])


def test_terminal_values_ties_share_the_mean_rank_and_2p_is_pm1():
    s = build(2, players=[{"score": 9}, {"score": 9}])
    s.phase = E.PHASE_GAME_OVER
    assert list(_terminal_values(s)) == pytest.approx([0.0, 0.0, 0.0, 0.0])
    s = build(2, players=[{"score": 9}, {"score": 8}])
    s.phase = E.PHASE_GAME_OVER
    assert list(_terminal_values(s)) == pytest.approx([1.0, -1.0, 0.0, 0.0])
    # cards break a score tie: fewer cards ranks higher (server ordering)
    s = build(2, players=[{"score": 9, "cards": T1_ZERO[:3]},
                          {"score": 9, "cards": T1_ZERO[3:5]}])
    s.phase = E.PHASE_GAME_OVER
    assert list(_terminal_values(s)) == pytest.approx([-1.0, 1.0, 0.0, 0.0])


def test_terminal_values_resigned_seats_rank_last():
    s = build(3, players=[{"score": 0}, {"score": 5}, {"score": 2}])
    E.resign(s, 0)
    s.phase = E.PHASE_GAME_OVER
    z = _terminal_values(s)
    assert z[0] == -1.0 and z[1] == 1.0 and z[2] == 0.0


@pytest.mark.parametrize("mode,n,layout", [("ONE_V_TWO", 3, None),
                                           ("TEAM", 4, "ADJACENT")])
def test_terminal_values_team_modes(mode, n, layout):
    s = build(n, mode, layout)
    s.phase = E.PHASE_GAME_OVER
    s.game_result = {"reason": "SCORE", "winningTeamIds": [1]}
    z = _terminal_values(s)
    for i, p in enumerate(s.players):
        assert z[i] == (1.0 if p.team_id == 1 else -1.0)
    s.game_result = {"reason": "SCORE", "winningTeamIds": [0, 1]}
    assert list(_terminal_values(s)) == [0.0] * 4          # tie
    s.game_result = {"reason": "SCORE", "winningTeamIds": []}
    assert list(_terminal_values(s)) == [0.0] * 4          # nobody qualified
    s.game_result = {"reason": "FORFEIT", "forfeitingTeamId": 0,
                     "winningTeamIds": [1]}
    z = _terminal_values(s)
    for i, p in enumerate(s.players):
        assert z[i] == (-1.0 if p.team_id == 0 else 1.0)
    assert z[n:].tolist() == [0.0] * (4 - n)


def test_seat_relative_round_trip():
    z = np.array([0.25, -0.5, 0.75, -1.0], dtype=np.float32)
    for seat in range(4):
        rel = seat_relative(z, seat)
        assert rel[0] == z[seat]
        assert np.allclose(seat_absolute(rel, seat), z)


# ── PUCT ──────────────────────────────────────────────────────────────────

def test_puct_finds_the_move_that_wins_next_ply():
    s, target = tactical_win_position()
    # sanity: the position really has exactly one immediately winning move
    winners = []
    for a in E.legal_actions(s):
        t = s.clone()
        E.apply(t, a)
        if t.players[0].score >= 15:
            winners.append(a)
    assert winners == [target]

    cfg = SearchConfig(sims=300, temperature_plies=0, universes=4)
    res = run_search(s, 0, UniformEvaluator(), ZeroEncoder(4), cfg,
                     np.random.default_rng(0))
    others = np.delete(res.visits, target)
    assert res.action == target
    assert res.visits[target] > 2 * others.max()
    assert res.root_value[0] > 0.5
    assert res.policy_target[target] == res.policy_target.max()
    assert res.stats["terminal"] > 0


def test_policy_target_is_a_distribution_over_legal_actions():
    s = E.new_game(3, rng=random.Random(2))
    mask = np.asarray(E.legal_mask(s), dtype=bool)
    for root in ("puct", "gumbel"):
        cfg = SearchConfig(sims=120, root=root, gumbel_m=8, temperature_plies=0)
        res = run_search(s, s.current_player, UniformEvaluator(), ZeroEncoder(4),
                         cfg, np.random.default_rng(1))
        t = res.policy_target
        assert t.shape == (NUM_ACTIONS,)
        assert (t >= 0).all()
        assert t.sum() == pytest.approx(1.0, abs=1e-5)
        assert not t[~mask].any(), "policy target must be zero on illegal moves"
        assert mask[res.action]
        assert int(res.visits.sum()) > 0


def test_forced_playouts_are_pruned_out_of_the_policy_target():
    s = E.new_game(2, rng=random.Random(6))
    rng = np.random.default_rng(3)
    common = dict(sims=200, temperature_plies=0, forced_playouts_k=2.0)
    kept = run_search(s, 0, UniformEvaluator(), ZeroEncoder(4),
                      SearchConfig(prune_policy_target=False, **common),
                      np.random.default_rng(3))
    pruned = run_search(s, 0, UniformEvaluator(), ZeroEncoder(4),
                        SearchConfig(prune_policy_target=True, **common),
                        np.random.default_rng(3))
    assert np.array_equal(kept.visits, pruned.visits)      # same search
    assert pruned.policy_target.max() > kept.policy_target.max()
    assert (pruned.policy_target > 0).sum() <= (kept.policy_target > 0).sum()
    assert pruned.policy_target.sum() == pytest.approx(1.0, abs=1e-5)


def test_dirichlet_noise_only_touches_the_root_and_only_legal_actions():
    s = E.new_game(2, rng=random.Random(8))
    mask = np.asarray(E.legal_mask(s), dtype=bool)
    cfg = SearchConfig(sims=60, noise=True, temperature_plies=0)
    tree = MCTS(cfg, np.random.default_rng(4))
    for _ in range(cfg.sims):
        leaf = tree.select_leaf(s, 0)
        if leaf is None:
            continue
        p, v = UniformEvaluator().evaluate([None], leaf.mask[None])
        tree.backup(leaf.token, p[0], v[0])
    P = np.array(tree.root.P)
    assert not P[~mask].any()
    assert P.sum() == pytest.approx(1.0, abs=1e-5)
    assert P[mask].std() > 0            # noise made the priors non-uniform


def test_backup_maps_leaf_relative_values_onto_absolute_seats():
    s = E.new_game(2, rng=random.Random(11))
    s.current_player = 1
    tree = MCTS(SearchConfig(sims=1, temperature_plies=0),
                np.random.default_rng(0))
    leaf = tree.select_leaf(s, 1)
    assert leaf is not None and leaf.seat == 1
    # index j is absolute seat (j + leaf_seat) % 4: seat 1 gets +1, seat 0 -1
    tree.backup(leaf.token, np.ones(NUM_ACTIONS) * leaf.mask,
                np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32))
    assert list(tree.result().root_value) == [-1.0, 1.0, 0.0, 0.0]


# ── symmetries ────────────────────────────────────────────────────────────

def test_2p_root_value_is_antisymmetric():
    s = E.new_game(2, rng=random.Random(12))
    cfg = SearchConfig(sims=150, temperature_plies=0, deck_reserve_penalty=None)
    res = run_search(s, s.current_player, RolloutEvaluator("random", 40,
                                                           np.random.default_rng(5)),
                     state_encoder, cfg, np.random.default_rng(5))
    assert res.root_value[0] == pytest.approx(-res.root_value[1], abs=1e-6)
    assert res.root_value[2] == 0.0 and res.root_value[3] == 0.0
    assert abs(res.root_value[0]) < 0.9        # not a solved position


def test_seat_relabelling_gives_the_rotated_result():
    """Searching a rotated seating must rotate the answer, nothing else."""
    s = E.new_game(3, rng=random.Random(13))
    for _ in range(6):                          # a few plies of play
        E.apply(s, E.legal_actions(s)[3])
    assert not any(False in p.reserved_public for p in s.players)
    cfg = SearchConfig(sims=200, temperature_plies=0, deck_reserve_penalty=None)

    def search(state):
        return run_search(state, state.current_player,
                          RolloutEvaluator("greedy", 40), state_encoder, cfg,
                          np.random.default_rng(17))

    base = search(s)
    n = s.num_players
    for shift in (1, 2):
        rot = search(rotate_seats(s, shift))
        expected = np.zeros(4, dtype=np.float32)
        for i in range(n):                      # seat i moved to (i + shift) % n
            expected[(i + shift) % n] = base.root_value[i]
        assert np.allclose(rot.root_value, expected, atol=1e-6)
        assert np.array_equal(rot.visits, base.visits)
        assert np.allclose(rot.policy_target, base.policy_target, atol=1e-6)


# ── stuck / terminal paths ────────────────────────────────────────────────

def test_stuck_leaves_resign_and_the_search_scores_them():
    s = stuck_position()
    assert not E.is_stuck(s)
    for a in E.legal_actions(s):
        t = s.clone()
        E.apply(t, a)
        assert E.is_stuck(t), f"seat 1 should be stuck after {a}"

    cfg = SearchConfig(sims=80, temperature_plies=0)
    res = run_search(s, 0, UniformEvaluator(), ZeroEncoder(4), cfg,
                     np.random.default_rng(0))
    assert res.stats["stuck"] >= res.stats["sims_run"] - 1
    # only the root itself was ever evaluated; every child is a stuck-resign
    # that ends the game, so no other simulation needed the evaluator
    assert res.stats["evaluated"] == 1
    assert res.stats["terminal"] == res.stats["sims_run"] - 1
    assert res.root_value[0] > 0.98 and res.root_value[1] < -0.98
    assert res.policy_target.sum() == pytest.approx(1.0, abs=1e-5)


def test_search_refuses_a_stuck_root():
    s = stuck_position()
    E.apply(s, E.legal_actions(s)[0])           # now seat 1 is stuck
    assert E.is_stuck(s)
    with pytest.raises(ValueError):
        run_search(s, 1, UniformEvaluator(), ZeroEncoder(4),
                   SearchConfig(sims=4), np.random.default_rng(0))


def test_terminal_leaves_need_no_evaluator_call():
    s, target = tactical_win_position()
    E.apply(s, target)                           # final round, seat 1 to move
    cfg = SearchConfig(sims=40, temperature_plies=0)
    tree = MCTS(cfg, np.random.default_rng(0))
    leaf = tree.select_leaf(s, 1)                # root expansion needs a call
    assert leaf is not None
    p, v = UniformEvaluator().evaluate([None], leaf.mask[None])
    tree.backup(leaf.token, p[0], v[0])
    assert tree.select_leaf(s, 1) is None        # every child ends the game
    assert tree.stats["terminal"] >= 1


# ── Gumbel root ───────────────────────────────────────────────────────────

def test_gumbel_root_returns_a_valid_improved_policy():
    s = E.new_game(2, rng=random.Random(21))
    mask = np.asarray(E.legal_mask(s), dtype=bool)
    cfg = SearchConfig(sims=96, root="gumbel", gumbel_m=8, temperature_plies=0)
    res = run_search(s, 0, GreedyValueEvaluator(), state_encoder, cfg,
                     np.random.default_rng(2))
    t = res.policy_target
    assert (t >= 0).all() and t.sum() == pytest.approx(1.0, abs=1e-5)
    assert not t[~mask].any()
    assert mask[res.action]
    # sequential halving concentrates the visits on a handful of candidates
    assert 0 < (res.visits > 0).sum() <= 8
    assert res.visits.sum() >= cfg.sims - 4


def test_gumbel_root_solves_the_tactical_position():
    s, target = tactical_win_position()
    cfg = SearchConfig(sims=200, root="gumbel", gumbel_m=16, temperature_plies=0)
    res = run_search(s, 0, UniformEvaluator(), ZeroEncoder(4), cfg,
                     np.random.default_rng(7))
    assert res.action == target
    assert res.policy_target[target] == res.policy_target.max()


# ── anti-clairvoyance penalty ─────────────────────────────────────────────

def test_deck_reserve_penalty_lowers_deck_reserve_visits():
    """The penalty must bite on the deck-reserve actions and nothing else."""
    s = E.new_game(2, rng=random.Random(31))
    deck = list(range(RESERVE_DECK_START, RESERVE_DECK_START + 3))
    cfg_off = SearchConfig(sims=400, temperature_plies=0,
                           deck_reserve_penalty=None, forced_playouts_k=0.0)
    cfg_on = SearchConfig(sims=400, temperature_plies=0,
                          deck_reserve_penalty=(1.0, 1.0, 1.0),
                          forced_playouts_k=0.0)
    off = run_search(s, 0, UniformEvaluator(), ZeroEncoder(4), cfg_off,
                     np.random.default_rng(0))
    on = run_search(s, 0, UniformEvaluator(), ZeroEncoder(4), cfg_on,
                    np.random.default_rng(0))
    assert on.visits[deck].sum() < off.visits[deck].sum()


# ── scheduler ─────────────────────────────────────────────────────────────

def test_scheduler_batches_one_leaf_per_tree():
    states = [E.new_game(2, rng=random.Random(40 + i)) for i in range(4)]
    cfg = SearchConfig(sims=50, temperature_plies=0)
    slots = [SearchSlot(MCTS(cfg, np.random.default_rng(i)), st,
                        st.current_player) for i, st in enumerate(states)]

    calls = []

    class Counting(UniformEvaluator):
        def evaluate(self, obs, mask):
            calls.append(len(mask))
            return super().evaluate(obs, mask)

    sched = Scheduler(slots, Counting(), ZeroEncoder(4))
    report = sched.run()
    assert report["sims"] == 4 * cfg.sims
    assert max(calls) <= 4                       # never more than one per tree
    assert report["batches"] == len(calls)
    assert report["sims_per_s"] > 0
    for res in sched.results():
        assert res.policy_target.sum() == pytest.approx(1.0, abs=1e-5)
        assert res.stats["sims_run"] == cfg.sims


def test_scheduler_matches_single_tree_search():
    """Batching must not change the algorithm (batch=1 reproduces it)."""
    s = E.new_game(2, rng=random.Random(41))
    cfg = SearchConfig(sims=60, temperature_plies=0)
    single = run_search(s, 0, UniformEvaluator(), ZeroEncoder(4), cfg,
                        np.random.default_rng(9))
    slot = SearchSlot(MCTS(cfg, np.random.default_rng(9)), s, 0)
    Scheduler([slot], UniformEvaluator(), ZeroEncoder(4)).run()
    batched = slot.result()
    assert np.array_equal(single.visits, batched.visits)
    assert np.allclose(single.root_value, batched.root_value)


# ── evaluators ────────────────────────────────────────────────────────────

def test_rollout_evaluator_returns_leaf_relative_values():
    s, target = tactical_win_position()
    ev = RolloutEvaluator("greedy", 30)
    mask = np.asarray(E.legal_mask(s), dtype=bool)[None]
    priors, values = ev.evaluate([(s, 0)], mask)
    assert priors.shape == (1, NUM_ACTIONS) and values.shape == (1, 4)
    assert priors[0][~mask[0]].sum() == 0
    assert priors.sum() == pytest.approx(1.0, abs=1e-6)
    # greedy plays the winning buy, so seat 0 (index 0 = leaf seat) wins
    assert values[0][0] == 1.0
    assert seat_absolute(values[0], 0)[1] == -1.0


def test_greedy_value_evaluator_prefers_the_stronger_side():
    s = build(2, players=[{"score": 10, "cards": T1_ZERO[:6]}, {"score": 1}])
    ev = GreedyValueEvaluator()
    z = ev.value(s)
    assert z[0] > 0.5 > z[1] and z[1] < 0
    assert list(z[2:]) == [0.0, 0.0]


def test_standings_values_score_a_truncated_game():
    s = build(3, players=[{"score": 8}, {"score": 3}, {"score": 5}])
    assert list(standings_values(s)) == pytest.approx([1.0, -1.0, 0.0, 0.0])
    t = build(4, "TEAM", "ADJACENT",
              players=[{"score": 9}, {"score": 9}, {"score": 2}, {"score": 2}])
    assert list(standings_values(t)) == [1.0, 1.0, -1.0, -1.0]


# ── same-player edges and the values.py hand-over ─────────────────────────

def test_choose_tile_edges_are_flagged_same_player():
    """A buy that qualifies two nobles keeps the turn with the same seat."""
    from splendor_ai.rules.cards import CARD_REWARD
    board_card = 0
    cards = []
    for colour, count in enumerate([3, 4, 3, 0, 3]):
        pool = [c for c in T1_ZERO if CARD_REWARD[c] == colour
                and c != board_card]
        cards += pool[:count]
    s = build(2, board=[[board_card, 1, 2, 3], [], []], tiles=[0, 5],
              gems=[4, 4, 4, 4, 4, 5],
              players=[{"cards": cards, "gems": [1, 1, 1, 1, 1, 0]}, {}])
    buy = BUY_BOARD_START + 0
    probe = s.clone()
    E.apply(probe, buy)
    assert probe.pending_tile_choice and probe.current_player == 0

    cfg = SearchConfig(sims=120, temperature_plies=0)
    tree = MCTS(cfg, np.random.default_rng(0))
    for _ in range(cfg.sims):
        leaf = tree.select_leaf(s, 0)
        if leaf is None:
            continue
        p, v = UniformEvaluator().evaluate([None], leaf.mask[None])
        tree.backup(leaf.token, p[0], v[0])
    assert tree.root.N[buy] > 0
    assert tree.root.same_player[buy] is True
    child = tree.root.children[buy]
    assert child.acting_seat == 0                    # still seat 0 to choose
    tile_edges = [a for a in range(CHOOSE_TILE_START, NUM_ACTIONS)
                  if child.N[a] > 0]
    assert tile_edges
    for a in tile_edges:
        assert child.same_player[a] is False         # the tile choice ends it


def test_terminal_values_switches_to_values_module_when_it_lands(monkeypatch):
    """The guarded import must prefer ``splendor_ai.values`` once it exists."""
    import sys
    import types

    import splendor_ai
    import splendor_ai.search.mcts as mcts

    fake = types.ModuleType("splendor_ai.values")
    sentinel = np.array([0.5, -0.5, 0.0, 0.0], dtype=np.float32)
    fake.terminal_values = lambda state: sentinel
    monkeypatch.setitem(sys.modules, "splendor_ai.values", fake)
    monkeypatch.setattr(splendor_ai, "values", fake, raising=False)
    mcts.reset_terminal_values()
    try:
        s = build(2, players=[{"score": 3}, {"score": 1}])
        s.phase = E.PHASE_GAME_OVER
        assert list(terminal_values(s)) == [0.5, -0.5, 0.0, 0.0]
    finally:
        # drop the cache so the next caller re-resolves (monkeypatch removes
        # the fake module only after this test returns)
        mcts.reset_terminal_values()
    assert mcts._TERMINAL_VALUES is None
