"""Arena, seat rotation, the anchor ladder and the Bradley–Terry fit.

Covers the three things that make an arena number trustworthy
(``docs/AI_DESIGN.md`` §1.7, ``docs/research/judges.md``):

* the **fit** recovers ratings that were put in by construction, is pinned to
  the anchor and reads draws as half a win;
* the **seat rotation** gives every bot every seat (INDIVIDUAL) and every role
  (1v2 solo/duo, 2v2 both sides), with paired seeds inside a group;
* a **whole run** in all five modes produces a report, with the STALE and
  truncation buckets wired through and identical results however many worker
  processes are used.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter

import numpy as np
import pytest

from splendor_ai import anchors as A
from splendor_ai import arena
from splendor_ai.arena import (
    ELO_SCALE, MODES, ArenaResults, build_schedule, build_tables,
    fit_bradley_terry, game_values, pair_compositions, pairwise_from_game,
    parse_mode, render_markdown, run_matches, seat_arrangements, seat_credit,
    table_credit, write_reports,
)

ALL_MODES = ["ind2", "ind3", "ind4", "ovt", "team"]


# ── anchors ───────────────────────────────────────────────────────────────

def test_anchor_ladder_is_the_frozen_five():
    assert A.ANCHOR_LADDER == ("random", "greedy", "mcts40", "mcts160",
                               "mcts640")
    assert A.ANCHOR_SIMS == {"mcts40": 40, "mcts160": 160, "mcts640": 640}
    for name, sims in A.ANCHOR_SIMS.items():
        bot = A.make_bot(name)
        assert bot.cfg.sims == sims
        # Evaluation-mode search: the training-only devices are off, and the
        # rungs differ in nothing but the simulation budget.
        assert bot.cfg.noise is False
        assert bot.cfg.forced_playouts_k == 0.0
        assert bot.cfg.temperature_plies == 0
        assert bot.evaluator.policy == "greedy"
        other = A.make_bot("mcts160")
        for field in ("c_puct", "universes", "root", "deck_reserve_penalty"):
            assert getattr(bot.cfg, field) == getattr(other.cfg, field)


@pytest.mark.parametrize("spec,kind,sims,ens", [
    ("random", "random", 0, False),
    ("greedy", "greedy", 0, False),
    ("mcts640", "mcts", 640, False),
    ("net:runs/x/weights/latest.pt", "net", 0, False),
    ("net:runs/x/weights/latest.pt:c5", "net", 0, True),
    ("search:runs/x/checkpoints/gen_0040.pt:400", "search", 400, False),
    ("search:C:\\models\\gen_0040.pt:200:c5", "search", 200, True),
])
def test_spec_parsing(spec, kind, sims, ens):
    parsed = A.parse_spec(spec)
    assert (parsed.kind, parsed.sims, parsed.ensemble) == (kind, sims, ens)
    if kind in ("net", "search"):
        assert parsed.path.endswith(".pt")


def test_spec_labels_and_factories_are_picklable():
    import pickle
    parsed = A.parse_spec("search:runs/nscc/weights/latest.pt:400")
    assert parsed.label == "search400:nscc"          # run dir, not 'latest'
    assert A.parse_spec("gen40=mcts160").label == "gen40"
    factory = A.make_factory("mcts40")
    bot = pickle.loads(pickle.dumps(factory))()
    assert bot.name == "mcts40" and bot.cfg.sims == 40


def test_anchor_ladder_helper_builds_every_rung():
    ladder = A.anchor_ladder(("random", "greedy", "mcts40"))
    assert list(ladder) == ["random", "greedy", "mcts40"]
    assert all(callable(f) for f in ladder.values())


def test_net_bots_play_and_the_c5_ensemble_is_equivariant(tmp_path):
    """A checkpoint drives both the policy bot and the search bot, and the
    optional C5 ensemble respects the §1.4 rotation identities."""
    import random
    import torch
    from splendor_ai.encode import encode
    from splendor_ai.model import NetConfig, SplendorNet, save_checkpoint
    from splendor_ai.rules import engine as E
    from splendor_ai.symmetry import action_perm, rotate_state

    torch.manual_seed(0)
    ckpt = str(tmp_path / "latest.pt")
    save_checkpoint(ckpt, SplendorNet(NetConfig(width=32, blocks=1)),
                    {"generation": 1})

    state = E.new_game(2, "INDIVIDUAL", None, rng=random.Random(3))
    for _ in range(6):
        E.apply(state, E.legal_actions(state)[0])

    policy = A.make_bot(f"net:{ckpt}")
    action = policy.act(state, state.current_player, np.random.default_rng(0))
    assert E.legal_mask(state)[action]

    ensemble = A.net_policy(ckpt, ensemble=True)
    mask = np.asarray(E.legal_mask(state), dtype=bool)
    priors, _ = ensemble.evaluator.evaluate(encode(state, 0)[None], mask[None])
    rotated = rotate_state(state, 2)
    rmask = np.asarray(E.legal_mask(rotated), dtype=bool)
    rpriors, _ = ensemble.evaluator.evaluate(encode(rotated, 0)[None],
                                             rmask[None])
    assert np.allclose(rpriors[0], priors[0][action_perm(2)], atol=1e-6)

    search = A.make_bot(f"search:{ckpt}:8")
    action = search.act(state, state.current_player, np.random.default_rng(0))
    assert E.legal_mask(state)[action]


# ── Bradley–Terry ─────────────────────────────────────────────────────────

def _synthetic_counts(true_elo, games_per_pair=4000, seed=0):
    """Simulate a full round robin from known Elo ratings."""
    rng = np.random.default_rng(seed)
    names = list(true_elo)
    counts = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p = 1.0 / (1.0 + 10 ** (-(true_elo[a] - true_elo[b]) / 400.0))
            wins = int(rng.binomial(games_per_pair, p))
            counts[(a, b)] = (wins, 0, games_per_pair - wins)
    return counts


def test_bradley_terry_recovers_known_ratings():
    true = {"random": 0.0, "greedy": 400.0, "mcts160": 700.0, "net": 1000.0}
    counts = _synthetic_counts(true, games_per_pair=4000, seed=7)
    fit = fit_bradley_terry(counts, anchor="random", bootstrap=60, seed=11)

    assert fit.converged
    assert fit.rating["random"] == pytest.approx(0.0)
    for name, target in true.items():
        assert fit.rating[name] == pytest.approx(target, abs=30.0), name
        if name != "random":                     # the anchor has no spread
            assert fit.lo[name] < target < fit.hi[name], name
    # ranking and monotonicity
    assert fit.sorted_names() == ["net", "mcts160", "greedy", "random"]
    assert fit.games["greedy"] == 12000


def test_bradley_terry_is_a_joint_fit_not_a_chain():
    """``net`` never plays ``random``; the joint fit still places it."""
    true = {"random": 0.0, "greedy": 400.0, "net": 900.0}
    counts = _synthetic_counts(true, games_per_pair=5000, seed=3)
    counts.pop(("net", "random"), None)
    counts.pop(("random", "net"), None)
    fit = fit_bradley_terry(counts, anchor="random")
    assert fit.rating["net"] == pytest.approx(900.0, abs=40.0)
    assert not fit.disconnected


def test_bradley_terry_single_pair_matches_the_closed_form():
    fit = fit_bradley_terry({("a", "b"): (75, 10, 15)}, anchor="a", prior=0.0)
    # draws as half: 80 vs 20 -> 400/ln10 * ln(20/80)
    assert (fit.rating["b"] - fit.rating["a"]) == pytest.approx(
        ELO_SCALE * math.log(20.0 / 80.0), abs=1e-6)


def test_bradley_terry_scale_is_ordinary_elo():
    """A 400-point gap must mean a 10:1 expected score."""
    fit = fit_bradley_terry({("a", "b"): (1000, 0, 100)}, anchor="b",
                            prior=0.0)
    delta = fit.rating["a"] - fit.rating["b"]
    expected = 1.0 / (1.0 + 10 ** (-delta / 400.0))
    assert expected == pytest.approx(1000 / 1100, abs=1e-6)


def test_bradley_terry_anchor_pinning_and_prior():
    counts = {("random", "greedy"): (0, 0, 40)}      # greedy won everything
    pinned = fit_bradley_terry(counts, anchor="random", anchor_rating=1500.0)
    assert pinned.rating["random"] == pytest.approx(1500.0)
    # the virtual draw keeps an unbeaten player at a sane, finite rating
    assert 1500.0 < pinned.rating["greedy"] < 1500.0 + 2000.0
    # without it the likelihood is maximised at infinity and the fit runs away
    unpinned = fit_bradley_terry(counts, anchor="random", prior=0.0)
    assert unpinned.rating["greedy"] > 10000.0


def test_bradley_terry_flags_a_disconnected_graph():
    counts = {("a", "b"): (10, 0, 5), ("c", "d"): (7, 0, 8)}
    fit = fit_bradley_terry(counts, anchor="a")
    assert [sorted(g) for g in fit.disconnected] == [["c", "d"]]


# ── seat rotation ─────────────────────────────────────────────────────────

def test_two_player_rotation_is_a_swap():
    seatings = seat_arrangements(MODES["ind2"], ("A", "B"))
    assert seatings == [("A", "B"), ("B", "A")]


@pytest.mark.parametrize("key", ["ind2", "ind3", "ind4"])
def test_individual_rotation_is_seat_balanced(key):
    """Every bot sits in every seat the same number of times."""
    mode = MODES[key]
    seatings = [s for base in pair_compositions(mode, "A", "B")
                for s in seat_arrangements(mode, base)]
    for seat in range(mode.num_players):
        counts = Counter(s[seat] for s in seatings)
        assert counts["A"] == counts["B"] == len(seatings) / 2, (key, seat)
    # and each bot occupies each seat at least once
    for name in ("A", "B"):
        occupied = {seat for s in seatings for seat in range(mode.num_players)
                    if s[seat] == name}
        assert occupied == set(range(mode.num_players))


def test_latin_square_for_a_table_of_distinct_bots():
    mode = MODES["ind4"]
    seatings = seat_arrangements(mode, ("A", "B", "C", "D"))
    assert len(seatings) == 4
    for seat in range(4):
        assert sorted(s[seat] for s in seatings) == ["A", "B", "C", "D"]
    for name in "ABCD":
        assert sorted(s.index(name) for s in seatings) == [0, 1, 2, 3]


def test_one_v_two_rotation_covers_both_roles_with_pure_sides():
    mode = MODES["ovt"]
    seatings = [s for base in pair_compositions(mode, "A", "B")
                for s in seat_arrangements(mode, base)]
    assert set(seatings) == {("A", "B", "B"), ("B", "A", "A")}
    for name in ("A", "B"):
        roles = Counter(mode.roles[seat] for s in seatings
                        for seat in range(3) if s[seat] == name)
        assert roles["solo"] == 1 and roles["duo"] == 2
    for s in seatings:                       # a side is never mixed
        assert s[1] == s[2]


@pytest.mark.parametrize("key", ["team", "team_opp"])
def test_team_rotation_swaps_the_sides_and_keeps_them_pure(key):
    mode = MODES[key]
    sides = mode.sides
    seatings = [s for base in pair_compositions(mode, "A", "B")
                for s in seat_arrangements(mode, base)]
    assert len(seatings) == 2
    for name in ("A", "B"):
        played = {tuple(sorted(seat for seat in range(4) if s[seat] == name))
                  for s in seatings}
        assert len(played) == 2                     # both sides
        assert set().union(*played) == {0, 1, 2, 3}  # every seat
    for s in seatings:
        for side in (0, 1):
            occupants = {s[seat] for seat in range(4) if sides[seat] == side}
            assert len(occupants) == 1              # homogeneous side


def test_within_side_rotation_covers_both_team_seats():
    """A heterogeneous side still gets both of its seats rotated."""
    seatings = seat_arrangements(MODES["team"], ("A", "B", "C", "C"))
    assert ("A", "B", "C", "C") in seatings and ("B", "A", "C", "C") in seatings
    assert ("C", "C", "A", "B") in seatings


# ── schedule ──────────────────────────────────────────────────────────────

def test_schedule_pairs_seeds_within_a_group_and_across_pairings():
    modes = [MODES[k] for k in ALL_MODES]
    jobs, tables = build_schedule(["A", "B", "C"], modes,
                                  games_per_pairing=6, seed=5, mixed=False)
    by_group = {}
    for job in jobs:
        by_group.setdefault((job["mode"], job["table"], job["group"]),
                            []).append(job)
    for key, group in by_group.items():
        seeds = {job["seed"] for job in group}
        assert len(seeds) == 1, key            # paired: one deal per group
        seatings = [tuple(job["seats"]) for job in group]
        assert len(seatings) == len(set(seatings))
    # common random numbers: group g of a mode is the same deal for every table
    per_mode_group = {}
    for job in jobs:
        per_mode_group.setdefault((job["mode"], job["group"]),
                                  set()).add(job["seed"])
    assert all(len(v) == 1 for v in per_mode_group.values())
    # every pairing is scheduled in every mode
    for mode in modes:
        members = {tuple(sorted(t.members)) for t in tables
                   if t.mode == mode.key}
        assert members == {("A", "B"), ("A", "C"), ("B", "C")}


def test_schedule_rounds_up_to_whole_rotations():
    jobs, _ = build_schedule(["A", "B"], [MODES["ind3"]],
                             games_per_pairing=4, seed=0)
    per_table = Counter(job["table"] for job in jobs)
    # 3 seatings per composition, 2 compositions, ceil(4/3) = 2 groups each
    assert set(per_table.values()) == {6}
    assert len(jobs) == 12


def test_mixed_tables_only_appear_for_individual_modes():
    tables = build_tables(MODES["ind4"], ["A", "B", "C", "D"], mixed=True)
    kinds = Counter(t.kind for t in tables)
    assert kinds["mixed"] >= 1
    assert all(len(set(t.base)) == 4 for t in tables if t.kind == "mixed")
    assert not [t for t in build_tables(MODES["team"], ["A", "B", "C", "D"],
                                        mixed=True) if t.kind == "mixed"]


def test_mode_parsing():
    assert parse_mode("ind3") is MODES["ind3"]
    spec = parse_mode("TEAM:4:OPPOSITE")
    assert (spec.mode, spec.num_players, spec.layout) == ("TEAM", 4, "OPPOSITE")
    assert parse_mode("INDIVIDUAL:3").roles == ("seat0", "seat1", "seat2")
    assert MODES["ovt"].roles == ("solo", "duo", "duo")
    with pytest.raises(ValueError):
        parse_mode("nonsense")


# ── outcome bookkeeping ───────────────────────────────────────────────────

def test_game_values_credit_and_pairwise():
    record = {"seats": ["A", "B", "A", "B"], "values": [1.0, -1.0, 1.0, -1.0]}
    assert game_values(record) == {"A": 1.0, "B": -1.0}
    assert table_credit(record) == {"A": 1.0, "B": 0.0}
    assert pairwise_from_game(record) == [(("A", "B"), 1.0)]
    tie = {"seats": ["A", "B"], "values": [0.0, 0.0]}
    assert table_credit(tie) == {"A": 0.5, "B": 0.5}
    assert pairwise_from_game(tie) == [(("A", "B"), 0.5)]
    assert seat_credit([1.0, -1.0, 1.0, -1.0]) == [1.0, 0.0, 1.0, 0.0]
    assert seat_credit([0.0, 0.0]) == [0.5, 0.5]


def test_stale_and_truncation_are_separate_buckets():
    results = ArenaResults(
        bots=["A", "B"], modes=[MODES["ind2"]],
        games=[
            {"i": 0, "mode": "ind2", "table": 0, "kind": "pair", "group": 0,
             "seed": 1, "seats": ["A", "B"], "values": [1.0, -1.0],
             "scores": [15, 3], "plies": 40, "reason": "SCORE",
             "truncated": False, "stuck": 0, "resigned": []},
            {"i": 1, "mode": "ind2", "table": 0, "kind": "pair", "group": 0,
             "seed": 1, "seats": ["B", "A"], "values": [1.0, -1.0],
             "scores": [4, 2], "plies": 400, "reason": "TRUNCATED",
             "truncated": True, "stuck": 0, "resigned": []},
            {"i": 2, "mode": "ind2", "table": 0, "kind": "pair", "group": 1,
             "seed": 2, "seats": ["A", "B"], "values": [1.0, -1.0],
             "scores": [7, 0], "plies": 22, "reason": "FORFEIT",
             "truncated": False, "stuck": 1, "resigned": [1]},
        ])
    summary = results.mode_stats("ind2")
    assert summary["reasons"] == {"SCORE": 1, "TRUNCATED": 1, "FORFEIT": 1}
    assert summary["trunc_rate"] == pytest.approx(1 / 3)
    assert summary["stale_rate"] == pytest.approx(1 / 3)
    stats = results.bot_stats("ind2")
    assert stats["A"]["wins"] == 2.0 and stats["B"]["wins"] == 1.0
    assert stats["A"]["seat_games"] == [2, 1]
    assert results.pair_counts("ind2")[("A", "B")] == [2.0, 0.0, 1.0]


# ── whole runs ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_results():
    """random vs greedy, 4 games per pairing, all five modes."""
    return run_matches({"random": "random", "greedy": "greedy"}, ALL_MODES,
                       games_per_pairing=4, seed=0, workers=1)


def test_tiny_arena_runs_every_mode(tiny_results):
    played = Counter(g["mode"] for g in tiny_results.games)
    assert set(played) == set(ALL_MODES)
    for key, count in played.items():
        assert count >= 4, key
    for game in tiny_results.games:
        spec = MODES[game["mode"]]
        assert len(game["seats"]) == spec.num_players
        assert len(game["values"]) == spec.num_players
        assert game["plies"] > 0


def test_tiny_arena_writes_a_report(tiny_results, tmp_path):
    md_path, json_path = write_reports(tiny_results,
                                       str(tmp_path / "arena.md"),
                                       bootstrap=50)
    assert os.path.exists(md_path) and os.path.exists(json_path)
    text = open(md_path, encoding="utf-8").read()
    assert "# Splendor arena report" in text
    assert "Bradley" in text and "95% CI" in text
    for key in ALL_MODES:
        assert f"## Mode `{key}`" in text
    assert "solo win%" in text                     # the 1v2 role split
    assert "Seat win share" in text
    payload = json.loads(open(json_path, encoding="utf-8").read())
    assert payload["ratings"]["anchor"] == "random"
    elo = payload["ratings"]["ratings"]
    assert elo["greedy"]["elo"] > elo["random"]["elo"] + 100
    assert payload["results"]["config"]["games_per_pairing"] == 4
    assert len(payload["results"]["games"]) == len(tiny_results.games)
    for key in ALL_MODES:                          # per-mode fits are present
        assert payload["modes"][key]["ratings"]["ratings"]["greedy"]["elo"] >= 0


def test_greedy_beats_random_and_the_fit_agrees(tiny_results):
    fit = fit_bradley_terry(tiny_results, anchor="random", bootstrap=50,
                            seed=1)
    assert fit.rating["random"] == 0.0
    assert fit.rating["greedy"] > 200.0
    assert fit.lo["greedy"] > 0.0
    stats = tiny_results.bot_stats()
    assert stats["greedy"]["win_rate"] > 0.8


def test_worker_pool_matches_the_single_process_run(tiny_results):
    parallel = run_matches({"random": "random", "greedy": "greedy"},
                           ALL_MODES, games_per_pairing=4, seed=0, workers=2)
    key = lambda r: [(g["i"], g["seats"], g["values"], g["plies"],
                      g["reason"]) for g in r.games]
    assert key(parallel) == key(tiny_results)


def test_truncation_cap_is_honoured():
    results = run_matches({"random": "random", "greedy": "greedy"}, ["ind2"],
                          games_per_pairing=2, seed=3, workers=1, max_plies=6)
    assert all(g["truncated"] and g["plies"] == 6 for g in results.games)
    assert results.mode_stats("ind2")["trunc_rate"] == 1.0
    assert results.mode_stats("ind2")["reasons"] == {
        "TRUNCATED": len(results.games)}


def test_bots_may_be_factories_or_objects():
    from splendor_ai.bots import GreedyBot, RandomBot
    results = run_matches({"r": RandomBot(), "g": lambda: GreedyBot()},
                          ["ind2"], games_per_pairing=2, seed=1, workers=1)
    assert len(results.games) == 2
    assert results.bot_stats("ind2")["g"]["win_rate"] >= 0.5


def test_cli_writes_a_report(tmp_path, capsys):
    out = tmp_path / "reports" / "cli.md"
    rc = arena.main(["--bots", "random", "greedy", "--modes", "ind2", "ovt",
                     "--games", "2", "--bootstrap", "20", "--seed", "4",
                     "--out", str(out), "--quiet"])
    assert rc == 0
    assert out.exists() and (tmp_path / "reports" / "cli.json").exists()
    report = json.loads((tmp_path / "reports" / "cli.json").read_text())
    assert set(report["modes"]) == {"ind2", "ovt"}
    assert report["config"]["bots"] == {"random": "random", "greedy": "greedy"}


def test_cli_rejects_duplicate_bot_names(tmp_path):
    with pytest.raises(SystemExit):
        arena.main(["--bots", "mcts40", "mcts40", "--modes", "ind2",
                    "--games", "1", "--out", str(tmp_path / "x.md")])


def test_missing_anchor_falls_back_loudly(tiny_results):
    """A report headed "anchor X = 0" that was centred on something else is
    exactly the incomparable number this module exists to prevent."""
    with pytest.warns(RuntimeWarning, match="did not play"):
        report = arena.build_report(tiny_results, anchor="not_here",
                                    bootstrap=0)
    assert report["anchor"] == "random"                # the first bot
    assert report["ratings"]["ratings"]["random"]["elo"] == 0.0


def test_report_renders_without_bootstrap(tiny_results):
    report = arena.build_report(tiny_results, bootstrap=0)
    text = render_markdown(report)
    assert "# Splendor arena report" in text
    assert "| `greedy` |" in text
