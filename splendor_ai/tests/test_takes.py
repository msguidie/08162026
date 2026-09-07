"""Gem-take legality — the fiddliest rule in the variant.

The server validates a colour list sequentially with ``canSelectGem`` and then
requires ``isGemTakeComplete``; the engine uses closed forms.  These tests
prove the two agree everywhere, prove the validation is order-independent (so
the canonical sorted ordering behind each take action index is sound), and
pin the individual rules against the real Node engine.
"""

import itertools
import random

import pytest

from splendor_ai.rules import engine as E
from splendor_ai.rules.actions import (
    NUM_TAKE_ACTIONS, TAKE1_START, TAKE2SAME_START, TAKE2_START, TAKE3_START,
    TAKE_PATTERNS, take_index,
)
from splendor_ai.tests.oracle import probe, requires_node

CFG = E.make_config(4)


# ── closed form vs. the literal port ──────────────────────────────────────

def _reference_mask(supply, p_gems):
    return [E.gem_take_accepted(pat, supply, p_gems, CFG)
            for pat in TAKE_PATTERNS]


def test_closed_form_matches_reference_exhaustive():
    """Every supply in {0..5}^5 x every hand size 0..10."""
    bad = []
    for supply in itertools.product(range(6), repeat=5):
        s = list(supply)
        for p in range(11):
            fast = E.take_legal_mask(s, p)
            ref = _reference_mask(s, p)
            if fast != ref:
                bad.append((s, p, [i for i in range(NUM_TAKE_ACTIONS)
                                   if fast[i] != ref[i]]))
                if len(bad) > 5:
                    break
    assert not bad, f"closed form disagrees with canSelectGem/isGemTakeComplete: {bad}"


def test_closed_form_matches_reference_high_supply():
    """Supplies above the 7-token cap can appear in doctored positions."""
    rng = random.Random(7)
    for _ in range(20000):
        s = [rng.randrange(0, 12) for _ in range(5)]
        p = rng.randrange(0, 11)
        assert E.take_legal_mask(s, p) == _reference_mask(s, p), (s, p)


def test_take_order_independence():
    """Acceptance never depends on the order the colours are validated in, so
    the sorted representative behind each action index is faithful."""
    rng = random.Random(11)
    for _ in range(4000):
        supply = [rng.randrange(0, 8) for _ in range(5)]
        p = rng.randrange(0, 11)
        for pat in TAKE_PATTERNS:
            base = E.gem_take_accepted(pat, supply, p, CFG)
            for perm in set(itertools.permutations(pat)):
                assert E.gem_take_accepted(list(perm), supply, p, CFG) is base, (
                    supply, p, pat, perm)


def test_non_representable_multisets_are_never_accepted():
    """The 25 colour multisets outside the 30-action space (e.g. [a,a,b] or
    three of a kind) must be rejected in every position."""
    extra = []
    for n in (2, 3):
        for combo in itertools.combinations_with_replacement(range(5), n):
            if tuple(sorted(combo)) not in {tuple(p) for p in TAKE_PATTERNS}:
                extra.append(list(combo))
    assert len(extra) == 25
    rng = random.Random(13)
    for _ in range(2000):
        supply = [rng.randrange(0, 8) for _ in range(5)]
        p = rng.randrange(0, 11)
        for combo in extra:
            assert not E.gem_take_accepted(combo, supply, p, CFG)


# ── individual rules ──────────────────────────────────────────────────────

def _take_actions(supply, p_gems):
    mask = E.take_legal_mask(supply, p_gems)
    return {i for i in range(NUM_TAKE_ACTIONS) if mask[i]}


def test_ten_token_cap_blocks_every_take():
    assert _take_actions([7, 7, 7, 7, 7], 10) == set()


def test_nine_tokens_forces_a_single():
    acts = _take_actions([7, 7, 7, 7, 7], 9)
    assert acts == {TAKE1_START + c for c in range(5)}


def test_eight_tokens_forces_a_pair():
    acts = _take_actions([7, 7, 7, 7, 7], 8)
    pairs = set(range(TAKE2_START, TAKE2_START + 10))
    doubles = set(range(TAKE2SAME_START, TAKE2SAME_START + 5))
    assert acts == pairs | doubles          # no triples, no singles


def test_seven_tokens_allows_triples_but_not_pairs_or_singles():
    acts = _take_actions([7, 7, 7, 7, 7], 7)
    triples = set(range(TAKE3_START, TAKE3_START + 10))
    doubles = set(range(TAKE2SAME_START, TAKE2SAME_START + 5))
    assert acts == triples | doubles


def test_take_two_same_needs_a_stack_of_four():
    for stack in range(0, 8):
        supply = [stack, 0, 0, 0, 0]
        legal = TAKE2SAME_START in _take_actions(supply, 0)
        assert legal == (stack >= 4), stack


def test_two_different_only_when_no_third_colour_or_cap():
    # three colours available and room to hold three: the 2-take is incomplete
    assert take_index([0, 1]) not in _take_actions([1, 1, 1, 0, 0], 0)
    # only two colours left: the 2-take is forced short and therefore legal
    assert take_index([0, 1]) in _take_actions([1, 1, 0, 0, 0], 0)
    # cap forces it even with a full supply
    assert take_index([0, 1]) in _take_actions([7, 7, 7, 7, 7], 8)


def test_single_only_when_forced():
    # one colour left, fewer than 4 of it -> cannot take 2 same, no other colour
    assert take_index([0]) in _take_actions([3, 0, 0, 0, 0], 0)
    # one colour left with >= 4 -> must take two of it, the single is illegal
    assert take_index([0]) not in _take_actions([4, 0, 0, 0, 0], 0)
    # ... unless the cap allows only one more token
    assert take_index([0]) in _take_actions([4, 0, 0, 0, 0], 9)


def test_empty_supply_has_no_takes():
    assert _take_actions([0, 0, 0, 0, 0], 0) == set()


# ── cross-checked against the Node engine ─────────────────────────────────

@requires_node
@pytest.mark.parametrize("supply,hand", [
    ([7, 7, 7, 7, 7], [0, 0, 0, 0, 0, 0]),
    ([7, 7, 7, 7, 7], [2, 2, 2, 1, 0, 0]),      # 7 tokens
    ([7, 7, 7, 7, 7], [2, 2, 2, 2, 0, 0]),      # 8 tokens
    ([7, 7, 7, 7, 7], [2, 2, 2, 2, 1, 0]),      # 9 tokens
    ([7, 7, 7, 7, 7], [2, 2, 2, 2, 2, 0]),      # 10 tokens
    ([2, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0]),
    ([3, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
    ([4, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
    ([1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
    ([1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]),
    ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
    ([4, 1, 0, 0, 0], [0, 0, 0, 0, 0, 3]),      # gold counts toward the cap
    ([4, 1, 0, 0, 0], [0, 0, 0, 0, 0, 9]),
])
def test_node_agrees_on_take_legality(supply, hand):
    probe(mode="INDIVIDUAL", n=2,
          state={"gems": supply + [5], "players": [{"gems": hand}, {}]},
          ops=[["probe"]]).run_all()


@requires_node
def test_node_agrees_on_random_take_positions():
    rng = random.Random(99)
    for _ in range(25):
        supply = [rng.randrange(0, 8) for _ in range(5)]
        hand = [rng.randrange(0, 3) for _ in range(5)] + [rng.randrange(0, 3)]
        if sum(hand) > 10:
            hand = [0, 0, 0, 0, 0, 0]
        probe(mode="INDIVIDUAL", n=2,
              state={"gems": supply + [5], "players": [{"gems": hand}, {}]},
              ops=[["probe"]]).run_all()


@requires_node
def test_incremental_select_gem_matches_reference():
    """The desktop SELECT_GEM path accepts colours one at a time; the engine's
    ``can_select_gem`` / ``is_gem_take_complete`` port must agree step by step."""
    supply = [4, 1, 0, 0, 0]
    spec = {
        "mode": "INDIVIDUAL", "n": 2,
        "state": {"gems": supply + [5],
                  "players": [{"gems": [0, 0, 0, 0, 0, 0]}, {}]},
        "ops": [["select", 0], ["select", 0]],
    }
    from splendor_ai.tests.oracle import node_probe
    data = node_probe(spec)
    selected = []
    cfg = E.make_config(2)
    for i, op in enumerate(spec["ops"]):
        color = op[1]
        adjusted = list(supply)
        for s in selected:
            adjusted[s] -= 1
        assert E.can_select_gem(color, selected, adjusted, 0, cfg)
        assert data["results"][i]["error"] is None
        selected.append(color)
        final = list(supply)
        for s in selected:
            final[s] -= 1
        complete = E.is_gem_take_complete(selected, final, 0, cfg)
        assert complete == (data["results"][i]["completed"] is True), i
