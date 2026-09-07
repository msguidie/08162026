"""The card / tile tables must be byte-identical to server/gameLogic.js."""

from splendor_ai.rules import cards as C
from splendor_ai.tests.oracle import requires_node


def test_sizes_and_ids():
    assert C.NUM_CARDS == 90
    assert C.NUM_TILES == 10
    assert [c.id for c in C.CARDS] == list(range(90))
    assert [t.id for t in C.TILES] == list(range(10))


def test_tier_layout():
    tiers = [c.tier for c in C.CARDS]
    assert tiers[:40] == [1] * 40
    assert tiers[40:70] == [2] * 30
    assert tiers[70:] == [3] * 20
    assert C.CARDS_BY_TIER[0] == tuple(range(40))
    assert C.CARDS_BY_TIER[1] == tuple(range(40, 70))
    assert C.CARDS_BY_TIER[2] == tuple(range(70, 90))


def test_cycle_structure():
    """Each addCycle emits five cards, one per reward colour, with the cost
    template rotated by the reward index."""
    for base in range(0, 90, 5):
        block = C.CARDS[base:base + 5]
        assert [c.reward for c in block] == [0, 1, 2, 3, 4]
        assert len({c.points for c in block}) == 1
        assert len({c.tier for c in block}) == 1
        template = block[0].cost
        for i, card in enumerate(block):
            assert card.cost == tuple(template[(j - i + 5) % 5] for j in range(5))


def test_known_cards():
    # addCycle(1, 0, [1, 1, 0, 1, 1]) -> ids 0..4
    assert C.CARDS[0].cost == (1, 1, 0, 1, 1)
    assert C.CARDS[0].points == 0 and C.CARDS[0].reward == 0
    # addCycle(1, 1, [0, 4, 0, 0, 0]) -> ids 35..39, the only 1-point tier 1
    assert [c.id for c in C.CARDS if c.tier == 1 and c.points == 1] == list(range(35, 40))
    assert C.CARDS[35].cost == (0, 4, 0, 0, 0)
    # tier 3 top card: addCycle(3, 5, [0, 3, 0, 0, 7]) -> ids 85..89
    assert C.CARDS[85].points == 5 and C.CARDS[85].cost == (0, 3, 0, 0, 7)
    assert max(c.points for c in C.CARDS) == 5


def test_tiles():
    assert all(t.points == 3 for t in C.TILES)
    # ids 0..4: two adjacent colours at 4
    for i in range(5):
        req = list(C.TILES[i].requirement)
        want = [0] * 5
        want[i] = 4
        want[(i + 1) % 5] = 4
        assert req == want
    # ids 5..9: three colours at 3, stepping by 2
    for i in range(5):
        req = list(C.TILES[5 + i].requirement)
        want = [0] * 5
        want[i] = 3
        want[(i + 2) % 5] = 3
        want[(i + 4) % 5] = 3
        assert req == want


@requires_node
def test_matches_node_exactly():
    diffs = C.self_test()
    assert diffs == [], "\n".join(diffs[:20])
