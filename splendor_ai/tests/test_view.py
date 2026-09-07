"""Per-seat information sets."""

import random

from splendor_ai.rules import engine as E
from splendor_ai.rules import view as V
from splendor_ai.rules.actions import RESERVE_BOARD_START, RESERVE_DECK_START


def _game():
    return E.new_game(3, rng=random.Random(4))


def test_board_reserved_cards_are_public_deck_reserved_are_not():
    s = _game()
    seat = s.current_player
    board_card = s.board[0][0]
    E.apply(s, RESERVE_BOARD_START)                 # face-up reserve
    while s.current_player != seat:
        E.apply(s, E.legal_actions(s)[0])
    deck_top = s.decks[1][-1]
    E.apply(s, RESERVE_DECK_START + 1)              # blind deck reserve

    me = s.players[seat]
    assert me.reserved == [board_card, deck_top]
    assert me.reserved_public == [True, False]

    own = V.public_view(s, seat)["players"][seat]["reserved"]
    assert [c["id"] for c in own] == [board_card, deck_top]

    other = (seat + 1) % s.num_players
    theirs = V.public_view(s, other)["players"][seat]["reserved"]
    assert theirs[0]["id"] == board_card and theirs[0]["hidden"] is False
    assert theirs[1]["id"] == V.HIDDEN_CARD and theirs[1]["hidden"] is True
    # the tier of a deck reserve IS public (RESERVE_FROM_DECK announces it)
    assert theirs[1]["tier"] == 2
    assert theirs[1]["cost"] == [-1] * 5


def test_decks_are_never_exposed():
    s = _game()
    view = V.public_view(s, 0)
    assert "decks" not in view
    assert view["deckCounts"] == s.deck_counts


def test_pending_choice_is_only_shown_to_the_acting_seat():
    s = _game()
    s.pending_tile_choice = [0, 1]
    s.turn_action = "BUY"
    assert V.public_view(s, s.current_player)["pendingTileChoice"] == [0, 1]
    other = (s.current_player + 1) % s.num_players
    assert V.public_view(s, other)["pendingTileChoice"] is None


def test_view_reports_teams_and_affordability():
    s = E.new_game(4, "TEAM", "OPPOSITE", rng=random.Random(6))
    v = V.public_view(s, 0)
    assert [p["teamId"] for p in v["players"]] == [0, 1, 0, 1]
    assert [p["isTeammate"] for p in v["players"]] == [False, False, True, False]
    assert len(v["affordable"]["board"]) == 3
    assert v["qualifiedTiles"] == []


def test_tableaus_and_scores_are_public():
    s = _game()
    for _ in range(40):
        acts = E.legal_actions(s)
        if not acts or s.is_over():
            break
        E.apply(s, acts[0])
    v = V.public_view(s, 1)
    for i, p in enumerate(s.players):
        assert v["players"][i]["cards"] == p.cards
        assert v["players"][i]["score"] == p.score
        assert v["players"][i]["discount"] == p.discount
