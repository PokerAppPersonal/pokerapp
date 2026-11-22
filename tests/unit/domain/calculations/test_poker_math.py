import pytest

from pokerapp.domain.cards.card import Card, Rank, Suit
from pokerapp.domain.calculations.poker_math import (
    combinations_count, exact_equity, full_deck, monte_carlo_equity, outs,
    remaining_cards,
)


def c(rank, suit):
    return Card(Rank(rank), Suit(suit))


def river_board():
    return [c(2, Suit.CLUBS), c(3, Suit.DIAMONDS), c(4, Suit.HEARTS),
            c(5, Suit.SPADES), c(9, Suit.CLUBS)]


def test_deck_has_52_unique_cards():
    deck = full_deck()
    assert len(deck) == 52
    assert len(set(deck)) == 52


def test_remaining_cards_excludes_known_cards():
    known = [c(14, Suit.SPADES), c(13, Suit.HEARTS)]
    remaining = remaining_cards(known)
    assert len(remaining) == 50
    assert not set(known).intersection(remaining)


def test_remaining_cards_rejects_duplicates_and_non_cards():
    card = c(14, Suit.SPADES)
    with pytest.raises(ValueError, match="Duplicate"):
        remaining_cards([card, card])
    with pytest.raises(ValueError, match="Card instances"):
        remaining_cards([card, "not a card"])


def test_combinations_count():
    assert combinations_count(47, 5) == 1_533_939
    assert combinations_count(10, 2) == 45
    assert combinations_count(3, 4) == 0


def test_exact_equity_river_tie():
    players = [[c(14, Suit.CLUBS), c(13, Suit.DIAMONDS)],
               [c(14, Suit.HEARTS), c(13, Suit.SPADES)]]
    assert exact_equity(players, river_board()) == [0.5, 0.5]


def test_exact_equity_river_win():
    players = [[c(14, Suit.CLUBS), c(13, Suit.DIAMONDS)],
               [c(7, Suit.HEARTS), c(7, Suit.DIAMONDS)]]
    assert exact_equity(players, river_board()) == [1.0, 0.0]


def test_exact_equity_three_way_split():
    players = [[c(14, Suit.CLUBS), c(13, Suit.DIAMONDS)],
               [c(14, Suit.HEARTS), c(13, Suit.SPADES)],
               [c(14, Suit.DIAMONDS), c(13, Suit.HEARTS)]]
    board = [c(2, Suit.CLUBS), c(3, Suit.DIAMONDS), c(4, Suit.HEARTS),
             c(5, Suit.SPADES), c(6, Suit.CLUBS)]
    assert exact_equity(players, board) == pytest.approx([1 / 3] * 3)


@pytest.mark.parametrize("players, board", [
    ([[c(14, Suit.CLUBS), c(13, Suit.CLUBS)]], []),
    ([[c(14, Suit.CLUBS)], [c(13, Suit.CLUBS), c(12, Suit.CLUBS)]], []),
    ([[c(14, Suit.CLUBS), c(13, Suit.CLUBS)],
      [c(14, Suit.CLUBS), c(12, Suit.DIAMONDS)]], []),
    ([[c(14, Suit.CLUBS), c(13, Suit.CLUBS)],
      [c(12, Suit.CLUBS), c(11, Suit.CLUBS)]],
     [c(2, Suit.CLUBS), c(3, Suit.CLUBS)]),
])
def test_exact_equity_rejects_invalid_inputs(players, board):
    with pytest.raises(ValueError):
        exact_equity(players, board)


def test_exact_equity_rejects_board_hole_collision():
    board = river_board()
    players = [[c(14, Suit.CLUBS), c(13, Suit.DIAMONDS)],
               [board[0], c(7, Suit.DIAMONDS)]]
    with pytest.raises(ValueError, match="Duplicate"):
        exact_equity(players, board)


def test_multiway_equity_shares_sum_to_one():
    players = [[c(14, Suit.CLUBS), c(13, Suit.DIAMONDS)],
               [c(7, Suit.HEARTS), c(7, Suit.DIAMONDS)],
               [c(12, Suit.CLUBS), c(11, Suit.DIAMONDS)]]
    equity = exact_equity(players, river_board())
    assert len(equity) == 3
    assert sum(equity) == pytest.approx(1.0)


def test_monte_carlo_reproducible():
    board = [c(2, Suit.CLUBS), c(3, Suit.DIAMONDS), c(4, Suit.HEARTS)]
    players = [[c(14, Suit.CLUBS), c(13, Suit.DIAMONDS)],
               [c(7, Suit.HEARTS), c(7, Suit.DIAMONDS)]]
    assert monte_carlo_equity(players, board, trials=50, seed=42) == (
        monte_carlo_equity(players, board, trials=50, seed=42))


def test_monte_carlo_converges_toward_exact_turn_equity():
    board = [c(2, Suit.CLUBS), c(3, Suit.DIAMONDS), c(4, Suit.HEARTS),
             c(5, Suit.SPADES)]
    players = [[c(14, Suit.CLUBS), c(13, Suit.DIAMONDS)],
               [c(7, Suit.HEARTS), c(7, Suit.DIAMONDS)]]
    exact = exact_equity(players, board)
    estimated = monte_carlo_equity(players, board, trials=5000, seed=2026)
    assert estimated == pytest.approx(exact, abs=0.04)
    assert sum(estimated) == pytest.approx(1.0)


@pytest.mark.parametrize("trials", [0, -1, 1.5, True])
def test_monte_carlo_rejects_invalid_trial_counts(trials):
    players = [[c(14, Suit.CLUBS), c(13, Suit.DIAMONDS)],
               [c(7, Suit.HEARTS), c(7, Suit.DIAMONDS)]]
    with pytest.raises(ValueError, match="positive integer"):
        monte_carlo_equity(players, river_board(), trials=trials)


def test_outs_finds_flush_improvers_and_excludes_known_cards():
    hole = [c(14, Suit.HEARTS), c(13, Suit.HEARTS)]
    board = [c(2, Suit.HEARTS), c(7, Suit.HEARTS), c(9, Suit.CLUBS)]
    candidates = outs(hole, board)
    assert candidates
    assert not set(candidates).intersection([*hole, *board])


def test_outs_rejects_duplicate_cards():
    hole = [c(14, Suit.HEARTS), c(13, Suit.HEARTS)]
    board = [c(2, Suit.HEARTS), c(7, Suit.HEARTS), hole[0]]
    with pytest.raises(ValueError, match="Duplicate"):
        outs(hole, board)
