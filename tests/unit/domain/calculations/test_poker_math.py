from pokerapp.domain.cards.card import Card, Rank, Suit
from pokerapp.domain.calculations.poker_math import (
    full_deck, remaining_cards, combinations_count, exact_equity, monte_carlo_equity,
)


def c(rank, suit):
    return Card(Rank(rank), Suit(suit))


def test_deck_has_52_unique_cards():
    deck = full_deck()
    assert len(deck) == 52
    assert len(set(deck)) == 52


def test_remaining_cards_excludes_known_cards():
    known = [c(14, Suit.SPADES), c(13, Suit.HEARTS)]
    remaining = remaining_cards(known)
    assert len(remaining) == 50
    assert not set(known).intersection(remaining)


def test_combination_count():
    assert combinations_count(47, 5) == 1_533_939
    assert combinations_count(10, 2) == 45
    assert combinations_count(3, 4) == 0


def test_exact_equity_river_tie():
    board = [c(2, Suit.CLUBS), c(3, Suit.DIAMONDS), c(4, Suit.HEARTS),
              c(5, Suit.SPADES), c(9, Suit.CLUBS)]
    players = [[c(14, Suit.CLUBS), c(13, Suit.DIAMONDS)],
               [c(14, Suit.HEARTS), c(13, Suit.SPADES)]]
    assert exact_equity(players, board) == [0.5, 0.5]


def test_exact_equity_river_win():
    board = [c(2, Suit.CLUBS), c(3, Suit.DIAMONDS), c(4, Suit.HEARTS),
              c(5, Suit.SPADES), c(9, Suit.CLUBS)]
    players = [[c(14, Suit.CLUBS), c(13, Suit.DIAMONDS)],
               [c(7, Suit.HEARTS), c(7, Suit.DIAMONDS)]]
    assert exact_equity(players, board) == [1.0, 0.0]


def test_monte_carlo_reproducible():
    board = [c(2, Suit.CLUBS), c(3, Suit.DIAMONDS), c(4, Suit.HEARTS)]
    players = [[c(14, Suit.CLUBS), c(13, Suit.DIAMONDS)],
               [c(7, Suit.HEARTS), c(7, Suit.DIAMONDS)]]
    assert monte_carlo_equity(players, board, trials=50, seed=42) == monte_carlo_equity(players, board, trials=50, seed=42)
