"""UI-independent Texas Hold'em calculation utilities."""
from itertools import combinations
import random
from typing import Sequence

from pokerapp.domain.cards.card import Card, Rank, Suit
from pokerapp.domain.hands.evaluator import HandEvaluator

RANKS = tuple(Rank)
SUITS = tuple(Suit)
STANDARD_DECK_SIZE = 52
BOARD_SIZE = 5


def full_deck() -> list[Card]:
    return [Card(rank, suit) for rank in RANKS for suit in SUITS]


def validate_unique(cards: Sequence[Card]) -> None:
    if len(set(cards)) != len(cards):
        raise ValueError("Duplicate cards are not allowed")


def remaining_cards(known_cards: Sequence[Card]) -> list[Card]:
    validate_unique(known_cards)
    known = set(known_cards)
    return [card for card in full_deck() if card not in known]


def combinations_count(n: int, k: int) -> int:
    if n < 0 or k < 0 or k > n:
        return 0
    from math import comb
    return comb(n, k)


def outs(hole_cards: Sequence[Card], board: Sequence[Card], target_board_size: int = 5) -> list[Card]:
    """Cards that strictly improve the evaluated hand on the next street."""
    if len(board) >= target_board_size:
        return []
    if len(board) not in (3, 4):
        raise ValueError("Outs calculation requires a flop or turn board")
    validate_unique([*hole_cards, *board])
    evaluator = HandEvaluator()
    current = evaluator.evaluate([*hole_cards, *board])[0].comparison_key
    return [card for card in remaining_cards([*hole_cards, *board])
            if evaluator.evaluate([*hole_cards, *board, card])[0].comparison_key > current]


def _validate_equity_inputs(players: Sequence[Sequence[Card]], board: Sequence[Card]) -> list[Card]:
    if len(players) < 2:
        raise ValueError("Equity requires at least two players")
    if any(len(hole) != 2 for hole in players):
        raise ValueError("Each player must have exactly two hole cards")
    known = [*board, *(card for hole in players for card in hole)]
    validate_unique(known)
    if len(board) > BOARD_SIZE:
        raise ValueError("Board cannot contain more than five cards")
    return remaining_cards(known)


def _winner_indices(players: Sequence[Sequence[Card]], board: Sequence[Card], evaluator: HandEvaluator) -> list[int]:
    complete = [list(hole) + list(board) for hole in players]
    return evaluator.highest_hand(complete)


def exact_equity(players: Sequence[Sequence[Card]], board: Sequence[Card]) -> list[float]:
    """Enumerate every possible board completion; return share of pot per player."""
    deck = _validate_equity_inputs(players, board)
    needed = BOARD_SIZE - len(board)
    if needed > len(deck):
        raise ValueError("Not enough cards to complete board")
    evaluator = HandEvaluator()
    totals = [0.0] * len(players)
    scenarios = 0
    for runout in combinations(deck, needed):
        winners = _winner_indices(players, [*board, *runout], evaluator)
        share = 1.0 / len(winners)
        for winner in winners:
            totals[winner] += share
        scenarios += 1
    return [value / scenarios for value in totals]


def monte_carlo_equity(players: Sequence[Sequence[Card]], board: Sequence[Card],
                        trials: int = 10000, seed: int | None = None) -> list[float]:
    """Estimate pot equity by randomly sampling board runouts."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    deck = _validate_equity_inputs(players, board)
    needed = BOARD_SIZE - len(board)
    if needed > len(deck):
        raise ValueError("Not enough cards to complete board")
    rng = random.Random(seed)
    evaluator = HandEvaluator()
    totals = [0.0] * len(players)
    for _ in range(trials):
        runout = rng.sample(deck, needed)
        winners = _winner_indices(players, [*board, *runout], evaluator)
        share = 1.0 / len(winners)
        for winner in winners:
            totals[winner] += share
    return [value / trials for value in totals]
