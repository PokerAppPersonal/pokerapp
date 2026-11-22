"""UI-independent Texas Hold'em calculation utilities."""
from itertools import combinations
import random
from math import comb
from typing import Sequence

from pokerapp.domain.cards.card import Card, Rank, Suit
from pokerapp.domain.hands.evaluator import HandEvaluator

RANKS = tuple(Rank)
SUITS = tuple(Suit)
STANDARD_DECK_SIZE = 52
BOARD_SIZE = 5
VALID_BOARD_SIZES = (0, 3, 4, 5)


def full_deck() -> list[Card]:
    """Return a fresh, complete 52-card deck."""
    return [Card(rank, suit) for rank in RANKS for suit in SUITS]


def validate_unique(cards: Sequence[Card]) -> None:
    """Validate that all supplied values are cards and no card is repeated."""
    if any(not isinstance(card, Card) for card in cards):
        raise ValueError("All values must be Card instances")
    if len(set(cards)) != len(cards):
        raise ValueError("Duplicate cards are not allowed")


def remaining_cards(known_cards: Sequence[Card]) -> list[Card]:
    """Return all cards not present in known_cards."""
    validate_unique(known_cards)
    known = set(known_cards)
    return [card for card in full_deck() if card not in known]


def combinations_count(n: int, k: int) -> int:
    """Return n choose k; invalid ranges have zero combinations."""
    if n < 0 or k < 0 or k > n:
        return 0
    return comb(n, k)


def outs(
    hole_cards: Sequence[Card],
    board: Sequence[Card],
    target_board_size: int = BOARD_SIZE,
) -> list[Card]:
    """Return next-card candidates that improve hand rank.

    These are rank-improving cards, not necessarily cards that win the pot.
    """
    if target_board_size not in (4, BOARD_SIZE):
        raise ValueError("target_board_size must be 4 or 5")
    if len(board) >= target_board_size:
        return []
    if len(board) not in (3, 4):
        raise ValueError("Outs calculation requires a flop or turn board")
    if len(hole_cards) != 2:
        raise ValueError("Each player must have exactly two hole cards")

    known = [*hole_cards, *board]
    validate_unique(known)
    evaluator = HandEvaluator()
    current = evaluator.evaluate(known)[0].comparison_key
    return [
        card
        for card in remaining_cards(known)
        if evaluator.evaluate([*known, card])[0].comparison_key > current
    ]


def _validate_equity_inputs(
    players: Sequence[Sequence[Card]],
    board: Sequence[Card],
) -> list[Card]:
    if len(players) < 2:
        raise ValueError("Equity requires at least two players")
    if len(board) not in VALID_BOARD_SIZES:
        raise ValueError("Board must contain 0, 3, 4, or 5 cards")
    if any(len(hole) != 2 for hole in players):
        raise ValueError("Each player must have exactly two hole cards")

    known = [*board, *(card for hole in players for card in hole)]
    validate_unique(known)
    return remaining_cards(known)


def _winner_indices(
    players: Sequence[Sequence[Card]],
    board: Sequence[Card],
    evaluator: HandEvaluator,
) -> list[int]:
    complete = [list(hole) + list(board) for hole in players]
    return evaluator.highest_hand(complete)


def exact_equity(
    players: Sequence[Sequence[Card]],
    board: Sequence[Card],
) -> list[float]:
    """Enumerate every board completion and return each player's pot share."""
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

    # combinations(deck, 0) always yields one scenario, but retain this guard
    # so future changes cannot silently divide by zero.
    if scenarios == 0:
        raise ValueError("No possible board completions")
    return [value / scenarios for value in totals]


def monte_carlo_equity(
    players: Sequence[Sequence[Card]],
    board: Sequence[Card],
    trials: int = 10000,
    seed: int | None = None,
) -> list[float]:
    """Estimate each player's pot share using reproducible random runouts."""
    if not isinstance(trials, int) or isinstance(trials, bool) or trials <= 0:
        raise ValueError("trials must be a positive integer")

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
