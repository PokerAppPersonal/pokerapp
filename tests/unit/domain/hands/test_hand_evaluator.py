from itertools import combinations

import pytest

from pokerapp.domain.cards.card import Card, Rank, Suit
from pokerapp.domain.hands.categories import HandCategory
from pokerapp.domain.hands.evaluator import HandEvaluator


RANKS = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
          "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
SUITS = {"c": Suit.CLUBS, "d": Suit.DIAMONDS, "h": Suit.HEARTS, "s": Suit.SPADES}


def hand(spec: str) -> list[Card]:
    return [Card(Rank(RANKS[token[0]]), SUITS[token[1]]) for token in spec.split()]


def oracle_five(cards: tuple[Card, ...]) -> tuple[int, tuple[int, ...]]:
    """Independent 5-card rank oracle, using count/rank sorting."""
    ranks = [int(c.rank) for c in cards]
    counts: dict[int, int] = {}
    for rank in ranks:
        counts[rank] = counts.get(rank, 0) + 1
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    shape = sorted(counts.values(), reverse=True)
    flush = len({c.suit for c in cards}) == 1
    unique = set(ranks)
    straight = None
    if len(unique) == 5:
        if max(unique) - min(unique) == 4:
            straight = max(unique)
        elif unique == {14, 2, 3, 4, 5}:
            straight = 5

    if flush and straight:
        return 8, (straight,)
    if shape == [4, 1]:
        quad = groups[0][1]
        kicker = next(rank for rank in ranks if rank != quad)
        return 7, (quad, kicker)
    if shape == [3, 2]:
        return 6, (groups[0][1], groups[1][1])
    if flush:
        return 5, tuple(sorted(ranks, reverse=True))
    if straight:
        return 4, (straight,)
    if shape == [3, 1, 1]:
        trip = groups[0][1]
        kickers = sorted((r for r in ranks if r != trip), reverse=True)
        return 3, (trip, *kickers)
    if shape == [2, 2, 1]:
        pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
        kicker = next(rank for rank, count in counts.items() if count == 1)
        return 2, (*pairs, kicker)
    if shape == [2, 1, 1, 1]:
        pair = groups[0][1]
        kickers = sorted((r for r in ranks if r != pair), reverse=True)
        return 1, (pair, *kickers)
    return 0, tuple(sorted(ranks, reverse=True))


def oracle_best(cards: list[Card]) -> tuple[int, tuple[int, ...]]:
    return max(oracle_five(combo) for combo in combinations(cards, 5))


@pytest.fixture
def evaluator() -> HandEvaluator:
    return HandEvaluator()


@pytest.mark.parametrize(
    "spec,category,tiebreakers",
    [
        ("Ah Kh Qh Jh Th", HandCategory.STRAIGHT_FLUSH, (14,)),
        ("9c 9d 9h 9s Ac", HandCategory.FOUR_OF_A_KIND, (9, 14)),
        ("Ac Ad Ah Ks Kd", HandCategory.FULL_HOUSE, (14, 13)),
        ("Ah Jh 9h 5h 2h", HandCategory.FLUSH, (14, 11, 9, 5, 2)),
        ("9c 8d 7h 6s 5c", HandCategory.STRAIGHT, (9,)),
        ("9c 9d 9h As Kc", HandCategory.SET, (9, 14, 13)),
        ("9c 9d 5h 5s Ac", HandCategory.TWO_PAIRS, (9, 5, 14)),
        ("9c 9d Ah Ks Qc", HandCategory.PAIR, (9, 14, 13, 12)),
        ("Ac Kd Qh Js 9c", HandCategory.HIGH_CARD, (14, 13, 12, 11, 9)),
    ],
)
def test_evaluate_each_category(evaluator, spec, category, tiebreakers):
    rank, best_cards = evaluator.evaluate(hand(spec))
    assert rank.category == category
    assert rank.tiebreakers == tiebreakers
    assert len(best_cards) == 5
    assert len(set(best_cards)) == 5


def test_evaluate_preflop_returns_two_cards(evaluator):
    cards = hand("Ah Kd")
    rank, selected = evaluator.evaluate(cards)
    assert rank.category == HandCategory.HIGH_CARD
    assert selected == cards


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("Ah Kh Qh Jh Th", "9c 9d 9h 9s Ac", 1),
        ("9c 9d 9h 9s Ac", "Ac Ad Ah Ks Kd", 1),
        ("Ac Ad Ah Ks Kd", "Ah Jh 9h 5h 2h", 1),
        ("9c 9d 5h 5s Ac", "9c 9d 5h 5s Kc", 1),
        ("9c 9d Ah Ks Qc", "9c 9d Ah Ks Jc", 1),
        ("Ac Kd Qh Js 9c", "Ac Kd Qh Js 8c", 1),
        ("Ah Kh Qh Jh Th", "As Ks Qs Js Ts", 0),
    ],
)
def test_compare_win_loss_and_tie(evaluator, a, b, expected):
    assert evaluator.compare(hand(a), hand(b)) == expected
    assert evaluator.compare(hand(b), hand(a)) == -expected


def test_highest_hand_multiple_winners(evaluator):
    players = [
        hand("Ah Ad 2c 3d 4h 5s 6c"),
        hand("Kc Kd 2c 3d 4h 5s 6c"),
        hand("Qc Qd 2c 3d 4h 5s 6c"),
    ]
    assert evaluator.highest_hand(players) == [0, 1, 2]


def test_highest_hand_board_playing_tie(evaluator):
    # The board's 5-high straight plays for every player; hole cards do not matter.
    board = hand("Ac 2d 3h 4s 5c")
    players = [hand("Kh Kd") + board, hand("Qh Qd") + board]
    assert evaluator.highest_hand(players) == [0, 1]


def test_highest_hand_board_playing_winner_and_loser(evaluator):
    board = hand("2c 3d 4h 5s 9c")
    players = [hand("Ac Kd") + board, hand("7h 7d") + board]
    assert evaluator.highest_hand(players) == [0]


def test_best_five_selection_matches_independent_oracle(evaluator):
    samples = [
        hand("Ah Kh Qh Jh Th 2c 3d"),  # straight flush
        hand("9c 9d 9h 9s Ac Kd 2h"),  # quads and kicker
        hand("9c 9d 9h 5s 5c 5d Ac"),  # two trips, best full house
        hand("Ad Kd 2h 3h 4h 5h 9h"),  # best flush
        hand("Ac Ad 2h 3s 4c 5d Kh"),  # wheel straight with pair
        hand("Kc Kd 9h 9s 5c 5d Ad"),  # three pairs, top two + kicker
        hand("9c 9d Ah Ks Qc Jd 2h"),  # pair kickers
        hand("2c 4d 6h 9s Jc Kd 7h"),  # high-card selection
    ]
    for cards in samples:
        rank, selected = evaluator.evaluate(cards)
        assert (int(rank.category), rank.tiebreakers) == oracle_best(cards)
        assert len(selected) == 5
        assert all(card in cards for card in selected)


@pytest.mark.parametrize("spec", ["Ah Ah", "2c 2c 3d 4h 5s", "Ac Kd Qh Js 9c 9c"])
def test_duplicate_cards_rejected(evaluator, spec):
    with pytest.raises(ValueError, match="Duplicate cards"):
        evaluator.evaluate(hand(spec))


@pytest.mark.parametrize("count", [0, 1, 3, 4, 8])
def test_unsupported_card_count_rejected(evaluator, count):
    cards = hand("2c 3d 4h 5s 6c 7d 8h")[:count]
    with pytest.raises(ValueError, match="Expected"):
        evaluator.evaluate(cards)


def test_highest_hand_requires_players(evaluator):
    with pytest.raises(ValueError, match="At least one hand"):
        evaluator.highest_hand([])


def test_highest_hand_requires_equal_card_counts(evaluator):
    with pytest.raises(ValueError, match="same number"):
        evaluator.highest_hand([hand("Ah Kd"), hand("Ah Kd Qh")])
