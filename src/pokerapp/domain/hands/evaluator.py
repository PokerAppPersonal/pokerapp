from collections import defaultdict
from typing import Sequence

from pokerapp.domain.cards.card import Card, Suit, Rank
from pokerapp.domain.hands.categories import HandCategory
from pokerapp.domain.hands.hand_rank import HandRank


# Assumes Card and Sequence are already in scope (same module as _contains_straight etc.).
#
# Contract shared by all six functions below, same as the straight/flush ones:
#   returns (cards, tiebreaker), or ([], ()) when the pattern is not present.
#   - cards: the best 5-card hand of that category, made-up group(s) first, then
#     kickers, each part high to low. Fewer than 5 cards only if fewer were passed
#     in (pre-flop: 2 cards).
#   - tiebreaker: rank values, group ranks first, then kickers. Tuples of the same
#     category compare correctly with < and >.
#
#   four of a kind -> (quad, kicker)
#   full house     -> (trips, pair)
#   set            -> (trips, kicker1, kicker2)
#   two pairs      -> (high_pair, low_pair, kicker)
#   pair           -> (pair, kicker1, kicker2, kicker3)
#   high card      -> (r1, r2, r3, r4, r5)
#
# Each function only detects its own pattern; deciding which category wins
# (straight flush > quads > full house > flush > straight > set > ...) is the
# caller's job. Kickers are simply the highest remaining cards.

_HAND_SIZE = 5


def _group_by_rank(cards: Sequence[Card]) -> dict[int, list[Card]]:
    """Map rank value -> cards of that rank, in input order."""
    groups: dict[int, list[Card]] = {}
    for card in cards:
        groups.setdefault(int(card.rank), []).append(card)
    return groups


def _ranks_with_at_least(groups: dict[int, list[Card]], size: int) -> list[int]:
    """Ranks that appear at least `size` times, highest first."""
    return sorted((r for r, g in groups.items() if len(g) >= size), reverse=True)


def _build_hand(
        cards: Sequence[Card],
        groups: dict[int, list[Card]],
        made: list[tuple[int, int]],
) -> tuple[Sequence[Card], tuple[int, ...]]:
    """Take `count` cards for each (rank, count) in `made`, then fill the hand
    up to five cards with the highest remaining cards (the kickers)."""
    chosen: list[Card] = []
    for rank, count in made:
        chosen.extend(groups[rank][:count])

    used = {rank for rank, _ in made}
    rest = sorted(
        (c for c in cards if int(c.rank) not in used),
        key=lambda c: int(c.rank),
        reverse=True,
    )
    kickers = rest[: _HAND_SIZE - len(chosen)]

    tiebreaker = tuple(rank for rank, _ in made) + tuple(int(c.rank) for c in kickers)
    return chosen + kickers, tiebreaker


def _contains_four_of_a_kind(cards: Sequence[Card]) -> tuple[Sequence[Card], tuple[int, ...]]:
    groups = _group_by_rank(cards)
    quads = _ranks_with_at_least(groups, 4)
    if not quads:
        return [], ()
    return _build_hand(cards, groups, [(quads[0], 4)])


def _contains_full_house(cards: Sequence[Card]) -> tuple[Sequence[Card], tuple[int, ...]]:
    groups = _group_by_rank(cards)
    trips = _ranks_with_at_least(groups, 3)
    if not trips:
        return [], ()
    best_trips = trips[0]
    # The pair may come from a second set of trips (its lowest two cards play).
    pairs = [r for r in _ranks_with_at_least(groups, 2) if r != best_trips]
    if not pairs:
        return [], ()
    return _build_hand(cards, groups, [(best_trips, 3), (pairs[0], 2)])


def _contains_set(cards: Sequence[Card]) -> tuple[Sequence[Card], tuple[int, ...]]:
    groups = _group_by_rank(cards)
    trips = _ranks_with_at_least(groups, 3)
    if not trips:
        return [], ()
    return _build_hand(cards, groups, [(trips[0], 3)])


def _contains_two_pairs(cards: Sequence[Card]) -> tuple[Sequence[Card], tuple[int, ...]]:
    groups = _group_by_rank(cards)
    pairs = _ranks_with_at_least(groups, 2)
    if len(pairs) < 2:
        return [], ()
    return _build_hand(cards, groups, [(pairs[0], 2), (pairs[1], 2)])


def _contains_pair(cards: Sequence[Card]) -> tuple[Sequence[Card], tuple[int, ...]]:
    groups = _group_by_rank(cards)
    pairs = _ranks_with_at_least(groups, 2)
    if not pairs:
        return [], ()
    return _build_hand(cards, groups, [(pairs[0], 2)])


def _contains_high_card(cards: Sequence[Card]) -> tuple[Sequence[Card], tuple[int, ...]]:
    """Always succeeds for a non-empty input: the top five cards by rank."""
    if not cards:
        return [], ()
    return _build_hand(cards, _group_by_rank(cards), [])


def _contains_flush(cards: Sequence[Card]) -> tuple[Sequence[Card], tuple[int, ...]]:
    suits = defaultdict(list)

    for card in cards:
        suits[card.suit].append(card)

    flush_cards = [
        card
        for suited_cards in suits.values()
        if len(suited_cards) >= 5
        for card in suited_cards
    ]

    if not flush_cards:
        return [], ()

    best_flush = max(
        (
            sorted(
                suited_cards,
                key=lambda card: card.rank,
                reverse=True
            )[:5]
            for suited_cards in suits.values()
            if len(suited_cards) >= 5
        ),
        key=lambda flush: tuple(card.rank for card in flush)
    )

    tiebreaker = tuple(card.rank for card in best_flush)

    return best_flush, tiebreaker

def _contains_straight(cards: Sequence[Card]) -> tuple[Sequence[Card], tuple[int, ...]]:
    by_rank: dict[int, Card] = {}
    for card in cards:
        by_rank.setdefault(int(card.rank), card)

    if int(Rank.ACE) in by_rank:
        by_rank[1] = by_rank[int(Rank.ACE)]

    for high in range(int(Rank.ACE), 4, -1):
        window = range(high, high - 5, -1)
        if all(value in by_rank for value in window):
            return [by_rank[value] for value in window], (high,)

    return [], ()

def _contains_straight_flush(cards: Sequence[Card]) -> tuple[Sequence[Card], tuple[int, ...]]:
    """Find the highest straight flush. Returns (five cards high to low, (high_card,)),
    or ([], ()) if none. A royal flush is just a straight flush with high card 14."""
    by_suit: dict[Suit, list[Card]] = defaultdict(list)
    for card in cards:
        by_suit[card.suit].append(card)

    best: tuple[Sequence[Card], tuple[int, ...]] = ([], ())
    for suited in by_suit.values():
        if len(suited) < 5:
            continue
        result = _contains_straight(suited)
        if result[1] > best[1]:  # () < any non-empty tuple, so this also handles "none"
            best = result
    return best


# Assumes HandCategory, HandRank, Card, Sequence and the eight _contains_* functions are in scope.


class HandEvaluator:
    """Evaluates poker hands of 2 (pre-flop), 5 (flop), 6 (turn) or 7 (river) cards.

    Hands are compared through HandRank.comparison_key: category first, then the
    tiebreakers tuple, which is directly comparable within a category.
    """

    VALID_SIZES = (2, 5, 6, 7)

    # Checked from strongest to weakest; the first match is the hand.
    _DETECTORS = (
        (HandCategory.STRAIGHT_FLUSH, _contains_straight_flush),
        (HandCategory.FOUR_OF_A_KIND, _contains_four_of_a_kind),
        (HandCategory.FULL_HOUSE, _contains_full_house),
        (HandCategory.FLUSH, _contains_flush),
        (HandCategory.STRAIGHT, _contains_straight),
        (HandCategory.SET, _contains_set),
        (HandCategory.TWO_PAIRS, _contains_two_pairs),
        (HandCategory.PAIR, _contains_pair),
        (HandCategory.HIGH_CARD, _contains_high_card),
    )

    def evaluate(self, cards: Sequence[Card]) -> tuple[HandRank, Sequence[Card]]:
        """Return (rank, best_five_cards) for the given cards.

        With 2 cards (pre-flop) the returned cards are just those two cards.
        Raises ValueError for an unsupported number of cards or duplicate cards.
        """
        self._validate(cards)
        for category, detect in self._DETECTORS:
            hand_cards, tiebreakers = detect(cards)
            if hand_cards:
                return HandRank(category, tiebreakers), hand_cards
        raise AssertionError("unreachable: high card matches any non-empty hand")

    def compare(self, cards_a: Sequence[Card], cards_b: Sequence[Card]) -> int:
        """Return 1 if hand A is stronger, -1 if B is stronger, 0 for a tie."""
        key_a = self._key(cards_a)
        key_b = self._key(cards_b)
        return (key_a > key_b) - (key_a < key_b)

    def highest_hand(self, hands: Sequence[Sequence[Card]]) -> list[int]:
        """Return the indices (ascending) of the winning hand(s).

        Each entry of `hands` is one player's complete card set (hole cards + board).
        More than one index is returned on a tie (split pot). All hands must have
        the same number of cards.
        """
        if not hands:
            raise ValueError("At least one hand is required")
        if len({len(h) for h in hands}) != 1:
            raise ValueError("All hands must contain the same number of cards")

        keys = [self._key(h) for h in hands]
        best = max(keys)
        return [i for i, key in enumerate(keys) if key == best]

    def _key(self, cards: Sequence[Card]) -> tuple:
        rank, _ = self.evaluate(cards)
        return rank.comparison_key

    def _validate(self, cards: Sequence[Card]) -> None:
        if len(cards) not in self.VALID_SIZES:
            raise ValueError(
                f"Expected {self.VALID_SIZES} cards, got {len(cards)}"
            )
        if len({(c.rank, c.suit) for c in cards}) != len(cards):
            raise ValueError("Duplicate cards in hand")




