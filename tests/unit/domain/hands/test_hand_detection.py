import pytest
import random
from itertools import combinations

from pokerapp.domain.cards.card import Suit, Rank, Card
from pokerapp.domain.cards.deck import Deck
from pokerapp.domain.hands.evaluator import (
    _contains_flush,
    _contains_straight,
    _contains_straight_flush,
    _contains_set,
    _contains_pair,
    _contains_two_pairs,
    _contains_full_house,
    _contains_high_card,
    _contains_four_of_a_kind
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
RANK_CHARS = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
              "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
SUIT_CHARS = {"c": Suit.CLUBS, "d": Suit.DIAMONDS, "h": Suit.HEARTS, "s": Suit.SPADES}
DECK_SIZE = 52


def make_card(rank: int, suit: Suit) -> Card:
    return Card(Rank(rank), suit)  # adjust if your constructor differs


def hand(spec: str) -> list[Card]:
    """'Ah Kd Ts' -> list of cards. T = ten."""
    return [make_card(RANK_CHARS[t[0]], SUIT_CHARS[t[1]]) for t in spec.split()]


def ranks(cards) -> list[int]:
    return [int(c.rank) for c in cards]


def suits(cards) -> set:
    return {c.suit for c in cards}


def assert_none(result):
    cards, tiebreaker = result
    assert list(cards) == []
    assert tiebreaker == ()


# --------------------------------------------------------------------------- #
# Brute-force oracles (check every 5-card combination)
# --------------------------------------------------------------------------- #
def _straight_high(five) -> int | None:
    rs = {int(c.rank) for c in five}
    if len(rs) != 5:
        return None
    if max(rs) - min(rs) == 4:
        return max(rs)
    if rs == {14, 2, 3, 4, 5}:
        return 5
    return None


def oracle_straight(cards) -> int | None:
    highs = [h for f in combinations(cards, 5) if (h := _straight_high(f))]
    return max(highs) if highs else None


def oracle_flush(cards) -> tuple[int, ...] | None:
    keys = [tuple(sorted(ranks(f), reverse=True))
            for f in combinations(cards, 5) if len(suits(f)) == 1]
    return max(keys) if keys else None


def oracle_straight_flush(cards) -> int | None:
    highs = [h for f in combinations(cards, 5)
             if len(suits(f)) == 1 and (h := _straight_high(f))]
    return max(highs) if highs else None


def check_against_oracles(cards):
    s_cards, s_tb = _contains_straight(cards)
    expected = oracle_straight(cards)
    assert s_tb == ((expected,) if expected else ()), cards

    f_cards, f_tb = _contains_flush(cards)
    expected_f = oracle_flush(cards)
    assert f_tb == (expected_f or ()), cards

    sf_cards, sf_tb = _contains_straight_flush(cards)
    expected_sf = oracle_straight_flush(cards)
    assert sf_tb == ((expected_sf,) if expected_sf else ()), cards
    if expected_sf:
        assert len(sf_cards) == 5 and len(suits(sf_cards)) == 1


# --------------------------------------------------------------------------- #
# Straight
# --------------------------------------------------------------------------- #
class TestStraight:
    @pytest.mark.parametrize("spec", ["Ah Kd", "2c 3c", "7h 7d", "Ah 2h"])
    def test_preflop_never_a_straight(self, spec):
        assert_none(_contains_straight(hand(spec)))

    @pytest.mark.parametrize("spec, high", [
        ("2c 3d 4h 5s 6c", 6),
        ("9c Td Jh Qs Kc", 13),
        ("Tc Jd Qh Ks Ac", 14),   # broadway
        ("Ac 2d 3h 4s 5c", 5),    # wheel
        ("6c 5d 4h 3s 2c", 6),    # unordered input
        ("5c 3d Ah 2s 4c", 5),    # unordered wheel
    ])
    def test_flop_straights(self, spec, high):
        cards, tb = _contains_straight(hand(spec))
        assert tb == (high,)
        assert len(cards) == 5

    @pytest.mark.parametrize("spec", [
        "2c 3d 4h 5s 7c",    # gap
        "2c 2d 4h 5s 6c",    # pair breaks it
        "Qc Kd Ah 2s 3c",    # no wrap-around
        "Kc Ad 2h 3s 4c",    # no wrap-around
        "2c 3d 4h 5s 5c",    # four consecutive + pair
        "Ac Kd Qh Js 9c",    # gap
        "7c 7d 7h 8s 9c",
    ])
    def test_flop_non_straights(self, spec):
        assert_none(_contains_straight(hand(spec)))

    def test_returned_cards_high_to_low(self):
        cards, _ = _contains_straight(hand("5c 9d 7h 6s 8c"))
        assert ranks(cards) == [9, 8, 7, 6, 5]

    def test_wheel_ace_is_last(self):
        cards, tb = _contains_straight(hand("3c Ad 5h 2s 4c"))
        assert ranks(cards) == [5, 4, 3, 2, 14]
        assert tb == (5,)

    def test_returned_cards_are_input_cards(self):
        original = hand("5c 9d 7h 6s 8c Kd")
        cards, _ = _contains_straight(original)
        assert all(c in original for c in cards)

    def test_turn_six_cards(self):
        assert _contains_straight(hand("2c 3d 4h 5s 6c Kd"))[1] == (6,)
        assert _contains_straight(hand("Kc 3d 4h 5s 6c 7d"))[1] == (7,)
        assert_none(_contains_straight(hand("2c 3d 4h 6s 7c Kd")))

    def test_six_card_run_takes_highest(self):
        cards, tb = _contains_straight(hand("4c 5d 6h 7s 8c 9d"))
        assert tb == (9,)
        assert ranks(cards) == [9, 8, 7, 6, 5]

    def test_river_seven_card_run_takes_highest(self):
        assert _contains_straight(hand("3c 4d 5h 6s 7c 8d 9h"))[1] == (9,)

    def test_wheel_loses_to_six_high(self):
        # A-2-3-4-5-6: the straight is 6-high, not the wheel
        assert _contains_straight(hand("Ac 2d 3h 4s 5c 6d"))[1] == (6,)

    def test_ace_high_beats_wheel_when_both_possible(self):
        # A,2,3,4,5 and T,J,Q,K,A cannot coexist in 7 cards, but A-high wins whenever available
        assert _contains_straight(hand("Tc Jd Qh Ks Ac 2d 3h"))[1] == (14,)

    def test_duplicates_dont_break_straight(self):
        assert _contains_straight(hand("5c 5d 6h 7s 8c 9d 9h"))[1] == (9,)
        assert _contains_straight(hand("Ac Ad 2h 3s 4c 5d 5h"))[1] == (5,)

    def test_trips_and_pairs_without_straight(self):
        assert_none(_contains_straight(hand("2c 2d 2h 9s 9c Kd Kh")))

    def test_straight_with_other_cards_around(self):
        assert _contains_straight(hand("2c 9d Th Js Qc Kd 2h"))[1] == (13,)

    def test_wheel_in_seven_cards_with_pair_of_aces(self):
        cards, tb = _contains_straight(hand("Ac Ad 2h 3s 4c 5d Kh"))
        assert tb == (5,)
        assert ranks(cards) == [5, 4, 3, 2, 14]

    def test_gutshots_and_open_enders_are_not_straights(self):
        assert_none(_contains_straight(hand("4c 5d 6h 7s Kc")))      # open-ended draw
        assert_none(_contains_straight(hand("4c 5d 7h 8s 9c Ad")))   # gutshot

    def test_does_not_mutate_input(self):
        original = hand("Ac 2d 3h 4s 5c 9d 9h")
        snapshot = list(original)
        _contains_straight(original)
        assert original == snapshot

    def test_accepts_tuple(self):
        assert _contains_straight(tuple(hand("2c 3d 4h 5s 6c")))[1] == (6,)


# --------------------------------------------------------------------------- #
# Flush
# --------------------------------------------------------------------------- #
class TestFlush:
    @pytest.mark.parametrize("spec", ["Ah Kh", "2c 3d", "7h 7d"])
    def test_preflop_never_a_flush(self, spec):
        assert_none(_contains_flush(hand(spec)))

    def test_flop_flush(self):
        cards, tb = _contains_flush(hand("2h 5h 9h Jh Ah"))
        assert tb == (14, 11, 9, 5, 2)
        assert ranks(cards) == [14, 11, 9, 5, 2]
        assert suits(cards) == {Suit.HEARTS}

    def test_flop_not_a_flush(self):
        assert_none(_contains_flush(hand("2h 5h 9h Jh Ad")))

    def test_four_suited_is_not_a_flush(self):
        assert_none(_contains_flush(hand("2h 5h 9h Jh Ad Ks 3c")))

    def test_turn_six_cards_five_suited(self):
        cards, tb = _contains_flush(hand("2s 5s 9s Js 3s Ad"))
        assert tb == (11, 9, 5, 3, 2)
        assert suits(cards) == {Suit.SPADES}

    def test_six_suited_takes_best_five(self):
        cards, tb = _contains_flush(hand("2d 4d 6d 8d Td Qd"))
        assert tb == (12, 10, 8, 6, 4)
        assert 2 not in ranks(cards)

    def test_river_seven_suited_takes_best_five(self):
        cards, tb = _contains_flush(hand("2c 3c 4c 9c Jc Kc Ac"))
        assert tb == (14, 13, 11, 9, 4)
        assert suits(cards) == {Suit.CLUBS}

    def test_off_suit_cards_ignored(self):
        cards, tb = _contains_flush(hand("Ad Kd 2h 3h 4h 5h 9h"))
        assert suits(cards) == {Suit.HEARTS}
        assert tb == (9, 5, 4, 3, 2)

    def test_higher_offsuit_cards_dont_leak_in(self):
        cards, tb = _contains_flush(hand("Ac Ad 2h 3h 4h 5h 7h"))
        assert tb == (7, 5, 4, 3, 2)

    def test_tiebreaker_comparison(self):
        better = _contains_flush(hand("Ah 9h 7h 5h 3h"))[1]
        worse = _contains_flush(hand("Ad 9d 7d 5d 2d"))[1]
        assert better > worse

    def test_flush_ignores_straightness(self):
        # a straight flush is also a flush
        cards, tb = _contains_flush(hand("5h 6h 7h 8h 9h"))
        assert tb == (9, 8, 7, 6, 5)

    def test_does_not_mutate_input(self):
        original = hand("2h 5h 9h Jh Ah 3c 4d")
        snapshot = list(original)
        _contains_flush(original)
        assert original == snapshot


# --------------------------------------------------------------------------- #
# Straight flush
# --------------------------------------------------------------------------- #
class TestStraightFlush:
    @pytest.mark.parametrize("spec", ["Ah Kh", "5h 6h", "2c 3d"])
    def test_preflop_never_a_straight_flush(self, spec):
        assert_none(_contains_straight_flush(hand(spec)))

    @pytest.mark.parametrize("spec, high", [
        ("2h 3h 4h 5h 6h", 6),
        ("5s 6s 7s 8s 9s", 9),
        ("Th Jh Qh Kh Ah", 14),   # royal
        ("Ac 2c 3c 4c 5c", 5),    # steel wheel
        ("9d 7d Td 8d Jd", 11),   # unordered
    ])
    def test_flop_straight_flushes(self, spec, high):
        cards, tb = _contains_straight_flush(hand(spec))
        assert tb == (high,)
        assert len(cards) == 5
        assert len(suits(cards)) == 1

    def test_flop_straight_and_flush_are_different_hands(self):
        # straight but not a flush
        assert_none(_contains_straight_flush(hand("5h 6h 7h 8h 9d")))
        # flush but not a straight
        assert_none(_contains_straight_flush(hand("5h 6h 7h 8h Jh")))

    def test_returned_order_high_to_low(self):
        cards, _ = _contains_straight_flush(hand("6h 9h 7h 5h 8h"))
        assert ranks(cards) == [9, 8, 7, 6, 5]

    def test_steel_wheel_ace_is_last(self):
        cards, tb = _contains_straight_flush(hand("3s As 5s 2s 4s"))
        assert ranks(cards) == [5, 4, 3, 2, 14]
        assert tb == (5,)

    def test_straight_and_flush_in_different_cards(self):
        # Hearts 2,3,4,5,9 (flush) + spade 6 (straight 2-6) but NOT a straight flush
        assert_none(_contains_straight_flush(hand("2h 3h 4h 5h 9h 6s")))
        assert_none(_contains_straight_flush(hand("2h 3h 4h 5h 9h 6s 7s")))

    def test_straight_flush_missing_one_suited_card(self):
        assert_none(_contains_straight_flush(hand("5h 6h 7h 8h 9d Th")))

    def test_turn_six_card_straight_flush(self):
        cards, tb = _contains_straight_flush(hand("5h 6h 7h 8h 9h Kd"))
        assert tb == (9,)

    def test_six_suited_run_takes_highest(self):
        cards, tb = _contains_straight_flush(hand("4h 5h 6h 7h 8h 9h"))
        assert tb == (9,)
        assert ranks(cards) == [9, 8, 7, 6, 5]

    def test_river_seven_suited_run_takes_highest(self):
        assert _contains_straight_flush(hand("3h 4h 5h 6h 7h 8h 9h"))[1] == (9,)

    def test_wheel_loses_to_six_high_straight_flush(self):
        assert _contains_straight_flush(hand("Ah 2h 3h 4h 5h 6h"))[1] == (6,)

    def test_wheel_straight_flush_with_extra_cards(self):
        cards, tb = _contains_straight_flush(hand("Ah 2h 3h 4h 5h Kd Ks"))
        assert tb == (5,)
        assert ranks(cards) == [5, 4, 3, 2, 14]

    def test_higher_offsuit_straight_does_not_override(self):
        # Hearts 5-9 is a straight flush; the spade ten makes a 10-high regular straight
        cards, tb = _contains_straight_flush(hand("5h 6h 7h 8h 9h Ts 2c"))
        assert tb == (9,)
        assert suits(cards) == {Suit.HEARTS}
        # and the plain straight check does see the higher straight
        assert _contains_straight(hand("5h 6h 7h 8h 9h Ts 2c"))[1] == (10,)

    def test_flush_has_more_than_five_but_sf_uses_subset(self):
        # Six hearts: 2,3,4,5,6 straight flush + K heart kicker
        cards, tb = _contains_straight_flush(hand("2h 3h 4h 5h 6h Kh Ad"))
        assert tb == (6,)
        assert 13 not in ranks(cards)

    def test_pair_inside_suited_cards_irrelevant(self):
        cards, tb = _contains_straight_flush(hand("5h 6h 7h 8h 9h 9d 9s"))
        assert tb == (9,)
        assert suits(cards) == {Suit.HEARTS}

    def test_duplicate_rank_other_suit_not_used(self):
        cards, tb = _contains_straight_flush(hand("5h 6h 7h 8h 9h 5d 6d"))
        assert suits(cards) == {Suit.HEARTS}

    def test_four_suited_connectors_not_enough(self):
        assert_none(_contains_straight_flush(hand("5h 6h 7h 8h 9d 9c Ts")))

    def test_royal_flush_among_seven(self):
        cards, tb = _contains_straight_flush(hand("Ts Js Qs Ks As 2d 2c"))
        assert tb == (14,)
        assert ranks(cards) == [14, 13, 12, 11, 10]

    def test_no_wraparound(self):
        assert_none(_contains_straight_flush(hand("Qh Kh Ah 2h 3h")))

    def test_straight_flush_beats_flush_tiebreaker(self):
        sf = _contains_straight_flush(hand("5h 6h 7h 8h 9h"))[1]
        assert sf == (9,)
        # Comparison across suits / higher: tuples compare correctly
        assert _contains_straight_flush(hand("Th Jh Qh Kh Ah"))[1] > sf

    def test_does_not_mutate_input(self):
        original = hand("5h 6h 7h 8h 9h Ts 2c")
        snapshot = list(original)
        _contains_straight_flush(original)
        assert original == snapshot


# --------------------------------------------------------------------------- #
# Consistency between the three functions
# --------------------------------------------------------------------------- #
class TestConsistency:
    @pytest.mark.parametrize("spec", [
        "5h 6h 7h 8h 9h", "Ah 2h 3h 4h 5h 6h", "Th Jh Qh Kh Ah 2d 3c",
        "3h 4h 5h 6h 7h 8h 9h",
    ])
    def test_straight_flush_implies_straight_and_flush(self, spec):
        cards = hand(spec)
        sf_cards, sf_tb = _contains_straight_flush(cards)
        assert sf_tb
        assert _contains_straight(cards)[1]
        assert _contains_flush(cards)[1]
        # the straight is at least as high as the straight flush
        assert _contains_straight(cards)[1] >= sf_tb


# --------------------------------------------------------------------------- #
# Randomised comparison with brute-force oracles (cards come from Deck)
# --------------------------------------------------------------------------- #
def shuffled_deck_cards(deck: Deck) -> list[Card]:
    """Reshuffle the deck and return all 52 cards in the new order."""
    deck.reset(shuffle=True)
    return deck.deal(DECK_SIZE)


@pytest.mark.parametrize("n_cards", [2, 5, 6, 7])
def test_random_hands_match_oracle(n_cards):
    deck = Deck(seed=20260101 + n_cards)
    for _ in range(4000):
        deck.reset(shuffle=True)
        cards = deck.deal(n_cards)
        if n_cards < 5:
            assert_none(_contains_straight(cards))
            assert_none(_contains_flush(cards))
            assert_none(_contains_straight_flush(cards))
        else:
            check_against_oracles(cards)


@pytest.mark.parametrize("n_cards", [5, 6, 7])
def test_forced_straight_flushes_match_oracle(n_cards):
    """Straight flushes are too rare in random hands, so plant them and
    fill the remaining slots with cards dealt from a shuffled Deck."""
    rng = random.Random(777 + n_cards)
    deck = Deck(seed=777 + n_cards)
    for _ in range(1500):
        suit = rng.choice(list(Suit))
        high = rng.randint(5, 14)
        run = [1 if r == 14 and high == 5 else r for r in range(high, high - 5, -1)]
        sf = [make_card(14 if r == 1 else r, suit) for r in run]
        filler = [c for c in shuffled_deck_cards(deck) if c not in sf][: n_cards - 5]
        cards = sf + filler
        rng.shuffle(cards)
        assert len(cards) == n_cards and len(set(cards)) == n_cards
        check_against_oracles(cards)
        assert _contains_straight_flush(cards)[1] >= (high,)


@pytest.mark.parametrize("n_cards", [5, 6, 7])
def test_forced_flush_plus_straight_off_suit(n_cards):
    """Hands with a guaranteed flush; the other cards may form an off-suit straight.
    A straight flush must only be reported when the oracle agrees."""
    rng = random.Random(4242 + n_cards)
    deck = Deck(seed=4242 + n_cards)
    for _ in range(1500):
        suit = rng.choice(list(Suit))
        order = shuffled_deck_cards(deck)
        suited = [c for c in order if c.suit == suit][:5]
        extra = [c for c in order if c not in suited][: n_cards - 5]
        cards = suited + extra
        rng.shuffle(cards)
        assert len(set(cards)) == n_cards
        assert _contains_flush(cards)[1]
        check_against_oracles(cards)


# Needs the helpers from test_hand_detection.py in scope:
#   hand, ranks, assert_none, make_card, shuffled_deck_cards, DECK_SIZE
# plus Deck, Suit, and the six functions from group_hands.py.


# --------------------------------------------------------------------------- #
# Brute-force oracle: rank every 5-card combination, keep the best
# --------------------------------------------------------------------------- #
def oracle_key(five) -> tuple[int, tuple[int, ...]]:
    """(category, tiebreaker). Categories: 8 straight flush, 7 quads, 6 full house,
    5 flush, 4 straight, 3 set, 2 two pairs, 1 pair, 0 high card.
    Works for 2-5 cards (straights/flushes need exactly 5)."""
    rs = [int(c.rank) for c in five]
    counts: dict[int, int] = {}
    for r in rs:
        counts[r] = counts.get(r, 0) + 1
    ordered = tuple(sorted(counts, key=lambda r: (counts[r], r), reverse=True))
    shape = sorted(counts.values(), reverse=True)

    is_flush = len(five) == 5 and len({c.suit for c in five}) == 1
    straight_high = None
    if len(five) == 5 and len(counts) == 5:
        if max(rs) - min(rs) == 4:
            straight_high = max(rs)
        elif set(rs) == {14, 2, 3, 4, 5}:
            straight_high = 5

    if is_flush and straight_high:
        return 8, (straight_high,)
    if shape[0] == 4:
        return 7, ordered
    if shape[:2] == [3, 2]:
        return 6, ordered
    if is_flush:
        return 5, tuple(sorted(rs, reverse=True))
    if straight_high:
        return 4, (straight_high,)
    if shape[0] == 3:
        return 3, ordered
    if shape[:2] == [2, 2]:
        return 2, ordered
    if shape[0] == 2:
        return 1, ordered
    return 0, ordered


def oracle_best(cards) -> tuple[int, tuple[int, ...]]:
    pool = combinations(cards, 5) if len(cards) >= 5 else [tuple(cards)]
    return max(oracle_key(f) for f in pool)


GROUP_FUNCS = {
    7: _contains_four_of_a_kind,
    6: _contains_full_house,
    3: _contains_set,
    2: _contains_two_pairs,
    1: _contains_pair,
    0: _contains_high_card,
}


def assert_valid_result(result, original):
    cards, tiebreaker = result
    assert len(cards) == min(5, len(original))
    assert len(set(cards)) == len(cards)               # no card used twice
    assert all(c in original for c in cards)           # only input cards
    # cards come as: group(s) first, then kickers; ranks agree with the tiebreaker
    assert list(dict.fromkeys(ranks(cards))) == list(tiebreaker)


def check_against_oracle(cards):
    """For every function: if the hand's true best category is lower than the
    function's, it must return nothing; if equal, it must match the oracle exactly.
    (If a higher category exists, the caller would pick that one instead.)"""
    best_cat, best_tb = oracle_best(cards)
    for category, func in GROUP_FUNCS.items():
        result = func(cards)
        if best_cat < category:
            assert_none(result)
        elif best_cat == category:
            assert result[1] == best_tb, (func.__name__, cards)
            assert_valid_result(result, cards)


# --------------------------------------------------------------------------- #
# Four of a kind
# --------------------------------------------------------------------------- #
class TestFourOfAKind:
    def test_flop(self):
        cards, tb = _contains_four_of_a_kind(hand("9c 9d 9h 9s 2c"))
        assert tb == (9, 2)
        assert ranks(cards) == [9, 9, 9, 9, 2]

    def test_quads_not_tripled_up_by_low_card(self):
        assert _contains_four_of_a_kind(hand("2c 2d 2h 2s Ac"))[1] == (2, 14)

    def test_kicker_is_highest_remaining_card(self):
        assert _contains_four_of_a_kind(hand("9c 9d 9h 9s Ac 2c"))[1] == (9, 14)   # turn
        assert _contains_four_of_a_kind(hand("9c 9d 9h 9s 3c Kd 2c"))[1] == (9, 13)  # river

    def test_kicker_taken_from_a_pair_or_trips(self):
        assert _contains_four_of_a_kind(hand("9c 9d 9h 9s Ac Ad 2c"))[1] == (9, 14)
        assert _contains_four_of_a_kind(hand("9c 9d 9h 9s Kc Kd Kh"))[1] == (9, 13)

    def test_ace_quads(self):
        cards, tb = _contains_four_of_a_kind(hand("Ac Ad Ah As 2c 3d 4h"))
        assert tb == (14, 4)
        assert ranks(cards) == [14, 14, 14, 14, 4]

    def test_unordered_input(self):
        assert _contains_four_of_a_kind(hand("2c 9s Ad 9h 9c 5d 9d"))[1] == (9, 14)

    def test_higher_quads_compare_higher(self):
        low = _contains_four_of_a_kind(hand("8c 8d 8h 8s Ac"))[1]
        high = _contains_four_of_a_kind(hand("9c 9d 9h 9s 2c"))[1]
        assert high > low

    def test_same_quads_kicker_decides(self):
        assert (_contains_four_of_a_kind(hand("9c 9d 9h 9s Ac"))[1]
                > _contains_four_of_a_kind(hand("9c 9d 9h 9s Kc"))[1])

    @pytest.mark.parametrize("spec", [
        "9c 9d", "9c 9d 9h 2s 3c", "9c 9d 2h 2s 3c", "9c 9d 9h 2s 2c",
        "9c 9d 9h 2s 3c 4d 5h", "Ac Kd Qh Js 9c",
    ])
    def test_none(self, spec):
        assert_none(_contains_four_of_a_kind(hand(spec)))


# --------------------------------------------------------------------------- #
# Full house
# --------------------------------------------------------------------------- #
class TestFullHouse:
    def test_flop(self):
        cards, tb = _contains_full_house(hand("9c 9d 9h 2s 2c"))
        assert tb == (9, 2)
        assert ranks(cards) == [9, 9, 9, 2, 2]

    def test_low_trips_high_pair(self):
        cards, tb = _contains_full_house(hand("2c 2d 2h 9s 9c"))
        assert tb == (2, 9)
        assert ranks(cards) == [2, 2, 2, 9, 9]

    def test_extra_card_ignored(self):
        cards, tb = _contains_full_house(hand("9c 9d 9h 2s 2c Ac"))
        assert tb == (9, 2)
        assert 14 not in ranks(cards)

    def test_two_trips_uses_lower_trips_as_pair(self):
        cards, tb = _contains_full_house(hand("9c 9d 9h 5s 5c 5d Ac"))
        assert tb == (9, 5)
        assert ranks(cards) == [9, 9, 9, 5, 5]

    def test_trips_with_two_pairs_picks_highest_pair(self):
        assert _contains_full_house(hand("9c 9d 9h 5s 5c 3d 3h"))[1] == (9, 5)
        assert _contains_full_house(hand("9c 9d 9h 3s 3c 5d 5h"))[1] == (9, 5)

    def test_higher_pair_beats_lower_trips_pair_choice(self):
        assert _contains_full_house(hand("5c 5d 5h Ks Kc 2d 2h"))[1] == (5, 13)

    def test_higher_trips_wins(self):
        assert (_contains_full_house(hand("Ac Ad Ah 2s 2c"))[1]
                > _contains_full_house(hand("Kc Kd Kh Qs Qc"))[1])

    def test_same_trips_pair_decides(self):
        assert (_contains_full_house(hand("9c 9d 9h Ks Kc"))[1]
                > _contains_full_house(hand("9c 9d 9h Qs Qc"))[1])

    @pytest.mark.parametrize("spec", [
        "9c 9d", "9c 9d 9h 2s 3c", "9c 9d 2h 2s 3c", "9c 9d 9h 9s 2c",
        "9c 9d 9h 2s 3c 4d 5h", "Ac Kd Qh Js 9c", "9c 9d 2h 2s 3c 3d 4h",
    ])
    def test_none(self, spec):
        assert_none(_contains_full_house(hand(spec)))


# --------------------------------------------------------------------------- #
# Set (three of a kind)
# --------------------------------------------------------------------------- #
class TestSet:
    def test_flop(self):
        cards, tb = _contains_set(hand("9c 9d 9h As Kc"))
        assert tb == (9, 14, 13)
        assert ranks(cards) == [9, 9, 9, 14, 13]

    def test_low_set(self):
        assert _contains_set(hand("2c 2d 2h Ks Qc"))[1] == (2, 13, 12)

    def test_river_picks_top_two_kickers(self):
        cards, tb = _contains_set(hand("9c 9d 9h As Kc Qd 2h"))
        assert tb == (9, 14, 13)
        assert len(cards) == 5

    def test_turn(self):
        assert _contains_set(hand("9c 9d 9h 3s Kc 5d"))[1] == (9, 13, 5)

    def test_unordered_input(self):
        assert _contains_set(hand("Kc 9d 2h 9h As 9c 4d"))[1] == (9, 14, 13)

    def test_higher_set_wins_kickers_break_ties(self):
        assert _contains_set(hand("Tc Td Th 2s 3c"))[1] > _contains_set(hand("9c 9d 9h As Kc"))[1]
        assert _contains_set(hand("9c 9d 9h As Kc"))[1] > _contains_set(hand("9c 9d 9h As Qc"))[1]

    @pytest.mark.parametrize("spec", [
        "9c 9d", "9c 9d 2h 2s 3c", "9c 9d Ah Ks Qc", "Ac Kd Qh Js 9c",
        "9c 9d 2h 2s 3c 3d 4h", "2c 3d 4h 5s 6c 7d 9h",
    ])
    def test_none(self, spec):
        assert_none(_contains_set(hand(spec)))


# --------------------------------------------------------------------------- #
# Two pairs
# --------------------------------------------------------------------------- #
class TestTwoPairs:
    def test_flop(self):
        cards, tb = _contains_two_pairs(hand("9c 9d 5h 5s Ac"))
        assert tb == (9, 5, 14)
        assert ranks(cards) == [9, 9, 5, 5, 14]

    def test_order_independent(self):
        assert _contains_two_pairs(hand("5h Ac 9d 5s 9c"))[1] == (9, 5, 14)

    def test_turn(self):
        assert _contains_two_pairs(hand("9c 9d 5h 5s Ac 2d"))[1] == (9, 5, 14)

    def test_river_kicker_is_highest_other_card(self):
        assert _contains_two_pairs(hand("9c 9d 5h 5s Ac Kd 2d"))[1] == (9, 5, 14)

    def test_three_pairs_uses_top_two(self):
        assert _contains_two_pairs(hand("Kc Kd 9h 9s 5c 5d Ad"))[1] == (13, 9, 14)

    def test_three_pairs_kicker_can_come_from_third_pair(self):
        cards, tb = _contains_two_pairs(hand("Kc Kd 9h 9s 5c 5d 2d"))
        assert tb == (13, 9, 5)
        assert ranks(cards) == [13, 13, 9, 9, 5]

    def test_higher_pair_decides_first(self):
        assert (_contains_two_pairs(hand("Kc Kd 2h 2s 3c"))[1]
                > _contains_two_pairs(hand("Qc Qd Jh Js Ac"))[1])

    def test_second_pair_then_kicker(self):
        assert (_contains_two_pairs(hand("9c 9d 6h 6s 2c"))[1]
                > _contains_two_pairs(hand("9c 9d 5h 5s Ac"))[1])
        assert (_contains_two_pairs(hand("9c 9d 5h 5s Ac"))[1]
                > _contains_two_pairs(hand("9c 9d 5h 5s Kc"))[1])

    @pytest.mark.parametrize("spec", [
        "9c 9d", "9c 5d", "9c 9d Ah Ks Qc", "9c 9d 9h As Kc",
        "Ac Kd Qh Js 9c", "9c 9d Ah Ks Qc Jd 2h",
    ])
    def test_none(self, spec):
        assert_none(_contains_two_pairs(hand(spec)))


# --------------------------------------------------------------------------- #
# Pair
# --------------------------------------------------------------------------- #
class TestPair:
    def test_preflop_pocket_pair(self):
        cards, tb = _contains_pair(hand("9c 9d"))
        assert tb == (9,)
        assert ranks(cards) == [9, 9]

    def test_flop(self):
        cards, tb = _contains_pair(hand("9c 9d Ah Ks Qc"))
        assert tb == (9, 14, 13, 12)
        assert ranks(cards) == [9, 9, 14, 13, 12]

    def test_low_pair_high_kickers(self):
        assert _contains_pair(hand("2c 2d Ah Ks Qc"))[1] == (2, 14, 13, 12)

    def test_turn(self):
        assert _contains_pair(hand("9c 9d Ah 3s Qc 2d"))[1] == (9, 14, 12, 3)

    def test_river_picks_top_three_kickers(self):
        cards, tb = _contains_pair(hand("9c 9d Ah Ks Qc Jd 2h"))
        assert tb == (9, 14, 13, 12)
        assert len(cards) == 5

    def test_pair_in_the_middle_of_unordered_input(self):
        assert _contains_pair(hand("Ah 9c Ks 2d 9d 7c Jh"))[1] == (9, 14, 13, 11)

    def test_higher_pair_wins(self):
        assert _contains_pair(hand("Tc Td 2h 3s 4c"))[1] > _contains_pair(hand("9c 9d Ah Ks Qc"))[1]

    def test_kickers_break_ties(self):
        assert _contains_pair(hand("9c 9d Ah Ks Qc"))[1] > _contains_pair(hand("9c 9d Ah Ks Jc"))[1]

    @pytest.mark.parametrize("spec", [
        "9c 5d", "Ah Kd", "Ac Kd Qh Js 9c", "2c 3d 4h 5s 6c 7d 9h",
    ])
    def test_none(self, spec):
        assert_none(_contains_pair(hand(spec)))


# --------------------------------------------------------------------------- #
# High card
# --------------------------------------------------------------------------- #
class TestHighCard:
    def test_preflop_two_cards(self):
        cards, tb = _contains_high_card(hand("2c Kd"))
        assert tb == (13, 2)
        assert ranks(cards) == [13, 2]

    def test_flop(self):
        cards, tb = _contains_high_card(hand("Ac Kd Qh Js 9c"))
        assert tb == (14, 13, 12, 11, 9)
        assert ranks(cards) == [14, 13, 12, 11, 9]

    def test_unordered_river_takes_top_five(self):
        cards, tb = _contains_high_card(hand("2c 4d 6h 9s Jc Kd 7h"))
        assert tb == (13, 11, 9, 7, 6)
        assert ranks(cards) == [13, 11, 9, 7, 6]

    def test_turn(self):
        assert _contains_high_card(hand("2c 4d 6h 9s Jc Kd"))[1] == (13, 11, 9, 6, 4)

    def test_returns_top_cards_even_when_a_better_hand_exists(self):
        assert _contains_high_card(hand("9c 9d Ah Ks Qc"))[1] == (14, 13, 12, 9, 9)

    def test_ace_is_high(self):
        assert _contains_high_card(hand("Ac 2d 3h 4s 6c"))[1] == (14, 6, 4, 3, 2)

    def test_tiebreaker_comparison(self):
        assert (_contains_high_card(hand("Ac Kd Qh Js 9c"))[1]
                > _contains_high_card(hand("Ac Kd Qh Js 8c"))[1])

    def test_empty_input(self):
        assert_none(_contains_high_card([]))


# --------------------------------------------------------------------------- #
# Generic behaviour
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("func", list(GROUP_FUNCS.values()))
def test_does_not_mutate_input_and_accepts_tuple(func):
    original = hand("9c 9d 9h 9s Ac Kd 2h")
    snapshot = list(original)
    func(original)
    assert original == snapshot
    assert func(tuple(original)) == func(original)


# --------------------------------------------------------------------------- #
# Randomised comparison with the brute-force oracle (cards dealt from Deck)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_cards", [2, 5, 6, 7])
def test_random_hands_match_oracle(n_cards):
    deck = Deck(seed=555 + n_cards)
    for _ in range(4000):
        deck.reset(shuffle=True)
        check_against_oracle(deck.deal(n_cards))


@pytest.mark.parametrize("n_cards", [5, 6, 7])
def test_planted_quads_match_oracle(n_cards):
    rng = random.Random(900 + n_cards)
    deck = Deck(seed=900 + n_cards)
    for _ in range(1500):
        rank = rng.randint(2, 14)
        quads = [make_card(rank, s) for s in Suit]
        filler = [c for c in shuffled_deck_cards(deck) if c not in quads][: n_cards - 4]
        cards = quads + filler
        rng.shuffle(cards)
        check_against_oracle(cards)
        assert _contains_four_of_a_kind(cards)[1][0] == rank


@pytest.mark.parametrize("n_cards", [5, 6, 7])
def test_planted_full_houses_match_oracle(n_cards):
    rng = random.Random(1200 + n_cards)
    deck = Deck(seed=1200 + n_cards)
    for _ in range(1500):
        trips_rank, pair_rank = rng.sample(range(2, 15), 2)
        planted = (
                [make_card(trips_rank, s) for s in rng.sample(list(Suit), 3)]
                + [make_card(pair_rank, s) for s in rng.sample(list(Suit), 2)]
        )
        filler = [c for c in shuffled_deck_cards(deck) if c not in planted][: n_cards - 5]
        cards = planted + filler
        rng.shuffle(cards)
        check_against_oracle(cards)
        assert _contains_full_house(cards)[1]  # trips + pair were planted