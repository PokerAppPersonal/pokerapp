from pokerapp.domain.cards.deck import Deck
import pytest

def test_deal_returns_correct_number_of_cards():
    deck = Deck(seed=42)

    cards = deck.deal(5)
    assert len(cards) == 5
    assert deck.remaining == 47



def test_dealing_entire_deck():
    deck = Deck(seed=42)

    cards = deck.deal(52)

    assert len(cards) == 52
    assert deck.remaining == 0


def test_cannot_deal_more_than_remaining():
    deck = Deck(seed=42)
    deck.deal(50)

    with pytest.raises(ValueError):
        deck.deal(3)


def test_same_seed_produces_same_order():
    deck1 = Deck(seed=42)
    deck2 = Deck(seed=42)

    assert deck1.deal(52) == deck2.deal(52)