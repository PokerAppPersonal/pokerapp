import random


from .card import Card, Suit, Rank




class Deck:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._cards = [
            Card(rank, suit)
            for rank in Rank
            for suit in Suit
        ]
        self._dealt = 0
        self.shuffle()


    def shuffle(self) -> None:
        self._rng.shuffle(self._cards)
        self._dealt = 0


    def deal(self, count: int = 1) -> list[Card]:
        if count < 0:
            raise ValueError("Cannot deal a negative number of cards")


        if count > self.remaining:
            raise ValueError("Not enough cards remaining")


        start = self._dealt
        end = start + count


        self._dealt = end
        return self._cards[start:end]


    def reset(self, shuffle: bool = False) -> None:
        self._dealt = 0


        if shuffle:
            self.shuffle()


    @property
    def remaining(self) -> int:
        return len(self._cards) - self._dealt
