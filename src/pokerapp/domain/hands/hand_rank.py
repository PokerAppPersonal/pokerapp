from dataclasses import dataclass

from pokerapp.domain.hands.categories import HandCategory


@dataclass(frozen=True, slots=True)
class HandRank:
    category: HandCategory
    tiebreakers: tuple[int, ...]

@property
def comparison_key(self) -> tuple:
    return self.category, self.tiebreakers
