from __future__ import annotations

"""Board construction and deck traversal helpers for the Monopoly rules engine.

This module owns the immutable layout of the standard board plus the mutable card
decks that are shared across gameplay, frontend rendering, and state
serialization. The rest of the project relies on these helpers to resolve space
lookups, card draws, wraparound movement, and colour-group membership.
"""

from collections import deque
from dataclasses import dataclass

from monopoly.cards import Card, create_chance_deck, create_community_chest_deck
from monopoly.constants import CHANCE, COMMUNITY_CHEST, INCOME_TAX_AMOUNT, LUXURY_TAX_AMOUNT
from monopoly.spaces import (
    ActionSpace,
    CardSpace,
    PropertySpace,
    RailroadPropertySpace,
    Space,
    StreetPropertySpace,
    TaxSpace,
    UtilityPropertySpace,
)


@dataclass(slots=True)
class Board:
    """Mutable game board state.

    The board combines the static list of spaces with the mutable Chance and
    Community Chest decks so a single object can answer rules questions and be
    serialized/restored as part of the full game state.
    """

    spaces: list[Space]
    chance_deck: deque[Card]
    community_chest_deck: deque[Card]

    def get_space(self, index: int) -> Space:
        """Return the board space at the given absolute index."""
        return self.spaces[index]

    def draw_card(self, deck_name: str) -> Card:
        """Draw the next card from the named deck and rotate non-kept cards.

        Keep-until-used cards leave the deck until an explicit call to
        :meth:`return_kept_card`, which matches Monopoly's jail-free-card flow.
        """
        deck = self.chance_deck if deck_name == CHANCE else self.community_chest_deck
        card = deck.popleft()
        if not card.keep_until_used:
            deck.append(card)
        return card

    def return_kept_card(self, card: Card) -> None:
        """Return a previously held keep-until-used card to the back of its deck."""
        deck = self.chance_deck if card.deck_name == CHANCE else self.community_chest_deck
        deck.append(card)

    def next_space_of_type(self, current_index: int, target_type: str) -> Space:
        """Find the next space of ``target_type`` while wrapping around the board."""
        for offset in range(1, len(self.spaces) + 1):
            candidate = self.spaces[(current_index + offset) % len(self.spaces)]
            if candidate.space_type == target_type:
                return candidate
        raise ValueError(f"No space of type {target_type} exists on the board.")

    def properties_in_color_group(self, color_group: str) -> list[StreetPropertySpace]:
        """Return the full street set for a colour group.

        Rules and agent heuristics both use this helper to answer monopoly,
        mortgage, and even-building questions.
        """
        return [
            space
            for space in self.spaces
            if isinstance(space, StreetPropertySpace) and space.color_group == color_group
        ]


def create_standard_board() -> Board:
    """Build the standard UK/Commonwealth Monopoly board and fresh decks."""
    spaces: list[Space] = [
        ActionSpace(0, "Go", "go", "Collect $200 when you pass."),
        StreetPropertySpace(1, "Old Kent Road", "street", 60, 30, color_group="Brown", rents=(2, 10, 30, 90, 160, 250), house_cost=50),
        CardSpace(2, "Community Chest", "card", COMMUNITY_CHEST),
        StreetPropertySpace(3, "Whitechapel Road", "street", 60, 30, color_group="Brown", rents=(4, 20, 60, 180, 320, 450), house_cost=50),
        TaxSpace(4, "Income Tax", "tax", INCOME_TAX_AMOUNT),
        RailroadPropertySpace(5, "King's Cross Station", "railroad", 200, 100),
        StreetPropertySpace(6, "The Angel Islington", "street", 100, 50, color_group="Light Blue", rents=(6, 30, 90, 270, 400, 550), house_cost=50),
        CardSpace(7, "Chance", "card", CHANCE),
        StreetPropertySpace(8, "Euston Road", "street", 100, 50, color_group="Light Blue", rents=(6, 30, 90, 270, 400, 550), house_cost=50),
        StreetPropertySpace(9, "Pentonville Road", "street", 120, 60, color_group="Light Blue", rents=(8, 40, 100, 300, 450, 600), house_cost=50),
        ActionSpace(10, "Jail / Just Visiting", "jail", "Players in jail stay here until released."),
        StreetPropertySpace(11, "Pall Mall", "street", 140, 70, color_group="Pink", rents=(10, 50, 150, 450, 625, 750), house_cost=100),
        UtilityPropertySpace(12, "Electric Company", "utility", 150, 75),
        StreetPropertySpace(13, "Whitehall", "street", 140, 70, color_group="Pink", rents=(10, 50, 150, 450, 625, 750), house_cost=100),
        StreetPropertySpace(14, "Northumberland Avenue", "street", 160, 80, color_group="Pink", rents=(12, 60, 180, 500, 700, 900), house_cost=100),
        RailroadPropertySpace(15, "Marylebone Station", "railroad", 200, 100),
        StreetPropertySpace(16, "Bow Street", "street", 180, 90, color_group="Orange", rents=(14, 70, 200, 550, 750, 950), house_cost=100),
        CardSpace(17, "Community Chest", "card", COMMUNITY_CHEST),
        StreetPropertySpace(18, "Marlborough Street", "street", 180, 90, color_group="Orange", rents=(14, 70, 200, 550, 750, 950), house_cost=100),
        StreetPropertySpace(19, "Vine Street", "street", 200, 100, color_group="Orange", rents=(16, 80, 220, 600, 800, 1000), house_cost=100),
        ActionSpace(20, "Free Parking", "free_parking", "No money is collected here in the standard rules."),
        StreetPropertySpace(21, "Strand", "street", 220, 110, color_group="Red", rents=(18, 90, 250, 700, 875, 1050), house_cost=150),
        CardSpace(22, "Chance", "card", CHANCE),
        StreetPropertySpace(23, "Fleet Street", "street", 220, 110, color_group="Red", rents=(18, 90, 250, 700, 875, 1050), house_cost=150),
        StreetPropertySpace(24, "Trafalgar Square", "street", 240, 120, color_group="Red", rents=(20, 100, 300, 750, 925, 1100), house_cost=150),
        RailroadPropertySpace(25, "Fenchurch St Station", "railroad", 200, 100),
        StreetPropertySpace(26, "Leicester Square", "street", 260, 130, color_group="Yellow", rents=(22, 110, 330, 800, 975, 1150), house_cost=150),
        StreetPropertySpace(27, "Coventry Street", "street", 260, 130, color_group="Yellow", rents=(22, 110, 330, 800, 975, 1150), house_cost=150),
        UtilityPropertySpace(28, "Water Works", "utility", 150, 75),
        StreetPropertySpace(29, "Piccadilly", "street", 280, 140, color_group="Yellow", rents=(24, 120, 360, 850, 1025, 1200), house_cost=150),
        ActionSpace(30, "Go To Jail", "go_to_jail", "Move directly to jail without collecting $200."),
        StreetPropertySpace(31, "Regent Street", "street", 300, 150, color_group="Green", rents=(26, 130, 390, 900, 1100, 1275), house_cost=200),
        StreetPropertySpace(32, "Oxford Street", "street", 300, 150, color_group="Green", rents=(26, 130, 390, 900, 1100, 1275), house_cost=200),
        CardSpace(33, "Community Chest", "card", COMMUNITY_CHEST),
        StreetPropertySpace(34, "Bond Street", "street", 320, 160, color_group="Green", rents=(28, 150, 450, 1000, 1200, 1400), house_cost=200),
        RailroadPropertySpace(35, "Liverpool St Station", "railroad", 200, 100),
        CardSpace(36, "Chance", "card", CHANCE),
        StreetPropertySpace(37, "Park Lane", "street", 350, 175, color_group="Dark Blue", rents=(35, 175, 500, 1100, 1300, 1500), house_cost=200),
        TaxSpace(38, "Super Tax", "tax", LUXURY_TAX_AMOUNT),
        StreetPropertySpace(39, "Mayfair", "street", 400, 200, color_group="Dark Blue", rents=(50, 200, 600, 1400, 1700, 2000), house_cost=200),
    ]
    return Board(spaces=spaces, chance_deck=create_chance_deck(), community_chest_deck=create_community_chest_deck())
