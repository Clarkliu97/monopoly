from __future__ import annotations

"""Board space models used by the Monopoly engine.

These dataclasses define the public and rule-facing state for every board space
type. The game engine and rule helpers rely on them for ownership, mortgage,
building, and rent calculations.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from monopoly.constants import HOTEL_BUILDING_COUNT, HOUSE_LIMIT

if TYPE_CHECKING:
    from monopoly.player import Player


@dataclass(slots=True)
class Space:
    """Base board space with immutable identity and type metadata."""

    index: int
    name: str
    space_type: str


@dataclass(slots=True)
class PropertySpace(Space):
    """Ownable board space with bank price, mortgage value, and owner state."""

    price: int
    mortgage_value: int
    owner: Player | None = field(default=None, init=False)
    mortgaged: bool = field(default=False, init=False)

    def is_owned(self) -> bool:
        """Whether this property currently belongs to a player."""
        return self.owner is not None

    def assign_owner(self, player: Player) -> None:
        """Assign ownership to a player and register the property on that player."""
        self.owner = player
        player.add_property(self)

    def release_to_bank(self) -> None:
        """Remove ownership and clear mortgage state when the bank reclaims the property."""
        if self.owner is not None:
            self.owner.remove_property(self)
        self.owner = None
        self.mortgaged = False


@dataclass(slots=True)
class StreetPropertySpace(PropertySpace):
    """Color-group street property supporting houses, hotels, and scaled rent."""

    color_group: str
    rents: tuple[int, int, int, int, int, int]
    house_cost: int
    building_count: int = 0

    @property
    def has_hotel(self) -> bool:
        """Whether the street is upgraded to a hotel."""
        return self.building_count == HOTEL_BUILDING_COUNT

    @property
    def house_count(self) -> int:
        """Return house-equivalent count, capping hotels as four houses for bank supply."""
        if self.has_hotel:
            return HOUSE_LIMIT
        return self.building_count

    def current_rent(self, owns_full_set: bool) -> int:
        """Return current rent, accounting for monopoly bonus, buildings, and mortgages."""
        if self.mortgaged:
            return 0
        if self.building_count > 0:
            return self.rents[self.building_count]
        if owns_full_set:
            return self.rents[0] * 2
        return self.rents[0]

    def release_to_bank(self) -> None:
        """Reset development before returning the street to the bank."""
        self.building_count = 0
        PropertySpace.release_to_bank(self)


@dataclass(slots=True)
class RailroadPropertySpace(PropertySpace):
    """Railroad property with rent that doubles as more railroads are owned."""

    def current_rent(self, owned_railroads: int) -> int:
        """Return railroad rent for the owner's current railroad count."""
        if self.mortgaged:
            return 0
        return 25 * (2 ** max(owned_railroads - 1, 0))


@dataclass(slots=True)
class UtilityPropertySpace(PropertySpace):
    """Utility property whose rent scales with the dice total and owned utilities."""

    def current_rent(self, owned_utilities: int, dice_total: int) -> int:
        """Return utility rent for the provided dice total and ownership count."""
        if self.mortgaged:
            return 0
        multiplier = 10 if owned_utilities == 2 else 4
        return multiplier * dice_total


@dataclass(slots=True)
class TaxSpace(Space):
    """Board space that charges a fixed tax to the landing player."""

    tax_amount: int


@dataclass(slots=True)
class CardSpace(Space):
    """Board space that draws from one named card deck."""

    deck_name: str


@dataclass(slots=True)
class ActionSpace(Space):
    """Non-property board space with descriptive notes for the UI."""

    notes: str = ""
