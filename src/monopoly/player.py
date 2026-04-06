from __future__ import annotations

"""Player state model used by the Monopoly engine.

The `Game` class mutates these objects directly while rule helpers and view
builders read from them to derive legal actions, value estimates, and UI state.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from monopoly.constants import HUMAN_ROLE, STARTING_CASH

if TYPE_CHECKING:
    from monopoly.spaces import PropertySpace


@dataclass(slots=True)
class Player:
    """Mutable player state tracked by the authoritative game engine."""

    name: str
    role: str = HUMAN_ROLE
    cash: int = STARTING_CASH
    position: int = 0
    in_jail: bool = False
    jail_turns: int = 0
    get_out_of_jail_cards: int = 0
    is_bankrupt: bool = False
    properties: list[PropertySpace] = field(default_factory=list)

    def can_afford(self, amount: int) -> bool:
        """Whether the player currently has enough cash to cover an amount."""
        return self.cash >= amount

    def receive(self, amount: int) -> None:
        """Add cash to the player's balance."""
        self.cash += amount

    def pay(self, amount: int) -> None:
        """Subtract cash from the player's balance."""
        self.cash -= amount

    def move_to(self, position: int) -> None:
        """Move the player's token to an absolute board position."""
        self.position = position

    def add_property(self, property_space: PropertySpace) -> None:
        """Register ownership of a property if it is not already listed."""
        if property_space not in self.properties:
            self.properties.append(property_space)

    def remove_property(self, property_space: PropertySpace) -> None:
        """Remove a property from the player's holdings if present."""
        if property_space in self.properties:
            self.properties.remove(property_space)

    def total_assets_value(self) -> int:
        """Estimate player net holdings as cash plus face value of owned properties."""
        total = self.cash
        for property_space in self.properties:
            total += property_space.price
        return total

    def summary(self) -> str:
        """Return a compact human-readable summary for debugging and logs."""
        property_names = ", ".join(property_space.name for property_space in self.properties) or "none"
        jail_status = "in jail" if self.in_jail else "free"
        return (
            f"{self.name} ({self.role}): cash=${self.cash}, position={self.position}, "
            f"properties=[{property_names}], jail={jail_status}"
        )
