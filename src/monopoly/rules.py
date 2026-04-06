from __future__ import annotations

"""Core Monopoly rule helpers used by the game engine and tests.

The :class:`MonopolyRules` helpers centralize the math and validation around
rent, building transactions, and mortgage flow. `game.py` delegates to these
helpers so the turn engine can focus on sequencing while this module keeps the
business rules explicit and testable.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from monopoly.constants import COLOR_GROUP_SIZES, HOTEL_BUILDING_COUNT, HOUSE_LIMIT, UNMORTGAGE_INTEREST_RATE
from monopoly.spaces import PropertySpace, RailroadPropertySpace, StreetPropertySpace, UtilityPropertySpace

if TYPE_CHECKING:
    from monopoly.board import Board
    from monopoly.player import Player


@dataclass(slots=True)
class PaymentResult:
    """Outcome of a payment attempt, including all player-facing messages."""

    success: bool
    messages: list[str]


@dataclass(slots=True)
class BuildingTransaction:
    """Result of building or selling improvements on a street.

    ``house_delta`` and ``hotel_delta`` express how the bank inventory should be
    updated after the action succeeds.
    """

    messages: list[str]
    house_delta: int = 0
    hotel_delta: int = 0


class MonopolyRules:
    """Static rule helpers for rent, building, and mortgage validation."""

    @staticmethod
    def player_owns_full_color_group(board: Board, player: Player, color_group: str) -> bool:
        """Return whether ``player`` owns every street in ``color_group``."""
        owned_count = sum(
            1
            for space in board.properties_in_color_group(color_group)
            if space.owner is player
        )
        return owned_count == COLOR_GROUP_SIZES[color_group]

    @staticmethod
    def calculate_rent(board: Board, property_space: PropertySpace, dice_total: int, railroad_multiplier: int = 1, utility_multiplier: int | None = None) -> int:
        """Calculate rent for any property subtype.

        Streets use monopoly-aware printed rents, railroads scale by the count of
        non-mortgaged railroads the owner controls, and utilities use either the
        standard 4x/10x dice multiplier or an override supplied by a card effect.
        """
        owner = property_space.owner
        if owner is None:
            return 0

        if isinstance(property_space, StreetPropertySpace):
            owns_full_set = MonopolyRules.player_owns_full_color_group(board, owner, property_space.color_group)
            return property_space.current_rent(owns_full_set)

        if isinstance(property_space, RailroadPropertySpace):
            owned_railroads = sum(
                1
                for candidate in owner.properties
                if isinstance(candidate, RailroadPropertySpace) and not candidate.mortgaged
            )
            return property_space.current_rent(owned_railroads) * railroad_multiplier

        if isinstance(property_space, UtilityPropertySpace):
            owned_utilities = sum(
                1
                for candidate in owner.properties
                if isinstance(candidate, UtilityPropertySpace) and not candidate.mortgaged
            )
            if utility_multiplier is not None:
                return dice_total * utility_multiplier
            return property_space.current_rent(owned_utilities, dice_total)

        return 0

    @staticmethod
    def can_build_house(board: Board, player: Player, street: StreetPropertySpace) -> tuple[bool, str]:
        """Validate whether a house or hotel can be added to ``street``.

        The check enforces Monopoly's complete-set, unmortgaged-group, even-build,
        and affordability rules before the bank inventory is considered.
        """
        if street.owner is not player:
            return False, f"{player.name} does not own {street.name}."
        if street.mortgaged:
            return False, f"{street.name} is mortgaged."
        if not MonopolyRules.player_owns_full_color_group(board, player, street.color_group):
            return False, f"{player.name} does not own the full {street.color_group} set."
        if street.building_count >= HOTEL_BUILDING_COUNT:
            return False, f"{street.name} already has a hotel."
        if player.cash < street.house_cost:
            return False, f"{player.name} cannot afford a building on {street.name}."

        group = board.properties_in_color_group(street.color_group)
        if any(group_member.mortgaged for group_member in group):
            return False, f"Buildings cannot be added while a {street.color_group} property is mortgaged."

        minimum_building_count = min(group_member.building_count for group_member in group)
        if street.building_count > minimum_building_count:
            return False, "Buildings must be added evenly across a color group."

        return True, "ok"

    @staticmethod
    def build_house(
        board: Board,
        player: Player,
        street: StreetPropertySpace,
        houses_remaining: int,
        hotels_remaining: int,
    ) -> BuildingTransaction:
        """Apply a successful build and return the corresponding bank deltas."""
        can_build, reason = MonopolyRules.can_build_house(board, player, street)
        if not can_build:
            raise ValueError(reason)

        if street.building_count < HOUSE_LIMIT and houses_remaining <= 0:
            raise ValueError("The bank has no houses remaining.")
        if street.building_count == HOUSE_LIMIT and hotels_remaining <= 0:
            raise ValueError("The bank has no hotels remaining.")

        player.pay(street.house_cost)
        street.building_count += 1
        if street.building_count == HOTEL_BUILDING_COUNT:
            return BuildingTransaction(
                messages=[f"{player.name} upgrades {street.name} to a hotel for ${street.house_cost}."],
                house_delta=4,
                hotel_delta=-1,
            )
        return BuildingTransaction(
            messages=[f"{player.name} builds on {street.name} for ${street.house_cost}. Buildings now: {street.building_count}."],
            house_delta=-1,
        )

    @staticmethod
    def can_sell_building(board: Board, player: Player, street: StreetPropertySpace) -> tuple[bool, str]:
        """Validate whether a building can be sold from ``street`` evenly."""
        if street.owner is not player:
            return False, f"{player.name} does not own {street.name}."
        if street.building_count == 0:
            return False, f"{street.name} has no buildings to sell."

        group = board.properties_in_color_group(street.color_group)
        maximum_building_count = max(group_member.building_count for group_member in group)
        if street.building_count < maximum_building_count:
            return False, "Buildings must be sold evenly across a color group."

        return True, "ok"

    @staticmethod
    def sell_building(
        board: Board,
        player: Player,
        street: StreetPropertySpace,
        houses_remaining: int,
    ) -> BuildingTransaction:
        """Sell one building from ``street`` and return the resulting bank deltas.

        Selling a hotel requires the bank to have four houses available so the
        hotel can collapse back into houses, matching the tabletop rule.
        """
        can_sell, reason = MonopolyRules.can_sell_building(board, player, street)
        if not can_sell:
            raise ValueError(reason)

        house_delta = 1
        hotel_delta = 0
        if street.building_count == HOTEL_BUILDING_COUNT:
            if houses_remaining < HOUSE_LIMIT:
                raise ValueError("The bank does not have enough houses to break a hotel back into houses.")
            house_delta = -HOUSE_LIMIT
            hotel_delta = 1

        street.building_count -= 1
        proceeds = street.house_cost // 2
        player.receive(proceeds)
        return BuildingTransaction(
            messages=[f"{player.name} sells a building from {street.name} for ${proceeds}. Buildings now: {street.building_count}."],
            house_delta=house_delta,
            hotel_delta=hotel_delta,
        )

    @staticmethod
    def can_mortgage(board: Board, player: Player, property_space: PropertySpace) -> tuple[bool, str]:
        """Validate whether ``property_space`` may be mortgaged right now."""
        if property_space.owner is not player:
            return False, f"{player.name} does not own {property_space.name}."
        if property_space.mortgaged:
            return False, f"{property_space.name} is already mortgaged."
        if isinstance(property_space, StreetPropertySpace):
            group = board.properties_in_color_group(property_space.color_group)
            if any(group_member.building_count > 0 for group_member in group):
                return False, f"All buildings in the {property_space.color_group} group must be sold first."
        return True, "ok"

    @staticmethod
    def mortgage_property(board: Board, player: Player, property_space: PropertySpace) -> list[str]:
        """Mortgage a property and return the player-facing result messages."""
        can_mortgage, reason = MonopolyRules.can_mortgage(board, player, property_space)
        if not can_mortgage:
            raise ValueError(reason)

        property_space.mortgaged = True
        player.receive(property_space.mortgage_value)
        return [f"{player.name} mortgages {property_space.name} for ${property_space.mortgage_value}."]

    @staticmethod
    def unmortgage_cost(property_space: PropertySpace) -> int:
        """Return the mortgage value plus the standard 10% interest surcharge."""
        return property_space.mortgage_value + math.ceil(property_space.mortgage_value * UNMORTGAGE_INTEREST_RATE)

    @staticmethod
    def unmortgage_property(player: Player, property_space: PropertySpace) -> list[str]:
        """Clear a mortgage after charging the owner the required payoff cost."""
        if property_space.owner is not player:
            raise ValueError(f"{player.name} does not own {property_space.name}.")
        if not property_space.mortgaged:
            raise ValueError(f"{property_space.name} is not mortgaged.")

        cost = MonopolyRules.unmortgage_cost(property_space)
        if player.cash < cost:
            raise ValueError(f"{player.name} cannot afford to unmortgage {property_space.name}.")

        player.pay(cost)
        property_space.mortgaged = False
        return [f"{player.name} unmortgages {property_space.name} for ${cost}."]

