from __future__ import annotations

"""Chance and Community Chest card definitions.

Each card effect is implemented as a small game-layer callback so decks can be
serialized by card name while still executing the full engine behaviour when
drawn during normal or interactive play.
"""

from collections import deque
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from monopoly.constants import JAIL_INDEX

if TYPE_CHECKING:
    from monopoly.game import Game
    from monopoly.player import Player


CardEffect = Callable[["Game", "Player"], list[str]]


@dataclass(slots=True)
class Card:
    """One deck card with display text and an executable game effect."""

    name: str
    description: str
    deck_name: str
    effect: CardEffect
    keep_until_used: bool = False

    def apply(self, game: Game, player: Player) -> list[str]:
        """Execute the card effect against the current game and player."""
        return self.effect(game, player)


def create_community_chest_deck() -> deque[Card]:
    """Create the standard Community Chest deck in draw order."""
    return deque(
        [
            Card("Advance to Go", "Move to Go and collect $200.", "community_chest", _advance_to_go),
            Card("Bank Error", "Collect $200.", "community_chest", _collect_200),
            Card("Doctor's Fee", "Pay $50.", "community_chest", _pay_doctors_fee),
            Card("Sale of Stock", "Collect $50.", "community_chest", _collect_50),
            Card("Get Out of Jail Free", "Keep this card until needed.", "community_chest", _get_out_of_jail, True),
            Card("Go to Jail", "Go directly to jail.", "community_chest", _go_to_jail),
            Card("Holiday Fund Matures", "Collect $100.", "community_chest", _collect_100),
            Card("Income Tax Refund", "Collect $20.", "community_chest", _collect_20),
            Card("Birthday", "Collect $10 from every player.", "community_chest", _collect_10_from_each),
            Card("Life Insurance Matures", "Collect $100.", "community_chest", _collect_100),
            Card("Hospital Fees", "Pay $100.", "community_chest", _pay_hospital_fees),
            Card("School Fees", "Pay $50.", "community_chest", _pay_school_fees),
            Card("Consultancy Fee", "Collect $25.", "community_chest", _collect_25),
            Card("Street Repairs", "Pay $40 per house and $115 per hotel.", "community_chest", _street_repairs),
            Card("Beauty Contest", "Collect $10.", "community_chest", _collect_10),
            Card("Inheritance", "Collect $100.", "community_chest", _collect_100),
        ]
    )


def create_chance_deck() -> deque[Card]:
    """Create the standard Chance deck in draw order."""
    return deque(
        [
            Card("Advance to Go", "Move to Go and collect $200.", "chance", _advance_to_go),
            Card("Advance to Trafalgar Square", "Move to Trafalgar Square and collect $200 if you pass Go.", "chance", _advance_to_trafalgar_square),
            Card("Advance to Mayfair", "Move to Mayfair.", "chance", _advance_to_mayfair),
            Card("Advance to Pall Mall", "Move to Pall Mall and collect $200 if you pass Go.", "chance", _advance_to_pall_mall),
            Card("Advance to Nearest Station", "Move to the nearest station.", "chance", _advance_to_nearest_station),
            Card("Advance to Nearest Station", "Move to the nearest station.", "chance", _advance_to_nearest_station),
            Card("Advance to Nearest Utility", "Move to the nearest utility.", "chance", _advance_to_nearest_utility),
            Card("Bank Pays Dividend", "Collect $50.", "chance", _collect_50),
            Card("Get Out of Jail Free", "Keep this card until needed.", "chance", _get_out_of_jail, True),
            Card("Go Back Three Spaces", "Move back three spaces.", "chance", _go_back_three_spaces),
            Card("Go to Jail", "Go directly to jail.", "chance", _go_to_jail),
            Card("General Repairs", "Pay $25 per house and $100 per hotel.", "chance", _general_repairs),
            Card("Speeding Fine", "Pay $15.", "chance", _pay_15),
            Card("Take a Trip to King's Cross Station", "Move to King's Cross Station and collect $200 if you pass Go.", "chance", _advance_to_kings_cross_station),
            Card("Chairman of the Board", "Pay each player $50.", "chance", _pay_each_player_50),
            Card("Building Loan Matures", "Collect $150.", "chance", _collect_150),
        ]
    )


def _advance_to_go(game: Game, player: Player) -> list[str]:
    """Move directly to Go and collect salary."""
    game.move_player_to(player, 0, collect_go_salary=True)
    return [f"{player.name} moves to Go and collects $200."]


def _collect_200(game: Game, player: Player) -> list[str]:
    """Award $200 to the player."""
    player.receive(200)
    return [f"{player.name} collects $200."]


def _pay_doctors_fee(game: Game, player: Player) -> list[str]:
    """Charge the standard doctor's-fee penalty."""
    return game.charge_player(player, None, 50, "Doctor's Fee")


def _get_out_of_jail(game: Game, player: Player) -> list[str]:
    """Give the player a reusable Get Out of Jail Free card."""
    player.get_out_of_jail_cards += 1
    return [f"{player.name} keeps a Get Out of Jail Free card."]


def _go_to_jail(game: Game, player: Player) -> list[str]:
    """Send the player directly to jail."""
    game.send_player_to_jail(player)
    return [f"{player.name} goes directly to jail."]


def _collect_10_from_each(game: Game, player: Player) -> list[str]:
    """Collect $10 from every other active player."""
    messages: list[str] = []
    for other_player in game.active_players:
        if other_player is player:
            continue
        messages.extend(game.charge_player(other_player, player, 10, "Birthday"))
    return messages


def _collect_20(game: Game, player: Player) -> list[str]:
    """Award $20 to the player."""
    player.receive(20)
    return [f"{player.name} collects $20."]


def _street_repairs(game: Game, player: Player) -> list[str]:
    """Charge per-building street repair costs."""
    return game.charge_player(player, None, game.calculate_repair_fee(player, 40, 115), "Street Repairs")


def _advance_to_trafalgar_square(game: Game, player: Player) -> list[str]:
    """Advance to Trafalgar Square and resolve the landing normally."""
    messages = [f"{player.name} advances to Trafalgar Square."]
    game.move_player_to(player, 24, collect_go_salary=player.position > 24)
    messages.extend(game.resolve_current_space(player, allow_property_purchase=True))
    return messages


def _advance_to_mayfair(game: Game, player: Player) -> list[str]:
    """Advance to Mayfair and resolve the landing normally."""
    messages = [f"{player.name} advances to Mayfair."]
    game.move_player_to(player, 39, collect_go_salary=player.position > 39)
    messages.extend(game.resolve_current_space(player, allow_property_purchase=True))
    return messages


def _advance_to_pall_mall(game: Game, player: Player) -> list[str]:
    """Advance to Pall Mall and resolve the landing normally."""
    messages = [f"{player.name} advances to Pall Mall."]
    game.move_player_to(player, 11, collect_go_salary=player.position > 11)
    messages.extend(game.resolve_current_space(player, allow_property_purchase=True))
    return messages


def _advance_to_nearest_station(game: Game, player: Player) -> list[str]:
    """Advance to the nearest railroad and apply the doubled-rent card rule."""
    target = game.board.next_space_of_type(player.position, "railroad")
    messages = [f"{player.name} advances to the nearest station: {target.name}."]
    game.move_player_to(player, target.index, collect_go_salary=target.index < player.position)
    messages.extend(game.resolve_current_space(player, allow_property_purchase=True, forced_railroad_multiplier=2))
    return messages


def _advance_to_nearest_utility(game: Game, player: Player) -> list[str]:
    """Advance to the nearest utility and resolve rent using the special multiplier."""
    target = game.board.next_space_of_type(player.position, "utility")
    messages = [f"{player.name} advances to the nearest utility: {target.name}."]
    game.move_player_to(player, target.index, collect_go_salary=target.index < player.position)
    roll = game.dice.roll()
    messages.append(f"{player.name} rolls {roll.die_one} and {roll.die_two} for the utility card (total {roll.total}).")
    messages.extend(
        game.resolve_current_space(
            player,
            allow_property_purchase=True,
            dice_total=roll.total,
            forced_utility_multiplier=10,
        )
    )
    return messages


def _collect_50(game: Game, player: Player) -> list[str]:
    """Award $50 to the player."""
    player.receive(50)
    return [f"{player.name} collects $50."]


def _collect_100(game: Game, player: Player) -> list[str]:
    """Award $100 to the player."""
    player.receive(100)
    return [f"{player.name} collects $100."]


def _collect_150(game: Game, player: Player) -> list[str]:
    """Award $150 to the player."""
    player.receive(150)
    return [f"{player.name} collects $150."]


def _collect_25(game: Game, player: Player) -> list[str]:
    """Award $25 to the player."""
    player.receive(25)
    return [f"{player.name} collects $25."]


def _collect_10(game: Game, player: Player) -> list[str]:
    """Award $10 to the player."""
    player.receive(10)
    return [f"{player.name} collects $10."]


def _go_back_three_spaces(game: Game, player: Player) -> list[str]:
    """Move back three spaces and resolve the destination normally."""
    target_index = (player.position - 3) % len(game.board.spaces)
    messages = [f"{player.name} goes back three spaces."]
    game.move_player_to(player, target_index, collect_go_salary=False)
    messages.extend(game.resolve_current_space(player, allow_property_purchase=True))
    return messages


def _general_repairs(game: Game, player: Player) -> list[str]:
    """Charge general-repair costs based on houses and hotels owned."""
    return game.charge_player(player, None, game.calculate_repair_fee(player, 25, 100), "General Repairs")


def _pay_15(game: Game, player: Player) -> list[str]:
    """Charge the speeding-fine penalty."""
    return game.charge_player(player, None, 15, "Speeding Fine")


def _pay_hospital_fees(game: Game, player: Player) -> list[str]:
    """Charge hospital fees."""
    return game.charge_player(player, None, 100, "Hospital Fees")


def _pay_school_fees(game: Game, player: Player) -> list[str]:
    """Charge school fees."""
    return game.charge_player(player, None, 50, "School Fees")


def _advance_to_kings_cross_station(game: Game, player: Player) -> list[str]:
    """Advance to King's Cross Station and resolve the landing normally."""
    messages = [f"{player.name} takes a trip to King's Cross Station."]
    game.move_player_to(player, 5, collect_go_salary=player.position > 5)
    messages.extend(game.resolve_current_space(player, allow_property_purchase=True))
    return messages


def _pay_each_player_50(game: Game, player: Player) -> list[str]:
    """Pay $50 to every other active player."""
    messages: list[str] = []
    for other_player in game.active_players:
        if other_player is player:
            continue
        messages.extend(game.charge_player(player, other_player, 50, "Chairman of the Board"))
    return messages
