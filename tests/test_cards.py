from __future__ import annotations

import unittest

from monopoly.dice import Dice
from monopoly.game import Game


class CardTests(unittest.TestCase):
    def test_repair_card_charges_houses_and_hotels(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        for space_index in (1, 3):
            street = game.board.get_space(space_index)
            street.assign_owner(player)
        game.board.get_space(1).building_count = 2
        game.board.get_space(3).building_count = 5
        street_repairs = next(card for card in game.board.community_chest_deck if card.name == "Street Repairs")

        messages = street_repairs.apply(game, player)

        self.assertEqual(1500 - 195, player.cash)
        self.assertTrue(any("pays $195" in message for message in messages))

    def test_chairman_of_the_board_skips_bankrupt_players(self) -> None:
        game = Game(["A", "B", "C"])
        player = game.players[0]
        game.players[2].is_bankrupt = True
        chairman = next(card for card in game.board.chance_deck if card.name == "Chairman of the Board")

        messages = chairman.apply(game, player)

        self.assertEqual(1450, player.cash)
        self.assertEqual(1550, game.players[1].cash)
        self.assertEqual(1500, game.players[2].cash)
        self.assertEqual(1, sum(1 for message in messages if "pays $50" in message))

    def test_nearest_utility_card_uses_forced_multiplier_when_owned(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(3, 4)]))
        player = game.players[0]
        owner = game.players[1]
        utility = game.board.get_space(12)
        utility.assign_owner(owner)
        player.position = 7
        advance = next(card for card in game.board.chance_deck if card.name == "Advance to Nearest Utility")

        messages = advance.apply(game, player)

        self.assertEqual(12, player.position)
        self.assertEqual(1500 - 70, player.cash)
        self.assertEqual(1500 + 70, owner.cash)
        self.assertTrue(any("nearest utility" in message for message in messages))
        self.assertTrue(any("utility card" in message for message in messages))

    def test_nearest_station_card_wraps_past_go_and_charges_double_rent(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        owner = game.players[1]
        station = game.board.get_space(5)
        station.assign_owner(owner)
        player.position = 36
        advance = next(card for card in game.board.chance_deck if card.name == "Advance to Nearest Station")

        messages = advance.apply(game, player)

        self.assertEqual(5, player.position)
        self.assertEqual(1650, player.cash)
        self.assertEqual(1550, owner.cash)
        self.assertTrue(any("nearest station" in message for message in messages))
        self.assertTrue(any("pays $50" in message for message in messages))

    def test_go_back_three_spaces_can_trigger_tax_space(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        player.position = 7
        tax_amount = game.board.get_space(4).tax_amount
        go_back_three = next(card for card in game.board.chance_deck if card.name == "Go Back Three Spaces")

        messages = go_back_three.apply(game, player)

        self.assertEqual(4, player.position)
        self.assertEqual(1500 - tax_amount, player.cash)
        self.assertTrue(any("Income Tax" in message for message in messages))

    def test_birthday_card_can_bankrupt_insolvent_players_and_collect_from_solvent_players(self) -> None:
        game = Game(["A", "B", "C"])
        player = game.players[0]
        bankrupt_target = game.players[1]
        solvent_target = game.players[2]
        bankrupt_target.cash = 0
        solvent_target.cash = 10
        birthday = next(card for card in game.board.community_chest_deck if card.name == "Birthday")

        messages = birthday.apply(game, player)

        self.assertEqual(1510, player.cash)
        self.assertTrue(bankrupt_target.is_bankrupt)
        self.assertEqual(0, solvent_target.cash)
        self.assertTrue(any("goes bankrupt" in message for message in messages))
        self.assertEqual(1, sum(1 for message in messages if "pays $10" in message))

    def test_get_out_of_jail_cards_from_both_decks_stack_on_the_player(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        chest_card = next(card for card in game.board.community_chest_deck if card.name == "Get Out of Jail Free")
        chance_card = next(card for card in game.board.chance_deck if card.name == "Get Out of Jail Free")

        chest_messages = chest_card.apply(game, player)
        chance_messages = chance_card.apply(game, player)

        self.assertEqual(2, player.get_out_of_jail_cards)
        self.assertTrue(any("keeps a Get Out of Jail Free card" in message for message in chest_messages))
        self.assertTrue(any("keeps a Get Out of Jail Free card" in message for message in chance_messages))


if __name__ == "__main__":
    unittest.main()