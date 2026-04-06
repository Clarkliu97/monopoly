from __future__ import annotations

from collections import deque
import unittest

from monopoly.cards import Card
from monopoly.dice import Dice
from monopoly.game import Game


class ScenarioTests(unittest.TestCase):
    def test_multi_turn_scenario_tracks_purchases_cards_doubles_and_jail(self) -> None:
        game = Game(["A", "B", "C"], dice=Dice(scripted_rolls=[(1, 2), (3, 4), (1, 2), (1, 1), (1, 2), (1, 2), (1, 2)]))
        player_a, player_b, player_c = game.players
        player_a.position = 38
        player_c.position = 39
        self._move_card_to_top(game.board.chance_deck, "Go to Jail")
        self._move_card_to_top(game.board.community_chest_deck, "Bank Error")

        report_one = game.take_turn()
        self.assertEqual(["Old Kent Road"], [space.name for space in player_a.properties])
        self.assertEqual(1640, player_a.cash)
        self.assertEqual(1, player_a.position)
        self.assertEqual(1, game.turn_counter)
        self.assertEqual("B", game.current_player.name)
        self.assertTrue(any("buys Old Kent Road" in message for message in report_one.messages))

        report_two = game.take_turn(auto_buy_unowned=False)
        self.assertTrue(player_b.in_jail)
        self.assertEqual(10, player_b.position)
        self.assertEqual(2, game.turn_counter)
        self.assertEqual("C", game.current_player.name)
        self.assertTrue(any("Go to Jail" in message for message in report_two.messages))

        report_three = game.take_turn(auto_buy_unowned=False)
        self.assertEqual(1900, player_c.cash)
        self.assertEqual(2, player_c.position)
        self.assertEqual(3, game.turn_counter)
        self.assertEqual("A", game.current_player.name)
        self.assertTrue(any("Bank Error" in message for message in report_three.messages))

        report_four = game.take_turn()
        self.assertEqual(["Old Kent Road", "Whitechapel Road", "The Angel Islington"], [space.name for space in player_a.properties])
        self.assertEqual(1480, player_a.cash)
        self.assertEqual(6, player_a.position)
        self.assertEqual(4, game.turn_counter)
        self.assertEqual("B", game.current_player.name)
        self.assertEqual(1, sum(1 for message in report_four.messages if "rolled doubles and takes another turn" in message))

        report_five = game.take_turn(auto_buy_unowned=False)
        self.assertTrue(player_b.in_jail)
        self.assertEqual(1, player_b.jail_turns)
        self.assertEqual(10, player_b.position)
        self.assertEqual(5, game.turn_counter)
        self.assertEqual("C", game.current_player.name)
        self.assertTrue(any("remains in jail" in message for message in report_five.messages))

        report_six = game.take_turn()
        self.assertEqual(["King's Cross Station"], [space.name for space in player_c.properties])
        self.assertEqual(1700, player_c.cash)
        self.assertEqual(5, player_c.position)
        self.assertEqual(6, game.turn_counter)
        self.assertEqual("A", game.current_player.name)
        self.assertTrue(any("buys King's Cross Station" in message for message in report_six.messages))

    def test_multi_turn_scenario_tracks_bankruptcy_asset_transfer_and_continued_play(self) -> None:
        game = Game(["A", "B", "C"], dice=Dice(scripted_rolls=[(1, 2), (1, 2), (1, 2), (1, 2)]))
        player_a, player_b, player_c = game.players
        mayfair = game.board.get_space(39)
        station = game.board.get_space(5)
        mayfair.assign_owner(player_a)
        mayfair.building_count = 5
        station.assign_owner(player_b)
        player_b.pay(station.price)
        player_a.position = 17
        player_b.position = 36
        player_c.position = 39
        game._recalculate_building_supply_from_board()
        self._move_card_to_top(game.board.community_chest_deck, "Advance to Go")

        report_one = game.take_turn(auto_buy_unowned=False)
        self.assertEqual(20, player_a.position)
        self.assertEqual(1, game.turn_counter)
        self.assertEqual("B", game.current_player.name)
        self.assertTrue(any("Free Parking" in message for message in report_one.messages))

        report_two = game.take_turn(auto_buy_unowned=False)
        self.assertTrue(player_b.is_bankrupt)
        self.assertIs(station.owner, player_a)
        self.assertTrue(station.mortgaged)
        self.assertEqual(2900, player_a.cash)
        self.assertEqual(2, game.turn_counter)
        self.assertEqual("A", game.current_player.name)
        self.assertTrue(any("goes bankrupt" in message for message in report_two.messages))
        self.assertTrue(any("King's Cross Station transfers to A." in message for message in report_two.messages))

        report_three = game.take_turn()
        self.assertEqual(23, player_a.position)
        self.assertEqual(2680, player_a.cash)
        self.assertEqual(3, game.turn_counter)
        self.assertEqual("C", game.current_player.name)
        self.assertIsNone(game.winner())
        self.assertTrue(any("buys Fleet Street" in message for message in report_three.messages))

        report_four = game.take_turn(auto_buy_unowned=False)
        self.assertEqual(0, player_c.position)
        self.assertEqual(1900, player_c.cash)
        self.assertEqual(4, game.turn_counter)
        self.assertEqual("A", game.current_player.name)
        self.assertEqual(["Mayfair", "King's Cross Station", "Fleet Street"], [space.name for space in player_a.properties])
        self.assertTrue(any("Advance to Go" in message for message in report_four.messages))

    @staticmethod
    def _move_card_to_top(deck: deque[Card], card_name: str) -> None:
        for card in list(deck):
            if card.name == card_name:
                deck.remove(card)
                deck.appendleft(card)
                return
        raise AssertionError(f"Card named {card_name!r} was not found.")


if __name__ == "__main__":
    unittest.main()