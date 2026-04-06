from __future__ import annotations

import unittest

from monopoly.game import Game
from monopoly.trading import TradeOffer


class TradingTests(unittest.TestCase):
    def test_trade_transfers_property_and_mortgage_interest(self) -> None:
        game = Game(["A", "B"])
        player_a = game.players[0]
        player_b = game.players[1]
        oriental = game.board.get_space(6)

        oriental.assign_owner(player_b)
        player_b.pay(oriental.price)
        game.mortgage_property(player_b, "The Angel Islington")
        starting_cash = player_a.cash

        trade = TradeOffer(
            proposer=player_a,
            receiver=player_b,
            proposer_cash=200,
            receiver_properties=[oriental],
        )

        messages = game.execute_trade(trade)

        self.assertIs(oriental.owner, player_a)
        self.assertLess(player_a.cash, starting_cash - 200 + 1)
        self.assertTrue(any("interest" in message for message in messages))

    def test_trade_validation_rejects_invalid_cash_and_self_trades(self) -> None:
        game = Game(["A", "B"])
        player_a = game.players[0]

        trade = TradeOffer(proposer=player_a, receiver=player_a, proposer_cash=2000)

        errors = trade.validate()

        self.assertTrue(any("trade with themselves" in error for error in errors))
        self.assertTrue(any("enough cash" in error for error in errors))

    def test_trade_can_transfer_multiple_mortgaged_properties_and_jail_cards(self) -> None:
        game = Game(["A", "B"])
        player_a = game.players[0]
        player_b = game.players[1]
        angel = game.board.get_space(6)
        station = game.board.get_space(5)
        angel.assign_owner(player_b)
        station.assign_owner(player_b)
        player_b.pay(angel.price + station.price)
        player_b.get_out_of_jail_cards = 1
        game.mortgage_property(player_b, "The Angel Islington")
        game.mortgage_property(player_b, "King's Cross Station")

        trade = TradeOffer(
            proposer=player_a,
            receiver=player_b,
            proposer_cash=300,
            receiver_properties=[angel, station],
            receiver_jail_cards=1,
        )

        messages = game.execute_trade(trade)

        self.assertIs(angel.owner, player_a)
        self.assertIs(station.owner, player_a)
        self.assertEqual(1, player_a.get_out_of_jail_cards)
        self.assertEqual(0, player_b.get_out_of_jail_cards)
        self.assertEqual(1185, player_a.cash)
        self.assertEqual(1650, player_b.cash)
        self.assertEqual(2, sum(1 for message in messages if "pays $" in message and "interest" in message))

    def test_trade_validation_rejects_properties_not_owned_by_the_listed_player(self) -> None:
        game = Game(["A", "B"])
        player_a = game.players[0]
        player_b = game.players[1]
        old_kent = game.board.get_space(1)

        trade = TradeOffer(
            proposer=player_a,
            receiver=player_b,
            proposer_properties=[old_kent],
        )

        errors = trade.validate()

        self.assertTrue(any("A does not own Old Kent Road" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
