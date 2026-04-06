from __future__ import annotations

import tempfile
import unittest

from monopoly.dice import Dice
from monopoly.game import Game
from monopoly.trading import TradeOffer


class GameTests(unittest.TestCase):
    def test_player_count_validation_rejects_invalid_counts(self) -> None:
        with self.assertRaises(ValueError):
            Game(["A"])
        with self.assertRaises(ValueError):
            Game(["A", "B", "C", "D", "E", "F", "G"])

    def test_starting_cash_can_be_configured(self) -> None:
        game = Game(["A", "B"], starting_cash=2000)

        self.assertEqual(2000, game.players[0].cash)
        self.assertEqual(2000, game.players[1].cash)
        self.assertEqual(2000, game.get_game_view().starting_cash)

    def test_starting_cash_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            Game(["A", "B"], starting_cash=0)

    def test_take_turn_can_buy_property(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1)]))
        game.players[0].position = 39
        report = game.take_turn()
        self.assertIn("A buys Old Kent Road for $60.", report.messages)

    def test_passing_go_collects_salary(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 2)]))
        game.players[0].position = 37
        starting_cash = game.players[0].cash
        report = game.take_turn(auto_buy_unowned=False)
        self.assertGreater(game.players[0].cash, starting_cash)
        self.assertTrue(any("passes Go" in message for message in report.messages))

    def test_declined_property_triggers_auction(self) -> None:
        game = Game(["A", "B", "C"], dice=Dice(scripted_rolls=[(1, 1)]))
        game.players[0].position = 39
        game.auction_bids_by_space[1] = {"A": 50, "B": 80, "C": 75}

        report = game.take_turn(auto_buy_unowned=False)

        self.assertTrue(any("Auction begins for Old Kent Road." in message for message in report.messages))
        self.assertTrue(any("B wins the auction for Old Kent Road at $80." in message for message in report.messages))
        self.assertIs(game.board.get_space(1).owner, game.players[1])

    def test_jail_get_out_of_jail_card_allows_normal_turn(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 2)]))
        player = game.players[0]
        player.in_jail = True
        player.position = 10
        player.get_out_of_jail_cards = 1

        report = game.take_turn(auto_buy_unowned=False)

        self.assertFalse(player.in_jail)
        self.assertEqual(13, player.position)
        self.assertTrue(any("uses a Get Out of Jail Free card" in message for message in report.messages))

    def test_third_failed_jail_turn_pays_fine_and_continues(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 2)]))
        player = game.players[0]
        player.in_jail = True
        player.position = 10
        player.jail_turns = 2
        game.auction_bids_by_space[13] = {}
        starting_cash = player.cash

        report = game.take_turn(auto_buy_unowned=False)

        self.assertFalse(player.in_jail)
        self.assertEqual(13, player.position)
        self.assertEqual(starting_cash - 50, player.cash)
        self.assertTrue(any("pays the jail fine and leaves jail" in message for message in report.messages))

    def test_rolling_doubles_to_leave_jail_consumes_turn(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(2, 2), (6, 6)]))
        player = game.players[0]
        player.in_jail = True
        player.position = 10

        game.take_turn(auto_buy_unowned=False)

        self.assertEqual(14, player.position)
        self.assertIs(game.current_player, game.players[1])

    def test_card_chain_can_send_player_from_chance_to_community_chest_then_jail(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(3, 4)]))
        player = game.players[0]
        player.position = 29
        go_back_three = next(card for card in game.board.chance_deck if card.name == "Go Back Three Spaces")
        go_to_jail = next(card for card in game.board.community_chest_deck if card.name == "Go to Jail")
        game.board.chance_deck.remove(go_back_three)
        game.board.community_chest_deck.remove(go_to_jail)
        game.board.chance_deck.appendleft(go_back_three)
        game.board.community_chest_deck.appendleft(go_to_jail)

        report = game.take_turn(auto_buy_unowned=False)

        self.assertTrue(player.in_jail)
        self.assertEqual(10, player.position)
        self.assertTrue(any("Go Back Three Spaces" in message for message in report.messages))
        self.assertTrue(any("Go to Jail" in message for message in report.messages))

    def test_take_turn_is_blocked_during_post_roll_phase(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 2)]))
        game.players[0].position = 9

        game.start_turn_interactive()

        with self.assertRaises(ValueError):
            game.take_turn()

    def test_bankruptcy_to_bank_returns_property_and_reclaims_building_supply(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        mediterranean = game.board.get_space(1)
        baltic = game.board.get_space(3)
        mediterranean.assign_owner(player)
        baltic.assign_owner(player)
        player.cash = 0
        mediterranean.building_count = 1
        baltic.building_count = 1
        game.houses_remaining = 30

        messages = game.charge_player(player, None, 500, "Tax Bill")

        self.assertTrue(player.is_bankrupt)
        self.assertIsNone(mediterranean.owner)
        self.assertEqual(32, game.houses_remaining)
        self.assertTrue(any("returns to the bank" in message for message in messages))

    def test_full_state_round_trip_preserves_frontend_state(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1), (1, 2)]), starting_cash=1800)
        game.players[0].position = 39
        game.start_turn_interactive()

        restored = Game.from_serialized_state(game.serialize_full_state())

        self.assertEqual(game.get_frontend_state(), restored.get_frontend_state())

    def test_full_state_round_trip_preserves_debug_next_rolls_by_player(self) -> None:
        game = Game(["A", "B"])
        game.debug_next_rolls_by_player = {"B": [(5, 6)]}

        restored = Game.from_serialized_state(game.serialize_full_state())

        self.assertEqual({"B": [(5, 6)]}, restored.debug_next_rolls_by_player)

    def test_full_state_round_trip_preserves_duplicate_named_chance_cards(self) -> None:
        game = Game(["A", "B"])
        nearest_station_cards = [card for card in game.board.chance_deck if card.name == "Advance to Nearest Station"]
        self.assertEqual(2, len(nearest_station_cards))
        game.board.chance_deck.remove(nearest_station_cards[0])
        game.board.chance_deck.remove(nearest_station_cards[1])
        game.board.chance_deck.appendleft(nearest_station_cards[1])
        game.board.chance_deck.appendleft(nearest_station_cards[0])

        restored = Game.from_serialized_state(game.serialize_full_state())

        restored_names = [card.name for card in list(restored.board.chance_deck)[:2]]
        self.assertEqual(["Advance to Nearest Station", "Advance to Nearest Station"], restored_names)

    def test_full_state_round_trip_preserves_pending_trade_and_trade_turn_bookkeeping(self) -> None:
        game = Game(["A", "B"])
        old_kent = game.board.get_space(1)
        whitechapel = game.board.get_space(3)
        old_kent.assign_owner(game.players[0])
        whitechapel.assign_owner(game.players[1])

        original_offer = TradeOffer(
            proposer=game.players[0],
            receiver=game.players[1],
            proposer_cash=50,
            receiver_properties=[whitechapel],
            note="Initial offer",
        )
        counter_offer = TradeOffer(
            proposer=game.players[1],
            receiver=game.players[0],
            proposer_properties=[whitechapel],
            receiver_properties=[old_kent],
            note="Counter offer",
        )

        game.propose_trade_interactive(original_offer)
        game.counter_trade_interactive(counter_offer)

        restored = Game.from_serialized_state(game.serialize_full_state())

        self.assertIsNotNone(restored.pending_trade_decision)
        self.assertEqual(1, restored.pending_trade_decision.counter_count)
        self.assertEqual(game.get_pending_action(), restored.get_pending_action())
        self.assertEqual(game._trade_proposals_this_turn, restored._trade_proposals_this_turn)
        self.assertEqual(game._blocked_trade_offer_signatures, restored._blocked_trade_offer_signatures)
        self.assertTrue(
            restored.is_trade_offer_blocked(
                restored.deserialize_trade_offer(game.serialize_trade_offer(original_offer))
            )
        )

    def test_forced_next_roll_is_consumed_for_matching_player(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1)]))
        game.debug_next_rolls_by_player = {"A": [(6, 5)]}

        report = game.take_turn(auto_buy_unowned=False)

        self.assertTrue(any("A rolls 6 and 5 (total 11)." in message for message in report.messages))
        self.assertEqual({}, game.debug_next_rolls_by_player)

    def test_save_and_load_file_restores_pending_state(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1)]))
        game.players[0].position = 39
        game.start_turn_interactive()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            path = handle.name

        try:
            game.save_to_file(path)
            restored = Game.load_from_file(path)
        finally:
            import os
            os.unlink(path)

        self.assertEqual(game.get_frontend_state(), restored.get_frontend_state())

    def test_two_doubles_grant_extra_rolls_until_a_non_double_ends_the_turn(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1), (2, 2), (1, 2)]))
        game.players[0].position = 9

        report = game.take_turn()

        self.assertEqual(18, game.players[0].position)
        self.assertEqual(1, game.turn_counter)
        self.assertIs(game.current_player, game.players[1])
        self.assertEqual(2, sum(1 for message in report.messages if "rolled doubles and takes another turn" in message))

    def test_third_consecutive_double_sends_player_to_jail_before_the_move(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 1), (2, 2), (3, 3)]))
        game.players[0].position = 8
        game.auction_bids_by_space[14] = {}

        report = game.take_turn(auto_buy_unowned=False)

        self.assertTrue(game.players[0].in_jail)
        self.assertEqual(10, game.players[0].position)
        self.assertTrue(any("rolled doubles three times and goes to jail" in message for message in report.messages))

    def test_failed_jail_attempt_increments_counter_and_leaves_player_in_place(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(1, 2)]))
        player = game.players[0]
        player.in_jail = True
        player.position = 10
        player.jail_turns = 0

        report = game.take_turn(auto_buy_unowned=False)

        self.assertTrue(player.in_jail)
        self.assertEqual(1, player.jail_turns)
        self.assertEqual(10, player.position)
        self.assertTrue(any("remains in jail" in message for message in report.messages))

    def test_jail_doubles_on_second_attempt_releases_player_and_moves_normally(self) -> None:
        game = Game(["A", "B"], dice=Dice(scripted_rolls=[(2, 2)]))
        player = game.players[0]
        player.in_jail = True
        player.position = 10
        player.jail_turns = 1

        report = game.take_turn(auto_buy_unowned=False)

        self.assertFalse(player.in_jail)
        self.assertEqual(14, player.position)
        self.assertEqual(0, player.jail_turns)
        self.assertTrue(any("rolls doubles and leaves jail" in message for message in report.messages))

    def test_auction_property_allows_single_valid_bidder_to_win(self) -> None:
        game = Game(["A", "B"])
        old_kent = game.board.get_space(1)

        result = game.auction_property(old_kent, {"A": 75, "B": 0})

        self.assertIs(result.winner, game.players[0])
        self.assertEqual(75, result.winning_bid)
        self.assertIs(old_kent.owner, game.players[0])
        self.assertTrue(any("wins the auction" in message for message in result.messages))

    def test_auction_property_ignores_unaffordable_bids_and_can_leave_property_with_bank(self) -> None:
        game = Game(["A", "B"])
        old_kent = game.board.get_space(1)

        result = game.auction_property(old_kent, {"A": 2000, "B": 1600})

        self.assertIsNone(result.winner)
        self.assertIsNone(old_kent.owner)
        self.assertTrue(any("cannot afford" in message for message in result.messages))
        self.assertTrue(any("bank keeps the property" in message for message in result.messages))

    def test_submit_auction_bid_enforces_scaled_minimum_raise(self) -> None:
        game = Game(["A", "B", "C"], dice=Dice(scripted_rolls=[(1, 1)]))
        game.players[0].position = 39
        game.start_turn_interactive()
        game.resolve_property_decision(False)
        game.submit_auction_bid("A", 80)

        with self.assertRaisesRegex(ValueError, r"minimum valid bid is \$85"):
            game.submit_auction_bid("B", 81)

    def test_pending_auction_minimum_raise_scales_with_property_value_and_current_bid(self) -> None:
        game = Game(["A", "B", "C"], dice=Dice(scripted_rolls=[(1, 1)]))
        game.players[0].position = 39
        game.start_turn_interactive()
        game.resolve_property_decision(False)
        game.submit_auction_bid("A", 300)

        pending_action = game.get_pending_action()

        self.assertIsNotNone(pending_action)
        self.assertIsNotNone(pending_action.auction)
        self.assertEqual(315, pending_action.auction.minimum_bid)

    def test_landing_on_a_mortgaged_property_does_not_charge_rent(self) -> None:
        game = Game(["A", "B"])
        tenant = game.players[0]
        owner = game.players[1]
        old_kent = game.board.get_space(1)
        old_kent.assign_owner(owner)
        old_kent.mortgaged = True
        tenant.position = 1

        messages = game.resolve_current_space(tenant, allow_property_purchase=False, dice_total=8)

        self.assertEqual(1500, tenant.cash)
        self.assertEqual(1500, owner.cash)
        self.assertEqual(["A lands on Old Kent Road.", "Old Kent Road is mortgaged, so no rent is due."], messages)


if __name__ == "__main__":
    unittest.main()
