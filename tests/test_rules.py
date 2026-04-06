from __future__ import annotations

import unittest

from monopoly.game import Game
from monopoly.rules import MonopolyRules
from monopoly.spaces import RailroadPropertySpace, UtilityPropertySpace


class RuleTests(unittest.TestCase):
    def _assign_properties(self, game: Game, player, space_indexes: tuple[int, ...]) -> None:
        for space_index in space_indexes:
            game.board.get_space(space_index).assign_owner(player)

    def test_even_building_rule_is_enforced(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        brown_one = game.board.get_space(1)
        brown_two = game.board.get_space(3)

        brown_one.assign_owner(player)
        brown_two.assign_owner(player)
        player.cash = 1000

        game.build_on_property(player, "Old Kent Road")
        with self.assertRaises(ValueError):
            game.build_on_property(player, "Old Kent Road")

    def test_mortgage_requires_buildings_to_be_sold_first(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        brown_one = game.board.get_space(1)
        brown_two = game.board.get_space(3)

        brown_one.assign_owner(player)
        brown_two.assign_owner(player)
        player.cash = 1000
        game.build_on_property(player, "Old Kent Road")
        game.build_on_property(player, "Whitechapel Road")

        with self.assertRaises(ValueError):
            game.mortgage_property(player, "Old Kent Road")

    def test_bank_house_and_hotel_supply_updates_with_building(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        for space_index in (1, 3):
            game.board.get_space(space_index).assign_owner(player)
        player.cash = 5000

        for _ in range(4):
            game.build_on_property(player, "Old Kent Road")
            game.build_on_property(player, "Whitechapel Road")

        self.assertEqual(24, game.houses_remaining)
        self.assertEqual(12, game.hotels_remaining)

        game.build_on_property(player, "Old Kent Road")

        self.assertEqual(28, game.houses_remaining)
        self.assertEqual(11, game.hotels_remaining)
        self.assertTrue(game.board.get_space(1).has_hotel)

    def test_build_fails_when_bank_has_no_houses(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        for space_index in (1, 3):
            game.board.get_space(space_index).assign_owner(player)
        player.cash = 1000
        game.houses_remaining = 0

        with self.assertRaises(ValueError):
            game.build_on_property(player, "Old Kent Road")

    def test_automatic_liquidation_sells_buildings_and_mortgages_before_bankruptcy(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        for space_index in (1, 3):
            game.board.get_space(space_index).assign_owner(player)
        player.cash = 0
        game.board.get_space(1).building_count = 1
        game.board.get_space(3).building_count = 1
        game.houses_remaining = 30

        messages = game.charge_player(player, None, 100, "Large Tax")

        self.assertFalse(player.is_bankrupt)
        self.assertEqual(10, player.cash)
        self.assertTrue(game.board.get_space(1).mortgaged)
        self.assertTrue(game.board.get_space(3).mortgaged)
        self.assertTrue(any("sells a building" in message for message in messages))
        self.assertTrue(any("mortgages" in message for message in messages))

    def test_bankruptcy_to_creditor_transfers_assets_and_declares_winner(self) -> None:
        game = Game(["A", "B"])
        debtor = game.players[0]
        creditor = game.players[1]
        old_kent_road = game.board.get_space(1)
        old_kent_road.assign_owner(debtor)
        debtor.pay(old_kent_road.price)
        debtor.cash = 0

        messages = game.charge_player(debtor, creditor, 500, "Massive Rent")

        self.assertTrue(debtor.is_bankrupt)
        self.assertIs(old_kent_road.owner, creditor)
        self.assertIs(game.winner(), creditor)
        self.assertTrue(any("goes bankrupt" in message for message in messages))

    def test_unmortgage_requires_interest_payment_and_clears_mortgage(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        oriental = game.board.get_space(6)
        oriental.assign_owner(player)
        game.mortgage_property(player, "The Angel Islington")
        player.cash = 100

        messages = game.unmortgage_property(player, "The Angel Islington")

        self.assertFalse(oriental.mortgaged)
        self.assertEqual(45, player.cash)
        self.assertTrue(any("unmortgages The Angel Islington for $55" in message for message in messages))

    def test_calculate_rent_honors_utility_multiplier_override(self) -> None:
        game = Game(["A", "B"])
        owner = game.players[0]
        utility = game.board.get_space(12)
        utility.assign_owner(owner)

        rent = MonopolyRules.calculate_rent(game.board, utility, dice_total=7, utility_multiplier=10)

        self.assertEqual(70, rent)

    def test_street_rent_doubles_when_full_color_group_is_owned_without_buildings(self) -> None:
        game = Game(["A", "B"])
        owner = game.players[0]
        old_kent = game.board.get_space(1)
        whitechapel = game.board.get_space(3)

        old_kent.assign_owner(owner)
        whitechapel.assign_owner(owner)

        self.assertEqual(4, MonopolyRules.calculate_rent(game.board, old_kent, dice_total=7))
        self.assertEqual(8, MonopolyRules.calculate_rent(game.board, whitechapel, dice_total=7))

    def test_railroad_rent_scales_by_non_mortgaged_owned_count(self) -> None:
        game = Game(["A", "B"])
        owner = game.players[0]
        railroads = [space for space in game.board.spaces if isinstance(space, RailroadPropertySpace)]

        expected_rents = {1: 25, 2: 50, 3: 100, 4: 200}
        for owned_count, expected_rent in expected_rents.items():
            with self.subTest(owned_count=owned_count):
                for railroad in railroads:
                    railroad.release_to_bank()
                owner.properties.clear()
                for railroad in railroads[:owned_count]:
                    railroad.assign_owner(owner)

                rent = MonopolyRules.calculate_rent(game.board, railroads[0], dice_total=9)

                self.assertEqual(expected_rent, rent)

        railroads[1].mortgaged = True
        reduced_rent = MonopolyRules.calculate_rent(game.board, railroads[0], dice_total=9)
        self.assertEqual(100, reduced_rent)

    def test_utility_rent_uses_four_times_or_ten_times_dice_total(self) -> None:
        game = Game(["A", "B"])
        owner = game.players[0]
        utilities = [space for space in game.board.spaces if isinstance(space, UtilityPropertySpace)]

        utilities[0].assign_owner(owner)
        for dice_total, expected in ((2, 8), (7, 28), (12, 48)):
            with self.subTest(owned_utilities=1, dice_total=dice_total):
                self.assertEqual(expected, MonopolyRules.calculate_rent(game.board, utilities[0], dice_total=dice_total))

        utilities[1].assign_owner(owner)
        for dice_total, expected in ((2, 20), (7, 70), (12, 120)):
            with self.subTest(owned_utilities=2, dice_total=dice_total):
                self.assertEqual(expected, MonopolyRules.calculate_rent(game.board, utilities[0], dice_total=dice_total))

    def test_build_requires_complete_set_and_no_mortgaged_group_member(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        angel = game.board.get_space(6)
        euston = game.board.get_space(8)
        pentonville = game.board.get_space(9)
        player.cash = 1000

        angel.assign_owner(player)
        euston.assign_owner(player)

        with self.assertRaises(ValueError):
            game.build_on_property(player, "The Angel Islington")

        pentonville.assign_owner(player)
        game.mortgage_property(player, "Pentonville Road")

        with self.assertRaises(ValueError):
            game.build_on_property(player, "The Angel Islington")

        game.unmortgage_property(player, "Pentonville Road")
        messages = game.build_on_property(player, "The Angel Islington")

        self.assertEqual(1, angel.building_count)
        self.assertTrue(any("builds on The Angel Islington" in message for message in messages))

    def test_unmortgage_cost_rounds_interest_up(self) -> None:
        game = Game(["A", "B"])
        mayfair = game.board.get_space(39)

        self.assertEqual(220, MonopolyRules.unmortgage_cost(mayfair))

    def test_sell_building_rejects_uneven_sale(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        for index in (1, 3):
            game.board.get_space(index).assign_owner(player)
        player.cash = 2000
        game.board.get_space(1).building_count = 1
        game.board.get_space(3).building_count = 0

        can_sell, reason = MonopolyRules.can_sell_building(game.board, player, game.board.get_space(3))

        self.assertFalse(can_sell)
        self.assertIn("no buildings", reason)

    def test_selling_hotel_requires_bank_to_have_four_houses_available(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        for index in (1, 3):
            game.board.get_space(index).assign_owner(player)
        hotel_street = game.board.get_space(1)
        sister_street = game.board.get_space(3)
        hotel_street.building_count = 5
        sister_street.building_count = 5

        with self.assertRaises(ValueError):
            MonopolyRules.sell_building(game.board, player, hotel_street, houses_remaining=3)

    def test_can_build_house_rejects_uneven_group_progress(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        for index in (1, 3):
            game.board.get_space(index).assign_owner(player)
        player.cash = 2000
        game.board.get_space(1).building_count = 1
        game.board.get_space(3).building_count = 0

        can_build, reason = MonopolyRules.can_build_house(game.board, player, game.board.get_space(1))

        self.assertFalse(can_build)
        self.assertIn("evenly", reason)

    def test_auto_liquidation_sells_buildings_before_mortgaging_and_stops_when_debt_is_covered(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        self._assign_properties(game, player, (1, 3, 12))
        game.board.get_space(1).building_count = 1
        game.board.get_space(3).building_count = 1
        game.houses_remaining = 30
        player.cash = 0

        messages = game.charge_player(player, None, 180, "Emergency Tax")

        sell_indices = [index for index, message in enumerate(messages) if "sells a building" in message]
        mortgage_indices = [index for index, message in enumerate(messages) if "mortgages" in message]

        self.assertFalse(player.is_bankrupt)
        self.assertEqual(5, player.cash)
        self.assertEqual(32, game.houses_remaining)
        self.assertEqual([0, 1], sell_indices)
        self.assertTrue(mortgage_indices)
        self.assertGreater(min(mortgage_indices), max(sell_indices))
        self.assertTrue(game.board.get_space(12).mortgaged)
        self.assertTrue(game.board.get_space(1).mortgaged)
        self.assertTrue(game.board.get_space(3).mortgaged)


if __name__ == "__main__":
    unittest.main()
