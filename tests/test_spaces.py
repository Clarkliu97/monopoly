from __future__ import annotations

import unittest

from monopoly.player import Player
from monopoly.spaces import RailroadPropertySpace, StreetPropertySpace, UtilityPropertySpace


class SpaceTests(unittest.TestCase):
    def test_property_assignment_and_release_updates_owner_player_and_mortgage_state(self) -> None:
        player = Player("A")
        street = StreetPropertySpace(
            index=1,
            name="Old Kent Road",
            space_type="street",
            price=60,
            mortgage_value=30,
            color_group="Brown",
            rents=(2, 10, 30, 90, 160, 250),
            house_cost=50,
        )

        street.assign_owner(player)
        street.mortgaged = True
        street.building_count = 3
        street.release_to_bank()

        self.assertIsNone(street.owner)
        self.assertFalse(street.mortgaged)
        self.assertEqual(0, street.building_count)
        self.assertEqual([], player.properties)

    def test_street_rent_scales_for_monopolies_buildings_and_hotels(self) -> None:
        street = StreetPropertySpace(
            index=3,
            name="Whitechapel Road",
            space_type="street",
            price=60,
            mortgage_value=30,
            color_group="Brown",
            rents=(4, 20, 60, 180, 320, 450),
            house_cost=50,
        )

        self.assertEqual(4, street.current_rent(owns_full_set=False))
        self.assertEqual(8, street.current_rent(owns_full_set=True))
        street.building_count = 2
        self.assertEqual(60, street.current_rent(owns_full_set=True))
        street.building_count = 5
        self.assertTrue(street.has_hotel)
        self.assertEqual(4, street.house_count)
        self.assertEqual(450, street.current_rent(owns_full_set=True))
        street.mortgaged = True
        self.assertEqual(0, street.current_rent(owns_full_set=True))

    def test_railroad_rent_doubles_with_additional_railroads(self) -> None:
        railroad = RailroadPropertySpace(
            index=5,
            name="Kings Cross Station",
            space_type="railroad",
            price=200,
            mortgage_value=100,
        )

        self.assertEqual(25, railroad.current_rent(owned_railroads=1))
        self.assertEqual(50, railroad.current_rent(owned_railroads=2))
        self.assertEqual(200, railroad.current_rent(owned_railroads=4))
        railroad.mortgaged = True
        self.assertEqual(0, railroad.current_rent(owned_railroads=4))

    def test_utility_rent_uses_dice_total_and_owned_utility_count(self) -> None:
        utility = UtilityPropertySpace(
            index=12,
            name="Electric Company",
            space_type="utility",
            price=150,
            mortgage_value=75,
        )

        self.assertEqual(28, utility.current_rent(owned_utilities=1, dice_total=7))
        self.assertEqual(70, utility.current_rent(owned_utilities=2, dice_total=7))
        utility.mortgaged = True
        self.assertEqual(0, utility.current_rent(owned_utilities=2, dice_total=7))


if __name__ == "__main__":
    unittest.main()