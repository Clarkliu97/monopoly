from __future__ import annotations

import unittest

from monopoly.game import Game


class PlayerTests(unittest.TestCase):
    def test_add_remove_property_and_total_assets_are_stable(self) -> None:
        game = Game(["A", "B"])
        player = game.players[0]
        mediterranean = game.board.get_space(1)

        mediterranean.assign_owner(player)
        player.add_property(mediterranean)
        player.remove_property(mediterranean)
        player.add_property(mediterranean)

        self.assertEqual(1, len(player.properties))
        self.assertEqual(player.cash + mediterranean.price, player.total_assets_value())

    def test_summary_includes_role_and_jail_status(self) -> None:
        game = Game(["A", "B"], player_roles=["ai", "human"])
        player = game.players[0]
        player.in_jail = True

        summary = player.summary()

        self.assertIn("A (ai)", summary)
        self.assertIn("jail=in jail", summary)


if __name__ == "__main__":
    unittest.main()