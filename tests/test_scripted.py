from __future__ import annotations

import unittest

from monopoly.agent.scripted import build_scripted_controller, default_scripted_profiles
from monopoly.game import Game


class ScriptedControllerTests(unittest.TestCase):
    def test_default_scripted_profiles_expose_all_supported_variants(self) -> None:
        profiles = default_scripted_profiles()

        self.assertEqual(
            {
                "conservative_liquidity_manager",
                "auction_value_shark",
                "expansionist_builder",
                "monopoly_denial_disruptor",
            },
            set(profiles),
        )
        self.assertTrue(all(profile.name == variant_name for variant_name, profile in profiles.items()))

    def test_build_scripted_controller_rejects_unknown_variant(self) -> None:
        with self.assertRaises(ValueError):
            build_scripted_controller("definitely_not_real")

    def test_scripted_controller_evaluate_state_values_tracks_relative_board_strength(self) -> None:
        controller = build_scripted_controller("expansionist_builder", seed=13)
        game = Game(["A", "B"], player_roles=["ai", "ai"])
        old_kent = game.board.get_space(1)
        baltic = game.board.get_space(3)
        old_kent.assign_owner(game.players[0])
        baltic.assign_owner(game.players[0])

        values = controller.evaluate_state_values(game.get_frontend_state(), ("A", "B"))

        self.assertGreater(values["A"], values["B"])

    def test_set_seed_makes_exploratory_choices_repeatable_for_same_state(self) -> None:
        controller = build_scripted_controller("auction_value_shark", seed=13)
        game = Game(["A", "B"], player_roles=["ai", "ai"])

        controller.set_seed(99)
        first = controller.choose_action(game, "A", explore=True)
        controller.set_seed(99)
        second = controller.choose_action(game, "A", explore=True)

        self.assertEqual(first.choice.action_id, second.choice.action_id)
        self.assertEqual(first.choice.action_label, second.choice.action_label)


if __name__ == "__main__":
    unittest.main()