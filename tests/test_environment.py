from __future__ import annotations

import unittest

from monopoly.agent.config import TrainingConfig
from monopoly.agent.environment import MonopolySelfPlayEnvironment
from monopoly.agent.reward import RewardFunction


class _ValueController:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[tuple[str, ...]] = []

    def choose_action(self, game, actor_name: str, explore: bool = True):
        del game, actor_name, explore
        raise AssertionError("choose_action should not be used in these helper tests")

    def evaluate_state_values(self, frontend_state, actor_names: tuple[str, ...]):
        del frontend_state
        self.calls.append(actor_names)
        return {actor_name: self.value for actor_name in actor_names}


class EnvironmentTests(unittest.TestCase):
    def test_create_game_applies_training_config_to_players_roles_cash_and_scripted_dice(self) -> None:
        environment = MonopolySelfPlayEnvironment(
            training_config=TrainingConfig(
                worker_count=1,
                episodes_per_worker=1,
                players_per_game=3,
                player_name_prefix="Bot",
                starting_cash=1800,
                scripted_rolls=((6, 6),),
            ),
            reward_function=RewardFunction(),
            controller=_ValueController(0.0),
        )

        game = environment.create_game(seed=5)

        self.assertEqual(("Bot1", "Bot2", "Bot3"), tuple(player.name for player in game.players))
        self.assertTrue(all(player.role == "ai" for player in game.players))
        self.assertTrue(all(player.cash == 1800 for player in game.players))
        self.assertEqual(12, game.dice.roll().total)

    def test_total_reward_sums_all_player_reward_trajectories(self) -> None:
        total = MonopolySelfPlayEnvironment._total_reward({"A": [1.0, -0.5], "B": [0.25]})

        self.assertEqual(0.75, total)

    def test_resolve_controller_prefers_player_specific_override(self) -> None:
        default_controller = _ValueController(1.0)
        override_controller = _ValueController(2.0)
        environment = MonopolySelfPlayEnvironment(
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, players_per_game=2),
            reward_function=RewardFunction(),
            controller=default_controller,
        )

        resolved = environment._resolve_controller("AI2", {"AI2": override_controller})

        self.assertIs(override_controller, resolved)
        self.assertIs(default_controller, environment._resolve_controller("AI1", {"AI2": override_controller}))

    def test_evaluate_learning_values_batches_when_one_controller_handles_all_players(self) -> None:
        controller = _ValueController(1.25)
        environment = MonopolySelfPlayEnvironment(
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, players_per_game=2),
            reward_function=RewardFunction(),
            controller=controller,
        )
        state = environment.create_game(seed=3).get_frontend_state()

        values = environment._evaluate_learning_values(state, ("AI1", "AI2"), controller_by_player=None)

        self.assertEqual({"AI1": 1.25, "AI2": 1.25}, values)
        self.assertEqual([("AI1", "AI2")], controller.calls)

    def test_evaluate_learning_values_splits_calls_across_mixed_controllers(self) -> None:
        default_controller = _ValueController(1.0)
        override_controller = _ValueController(2.5)
        environment = MonopolySelfPlayEnvironment(
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, players_per_game=2),
            reward_function=RewardFunction(),
            controller=default_controller,
        )
        state = environment.create_game(seed=3).get_frontend_state()

        values = environment._evaluate_learning_values(
            state,
            ("AI1", "AI2"),
            controller_by_player={"AI1": default_controller, "AI2": override_controller},
        )

        self.assertEqual({"AI1": 1.0, "AI2": 2.5}, values)
        self.assertEqual([("AI1",)], default_controller.calls)
        self.assertEqual([("AI2",)], override_controller.calls)


if __name__ == "__main__":
    unittest.main()