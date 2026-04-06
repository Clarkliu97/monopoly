from __future__ import annotations

import unittest

from monopoly.agent.action_space import MonopolyActionSpace
from monopoly.agent.config import HeuristicWeights, PolicyConfig, RewardWeights, TrainingConfig
from monopoly.agent.features import ObservationEncoder
from monopoly.agent.league import LeaguePolicySpec
from monopoly.agent.model import TorchPolicyModel
from monopoly.agent.worker_pool import PersistentRolloutWorkerPool


class WorkerPoolTests(unittest.TestCase):
    def _make_pool(self) -> PersistentRolloutWorkerPool:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        model = TorchPolicyModel(encoder.observation_size, action_space.action_count, seed=3, hidden_sizes=(32, 32), device="cpu")
        return PersistentRolloutWorkerPool(
            initial_policy_state=model.to_state(),
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, max_steps_per_episode=1, players_per_game=2),
            policy_config=PolicyConfig(seed=3, hidden_sizes=(32, 32), minibatch_size=8, ppo_epochs=1, device="cpu"),
            reward_weights=RewardWeights(),
            heuristic_weights=HeuristicWeights(),
        )

    def test_collect_rejects_closed_pool(self) -> None:
        pool = self._make_pool()
        try:
            pool.close()
            with self.assertRaises(RuntimeError):
                pool.collect({}, iteration_index=0, heuristic_scale=1.0, use_heuristic_bias=False)
        finally:
            pool.close()

    def test_build_opponent_controller_rejects_invalid_league_specs(self) -> None:
        pool = self._make_pool()
        try:
            with self.assertRaises(ValueError):
                pool._build_opponent_controller(LeaguePolicySpec(label="broken_scripted", source="scripted"), seed=5)
            with self.assertRaises(ValueError):
                pool._build_opponent_controller(LeaguePolicySpec(label="broken_recent", source="recent"), seed=5)
        finally:
            pool.close()

    def test_build_opponent_controller_applies_policy_spec_heuristic_settings(self) -> None:
        pool = self._make_pool()
        try:
            encoder = ObservationEncoder()
            action_space = MonopolyActionSpace()
            model = TorchPolicyModel(encoder.observation_size, action_space.action_count, seed=9, hidden_sizes=(16, 16), device="cpu")
            controller = pool._build_opponent_controller(
                LeaguePolicySpec(
                    label="recent_snapshot",
                    source="recent",
                    policy_state=model.to_state(),
                    use_heuristic_bias=True,
                    heuristic_scale=0.35,
                ),
                seed=5,
            )

            self.assertTrue(controller.use_heuristic_bias)
            self.assertAlmostEqual(0.35, controller.heuristic_scale, places=6)
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()