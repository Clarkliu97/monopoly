from __future__ import annotations

import unittest

import numpy as np
import torch

from monopoly.agent.model import TorchPolicyModel, TrainingExample


class PolicyModelTests(unittest.TestCase):
    def test_batch_tensors_preserves_shapes_and_dtypes(self) -> None:
        model = TorchPolicyModel(observation_size=4, action_count=3, seed=5, hidden_sizes=(8, 8), device="cpu", model_type="mlp")
        examples = [
            TrainingExample(
                observation=[1.0, 2.0, 3.0, 4.0],
                action_mask=[True, False, True],
                heuristic_bias=[0.0, 0.5, -0.5],
                action_id=2,
                discounted_return=1.5,
                advantage=0.25,
                old_log_probability=-0.7,
                threat_target=0.1,
            ),
            TrainingExample(
                observation=[4.0, 3.0, 2.0, 1.0],
                action_mask=[False, True, True],
                heuristic_bias=[1.0, 0.0, 0.0],
                action_id=1,
                discounted_return=0.5,
                advantage=-0.1,
                old_log_probability=-0.2,
                threat_target=-0.3,
            ),
        ]

        batch = model.batch_tensors(examples)

        self.assertEqual(torch.float32, batch["observations"].dtype)
        self.assertEqual(torch.bool, batch["action_masks"].dtype)
        self.assertEqual(torch.long, batch["actions"].dtype)
        self.assertEqual((2, 4), tuple(batch["observations"].shape))
        self.assertEqual((2, 3), tuple(batch["action_masks"].shape))

    def test_apply_mask_adds_heuristics_and_blocks_illegal_actions(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        mask = torch.tensor([[True, False, True]], dtype=torch.bool)
        heuristic = torch.tensor([[0.5, 0.5, -1.0]], dtype=torch.float32)

        masked = TorchPolicyModel._apply_mask(logits, mask, heuristic, use_heuristic_bias=True)

        self.assertAlmostEqual(1.5, float(masked[0, 0]))
        self.assertLess(float(masked[0, 1]), -1e8)
        self.assertAlmostEqual(2.0, float(masked[0, 2]))

    def test_set_seed_makes_exploratory_sampling_repeatable(self) -> None:
        model = TorchPolicyModel(observation_size=4, action_count=3, seed=5, hidden_sizes=(8, 8), device="cpu", model_type="mlp")
        model.zero_parameters()
        observation = np.zeros(4, dtype=np.float32)
        action_mask = np.asarray([True, True, True], dtype=np.bool_)
        heuristic_bias = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)

        model.set_seed(99)
        first = model.act(observation, action_mask, heuristic_bias, explore=True)
        model.set_seed(99)
        second = model.act(observation, action_mask, heuristic_bias, explore=True)

        self.assertEqual(first.action_id, second.action_id)
        self.assertEqual(first.log_probability, second.log_probability)

    def test_predict_values_accepts_single_observation(self) -> None:
        model = TorchPolicyModel(observation_size=4, action_count=2, seed=5, hidden_sizes=(8, 8), device="cpu", model_type="mlp")
        values = model.predict_values(np.zeros(4, dtype=np.float32))

        self.assertEqual((1,), values.shape)


if __name__ == "__main__":
    unittest.main()