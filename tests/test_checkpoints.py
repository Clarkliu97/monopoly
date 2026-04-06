from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from monopoly.agent.checkpoints import default_checkpoint_path, resolve_checkpoint_path
from monopoly.agent.config import PolicyConfig, TrainingConfig
from monopoly.agent.league import LeagueEpisodeAssignment, LeagueManager, LeaguePolicySpec


class CheckpointAndLeagueTests(unittest.TestCase):
    def test_default_checkpoint_path_uses_conventional_directory_and_filename(self) -> None:
        path = default_checkpoint_path("C:/tmp/project")

        self.assertEqual(Path("C:/tmp/project/.checkpoints/latest.pt"), path)

    def test_resolve_checkpoint_path_prefers_existing_requested_path_and_falls_back_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            explicit = temp_path / "custom.pt"
            explicit.write_bytes(b"x")
            fallback = temp_path / ".checkpoints" / "latest.pt"
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_bytes(b"y")

            self.assertEqual(explicit.resolve(), resolve_checkpoint_path(explicit))

            previous_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_path)
                self.assertEqual(fallback.resolve(), resolve_checkpoint_path("missing.pt"))
            finally:
                os.chdir(previous_cwd)

    def test_resolve_checkpoint_path_raises_when_checkpoint_is_required_and_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint_path("definitely_missing.pt", require_exists=True)

    def test_league_policy_spec_and_assignment_round_trip_without_sharing_mutable_state(self) -> None:
        spec = LeaguePolicySpec(label="recent_1", source="recent", policy_state={"weights": [1, 2, 3]}, heuristic_scale=0.5)
        assignment = LeagueEpisodeAssignment(seat_labels=("current", "recent_1"), learning_player_names=("AI1",))

        restored_spec = LeaguePolicySpec.from_state(spec.to_state())
        restored_spec.policy_state["weights"].append(4)
        restored_assignment = LeagueEpisodeAssignment.from_state(assignment.to_state())

        self.assertEqual([1, 2, 3], spec.policy_state["weights"])
        self.assertEqual(("current", "recent_1"), restored_assignment.seat_labels)

    def test_league_manager_state_round_trip_and_weight_resolution_preserve_available_sources(self) -> None:
        training_config = TrainingConfig(
            worker_count=1,
            episodes_per_worker=1,
            players_per_game=2,
            use_league_self_play=True,
            league_snapshot_interval=1,
            league_recent_snapshot_count=2,
        )
        policy_config = PolicyConfig(seed=17, hidden_sizes=(64, 64), minibatch_size=8, ppo_epochs=1, device="cpu")
        manager = LeagueManager(training_config, policy_config)

        manager.record_snapshot({"weights": [1]}, 1)
        promoted = manager.maybe_promote_best({"weights": [2]}, iteration_index=2, benchmark_score=0.75)
        restored = LeagueManager.from_state(manager.to_state(), training_config=training_config, policy_config=policy_config)
        resolved = restored._resolve_available_weights({"best": 0.0, "recent": 0.0, "scripted": 0.0}, restored._source_pools({}))

        self.assertTrue(promoted)
        self.assertEqual(manager.opponent_pool_size(), restored.opponent_pool_size())
        self.assertIn("best_0002", {spec.label for spec in restored.benchmark_specs({"weights": [3]}, 3)})
        self.assertAlmostEqual(1.0 / 3.0, resolved["best"], places=6)
        self.assertAlmostEqual(1.0 / 3.0, resolved["recent"], places=6)
        self.assertAlmostEqual(1.0 / 3.0, resolved["scripted"], places=6)


if __name__ == "__main__":
    unittest.main()