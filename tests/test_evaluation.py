from __future__ import annotations

import unittest

from monopoly.agent.evaluation import CheckpointEvaluator, EvaluationGameResult, TournamentSummary
from monopoly.agent.league import LeaguePolicySpec


class EvaluationTests(unittest.TestCase):
    def test_placements_from_assets_assigns_shared_places_on_ties(self) -> None:
        placements = CheckpointEvaluator._placements_from_assets({"A": 2000, "B": 1500, "C": 1500})

        self.assertEqual(1.0, placements["A"])
        self.assertEqual(2.0, placements["B"])
        self.assertEqual(2.0, placements["C"])

    def test_pair_score_and_elo_update_reward_better_placement(self) -> None:
        left_score = CheckpointEvaluator._pair_score(1.0, 2.0)
        left_rating, right_rating = CheckpointEvaluator._update_elo(1000.0, 1000.0, left_score)

        self.assertEqual(1.0, left_score)
        self.assertGreater(left_rating, 1000.0)
        self.assertLess(right_rating, 1000.0)

    def test_build_benchmark_summary_preserves_cross_play_rates_for_simple_tournament(self) -> None:
        tournament = TournamentSummary(
            game_count=2,
            average_steps=10.0,
            win_counts={"alpha": 1, "beta": 0},
            draw_count=1,
            average_assets={"alpha": 1600.0, "beta": 1400.0},
            average_rent_potential_trend=0.5,
            average_monopoly_denial_events=0.25,
            average_board_strength_trend=0.1,
            average_auction_bid_quality=1.2,
            results=(
                EvaluationGameResult(
                    seed=1,
                    lineup_labels=("alpha", "beta"),
                    winner_name="Seat 1",
                    winner_label="alpha",
                    step_count=8,
                    player_assets={"Seat 1": 1700, "Seat 2": 1200},
                    rent_potential_trend=0.4,
                    monopoly_denial_events=0.0,
                    board_strength_trend=0.2,
                    auction_bid_quality_sum=0.0,
                    auction_bid_quality_count=0,
                ),
                EvaluationGameResult(
                    seed=2,
                    lineup_labels=("alpha", "beta"),
                    winner_name=None,
                    winner_label=None,
                    step_count=12,
                    player_assets={"Seat 1": 1500, "Seat 2": 1500},
                    rent_potential_trend=0.6,
                    monopoly_denial_events=0.5,
                    board_strength_trend=0.0,
                    auction_bid_quality_sum=2.4,
                    auction_bid_quality_count=2,
                ),
            ),
        )
        summary = CheckpointEvaluator(device="cpu")._build_benchmark_summary(tournament)

        self.assertEqual(2, summary.game_count)
        self.assertIn("beta", summary.cross_play_win_rates["alpha"])
        self.assertGreater(summary.elo_ratings["alpha"], summary.elo_ratings["beta"])

    def test_build_controller_from_policy_spec_rejects_missing_scripted_variant(self) -> None:
        evaluator = CheckpointEvaluator(device="cpu")

        with self.assertRaises(ValueError):
            evaluator._build_controller_from_policy_spec(LeaguePolicySpec(label="broken", source="scripted"))


if __name__ == "__main__":
    unittest.main()