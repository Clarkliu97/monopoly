from __future__ import annotations

"""Checkpoint and policy evaluation helpers for Monopoly agents."""

from dataclasses import dataclass
from pathlib import Path
from itertools import combinations

from monopoly.agent.board_analysis import analyze_transition
from monopoly.agent.checkpoints import load_agent_controller_from_checkpoint
from monopoly.agent.controller import AgentPolicyController
from monopoly.agent.action_space import MonopolyActionSpace
from monopoly.agent.features import ObservationEncoder
from monopoly.agent.heuristics import HeuristicScorer
from monopoly.agent.league import LeaguePolicySpec
from monopoly.agent.model import TorchPolicyModel
from monopoly.agent.scripted import ScriptedPolicyController, build_scripted_controller
from monopoly.constants import AI_ROLE
from monopoly.dice import Dice
from monopoly.game import Game


@dataclass(frozen=True, slots=True)
class EvaluationGameResult:
    """Recorded result and diagnostics for one evaluation game."""

    seed: int
    lineup_labels: tuple[str, ...]
    winner_name: str | None
    winner_label: str | None
    step_count: int
    player_assets: dict[str, int]
    rent_potential_trend: float
    monopoly_denial_events: float
    board_strength_trend: float
    auction_bid_quality_sum: float
    auction_bid_quality_count: int


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Aggregated summary for repeated self-play against one checkpoint."""

    checkpoint_path: str
    game_count: int
    win_count: int
    draw_count: int
    average_steps: float
    average_assets: float
    average_rent_potential_trend: float
    average_monopoly_denial_events: float
    average_board_strength_trend: float
    average_auction_bid_quality: float
    results: tuple[EvaluationGameResult, ...]


@dataclass(frozen=True, slots=True)
class TournamentSummary:
    """Aggregated summary for a multi-checkpoint tournament run."""

    game_count: int
    average_steps: float
    win_counts: dict[str, int]
    draw_count: int
    average_assets: dict[str, float]
    average_rent_potential_trend: float
    average_monopoly_denial_events: float
    average_board_strength_trend: float
    average_auction_bid_quality: float
    results: tuple[EvaluationGameResult, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkParticipantSummary:
    """Per-participant benchmark statistics including Elo and asset trends."""

    label: str
    win_rate: float
    average_placement: float
    average_assets: float
    average_asset_differential: float
    average_rent_potential_trend: float
    average_monopoly_denial_events: float
    average_board_strength_trend: float
    average_auction_bid_quality: float
    elo_rating: float


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteSummary:
    """Cross-play benchmark summary with pairwise win rates and Elo ratings."""

    game_count: int
    draw_count: int
    average_steps: float
    participants: tuple[BenchmarkParticipantSummary, ...]
    cross_play_win_rates: dict[str, dict[str, float]]
    elo_ratings: dict[str, float]
    results: tuple[EvaluationGameResult, ...]


class CheckpointEvaluator:
    """Run checkpoint, tournament, and league-policy benchmark evaluations."""

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = device

    def evaluate_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        seeds: list[int],
        players_per_game: int = 2,
        max_steps: int = 400,
    ) -> EvaluationSummary:
        """Evaluate one checkpoint against copies of itself across several seeds."""
        label = Path(checkpoint_path).stem
        controllers = [load_agent_controller_from_checkpoint(checkpoint_path, device=self.device) for _ in range(players_per_game)]
        results = [
            self._play_game([label] * players_per_game, controllers, seed=seed, max_steps=max_steps)
            for seed in seeds
        ]
        average_steps = 0.0 if not results else sum(result.step_count for result in results) / float(len(results))
        total_assets = 0.0 if not results else sum(sum(result.player_assets.values()) for result in results) / float(len(results))
        average_rent_potential_trend = 0.0 if not results else sum(result.rent_potential_trend for result in results) / float(len(results))
        average_monopoly_denial_events = 0.0 if not results else sum(result.monopoly_denial_events for result in results) / float(len(results))
        average_board_strength_trend = 0.0 if not results else sum(result.board_strength_trend for result in results) / float(len(results))
        total_bid_quality = sum(result.auction_bid_quality_sum for result in results)
        total_bid_quality_count = sum(result.auction_bid_quality_count for result in results)
        return EvaluationSummary(
            checkpoint_path=str(Path(checkpoint_path)),
            game_count=len(results),
            win_count=sum(1 for result in results if result.winner_label == label),
            draw_count=sum(1 for result in results if result.winner_label is None),
            average_steps=average_steps,
            average_assets=total_assets,
            average_rent_potential_trend=average_rent_potential_trend,
            average_monopoly_denial_events=average_monopoly_denial_events,
            average_board_strength_trend=average_board_strength_trend,
            average_auction_bid_quality=0.0 if total_bid_quality_count <= 0 else total_bid_quality / float(total_bid_quality_count),
            results=tuple(results),
        )

    def run_tournament(
        self,
        checkpoint_paths: list[str | Path],
        *,
        seeds: list[int],
        players_per_game: int = 2,
        max_steps: int = 400,
    ) -> TournamentSummary:
        """Run a checkpoint tournament and aggregate win and asset summaries."""
        if len(checkpoint_paths) < 2:
            raise ValueError("Tournament mode requires at least two checkpoints.")
        labels = [Path(path).stem for path in checkpoint_paths]
        controllers = [load_agent_controller_from_checkpoint(path, device=self.device) for path in checkpoint_paths]
        return self._run_tournament_from_controllers(labels, controllers, seeds=seeds, players_per_game=players_per_game, max_steps=max_steps)

    def run_benchmark_suite(
        self,
        checkpoint_paths: list[str | Path],
        *,
        seeds: list[int],
        players_per_game: int = 2,
        max_steps: int = 400,
    ) -> BenchmarkSuiteSummary:
        """Run cross-play benchmarking across saved checkpoints and build Elo summaries."""
        labels = [Path(path).stem for path in checkpoint_paths]
        controllers = [load_agent_controller_from_checkpoint(path, device=self.device) for path in checkpoint_paths]
        tournament = self._run_tournament_from_controllers(labels, controllers, seeds=seeds, players_per_game=players_per_game, max_steps=max_steps)
        return self._build_benchmark_summary(tournament)

    def run_policy_benchmark(
        self,
        policy_specs: list[LeaguePolicySpec],
        *,
        seeds: list[int],
        players_per_game: int = 2,
        max_steps: int = 400,
    ) -> BenchmarkSuiteSummary:
        """Benchmark an in-memory set of policy specs, including scripted opponents."""
        if len(policy_specs) < 2:
            raise ValueError("Benchmark mode requires at least two policy specs.")
        labels = [spec.label for spec in policy_specs]
        controllers = [self._build_controller_from_policy_spec(spec) for spec in policy_specs]
        tournament = self._run_tournament_from_controllers(labels, controllers, seeds=seeds, players_per_game=players_per_game, max_steps=max_steps)
        return self._build_benchmark_summary(tournament)

    def _run_tournament_from_controllers(
        self,
        labels: list[str],
        controllers: list[AgentPolicyController],
        *,
        seeds: list[int],
        players_per_game: int,
        max_steps: int,
    ) -> TournamentSummary:
        """Run round-robin seat-rotation games for the supplied controllers."""
        results: list[EvaluationGameResult] = []
        win_counts = {label: 0 for label in labels}
        asset_totals = {label: 0.0 for label in labels}
        asset_samples = {label: 0 for label in labels}
        draw_count = 0

        for seed in seeds:
            for rotation in range(len(controllers)):
                lineup_labels = [labels[(rotation + seat) % len(labels)] for seat in range(players_per_game)]
                lineup_controllers = [controllers[(rotation + seat) % len(controllers)] for seat in range(players_per_game)]
                result = self._play_game(lineup_labels, lineup_controllers, seed=seed, max_steps=max_steps)
                results.append(result)
                if result.winner_label is None:
                    draw_count += 1
                else:
                    win_counts[result.winner_label] += 1
                for label, assets in zip(result.lineup_labels, result.player_assets.values(), strict=True):
                    asset_totals[label] += assets
                    asset_samples[label] += 1

        average_steps = 0.0 if not results else sum(result.step_count for result in results) / float(len(results))
        average_rent_potential_trend = 0.0 if not results else sum(result.rent_potential_trend for result in results) / float(len(results))
        average_monopoly_denial_events = 0.0 if not results else sum(result.monopoly_denial_events for result in results) / float(len(results))
        average_board_strength_trend = 0.0 if not results else sum(result.board_strength_trend for result in results) / float(len(results))
        total_bid_quality = sum(result.auction_bid_quality_sum for result in results)
        total_bid_quality_count = sum(result.auction_bid_quality_count for result in results)
        average_assets = {
            label: 0.0 if asset_samples[label] == 0 else asset_totals[label] / float(asset_samples[label])
            for label in labels
        }
        return TournamentSummary(
            game_count=len(results),
            average_steps=average_steps,
            win_counts=win_counts,
            draw_count=draw_count,
            average_assets=average_assets,
            average_rent_potential_trend=average_rent_potential_trend,
            average_monopoly_denial_events=average_monopoly_denial_events,
            average_board_strength_trend=average_board_strength_trend,
            average_auction_bid_quality=0.0 if total_bid_quality_count <= 0 else total_bid_quality / float(total_bid_quality_count),
            results=tuple(results),
        )

    def _build_benchmark_summary(self, tournament: TournamentSummary) -> BenchmarkSuiteSummary:
        """Transform raw tournament outcomes into participant metrics and Elo ratings."""
        participant_labels = tuple(tournament.win_counts.keys())
        ratings = {label: 1000.0 for label in participant_labels}
        appearances = {label: 0 for label in participant_labels}
        placement_totals = {label: 0.0 for label in participant_labels}
        asset_totals = {label: 0.0 for label in participant_labels}
        asset_deltas = {label: 0.0 for label in participant_labels}
        rent_totals = {label: 0.0 for label in participant_labels}
        denial_totals = {label: 0.0 for label in participant_labels}
        board_totals = {label: 0.0 for label in participant_labels}
        bid_quality_totals = {label: 0.0 for label in participant_labels}
        bid_quality_samples = {label: 0 for label in participant_labels}
        pair_games = {label: {other: 0 for other in participant_labels if other != label} for label in participant_labels}
        pair_wins = {label: {other: 0.0 for other in participant_labels if other != label} for label in participant_labels}

        for result in tournament.results:
            placements = self._placements_from_assets(result.player_assets)
            seat_labels = tuple(result.lineup_labels)
            seat_players = tuple(result.player_assets.keys())
            for player_name, label in zip(seat_players, seat_labels, strict=True):
                appearances[label] += 1
                asset_value = float(result.player_assets[player_name])
                other_assets = [float(value) for other_name, value in result.player_assets.items() if other_name != player_name]
                asset_totals[label] += asset_value
                asset_deltas[label] += asset_value - (0.0 if not other_assets else sum(other_assets) / float(len(other_assets)))
                placement_totals[label] += placements[player_name]
                rent_totals[label] += result.rent_potential_trend
                denial_totals[label] += result.monopoly_denial_events
                board_totals[label] += result.board_strength_trend
                if result.auction_bid_quality_count > 0:
                    bid_quality_totals[label] += result.auction_bid_quality_sum / float(result.auction_bid_quality_count)
                    bid_quality_samples[label] += 1
            for left_index, right_index in combinations(range(len(seat_labels)), 2):
                left_label = seat_labels[left_index]
                right_label = seat_labels[right_index]
                left_player = seat_players[left_index]
                right_player = seat_players[right_index]
                pair_games[left_label][right_label] += 1
                pair_games[right_label][left_label] += 1
                left_score = self._pair_score(placements[left_player], placements[right_player])
                right_score = 1.0 - left_score
                pair_wins[left_label][right_label] += left_score
                pair_wins[right_label][left_label] += right_score
                ratings[left_label], ratings[right_label] = self._update_elo(ratings[left_label], ratings[right_label], left_score)

        participant_summaries = tuple(
            BenchmarkParticipantSummary(
                label=label,
                win_rate=0.0 if tournament.game_count <= 0 else tournament.win_counts.get(label, 0) / float(tournament.game_count),
                average_placement=0.0 if appearances[label] <= 0 else placement_totals[label] / float(appearances[label]),
                average_assets=0.0 if appearances[label] <= 0 else asset_totals[label] / float(appearances[label]),
                average_asset_differential=0.0 if appearances[label] <= 0 else asset_deltas[label] / float(appearances[label]),
                average_rent_potential_trend=0.0 if appearances[label] <= 0 else rent_totals[label] / float(appearances[label]),
                average_monopoly_denial_events=0.0 if appearances[label] <= 0 else denial_totals[label] / float(appearances[label]),
                average_board_strength_trend=0.0 if appearances[label] <= 0 else board_totals[label] / float(appearances[label]),
                average_auction_bid_quality=0.0 if bid_quality_samples[label] <= 0 else bid_quality_totals[label] / float(bid_quality_samples[label]),
                elo_rating=ratings[label],
            )
            for label in participant_labels
        )
        cross_play_win_rates = {
            label: {
                other: 0.0 if pair_games[label][other] <= 0 else pair_wins[label][other] / float(pair_games[label][other])
                for other in pair_games[label]
            }
            for label in participant_labels
        }
        return BenchmarkSuiteSummary(
            game_count=tournament.game_count,
            draw_count=tournament.draw_count,
            average_steps=tournament.average_steps,
            participants=participant_summaries,
            cross_play_win_rates=cross_play_win_rates,
            elo_ratings=ratings,
            results=tournament.results,
        )

    def _build_controller_from_policy_spec(self, spec: LeaguePolicySpec) -> AgentPolicyController:
        """Construct a controller from either a scripted spec or serialized model state."""
        if spec.source == "scripted":
            if spec.scripted_variant is None:
                raise ValueError(f"Scripted league spec {spec.label} is missing a scripted variant.")
            controller = build_scripted_controller(spec.scripted_variant, seed=7)
            controller.set_seed(7)
            return controller
        if spec.policy_state is None:
            raise ValueError(f"League spec {spec.label} is missing policy state for source {spec.source}.")
        model = TorchPolicyModel.from_state(spec.policy_state, device_override=self.device)
        controller = AgentPolicyController(
            policy_model=model,
            observation_encoder=ObservationEncoder(),
            action_space=MonopolyActionSpace(),
            heuristic_scorer=HeuristicScorer(),
        )
        controller.configure_heuristics(
            heuristic_scale=spec.heuristic_scale,
            use_heuristic_bias=spec.use_heuristic_bias,
        )
        return controller

    @staticmethod
    def _placements_from_assets(player_assets: dict[str, int]) -> dict[str, float]:
        """Assign placement numbers from descending asset totals, with ties sharing place."""
        return {
            player_name: 1.0 + sum(1 for other_assets in player_assets.values() if other_assets > assets)
            for player_name, assets in player_assets.items()
        }

    @staticmethod
    def _pair_score(left_placement: float, right_placement: float) -> float:
        """Convert two placements into a pairwise score for Elo updates."""
        if left_placement < right_placement:
            return 1.0
        if left_placement > right_placement:
            return 0.0
        return 0.5

    @staticmethod
    def _update_elo(left_rating: float, right_rating: float, left_score: float, *, k_factor: float = 24.0) -> tuple[float, float]:
        """Apply one Elo update step for a pairwise result."""
        expected_left = 1.0 / (1.0 + 10.0 ** ((right_rating - left_rating) / 400.0))
        expected_right = 1.0 - expected_left
        right_score = 1.0 - left_score
        return (
            left_rating + k_factor * (left_score - expected_left),
            right_rating + k_factor * (right_score - expected_right),
        )

    def _play_game(
        self,
        lineup_labels: list[str],
        controllers: list[AgentPolicyController | ScriptedPolicyController],
        *,
        seed: int,
        max_steps: int,
    ) -> EvaluationGameResult:
        """Play one deterministic evaluation game and collect aggregate diagnostics."""
        player_names = [f"Seat {index + 1}" for index in range(len(lineup_labels))]
        controller_by_player = {player_name: controller for player_name, controller in zip(player_names, controllers, strict=True)}
        label_by_player = {player_name: label for player_name, label in zip(player_names, lineup_labels, strict=True)}
        game = Game(
            player_names=player_names,
            player_roles=[AI_ROLE] * len(player_names),
            dice=Dice(seed=seed),
        )

        steps = 0
        rent_potential_trend = 0.0
        monopoly_denial_events = 0.0
        board_strength_trend = 0.0
        auction_bid_quality_sum = 0.0
        auction_bid_quality_count = 0
        while game.winner() is None and steps < max_steps:
            turn_plan = game.get_active_turn_plan()
            controller = controller_by_player[turn_plan.player_name]
            previous_state = game.get_frontend_state()
            decision = controller.choose_action(game, turn_plan.player_name, explore=False)
            trade_offer = None if decision.choice.trade_offer_payload is None else game.deserialize_trade_offer(decision.choice.trade_offer_payload)
            game.execute_legal_action(
                decision.choice.legal_action,
                bid_amount=decision.choice.bid_amount,
                trade_offer=trade_offer,
            )
            next_state = game.get_frontend_state()
            diagnostics = analyze_transition(previous_state, next_state)
            rent_potential_trend += diagnostics.rent_potential_delta
            monopoly_denial_events += diagnostics.monopoly_denial_events
            board_strength_trend += diagnostics.board_strength_trend
            if diagnostics.auction_bid_quality is not None:
                auction_bid_quality_sum += diagnostics.auction_bid_quality
                auction_bid_quality_count += 1
            steps += 1

        winner = game.winner()
        player_assets = {player.name: player.total_assets_value() for player in game.players}
        winner_name = None if winner is None else winner.name
        return EvaluationGameResult(
            seed=seed,
            lineup_labels=tuple(lineup_labels),
            winner_name=winner_name,
            winner_label=None if winner_name is None else label_by_player[winner_name],
            step_count=steps,
            player_assets=player_assets,
            rent_potential_trend=rent_potential_trend,
            monopoly_denial_events=monopoly_denial_events,
            board_strength_trend=board_strength_trend,
            auction_bid_quality_sum=auction_bid_quality_sum,
            auction_bid_quality_count=auction_bid_quality_count,
        )