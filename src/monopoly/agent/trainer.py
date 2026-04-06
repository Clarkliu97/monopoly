from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn.functional as F
from tqdm import tqdm

from monopoly.agent.action_space import MonopolyActionSpace
from monopoly.agent.config import HeuristicWeights, PolicyConfig, RewardWeights, TrainingConfig
from monopoly.agent.controller import AgentPolicyController
from monopoly.agent.environment import MonopolySelfPlayEnvironment, SelfPlayEpisode
from monopoly.agent.features import ObservationEncoder
from monopoly.agent.heuristics import HeuristicScorer
from monopoly.agent.league import LeagueManager, LeagueRolloutPlan
from monopoly.agent.model import TorchPolicyModel, TrainingExample
from monopoly.agent.reward import RewardFunction
from monopoly.agent.worker_pool import PersistentRolloutWorkerPool, RolloutWorkerResult

if TYPE_CHECKING:
    from monopoly.agent.evaluation import BenchmarkSuiteSummary


CURRENT_CHECKPOINT_SCHEMA_VERSION = 5


@dataclass(frozen=True, slots=True)
class TrainerCheckpoint:
    schema_version: int
    policy_state: dict[str, Any]
    policy_config: dict[str, Any]
    reward_weights: dict[str, Any]
    heuristic_weights: dict[str, Any]
    training_config: dict[str, Any]
    completed_iterations: int
    league_state: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class TrainingIterationStats:
    iteration_index: int
    example_count: int
    episode_count: int
    average_steps: float
    average_macro_steps: float
    average_raw_actions: float
    auction_truncation_rate: float
    average_total_reward: float
    average_rent_potential_trend: float
    average_monopoly_denial_events: float
    average_board_strength_trend: float
    average_auction_bid_quality: float
    truncated_bootstrap_rate: float
    rollout_seconds: float
    update_seconds: float
    rollout_examples_per_second: float
    rollout_steps_per_second: float
    league_pool_size: int
    league_source_weights: dict[str, float]
    benchmark_current_win_rate: float
    benchmark_current_elo: float
    win_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class TrainingProgressUpdate:
    iteration_index: int
    total_iterations: int
    phase: str
    completed: int
    total: int
    message: str = ""


class ParallelSelfPlayTrainer:
    def __init__(
        self,
        policy_model: TorchPolicyModel,
        training_config: TrainingConfig | None = None,
        policy_config: PolicyConfig | None = None,
        reward_weights: RewardWeights | None = None,
        heuristic_weights: HeuristicWeights | None = None,
    ) -> None:
        self.policy_model = policy_model
        self.training_config = training_config or TrainingConfig()
        self.policy_config = policy_config or PolicyConfig()
        self.reward_weights = reward_weights or RewardWeights()
        self.heuristic_weights = heuristic_weights or HeuristicWeights()
        self._optimizer = torch.optim.Adam(self.policy_model.network.parameters(), lr=self.policy_config.learning_rate)
        self._worker_pool: PersistentRolloutWorkerPool | None = None
        self._league_manager = LeagueManager(self.training_config, self.policy_config) if self.training_config.use_league_self_play else None
        self._last_rollout_plan: LeagueRolloutPlan | None = None
        self.completed_iterations = 0

    def train(
        self,
        iteration_count: int,
        progress_callback: Callable[[TrainingIterationStats], None] | None = None,
        status_callback: Callable[[TrainingProgressUpdate], None] | None = None,
        show_update_progress: bool = False,
    ) -> list[TrainingIterationStats]:
        stats: list[TrainingIterationStats] = []
        start_iteration = self.completed_iterations
        total_iterations = start_iteration + iteration_count
        for iteration_offset in range(iteration_count):
            iteration_index = start_iteration + iteration_offset
            self._apply_learning_rate_schedule(iteration_index, total_iterations)
            self._emit_status(
                status_callback,
                iteration_index=iteration_index,
                total_iterations=total_iterations,
                phase="rollout",
                completed=0,
                total=max(1, self.training_config.worker_count),
                message="Collecting self-play episodes",
            )
            rollout_start = time.perf_counter()
            worker_results = self._collect_worker_results(
                iteration_index,
                total_iterations=total_iterations,
                status_callback=status_callback,
            )
            rollout_seconds = time.perf_counter() - rollout_start
            examples = [example for result in worker_results for episode in result.episodes for example in episode.training_examples]
            self._emit_status(
                status_callback,
                iteration_index=iteration_index,
                total_iterations=total_iterations,
                phase="update",
                completed=0,
                total=max(1, self._estimate_update_steps(len(examples))),
                message="Updating policy",
            )
            update_start = time.perf_counter()
            self._update_policy(
                list(examples),
                show_progress=show_update_progress,
                iteration_index=iteration_index,
                total_iterations=total_iterations,
                status_callback=status_callback,
            )
            update_seconds = time.perf_counter() - update_start
            self._emit_status(
                status_callback,
                iteration_index=iteration_index,
                total_iterations=total_iterations,
                phase="benchmark",
                completed=0,
                total=1,
                message="Running benchmark",
            )
            benchmark_summary = self._run_benchmark_suite(iteration_index + 1)
            self._emit_status(
                status_callback,
                iteration_index=iteration_index,
                total_iterations=total_iterations,
                phase="benchmark",
                completed=1,
                total=1,
                message="Benchmark complete",
            )
            iteration_stats = self._build_iteration_stats(
                iteration_index,
                worker_results,
                list(examples),
                rollout_seconds=rollout_seconds,
                update_seconds=update_seconds,
                benchmark_summary=benchmark_summary,
            )
            stats.append(iteration_stats)
            self.completed_iterations = iteration_index + 1
            if self._league_manager is not None:
                self._league_manager.record_snapshot(self.policy_model.to_state(), self.completed_iterations)
            if progress_callback is not None:
                progress_callback(iteration_stats)
            if self.training_config.checkpoint_interval > 0 and self.completed_iterations % self.training_config.checkpoint_interval == 0:
                self._emit_status(
                    status_callback,
                    iteration_index=iteration_index,
                    total_iterations=total_iterations,
                    phase="checkpoint",
                    completed=0,
                    total=1,
                    message="Saving checkpoint",
                )
                checkpoint_path = Path(self.training_config.output_directory) / f"iteration_{self.completed_iterations:04d}.pt"
                self.save_checkpoint(checkpoint_path)
                self._emit_status(
                    status_callback,
                    iteration_index=iteration_index,
                    total_iterations=total_iterations,
                    phase="checkpoint",
                    completed=1,
                    total=1,
                    message="Checkpoint saved",
                )
        return stats

    def save_checkpoint(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = asdict(
            TrainerCheckpoint(
                schema_version=CURRENT_CHECKPOINT_SCHEMA_VERSION,
                policy_state=self.policy_model.to_state(),
                policy_config=asdict(self.policy_config),
                reward_weights=asdict(self.reward_weights),
                heuristic_weights=asdict(self.heuristic_weights),
                training_config=asdict(self.training_config),
                completed_iterations=self.completed_iterations,
                league_state=None if self._league_manager is None else self._league_manager.to_state(),
            )
        )
        checkpoint["optimizer_state"] = self._optimizer.state_dict()
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, file_path: str | Path, device_override: str | None = None) -> ParallelSelfPlayTrainer:
        payload = torch.load(Path(file_path), map_location="cpu")
        schema_version = int(payload.get("schema_version", 1))
        if schema_version != CURRENT_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint schema version {schema_version} does not match the current trainer schema version {CURRENT_CHECKPOINT_SCHEMA_VERSION}. Start a fresh training run with the new schema."
            )
        policy_model = TorchPolicyModel.from_state(payload["policy_state"], device_override=device_override)
        policy_config = PolicyConfig(**payload["policy_config"])
        if device_override is not None:
            policy_config.device = device_override
        trainer = cls(
            policy_model=policy_model,
            training_config=TrainingConfig(**payload["training_config"]),
            policy_config=policy_config,
            reward_weights=RewardWeights(**payload["reward_weights"]),
            heuristic_weights=HeuristicWeights(**payload["heuristic_weights"]),
        )
        trainer.completed_iterations = cls._resolve_completed_iterations(payload, Path(file_path))
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is not None:
            trainer._optimizer.load_state_dict(optimizer_state)
        league_state = payload.get("league_state")
        if trainer.training_config.use_league_self_play:
            trainer._league_manager = LeagueManager.from_state(
                league_state or {},
                training_config=trainer.training_config,
                policy_config=trainer.policy_config,
            )
        return trainer

    def close(self) -> None:
        if self._worker_pool is not None:
            self._worker_pool.close()
            self._worker_pool = None

    def refresh_league_manager(self) -> None:
        if not self.training_config.use_league_self_play:
            self._league_manager = None
            return
        existing_state = None if self._league_manager is None else self._league_manager.to_state()
        self._league_manager = LeagueManager.from_state(
            existing_state or {},
            training_config=self.training_config,
            policy_config=self.policy_config,
        )

    def _collect_worker_results(
        self,
        iteration_index: int,
        *,
        total_iterations: int | None = None,
        status_callback: Callable[[TrainingProgressUpdate], None] | None = None,
    ) -> list[RolloutWorkerResult]:
        worker_pool = self._get_or_create_worker_pool()
        policy_state = self.policy_model.to_state()
        rollout_plan = None if self._league_manager is None else self._league_manager.build_rollout_plan(policy_state, iteration_index)
        self._last_rollout_plan = rollout_plan
        return worker_pool.collect(
            policy_state,
            iteration_index,
            heuristic_scale=self.current_heuristic_scale(iteration_index),
            use_heuristic_bias=self.policy_config.use_heuristic_bias,
            league_policy_specs=None if rollout_plan is None else rollout_plan.policy_specs,
            league_assignments_by_worker=None if rollout_plan is None else rollout_plan.assignments_by_worker,
            progress_callback=None if status_callback is None else lambda completed, total: self._emit_status(
                status_callback,
                iteration_index=iteration_index,
                total_iterations=iteration_index + 1 if total_iterations is None else total_iterations,
                phase="rollout",
                completed=completed,
                total=total,
                message=f"Collected {completed}/{total} worker batches",
            ),
        )

    def _get_or_create_worker_pool(self) -> PersistentRolloutWorkerPool:
        if self._worker_pool is None or self._worker_pool.worker_count != self.training_config.worker_count:
            self.close()
            self._worker_pool = PersistentRolloutWorkerPool(
                initial_policy_state=self.policy_model.to_state(),
                training_config=self.training_config,
                policy_config=self.policy_config,
                reward_weights=self.reward_weights,
                heuristic_weights=self.heuristic_weights,
            )
        return self._worker_pool

    def _update_policy(
        self,
        examples: list[TrainingExample],
        *,
        show_progress: bool = False,
        iteration_index: int = 0,
        total_iterations: int = 0,
        status_callback: Callable[[TrainingProgressUpdate], None] | None = None,
    ) -> None:
        if not examples:
            self._emit_status(
                status_callback,
                iteration_index=iteration_index,
                total_iterations=total_iterations,
                phase="update",
                completed=1,
                total=1,
                message="No training examples this iteration",
            )
            return

        batch = self.policy_model.batch_tensors(examples)
        batch["advantages"] = self._normalize_advantages(batch["advantages"])
        batch_size = batch["observations"].shape[0]
        minibatch_size = max(1, min(self.policy_config.minibatch_size, batch_size))
        minibatches_per_epoch = (batch_size + minibatch_size - 1) // minibatch_size
        total_update_steps = self.policy_config.ppo_epochs * minibatches_per_epoch

        progress_bar = None
        if show_progress:
            progress_bar = tqdm(
                total=total_update_steps,
                desc=f"PPO {iteration_index + 1}",
                unit="mb",
                leave=False,
                position=1,
            )

        try:
            completed_update_steps = 0
            for _ in range(self.policy_config.ppo_epochs):
                for minibatch in self.policy_model.iter_minibatches(batch, minibatch_size=minibatch_size, shuffle=True):
                    log_probabilities, entropy, values, threat_predictions = self.policy_model.evaluate_batch(
                        minibatch,
                        use_heuristic_bias=self.policy_config.use_heuristic_bias,
                    )
                    ratios = torch.exp(log_probabilities - minibatch["old_log_probabilities"])
                    unclipped = ratios * minibatch["advantages"]
                    clipped = torch.clamp(ratios, 1.0 - self.policy_config.ppo_clip_ratio, 1.0 + self.policy_config.ppo_clip_ratio) * minibatch["advantages"]
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = F.mse_loss(values, minibatch["returns"])
                    threat_loss = F.mse_loss(threat_predictions, minibatch["threat_targets"])
                    entropy_bonus = entropy.mean()
                    loss = (
                        policy_loss
                        + self.policy_config.value_loss_weight * value_loss
                        + self.policy_config.threat_loss_weight * threat_loss
                        - self.policy_config.entropy_weight * entropy_bonus
                    )

                    self._optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_model.network.parameters(), self.policy_config.gradient_clip)
                    self._optimizer.step()
                    completed_update_steps += 1
                    self._emit_status(
                        status_callback,
                        iteration_index=iteration_index,
                        total_iterations=total_iterations,
                        phase="update",
                        completed=completed_update_steps,
                        total=total_update_steps,
                        message="Updating policy network",
                    )
                    if progress_bar is not None:
                        progress_bar.update(1)
                        progress_bar.set_postfix(loss=f"{loss.item():.4f}", value=f"{value_loss.item():.4f}", threat=f"{threat_loss.item():.4f}")
                if self.policy_model.device.type == "cuda":
                    torch.cuda.synchronize(self.policy_model.device)
        finally:
            if progress_bar is not None:
                progress_bar.close()

    def _build_iteration_stats(
        self,
        iteration_index: int,
        worker_results: list[RolloutWorkerResult],
        examples: list[TrainingExample],
        *,
        rollout_seconds: float,
        update_seconds: float,
        benchmark_summary: BenchmarkSuiteSummary | None,
    ) -> TrainingIterationStats:
        all_episodes = [episode for result in worker_results for episode in result.episodes]
        win_counts: dict[str, int] = {}
        for episode in all_episodes:
            if episode.winner_name is None:
                win_counts.setdefault("draw", 0)
                win_counts["draw"] += 1
            else:
                win_counts.setdefault(episode.winner_name, 0)
                win_counts[episode.winner_name] += 1
        average_steps = 0.0 if not all_episodes else sum(episode.step_count for episode in all_episodes) / float(len(all_episodes))
        average_macro_steps = 0.0 if not all_episodes else sum(episode.macro_step_count for episode in all_episodes) / float(len(all_episodes))
        average_raw_actions = 0.0 if not all_episodes else sum(episode.raw_action_count for episode in all_episodes) / float(len(all_episodes))
        total_auctions = sum(episode.auction_count for episode in all_episodes)
        truncated_auctions = sum(episode.truncated_auction_count for episode in all_episodes)
        average_total_reward = 0.0 if not all_episodes else sum(episode.total_reward for episode in all_episodes) / float(len(all_episodes))
        average_rent_potential_trend = 0.0 if not all_episodes else sum(episode.rent_potential_trend for episode in all_episodes) / float(len(all_episodes))
        average_monopoly_denial_events = 0.0 if not all_episodes else sum(episode.monopoly_denial_events for episode in all_episodes) / float(len(all_episodes))
        average_board_strength_trend = 0.0 if not all_episodes else sum(episode.board_strength_trend for episode in all_episodes) / float(len(all_episodes))
        total_bid_quality = sum(episode.auction_bid_quality_sum for episode in all_episodes)
        total_bid_quality_count = sum(episode.auction_bid_quality_count for episode in all_episodes)
        truncated_episode_count = sum(1 for episode in all_episodes if episode.truncated_episode)
        used_truncated_bootstrap_count = sum(1 for episode in all_episodes if episode.used_truncated_bootstrap)
        total_steps = sum(episode.step_count for episode in all_episodes)
        current_benchmark_participant = None if benchmark_summary is None else next(
            (participant for participant in benchmark_summary.participants if participant.label == "current"),
            None,
        )
        return TrainingIterationStats(
            iteration_index=iteration_index,
            example_count=len(examples),
            episode_count=len(all_episodes),
            average_steps=average_steps,
            average_macro_steps=average_macro_steps,
            average_raw_actions=average_raw_actions,
            auction_truncation_rate=0.0 if total_auctions <= 0 else truncated_auctions / float(total_auctions),
            average_total_reward=average_total_reward,
            average_rent_potential_trend=average_rent_potential_trend,
            average_monopoly_denial_events=average_monopoly_denial_events,
            average_board_strength_trend=average_board_strength_trend,
            average_auction_bid_quality=0.0 if total_bid_quality_count <= 0 else total_bid_quality / float(total_bid_quality_count),
            truncated_bootstrap_rate=0.0 if truncated_episode_count <= 0 else used_truncated_bootstrap_count / float(truncated_episode_count),
            rollout_seconds=rollout_seconds,
            update_seconds=update_seconds,
            rollout_examples_per_second=0.0 if rollout_seconds <= 0.0 else len(examples) / rollout_seconds,
            rollout_steps_per_second=0.0 if rollout_seconds <= 0.0 else total_steps / rollout_seconds,
            league_pool_size=0 if self._league_manager is None else self._league_manager.opponent_pool_size(),
            league_source_weights={} if self._last_rollout_plan is None else dict(self._last_rollout_plan.source_weights),
            benchmark_current_win_rate=0.0 if current_benchmark_participant is None else current_benchmark_participant.win_rate,
            benchmark_current_elo=0.0 if current_benchmark_participant is None else current_benchmark_participant.elo_rating,
            win_counts=win_counts,
        )

    def _run_benchmark_suite(self, completed_iteration_count: int) -> BenchmarkSuiteSummary | None:
        if self._league_manager is None:
            return None
        interval = max(0, int(self.training_config.benchmark_interval))
        if interval <= 0 or completed_iteration_count <= 0 or completed_iteration_count % interval != 0:
            return None
        policy_specs = list(self._league_manager.benchmark_specs(self.policy_model.to_state(), completed_iteration_count))
        if len(policy_specs) < 2:
            return None
        seeds = [
            int(self.training_config.benchmark_seed) + index * int(self.training_config.benchmark_seed_step)
            for index in range(max(1, int(self.training_config.benchmark_games)))
        ]
        from monopoly.agent.evaluation import CheckpointEvaluator

        evaluator = CheckpointEvaluator(device="cpu")
        summary = evaluator.run_policy_benchmark(
            policy_specs,
            seeds=seeds,
            players_per_game=max(2, int(self.training_config.benchmark_players_per_game)),
            max_steps=max(1, int(self.training_config.benchmark_max_steps)),
        )
        current_participant = next((participant for participant in summary.participants if participant.label == "current"), None)
        if current_participant is not None:
            self._league_manager.maybe_promote_best(
                self.policy_model.to_state(),
                completed_iteration_count,
                current_participant.elo_rating,
            )
        return summary

    def _normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        normalized = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        advantage_clip = self.policy_config.advantage_clip
        if advantage_clip > 0.0:
            normalized = torch.clamp(normalized, -advantage_clip, advantage_clip)
        return normalized

    def _apply_learning_rate_schedule(self, iteration_index: int, iteration_count: int) -> None:
        schedule = self.policy_config.learning_rate_schedule
        if schedule == "none":
            learning_rate = self.policy_config.learning_rate
        elif schedule == "linear":
            if iteration_count <= 1:
                learning_rate = self.policy_config.learning_rate
            else:
                progress = iteration_index / float(iteration_count - 1)
                base_learning_rate = self.policy_config.learning_rate
                minimum_learning_rate = min(base_learning_rate, self.policy_config.minimum_learning_rate)
                learning_rate = base_learning_rate - (base_learning_rate - minimum_learning_rate) * progress
        else:
            raise ValueError(f"Unsupported learning-rate schedule: {schedule}")

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = learning_rate

    def _estimate_update_steps(self, example_count: int) -> int:
        if example_count <= 0:
            return 1
        minibatch_size = max(1, min(self.policy_config.minibatch_size, example_count))
        minibatches_per_epoch = (example_count + minibatch_size - 1) // minibatch_size
        return max(1, self.policy_config.ppo_epochs * minibatches_per_epoch)

    @staticmethod
    def _emit_status(
        status_callback: Callable[[TrainingProgressUpdate], None] | None,
        *,
        iteration_index: int,
        total_iterations: int,
        phase: str,
        completed: int,
        total: int,
        message: str,
    ) -> None:
        if status_callback is None:
            return
        status_callback(
            TrainingProgressUpdate(
                iteration_index=iteration_index,
                total_iterations=total_iterations,
                phase=phase,
                completed=completed,
                total=max(1, total),
                message=message,
            )
        )

    def current_heuristic_scale(self, iteration_index: int | None = None) -> float:
        if not self.policy_config.use_heuristic_bias:
            return 0.0
        iteration_value = self.completed_iterations if iteration_index is None else max(0, int(iteration_index))
        schedule = self.policy_config.heuristic_anneal_schedule
        start = float(self.policy_config.heuristic_bias_start)
        end = float(self.policy_config.heuristic_bias_end)
        anneal_iterations = max(0, int(self.policy_config.heuristic_anneal_iterations))
        if schedule == "none":
            return max(0.0, start)
        if schedule == "linear":
            if anneal_iterations <= 0:
                return max(0.0, end)
            progress = min(1.0, iteration_value / float(anneal_iterations))
            return max(0.0, start + (end - start) * progress)
        raise ValueError(f"Unsupported heuristic anneal schedule: {schedule}")

    @staticmethod
    def _resolve_completed_iterations(payload: dict[str, Any], checkpoint_path: Path) -> int:
        completed_iterations = payload.get("completed_iterations")
        if isinstance(completed_iterations, int) and completed_iterations >= 0:
            if completed_iterations == 0 and checkpoint_path.name == "latest.pt":
                inferred_from_directory = ParallelSelfPlayTrainer._infer_iteration_from_directory(checkpoint_path)
                if inferred_from_directory is not None:
                    return inferred_from_directory
            return completed_iterations

        inferred_from_name = ParallelSelfPlayTrainer._infer_iteration_from_path(checkpoint_path)
        if inferred_from_name is not None:
            return inferred_from_name

        inferred_from_directory = ParallelSelfPlayTrainer._infer_iteration_from_directory(checkpoint_path)
        if inferred_from_directory is not None:
            return inferred_from_directory

        return 0

    @staticmethod
    def _infer_iteration_from_path(checkpoint_path: Path) -> int | None:
        match = re.search(r"iteration_(\d+)\.pt$", checkpoint_path.name)
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _infer_iteration_from_directory(checkpoint_path: Path) -> int | None:
        max_iteration: int | None = None
        for candidate in checkpoint_path.parent.glob("iteration_*.pt"):
            match = re.search(r"iteration_(\d+)\.pt$", candidate.name)
            if match is None:
                continue
            candidate_iteration = int(match.group(1))
            if max_iteration is None or candidate_iteration > max_iteration:
                max_iteration = candidate_iteration
        return max_iteration

    def __enter__(self) -> ParallelSelfPlayTrainer:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass