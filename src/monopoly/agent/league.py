from __future__ import annotations

"""League self-play policy selection and assignment helpers."""

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

from monopoly.agent.config import PolicyConfig, TrainingConfig


@dataclass(frozen=True, slots=True)
class LeaguePolicySpec:
    """Serializable description of one opponent policy source in the self-play league."""

    label: str
    source: str
    policy_state: dict[str, Any] | None = None
    scripted_variant: str | None = None
    use_heuristic_bias: bool = False
    heuristic_scale: float = 0.0

    def to_state(self) -> dict[str, Any]:
        """Serialize this policy spec while deep-copying any embedded model state."""
        return {
            "label": self.label,
            "source": self.source,
            "policy_state": None if self.policy_state is None else copy.deepcopy(self.policy_state),
            "scripted_variant": self.scripted_variant,
            "use_heuristic_bias": self.use_heuristic_bias,
            "heuristic_scale": self.heuristic_scale,
        }

    @classmethod
    def from_state(cls, payload: dict[str, Any]) -> LeaguePolicySpec:
        """Restore a policy spec from `to_state()` output."""
        return cls(
            label=str(payload["label"]),
            source=str(payload.get("source", "recent")),
            policy_state=None if payload.get("policy_state") is None else copy.deepcopy(payload["policy_state"]),
            scripted_variant=None if payload.get("scripted_variant") is None else str(payload.get("scripted_variant")),
            use_heuristic_bias=bool(payload.get("use_heuristic_bias", False)),
            heuristic_scale=float(payload.get("heuristic_scale", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class LeagueEpisodeAssignment:
    """One worker episode seating plan for current and opponent policies."""

    seat_labels: tuple[str, ...]
    learning_player_names: tuple[str, ...]

    def to_state(self) -> dict[str, Any]:
        """Serialize the seat assignment to plain data."""
        return {
            "seat_labels": list(self.seat_labels),
            "learning_player_names": list(self.learning_player_names),
        }

    @classmethod
    def from_state(cls, payload: dict[str, Any]) -> LeagueEpisodeAssignment:
        """Restore a seat assignment from serialized data."""
        return cls(
            seat_labels=tuple(str(label) for label in payload.get("seat_labels", ())),
            learning_player_names=tuple(str(name) for name in payload.get("learning_player_names", ())),
        )


@dataclass(frozen=True, slots=True)
class LeagueRolloutPlan:
    """Complete league rollout plan for all workers in one training iteration."""

    policy_specs: tuple[LeaguePolicySpec, ...]
    assignments_by_worker: tuple[tuple[LeagueEpisodeAssignment, ...], ...]
    source_weights: dict[str, float]


class LeagueManager:
    """Manage recent, best, and scripted opponent pools for league self-play."""

    def __init__(self, training_config: TrainingConfig, policy_config: PolicyConfig) -> None:
        self.training_config = training_config
        self.policy_config = policy_config
        self._recent_snapshots: deque[LeaguePolicySpec] = deque(maxlen=max(1, training_config.league_recent_snapshot_count))
        self._best_snapshot: LeaguePolicySpec | None = None
        self._best_benchmark_score: float | None = None

    def to_state(self) -> dict[str, Any]:
        """Serialize the current league snapshot state for checkpointing."""
        return {
            "recent_snapshots": [snapshot.to_state() for snapshot in self._recent_snapshots],
            "best_snapshot": None if self._best_snapshot is None else self._best_snapshot.to_state(),
            "best_benchmark_score": self._best_benchmark_score,
        }

    @classmethod
    def from_state(
        cls,
        payload: dict[str, Any],
        *,
        training_config: TrainingConfig,
        policy_config: PolicyConfig,
    ) -> LeagueManager:
        """Restore a league manager from serialized state and fresh configs."""
        manager = cls(training_config=training_config, policy_config=policy_config)
        for snapshot_payload in payload.get("recent_snapshots", ()): 
            manager._recent_snapshots.append(LeaguePolicySpec.from_state(snapshot_payload))
        best_payload = payload.get("best_snapshot")
        manager._best_snapshot = None if best_payload is None else LeaguePolicySpec.from_state(best_payload)
        best_score = payload.get("best_benchmark_score")
        manager._best_benchmark_score = None if best_score is None else float(best_score)
        return manager

    def record_snapshot(self, current_policy_state: dict[str, Any], iteration_index: int) -> None:
        """Record a periodic current-policy snapshot for future league opponent sampling."""
        if not self.training_config.use_league_self_play:
            return
        interval = max(1, int(self.training_config.league_snapshot_interval))
        if iteration_index <= 0 or iteration_index % interval != 0:
            return
        snapshot = LeaguePolicySpec(
            label=f"iteration_{iteration_index:04d}",
            source="recent",
            policy_state=copy.deepcopy(current_policy_state),
        )
        self._recent_snapshots.append(snapshot)
        if self._best_snapshot is None:
            self._best_snapshot = LeaguePolicySpec(
                label=f"best_{iteration_index:04d}",
                source="best",
                policy_state=copy.deepcopy(current_policy_state),
            )

    def maybe_promote_best(self, current_policy_state: dict[str, Any], iteration_index: int, benchmark_score: float) -> bool:
        """Promote the current policy to best-so-far if its benchmark score improves."""
        score = float(benchmark_score)
        if self._best_benchmark_score is not None and score < self._best_benchmark_score:
            return False
        self._best_benchmark_score = score
        self._best_snapshot = LeaguePolicySpec(
            label=f"best_{iteration_index:04d}",
            source="best",
            policy_state=copy.deepcopy(current_policy_state),
        )
        return True

    def opponent_pool_size(self) -> int:
        """Return the number of distinct opponent labels currently available."""
        labels = {snapshot.label for snapshot in self._recent_snapshots}
        if self._best_snapshot is not None:
            labels.add(self._best_snapshot.label)
        labels.update(spec.label for spec in self._scripted_specs())
        return len(labels)

    def source_weights(self, iteration_index: int) -> dict[str, float]:
        """Return scheduled league source weights for the given training iteration."""
        if iteration_index <= 20:
            return self._interpolate_weights(iteration_index, 0, 20, (0.0, 0.0, 1.0), (0.2, 0.1, 0.7))
        if iteration_index <= 50:
            return self._interpolate_weights(iteration_index, 20, 50, (0.2, 0.1, 0.7), (0.4, 0.1, 0.5))
        if iteration_index <= 150:
            return self._interpolate_weights(iteration_index, 50, 150, (0.4, 0.1, 0.5), (0.25, 0.25, 0.5))
        if iteration_index <= 200:
            return self._interpolate_weights(iteration_index, 150, 200, (0.25, 0.25, 0.5), (0.4, 0.4, 0.2))
        return {"best": 0.4, "recent": 0.4, "scripted": 0.2}

    def build_rollout_plan(self, current_policy_state: dict[str, Any], iteration_index: int) -> LeagueRolloutPlan | None:
        """Build per-worker opponent assignments for one league self-play rollout batch."""
        if not self.training_config.use_league_self_play or self.training_config.players_per_game < 2:
            return None
        current_spec = LeaguePolicySpec(label="current", source="current", policy_state=copy.deepcopy(current_policy_state))
        source_pools = self._source_pools(current_policy_state)
        effective_weights = self._resolve_available_weights(self.source_weights(iteration_index), source_pools)
        if not effective_weights:
            return None
        all_specs = self._dedupe_specs((current_spec, *[spec for specs in source_pools.values() for spec in specs]))
        assignments_by_worker: list[tuple[LeagueEpisodeAssignment, ...]] = []
        players_per_game = max(2, int(self.training_config.players_per_game))
        for worker_index in range(max(1, int(self.training_config.worker_count))):
            worker_assignments: list[LeagueEpisodeAssignment] = []
            for episode_index in range(max(1, int(self.training_config.episodes_per_worker))):
                rng = random.Random(self.policy_config.seed + iteration_index * 100_000 + worker_index * 1_000 + episode_index)
                learning_seat = rng.randrange(players_per_game)
                seat_labels: list[str] = []
                for seat_index in range(players_per_game):
                    if seat_index == learning_seat:
                        seat_labels.append(current_spec.label)
                        continue
                    source_name = self._sample_source(rng, effective_weights)
                    source_specs = source_pools[source_name]
                    seat_labels.append(source_specs[rng.randrange(len(source_specs))].label)
                worker_assignments.append(
                    LeagueEpisodeAssignment(
                        seat_labels=tuple(seat_labels),
                        learning_player_names=(f"{self.training_config.player_name_prefix}{learning_seat + 1}",),
                    )
                )
            assignments_by_worker.append(tuple(worker_assignments))
        return LeagueRolloutPlan(
            policy_specs=all_specs,
            assignments_by_worker=tuple(assignments_by_worker),
            source_weights=effective_weights,
        )

    def benchmark_specs(self, current_policy_state: dict[str, Any], iteration_index: int) -> tuple[LeaguePolicySpec, ...]:
        """Return the current benchmark candidate set for evaluation and tournaments."""
        specs: list[LeaguePolicySpec] = [
            LeaguePolicySpec(label="current", source="current", policy_state=copy.deepcopy(current_policy_state)),
        ]
        if self._best_snapshot is not None:
            specs.append(self._best_snapshot)
        if self._recent_snapshots:
            specs.append(self._recent_snapshots[-1])
        specs.extend(self._scripted_specs())
        return self._dedupe_specs(tuple(specs))

    @staticmethod
    def _interpolate_weights(
        iteration_index: int,
        start_iteration: int,
        end_iteration: int,
        start_weights: tuple[float, float, float],
        end_weights: tuple[float, float, float],
    ) -> dict[str, float]:
        """Linearly interpolate scheduled source weights between two iteration anchors."""
        if end_iteration <= start_iteration:
            progress = 1.0
        else:
            progress = max(0.0, min(1.0, (iteration_index - start_iteration) / float(end_iteration - start_iteration)))
        best_weight = start_weights[0] + (end_weights[0] - start_weights[0]) * progress
        recent_weight = start_weights[1] + (end_weights[1] - start_weights[1]) * progress
        scripted_weight = start_weights[2] + (end_weights[2] - start_weights[2]) * progress
        return {"best": best_weight, "recent": recent_weight, "scripted": scripted_weight}

    def _scripted_specs(self) -> tuple[LeaguePolicySpec, ...]:
        """Return league policy specs for configured scripted opponents."""
        if not self.training_config.league_use_scripted_opponents:
            return ()
        return tuple(
            LeaguePolicySpec(
                label=f"scripted_{variant_name}",
                source="scripted",
                scripted_variant=variant_name,
            )
            for variant_name in self.training_config.league_scripted_variants
        )

    def _source_pools(self, current_policy_state: dict[str, Any]) -> dict[str, tuple[LeaguePolicySpec, ...]]:
        """Return currently available opponent pools grouped by source type."""
        del current_policy_state
        pools: dict[str, tuple[LeaguePolicySpec, ...]] = {}
        if self._best_snapshot is not None:
            pools["best"] = (self._best_snapshot,)
        if self._recent_snapshots:
            pools["recent"] = tuple(self._recent_snapshots)
        scripted_specs = self._scripted_specs()
        if scripted_specs:
            pools["scripted"] = scripted_specs
        return pools

    @staticmethod
    def _resolve_available_weights(
        requested_weights: dict[str, float],
        source_pools: dict[str, tuple[LeaguePolicySpec, ...]],
    ) -> dict[str, float]:
        """Normalize source weights over only the pools that are currently populated."""
        available_weights = {
            source_name: max(0.0, requested_weights.get(source_name, 0.0))
            for source_name, specs in source_pools.items()
            if specs
        }
        total = sum(available_weights.values())
        if total <= 0.0:
            count = len(available_weights)
            return {} if count <= 0 else {source_name: 1.0 / float(count) for source_name in available_weights}
        return {source_name: weight / total for source_name, weight in available_weights.items()}

    @staticmethod
    def _sample_source(rng: random.Random, weights: dict[str, float]) -> str:
        """Sample one source name according to the already-normalized source weights."""
        sample = rng.random()
        cumulative = 0.0
        source_names = list(weights.keys())
        for source_name in source_names:
            cumulative += weights[source_name]
            if sample <= cumulative:
                return source_name
        return source_names[-1]

    @staticmethod
    def _dedupe_specs(specs: tuple[LeaguePolicySpec, ...]) -> tuple[LeaguePolicySpec, ...]:
        """Preserve policy-spec order while removing duplicate labels."""
        deduped: list[LeaguePolicySpec] = []
        seen_labels: set[str] = set()
        for spec in specs:
            if spec.label in seen_labels:
                continue
            deduped.append(spec)
            seen_labels.add(spec.label)
        return tuple(deduped)
