from __future__ import annotations

"""Persistent rollout worker threads for parallel self-play collection."""

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

from monopoly.agent.action_space import MonopolyActionSpace
from monopoly.agent.config import HeuristicWeights, PolicyConfig, RewardWeights, TrainingConfig
from monopoly.agent.controller import AgentPolicyController
from monopoly.agent.environment import MonopolySelfPlayEnvironment, SelfPlayEpisode
from monopoly.agent.features import ObservationEncoder
from monopoly.agent.heuristics import HeuristicScorer
from monopoly.agent.league import LeagueEpisodeAssignment, LeaguePolicySpec
from monopoly.agent.model import TorchPolicyModel
from monopoly.agent.reward import RewardFunction
from monopoly.agent.scripted import ScriptedPolicyController, build_scripted_controller


@dataclass(frozen=True, slots=True)
class RolloutWorkerResult:
    """Completed rollout batch returned from one worker thread."""

    worker_index: int
    episodes: tuple[SelfPlayEpisode, ...]


@dataclass(frozen=True, slots=True)
class _CollectRequest:
    """Work item sent to one rollout worker for a training iteration."""

    iteration_index: int
    policy_state: dict[str, Any]
    heuristic_scale: float
    use_heuristic_bias: bool
    league_policy_specs: tuple[LeaguePolicySpec, ...] | None = None
    league_assignments: tuple[LeagueEpisodeAssignment, ...] | None = None


@dataclass(frozen=True, slots=True)
class _CollectResponse:
    """Worker response carrying either collected episodes or a raised error."""

    worker_index: int
    episodes: tuple[SelfPlayEpisode, ...] = ()
    error: BaseException | None = None


_SHUTDOWN = object()


class PersistentRolloutWorkerPool:
    """Keep rollout workers alive across iterations to amortize environment startup cost."""

    def __init__(
        self,
        *,
        initial_policy_state: dict[str, Any],
        training_config: TrainingConfig,
        policy_config: PolicyConfig,
        reward_weights: RewardWeights,
        heuristic_weights: HeuristicWeights,
    ) -> None:
        self._training_config = training_config
        self._policy_config = policy_config
        self._reward_weights = reward_weights
        self._heuristic_weights = heuristic_weights
        self._response_queue: queue.Queue[_CollectResponse] = queue.Queue()
        self._request_queues: list[queue.Queue[object]] = []
        self._threads: list[threading.Thread] = []
        self._closed = False

        for worker_index in range(self._training_config.worker_count):
            request_queue: queue.Queue[object] = queue.Queue(maxsize=1)
            thread = threading.Thread(
                target=self._run_worker,
                name=f"rollout-worker-{worker_index}",
                args=(worker_index, initial_policy_state, request_queue),
                daemon=True,
            )
            thread.start()
            self._request_queues.append(request_queue)
            self._threads.append(thread)

    @property
    def worker_count(self) -> int:
        """Return the number of active worker threads managed by this pool."""
        return len(self._threads)

    def collect(
        self,
        policy_state: dict[str, Any],
        iteration_index: int,
        *,
        heuristic_scale: float,
        use_heuristic_bias: bool,
        league_policy_specs: tuple[LeaguePolicySpec, ...] | None = None,
        league_assignments_by_worker: tuple[tuple[LeagueEpisodeAssignment, ...], ...] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[RolloutWorkerResult]:
        """Dispatch one rollout request to each worker and gather ordered results."""
        if self._closed:
            raise RuntimeError("Cannot collect rollouts from a closed worker pool.")

        for worker_index, request_queue in enumerate(self._request_queues):
            request_queue.put(
                _CollectRequest(
                    iteration_index=iteration_index,
                    policy_state=policy_state,
                    heuristic_scale=heuristic_scale,
                    use_heuristic_bias=use_heuristic_bias,
                    league_policy_specs=league_policy_specs,
                    league_assignments=None if league_assignments_by_worker is None else league_assignments_by_worker[worker_index],
                )
            )

        results_by_index: dict[int, RolloutWorkerResult] = {}
        completed_workers = 0
        for _ in range(self.worker_count):
            response = self._response_queue.get()
            if response.error is not None:
                self.close()
                raise RuntimeError(f"Rollout worker {response.worker_index} failed.") from response.error
            results_by_index[response.worker_index] = RolloutWorkerResult(
                worker_index=response.worker_index,
                episodes=response.episodes,
            )
            completed_workers += 1
            if progress_callback is not None:
                progress_callback(completed_workers, self.worker_count)
        return [results_by_index[index] for index in range(self.worker_count)]

    def close(self) -> None:
        """Signal worker shutdown and join their threads exactly once."""
        if self._closed:
            return
        self._closed = True
        for request_queue in self._request_queues:
            request_queue.put(_SHUTDOWN)
        for thread in self._threads:
            thread.join(timeout=5.0)

    def _run_worker(self, worker_index: int, initial_policy_state: dict[str, Any], request_queue: queue.Queue[object]) -> None:
        """Worker thread loop that reloads policy state and executes requested rollouts."""
        model = TorchPolicyModel.from_state(initial_policy_state, device_override="cpu")
        controller = AgentPolicyController(
            policy_model=model,
            observation_encoder=ObservationEncoder(),
            action_space=MonopolyActionSpace(),
            heuristic_scorer=HeuristicScorer(self._heuristic_weights),
        )
        environment = MonopolySelfPlayEnvironment(
            training_config=self._training_config,
            reward_function=RewardFunction(self._reward_weights),
            controller=controller,
            discount_gamma=self._policy_config.discount_gamma,
            gae_lambda=self._policy_config.gae_lambda,
            bootstrap_truncated_episodes=self._policy_config.bootstrap_truncated_episodes,
        )

        while True:
            request = request_queue.get()
            if request is _SHUTDOWN:
                return
            assert isinstance(request, _CollectRequest)
            try:
                model.load_state(request.policy_state, load_rng_state=False)
                controller.configure_heuristics(
                    heuristic_scale=request.heuristic_scale,
                    use_heuristic_bias=request.use_heuristic_bias,
                )
                episodes: list[SelfPlayEpisode] = []
                base_seed = self._policy_config.seed + request.iteration_index * 100_000 + worker_index * 1_000
                for episode_index in range(self._training_config.episodes_per_worker):
                    episode_seed = base_seed + episode_index
                    if request.league_policy_specs is None or request.league_assignments is None:
                        model.set_seed(episode_seed)
                        episodes.append(environment.run_episode(seed=episode_seed, explore=True))
                        continue
                    assignment = request.league_assignments[episode_index]
                    controller_by_player: dict[str, AgentPolicyController] = {}
                    specs_by_label = {spec.label: spec for spec in request.league_policy_specs}
                    for seat_index, label in enumerate(assignment.seat_labels):
                        player_name = f"{self._training_config.player_name_prefix}{seat_index + 1}"
                        if label == "current":
                            model.set_seed(episode_seed + seat_index)
                            controller.configure_heuristics(
                                heuristic_scale=request.heuristic_scale,
                                use_heuristic_bias=request.use_heuristic_bias,
                            )
                            controller_by_player[player_name] = controller
                            continue
                        spec = specs_by_label[label]
                        opponent_controller = self._build_opponent_controller(
                            spec,
                            seed=episode_seed + (seat_index + 1) * 97,
                        )
                        controller_by_player[player_name] = opponent_controller
                    episodes.append(
                        environment.run_episode(
                            seed=episode_seed,
                            explore=True,
                            controller_by_player=controller_by_player,
                            learning_player_names=assignment.learning_player_names,
                        )
                    )
                self._response_queue.put(_CollectResponse(worker_index=worker_index, episodes=tuple(episodes)))
            except BaseException as error:
                self._response_queue.put(_CollectResponse(worker_index=worker_index, error=error))

    def _build_opponent_controller(self, spec: LeaguePolicySpec, *, seed: int) -> AgentPolicyController | ScriptedPolicyController:
        """Construct one non-current-seat controller from a league policy specification."""
        if spec.source == "scripted":
            if spec.scripted_variant is None:
                raise ValueError(f"Scripted league spec {spec.label} is missing a scripted variant.")
            controller = build_scripted_controller(
                spec.scripted_variant,
                seed=seed,
                observation_encoder=ObservationEncoder(),
                action_space=MonopolyActionSpace(),
            )
            controller.set_seed(seed)
            return controller
        if spec.policy_state is None:
            raise ValueError(f"League spec {spec.label} is missing policy state for source {spec.source}.")
        opponent_model = TorchPolicyModel.from_state(spec.policy_state, device_override="cpu")
        opponent_model.set_seed(seed)
        opponent_controller = AgentPolicyController(
            policy_model=opponent_model,
            observation_encoder=ObservationEncoder(),
            action_space=MonopolyActionSpace(),
            heuristic_scorer=HeuristicScorer(self._heuristic_weights),
        )
        opponent_controller.configure_heuristics(
            heuristic_scale=spec.heuristic_scale,
            use_heuristic_bias=spec.use_heuristic_bias,
        )
        return opponent_controller