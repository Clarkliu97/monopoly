from __future__ import annotations

"""Checkpoint loading helpers for agent inference and evaluation flows."""

from pathlib import Path

from monopoly.agent.controller import AgentPolicyController, GameProcessAgentHost
from monopoly.agent.action_space import MonopolyActionSpace
from monopoly.agent.features import ObservationEncoder
from monopoly.agent.heuristics import HeuristicScorer
from monopoly.agent.trainer import ParallelSelfPlayTrainer


DEFAULT_CHECKPOINT_DIRECTORY = ".checkpoints"
DEFAULT_CHECKPOINT_FILE_NAME = "latest.pt"


def default_checkpoint_path(base_directory: str | Path | None = None) -> Path:
    """Return the conventional latest-checkpoint path under the chosen root directory."""
    root = Path.cwd() if base_directory is None else Path(base_directory)
    return root / DEFAULT_CHECKPOINT_DIRECTORY / DEFAULT_CHECKPOINT_FILE_NAME


def resolve_checkpoint_path(requested_path: str | Path | None = None, *, require_exists: bool = False) -> Path | None:
    """Resolve an explicit or default checkpoint path, optionally requiring that it exists."""
    candidates: list[Path] = []
    if requested_path is not None:
        candidates.append(Path(requested_path))
    candidates.append(default_checkpoint_path())

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved

    if require_exists:
        missing_target = Path(requested_path) if requested_path is not None else default_checkpoint_path()
        raise FileNotFoundError(
            f"AI checkpoint was not found. Looked for {missing_target}. Train a model first or pass an explicit checkpoint path."
        )
    return None


def load_agent_controller_from_checkpoint(checkpoint_path: str | Path, *, device: str = "cpu") -> AgentPolicyController:
    """Load an agent controller from a saved trainer checkpoint after schema checks."""
    trainer = ParallelSelfPlayTrainer.load_checkpoint(checkpoint_path, device_override=device)
    encoder = ObservationEncoder()
    action_space = MonopolyActionSpace()
    if trainer.policy_model.observation_size != encoder.observation_size:
        raise ValueError(
            f"Checkpoint observation size {trainer.policy_model.observation_size} does not match the current encoder size {encoder.observation_size}. Start a fresh training run with the new schema."
        )
    if trainer.policy_model.action_count != action_space.action_count:
        raise ValueError(
            f"Checkpoint action count {trainer.policy_model.action_count} does not match the current action space size {action_space.action_count}. Start a fresh training run with the new schema."
        )
    controller = AgentPolicyController(
        policy_model=trainer.policy_model,
        observation_encoder=encoder,
        action_space=action_space,
        heuristic_scorer=HeuristicScorer(trainer.heuristic_weights),
    )
    controller.configure_heuristics(
        heuristic_scale=trainer.current_heuristic_scale(),
        use_heuristic_bias=trainer.policy_config.use_heuristic_bias,
    )
    return controller


def load_agent_host_from_checkpoint(checkpoint_path: str | Path, *, device: str = "cpu") -> GameProcessAgentHost:
    """Load a checkpoint and wrap its controller in a live game host adapter."""
    return GameProcessAgentHost(load_agent_controller_from_checkpoint(checkpoint_path, device=device))