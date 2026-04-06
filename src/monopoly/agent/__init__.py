"""Convenience exports for the Monopoly agent package.

Importers use this module as the stable public surface for training, inference,
evaluation, and checkpoint management components.
"""

from monopoly.agent.action_space import AgentActionChoice, MonopolyActionSpace
from monopoly.agent.checkpoints import DEFAULT_CHECKPOINT_DIRECTORY, DEFAULT_CHECKPOINT_FILE_NAME, load_agent_host_from_checkpoint, resolve_checkpoint_path
from monopoly.agent.config import HeuristicWeights, PolicyConfig, RewardWeights, TrainingConfig
from monopoly.agent.controller import AgentPolicyController, GameProcessAgentHost
from monopoly.agent.evaluation import BenchmarkParticipantSummary, BenchmarkSuiteSummary, CheckpointEvaluator, EvaluationGameResult, EvaluationSummary, TournamentSummary
from monopoly.agent.environment import MonopolySelfPlayEnvironment, SelfPlayEpisode
from monopoly.agent.features import ObservationEncoder
from monopoly.agent.heuristics import HeuristicScorer
from monopoly.agent.league import LeagueEpisodeAssignment, LeagueManager, LeaguePolicySpec, LeagueRolloutPlan
from monopoly.agent.model import LightweightPolicyModel, PolicyDecision, TorchPolicyModel, TrainingExample
from monopoly.agent.reward import RewardBreakdown, RewardFunction
from monopoly.agent.scripted import ScriptedPolicyController, ScriptedPolicyProfile, build_scripted_controller, default_scripted_profiles
from monopoly.agent.trainer import ParallelSelfPlayTrainer, TrainerCheckpoint, TrainingIterationStats

__all__ = [
    "AgentActionChoice",
    "AgentPolicyController",
    "BenchmarkParticipantSummary",
    "BenchmarkSuiteSummary",
    "CheckpointEvaluator",
    "DEFAULT_CHECKPOINT_DIRECTORY",
    "DEFAULT_CHECKPOINT_FILE_NAME",
    "EvaluationGameResult",
    "EvaluationSummary",
    "GameProcessAgentHost",
    "HeuristicScorer",
    "HeuristicWeights",
    "LeagueEpisodeAssignment",
    "LeagueManager",
    "LeaguePolicySpec",
    "LeagueRolloutPlan",
    "LightweightPolicyModel",
    "load_agent_host_from_checkpoint",
    "MonopolyActionSpace",
    "MonopolySelfPlayEnvironment",
    "ObservationEncoder",
    "PolicyConfig",
    "PolicyDecision",
    "ParallelSelfPlayTrainer",
    "RewardBreakdown",
    "RewardFunction",
    "RewardWeights",
    "resolve_checkpoint_path",
    "ScriptedPolicyController",
    "ScriptedPolicyProfile",
    "SelfPlayEpisode",
    "build_scripted_controller",
    "default_scripted_profiles",
    "TorchPolicyModel",
    "TrainerCheckpoint",
    "TrainingConfig",
    "TrainingExample",
    "TrainingIterationStats",
    "TournamentSummary",
]