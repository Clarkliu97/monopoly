from __future__ import annotations

"""Agent-side controller adapters for choosing and executing legal game actions."""

from dataclasses import dataclass

import numpy as np

from monopoly.api import FrontendStateView
from monopoly.agent.action_space import AgentActionChoice, MonopolyActionSpace
from monopoly.agent.features import ObservationEncoder
from monopoly.agent.heuristics import HeuristicScorer
from monopoly.agent.model import PolicyDecision, TorchPolicyModel
from monopoly.game import Game


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    """One chosen action plus the tensors needed to train from that decision."""

    choice: AgentActionChoice
    decision: PolicyDecision
    observation: np.ndarray
    action_mask: np.ndarray
    heuristic_bias: np.ndarray


def filter_invalid_trade_choices(
    game: Game,
    action_mask: np.ndarray,
    choices: dict[int, AgentActionChoice],
) -> tuple[np.ndarray, dict[int, AgentActionChoice]]:
    """Drop trade choices whose payloads are invalid, stale, or already blocked."""
    filtered_mask = action_mask.copy()
    filtered_choices = dict(choices)
    for action_id, choice in list(filtered_choices.items()):
        if choice.trade_offer_payload is None:
            continue
        try:
            trade_offer = game.deserialize_trade_offer(choice.trade_offer_payload)
        except Exception:
            filtered_mask[action_id] = False
            filtered_choices.pop(action_id, None)
            continue
        if trade_offer.validate() or game.is_trade_offer_blocked(trade_offer):
            filtered_mask[action_id] = False
            filtered_choices.pop(action_id, None)
    if filtered_choices:
        return filtered_mask, filtered_choices
    return action_mask, choices


class AgentPolicyController:
    """Map a game state into one legal action using observations, masks, and a policy model."""

    def __init__(
        self,
        policy_model: TorchPolicyModel,
        observation_encoder: ObservationEncoder | None = None,
        action_space: MonopolyActionSpace | None = None,
        heuristic_scorer: HeuristicScorer | None = None,
    ) -> None:
        self.policy_model = policy_model
        self.observation_encoder = observation_encoder or ObservationEncoder()
        self.action_space = action_space or MonopolyActionSpace()
        self.heuristic_scorer = heuristic_scorer or HeuristicScorer()
        self.heuristic_scale = 1.0
        self.use_heuristic_bias = True

    def configure_heuristics(self, *, heuristic_scale: float | None = None, use_heuristic_bias: bool | None = None) -> None:
        """Adjust runtime heuristic-bias behavior without rebuilding the controller."""
        if heuristic_scale is not None:
            self.heuristic_scale = max(0.0, float(heuristic_scale))
        if use_heuristic_bias is not None:
            self.use_heuristic_bias = bool(use_heuristic_bias)

    def choose_action(self, game: Game, actor_name: str, explore: bool = True) -> ControllerDecision:
        """Encode the current state, score legal actions, and sample or select one action."""
        frontend_state = game.get_frontend_state()
        turn_plan = game.get_turn_plan(actor_name)
        observation = self.observation_encoder.encode(frontend_state, actor_name)
        action_mask, choices = self.action_space.build_mask(turn_plan, frontend_state)
        action_mask, choices = filter_invalid_trade_choices(game, action_mask, choices)
        heuristic_bias = self.heuristic_scorer.score(
            frontend_state,
            choices,
            actor_name,
            self.action_space.action_count,
            heuristic_scale=self.heuristic_scale if self.use_heuristic_bias else 0.0,
        )
        decision = self.policy_model.act(
            observation,
            action_mask,
            heuristic_bias,
            explore=explore,
            use_heuristic_bias=self.use_heuristic_bias,
        )
        return ControllerDecision(
            choice=choices[decision.action_id],
            decision=decision,
            observation=observation,
            action_mask=action_mask,
            heuristic_bias=heuristic_bias,
        )

    def evaluate_state_values(self, frontend_state: FrontendStateView, actor_names: tuple[str, ...]) -> dict[str, float]:
        """Predict value estimates for one or more named actors from the same public state."""
        observations = np.asarray(
            [self.observation_encoder.encode(frontend_state, actor_name) for actor_name in actor_names],
            dtype=np.float32,
        )
        values = self.policy_model.predict_values(observations)
        return {actor_name: float(value) for actor_name, value in zip(actor_names, values, strict=True)}


class GameProcessAgentHost:
    """Drive AI turns directly against a live `Game` instance until control returns to a human."""

    def __init__(self, controller: AgentPolicyController) -> None:
        self.controller = controller

    def play_ai_actions(self, game: Game, *, explore: bool = True, max_actions: int = 128) -> list[ControllerDecision]:
        """Execute consecutive AI actions until the game ends, a human turn begins, or the cap is hit."""
        decisions: list[ControllerDecision] = []
        for _ in range(max_actions):
            turn_plan = game.get_active_turn_plan()
            if turn_plan.player_role != "ai":
                break
            decision = self.controller.choose_action(game, turn_plan.player_name, explore=explore)
            decisions.append(decision)
            trade_offer = None if decision.choice.trade_offer_payload is None else game.deserialize_trade_offer(decision.choice.trade_offer_payload)
            game.execute_legal_action(
                decision.choice.legal_action,
                bid_amount=decision.choice.bid_amount,
                trade_offer=trade_offer,
            )
            if game.winner() is not None:
                break
        return decisions