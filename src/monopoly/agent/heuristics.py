from __future__ import annotations

"""Rule-of-thumb priors used to bias or benchmark policy decisions."""

import numpy as np

from monopoly.agent.action_space import AgentActionChoice
from monopoly.agent.config import HeuristicWeights
from monopoly.api import FrontendStateView


class HeuristicScorer:
    """Assign lightweight prior scores to action-space choices."""

    def __init__(self, weights: HeuristicWeights | None = None) -> None:
        """Create a scorer with the provided heuristic weights or defaults."""
        self.weights = weights or HeuristicWeights()

    def score(
        self,
        frontend_state: FrontendStateView,
        choices: dict[int, AgentActionChoice],
        actor_name: str,
        action_count: int,
        *,
        heuristic_scale: float = 1.0,
    ) -> np.ndarray:
        """Return per-action heuristic logits aligned with the discrete action space."""
        scores = np.zeros(action_count, dtype=np.float32)
        heuristic_scale = max(0.0, float(heuristic_scale))
        if heuristic_scale <= 0.0:
            return scores
        actor = next(player for player in frontend_state.game_view.players if player.name == actor_name)
        weights = self.weights
        pending_action = frontend_state.game_view.pending_action
        actor_cash = float(actor.cash)
        cash_buffer_target = max(1.0, weights.cash_buffer_target)
        positive_cash_margin = max(0.0, (actor_cash - weights.cash_buffer_target) / 500.0)
        negative_cash_margin = max(0.0, (weights.cash_buffer_target - actor_cash) / 500.0)
        jail_fine_margin = max(0.0, (actor_cash - 50.0) / 500.0)
        buy_price = 0.0 if pending_action is None or pending_action.price is None else float(pending_action.price)
        affordability_margin = (actor_cash - buy_price) / cash_buffer_target
        low_cash_mortgage_bonus = weights.low_cash_mortgage_bonus if actor_cash < weights.cash_buffer_target else 0.0
        confirm_action_type = None if pending_action is None or pending_action.property_action is None else pending_action.property_action.action_type

        for action_id, choice in choices.items():
            label = choice.action_label
            score = 0.0
            if label == "start_turn":
                score = weights.start_turn_bias
            elif label == "end_turn":
                score = weights.end_turn_bias
            elif label == "buy_property":
                score = weights.buy_property_bias + max(-0.5, affordability_margin)
            elif label == "decline_property":
                score = weights.decline_property_bias
            elif label.startswith("build:"):
                score = weights.build_bias + positive_cash_margin
            elif label.startswith("sell:"):
                score = weights.sell_building_bias + negative_cash_margin
            elif label.startswith("mortgage:"):
                score = weights.mortgage_bias + low_cash_mortgage_bonus
            elif label.startswith("unmortgage:"):
                score = weights.unmortgage_bias + positive_cash_margin
            elif label == "pass_auction":
                score = weights.pass_auction_bias + negative_cash_margin
            elif label == "auction_bid_min":
                score = weights.auction_min_bid_bias
            elif label == "auction_bid_value":
                score = weights.auction_value_bid_bias
            elif label == "auction_bid_premium":
                score = weights.auction_premium_bid_bias
            elif label == "auction_bid_aggressive":
                score = weights.auction_aggressive_bid_bias
            elif label == "jail_use_card":
                score = weights.jail_use_card_bias
            elif label == "jail_pay_fine":
                score = weights.jail_pay_fine_bias + jail_fine_margin
            elif label == "jail_roll":
                score = weights.jail_roll_bias
            elif label == "accept_trade":
                score = weights.accept_trade_bias
            elif label == "reject_trade":
                score = weights.reject_trade_bias
            elif label == "confirm_property_action" and confirm_action_type is not None:
                if confirm_action_type == "build":
                    score = weights.build_bias
                elif confirm_action_type == "sell_building":
                    score = weights.sell_building_bias
                elif confirm_action_type == "mortgage":
                    score = weights.mortgage_bias
                elif confirm_action_type == "unmortgage":
                    score = weights.unmortgage_bias
            elif label == "cancel_property_action":
                score = -0.05
            scores[action_id] = score * heuristic_scale
        return scores