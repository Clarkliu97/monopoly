from __future__ import annotations

"""Hand-authored baseline controllers for Monopoly agents.

These scripted policies provide deterministic or lightly stochastic baselines for
training, benchmarking, and fallback AI seats. They score legal actions using
public board analysis instead of a learned model.
"""

from dataclasses import dataclass

import numpy as np

from monopoly.agent.action_space import AgentActionChoice, MonopolyActionSpace
from monopoly.agent.board_analysis import COLOR_GROUP_SEQUENCE, analyze_board, strongest_opponent_name
from monopoly.agent.controller import ControllerDecision, filter_invalid_trade_choices
from monopoly.agent.features import ObservationEncoder
from monopoly.agent.model import PolicyDecision
from monopoly.api import BoardSpaceView, FrontendStateView, PendingActionView, PlayerView, TradeDecisionView, TurnPlanView
from monopoly.game import Game


@dataclass(frozen=True, slots=True)
class ScriptedPolicyProfile:
    """Parameter bundle controlling one scripted opponent personality."""

    name: str
    cash_buffer_target: float
    property_pressure_weight: float
    expansion_weight: float
    build_weight: float
    liquidity_weight: float
    auction_aggression: float
    denial_weight: float
    mortgage_urgency: float
    trade_margin: float
    accept_trade_bias: float = 0.0
    counter_trade_bias: float = 0.0
    counter_trade_margin: float = 8.0
    completion_trade_bonus: float = 0.0
    expansion_trade_bonus: float = 0.0
    denial_trade_bonus: float = 0.0
    cash_sale_trade_bonus: float = 0.0
    swap_trade_bonus: float = 0.0
    stochastic_margin: float = 0.2


def default_scripted_profiles() -> dict[str, ScriptedPolicyProfile]:
    """Return the built-in scripted opponent variants keyed by their stable names."""
    return {
        "conservative_liquidity_manager": ScriptedPolicyProfile(
            name="conservative_liquidity_manager",
            cash_buffer_target=450.0,
            property_pressure_weight=0.75,
            expansion_weight=0.65,
            build_weight=0.7,
            liquidity_weight=1.5,
            auction_aggression=0.7,
            denial_weight=0.75,
            mortgage_urgency=1.6,
            trade_margin=35.0,
            accept_trade_bias=-6.0,
            counter_trade_bias=9.0,
            counter_trade_margin=18.0,
            completion_trade_bonus=4.0,
            expansion_trade_bonus=2.0,
            denial_trade_bonus=3.0,
            cash_sale_trade_bonus=10.0,
            swap_trade_bonus=-1.0,
        ),
        "auction_value_shark": ScriptedPolicyProfile(
            name="auction_value_shark",
            cash_buffer_target=250.0,
            property_pressure_weight=1.0,
            expansion_weight=0.85,
            build_weight=0.75,
            liquidity_weight=0.9,
            auction_aggression=1.55,
            denial_weight=1.35,
            mortgage_urgency=1.0,
            trade_margin=20.0,
            accept_trade_bias=-2.0,
            counter_trade_bias=4.0,
            counter_trade_margin=12.0,
            completion_trade_bonus=6.0,
            expansion_trade_bonus=4.0,
            denial_trade_bonus=8.0,
            cash_sale_trade_bonus=3.0,
            swap_trade_bonus=2.0,
        ),
        "expansionist_builder": ScriptedPolicyProfile(
            name="expansionist_builder",
            cash_buffer_target=225.0,
            property_pressure_weight=0.95,
            expansion_weight=1.45,
            build_weight=1.55,
            liquidity_weight=0.8,
            auction_aggression=1.1,
            denial_weight=0.9,
            mortgage_urgency=0.95,
            trade_margin=15.0,
            accept_trade_bias=3.0,
            counter_trade_bias=7.5,
            counter_trade_margin=6.0,
            completion_trade_bonus=14.0,
            expansion_trade_bonus=11.0,
            denial_trade_bonus=4.0,
            cash_sale_trade_bonus=-2.0,
            swap_trade_bonus=5.0,
        ),
        "monopoly_denial_disruptor": ScriptedPolicyProfile(
            name="monopoly_denial_disruptor",
            cash_buffer_target=300.0,
            property_pressure_weight=0.9,
            expansion_weight=0.8,
            build_weight=0.9,
            liquidity_weight=1.0,
            auction_aggression=1.2,
            denial_weight=1.7,
            mortgage_urgency=1.05,
            trade_margin=10.0,
            accept_trade_bias=-1.0,
            counter_trade_bias=8.5,
            counter_trade_margin=10.0,
            completion_trade_bonus=6.0,
            expansion_trade_bonus=4.0,
            denial_trade_bonus=15.0,
            cash_sale_trade_bonus=1.0,
            swap_trade_bonus=4.0,
        ),
    }


class ScriptedPolicyController:
    """Choose legal actions by scoring them with handcrafted strategic heuristics."""

    def __init__(
        self,
        profile: ScriptedPolicyProfile,
        *,
        seed: int = 7,
        observation_encoder: ObservationEncoder | None = None,
        action_space: MonopolyActionSpace | None = None,
    ) -> None:
        self.profile = profile
        self.observation_encoder = observation_encoder or ObservationEncoder()
        self.action_space = action_space or MonopolyActionSpace()
        self._rng = np.random.default_rng(seed)

    def set_seed(self, seed: int) -> None:
        """Reset the stochastic tie-breaker RNG for reproducible scripted choices."""
        self._rng = np.random.default_rng(seed)

    def choose_action(self, game: Game, actor_name: str, explore: bool = True) -> ControllerDecision:
        """Score the actor's legal actions and pick the best or sample among near-best choices."""
        frontend_state = game.get_frontend_state()
        turn_plan = game.get_turn_plan(actor_name)
        observation = self.observation_encoder.encode(frontend_state, actor_name)
        action_mask, choices = self.action_space.build_mask(turn_plan, frontend_state)
        action_mask, choices = filter_invalid_trade_choices(game, action_mask, choices)
        analysis = analyze_board(frontend_state, actor_name)
        scores_by_action_id = {
            action_id: self._score_choice(frontend_state, turn_plan, analysis, actor_name, choice)
            for action_id, choice in choices.items()
        }
        best_score = max(scores_by_action_id.values(), default=0.0)
        candidate_action_ids = [
            action_id for action_id, score in scores_by_action_id.items() if score >= best_score - self.profile.stochastic_margin
        ]
        if not candidate_action_ids:
            candidate_action_ids = list(choices.keys())
        chosen_action_id = candidate_action_ids[0] if not explore else int(self._rng.choice(np.asarray(candidate_action_ids, dtype=np.int64)))
        choice = choices[chosen_action_id]
        return ControllerDecision(
            choice=choice,
            decision=PolicyDecision(
                action_id=choice.action_id,
                log_probability=0.0,
                value=self.evaluate_state_values(frontend_state, (actor_name,))[actor_name],
            ),
            observation=observation,
            action_mask=action_mask,
            heuristic_bias=np.zeros(self.action_space.action_count, dtype=np.float32),
        )

    def evaluate_state_values(self, frontend_state: FrontendStateView, actor_names: tuple[str, ...]) -> dict[str, float]:
        """Estimate simple value targets by scaling public board-strength scores."""
        analysis_by_actor = {actor_name: analyze_board(frontend_state, actor_name) for actor_name in actor_names}
        return {
            actor_name: analysis_by_actor[actor_name].player_metrics[actor_name].board_strength / 1000.0
            for actor_name in actor_names
        }

    def _score_choice(
        self,
        frontend_state: FrontendStateView,
        turn_plan: TurnPlanView,
        analysis,
        actor_name: str,
        choice: AgentActionChoice,
    ) -> float:
        """Dispatch one legal action choice to the appropriate handcrafted scorer."""
        actor = next(player for player in frontend_state.game_view.players if player.name == actor_name)
        actor_metrics = analysis.player_metrics[actor_name]
        pending_action = frontend_state.game_view.pending_action
        label = choice.action_label
        if label == "start_turn":
            return 1000.0
        if label == "end_turn":
            return self._score_end_turn(frontend_state, analysis, actor_name, actor)
        if label in {"buy_property", "decline_property"}:
            buy_score = self._score_property_acquisition(frontend_state, analysis, actor, actor_metrics, pending_action)
            return buy_score if label == "buy_property" else -buy_score
        if label.startswith("build:"):
            return self._score_build_action(frontend_state, actor, choice)
        if label == "confirm_property_action":
            return self._score_confirm_property_action(frontend_state, analysis, actor_name, actor, pending_action)
        if label.startswith("sell:"):
            return self._score_sell_action(frontend_state, actor, choice)
        if label.startswith("mortgage:"):
            return self._score_mortgage_action(frontend_state, actor, choice)
        if label.startswith("unmortgage:"):
            return self._score_unmortgage_action(frontend_state, actor, choice)
        if label == "cancel_property_action":
            return -4.0
        if label == "pass_auction" or label.startswith("auction_bid_"):
            return self._score_auction_choice(frontend_state, analysis, actor_name, actor, choice)
        if label.startswith("jail_"):
            return self._score_jail_choice(actor, label)
        if label in {"accept_trade", "reject_trade"}:
            trade_score = self._score_trade_decision(frontend_state, analysis, actor_name, pending_action)
            if label == "accept_trade":
                return trade_score + self.profile.accept_trade_bias
            return -trade_score
        if label.startswith("trade_"):
            return self._score_trade_offer(frontend_state, analysis, actor_name, choice)
        return 0.0

    def _score_property_acquisition(self, frontend_state: FrontendStateView, analysis, actor: PlayerView, actor_metrics, pending_action: PendingActionView | None) -> float:
        if pending_action is None or pending_action.property_index is None:
            return 0.0
        space = frontend_state.board_spaces[pending_action.property_index]
        pressure = analysis.estimated_space_pressure_by_index.get(space.index, 0.0)
        color_pressure = self._group_progress(analysis, actor.name, space)
        strongest = strongest_opponent_name(analysis, actor.name)
        opponent_progress = 0.0 if strongest is None else self._group_progress(analysis, strongest, space)
        cash_after = float(actor.cash - (pending_action.price or 0))
        liquidity_penalty = max(0.0, self.profile.cash_buffer_target - cash_after) * self.profile.liquidity_weight / 150.0
        return (
            pressure * self.profile.property_pressure_weight / 40.0
            + color_pressure * 12.0 * self.profile.expansion_weight
            + opponent_progress * 10.0 * self.profile.denial_weight
            + actor_metrics.buildable_monopoly_count * self.profile.build_weight
            - liquidity_penalty
        )

    def _score_build_action(self, frontend_state: FrontendStateView, actor: PlayerView, choice: AgentActionChoice) -> float:
        property_name = choice.legal_action.property_name or choice.action_label.partition(":")[2]
        space = self._space_by_name(frontend_state, property_name)
        if space is None:
            return 0.0
        analysis = analyze_board(frontend_state, actor.name)
        house_cost = float(space.house_cost or 0)
        cash_after = float(actor.cash) - house_cost
        current_pressure = analysis.estimated_space_pressure_by_index.get(space.index, 0.0)
        monopoly_progress = self._group_progress(analysis, actor.name, space)
        improvement_value = house_cost * (0.35 + 0.2 * float(space.building_count or 0))
        threat_penalty = self._liquidity_shortfall_penalty(actor.cash, house_cost, strongest_opponent_name(analysis, actor.name), analysis)
        return (
            current_pressure / 45.0
            + improvement_value / 30.0
            + monopoly_progress * 10.0 * self.profile.build_weight
            - threat_penalty
        )

    def _score_confirm_property_action(
        self,
        frontend_state: FrontendStateView,
        analysis,
        actor_name: str,
        actor: PlayerView,
        pending_action: PendingActionView | None,
    ) -> float:
        if pending_action is None or pending_action.property_action is None:
            return 0.0
        property_action = pending_action.property_action
        property_name = property_action.property_name
        space = self._space_by_name(frontend_state, property_name)
        if space is None:
            return 0.0
        if property_action.action_type == "build":
            synthetic_choice = AgentActionChoice(action_id=-1, action_label=f"build:{property_name}", legal_action=self._synthetic_legal_action(actor_name, property_name))
            return self._score_build_action(frontend_state, actor, synthetic_choice) + 1.0
        if property_action.action_type == "sell_building":
            synthetic_choice = AgentActionChoice(action_id=-1, action_label=f"sell:{property_name}", legal_action=self._synthetic_legal_action(actor_name, property_name))
            return self._score_sell_action(frontend_state, actor, synthetic_choice) + 1.0
        if property_action.action_type == "mortgage":
            synthetic_choice = AgentActionChoice(action_id=-1, action_label=f"mortgage:{property_name}", legal_action=self._synthetic_legal_action(actor_name, property_name))
            return self._score_mortgage_action(frontend_state, actor, synthetic_choice) + 1.0
        if property_action.action_type == "unmortgage":
            synthetic_choice = AgentActionChoice(action_id=-1, action_label=f"unmortgage:{property_name}", legal_action=self._synthetic_legal_action(actor_name, property_name))
            return self._score_unmortgage_action(frontend_state, actor, synthetic_choice) + 1.0
        return self._score_end_turn(frontend_state, analysis, actor_name, actor)

    def _score_sell_action(self, frontend_state: FrontendStateView, actor: PlayerView, choice: AgentActionChoice) -> float:
        property_name = choice.legal_action.property_name or choice.action_label.partition(":")[2]
        space = self._space_by_name(frontend_state, property_name)
        if space is None:
            return -5.0
        analysis = analyze_board(frontend_state, actor.name)
        house_relief = float(space.house_cost or 0) * 0.5
        strategic_loss = self._estimate_property_strategic_value(frontend_state, analysis, actor.name, property_name) / 45.0
        monopoly_penalty = 10.0 if self._owns_monopoly(frontend_state, actor.name, space.color_group) else 0.0
        relief_score = self._liquidity_relief_score(actor.cash, house_relief)
        return relief_score - strategic_loss - monopoly_penalty - float(space.building_count or 0) * 0.75

    def _score_mortgage_action(self, frontend_state: FrontendStateView, actor: PlayerView, choice: AgentActionChoice) -> float:
        property_name = choice.legal_action.property_name or choice.action_label.partition(":")[2]
        space = self._space_by_name(frontend_state, property_name)
        if space is None:
            return 0.0
        mortgage_value = float(space.price or 0) * 0.5
        analysis = analyze_board(frontend_state, actor.name)
        strategic_loss = self._estimate_property_strategic_value(frontend_state, analysis, actor.name, property_name) / 35.0
        monopoly_penalty = 16.0 if self._owns_monopoly(frontend_state, actor.name, space.color_group) else 0.0
        relief_score = self._liquidity_relief_score(actor.cash, mortgage_value)
        return relief_score + mortgage_value / 180.0 - strategic_loss - monopoly_penalty

    def _score_unmortgage_action(self, frontend_state: FrontendStateView, actor: PlayerView, choice: AgentActionChoice) -> float:
        property_name = choice.legal_action.property_name or choice.action_label.partition(":")[2]
        space = self._space_by_name(frontend_state, property_name)
        if space is None:
            return 0.0
        analysis = analyze_board(frontend_state, actor.name)
        unmortgage_cost = float(space.price or 0) * 0.55
        cash_after = float(actor.cash) - unmortgage_cost
        recovered_value = self._estimate_property_strategic_value(frontend_state, analysis, actor.name, property_name) / 28.0
        threat_penalty = self._liquidity_shortfall_penalty(actor.cash, unmortgage_cost, strongest_opponent_name(analysis, actor.name), analysis)
        return recovered_value - threat_penalty

    def _score_auction_choice(self, frontend_state: FrontendStateView, analysis, actor_name: str, actor: PlayerView, choice: AgentActionChoice) -> float:
        pending_action = frontend_state.game_view.pending_action
        if pending_action is None or pending_action.auction is None:
            return 0.0 if choice.action_label == "pass_auction" else -10.0
        auction = pending_action.auction
        space = frontend_state.board_spaces[auction.property_index]
        pressure = analysis.estimated_space_pressure_by_index.get(space.index, 0.0)
        actor_progress = self._group_progress(analysis, actor_name, space)
        strongest = strongest_opponent_name(analysis, actor_name)
        opponent_progress = 0.0 if strongest is None else self._group_progress(analysis, strongest, space)
        completion_bonus = 18.0 if actor_progress >= 0.5 else 0.0
        denial_bonus = 22.0 if opponent_progress >= 0.5 else 0.0
        desirability = (
            pressure * self.profile.property_pressure_weight / 35.0
            + actor_progress * 14.0 * self.profile.expansion_weight
            + opponent_progress * 12.0 * self.profile.denial_weight
            + completion_bonus
            + denial_bonus
        )
        if choice.action_label == "pass_auction":
            return self._liquidity_shortfall_penalty(actor.cash, 0.0, strongest, analysis) - desirability
        bid_amount = float(choice.bid_amount or auction.minimum_bid)
        intrinsic_value = self._estimate_property_strategic_value(frontend_state, analysis, actor_name, space.name)
        max_bid = float(space.price or 0) + completion_bonus * 2.5 + denial_bonus * 2.0 + intrinsic_value * 0.1
        liquidity_penalty = self._liquidity_shortfall_penalty(actor.cash, bid_amount, strongest, analysis)
        overpay_penalty = max(0.0, bid_amount - max_bid) / 18.0
        aggression_bonus = self.profile.auction_aggression * (1.5 if choice.action_label in {"auction_bid_aggressive", "auction_bid_denial"} else 1.0)
        return desirability + aggression_bonus - liquidity_penalty - overpay_penalty

    def _score_jail_choice(self, actor: PlayerView, label: str) -> float:
        if label == "jail_use_card":
            return 9.0 + float(actor.get_out_of_jail_cards)
        if label == "jail_pay_fine":
            return 6.0 - max(0.0, self.profile.cash_buffer_target - float(actor.cash)) / 100.0
        if label == "jail_roll":
            return 5.0 + max(0.0, self.profile.cash_buffer_target - float(actor.cash)) / 120.0
        return 0.0

    def _score_trade_decision(
        self,
        frontend_state: FrontendStateView,
        analysis,
        actor_name: str,
        pending_action: PendingActionView | None,
    ) -> float:
        if pending_action is None or pending_action.trade is None:
            return -1.0
        trade = pending_action.trade
        if actor_name != trade.receiver_name:
            return -1.0
        return self._trade_net_value(
            frontend_state,
            analysis,
            actor_name,
            receive_cash=float(trade.proposer_cash),
            give_cash=float(trade.receiver_cash),
            receive_properties=trade.proposer_property_names,
            give_properties=trade.receiver_property_names,
        ) - self.profile.trade_margin

    def _score_trade_offer(
        self,
        frontend_state: FrontendStateView,
        analysis,
        actor_name: str,
        choice: AgentActionChoice,
    ) -> float:
        payload = choice.trade_offer_payload
        if payload is None:
            return -5.0
        score = self._trade_net_value(
            frontend_state,
            analysis,
            actor_name,
            receive_cash=float(payload.get("receiver_cash", 0)),
            give_cash=float(payload.get("proposer_cash", 0)),
            receive_properties=tuple(str(name) for name in payload.get("receiver_property_names", ())),
            give_properties=tuple(str(name) for name in payload.get("proposer_property_names", ())),
        )
        if not payload.get("receiver_property_names") and not payload.get("proposer_property_names"):
            score -= 8.0
        score += self._trade_family_bonus(choice.action_label)
        pending_action = frontend_state.game_view.pending_action
        if choice.legal_action.action_type == "counter_trade":
            current_offer_score = self._score_trade_decision(frontend_state, analysis, actor_name, pending_action)
            score += self.profile.counter_trade_bias
            if current_offer_score > 0.0:
                score -= current_offer_score * 0.6
            else:
                score += min(12.0, -current_offer_score * 0.35)
            if score < self.profile.counter_trade_margin:
                score -= 6.0
        return score - self.profile.trade_margin * 0.5

    def _trade_family_bonus(self, action_label: str) -> float:
        """Apply profile-specific trade-template bonuses based on the template family label."""
        if action_label.startswith("trade_request_completion"):
            return self.profile.completion_trade_bonus
        if action_label.startswith("trade_request_expansion"):
            return self.profile.expansion_trade_bonus
        if action_label.startswith("trade_request_denial"):
            return self.profile.denial_trade_bonus
        if action_label.startswith("trade_sell_"):
            return self.profile.cash_sale_trade_bonus
        if action_label.startswith("trade_swap_"):
            return self.profile.swap_trade_bonus
        return 0.0

    def _trade_net_value(
        self,
        frontend_state: FrontendStateView,
        analysis,
        actor_name: str,
        *,
        receive_cash: float,
        give_cash: float,
        receive_properties: tuple[str, ...],
        give_properties: tuple[str, ...],
    ) -> float:
        strongest = strongest_opponent_name(analysis, actor_name)
        receive_value = sum(self._estimate_property_strategic_value(frontend_state, analysis, actor_name, property_name) for property_name in receive_properties)
        give_value = sum(self._estimate_property_strategic_value(frontend_state, analysis, actor_name, property_name) for property_name in give_properties)
        receive_bonus = sum(self._trade_completion_bonus(frontend_state, actor_name, property_name, gaining=True) for property_name in receive_properties)
        give_penalty = sum(self._trade_completion_bonus(frontend_state, actor_name, property_name, gaining=False) for property_name in give_properties)
        opponent_penalty = 0.0 if strongest is None else sum(
            self._trade_opponent_risk_penalty(frontend_state, strongest, property_name) for property_name in give_properties
        )
        cash_swing = receive_cash - give_cash
        return cash_swing + (receive_value - give_value) + receive_bonus - give_penalty - opponent_penalty

    def _score_end_turn(self, frontend_state: FrontendStateView, analysis, actor_name: str, actor: PlayerView) -> float:
        strongest = strongest_opponent_name(analysis, actor_name)
        return (
            4.0
            + analysis.player_metrics[actor_name].buildable_monopoly_count * 0.5
            - self._liquidity_shortfall_penalty(actor.cash, 0.0, strongest, analysis)
        )

    def _estimate_property_strategic_value(
        self,
        frontend_state: FrontendStateView,
        analysis,
        actor_name: str,
        property_name: str,
    ) -> float:
        space = self._space_by_name(frontend_state, property_name)
        if space is None:
            return 0.0
        pressure = analysis.estimated_space_pressure_by_index.get(space.index, 0.0)
        actor_progress = self._group_progress(analysis, actor_name, space)
        strongest = strongest_opponent_name(analysis, actor_name)
        opponent_progress = 0.0 if strongest is None else self._group_progress(analysis, strongest, space)
        buildable_bonus = 35.0 if self._owns_monopoly(frontend_state, actor_name, space.color_group) and not space.mortgaged else 0.0
        return (
            pressure
            + float(space.price or 0) * 0.25
            + actor_progress * 60.0 * self.profile.expansion_weight
            + opponent_progress * 55.0 * self.profile.denial_weight
            + buildable_bonus
            + float(space.house_cost or 0) * float(space.building_count or 0) * 0.75
        )

    def _trade_completion_bonus(self, frontend_state: FrontendStateView, actor_name: str, property_name: str, *, gaining: bool) -> float:
        space = self._space_by_name(frontend_state, property_name)
        if space is None or space.color_group is None:
            return 0.0
        progress = self._group_progress_with_transfer(frontend_state, actor_name, property_name, gaining=gaining)
        current = self._group_progress_for_player(frontend_state, actor_name, space.color_group)
        if gaining and current < 1.0 and progress >= 1.0:
            return 120.0
        if gaining and current < 0.75 and progress >= 0.75:
            return 45.0
        if not gaining and current >= 1.0 and progress < 1.0:
            return 140.0
        if not gaining and current >= 0.75 and progress < 0.75:
            return 50.0
        return 0.0

    def _trade_opponent_risk_penalty(self, frontend_state: FrontendStateView, opponent_name: str, property_name: str) -> float:
        space = self._space_by_name(frontend_state, property_name)
        if space is None or space.color_group is None:
            return 0.0
        current = self._group_progress_for_player(frontend_state, opponent_name, space.color_group)
        after_gain = self._group_progress_with_transfer(frontend_state, opponent_name, property_name, gaining=True)
        if current < 1.0 and after_gain >= 1.0:
            return 160.0
        if current < 0.75 and after_gain >= 0.75:
            return 60.0
        return 0.0

    def _liquidity_relief_score(self, current_cash: float, cash_relief: float) -> float:
        current_shortfall = max(0.0, self.profile.cash_buffer_target - current_cash)
        next_shortfall = max(0.0, self.profile.cash_buffer_target - (current_cash + cash_relief))
        return (current_shortfall - next_shortfall) * self.profile.mortgage_urgency / 18.0

    def _liquidity_shortfall_penalty(self, current_cash: float, spend_amount: float, strongest_opponent: str | None, analysis) -> float:
        threat_pressure = 0.0 if strongest_opponent is None else analysis.player_metrics[strongest_opponent].rent_potential / 300.0
        cash_after = current_cash - spend_amount
        return max(0.0, self.profile.cash_buffer_target + threat_pressure - cash_after) * self.profile.liquidity_weight / 90.0

    def _owns_monopoly(self, frontend_state: FrontendStateView, player_name: str, color_group: str | None) -> bool:
        if color_group is None or color_group not in COLOR_GROUP_SEQUENCE:
            return False
        group_spaces = [space for space in frontend_state.board_spaces if space.color_group == color_group]
        return bool(group_spaces) and all(space.owner_name == player_name for space in group_spaces)

    def _group_progress_for_player(self, frontend_state: FrontendStateView, player_name: str, color_group: str | None) -> float:
        if color_group is None or color_group not in COLOR_GROUP_SEQUENCE:
            return 0.0
        group_spaces = [space for space in frontend_state.board_spaces if space.color_group == color_group]
        if not group_spaces:
            return 0.0
        owned_count = sum(1 for space in group_spaces if space.owner_name == player_name)
        return owned_count / float(len(group_spaces))

    def _group_progress_with_transfer(self, frontend_state: FrontendStateView, player_name: str, property_name: str, *, gaining: bool) -> float:
        space = self._space_by_name(frontend_state, property_name)
        if space is None or space.color_group is None or space.color_group not in COLOR_GROUP_SEQUENCE:
            return 0.0
        group_spaces = [group_space for group_space in frontend_state.board_spaces if group_space.color_group == space.color_group]
        if not group_spaces:
            return 0.0
        owned_count = sum(1 for group_space in group_spaces if group_space.owner_name == player_name)
        if gaining and space.owner_name != player_name:
            owned_count += 1
        if not gaining and space.owner_name == player_name:
            owned_count -= 1
        return max(0.0, owned_count / float(len(group_spaces)))

    def _group_progress(self, analysis, player_name: str, space: BoardSpaceView) -> float:
        if space.color_group not in COLOR_GROUP_SEQUENCE or player_name not in analysis.player_metrics:
            return 0.0
        return analysis.player_metrics[player_name].color_progress[COLOR_GROUP_SEQUENCE.index(space.color_group)]

    @staticmethod
    def _space_by_name(frontend_state: FrontendStateView, property_name: str | None) -> BoardSpaceView | None:
        if property_name is None:
            return None
        return next((space for space in frontend_state.board_spaces if space.name == property_name), None)

    def _space_pressure_by_name(self, frontend_state: FrontendStateView, property_name: str) -> float:
        space = self._space_by_name(frontend_state, property_name)
        if space is None:
            return 0.0
        return float(space.price or 0) + float(space.house_cost or 0) * float(space.building_count or 0)

    @staticmethod
    def _synthetic_legal_action(actor_name: str, property_name: str):
        from monopoly.api import LegalActionOption

        return LegalActionOption(
            action_type="synthetic_property_action",
            actor_name=actor_name,
            actor_role="ai",
            handler_name="synthetic_property_action",
            description="synthetic property action",
            property_name=property_name,
        )


def build_scripted_controller(
    variant_name: str,
    *,
    seed: int = 7,
    observation_encoder: ObservationEncoder | None = None,
    action_space: MonopolyActionSpace | None = None,
) -> ScriptedPolicyController:
    """Construct one scripted controller from a named built-in profile variant."""
    profiles = default_scripted_profiles()
    if variant_name not in profiles:
        raise ValueError(f"Unsupported scripted opponent variant: {variant_name}")
    return ScriptedPolicyController(
        profiles[variant_name],
        seed=seed,
        observation_encoder=observation_encoder,
        action_space=action_space,
    )