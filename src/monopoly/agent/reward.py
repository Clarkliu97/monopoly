from __future__ import annotations

"""Reward shaping for self-play and evaluation transitions."""

from dataclasses import dataclass

from monopoly.agent.board_analysis import analyze_board, max_opponent_buildability, max_opponent_rent_pressure, relative_board_strength
from monopoly.agent.config import RewardWeights
from monopoly.api import FrontendStateView
from monopoly.constants import COLOR_GROUP_SIZES


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Named decomposition of one transition reward signal."""

    total: float
    cash_delta: float
    net_worth_delta: float
    property_delta: float
    monopoly_delta: float
    rent_potential_delta: float
    buildable_monopoly_delta: float
    near_monopoly_delta: float
    cluster_strength_delta: float
    relative_board_strength_delta: float
    opponent_rent_pressure_delta: float
    opponent_buildability_denial_delta: float
    strategic_property_delta: float
    bankruptcy_delta: float
    terminal_delta: float
    jail_delta: float
    turn_delta: float
    auction_overpay_delta: float
    auction_liquidity_delta: float


class RewardFunction:
    """Compute actor-relative reward deltas between two frontend states."""

    def __init__(self, weights: RewardWeights | None = None) -> None:
        """Create a reward function with the provided shaping weights or defaults."""
        self.weights = weights or RewardWeights()

    def score_transition(self, previous_state: FrontendStateView, next_state: FrontendStateView, actor_name: str) -> RewardBreakdown:
        """Score one environment transition for the specified actor."""
        previous_player = self._player_lookup(previous_state, actor_name)
        next_player = self._player_lookup(next_state, actor_name)
        previous_cash = previous_player.cash if previous_player is not None else 0
        next_cash = next_player.cash if next_player is not None else 0
        cash_delta = (next_cash - previous_cash) * self.weights.cash_delta_weight
        previous_analysis = analyze_board(previous_state, actor_name)
        next_analysis = analyze_board(next_state, actor_name)
        previous_metrics = previous_analysis.player_metrics.get(actor_name)
        next_metrics = next_analysis.player_metrics.get(actor_name)

        previous_net_worth = self._estimate_net_worth(previous_state, actor_name)
        next_net_worth = self._estimate_net_worth(next_state, actor_name)
        net_worth_delta = (next_net_worth - previous_net_worth) * self.weights.net_worth_delta_weight

        previous_property_count = 0 if previous_player is None else len(previous_player.properties)
        next_property_count = 0 if next_player is None else len(next_player.properties)
        property_delta = (next_property_count - previous_property_count) * self.weights.property_gain_weight

        previous_monopolies = self._monopoly_count(previous_state, actor_name)
        next_monopolies = self._monopoly_count(next_state, actor_name)
        monopoly_delta = (next_monopolies - previous_monopolies) * self.weights.monopoly_gain_weight
        rent_potential_delta = 0.0 if previous_metrics is None or next_metrics is None else (next_metrics.rent_potential - previous_metrics.rent_potential) * self.weights.rent_potential_weight
        buildable_monopoly_delta = 0.0 if previous_metrics is None or next_metrics is None else (next_metrics.buildable_monopoly_count - previous_metrics.buildable_monopoly_count) * self.weights.buildable_monopoly_weight
        near_monopoly_delta = 0.0 if previous_metrics is None or next_metrics is None else (next_metrics.near_monopoly_count - previous_metrics.near_monopoly_count) * self.weights.near_monopoly_weight
        cluster_strength_delta = 0.0 if previous_metrics is None or next_metrics is None else (next_metrics.cluster_strength - previous_metrics.cluster_strength) * self.weights.cluster_strength_weight
        relative_board_strength_delta = (relative_board_strength(next_analysis, actor_name) - relative_board_strength(previous_analysis, actor_name)) * self.weights.relative_board_strength_weight
        opponent_rent_pressure_delta = (max_opponent_rent_pressure(previous_analysis, actor_name) - max_opponent_rent_pressure(next_analysis, actor_name)) * self.weights.opponent_rent_pressure_weight
        opponent_buildability_denial_delta = (max_opponent_buildability(previous_analysis, actor_name) - max_opponent_buildability(next_analysis, actor_name)) * self.weights.opponent_buildability_denial_weight
        strategic_property_delta = 0.0 if previous_metrics is None or next_metrics is None else (next_metrics.strategic_property_value - previous_metrics.strategic_property_value) * self.weights.strategic_property_weight

        bankruptcy_delta = 0.0
        if previous_player is not None and next_player is not None and not previous_player.is_bankrupt and next_player.is_bankrupt:
            bankruptcy_delta += self.weights.bankruptcy_penalty
        previous_opponents_bankrupt = self._bankrupt_opponents(previous_state, actor_name)
        next_opponents_bankrupt = self._bankrupt_opponents(next_state, actor_name)
        bankruptcy_delta += (next_opponents_bankrupt - previous_opponents_bankrupt) * self.weights.opponent_bankruptcy_reward

        jail_delta = 0.0
        if previous_player is not None and next_player is not None and not previous_player.in_jail and next_player.in_jail:
            jail_delta += self.weights.jail_enter_penalty

        terminal_delta = 0.0
        winner_name = self._winner_name(next_state)
        if winner_name == actor_name:
            terminal_delta += self.weights.win_reward
        elif winner_name is not None and previous_player is not None and next_player is not None and next_player.is_bankrupt:
            terminal_delta += self.weights.loss_penalty

        turn_delta = self.weights.turn_completion_reward if next_state.game_view.current_player_name != previous_state.game_view.current_player_name else 0.0

        auction_overpay_delta, auction_liquidity_delta = self._score_auction_outcome(
            previous_state,
            next_state,
            actor_name,
            previous_cash=float(previous_cash),
            next_cash=float(next_cash),
        )

        total = (
            cash_delta
            + net_worth_delta
            + property_delta
            + monopoly_delta
            + rent_potential_delta
            + buildable_monopoly_delta
            + near_monopoly_delta
            + cluster_strength_delta
            + relative_board_strength_delta
            + opponent_rent_pressure_delta
            + opponent_buildability_denial_delta
            + strategic_property_delta
            + bankruptcy_delta
            + terminal_delta
            + jail_delta
            + turn_delta
            + auction_overpay_delta
            + auction_liquidity_delta
        )
        return RewardBreakdown(
            total=total,
            cash_delta=cash_delta,
            net_worth_delta=net_worth_delta,
            property_delta=property_delta,
            monopoly_delta=monopoly_delta,
            rent_potential_delta=rent_potential_delta,
            buildable_monopoly_delta=buildable_monopoly_delta,
            near_monopoly_delta=near_monopoly_delta,
            cluster_strength_delta=cluster_strength_delta,
            relative_board_strength_delta=relative_board_strength_delta,
            opponent_rent_pressure_delta=opponent_rent_pressure_delta,
            opponent_buildability_denial_delta=opponent_buildability_denial_delta,
            strategic_property_delta=strategic_property_delta,
            bankruptcy_delta=bankruptcy_delta,
            terminal_delta=terminal_delta,
            jail_delta=jail_delta,
            turn_delta=turn_delta,
            auction_overpay_delta=auction_overpay_delta,
            auction_liquidity_delta=auction_liquidity_delta,
        )

    @staticmethod
    def _player_lookup(state: FrontendStateView, actor_name: str):
        """Find a player view by name inside a serialized frontend state."""
        return next((player for player in state.game_view.players if player.name == actor_name), None)

    @staticmethod
    def _estimate_net_worth(state: FrontendStateView, actor_name: str) -> float:
        """Estimate liquid and board value held by the actor."""
        player = RewardFunction._player_lookup(state, actor_name)
        if player is None:
            return 0.0
        total = float(player.cash)
        for space in state.board_spaces:
            if space.owner_name != actor_name:
                continue
            if space.price is not None:
                total += float(space.price)
            if space.house_cost is not None and space.building_count is not None:
                total += float(space.house_cost * min(space.building_count, 4))
        return total

    @staticmethod
    def _monopoly_count(state: FrontendStateView, actor_name: str) -> int:
        """Count fully completed color groups owned by the actor."""
        color_counts = {color: 0 for color in COLOR_GROUP_SIZES}
        for space in state.board_spaces:
            if space.color_group and space.owner_name == actor_name:
                color_counts[space.color_group] += 1
        return sum(1 for color_group, expected_count in COLOR_GROUP_SIZES.items() if color_counts[color_group] == expected_count)

    @staticmethod
    def _bankrupt_opponents(state: FrontendStateView, actor_name: str) -> int:
        """Count bankrupt opponents still visible in the serialized state."""
        return sum(1 for player in state.game_view.players if player.name != actor_name and player.is_bankrupt)

    @staticmethod
    def _winner_name(state: FrontendStateView) -> str | None:
        """Return the sole non-bankrupt player if the game is effectively over."""
        active_players = [player for player in state.game_view.players if not player.is_bankrupt]
        if len(active_players) == 1:
            return active_players[0].name
        return None

    def _score_auction_outcome(
        self,
        previous_state: FrontendStateView,
        next_state: FrontendStateView,
        actor_name: str,
        *,
        previous_cash: float,
        next_cash: float,
    ) -> tuple[float, float]:
        """Score delayed auction consequences once an auction resolves."""
        previous_pending_action = previous_state.game_view.pending_action
        if previous_pending_action is None or previous_pending_action.action_type != "auction" or previous_pending_action.auction is None:
            return 0.0, 0.0

        auction = previous_pending_action.auction
        property_space_before = previous_state.board_spaces[auction.property_index]
        property_space_after = next_state.board_spaces[auction.property_index]
        acquired_property = property_space_before.owner_name != actor_name and property_space_after.owner_name == actor_name
        if not acquired_property:
            return 0.0, 0.0

        property_price = float(property_space_after.price or property_space_before.price or 0)
        amount_paid = max(0.0, previous_cash - next_cash)
        reserve_cash_target = float(next_state.game_view.starting_cash) * max(0.0, self.weights.auction_cash_reserve_ratio)
        overpay = max(0.0, amount_paid - property_price)
        liquidity_gap = max(0.0, reserve_cash_target - next_cash)
        return (
            -overpay * self.weights.auction_overpay_weight,
            -liquidity_gap * self.weights.auction_liquidity_weight,
        )