from __future__ import annotations

"""Self-play environment for Monopoly RL rollouts.

This module bridges the authoritative `Game` engine and the agent-training stack.
It creates AI-only games, executes controller decisions, aggregates rewards, and
packages the resulting trajectories into training examples.
"""

from dataclasses import dataclass

from monopoly.agent.board_analysis import TransitionDiagnostics, analyze_transition, estimate_actor_threat
from monopoly.agent.config import TrainingConfig
from monopoly.agent.controller import AgentPolicyController, ControllerDecision
from monopoly.agent.model import TrainingExample
from monopoly.agent.reward import RewardFunction
from monopoly.constants import AI_ROLE
from monopoly.dice import Dice
from monopoly.game import Game


@dataclass(frozen=True, slots=True)
class SelfPlayEpisode:
    """Collected data and diagnostics for one completed self-play episode."""

    training_examples: tuple[TrainingExample, ...]
    winner_name: str | None
    step_count: int
    macro_step_count: int
    raw_action_count: int
    auction_count: int
    truncated_auction_count: int
    total_reward: float
    rent_potential_trend: float
    monopoly_denial_events: float
    board_strength_trend: float
    auction_bid_quality_sum: float
    auction_bid_quality_count: int
    truncated_episode: bool
    used_truncated_bootstrap: bool


class MonopolySelfPlayEnvironment:
    """Run self-play episodes and transform them into PPO-style training data."""

    def __init__(
        self,
        training_config: TrainingConfig,
        reward_function: RewardFunction,
        controller: AgentPolicyController,
        discount_gamma: float = 0.97,
        gae_lambda: float = 0.95,
        bootstrap_truncated_episodes: bool = True,
    ) -> None:
        self.training_config = training_config
        self.reward_function = reward_function
        self.controller = controller
        self.discount_gamma = discount_gamma
        self.gae_lambda = gae_lambda
        self.bootstrap_truncated_episodes = bootstrap_truncated_episodes

    def create_game(self, seed: int | None = None) -> Game:
        """Create one AI-only training game from the current training configuration."""
        player_names = [f"{self.training_config.player_name_prefix}{index + 1}" for index in range(self.training_config.players_per_game)]
        player_roles = [AI_ROLE] * len(player_names)
        dice = Dice(seed=seed, scripted_rolls=list(self.training_config.scripted_rolls) or None)
        return Game(player_names=player_names, player_roles=player_roles, starting_cash=self.training_config.starting_cash, dice=dice)

    def run_episode(
        self,
        seed: int | None = None,
        explore: bool = True,
        *,
        controller_by_player: dict[str, AgentPolicyController] | None = None,
        learning_player_names: tuple[str, ...] | None = None,
    ) -> SelfPlayEpisode:
        """Run one self-play episode and return training examples plus rollout diagnostics."""
        game = self.create_game(seed=seed)
        player_names = tuple(player.name for player in game.players)
        active_learning_players = player_names if learning_player_names is None else tuple(
            player_name for player_name in learning_player_names if player_name in player_names
        )
        per_player_decisions: dict[str, list[tuple[ControllerDecision, int, float]]] = {
            player_name: [] for player_name in active_learning_players
        }
        reward_trajectories: dict[str, list[float]] = {player_name: [] for player_name in active_learning_players}
        value_trajectories: dict[str, list[float]] = {player_name: [] for player_name in active_learning_players}
        next_value_trajectories: dict[str, list[float]] = {player_name: [] for player_name in active_learning_players}
        terminal_trajectories: dict[str, list[bool]] = {player_name: [] for player_name in active_learning_players}
        total_reward = 0.0
        transition_index = 0
        raw_action_count = 0
        auction_count = 0
        truncated_auction_count = 0
        rent_potential_trend_total = 0.0
        monopoly_denial_events_total = 0.0
        board_strength_trend_total = 0.0
        auction_bid_quality_sum = 0.0
        auction_bid_quality_count = 0
        max_actions_per_episode = max(1, self.training_config.max_actions_per_episode)

        while (
            game.winner() is None
            and transition_index < self.training_config.max_steps_per_episode
            and raw_action_count < max_actions_per_episode
        ):
            if self.training_config.use_auction_macro_steps and game.pending_auction is not None:
                macro_step_result = self._collect_auction_macro_step(
                    game,
                    active_learning_players,
                    per_player_decisions,
                    reward_trajectories,
                    value_trajectories,
                    next_value_trajectories,
                    terminal_trajectories,
                    transition_index,
                    remaining_raw_actions=max_actions_per_episode - raw_action_count,
                    explore=explore,
                    controller_by_player=controller_by_player,
                )
                if macro_step_result.raw_actions_used <= 0:
                    break
                raw_action_count += macro_step_result.raw_actions_used
                auction_count += 1
                truncated_auction_count += 1 if macro_step_result.auction_truncated else 0
                rent_potential_trend_total += macro_step_result.transition_diagnostics.rent_potential_delta
                monopoly_denial_events_total += macro_step_result.transition_diagnostics.monopoly_denial_events
                board_strength_trend_total += macro_step_result.transition_diagnostics.board_strength_trend
                if macro_step_result.transition_diagnostics.auction_bid_quality is not None:
                    auction_bid_quality_sum += macro_step_result.transition_diagnostics.auction_bid_quality
                    auction_bid_quality_count += 1
                transition_index += 1
                total_reward = self._total_reward(reward_trajectories)
                continue

            active_plan = game.get_active_turn_plan()
            actor_name = active_plan.player_name
            previous_state = game.get_frontend_state()
            previous_values = self._evaluate_learning_values(previous_state, active_learning_players, controller_by_player)
            acting_controller = self._resolve_controller(actor_name, controller_by_player)
            decision = acting_controller.choose_action(game, actor_name, explore=explore)
            threat_target = estimate_actor_threat(previous_state, actor_name)
            if actor_name in per_player_decisions:
                per_player_decisions[actor_name].append((decision, transition_index, threat_target))
            trade_offer = None if decision.choice.trade_offer_payload is None else game.deserialize_trade_offer(decision.choice.trade_offer_payload)
            game.execute_legal_action(
                decision.choice.legal_action,
                bid_amount=decision.choice.bid_amount,
                trade_offer=trade_offer,
            )
            next_state = game.get_frontend_state()
            next_values = self._evaluate_learning_values(next_state, active_learning_players, controller_by_player)
            transition_diagnostics = analyze_transition(previous_state, next_state)
            transition_terminal = game.winner() is not None
            for player_name in active_learning_players:
                reward = self.reward_function.score_transition(previous_state, next_state, player_name).total
                reward_trajectories[player_name].append(reward)
                value_trajectories[player_name].append(previous_values[player_name])
                next_value_trajectories[player_name].append(next_values[player_name])
                terminal_trajectories[player_name].append(transition_terminal)
            raw_action_count += 1
            rent_potential_trend_total += transition_diagnostics.rent_potential_delta
            monopoly_denial_events_total += transition_diagnostics.monopoly_denial_events
            board_strength_trend_total += transition_diagnostics.board_strength_trend
            if transition_diagnostics.auction_bid_quality is not None:
                auction_bid_quality_sum += transition_diagnostics.auction_bid_quality
                auction_bid_quality_count += 1
            transition_index += 1
            total_reward = self._total_reward(reward_trajectories)

        winner = game.winner()
        winner_name = None if winner is None else winner.name
        episode_truncated = winner_name is None and (
            transition_index >= self.training_config.max_steps_per_episode or raw_action_count >= max_actions_per_episode
        )
        training_examples: list[TrainingExample] = []
        gae_by_player = {
            player_name: self._compute_gae(
                reward_trajectories[player_name],
                value_trajectories[player_name],
                next_value_trajectories[player_name],
                terminal_trajectories[player_name],
                bootstrap_final_transition=self.bootstrap_truncated_episodes and episode_truncated,
            )
            for player_name in active_learning_players
        }
        for player_name, decisions in per_player_decisions.items():
            returns, advantages = gae_by_player[player_name]
            for decision, decision_transition_index, threat_target in decisions:
                discounted_return = returns[decision_transition_index]
                training_examples.append(
                    TrainingExample(
                        observation=decision.observation.tolist(),
                        action_mask=decision.action_mask.tolist(),
                        heuristic_bias=decision.heuristic_bias.tolist(),
                        action_id=decision.decision.action_id,
                        discounted_return=discounted_return,
                        advantage=advantages[decision_transition_index],
                        old_log_probability=decision.decision.log_probability,
                        threat_target=threat_target,
                    )
                )

        return SelfPlayEpisode(
            training_examples=tuple(training_examples),
            winner_name=winner_name,
            step_count=transition_index,
            macro_step_count=transition_index,
            raw_action_count=raw_action_count,
            auction_count=auction_count,
            truncated_auction_count=truncated_auction_count,
            total_reward=total_reward,
            rent_potential_trend=rent_potential_trend_total,
            monopoly_denial_events=monopoly_denial_events_total,
            board_strength_trend=board_strength_trend_total,
            auction_bid_quality_sum=auction_bid_quality_sum,
            auction_bid_quality_count=auction_bid_quality_count,
            truncated_episode=episode_truncated,
            used_truncated_bootstrap=episode_truncated and self.bootstrap_truncated_episodes and transition_index > 0,
        )

    def _compute_gae(
        self,
        rewards: list[float],
        state_values: list[float],
        next_state_values: list[float],
        terminal_flags: list[bool],
        *,
        bootstrap_final_transition: bool,
    ) -> tuple[list[float], list[float]]:
        """Compute discounted returns and generalized advantage estimates for one trajectory."""
        if not rewards:
            return [], []
        advantages = [0.0] * len(rewards)
        returns = [0.0] * len(rewards)
        gae = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            is_terminal = terminal_flags[index]
            use_bootstrap = not is_terminal and (index < len(rewards) - 1 or bootstrap_final_transition)
            next_value = next_state_values[index] if use_bootstrap else 0.0
            nonterminal = 0.0 if is_terminal else (1.0 if use_bootstrap else 0.0)
            delta = rewards[index] + self.discount_gamma * next_value - state_values[index]
            gae = delta + self.discount_gamma * self.gae_lambda * nonterminal * gae
            advantages[index] = gae
            returns[index] = gae + state_values[index]
        return returns, advantages

    def _collect_auction_macro_step(
        self,
        game: Game,
        player_names: tuple[str, ...],
        per_player_decisions: dict[str, list[tuple[ControllerDecision, int, float]]],
        reward_trajectories: dict[str, list[float]],
        value_trajectories: dict[str, list[float]],
        next_value_trajectories: dict[str, list[float]],
        terminal_trajectories: dict[str, list[bool]],
        transition_index: int,
        *,
        remaining_raw_actions: int,
        explore: bool,
        controller_by_player: dict[str, AgentPolicyController] | None,
    ) -> _AuctionMacroStepResult:
        """Collapse a whole pending auction into one macro-transition when configured to do so."""
        previous_state = game.get_frontend_state()
        previous_values = self._evaluate_learning_values(previous_state, player_names, controller_by_player)
        macro_decisions: list[tuple[str, ControllerDecision, float]] = []
        raw_actions_used = 0

        while game.pending_auction is not None and game.winner() is None and raw_actions_used < remaining_raw_actions:
            active_plan = game.get_active_turn_plan()
            actor_name = active_plan.player_name
            bidder_state = game.get_frontend_state()
            decision = self._resolve_controller(actor_name, controller_by_player).choose_action(game, actor_name, explore=explore)
            threat_target = estimate_actor_threat(bidder_state, actor_name)
            macro_decisions.append((actor_name, decision, threat_target))
            trade_offer = None if decision.choice.trade_offer_payload is None else game.deserialize_trade_offer(decision.choice.trade_offer_payload)
            game.execute_legal_action(
                decision.choice.legal_action,
                bid_amount=decision.choice.bid_amount,
                trade_offer=trade_offer,
            )
            raw_actions_used += 1

        if raw_actions_used <= 0:
            return _AuctionMacroStepResult(
                raw_actions_used=0,
                auction_truncated=False,
                transition_diagnostics=TransitionDiagnostics(0.0, 0.0, 0.0, None),
            )

        next_state = game.get_frontend_state()
        next_values = self._evaluate_learning_values(next_state, player_names, controller_by_player)
        transition_diagnostics = analyze_transition(previous_state, next_state)
        transition_terminal = game.winner() is not None
        for actor_name, decision, threat_target in macro_decisions:
            if actor_name in per_player_decisions:
                per_player_decisions[actor_name].append((decision, transition_index, threat_target))
        for player_name in player_names:
            reward = self.reward_function.score_transition(previous_state, next_state, player_name).total
            reward_trajectories[player_name].append(reward)
            value_trajectories[player_name].append(previous_values[player_name])
            next_value_trajectories[player_name].append(next_values[player_name])
            terminal_trajectories[player_name].append(transition_terminal)
        return _AuctionMacroStepResult(
            raw_actions_used=raw_actions_used,
            auction_truncated=game.pending_auction is not None,
            transition_diagnostics=transition_diagnostics,
        )

    @staticmethod
    def _total_reward(reward_trajectories: dict[str, list[float]]) -> float:
        """Sum all per-player rewards accumulated during the episode."""
        return sum(sum(rewards) for rewards in reward_trajectories.values())

    def _resolve_controller(
        self,
        actor_name: str,
        controller_by_player: dict[str, AgentPolicyController] | None,
    ) -> AgentPolicyController:
        """Return the controller assigned to the acting seat, falling back to the default one."""
        if controller_by_player is None:
            return self.controller
        return controller_by_player.get(actor_name, self.controller)

    def _evaluate_learning_values(
        self,
        frontend_state,
        learning_player_names: tuple[str, ...],
        controller_by_player: dict[str, AgentPolicyController] | None,
    ) -> dict[str, float]:
        """Query value estimates for the tracked learning players, batching when possible."""
        if not learning_player_names:
            return {}
        if controller_by_player is None:
            return self.controller.evaluate_state_values(frontend_state, learning_player_names)
        controllers = {player_name: self._resolve_controller(player_name, controller_by_player) for player_name in learning_player_names}
        unique_controllers = {id(controller): controller for controller in controllers.values()}
        if len(unique_controllers) == 1:
            controller = next(iter(unique_controllers.values()))
            return controller.evaluate_state_values(frontend_state, learning_player_names)
        return {
            player_name: controllers[player_name].evaluate_state_values(frontend_state, (player_name,))[player_name]
            for player_name in learning_player_names
        }


@dataclass(frozen=True, slots=True)
class _AuctionMacroStepResult:
    """Internal summary of one macro-stepped auction resolution."""

    raw_actions_used: int
    auction_truncated: bool
    transition_diagnostics: TransitionDiagnostics
