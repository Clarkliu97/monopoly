from __future__ import annotations

"""Training, reward, and heuristic configuration dataclasses for the agent stack."""

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class RewardWeights:
    """Weights controlling how transition reward components are combined."""

    win_reward: float = 2.0
    loss_penalty: float = -2.0
    cash_delta_weight: float = 0.0015
    net_worth_delta_weight: float = 0.00075
    property_gain_weight: float = 0.05
    monopoly_gain_weight: float = 0.2
    rent_potential_weight: float = 0.0015
    buildable_monopoly_weight: float = 0.18
    near_monopoly_weight: float = 0.07
    cluster_strength_weight: float = 0.04
    relative_board_strength_weight: float = 0.00075
    opponent_rent_pressure_weight: float = 0.001
    opponent_buildability_denial_weight: float = 0.08
    strategic_property_weight: float = 0.0006
    bankruptcy_penalty: float = -1.5
    opponent_bankruptcy_reward: float = 0.35
    jail_enter_penalty: float = -0.03
    turn_completion_reward: float = 0.005
    auction_overpay_weight: float = 0.0015
    auction_liquidity_weight: float = 0.00075
    auction_cash_reserve_ratio: float = 0.2


@dataclass(slots=True)
class HeuristicWeights:
    """Bias values used by heuristic action scoring."""

    buy_property_bias: float = 0.6
    decline_property_bias: float = -0.2
    build_bias: float = 0.45
    sell_building_bias: float = -0.15
    mortgage_bias: float = -0.25
    low_cash_mortgage_bonus: float = 0.5
    unmortgage_bias: float = 0.2
    auction_min_bid_bias: float = 0.0
    auction_value_bid_bias: float = 0.0
    auction_premium_bid_bias: float = 0.0
    auction_aggressive_bid_bias: float = 0.0
    pass_auction_bias: float = 0.0
    jail_use_card_bias: float = 0.3
    jail_pay_fine_bias: float = 0.05
    jail_roll_bias: float = 0.1
    accept_trade_bias: float = -0.2
    reject_trade_bias: float = 0.2
    start_turn_bias: float = 0.1
    end_turn_bias: float = 0.1
    cash_buffer_target: float = 250.0


@dataclass(slots=True)
class PolicyConfig:
    """Hyperparameters describing the policy model and PPO optimization setup."""

    learning_rate: float = 0.0003
    minimum_learning_rate: float = 0.00003
    learning_rate_schedule: str = "linear"
    model_type: str = "transformer"
    discount_gamma: float = 0.97
    gae_lambda: float = 0.95
    bootstrap_truncated_episodes: bool = True
    use_heuristic_bias: bool = False
    heuristic_anneal_schedule: str = "linear"
    heuristic_bias_start: float = 1.0
    heuristic_bias_end: float = 0.0
    heuristic_anneal_iterations: int = 100
    entropy_weight: float = 0.003
    gradient_clip: float = 1.0
    advantage_clip: float = 5.0
    seed: int = 7
    hidden_sizes: tuple[int, ...] = (512, 512, 256, 256)
    transformer_embedding_size: int = 128
    transformer_heads: int = 8
    transformer_layers: int = 2
    ppo_clip_ratio: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 512
    value_loss_weight: float = 0.5
    threat_loss_weight: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class TrainingConfig:
    """Runtime settings for rollout collection, checkpoints, and league play."""

    worker_count: int = 2
    episodes_per_worker: int = 2
    max_steps_per_episode: int = 400
    max_actions_per_episode: int = 2000
    players_per_game: int = 4
    starting_cash: int = 1500
    checkpoint_interval: int = 5
    output_directory: str = ".checkpoints"
    player_name_prefix: str = "AI"
    use_auction_macro_steps: bool = True
    use_league_self_play: bool = False
    league_snapshot_interval: int = 10
    league_recent_snapshot_count: int = 4
    league_use_heuristic_baseline: bool = False
    league_use_scripted_opponents: bool = True
    league_scripted_variants: tuple[str, ...] = (
        "conservative_liquidity_manager",
        "auction_value_shark",
        "expansionist_builder",
        "monopoly_denial_disruptor",
    )
    benchmark_interval: int = 10
    benchmark_games: int = 4
    benchmark_seed: int = 10_000
    benchmark_seed_step: int = 1
    benchmark_max_steps: int = 400
    benchmark_players_per_game: int = 2
    scripted_rolls: tuple[tuple[int, int], ...] = field(default_factory=tuple)