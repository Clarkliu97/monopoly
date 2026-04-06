from __future__ import annotations

"""Observation encoding for Monopoly RL policies.

The encoder projects frontend state into a fixed-size numeric vector containing
global turn context, actor-relative player features, and per-space board state.
"""

import numpy as np

from monopoly.agent.board_analysis import COLOR_GROUP_SEQUENCE, analyze_board, estimate_actor_threat, relative_board_strength, strongest_opponent_name
from monopoly.api import BoardSpaceView, FrontendStateView, PlayerView
from monopoly.constants import BANK_HOTELS, BANK_HOUSES, BOARD_SIZE, MAX_PLAYERS, POST_ROLL_PHASE, PRE_ROLL_PHASE, STARTING_CASH


_PENDING_TYPES = ("none", "property_purchase", "auction", "jail_decision", "property_action", "trade_decision")
_TURN_PHASES = ("none", PRE_ROLL_PHASE, "in_turn", POST_ROLL_PHASE)
_SPACE_TYPE_ORDER = ("street", "railroad", "utility", "card", "tax", "go", "jail", "free_parking", "go_to_jail", "other")
_COLOR_GROUPS = COLOR_GROUP_SEQUENCE
_AUCTION_FEATURE_SIZE = 16
_TRADE_FEATURE_SIZE = 14
_PLAYER_FEATURE_SIZE = 20 + len(_COLOR_GROUPS)
_SPACE_OWNER_FEATURE_SIZE = MAX_PLAYERS + 1
_SPACE_NUMERIC_FEATURE_SIZE = 9
GLOBAL_FEATURE_SIZE = 5 + len(_TURN_PHASES) + len(_PENDING_TYPES) + _AUCTION_FEATURE_SIZE + _TRADE_FEATURE_SIZE
PLAYER_FEATURE_SIZE = _PLAYER_FEATURE_SIZE
SPACE_FEATURE_SIZE = len(_SPACE_TYPE_ORDER) + _SPACE_OWNER_FEATURE_SIZE + len(_COLOR_GROUPS) + _SPACE_NUMERIC_FEATURE_SIZE
OBSERVATION_PLAYER_COUNT = MAX_PLAYERS
OBSERVATION_SPACE_COUNT = BOARD_SIZE


class ObservationEncoder:
    """Convert a frontend state view into the model's flat observation tensor."""

    def __init__(self) -> None:
        self._observation_size = self._compute_observation_size()
        self._turn_phase_vectors = self._build_one_hot_lookup(_TURN_PHASES)
        self._pending_type_vectors = self._build_one_hot_lookup(_PENDING_TYPES)
        self._space_type_vectors = self._build_one_hot_lookup(_SPACE_TYPE_ORDER)
        self._color_vectors = self._build_one_hot_lookup(_COLOR_GROUPS)
        self._zero_player_features = np.zeros(_PLAYER_FEATURE_SIZE, dtype=np.float32)
        self._space_type_default = self._space_type_vectors["other"]
        self._empty_color_vector = np.zeros(len(_COLOR_GROUPS), dtype=np.float32)

    @property
    def observation_size(self) -> int:
        """Total flattened feature length consumed by policy models."""
        return self._observation_size

    @property
    def observation_layout(self) -> dict[str, int]:
        """Describe the high-level layout of the flattened observation vector."""
        return {
            "global_size": GLOBAL_FEATURE_SIZE,
            "player_count": OBSERVATION_PLAYER_COUNT,
            "player_size": PLAYER_FEATURE_SIZE,
            "space_count": OBSERVATION_SPACE_COUNT,
            "space_size": SPACE_FEATURE_SIZE,
        }

    def encode(self, frontend_state: FrontendStateView, actor_name: str) -> np.ndarray:
        """Encode one actor-relative game state into a float32 observation vector."""
        game_view = frontend_state.game_view
        board_spaces = frontend_state.board_spaces
        analysis = analyze_board(frontend_state, actor_name)
        features = np.empty(self._observation_size, dtype=np.float32)
        index = 0

        features[index:index + 5] = (
            game_view.turn_counter / 200.0,
            game_view.houses_remaining / float(BANK_HOUSES),
            game_view.hotels_remaining / float(BANK_HOTELS),
            game_view.starting_cash / 2000.0,
            len(frontend_state.active_turn_plan.legal_actions) / 64.0,
        )
        index += 5

        turn_phase = game_view.current_turn_phase or "none"
        turn_phase_vector = self._turn_phase_vectors.get(turn_phase, self._turn_phase_vectors["none"])
        features[index:index + len(_TURN_PHASES)] = turn_phase_vector
        index += len(_TURN_PHASES)

        pending_type = "none" if game_view.pending_action is None else game_view.pending_action.action_type
        pending_type_vector = self._pending_type_vectors.get(pending_type, self._pending_type_vectors["none"])
        features[index:index + len(_PENDING_TYPES)] = pending_type_vector
        index += len(_PENDING_TYPES)

        auction_features = self._encode_auction_features(frontend_state, actor_name, analysis)
        features[index:index + _AUCTION_FEATURE_SIZE] = auction_features
        index += _AUCTION_FEATURE_SIZE

        trade_features = self._encode_trade_features(frontend_state, actor_name, analysis)
        features[index:index + _TRADE_FEATURE_SIZE] = trade_features
        index += _TRADE_FEATURE_SIZE

        ordered_players = analysis.ordered_players
        for slot in range(MAX_PLAYERS):
            if slot < len(ordered_players):
                player = ordered_players[slot]
                player_metrics = analysis.player_metrics[player.name]
                features[index:index + _PLAYER_FEATURE_SIZE] = self._encode_player(
                    player,
                    actor_name,
                    slot == 0,
                    game_view.starting_cash,
                    player_metrics,
                    relative_board_strength(analysis, player.name),
                )
            else:
                features[index:index + _PLAYER_FEATURE_SIZE] = self._zero_player_features
            index += _PLAYER_FEATURE_SIZE

        for space in board_spaces:
            index = self._encode_space(features, index, space, actor_name, analysis)

        return features

    def _compute_observation_size(self) -> int:
        """Compute the flattened vector length from the configured sub-block sizes."""
        globals_size = GLOBAL_FEATURE_SIZE
        players_size = MAX_PLAYERS * _PLAYER_FEATURE_SIZE
        per_space_size = SPACE_FEATURE_SIZE
        return globals_size + players_size + BOARD_SIZE * per_space_size

    @staticmethod
    def _build_one_hot_lookup(vocabulary: tuple[str, ...]) -> dict[str, np.ndarray]:
        """Build reusable one-hot vectors for a small categorical vocabulary."""
        lookup: dict[str, np.ndarray] = {}
        identity = np.eye(len(vocabulary), dtype=np.float32)
        for index, value in enumerate(vocabulary):
            lookup[value] = identity[index]
        return lookup

    @staticmethod
    def _players_in_actor_relative_order(players: tuple[PlayerView, ...], actor_name: str) -> list[PlayerView]:
        """Rotate players so the acting player always occupies slot zero."""
        players_list = list(players)
        actor_index = next((index for index, player in enumerate(players_list) if player.name == actor_name), 0)
        return players_list[actor_index:] + players_list[:actor_index]

    @staticmethod
    def _encode_player(player: PlayerView, actor_name: str, is_actor_slot: bool, starting_cash: int, player_metrics, relative_strength: float) -> tuple[float, ...]:
        """Encode one player's scalar and strategic metrics into a fixed block."""
        return (
            1.0,
            1.0 if is_actor_slot else 0.0,
            1.0 if player.name == actor_name else 0.0,
            player.cash / max(1.0, float(starting_cash)),
            player.position / float(BOARD_SIZE - 1),
            1.0 if player.in_jail else 0.0,
            player.jail_turns / 3.0,
            min(player.get_out_of_jail_cards, 4) / 4.0,
            1.0 if player.is_bankrupt else 0.0,
            player_metrics.properties_owned / 28.0,
            player_metrics.gross_property_value / 5000.0,
            player_metrics.developed_value / 3000.0,
            player_metrics.mortgage_share,
            player_metrics.monopoly_count / 8.0,
            player_metrics.near_monopoly_count / 8.0,
            player_metrics.buildable_monopoly_count / 8.0,
            player_metrics.rent_potential / 2500.0,
            player_metrics.cluster_strength / 20.0,
            max(-2.0, min(2.0, relative_strength / 4000.0)),
            max(-2.0, min(2.0, (player_metrics.board_strength - float(starting_cash)) / 6000.0)),
            *player_metrics.color_progress,
        )

    def _encode_space(self, features: np.ndarray, index: int, space: BoardSpaceView, actor_name: str, analysis) -> int:
        """Encode one board space and advance the write index."""
        space_type_vector = self._space_type_vectors.get(space.space_type or "other", self._space_type_default)
        features[index:index + len(_SPACE_TYPE_ORDER)] = space_type_vector
        index += len(_SPACE_TYPE_ORDER)

        owner_slot_vector = np.zeros(_SPACE_OWNER_FEATURE_SIZE, dtype=np.float32)
        owner_slot = analysis.owner_slot_by_space_index.get(space.index, -1)
        owner_slot_vector[0 if owner_slot < 0 else owner_slot + 1] = 1.0
        features[index:index + _SPACE_OWNER_FEATURE_SIZE] = owner_slot_vector
        index += _SPACE_OWNER_FEATURE_SIZE

        color_vector = self._color_vectors.get(space.color_group or "", self._empty_color_vector)
        features[index:index + len(_COLOR_GROUPS)] = color_vector
        index += len(_COLOR_GROUPS)

        features[index:index + _SPACE_NUMERIC_FEATURE_SIZE] = (
            1.0 if actor_name in space.occupant_names else 0.0,
            len(space.occupant_names) / float(MAX_PLAYERS),
            1.0 if space.mortgaged else 0.0,
            0.0 if space.building_count is None else space.building_count / 5.0,
            0.0 if space.price is None else space.price / 400.0,
            0.0 if space.house_cost is None else space.house_cost / 200.0,
            analysis.owner_group_progress_by_space_index.get(space.index, 0.0),
            analysis.owner_group_buildable_by_space_index.get(space.index, 0.0),
            analysis.estimated_space_pressure_by_index.get(space.index, 0.0) / 1000.0,
        )
        return index + _SPACE_NUMERIC_FEATURE_SIZE

    def _encode_auction_features(self, frontend_state: FrontendStateView, actor_name: str, analysis) -> np.ndarray:
        """Encode auction-specific context when an auction is currently pending."""
        pending_action = frontend_state.game_view.pending_action
        if pending_action is None or pending_action.action_type != "auction" or pending_action.auction is None:
            return np.zeros(_AUCTION_FEATURE_SIZE, dtype=np.float32)

        auction = pending_action.auction
        property_space = frontend_state.board_spaces[auction.property_index]
        property_price = float(property_space.price or 0)
        safe_property_price = max(1.0, property_price)
        starting_cash = max(1.0, float(frontend_state.game_view.starting_cash or STARTING_CASH))
        eligible_count = len(auction.eligible_player_names) / float(MAX_PLAYERS)
        active_count = len(auction.active_player_names) / float(MAX_PLAYERS)
        current_bid = float(auction.current_bid)
        minimum_bid = float(auction.minimum_bid)
        actor_color_progress, opponent_color_progress = self._auction_color_progress(frontend_state, actor_name, property_space.color_group)
        actor_metrics = analysis.player_metrics[actor_name]
        strongest_opponent = strongest_opponent_name(analysis, actor_name)
        current_winner_metrics = None if auction.current_winner_name is None else analysis.player_metrics.get(auction.current_winner_name)
        return np.asarray(
            (
                1.0,
                1.0 if pending_action.player_name == actor_name else 0.0,
                1.0 if auction.current_winner_name == actor_name else 0.0,
                eligible_count,
                active_count,
                minimum_bid / starting_cash,
                current_bid / starting_cash,
                property_price / 400.0,
                min(3.0, minimum_bid / safe_property_price),
                min(3.0, current_bid / safe_property_price),
                actor_color_progress,
                opponent_color_progress,
                actor_metrics.board_strength / 6000.0,
                0.0 if current_winner_metrics is None else current_winner_metrics.board_strength / 6000.0,
                0.0 if strongest_opponent is None else analysis.player_metrics[strongest_opponent].board_strength / 6000.0,
                estimate_actor_threat(frontend_state, actor_name),
            ),
            dtype=np.float32,
        )

    def _encode_trade_features(self, frontend_state: FrontendStateView, actor_name: str, analysis) -> np.ndarray:
        """Encode trade-opportunity and pending-trade context for the actor."""
        pending_action = frontend_state.game_view.pending_action
        starting_cash = max(1.0, float(frontend_state.game_view.starting_cash or STARTING_CASH))
        board_spaces = frontend_state.board_spaces
        legal_trade_targets = tuple(
            action.target_player_name
            for action in frontend_state.active_turn_plan.legal_actions
            if action.action_type == "propose_trade" and action.target_player_name is not None
        )
        actor = next((player for player in frontend_state.game_view.players if player.name == actor_name), None)
        if actor is None:
            return np.zeros(_TRADE_FEATURE_SIZE, dtype=np.float32)
        actor_property_values = sorted(
            (float(space.price or 0) for space in board_spaces if space.owner_name == actor_name and space.price is not None),
            reverse=True,
        )
        best_offer_value = actor_property_values[0] if actor_property_values else 0.0
        receivable_values: list[float] = []
        best_monopoly_swing = 0.0
        actor_metrics = analysis.player_metrics[actor_name]
        for target_name in legal_trade_targets:
            for space in board_spaces:
                if space.owner_name != target_name or space.price is None:
                    continue
                receivable_values.append(float(space.price))
                if space.color_group in _COLOR_GROUPS:
                    color_index = _COLOR_GROUPS.index(space.color_group)
                    best_monopoly_swing = max(best_monopoly_swing, min(1.0, actor_metrics.color_progress[color_index] + 1.0 / 3.0))
        best_receivable_value = max(receivable_values, default=0.0)

        pending_trade_flag = 0.0
        actor_is_trade_proposer = 0.0
        actor_is_trade_receiver = 0.0
        pending_proposer_cash = 0.0
        pending_receiver_cash = 0.0
        pending_proposer_property_count = 0.0
        pending_receiver_property_count = 0.0
        pending_fairness = 0.0
        pending_strategic_swing = 0.0
        if pending_action is not None and pending_action.action_type == "trade_decision" and pending_action.trade is not None:
            trade = pending_action.trade
            pending_trade_flag = 1.0
            actor_is_trade_proposer = 1.0 if trade.proposer_name == actor_name else 0.0
            actor_is_trade_receiver = 1.0 if trade.receiver_name == actor_name else 0.0
            pending_proposer_cash = trade.proposer_cash / starting_cash
            pending_receiver_cash = trade.receiver_cash / starting_cash
            pending_proposer_property_count = len(trade.proposer_property_names) / 4.0
            pending_receiver_property_count = len(trade.receiver_property_names) / 4.0
            proposer_property_value = sum(float(self._space_price(board_spaces, name)) for name in trade.proposer_property_names)
            receiver_property_value = sum(float(self._space_price(board_spaces, name)) for name in trade.receiver_property_names)
            proposer_total = proposer_property_value + float(trade.proposer_cash)
            receiver_total = receiver_property_value + float(trade.receiver_cash)
            if trade.proposer_name == actor_name:
                pending_fairness = (receiver_total - proposer_total) / starting_cash
            elif trade.receiver_name == actor_name:
                pending_fairness = (proposer_total - receiver_total) / starting_cash
            pending_strategic_swing = best_monopoly_swing

        return np.asarray(
            (
                pending_trade_flag,
                actor_is_trade_proposer,
                actor_is_trade_receiver,
                pending_proposer_cash,
                pending_receiver_cash,
                pending_proposer_property_count,
                pending_receiver_property_count,
                max(-2.0, min(2.0, pending_fairness)),
                len(legal_trade_targets) / float(max(1, MAX_PLAYERS - 1)),
                best_offer_value / 400.0,
                best_receivable_value / 400.0,
                min(2.0, actor.cash / starting_cash),
                best_monopoly_swing,
                pending_strategic_swing,
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _space_price(board_spaces: tuple[BoardSpaceView, ...], property_name: str) -> int:
        """Look up a property purchase price by name from serialized board spaces."""
        space = next((candidate for candidate in board_spaces if candidate.name == property_name), None)
        return 0 if space is None or space.price is None else int(space.price)

    @staticmethod
    def _auction_color_progress(frontend_state: FrontendStateView, actor_name: str, color_group: str | None) -> tuple[float, float]:
        """Measure actor and opponent ownership progress inside one color group."""
        if color_group is None or color_group not in _COLOR_GROUPS:
            return 0.0, 0.0
        group_size = float(len([space for space in frontend_state.board_spaces if space.color_group == color_group]))
        actor_count = 0
        opponent_max = 0
        for player in frontend_state.game_view.players:
            owned_in_group = sum(1 for space in frontend_state.board_spaces if space.color_group == color_group and space.owner_name == player.name)
            if player.name == actor_name:
                actor_count = owned_in_group
            else:
                opponent_max = max(opponent_max, owned_in_group)
        return actor_count / group_size, opponent_max / group_size
