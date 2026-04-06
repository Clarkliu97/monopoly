from __future__ import annotations

"""Public-state board analysis used by the agent stack.

This module converts `FrontendStateView` snapshots into denser strategic metrics
that are shared by observation encoding, reward shaping, and heuristic action
selection. The analysis intentionally stays on the public-information side of the
project so it can be reused consistently by local play, online play, and RL.
"""

from dataclasses import dataclass

from monopoly.api import BoardSpaceView, FrontendStateView, PlayerView
from monopoly.constants import COLOR_GROUP_SIZES


COLOR_GROUP_SEQUENCE = tuple(COLOR_GROUP_SIZES)


@dataclass(frozen=True, slots=True)
class PlayerBoardMetrics:
    """Strategic board features derived for one player from public state only."""

    cash: float
    properties_owned: int
    gross_property_value: float
    developed_value: float
    mortgage_burden: float
    mortgage_share: float
    monopoly_count: int
    near_monopoly_count: int
    buildable_monopoly_count: int
    rent_potential: float
    cluster_strength: float
    strategic_property_value: float
    board_strength: float
    color_progress: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BoardAnalysis:
    """Actor-relative board analysis shared across the agent pipeline."""

    ordered_players: tuple[PlayerView, ...]
    slot_by_name: dict[str, int]
    player_metrics: dict[str, PlayerBoardMetrics]
    owner_slot_by_space_index: dict[int, int]
    owner_group_progress_by_space_index: dict[int, float]
    owner_group_buildable_by_space_index: dict[int, float]
    estimated_space_pressure_by_index: dict[int, float]


@dataclass(frozen=True, slots=True)
class TransitionDiagnostics:
    """High-level state-delta diagnostics used by reward shaping and evaluation."""

    rent_potential_delta: float
    monopoly_denial_events: float
    board_strength_trend: float
    auction_bid_quality: float | None


def analyze_board(frontend_state: FrontendStateView, actor_name: str) -> BoardAnalysis:
    """Summarize the board into actor-relative ownership and strength metrics."""
    players = frontend_state.game_view.players
    board_spaces = frontend_state.board_spaces
    ordered_players = tuple(_players_in_actor_relative_order(players, actor_name))
    slot_by_name = {player.name: index for index, player in enumerate(ordered_players)}

    group_spaces = {
        color_group: tuple(space for space in board_spaces if space.color_group == color_group)
        for color_group in COLOR_GROUP_SEQUENCE
    }
    railroad_counts = _property_type_counts(board_spaces, "railroad")
    utility_counts = _property_type_counts(board_spaces, "utility")

    player_metrics = {
        player.name: _build_player_metrics(
            frontend_state,
            player,
            board_spaces,
            group_spaces,
            railroad_counts,
            utility_counts,
        )
        for player in players
    }

    owner_slot_by_space_index: dict[int, int] = {}
    owner_group_progress_by_space_index: dict[int, float] = {}
    owner_group_buildable_by_space_index: dict[int, float] = {}
    estimated_space_pressure_by_index: dict[int, float] = {}
    for space in board_spaces:
        owner_name = space.owner_name
        owner_slot_by_space_index[space.index] = -1 if owner_name is None else slot_by_name.get(owner_name, -1)
        if owner_name is None or owner_name not in player_metrics:
            owner_group_progress_by_space_index[space.index] = 0.0
            owner_group_buildable_by_space_index[space.index] = 0.0
            estimated_space_pressure_by_index[space.index] = 0.0
            continue
        owner_metrics = player_metrics[owner_name]
        color_index = COLOR_GROUP_SEQUENCE.index(space.color_group) if space.color_group in COLOR_GROUP_SIZES else None
        owner_group_progress_by_space_index[space.index] = 0.0 if color_index is None else owner_metrics.color_progress[color_index]
        owner_group_buildable_by_space_index[space.index] = 1.0 if _space_owner_can_build(space, owner_name, group_spaces) else 0.0
        estimated_space_pressure_by_index[space.index] = _estimate_space_pressure(
            space,
            owner_name,
            group_spaces,
            railroad_counts,
            utility_counts,
        )

    return BoardAnalysis(
        ordered_players=ordered_players,
        slot_by_name=slot_by_name,
        player_metrics=player_metrics,
        owner_slot_by_space_index=owner_slot_by_space_index,
        owner_group_progress_by_space_index=owner_group_progress_by_space_index,
        owner_group_buildable_by_space_index=owner_group_buildable_by_space_index,
        estimated_space_pressure_by_index=estimated_space_pressure_by_index,
    )


def estimate_actor_threat(frontend_state: FrontendStateView, actor_name: str) -> float:
    """Estimate how far the actor trails the strongest opponent on board strength."""
    analysis = analyze_board(frontend_state, actor_name)
    actor_metrics = analysis.player_metrics.get(actor_name)
    if actor_metrics is None:
        return 0.0
    opponent_strengths = [
        metrics.board_strength
        for player_name, metrics in analysis.player_metrics.items()
        if player_name != actor_name
    ]
    if not opponent_strengths:
        return 0.0
    threat_gap = (max(opponent_strengths) - actor_metrics.board_strength) / 4000.0
    return max(-2.0, min(2.0, threat_gap))


def relative_board_strength(analysis: BoardAnalysis, actor_name: str) -> float:
    """Return the actor's board strength minus the strongest opposing score."""
    actor_metrics = analysis.player_metrics.get(actor_name)
    if actor_metrics is None:
        return 0.0
    opponent_strengths = [
        metrics.board_strength
        for player_name, metrics in analysis.player_metrics.items()
        if player_name != actor_name
    ]
    if not opponent_strengths:
        return actor_metrics.board_strength
    return actor_metrics.board_strength - max(opponent_strengths)


def max_opponent_rent_pressure(analysis: BoardAnalysis, actor_name: str) -> float:
    """Return the highest rent-pressure score among the actor's opponents."""
    return max(
        (metrics.rent_potential for player_name, metrics in analysis.player_metrics.items() if player_name != actor_name),
        default=0.0,
    )


def max_opponent_buildability(analysis: BoardAnalysis, actor_name: str) -> float:
    """Return the strongest opponent buildability signal used in reward shaping."""
    return max(
        (
            metrics.buildable_monopoly_count + 0.5 * metrics.near_monopoly_count
            for player_name, metrics in analysis.player_metrics.items()
            if player_name != actor_name
        ),
        default=0.0,
    )


def strongest_opponent_name(analysis: BoardAnalysis, actor_name: str) -> str | None:
    """Return the name of the opponent with the highest board-strength score."""
    strongest_name = None
    strongest_score = None
    for player_name, metrics in analysis.player_metrics.items():
        if player_name == actor_name:
            continue
        if strongest_score is None or metrics.board_strength > strongest_score:
            strongest_name = player_name
            strongest_score = metrics.board_strength
    return strongest_name


def analyze_transition(previous_state: FrontendStateView, next_state: FrontendStateView) -> TransitionDiagnostics:
    """Compare two public snapshots and summarize strategic movement between them."""
    previous_players = previous_state.game_view.players
    next_players = next_state.game_view.players
    if not previous_players or not next_players:
        return TransitionDiagnostics(0.0, 0.0, 0.0, None)
    previous_analysis = analyze_board(previous_state, previous_players[0].name)
    next_analysis = analyze_board(next_state, next_players[0].name)
    shared_names = tuple(sorted({player.name for player in previous_players} & {player.name for player in next_players}))
    if not shared_names:
        return TransitionDiagnostics(0.0, 0.0, 0.0, None)
    rent_potential_delta = sum(
        next_analysis.player_metrics[name].rent_potential - previous_analysis.player_metrics[name].rent_potential
        for name in shared_names
    ) / float(len(shared_names))
    board_strength_trend = sum(
        _relative_strength_for_name(next_analysis, name) - _relative_strength_for_name(previous_analysis, name)
        for name in shared_names
    ) / float(len(shared_names))
    monopoly_denial_events = sum(
        max(0.0, previous_analysis.player_metrics[name].buildable_monopoly_count - next_analysis.player_metrics[name].buildable_monopoly_count)
        + 0.5 * max(0.0, previous_analysis.player_metrics[name].near_monopoly_count - next_analysis.player_metrics[name].near_monopoly_count)
        for name in shared_names
    )
    return TransitionDiagnostics(
        rent_potential_delta=rent_potential_delta,
        monopoly_denial_events=monopoly_denial_events,
        board_strength_trend=board_strength_trend,
        auction_bid_quality=_auction_bid_quality(previous_state, next_state),
    )


def _players_in_actor_relative_order(players: tuple[PlayerView, ...], actor_name: str) -> list[PlayerView]:
    """Rotate player order so the requested actor always occupies slot zero."""
    players_list = list(players)
    actor_index = next((index for index, player in enumerate(players_list) if player.name == actor_name), 0)
    return players_list[actor_index:] + players_list[:actor_index]


def _relative_strength_for_name(analysis: BoardAnalysis, player_name: str) -> float:
    """Return one player's lead or deficit relative to their strongest opponent."""
    metrics = analysis.player_metrics[player_name]
    opponent_best = max(
        (other_metrics.board_strength for other_name, other_metrics in analysis.player_metrics.items() if other_name != player_name),
        default=0.0,
    )
    return metrics.board_strength - opponent_best


def _auction_bid_quality(previous_state: FrontendStateView, next_state: FrontendStateView) -> float | None:
    """Estimate how expensive the resolved auction win was relative to face value."""
    pending_action = previous_state.game_view.pending_action
    if pending_action is None or pending_action.action_type != "auction" or pending_action.auction is None:
        return None
    auction = pending_action.auction
    property_before = previous_state.board_spaces[auction.property_index]
    property_after = next_state.board_spaces[auction.property_index]
    winner_name = property_after.owner_name
    if winner_name is None or property_before.owner_name == winner_name:
        return None
    previous_player = next((player for player in previous_state.game_view.players if player.name == winner_name), None)
    next_player = next((player for player in next_state.game_view.players if player.name == winner_name), None)
    if previous_player is None or next_player is None:
        return None
    property_price = float(property_after.price or property_before.price or 0)
    if property_price <= 0.0:
        return None
    amount_paid = max(0.0, float(previous_player.cash - next_player.cash))
    return amount_paid / property_price


def _build_player_metrics(
    frontend_state: FrontendStateView,
    player: PlayerView,
    board_spaces: tuple[BoardSpaceView, ...],
    group_spaces: dict[str, tuple[BoardSpaceView, ...]],
    railroad_counts: dict[str, int],
    utility_counts: dict[str, int],
) -> PlayerBoardMetrics:
    """Aggregate public ownership, value, and development signals for one player."""
    owned_spaces = tuple(space for space in board_spaces if space.owner_name == player.name)
    properties_owned = len(owned_spaces)
    gross_property_value = sum(float(space.price or 0) for space in owned_spaces)
    developed_value = sum(float((space.house_cost or 0) * (space.building_count or 0)) for space in owned_spaces)
    mortgaged_properties = sum(1 for space in owned_spaces if space.mortgaged)
    mortgage_burden = sum(float(space.price or 0) * 0.5 for space in owned_spaces if space.mortgaged)
    mortgage_share = 0.0 if properties_owned == 0 else mortgaged_properties / float(properties_owned)

    color_progress: list[float] = []
    monopoly_count = 0
    near_monopoly_count = 0
    buildable_monopoly_count = 0
    cluster_strength = 0.0
    for color_group in COLOR_GROUP_SEQUENCE:
        spaces_in_group = group_spaces[color_group]
        group_size = max(1, len(spaces_in_group))
        owned_count = sum(1 for space in spaces_in_group if space.owner_name == player.name)
        progress = owned_count / float(group_size)
        color_progress.append(progress)
        average_group_price = sum(float(space.price or 0) for space in spaces_in_group) / float(group_size)
        cluster_strength += (progress * progress) * max(1.0, average_group_price / 100.0)
        if owned_count == group_size:
            monopoly_count += 1
            if all(not space.mortgaged for space in spaces_in_group):
                buildable_monopoly_count += 1
        elif owned_count == group_size - 1:
            near_monopoly_count += 1

    rent_potential = sum(
        _estimate_space_pressure(space, player.name, group_spaces, railroad_counts, utility_counts)
        for space in owned_spaces
    )
    strategic_property_value = sum(
        _estimate_space_pressure(space, player.name, group_spaces, railroad_counts, utility_counts) + float(space.price or 0) * 0.35
        for space in owned_spaces
    )
    board_strength = (
        float(player.cash)
        + gross_property_value * 0.75
        + developed_value * 1.1
        + rent_potential * 1.8
        + strategic_property_value * 0.35
        + float(monopoly_count) * 200.0
        + float(buildable_monopoly_count) * 150.0
        + float(near_monopoly_count) * 80.0
        + cluster_strength * 80.0
        - mortgage_burden * 0.4
    )
    return PlayerBoardMetrics(
        cash=float(player.cash),
        properties_owned=properties_owned,
        gross_property_value=gross_property_value,
        developed_value=developed_value,
        mortgage_burden=mortgage_burden,
        mortgage_share=mortgage_share,
        monopoly_count=monopoly_count,
        near_monopoly_count=near_monopoly_count,
        buildable_monopoly_count=buildable_monopoly_count,
        rent_potential=rent_potential,
        cluster_strength=cluster_strength,
        strategic_property_value=strategic_property_value,
        board_strength=board_strength,
        color_progress=tuple(color_progress),
    )


def _property_type_counts(board_spaces: tuple[BoardSpaceView, ...], space_type: str) -> dict[str, int]:
    """Count owned spaces of one property type for each owner name."""
    counts: dict[str, int] = {}
    for space in board_spaces:
        if space.space_type != space_type or space.owner_name is None:
            continue
        counts[space.owner_name] = counts.get(space.owner_name, 0) + 1
    return counts


def _space_owner_can_build(
    space: BoardSpaceView,
    owner_name: str,
    group_spaces: dict[str, tuple[BoardSpaceView, ...]],
) -> bool:
    """Whether the current owner holds an unmortgaged monopoly for this space."""
    if space.color_group is None or space.color_group not in group_spaces:
        return False
    spaces_in_group = group_spaces[space.color_group]
    if not spaces_in_group or any(group_space.owner_name != owner_name for group_space in spaces_in_group):
        return False
    return all(not group_space.mortgaged for group_space in spaces_in_group)


def _estimate_space_pressure(
    space: BoardSpaceView,
    owner_name: str,
    group_spaces: dict[str, tuple[BoardSpaceView, ...]],
    railroad_counts: dict[str, int],
    utility_counts: dict[str, int],
) -> float:
    """Estimate how threatening a space is to opponents based on public state."""
    if space.owner_name != owner_name:
        return 0.0
    price = float(space.price or 0)
    if price <= 0.0:
        return 0.0
    if space.space_type == "street":
        owned_count = 0
        group_size = 1
        monopoly = False
        buildable = False
        if space.color_group is not None and space.color_group in group_spaces:
            spaces_in_group = group_spaces[space.color_group]
            group_size = max(1, len(spaces_in_group))
            owned_count = sum(1 for group_space in spaces_in_group if group_space.owner_name == owner_name)
            monopoly = owned_count == group_size
            buildable = monopoly and all(not group_space.mortgaged for group_space in spaces_in_group)
        pressure = price * 0.18 * (1.0 + 0.35 * max(0, owned_count - 1))
        if monopoly:
            pressure *= 1.7
        if buildable:
            pressure *= 1.15
        building_count = space.building_count or 0
        if building_count <= 4:
            pressure *= 1.0 + 0.7 * building_count
        else:
            pressure *= 4.5
        if space.mortgaged:
            pressure *= 0.15
        return pressure
    if space.space_type == "railroad":
        count = railroad_counts.get(owner_name, 1)
        pressure = price * 0.24 * (1.0 + 0.55 * max(0, count - 1))
        if space.mortgaged:
            pressure *= 0.2
        return pressure
    if space.space_type == "utility":
        count = utility_counts.get(owner_name, 1)
        pressure = price * (0.22 if count <= 1 else 0.38)
        if space.mortgaged:
            pressure *= 0.2
        return pressure
    pressure = price * 0.1
    if space.mortgaged:
        pressure *= 0.2
    return pressure