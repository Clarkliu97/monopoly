from __future__ import annotations

"""Public state-transfer objects shared across gameplay, UI, and agents.

These frozen dataclasses define the serialized contract between the core game
engine, the pygame frontend, the online transport layer, and the RL agent code.
Each view captures a stable, read-only snapshot of part of the game state so the
rest of the project can communicate without depending on mutable engine objects.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping

from monopoly.constants import HUMAN_ROLE


@dataclass(frozen=True, slots=True)
class PlayerView:
    """Read-only projection of a player's public state."""

    name: str
    role: str
    cash: int
    position: int
    in_jail: bool
    jail_turns: int
    get_out_of_jail_cards: int
    is_bankrupt: bool
    properties: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "cash": self.cash,
            "position": self.position,
            "in_jail": self.in_jail,
            "jail_turns": self.jail_turns,
            "get_out_of_jail_cards": self.get_out_of_jail_cards,
            "is_bankrupt": self.is_bankrupt,
            "properties": list(self.properties),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlayerView:
        return cls(
            name=str(data["name"]),
            role=str(data["role"]),
            cash=int(data["cash"]),
            position=int(data["position"]),
            in_jail=bool(data["in_jail"]),
            jail_turns=int(data["jail_turns"]),
            get_out_of_jail_cards=int(data["get_out_of_jail_cards"]),
            is_bankrupt=bool(data["is_bankrupt"]),
            properties=tuple(str(item) for item in data["properties"]),
        )


@dataclass(frozen=True, slots=True)
class AuctionView:
    """Serialized snapshot of an in-progress property auction."""

    property_name: str
    property_index: int
    current_bid: int
    current_winner_name: str | None
    current_bidder_name: str | None
    eligible_player_names: tuple[str, ...]
    active_player_names: tuple[str, ...]
    minimum_bid: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "property_name": self.property_name,
            "property_index": self.property_index,
            "current_bid": self.current_bid,
            "current_winner_name": self.current_winner_name,
            "current_bidder_name": self.current_bidder_name,
            "eligible_player_names": list(self.eligible_player_names),
            "active_player_names": list(self.active_player_names),
            "minimum_bid": self.minimum_bid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AuctionView:
        return cls(
            property_name=str(data["property_name"]),
            property_index=int(data["property_index"]),
            current_bid=int(data["current_bid"]),
            current_winner_name=_optional_str(data.get("current_winner_name")),
            current_bidder_name=_optional_str(data.get("current_bidder_name")),
            eligible_player_names=tuple(str(item) for item in data["eligible_player_names"]),
            active_player_names=tuple(str(item) for item in data["active_player_names"]),
            minimum_bid=int(data["minimum_bid"]),
        )


@dataclass(frozen=True, slots=True)
class JailDecisionView:
    """Prompt data for a player's jail-release decision."""

    player_name: str
    player_role: str
    available_actions: tuple[str, ...]
    fine_amount: int
    has_get_out_of_jail_card: bool
    jail_turns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_name": self.player_name,
            "player_role": self.player_role,
            "available_actions": list(self.available_actions),
            "fine_amount": self.fine_amount,
            "has_get_out_of_jail_card": self.has_get_out_of_jail_card,
            "jail_turns": self.jail_turns,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JailDecisionView:
        return cls(
            player_name=str(data["player_name"]),
            player_role=str(data["player_role"]),
            available_actions=tuple(str(item) for item in data["available_actions"]),
            fine_amount=int(data["fine_amount"]),
            has_get_out_of_jail_card=bool(data["has_get_out_of_jail_card"]),
            jail_turns=int(data["jail_turns"]),
        )


@dataclass(frozen=True, slots=True)
class PropertyActionView:
    """Prompt data for build, sell, mortgage, or unmortgage actions."""

    action_type: str
    player_name: str
    player_role: str
    property_name: str
    property_index: int
    cash_effect: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "player_name": self.player_name,
            "player_role": self.player_role,
            "property_name": self.property_name,
            "property_index": self.property_index,
            "cash_effect": self.cash_effect,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PropertyActionView:
        return cls(
            action_type=str(data["action_type"]),
            player_name=str(data["player_name"]),
            player_role=str(data["player_role"]),
            property_name=str(data["property_name"]),
            property_index=int(data["property_index"]),
            cash_effect=int(data["cash_effect"]),
            description=str(data["description"]),
        )


@dataclass(frozen=True, slots=True)
class TradeDecisionView:
    """Public description of a trade offer awaiting acceptance or countering."""

    proposer_name: str
    proposer_role: str
    receiver_name: str
    receiver_role: str
    proposer_cash: int
    receiver_cash: int
    proposer_property_names: tuple[str, ...]
    receiver_property_names: tuple[str, ...]
    proposer_jail_cards: int
    receiver_jail_cards: int
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposer_name": self.proposer_name,
            "proposer_role": self.proposer_role,
            "receiver_name": self.receiver_name,
            "receiver_role": self.receiver_role,
            "proposer_cash": self.proposer_cash,
            "receiver_cash": self.receiver_cash,
            "proposer_property_names": list(self.proposer_property_names),
            "receiver_property_names": list(self.receiver_property_names),
            "proposer_jail_cards": self.proposer_jail_cards,
            "receiver_jail_cards": self.receiver_jail_cards,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TradeDecisionView:
        return cls(
            proposer_name=str(data["proposer_name"]),
            proposer_role=str(data["proposer_role"]),
            receiver_name=str(data["receiver_name"]),
            receiver_role=str(data["receiver_role"]),
            proposer_cash=int(data["proposer_cash"]),
            receiver_cash=int(data["receiver_cash"]),
            proposer_property_names=tuple(str(item) for item in data["proposer_property_names"]),
            receiver_property_names=tuple(str(item) for item in data["receiver_property_names"]),
            proposer_jail_cards=int(data["proposer_jail_cards"]),
            receiver_jail_cards=int(data["receiver_jail_cards"]),
            note=str(data["note"]),
        )


@dataclass(frozen=True, slots=True)
class PendingActionView:
    """Unified description of the currently blocking interactive decision."""

    action_type: str
    player_name: str
    player_role: str
    turn_phase: str
    prompt: str
    available_actions: tuple[str, ...]
    property_name: str | None = None
    property_index: int | None = None
    price: int | None = None
    auction: AuctionView | None = None
    jail: JailDecisionView | None = None
    property_action: PropertyActionView | None = None
    trade: TradeDecisionView | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "player_name": self.player_name,
            "player_role": self.player_role,
            "turn_phase": self.turn_phase,
            "prompt": self.prompt,
            "available_actions": list(self.available_actions),
            "property_name": self.property_name,
            "property_index": self.property_index,
            "price": self.price,
            "auction": None if self.auction is None else self.auction.to_dict(),
            "jail": None if self.jail is None else self.jail.to_dict(),
            "property_action": None if self.property_action is None else self.property_action.to_dict(),
            "trade": None if self.trade is None else self.trade.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PendingActionView:
        auction_data = data.get("auction")
        jail_data = data.get("jail")
        property_action_data = data.get("property_action")
        trade_data = data.get("trade")
        return cls(
            action_type=str(data["action_type"]),
            player_name=str(data["player_name"]),
            player_role=str(data["player_role"]),
            turn_phase=str(data["turn_phase"]),
            prompt=str(data["prompt"]),
            available_actions=tuple(str(item) for item in data["available_actions"]),
            property_name=_optional_str(data.get("property_name")),
            property_index=_optional_int(data.get("property_index")),
            price=_optional_int(data.get("price")),
            auction=None if auction_data is None else AuctionView.from_dict(auction_data),
            jail=None if jail_data is None else JailDecisionView.from_dict(jail_data),
            property_action=None if property_action_data is None else PropertyActionView.from_dict(property_action_data),
            trade=None if trade_data is None else TradeDecisionView.from_dict(trade_data),
        )


@dataclass(frozen=True, slots=True)
class GameView:
    """High-level board summary used by the frontend and replay system."""

    turn_counter: int
    current_player_name: str | None
    current_player_role: str | None
    current_turn_phase: str | None
    starting_cash: int
    houses_remaining: int
    hotels_remaining: int
    players: tuple[PlayerView, ...]
    pending_action: PendingActionView | None = None
    blocked_trade_offer_signatures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_counter": self.turn_counter,
            "current_player_name": self.current_player_name,
            "current_player_role": self.current_player_role,
            "current_turn_phase": self.current_turn_phase,
            "starting_cash": self.starting_cash,
            "houses_remaining": self.houses_remaining,
            "hotels_remaining": self.hotels_remaining,
            "players": [player.to_dict() for player in self.players],
            "pending_action": None if self.pending_action is None else self.pending_action.to_dict(),
            "blocked_trade_offer_signatures": list(self.blocked_trade_offer_signatures),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GameView:
        pending_action_data = data.get("pending_action")
        return cls(
            turn_counter=int(data["turn_counter"]),
            current_player_name=_optional_str(data.get("current_player_name")),
            current_player_role=_optional_str(data.get("current_player_role")),
            current_turn_phase=_optional_str(data.get("current_turn_phase")),
            starting_cash=int(data["starting_cash"]),
            houses_remaining=int(data["houses_remaining"]),
            hotels_remaining=int(data["hotels_remaining"]),
            players=tuple(PlayerView.from_dict(player) for player in data["players"]),
            pending_action=None if pending_action_data is None else PendingActionView.from_dict(pending_action_data),
            blocked_trade_offer_signatures=tuple(str(item) for item in data.get("blocked_trade_offer_signatures", [])),
        )


@dataclass(frozen=True, slots=True)
class InteractionResult:
    """Serialized result of an executed action or AI step.

    This wrapper keeps the narration emitted by the game engine together with the
    updated `GameView` and any still-pending action so UI and training code can
    react to one completed step atomically.
    """

    messages: tuple[str, ...] = field(default_factory=tuple)
    game_view: GameView | None = None
    pending_action: PendingActionView | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": list(self.messages),
            "game_view": None if self.game_view is None else self.game_view.to_dict(),
            "pending_action": None if self.pending_action is None else self.pending_action.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InteractionResult:
        game_view_data = data.get("game_view")
        pending_action_data = data.get("pending_action")
        return cls(
            messages=tuple(str(item) for item in data.get("messages", [])),
            game_view=None if game_view_data is None else GameView.from_dict(game_view_data),
            pending_action=None if pending_action_data is None else PendingActionView.from_dict(pending_action_data),
        )


@dataclass(frozen=True, slots=True)
class LegalActionOption:
    """One legal action presented to a human player or agent policy.

    The engine emits these objects as the executable action menu for the active
    turn. Frontend code uses the descriptive fields for UI labels, while agent
    code relies on the stable `action_type` and handler metadata.
    """

    action_type: str
    actor_name: str
    actor_role: str
    handler_name: str
    description: str
    property_name: str | None = None
    target_player_name: str | None = None
    fixed_choice: str | None = None
    min_bid: int | None = None
    max_bid: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "actor_name": self.actor_name,
            "actor_role": self.actor_role,
            "handler_name": self.handler_name,
            "description": self.description,
            "property_name": self.property_name,
            "target_player_name": self.target_player_name,
            "fixed_choice": self.fixed_choice,
            "min_bid": self.min_bid,
            "max_bid": self.max_bid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LegalActionOption:
        return cls(
            action_type=str(data["action_type"]),
            actor_name=str(data["actor_name"]),
            actor_role=str(data["actor_role"]),
            handler_name=str(data["handler_name"]),
            description=str(data["description"]),
            property_name=_optional_str(data.get("property_name")),
            target_player_name=_optional_str(data.get("target_player_name")),
            fixed_choice=_optional_str(data.get("fixed_choice")),
            min_bid=_optional_int(data.get("min_bid")),
            max_bid=_optional_int(data.get("max_bid")),
        )


@dataclass(frozen=True, slots=True)
class TurnPlanView:
    """Read-only snapshot of what one player may do right now.

    This is the primary decision surface for interactive play and for the RL
    action space. It identifies whether the player is currently active, whether a
    blocking prompt is present, and the exact legal actions available.
    """

    player_name: str
    player_role: str
    turn_phase: str | None
    is_current_player: bool
    has_pending_action: bool
    pending_action: PendingActionView | None
    legal_actions: tuple[LegalActionOption, ...]
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_name": self.player_name,
            "player_role": self.player_role,
            "turn_phase": self.turn_phase,
            "is_current_player": self.is_current_player,
            "has_pending_action": self.has_pending_action,
            "pending_action": None if self.pending_action is None else self.pending_action.to_dict(),
            "legal_actions": [action.to_dict() for action in self.legal_actions],
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TurnPlanView:
        pending_action_data = data.get("pending_action")
        return cls(
            player_name=str(data["player_name"]),
            player_role=str(data["player_role"]),
            turn_phase=_optional_str(data.get("turn_phase")),
            is_current_player=bool(data["is_current_player"]),
            has_pending_action=bool(data["has_pending_action"]),
            pending_action=None if pending_action_data is None else PendingActionView.from_dict(pending_action_data),
            legal_actions=tuple(LegalActionOption.from_dict(action) for action in data["legal_actions"]),
            reason=_optional_str(data.get("reason")),
        )


@dataclass(frozen=True, slots=True)
class BoardSpaceView:
    """Public description of a board space as rendered in the UI.

    The view flattens both ownable and non-ownable spaces into one serializable
    shape so the frontend can render a uniform board while still surfacing
    property-specific fields like owner, mortgage state, and building count.
    """

    index: int
    name: str
    space_type: str
    occupant_names: tuple[str, ...]
    owner_name: str | None = None
    color_group: str | None = None
    mortgaged: bool = False
    building_count: int | None = None
    price: int | None = None
    house_cost: int | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "space_type": self.space_type,
            "occupant_names": list(self.occupant_names),
            "owner_name": self.owner_name,
            "color_group": self.color_group,
            "mortgaged": self.mortgaged,
            "building_count": self.building_count,
            "price": self.price,
            "house_cost": self.house_cost,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BoardSpaceView:
        return cls(
            index=int(data["index"]),
            name=str(data["name"]),
            space_type=str(data["space_type"]),
            occupant_names=tuple(str(item) for item in data["occupant_names"]),
            owner_name=_optional_str(data.get("owner_name")),
            color_group=_optional_str(data.get("color_group")),
            mortgaged=bool(data.get("mortgaged", False)),
            building_count=_optional_int(data.get("building_count")),
            price=_optional_int(data.get("price")),
            house_cost=_optional_int(data.get("house_cost")),
            notes=_optional_str(data.get("notes")),
        )


@dataclass(frozen=True, slots=True)
class AIPlayerSetup:
    """Configuration override for one AI-controlled seat.

    The GUI and online host flows use this object to attach a checkpoint or
    scripted variant plus an action cooldown to a specific player name.
    """

    player_name: str
    checkpoint_path: str | None = None
    action_cooldown_seconds: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_name": self.player_name,
            "checkpoint_path": self.checkpoint_path,
            "action_cooldown_seconds": self.action_cooldown_seconds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AIPlayerSetup:
        return cls(
            player_name=str(data["player_name"]),
            checkpoint_path=_optional_str(data.get("checkpoint_path")),
            action_cooldown_seconds=float(data.get("action_cooldown_seconds", 2.0)),
        )


@dataclass(frozen=True, slots=True)
class GameSetup:
    """Serialized game-creation payload shared by local and hosted games."""

    player_names: tuple[str, ...]
    starting_cash: int
    player_roles: tuple[str, ...] | None = None
    ai_checkpoint_path: str | None = None
    ai_player_setups: tuple[AIPlayerSetup, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_names": list(self.player_names),
            "starting_cash": self.starting_cash,
            "player_roles": list(self.player_roles) if self.player_roles is not None else None,
            "ai_checkpoint_path": self.ai_checkpoint_path,
            "ai_player_setups": [item.to_dict() for item in self.ai_player_setups] if self.ai_player_setups is not None else None,
        }

    def resolved_player_roles(self) -> tuple[str, ...]:
        """Return explicit roles, defaulting omitted setups to all-human play."""
        if self.player_roles is None:
            return tuple(HUMAN_ROLE for _ in self.player_names)
        return self.player_roles

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GameSetup:
        player_roles = data.get("player_roles")
        ai_player_setups = data.get("ai_player_setups")
        return cls(
            player_names=tuple(str(item) for item in data["player_names"]),
            starting_cash=int(data["starting_cash"]),
            player_roles=None if player_roles is None else tuple(str(item) for item in player_roles),
            ai_checkpoint_path=_optional_str(data.get("ai_checkpoint_path")),
            ai_player_setups=None if ai_player_setups is None else tuple(AIPlayerSetup.from_dict(item) for item in ai_player_setups),
        )


@dataclass(frozen=True, slots=True)
class OnlineSeatView:
    """Public state for one seat in an online lobby or hosted match."""

    seat_index: int
    status: str
    player_name: str | None = None
    player_role: str = HUMAN_ROLE
    is_host: bool = False
    is_claimable: bool = False
    checkpoint_path: str | None = None
    action_cooldown_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat_index": self.seat_index,
            "status": self.status,
            "player_name": self.player_name,
            "player_role": self.player_role,
            "is_host": self.is_host,
            "is_claimable": self.is_claimable,
            "checkpoint_path": self.checkpoint_path,
            "action_cooldown_seconds": self.action_cooldown_seconds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OnlineSeatView:
        cooldown = data.get("action_cooldown_seconds")
        return cls(
            seat_index=int(data["seat_index"]),
            status=str(data["status"]),
            player_name=_optional_str(data.get("player_name")),
            player_role=str(data.get("player_role", HUMAN_ROLE)),
            is_host=bool(data.get("is_host", False)),
            is_claimable=bool(data.get("is_claimable", False)),
            checkpoint_path=_optional_str(data.get("checkpoint_path")),
            action_cooldown_seconds=None if cooldown is None else float(cooldown),
        )


@dataclass(frozen=True, slots=True)
class OnlineSessionView:
    """Serializable snapshot of the host-authoritative online session state."""

    session_code: str
    state: str
    host_player_name: str
    seat_count: int
    starting_cash: int
    seats: tuple[OnlineSeatView, ...]
    paused_reason: str | None = None
    paused_seat_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_code": self.session_code,
            "state": self.state,
            "host_player_name": self.host_player_name,
            "seat_count": self.seat_count,
            "starting_cash": self.starting_cash,
            "seats": [seat.to_dict() for seat in self.seats],
            "paused_reason": self.paused_reason,
            "paused_seat_index": self.paused_seat_index,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OnlineSessionView:
        return cls(
            session_code=str(data["session_code"]),
            state=str(data["state"]),
            host_player_name=str(data["host_player_name"]),
            seat_count=int(data["seat_count"]),
            starting_cash=int(data["starting_cash"]),
            seats=tuple(OnlineSeatView.from_dict(item) for item in data.get("seats", [])),
            paused_reason=_optional_str(data.get("paused_reason")),
            paused_seat_index=_optional_int(data.get("paused_seat_index")),
        )


@dataclass(frozen=True, slots=True)
class FrontendStateView:
    """Top-level UI snapshot combining board, game summary, and active turn plan."""

    game_view: GameView
    active_turn_plan: TurnPlanView
    board_spaces: tuple[BoardSpaceView, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_view": self.game_view.to_dict(),
            "active_turn_plan": self.active_turn_plan.to_dict(),
            "board_spaces": [space.to_dict() for space in self.board_spaces],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrontendStateView:
        return cls(
            game_view=GameView.from_dict(data["game_view"]),
            active_turn_plan=TurnPlanView.from_dict(data["active_turn_plan"]),
            board_spaces=tuple(BoardSpaceView.from_dict(space) for space in data["board_spaces"]),
        )


def _optional_str(value: Any) -> str | None:
    """Normalize an optional serialized value into `str | None`."""
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    """Normalize an optional serialized value into `int | None`."""
    if value is None:
        return None
    return int(value)
