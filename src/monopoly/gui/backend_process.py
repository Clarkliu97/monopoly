from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import secrets
from typing import Any, Callable, Mapping

from monopoly.agent import GameProcessAgentHost, build_scripted_controller, default_scripted_profiles, load_agent_host_from_checkpoint, resolve_checkpoint_path
from monopoly.api import AIPlayerSetup, GameSetup, OnlineSeatView, OnlineSessionView
from monopoly.constants import AI_ROLE, HUMAN_ROLE, MAX_PLAYERS, MIN_PLAYERS, STARTING_CASH
from monopoly.game import Game
from monopoly.logging_utils import configure_process_logging
from monopoly.gui.transport import serve_socket_requests


ONLINE_SESSION_STATE_LOBBY = "lobby"
ONLINE_SESSION_STATE_IN_GAME = "in_game"
ONLINE_SESSION_STATE_PAUSED = "paused"

ONLINE_SEAT_STATUS_HOST = "host"
ONLINE_SEAT_STATUS_OPEN = "open"
ONLINE_SEAT_STATUS_CLOSED = "closed"
ONLINE_SEAT_STATUS_CONNECTED = "connected"
ONLINE_SEAT_STATUS_DISCONNECTED = "disconnected"
ONLINE_SEAT_STATUS_AI = "ai"
SCRIPTED_AI_SELECTION_PREFIX = "scripted:"


logger = logging.getLogger("monopoly.gui.backend")


@dataclass(slots=True)
class _OnlineSeatState:
    seat_index: int
    status: str
    player_name: str | None = None
    player_role: str = HUMAN_ROLE
    is_host: bool = False
    checkpoint_path: str | None = None
    action_cooldown_seconds: float | None = None
    session_token: str | None = None
    reconnect_token: str | None = None

    def to_view(self) -> OnlineSeatView:
        return OnlineSeatView(
            seat_index=self.seat_index,
            status=self.status,
            player_name=self.player_name,
            player_role=self.player_role,
            is_host=self.is_host,
            is_claimable=self.status == ONLINE_SEAT_STATUS_OPEN,
            checkpoint_path=self.checkpoint_path,
            action_cooldown_seconds=self.action_cooldown_seconds,
        )


@dataclass(slots=True)
class _OnlineSessionState:
    session_code: str
    state: str
    host_player_name: str
    starting_cash: int
    seats: list[_OnlineSeatState]
    ai_checkpoint_path: str | None = None
    paused_reason: str | None = None
    paused_seat_index: int | None = None

    def to_view(self) -> OnlineSessionView:
        return OnlineSessionView(
            session_code=self.session_code,
            state=self.state,
            host_player_name=self.host_player_name,
            seat_count=len(self.seats),
            starting_cash=self.starting_cash,
            seats=tuple(seat.to_view() for seat in self.seats),
            paused_reason=self.paused_reason,
            paused_seat_index=self.paused_seat_index,
        )


class BackendRuntime:
    def __init__(
        self,
        debug_enabled: bool = False,
        default_ai_checkpoint_path: str | None = None,
        ai_device: str = "cpu",
        max_ai_actions_per_turn: int = 24,
    ) -> None:
        self.game: Game | None = None
        self.debug_enabled = debug_enabled
        self.default_ai_checkpoint_path = default_ai_checkpoint_path
        self.ai_device = ai_device
        self.ai_checkpoint_path: str | None = None
        self.ai_checkpoint_paths_by_player: dict[str, str] = {}
        self.ai_cooldowns_by_player: dict[str, float] = {}
        self._ai_hosts_by_player: dict[str, Any] = {}
        self.max_ai_actions_per_turn = max_ai_actions_per_turn
        self._ai_turn_guard_signature: tuple[int, str, str | None] | None = None
        self._ai_turn_guard_actions = 0
        self.online_session: _OnlineSessionState | None = None
        self._event_publisher: Callable[[str, Mapping[str, Any]], None] | None = None

    def set_event_publisher(self, publisher: Callable[[str, Mapping[str, Any]], None] | None) -> None:
        self._event_publisher = publisher

    def handle_command(self, command: Mapping[str, Any]) -> dict[str, Any]:
        action = str(command.get("command", "<missing>"))
        try:
            logger.debug("Handling backend command %s", action)
            action = str(command["command"])
            if action == "create_online_lobby":
                return self._handle_create_online_lobby(command)
            if action == "get_online_session":
                return self._handle_get_online_session(command)
            if action == "open_online_slot":
                return self._handle_open_online_slot(command)
            if action == "close_online_slot":
                return self._handle_close_online_slot(command)
            if action == "assign_ai_to_online_slot":
                return self._handle_assign_ai_to_online_slot(command)
            if action == "clear_online_slot":
                return self._handle_clear_online_slot(command)
            if action == "claim_online_slot":
                return self._handle_claim_online_slot(command)
            if action == "disconnect_online_slot":
                return self._handle_disconnect_online_slot(command)
            if action == "reconnect_online_slot":
                return self._handle_reconnect_online_slot(command)
            if action == "replace_disconnected_online_slot_with_ai":
                return self._handle_replace_disconnected_online_slot_with_ai(command)
            if action == "start_online_game":
                return self._handle_start_online_game(command)
            if action == "create_game":
                return self._handle_create_game(command)
            if action == "get_state":
                return self._ok_response(frontend_state=self._require_game().get_serialized_frontend_state())
            if action == "execute_action":
                return self._handle_execute_action(command)
            if action == "step_ai":
                return self._handle_step_ai(command)
            if action == "save_game":
                return self._handle_save_game(command)
            if action == "load_game":
                return self._handle_load_game(command)
            if action == "get_debug_state":
                return self._handle_get_debug_state()
            if action == "apply_debug_state":
                return self._handle_apply_debug_state(command)
            if action == "shutdown":
                logger.info("Backend shutdown requested.")
                return self._ok_response(shutting_down=True)
            raise ValueError(f"Unsupported backend command: {action}")
        except Exception as exc:
            logger.exception("Backend command %s failed: %s", action, exc)
            return {"ok": False, "error": str(exc)}

    def _handle_create_game(self, command: Mapping[str, Any]) -> dict[str, Any]:
        self.online_session = None
        setup = GameSetup.from_dict(command["setup"])
        logger.info("Creating local game with %d players and starting cash %d.", len(setup.player_names), setup.starting_cash)
        self.game = Game(
            player_names=list(setup.player_names),
            player_roles=list(setup.resolved_player_roles()),
            starting_cash=setup.starting_cash,
        )
        self._configure_ai_runtime(setup)
        return self._ok_response(
            frontend_state=self.game.get_serialized_frontend_state(),
            game_setup=self.game.get_serialized_game_setup(),
        )

    def _handle_execute_action(self, command: Mapping[str, Any]) -> dict[str, Any]:
        game = self._require_game()
        self._validate_online_action_access(command)
        interaction = game.execute_serialized_action(
            action_payload=command["action"],
            bid_amount=None if command.get("bid_amount") is None else int(command["bid_amount"]),
            trade_offer_payload=command.get("trade_offer"),
        )
        if self.online_session is not None and self.online_session.state in {ONLINE_SESSION_STATE_IN_GAME, ONLINE_SESSION_STATE_PAUSED}:
            self._broadcast_online_session(self.online_session, include_game_state=True)
        return self._ok_response(
            interaction=interaction,
            frontend_state=game.get_serialized_frontend_state(),
            game_setup=game.get_serialized_game_setup(),
        )

    def _handle_step_ai(self, command: Mapping[str, Any] | None = None) -> dict[str, Any]:
        game = self._require_game()
        if self.online_session is not None and self.online_session.state == ONLINE_SESSION_STATE_PAUSED:
            raise ValueError("Online gameplay is paused while waiting for a disconnected player to reconnect or be replaced.")
        if self.online_session is not None and self.online_session.state == ONLINE_SESSION_STATE_IN_GAME:
            self._require_online_host_token(command)
        active_turn_plan = game.get_active_turn_plan()
        if active_turn_plan.player_role != AI_ROLE:
            raise ValueError("The active turn does not belong to an AI player.")
        logger.debug("Stepping AI turn for %s in phase %s.", active_turn_plan.player_name, active_turn_plan.turn_phase)
        ai_host = self._ai_hosts_by_player.get(active_turn_plan.player_name)
        if ai_host is None:
            raise ValueError(f"No AI controller is configured for {active_turn_plan.player_name}.")

        turn_signature = (game.turn_counter, active_turn_plan.player_name, active_turn_plan.turn_phase)
        action_budget_exceeded = self._advance_ai_turn_guard(turn_signature) > self.max_ai_actions_per_turn
        if action_budget_exceeded:
            fallback_action = self._select_ai_fallback_action(active_turn_plan)
            interaction = game.execute_legal_action(fallback_action)
        else:
            decision = ai_host.controller.choose_action(game, active_turn_plan.player_name, explore=False)
            trade_offer = None if decision.choice.trade_offer_payload is None else game.deserialize_trade_offer(decision.choice.trade_offer_payload)
            interaction = game.execute_legal_action(
                decision.choice.legal_action,
                bid_amount=decision.choice.bid_amount,
                trade_offer=trade_offer,
            )
        if self.online_session is not None and self.online_session.state in {ONLINE_SESSION_STATE_IN_GAME, ONLINE_SESSION_STATE_PAUSED}:
            self._broadcast_online_session(self.online_session, include_game_state=True)
        return self._ok_response(
            actor_name=active_turn_plan.player_name,
            cooldown_seconds=self.ai_cooldowns_by_player.get(active_turn_plan.player_name, 2.0),
            interaction=interaction.to_dict(),
            frontend_state=game.get_serialized_frontend_state(),
            game_setup=game.get_serialized_game_setup(),
        )

    def _handle_save_game(self, command: Mapping[str, Any]) -> dict[str, Any]:
        game = self._require_game()
        file_path = str(command["file_path"])
        game.save_to_file(file_path)
        logger.info("Saved game state to %s.", file_path)
        return self._ok_response(file_path=file_path)

    def _handle_load_game(self, command: Mapping[str, Any]) -> dict[str, Any]:
        file_path = str(command["file_path"])
        self.online_session = None
        self.game = Game.load_from_file(file_path)
        self._configure_ai_runtime(None)
        logger.info("Loaded game state from %s.", file_path)
        return self._ok_response(
            file_path=file_path,
            frontend_state=self.game.get_serialized_frontend_state(),
            game_setup=self.game.get_serialized_game_setup(),
        )

    def _handle_get_debug_state(self) -> dict[str, Any]:
        self._require_debug_enabled()
        return self._ok_response(full_state=self._require_game().serialize_full_state())

    def _handle_apply_debug_state(self, command: Mapping[str, Any]) -> dict[str, Any]:
        self._require_debug_enabled()
        full_state = command.get("full_state")
        if not isinstance(full_state, Mapping):
            raise ValueError("Debug state payload must include a full_state object.")
        self.online_session = None
        self.game = Game.from_serialized_state(full_state)
        self._configure_ai_runtime(None)
        return self._ok_response(
            frontend_state=self.game.get_serialized_frontend_state(),
            game_setup=self.game.get_serialized_game_setup(),
            full_state=self.game.serialize_full_state(),
        )

    def _handle_create_online_lobby(self, command: Mapping[str, Any]) -> dict[str, Any]:
        host_player_name = str(command["host_player_name"]).strip()
        if not host_player_name:
            raise ValueError("Host player name cannot be empty.")
        seat_count = int(command.get("seat_count", MIN_PLAYERS))
        if not MIN_PLAYERS <= seat_count <= MAX_PLAYERS:
            raise ValueError(f"Online lobbies support between {MIN_PLAYERS} and {MAX_PLAYERS} seats.")
        starting_cash = int(command.get("starting_cash", STARTING_CASH))
        if starting_cash <= 0:
            raise ValueError("Starting cash must be positive.")

        host_seat = _OnlineSeatState(
            seat_index=0,
            status=ONLINE_SEAT_STATUS_HOST,
            player_name=host_player_name,
            player_role=HUMAN_ROLE,
            is_host=True,
            session_token=self._generate_token(),
            reconnect_token=self._generate_token(),
        )
        seats = [host_seat]
        for seat_index in range(1, seat_count):
            seats.append(_OnlineSeatState(seat_index=seat_index, status=ONLINE_SEAT_STATUS_OPEN))

        self.game = None
        self._reset_ai_turn_guard()
        self.online_session = _OnlineSessionState(
            session_code=self._generate_session_code(),
            state=ONLINE_SESSION_STATE_LOBBY,
            host_player_name=host_player_name,
            starting_cash=starting_cash,
            seats=seats,
            ai_checkpoint_path=None if command.get("ai_checkpoint_path") is None else str(command.get("ai_checkpoint_path")),
        )
        logger.info("Created online lobby %s for host %s with %d seats.", self.online_session.session_code, host_player_name, seat_count)
        self._broadcast_online_session(self.online_session)
        return self._session_response(host_seat)

    def _handle_get_online_session(self, command: Mapping[str, Any]) -> dict[str, Any]:
        session = self._require_online_session(command.get("session_code"))
        seat = self._seat_for_session_token(command.get("session_token"))
        return self._session_response(
            seat,
            session=session,
            include_game_state=session.state in {ONLINE_SESSION_STATE_IN_GAME, ONLINE_SESSION_STATE_PAUSED} and self.game is not None,
        )

    def _handle_open_online_slot(self, command: Mapping[str, Any]) -> dict[str, Any]:
        session = self._require_online_host_token(command)
        seat = self._editable_online_seat(command, session)
        if seat.status not in {ONLINE_SEAT_STATUS_CLOSED, ONLINE_SEAT_STATUS_OPEN}:
            raise ValueError("Only closed slots can be opened.")
        self._reset_seat_to_open(seat)
        logger.info("Opened online seat %d in lobby %s.", seat.seat_index, session.session_code)
        self._broadcast_online_session(session)
        return self._session_response(self._host_seat(), session=session)

    def _handle_close_online_slot(self, command: Mapping[str, Any]) -> dict[str, Any]:
        session = self._require_online_host_token(command)
        seat = self._editable_online_seat(command, session)
        if seat.status not in {ONLINE_SEAT_STATUS_OPEN, ONLINE_SEAT_STATUS_CLOSED, ONLINE_SEAT_STATUS_DISCONNECTED}:
            raise ValueError("Only open or disconnected slots can be closed.")
        self._reset_seat_to_closed(seat)
        logger.info("Closed online seat %d in lobby %s.", seat.seat_index, session.session_code)
        self._broadcast_online_session(session)
        return self._session_response(self._host_seat(), session=session)

    def _handle_assign_ai_to_online_slot(self, command: Mapping[str, Any]) -> dict[str, Any]:
        session = self._require_online_host_token(command)
        seat = self._editable_online_seat(command, session)
        if seat.status in {ONLINE_SEAT_STATUS_CONNECTED, ONLINE_SEAT_STATUS_HOST}:
            raise ValueError("Cannot replace an occupied human seat with AI.")
        player_name = str(command.get("player_name") or f"AI Player {seat.seat_index + 1}").strip()
        if not player_name:
            raise ValueError("AI player name cannot be empty.")
        seat.status = ONLINE_SEAT_STATUS_AI
        seat.player_name = player_name
        seat.player_role = AI_ROLE
        seat.checkpoint_path = None if command.get("checkpoint_path") is None else str(command.get("checkpoint_path"))
        seat.action_cooldown_seconds = float(command.get("action_cooldown_seconds", 2.0))
        seat.session_token = None
        seat.reconnect_token = None
        logger.info("Assigned AI %s to online seat %d in lobby %s.", player_name, seat.seat_index, session.session_code)
        self._broadcast_online_session(session)
        return self._session_response(self._host_seat(), session=session)

    def _handle_clear_online_slot(self, command: Mapping[str, Any]) -> dict[str, Any]:
        session = self._require_online_host_token(command)
        seat = self._editable_online_seat(command, session)
        if seat.status not in {ONLINE_SEAT_STATUS_AI, ONLINE_SEAT_STATUS_DISCONNECTED, ONLINE_SEAT_STATUS_OPEN, ONLINE_SEAT_STATUS_CLOSED}:
            raise ValueError("Only AI, open, closed, or disconnected slots can be cleared.")
        self._reset_seat_to_open(seat)
        self._broadcast_online_session(session)
        return self._session_response(self._host_seat(), session=session)

    def _handle_claim_online_slot(self, command: Mapping[str, Any]) -> dict[str, Any]:
        session = self._require_online_session(command.get("session_code"))
        self._require_online_lobby_state(session)
        seat = self._seat_from_command(command, session)
        if seat.status != ONLINE_SEAT_STATUS_OPEN:
            raise ValueError("Only open untaken human seats can be claimed.")
        player_name = str(command["player_name"]).strip()
        if not player_name:
            raise ValueError("Player name cannot be empty.")
        if self._is_online_player_name_taken(player_name, exclude_index=seat.seat_index):
            raise ValueError(f"Player name {player_name} is already taken in this lobby.")
        seat.status = ONLINE_SEAT_STATUS_CONNECTED
        seat.player_name = player_name
        seat.player_role = HUMAN_ROLE
        seat.session_token = self._generate_token()
        seat.reconnect_token = self._generate_token()
        seat.checkpoint_path = None
        seat.action_cooldown_seconds = None
        logger.info("Player %s claimed online seat %d in lobby %s.", player_name, seat.seat_index, session.session_code)
        self._broadcast_online_session(session)
        return self._session_response(seat, session=session)

    def _handle_disconnect_online_slot(self, command: Mapping[str, Any]) -> dict[str, Any]:
        session = self._require_online_session(command.get("session_code"))
        seat = self._require_seat_session_token(command.get("session_token"))
        if seat.is_host:
            raise ValueError("Host seat cannot be disconnected through the player disconnect command.")
        if seat.status != ONLINE_SEAT_STATUS_CONNECTED:
            raise ValueError("Only connected player seats can be disconnected.")
        seat.status = ONLINE_SEAT_STATUS_DISCONNECTED
        seat.session_token = None
        if session.state == ONLINE_SESSION_STATE_IN_GAME:
            self._pause_online_session_for_disconnect(session, seat)
        logger.warning("Player %s disconnected from online seat %d in lobby %s.", seat.player_name, seat.seat_index, session.session_code)
        self._broadcast_online_session(session, include_game_state=self.game is not None)
        return self._session_response(None, session=session)

    def _handle_reconnect_online_slot(self, command: Mapping[str, Any]) -> dict[str, Any]:
        session = self._require_online_session(command.get("session_code"))
        seat = self._require_seat_reconnect_token(command.get("reconnect_token"))
        if seat.status not in {ONLINE_SEAT_STATUS_DISCONNECTED, ONLINE_SEAT_STATUS_CONNECTED, ONLINE_SEAT_STATUS_HOST}:
            raise ValueError("This seat cannot be reconnected.")
        seat.status = ONLINE_SEAT_STATUS_HOST if seat.is_host else ONLINE_SEAT_STATUS_CONNECTED
        seat.session_token = self._generate_token()
        self._resume_online_session_if_ready(session)
        logger.info("Player %s reconnected to online seat %d in lobby %s.", seat.player_name, seat.seat_index, session.session_code)
        self._broadcast_online_session(session, include_game_state=self.game is not None)
        return self._session_response(
            seat,
            session=session,
            include_game_state=session.state in {ONLINE_SESSION_STATE_IN_GAME, ONLINE_SESSION_STATE_PAUSED} and self.game is not None,
        )

    def _handle_replace_disconnected_online_slot_with_ai(self, command: Mapping[str, Any]) -> dict[str, Any]:
        session = self._require_online_host_token(command)
        if session.state not in {ONLINE_SESSION_STATE_IN_GAME, ONLINE_SESSION_STATE_PAUSED}:
            raise ValueError("Disconnected seats can only be replaced after the online game has started.")
        seat = self._seat_from_command(command, session)
        if seat.is_host:
            raise ValueError("The host seat cannot be replaced with AI.")
        if seat.status != ONLINE_SEAT_STATUS_DISCONNECTED:
            raise ValueError("Only disconnected player seats can be replaced with AI.")
        if seat.player_name is None:
            raise ValueError("Disconnected seat is missing its player identity.")

        checkpoint_path = session.ai_checkpoint_path if command.get("checkpoint_path") is None else str(command.get("checkpoint_path"))
        action_cooldown_seconds = float(command.get("action_cooldown_seconds", 2.0))
        seat.status = ONLINE_SEAT_STATUS_AI
        seat.player_role = AI_ROLE
        seat.checkpoint_path = checkpoint_path
        seat.action_cooldown_seconds = action_cooldown_seconds
        seat.session_token = None
        seat.reconnect_token = None

        player = self._player_by_name(seat.player_name)
        player.role = AI_ROLE
        self._configure_ai_player(seat.player_name, checkpoint_path=checkpoint_path, action_cooldown_seconds=action_cooldown_seconds)
        self._resume_online_session_if_ready(session)
        logger.warning("Replaced disconnected seat %d with AI player %s in lobby %s.", seat.seat_index, seat.player_name, session.session_code)
        self._broadcast_online_session(session, include_game_state=self.game is not None)
        return self._session_response(self._host_seat(), session=session, include_game_state=self.game is not None)

    def _handle_start_online_game(self, command: Mapping[str, Any]) -> dict[str, Any]:
        session = self._require_online_host_token(command)
        self._require_online_lobby_state(session)
        included_seats = [seat for seat in session.seats if seat.status in {ONLINE_SEAT_STATUS_HOST, ONLINE_SEAT_STATUS_CONNECTED, ONLINE_SEAT_STATUS_AI}]
        if len(included_seats) < MIN_PLAYERS:
            raise ValueError(f"At least {MIN_PLAYERS} occupied or AI seats are required to start an online game.")
        if any(seat.status == ONLINE_SEAT_STATUS_DISCONNECTED for seat in session.seats):
            raise ValueError("Disconnected players must reconnect or be removed before the host can start the game.")

        player_names = [seat.player_name for seat in included_seats if seat.player_name is not None]
        player_roles = [seat.player_role for seat in included_seats]
        ai_player_setups = [
            AIPlayerSetup(
                player_name=seat.player_name or f"AI Player {seat.seat_index + 1}",
                checkpoint_path=seat.checkpoint_path,
                action_cooldown_seconds=2.0 if seat.action_cooldown_seconds is None else seat.action_cooldown_seconds,
            )
            for seat in included_seats
            if seat.player_role == AI_ROLE
        ]
        setup = GameSetup(
            player_names=tuple(player_names),
            starting_cash=session.starting_cash,
            player_roles=tuple(player_roles),
            ai_checkpoint_path=session.ai_checkpoint_path,
            ai_player_setups=tuple(ai_player_setups) if ai_player_setups else None,
        )
        self.game = Game(
            player_names=list(setup.player_names),
            player_roles=list(setup.resolved_player_roles()),
            starting_cash=setup.starting_cash,
        )
        self._configure_ai_runtime(setup)
        session.state = ONLINE_SESSION_STATE_IN_GAME
        session.paused_reason = None
        session.paused_seat_index = None
        logger.info("Started online game for lobby %s with %d players.", session.session_code, len(player_names))
        self._broadcast_online_session(session, include_game_state=True)
        return self._session_response(self._host_seat(), session=session, include_game_state=True)

    def _configure_ai_runtime(self, setup: GameSetup | None) -> None:
        game = self._require_game()
        has_ai_players = any(player.role == AI_ROLE for player in game.players)
        if not has_ai_players:
            self.ai_checkpoint_path = None
            self.ai_checkpoint_paths_by_player = {}
            self.ai_cooldowns_by_player = {}
            self._ai_hosts_by_player = {}
            self._reset_ai_turn_guard()
            return

        setup_by_player = self._ai_setup_by_player(setup)
        host_cache: dict[str, Any] = {}
        ai_checkpoint_paths_by_player: dict[str, str] = {}
        ai_cooldowns_by_player: dict[str, float] = {}
        ai_hosts_by_player: dict[str, Any] = {}

        for player in game.players:
            if player.role != AI_ROLE:
                continue
            ai_setup = setup_by_player.get(player.name)
            requested_selection = None if ai_setup is None else ai_setup.checkpoint_path
            normalized_selection = self._normalize_ai_selection(
                requested_selection or (None if setup is None else setup.ai_checkpoint_path) or self.default_ai_checkpoint_path,
            )
            ai_hosts_by_player[player.name] = self._load_ai_host(normalized_selection, player_name=player.name, checkpoint_host_cache=host_cache)
            ai_checkpoint_paths_by_player[player.name] = normalized_selection
            ai_cooldowns_by_player[player.name] = max(0.0, 2.0 if ai_setup is None else ai_setup.action_cooldown_seconds)

        self.ai_checkpoint_path = next(iter(ai_checkpoint_paths_by_player.values()), None)
        self.ai_checkpoint_paths_by_player = ai_checkpoint_paths_by_player
        self.ai_cooldowns_by_player = ai_cooldowns_by_player
        self._ai_hosts_by_player = ai_hosts_by_player
        self._reset_ai_turn_guard()

    def _configure_ai_player(self, player_name: str, *, checkpoint_path: str | None, action_cooldown_seconds: float) -> None:
        resolved_path_str = self._normalize_ai_selection(checkpoint_path or self.default_ai_checkpoint_path)
        ai_host = self._ai_hosts_by_player.get(player_name)
        if ai_host is None or self.ai_checkpoint_paths_by_player.get(player_name) != resolved_path_str:
            ai_host = self._load_ai_host(resolved_path_str, player_name=player_name)
        self._ai_hosts_by_player[player_name] = ai_host
        self.ai_checkpoint_paths_by_player[player_name] = resolved_path_str
        self.ai_cooldowns_by_player[player_name] = max(0.0, action_cooldown_seconds)
        self.ai_checkpoint_path = next(iter(self.ai_checkpoint_paths_by_player.values()), resolved_path_str)
        self._reset_ai_turn_guard()

    def _normalize_ai_selection(self, selection: str | None) -> str:
        if selection is None:
            raise FileNotFoundError("AI checkpoint could not be resolved.")
        scripted_variant = self._scripted_variant_from_selection(selection)
        if scripted_variant is not None:
            return f"{SCRIPTED_AI_SELECTION_PREFIX}{scripted_variant}"
        resolved_path = resolve_checkpoint_path(selection, require_exists=True)
        if resolved_path is None:
            raise FileNotFoundError("AI checkpoint could not be resolved.")
        return str(resolved_path)

    def _load_ai_host(
        self,
        selection: str,
        *,
        player_name: str,
        checkpoint_host_cache: dict[str, Any] | None = None,
    ) -> Any:
        scripted_variant = self._scripted_variant_from_selection(selection)
        if scripted_variant is not None:
            seed = sum(ord(character) for character in player_name) or 7
            return GameProcessAgentHost(build_scripted_controller(scripted_variant, seed=seed))

        resolved_path = resolve_checkpoint_path(selection, require_exists=True)
        if resolved_path is None:
            raise FileNotFoundError("AI checkpoint could not be resolved.")
        resolved_path_str = str(resolved_path)
        if checkpoint_host_cache is not None:
            cached_host = checkpoint_host_cache.get(resolved_path_str)
            if cached_host is None:
                cached_host = load_agent_host_from_checkpoint(resolved_path, device=self.ai_device)
                checkpoint_host_cache[resolved_path_str] = cached_host
            return cached_host
        return load_agent_host_from_checkpoint(resolved_path, device=self.ai_device)

    @staticmethod
    def _scripted_variant_from_selection(selection: str | None) -> str | None:
        if selection is None:
            return None
        normalized = str(selection).strip()
        if not normalized.startswith(SCRIPTED_AI_SELECTION_PREFIX):
            return None
        variant = normalized.removeprefix(SCRIPTED_AI_SELECTION_PREFIX).strip()
        if variant not in default_scripted_profiles():
            raise ValueError(f"Unsupported scripted opponent variant: {variant}")
        return variant

    @staticmethod
    def _ai_setup_by_player(setup: GameSetup | None) -> dict[str, AIPlayerSetup]:
        if setup is None or setup.ai_player_setups is None:
            return {}
        return {item.player_name: item for item in setup.ai_player_setups}

    def _require_game(self) -> Game:
        if self.game is None:
            raise ValueError("No game has been created yet.")
        return self.game

    def _require_online_session(self, session_code: Any = None) -> _OnlineSessionState:
        if self.online_session is None:
            raise ValueError("No online lobby has been created yet.")
        if session_code is not None and str(session_code) != self.online_session.session_code:
            raise ValueError("Online session code does not match the active lobby.")
        return self.online_session

    def _require_online_lobby_state(self, session: _OnlineSessionState) -> None:
        if session.state != ONLINE_SESSION_STATE_LOBBY:
            raise ValueError("This command is only available while the online lobby is waiting to start.")

    def _require_online_host_token(self, command: Mapping[str, Any] | None) -> _OnlineSessionState:
        session = self._require_online_session(None if command is None else command.get("session_code"))
        if session.state == ONLINE_SESSION_STATE_IN_GAME and command is None:
            return session
        if self.online_session is None:
            return session
        token = None if command is None else command.get("session_token")
        seat = self._require_seat_session_token(token)
        if not seat.is_host:
            raise ValueError("Only the host can perform this online action.")
        return session

    def _require_seat_session_token(self, token: Any) -> _OnlineSeatState:
        seat = self._seat_for_session_token(token)
        if seat is None:
            raise ValueError("A valid online session token is required for this action.")
        return seat

    def _require_seat_reconnect_token(self, token: Any) -> _OnlineSeatState:
        session = self._require_online_session()
        for seat in session.seats:
            if seat.reconnect_token is not None and seat.reconnect_token == token:
                return seat
        raise ValueError("A valid reconnect token is required for this action.")

    def _seat_for_session_token(self, token: Any) -> _OnlineSeatState | None:
        if token is None or self.online_session is None:
            return None
        token_str = str(token)
        for seat in self.online_session.seats:
            if seat.session_token == token_str:
                return seat
        return None

    def _seat_from_command(self, command: Mapping[str, Any], session: _OnlineSessionState) -> _OnlineSeatState:
        seat_index = int(command["seat_index"])
        if not 0 <= seat_index < len(session.seats):
            raise ValueError(f"Seat index {seat_index} is out of range.")
        return session.seats[seat_index]

    def _editable_online_seat(self, command: Mapping[str, Any], session: _OnlineSessionState) -> _OnlineSeatState:
        self._require_online_lobby_state(session)
        seat = self._seat_from_command(command, session)
        if seat.is_host:
            raise ValueError("The host seat cannot be edited through slot controls.")
        return seat

    def _host_seat(self) -> _OnlineSeatState:
        session = self._require_online_session()
        for seat in session.seats:
            if seat.is_host:
                return seat
        raise ValueError("Online lobby is missing its host seat.")

    def _validate_online_action_access(self, command: Mapping[str, Any]) -> None:
        if self.online_session is None:
            return
        if self.online_session.state == ONLINE_SESSION_STATE_PAUSED:
            raise ValueError("Online gameplay is paused while waiting for a disconnected player to reconnect or be replaced.")
        if self.online_session.state != ONLINE_SESSION_STATE_IN_GAME:
            return
        seat = self._require_seat_session_token(command.get("session_token"))
        action_payload = command.get("action")
        if not isinstance(action_payload, Mapping):
            raise ValueError("Online gameplay actions must include an action payload.")
        actor_name = str(action_payload.get("actor_name"))
        if seat.player_name != actor_name:
            raise ValueError(f"This connection does not control {actor_name}.")
        if seat.player_role != HUMAN_ROLE:
            raise ValueError("AI-controlled seats cannot submit direct gameplay actions.")

    def _broadcast_online_session(self, session: _OnlineSessionState, *, include_game_state: bool = False) -> None:
        if self._event_publisher is None:
            return
        payload: dict[str, Any] = {"online_session": session.to_view().to_dict()}
        if include_game_state and self.game is not None:
            payload["frontend_state"] = self.game.get_serialized_frontend_state()
            payload["game_setup"] = self.game.get_serialized_game_setup()
        self._event_publisher("online_session_updated", payload)

    def _is_online_player_name_taken(self, player_name: str, *, exclude_index: int | None = None) -> bool:
        session = self._require_online_session()
        normalized = player_name.casefold()
        for seat in session.seats:
            if exclude_index is not None and seat.seat_index == exclude_index:
                continue
            if seat.player_name is not None and seat.player_name.casefold() == normalized:
                return True
        return False

    @staticmethod
    def _pause_online_session_for_disconnect(session: _OnlineSessionState, seat: _OnlineSeatState) -> None:
        session.state = ONLINE_SESSION_STATE_PAUSED
        session.paused_reason = "player_disconnected"
        session.paused_seat_index = seat.seat_index

    def _resume_online_session_if_ready(self, session: _OnlineSessionState) -> None:
        if session.state != ONLINE_SESSION_STATE_PAUSED:
            return
        if any(seat.status == ONLINE_SEAT_STATUS_DISCONNECTED for seat in session.seats):
            return
        session.state = ONLINE_SESSION_STATE_IN_GAME
        session.paused_reason = None
        session.paused_seat_index = None

    def _player_by_name(self, player_name: str) -> Any:
        game = self._require_game()
        for player in game.players:
            if player.name == player_name:
                return player
        raise ValueError(f"Online game is missing player {player_name}.")

    @staticmethod
    def _generate_token() -> str:
        return secrets.token_urlsafe(18)

    @staticmethod
    def _generate_session_code() -> str:
        return secrets.token_hex(3).upper()

    @staticmethod
    def _reset_seat_to_open(seat: _OnlineSeatState) -> None:
        seat.status = ONLINE_SEAT_STATUS_OPEN
        seat.player_name = None
        seat.player_role = HUMAN_ROLE
        seat.checkpoint_path = None
        seat.action_cooldown_seconds = None
        seat.session_token = None
        seat.reconnect_token = None

    @staticmethod
    def _reset_seat_to_closed(seat: _OnlineSeatState) -> None:
        seat.status = ONLINE_SEAT_STATUS_CLOSED
        seat.player_name = None
        seat.player_role = HUMAN_ROLE
        seat.checkpoint_path = None
        seat.action_cooldown_seconds = None
        seat.session_token = None
        seat.reconnect_token = None

    def _session_response(
        self,
        seat: _OnlineSeatState | None,
        *,
        session: _OnlineSessionState | None = None,
        include_game_state: bool = False,
    ) -> dict[str, Any]:
        active_session = self._require_online_session() if session is None else session
        payload: dict[str, Any] = {
            "online_session": active_session.to_view().to_dict(),
            "session_token": None if seat is None else seat.session_token,
            "reconnect_token": None if seat is None else seat.reconnect_token,
            "player_name": None if seat is None else seat.player_name,
            "seat_index": None if seat is None else seat.seat_index,
            "is_host": False if seat is None else seat.is_host,
        }
        if include_game_state and self.game is not None:
            payload["frontend_state"] = self.game.get_serialized_frontend_state()
            payload["game_setup"] = self.game.get_serialized_game_setup()
        return self._ok_response(**payload)

    def _require_debug_enabled(self) -> None:
        if not self.debug_enabled:
            raise ValueError("Debug mode is disabled.")

    def _advance_ai_turn_guard(self, signature: tuple[int, str, str | None]) -> int:
        if self._ai_turn_guard_signature != signature:
            self._ai_turn_guard_signature = signature
            self._ai_turn_guard_actions = 1
        else:
            self._ai_turn_guard_actions += 1
        return self._ai_turn_guard_actions

    def _reset_ai_turn_guard(self) -> None:
        self._ai_turn_guard_signature = None
        self._ai_turn_guard_actions = 0

    @staticmethod
    def _select_ai_fallback_action(turn_plan: Any) -> Any:
        preferred_action_types = (
            "end_turn",
            "confirm_property_action",
            "reject_trade",
            "decline_property",
            "pass_auction",
            "jail_roll",
            "cancel_property_action",
            "start_turn",
        )
        for action_type in preferred_action_types:
            for legal_action in turn_plan.legal_actions:
                if legal_action.action_type == action_type:
                    return legal_action
        if not turn_plan.legal_actions:
            raise ValueError(f"No legal fallback actions are available for {turn_plan.player_name}.")
        return turn_plan.legal_actions[0]

    @staticmethod
    def _ok_response(**payload: Any) -> dict[str, Any]:
        return {"ok": True, "payload": payload}


def run_backend_process(host: str, port: int, debug_enabled: bool = False) -> None:
    log_path = configure_process_logging("backend")
    logger.info("Backend process starting on %s:%d. Log file: %s", host, port, log_path)
    try:
        runtime = BackendRuntime(debug_enabled=debug_enabled)
        serve_socket_requests(host, port, runtime.handle_command)
    except Exception:
        logger.critical("Backend process terminated unexpectedly.", exc_info=True)
        raise
    finally:
        logger.info("Backend process stopped.")