from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from typing import Any

from monopoly.api import FrontendStateView, InteractionResult, LegalActionOption, OnlineSessionView
from monopoly.gui.transport import SocketTransportClient


logger = logging.getLogger("monopoly.gui.frontend.controller")


class BackendClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.owns_server = True
        self.transport = SocketTransportClient(host, port)

    def reconnect(self, host: str, port: int, *, owns_server: bool = False) -> None:
        self.transport.close()
        self.host = host
        self.port = port
        self.owns_server = owns_server
        self.transport = SocketTransportClient(host, port)
        logger.info("Reconnected backend client to %s:%d (owns_server=%s).", host, port, owns_server)

    def create_game(
        self,
        player_names: list[str],
        starting_cash: int,
        *,
        player_roles: list[str] | None = None,
        ai_checkpoint_path: str | None = None,
        ai_player_setups: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            {
                "command": "create_game",
                "setup": {
                    "player_names": player_names,
                    "starting_cash": starting_cash,
                    "player_roles": player_roles,
                    "ai_checkpoint_path": ai_checkpoint_path,
                    "ai_player_setups": ai_player_setups,
                },
            }
        )

    def get_state(self) -> dict[str, Any]:
        return self._request({"command": "get_state"})

    def execute_action(
        self,
        action_payload: dict[str, Any],
        *,
        bid_amount: int | None = None,
        trade_offer: dict[str, Any] | None = None,
        session_token: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            {
                "command": "execute_action",
                "action": action_payload,
                "bid_amount": bid_amount,
                "trade_offer": trade_offer,
                "session_token": session_token,
            }
        )

    def step_ai(self, *, session_token: str | None = None) -> dict[str, Any]:
        return self._request({"command": "step_ai", "session_token": session_token})

    def save_game(self, file_path: str) -> dict[str, Any]:
        return self._request({"command": "save_game", "file_path": file_path})

    def load_game(self, file_path: str) -> dict[str, Any]:
        return self._request({"command": "load_game", "file_path": file_path})

    def create_online_lobby(self, host_player_name: str, seat_count: int, starting_cash: int) -> dict[str, Any]:
        return self._request(
            {
                "command": "create_online_lobby",
                "host_player_name": host_player_name,
                "seat_count": seat_count,
                "starting_cash": starting_cash,
            }
        )

    def get_online_session(self, *, session_code: str | None = None, session_token: str | None = None) -> dict[str, Any]:
        return self._request(
            {
                "command": "get_online_session",
                "session_code": session_code,
                "session_token": session_token,
            }
        )

    def open_online_slot(self, session_token: str, seat_index: int) -> dict[str, Any]:
        return self._request(
            {
                "command": "open_online_slot",
                "session_token": session_token,
                "seat_index": seat_index,
            }
        )

    def close_online_slot(self, session_token: str, seat_index: int) -> dict[str, Any]:
        return self._request(
            {
                "command": "close_online_slot",
                "session_token": session_token,
                "seat_index": seat_index,
            }
        )

    def assign_ai_to_online_slot(
        self,
        session_token: str,
        seat_index: int,
        *,
        player_name: str | None = None,
        checkpoint_path: str | None = None,
        action_cooldown_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self._request(
            {
                "command": "assign_ai_to_online_slot",
                "session_token": session_token,
                "seat_index": seat_index,
                "player_name": player_name,
                "checkpoint_path": checkpoint_path,
                "action_cooldown_seconds": action_cooldown_seconds,
            }
        )

    def clear_online_slot(self, session_token: str, seat_index: int) -> dict[str, Any]:
        return self._request(
            {
                "command": "clear_online_slot",
                "session_token": session_token,
                "seat_index": seat_index,
            }
        )

    def claim_online_slot(self, session_code: str, seat_index: int, player_name: str) -> dict[str, Any]:
        return self._request(
            {
                "command": "claim_online_slot",
                "session_code": session_code,
                "seat_index": seat_index,
                "player_name": player_name,
            }
        )

    def disconnect_online_slot(self, session_token: str, *, session_code: str | None = None) -> dict[str, Any]:
        return self._request(
            {
                "command": "disconnect_online_slot",
                "session_token": session_token,
                "session_code": session_code,
            }
        )

    def reconnect_online_slot(self, reconnect_token: str, *, session_code: str | None = None) -> dict[str, Any]:
        return self._request(
            {
                "command": "reconnect_online_slot",
                "reconnect_token": reconnect_token,
                "session_code": session_code,
            }
        )

    def replace_disconnected_online_slot_with_ai(
        self,
        session_token: str,
        seat_index: int,
        *,
        checkpoint_path: str | None = None,
        action_cooldown_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self._request(
            {
                "command": "replace_disconnected_online_slot_with_ai",
                "session_token": session_token,
                "seat_index": seat_index,
                "checkpoint_path": checkpoint_path,
                "action_cooldown_seconds": action_cooldown_seconds,
            }
        )

    def start_online_game(self, session_token: str, *, session_code: str | None = None) -> dict[str, Any]:
        return self._request(
            {
                "command": "start_online_game",
                "session_token": session_token,
                "session_code": session_code,
            }
        )

    def shutdown(self) -> None:
        if self.owns_server:
            try:
                self._request({"command": "shutdown"})
            except Exception:
                pass
        self.transport.close()
        logger.info("Backend client shutdown complete.")

    def close(self) -> None:
        self.transport.close()

    def drain_events(self) -> list[dict[str, Any]]:
        return self.transport.drain_events()

    @property
    def is_closed(self) -> bool:
        return self.transport.is_closed

    def _request(self, command: dict[str, Any]) -> dict[str, Any]:
        self._reconnect_owned_transport_if_needed()
        logger.debug("Sending backend command %s.", command.get("command"))
        try:
            response = self.transport.request(command)
        except ConnectionError:
            if not self.owns_server:
                raise
            logger.warning("Owned backend transport was closed. Attempting reconnect for command %s.", command.get("command"))
            self._reconnect_owned_transport_if_needed(force=True)
            response = self.transport.request(command)
        if not response.get("ok"):
            logger.error("Backend command %s failed: %s", command.get("command"), response.get("error"))
            raise ValueError(response.get("error", "Unknown backend error."))
        return response["payload"]

    def _reconnect_owned_transport_if_needed(self, *, force: bool = False) -> None:
        if not self.owns_server:
            return
        if not force and not self.transport.is_closed:
            return
        try:
            self.transport.close()
        except Exception:
            pass
        self.transport = SocketTransportClient(self.host, self.port)
        logger.info("Re-established owned backend transport to %s:%d.", self.host, self.port)


@dataclass(slots=True)
class FrontendController:
    client: BackendClient | None
    frontend_state: FrontendStateView | None = None
    online_session: OnlineSessionView | None = None
    session_token: str | None = None
    reconnect_token: str | None = None
    online_player_name: str | None = None
    online_seat_index: int | None = None
    is_online_host: bool = False
    selected_space_index: int = 0
    message_history: list[str] = field(default_factory=list)
    status_message: str = "Ready."
    last_error: str | None = None
    replay_frames: list[dict[str, Any]] = field(default_factory=list)

    def start_game(
        self,
        player_names: list[str],
        starting_cash: int,
        *,
        player_roles: list[str] | None = None,
        ai_checkpoint_path: str | None = None,
        ai_player_setups: list[dict[str, Any]] | None = None,
    ) -> None:
        logger.info("Starting local game for %d players.", len(player_names))
        payload = self._require_client().create_game(
            player_names,
            starting_cash,
            player_roles=player_roles,
            ai_checkpoint_path=ai_checkpoint_path,
            ai_player_setups=ai_player_setups,
        )
        self._clear_online_session_state()
        self.message_history = [f"Game created for {len(player_names)} players with starting cash ${starting_cash}."]
        self._apply_payload(payload)
        self.status_message = "Game created."
        self.last_error = None
        self._record_replay_frame("create_game")

    def load_game(self, file_path: str) -> None:
        logger.info("Loading saved game from %s.", file_path)
        payload = self._require_client().load_game(file_path)
        self._clear_online_session_state()
        self.frontend_state = FrontendStateView.from_dict(payload["frontend_state"])
        self.selected_space_index = min(self.selected_space_index, len(self.frontend_state.board_spaces) - 1)
        self.message_history = [f"Loaded saved game from {file_path}."]
        self.status_message = f"Loaded {file_path}."
        self.last_error = None
        self._record_replay_frame("load_game")

    def save_game(self, file_path: str) -> None:
        logger.info("Saving current game to %s.", file_path)
        self._require_client().save_game(file_path)
        self.message_history.append(f"Saved game to {file_path}.")
        self.status_message = f"Saved {file_path}."
        self.last_error = None

    def execute_action(
        self,
        action: LegalActionOption,
        *,
        bid_amount: int | None = None,
        trade_offer: dict[str, Any] | None = None,
    ) -> InteractionResult:
        logger.debug("Executing frontend action %s for %s.", action.action_type, action.actor_name)
        payload = self._require_client().execute_action(
            action.to_dict(),
            bid_amount=bid_amount,
            trade_offer=trade_offer,
            session_token=self.session_token,
        )
        interaction = InteractionResult.from_dict(payload["interaction"])
        self.frontend_state = FrontendStateView.from_dict(payload["frontend_state"])
        self.selected_space_index = min(self.selected_space_index, len(self.frontend_state.board_spaces) - 1)
        self.message_history.extend(interaction.messages)
        self.status_message = interaction.messages[-1] if interaction.messages else action.description
        self.last_error = None
        self._record_replay_frame(action.handler_name)
        return interaction

    def step_ai(self) -> InteractionResult:
        logger.debug("Requesting AI step from backend.")
        payload = self._require_client().step_ai(session_token=self.session_token)
        interaction = InteractionResult.from_dict(payload["interaction"])
        self.frontend_state = FrontendStateView.from_dict(payload["frontend_state"])
        self.selected_space_index = min(self.selected_space_index, len(self.frontend_state.board_spaces) - 1)
        self.message_history.extend(interaction.messages)
        self.status_message = interaction.messages[-1] if interaction.messages else f"{payload['actor_name']} completed an AI action."
        self.last_error = None
        self._record_replay_frame(f"ai:{payload['actor_name']}")
        return interaction

    def save_replay(self, file_path: str) -> None:
        logger.info("Saving replay to %s.", file_path)
        payload = {
            "version": 1,
            "frames": self.replay_frames,
        }
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.message_history.append(f"Saved replay to {file_path}.")
        self.status_message = f"Saved replay to {file_path}."
        self.last_error = None

    def load_replay(self, file_path: str) -> list[dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("Replay file does not contain any frames.")
        return frames

    def get_debug_state(self) -> dict[str, Any]:
        response = self._require_client()._request({"command": "get_debug_state"})
        return dict(response["full_state"])

    def apply_debug_state(self, full_state: dict[str, Any]) -> None:
        logger.info("Applying debug state to frontend controller.")
        payload = self._require_client()._request({"command": "apply_debug_state", "full_state": full_state})
        self.frontend_state = FrontendStateView.from_dict(payload["frontend_state"])
        self.selected_space_index = min(self.selected_space_index, len(self.frontend_state.board_spaces) - 1)
        self.message_history.append("Applied debug state.")
        self.status_message = "Applied debug state."
        self.last_error = None
        self._record_replay_frame("apply_debug_state")

    def select_space(self, index: int) -> None:
        if self.frontend_state is None:
            return
        self.selected_space_index = max(0, min(index, len(self.frontend_state.board_spaces) - 1))

    def set_error(self, message: str) -> None:
        self.last_error = message
        self.status_message = message
        self.message_history.append(f"Error: {message}")
        logger.error("%s", message)

    def shutdown(self) -> None:
        if self.client is not None:
            self.client.shutdown()
        logger.info("Frontend controller shutdown complete.")

    def connect_to_backend(self, host: str, port: int, *, owns_server: bool = False) -> None:
        if self.client is None:
            self.client = BackendClient(host, port)
            self.client.owns_server = owns_server
        else:
            self.client.reconnect(host, port, owns_server=owns_server)
        self.frontend_state = None
        self.online_session = None
        self.session_token = None
        self.reconnect_token = None
        self.online_player_name = None
        self.online_seat_index = None
        self.is_online_host = False
        self.status_message = f"Connected to {host}:{port}."
        self.last_error = None
        logger.info("Connected frontend controller to backend at %s:%d (owns_server=%s).", host, port, owns_server)

    def _clear_online_session_state(self) -> None:
        self.online_session = None
        self.session_token = None
        self.reconnect_token = None
        self.online_player_name = None
        self.online_seat_index = None
        self.is_online_host = False

    def create_online_lobby(self, host_player_name: str, seat_count: int, starting_cash: int) -> None:
        logger.info("Creating online lobby as %s with %d seats.", host_player_name, seat_count)
        payload = self._require_client().create_online_lobby(host_player_name, seat_count, starting_cash)
        self._apply_online_payload(payload)
        self.message_history = [f"Created online lobby {self.online_session.session_code}."] if self.online_session is not None else []
        self.status_message = "Online lobby created."
        self.last_error = None

    def refresh_online_session(self, *, session_code: str | None = None) -> None:
        logger.debug("Refreshing online session %s.", session_code or "<current>")
        payload = self._require_client().get_online_session(session_code=session_code, session_token=self.session_token)
        self._apply_online_refresh_payload(payload)
        self.status_message = "Online lobby updated."
        self.last_error = None

    def open_online_slot(self, seat_index: int) -> None:
        payload = self._require_client().open_online_slot(self._require_session_token(), seat_index)
        self._apply_online_payload(payload)

    def close_online_slot(self, seat_index: int) -> None:
        payload = self._require_client().close_online_slot(self._require_session_token(), seat_index)
        self._apply_online_payload(payload)

    def assign_ai_to_online_slot(
        self,
        seat_index: int,
        *,
        player_name: str | None = None,
        checkpoint_path: str | None = None,
        action_cooldown_seconds: float | None = None,
    ) -> None:
        payload = self._require_client().assign_ai_to_online_slot(
            self._require_session_token(),
            seat_index,
            player_name=player_name,
            checkpoint_path=checkpoint_path,
            action_cooldown_seconds=action_cooldown_seconds,
        )
        self._apply_online_payload(payload)

    def clear_online_slot(self, seat_index: int) -> None:
        payload = self._require_client().clear_online_slot(self._require_session_token(), seat_index)
        self._apply_online_payload(payload)

    def claim_online_slot(self, session_code: str, seat_index: int, player_name: str) -> None:
        logger.info("Joining lobby %s as %s in seat %d.", session_code, player_name, seat_index)
        payload = self._require_client().claim_online_slot(session_code, seat_index, player_name)
        self._apply_online_payload(payload)
        self.status_message = f"Joined lobby {session_code}."
        self.last_error = None

    def disconnect_online_slot(self) -> None:
        payload = self._require_client().disconnect_online_slot(self._require_session_token(), session_code=None if self.online_session is None else self.online_session.session_code)
        self._apply_online_payload(payload)

    def reconnect_online_slot(self, reconnect_token: str, *, session_code: str | None = None) -> None:
        logger.info("Attempting online reconnect for session %s.", session_code or "<unknown>")
        payload = self._require_client().reconnect_online_slot(reconnect_token, session_code=session_code)
        self._apply_online_payload(payload)
        if "frontend_state" in payload:
            self.frontend_state = FrontendStateView.from_dict(payload["frontend_state"])
            self.selected_space_index = min(self.selected_space_index, len(self.frontend_state.board_spaces) - 1)

    def replace_disconnected_online_slot_with_ai(self, seat_index: int) -> None:
        payload = self._require_client().replace_disconnected_online_slot_with_ai(self._require_session_token(), seat_index)
        self._apply_online_payload(payload)
        if "frontend_state" in payload:
            self.frontend_state = FrontendStateView.from_dict(payload["frontend_state"])
            self.selected_space_index = min(self.selected_space_index, len(self.frontend_state.board_spaces) - 1)

    def start_online_game(self) -> None:
        logger.info("Starting online game for current lobby.")
        payload = self._require_client().start_online_game(self._require_session_token(), session_code=None if self.online_session is None else self.online_session.session_code)
        self._apply_online_payload(payload)
        if "frontend_state" in payload:
            self.frontend_state = FrontendStateView.from_dict(payload["frontend_state"])
            self.selected_space_index = min(self.selected_space_index, len(self.frontend_state.board_spaces) - 1)
        self.status_message = "Online game started."
        self.last_error = None

    def _apply_payload(self, payload: dict[str, Any]) -> None:
        self.frontend_state = FrontendStateView.from_dict(payload["frontend_state"])
        self.selected_space_index = min(self.selected_space_index, len(self.frontend_state.board_spaces) - 1)

    def drain_online_events(self) -> bool:
        client = self.client
        if client is None:
            return False
        changed = False
        for event in client.drain_events():
            if event.get("event") != "online_session_updated":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            self._apply_online_snapshot_payload(payload)
            changed = True
        if changed:
            self.status_message = "Online session updated."
            self.last_error = None
            logger.debug("Applied online session update event.")
        return changed

    def _apply_online_payload(self, payload: dict[str, Any]) -> None:
        self._apply_online_snapshot_payload(payload)
        if payload.get("session_token") is not None:
            self.session_token = str(payload["session_token"])
        if payload.get("reconnect_token") is not None:
            self.reconnect_token = str(payload["reconnect_token"])
        self.online_player_name = None if payload.get("player_name") is None else str(payload["player_name"])
        self.online_seat_index = None if payload.get("seat_index") is None else int(payload["seat_index"])
        self.is_online_host = bool(payload.get("is_host", False))

    def _apply_online_refresh_payload(self, payload: dict[str, Any]) -> None:
        self._apply_online_snapshot_payload(payload)
        if payload.get("session_token") is not None:
            self.session_token = str(payload["session_token"])
        if payload.get("reconnect_token") is not None:
            self.reconnect_token = str(payload["reconnect_token"])
        if payload.get("player_name") is not None:
            self.online_player_name = str(payload["player_name"])
        if payload.get("seat_index") is not None:
            self.online_seat_index = int(payload["seat_index"])
        self.is_online_host = bool(payload.get("is_host", self.is_online_host))

    def _apply_online_snapshot_payload(self, payload: dict[str, Any]) -> None:
        online_session = payload.get("online_session")
        if online_session is not None:
            self.online_session = OnlineSessionView.from_dict(online_session)
        if "frontend_state" in payload:
            self.frontend_state = FrontendStateView.from_dict(payload["frontend_state"])
            self.selected_space_index = min(self.selected_space_index, len(self.frontend_state.board_spaces) - 1)

    def _require_session_token(self) -> str:
        if self.session_token is None:
            raise ValueError("No online session token is available for this client.")
        return self.session_token

    def _require_client(self) -> BackendClient:
        if self.client is None:
            raise ValueError("Backend connection is not established.")
        return self.client

    def _record_replay_frame(self, label: str) -> None:
        if self.frontend_state is None:
            return
        self.replay_frames.append(
            {
                "label": label,
                "selected_space_index": self.selected_space_index,
                "status_message": self.status_message,
                "message_history": list(self.message_history),
                "frontend_state": self.frontend_state.to_dict(),
            }
        )
