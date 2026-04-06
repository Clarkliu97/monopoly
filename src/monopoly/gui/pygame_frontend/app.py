from __future__ import annotations

from dataclasses import dataclass
from html import escape
import logging
import multiprocessing as mp
from pathlib import Path
import random
import re
from typing import Any

import pygame
import pygame_gui
from pygame_gui.elements import UIButton, UIDropDownMenu, UILabel, UIPanel, UITextBox, UITextEntryLine, UIWindow

from monopoly.agent import DEFAULT_CHECKPOINT_DIRECTORY, DEFAULT_CHECKPOINT_FILE_NAME, default_scripted_profiles
from monopoly.api import FrontendStateView, LegalActionOption, PlayerView
from monopoly.constants import AI_ROLE, BANK_HOTELS, BANK_HOUSES, HUMAN_ROLE, JAIL_FINE, JAIL_INDEX, PLAYER_ROLES, POST_ROLL_PHASE, PRE_ROLL_PHASE, IN_TURN_PHASE
from monopoly.gui.backend_process import run_backend_process
from monopoly.gui.pygame_frontend.board import BoardRenderer
from monopoly.gui.pygame_frontend.controller import BackendClient, FrontendController
from monopoly.gui.rendezvous import DEFAULT_RENDEZVOUS_HOST, DEFAULT_RENDEZVOUS_PORT, RendezvousClient
from monopoly.gui.transport import find_free_port
from monopoly.gui.pygame_frontend import theme
from monopoly.logging_utils import configure_process_logging


LOCAL_PLAY_MODE = "local_play"
SCRIPTED_AI_SELECTION_PREFIX = "scripted:"


logger = logging.getLogger("monopoly.gui.frontend.app")


@dataclass(slots=True)
class TextPromptState:
    kind: str
    window: UIWindow
    entry: UITextEntryLine
    submit_button: UIButton
    cancel_button: UIButton
    title: str
    action: LegalActionOption | None = None


@dataclass(slots=True)
class TradePromptState:
    window: UIWindow
    submit_button: UIButton
    cancel_button: UIButton
    action: LegalActionOption
    proposer: PlayerView
    receiver: PlayerView
    proposer_cash_label: UILabel
    receiver_cash_label: UILabel
    proposer_jail_cards_label: UILabel
    receiver_jail_cards_label: UILabel
    proposer_cash_value: int
    receiver_cash_value: int
    proposer_jail_cards_value: int
    receiver_jail_cards_value: int
    proposer_selected_properties: set[str]
    receiver_selected_properties: set[str]
    proposer_property_buttons: list[UIButton]
    receiver_property_buttons: list[UIButton]
    proposer_property_page: int
    receiver_property_page: int
    note_entry: UITextEntryLine
    accept_button: UIButton | None = None
    reject_action: LegalActionOption | None = None
    accept_action: LegalActionOption | None = None


@dataclass(slots=True)
class AnimationSegment:
    dice_values: tuple[int, int] | None
    path: list[int]


@dataclass(slots=True)
class AnimationState:
    previous_state: FrontendStateView
    final_state: FrontendStateView
    player_name: str | None
    steps: list[AnimationSegment]
    speed: float
    elapsed: float = 0.0
    step_index: int = 0
    phase: str = "dice"

    @property
    def current_step(self) -> AnimationSegment:
        return self.steps[self.step_index]

    @property
    def dice_duration(self) -> float:
        return 0.9 / self.speed

    @property
    def move_duration(self) -> float:
        if len(self.current_step.path) <= 1:
            return 0.0
        return max(0.18, 0.22 * (len(self.current_step.path) - 1)) / self.speed


@dataclass(slots=True)
class ReplayState:
    frames: list[dict[str, Any]]
    current_index: int = 0
    is_playing: bool = False
    elapsed: float = 0.0


@dataclass(slots=True)
class DebugPanelState:
    selected_player_name: str
    player_select: UIDropDownMenu | None
    player_role_dropdown: UIDropDownMenu | None
    player_in_jail_dropdown: UIDropDownMenu | None
    player_bankrupt_dropdown: UIDropDownMenu | None
    player_cash_entry: UITextEntryLine | None
    player_position_entry: UITextEntryLine | None
    player_jail_cards_entry: UITextEntryLine | None
    player_jail_turns_entry: UITextEntryLine | None
    queued_roll_label: UILabel | None
    next_die_one_entry: UITextEntryLine | None
    next_die_two_entry: UITextEntryLine | None
    apply_player_button: UIButton | None
    set_next_roll_button: UIButton | None
    clear_next_roll_button: UIButton | None
    runtime_current_player_dropdown: UIDropDownMenu | None
    runtime_turn_phase_dropdown: UIDropDownMenu | None
    runtime_pending_action_dropdown: UIDropDownMenu | None
    runtime_turn_counter_entry: UITextEntryLine | None
    runtime_houses_entry: UITextEntryLine | None
    runtime_hotels_entry: UITextEntryLine | None
    runtime_continuation_player_dropdown: UIDropDownMenu | None
    runtime_continuation_doubles_entry: UITextEntryLine | None
    runtime_continuation_rolled_double_dropdown: UIDropDownMenu | None
    runtime_auction_current_bid_entry: UITextEntryLine | None
    runtime_auction_bidder_index_entry: UITextEntryLine | None
    runtime_auction_winner_dropdown: UIDropDownMenu | None
    runtime_trade_proposer_dropdown: UIDropDownMenu | None
    runtime_trade_receiver_dropdown: UIDropDownMenu | None
    runtime_trade_proposer_cash_entry: UITextEntryLine | None
    runtime_trade_receiver_cash_entry: UITextEntryLine | None
    runtime_trade_note_entry: UITextEntryLine | None
    apply_runtime_button: UIButton
    property_space_index: int | None
    property_owner_dropdown: UIDropDownMenu | None
    property_mortgaged_dropdown: UIDropDownMenu | None
    property_building_dropdown: UIDropDownMenu | None
    apply_property_button: UIButton | None


@dataclass(slots=True)
class SetupFieldState:
    label: UILabel
    entry: UITextEntryLine
    role_dropdown: UIDropDownMenu
    checkpoint_dropdown: UIDropDownMenu | None
    cooldown_entry: UITextEntryLine | None


@dataclass(slots=True)
class AIStepState:
    actor_name: str
    remaining_delay: float


class MonopolyPygameApp:
    def __init__(self, host: str | None, port: int | None, discovery_host: str | None = None, discovery_port: int | None = None, debug_mode: bool = False) -> None:
        pygame.init()
        pygame.display.set_caption("Monopoly Game Night")
        self.screen = pygame.display.set_mode((theme.WINDOW_WIDTH, theme.WINDOW_HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.manager = pygame_gui.UIManager((theme.WINDOW_WIDTH, theme.WINDOW_HEIGHT), theme.UI_THEME_PATH)
        self.manager.preload_fonts([
            {"name": "noto_sans", "point_size": 14, "style": "bold", "antialiased": "1"},
            {"name": "noto_sans", "point_size": 15, "style": "regular", "antialiased": "1"},
        ])
        initial_client = None if host is None or port is None else BackendClient(host, port)
        self.controller = FrontendController(initial_client)
        self.debug_mode = debug_mode
        self._managed_backend_process: mp.Process | None = None
        self.board_rect = pygame.Rect(0, 0, 0, 0)
        self.board_renderer = BoardRenderer(self.board_rect)
        self.running = True
        self.screen_mode = "setup"
        self.hovered_space_index: int | None = None
        self.sidebar_width = theme.SIDEBAR_WIDTH
        self.animation_speed_text = "1.0"
        self.animation_speed = 1.0
        self._animation: AnimationState | None = None
        self._property_submenu_mode: str | None = None
        self._property_submenu_page = 0

        self._elements: list[Any] = []
        self._action_buttons: dict[UIButton, Any] = {}
        self._ui_commands: dict[UIButton, tuple[str, Any]] = {}
        self._dropdown_values: dict[Any, str] = {}
        self._setup_fields: list[SetupFieldState] = []
        self._setup_role_dropdowns: dict[UIDropDownMenu, int] = {}
        self._setup_checkpoint_dropdowns: dict[UIDropDownMenu, int] = {}
        self._host_lobby_slot_dropdowns: dict[UIDropDownMenu, int] = {}
        self._host_lobby_ai_checkpoint_dropdowns: dict[UIDropDownMenu, int] = {}
        self._host_lobby_ai_speed_entries: dict[int, UITextEntryLine] = {}
        self._host_lobby_ai_apply_buttons: dict[UIButton, int] = {}
        self._prompt: TextPromptState | None = None
        self._trade_prompt: TradePromptState | None = None
        self._replay: ReplayState | None = None
        self._debug_panel: DebugPanelState | None = None
        self._debug_active_tab = "player"
        self._debug_selected_player_name: str | None = None
        self._debug_validation_messages: dict[str, str] = {}
        self._debug_field_elements: dict[str, Any] = {}
        self._debug_invalid_elements: list[Any] = []
        self._pending_ai_step: AIStepState | None = None

        self.player_count = 2
        self.player_names = [f"Player {index}" for index in range(1, 7)]
        self.player_roles = [HUMAN_ROLE for _ in range(6)]
        self.setup_game_mode = LOCAL_PLAY_MODE
        self.starting_cash = "1500"
        self.ai_checkpoint_options = self._discover_checkpoint_options()
        default_checkpoint_path = self._default_checkpoint_option()
        self.ai_checkpoint_paths = [default_checkpoint_path for _ in range(6)]
        self.ai_checkpoint_path = default_checkpoint_path
        self.ai_cooldown_texts = ["2.0" for _ in range(6)]
        self.online_remote_host = "127.0.0.1" if host is None else host
        self.online_remote_port = "0" if port is None else str(port)
        self.online_session_code = ""
        self.online_join_player_name = "Guest"
        self.online_reconnect_token = ""
        self.online_discovery_host = DEFAULT_RENDEZVOUS_HOST if discovery_host is None else discovery_host
        self.online_discovery_port = str(DEFAULT_RENDEZVOUS_PORT if discovery_port is None else discovery_port)

        self._recalculate_layout(*self.screen.get_size())
        self._build_setup_screen()

    def run(self) -> None:
        while self.running:
            time_delta = self.clock.tick(theme.FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    self._resize_window(event.w, event.h)
                self.manager.process_events(event)
                self._handle_event(event)

            self._update_animation(time_delta)
            self._process_online_events()
            self._update_ai_turns(time_delta)
            self.manager.update(time_delta)
            self._draw_frame()
            pygame.display.flip()

        self._unregister_discovery_lobby()
        self.controller.shutdown()
        self._stop_managed_backend()
        pygame.quit()

    def _handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEMOTION and self.screen_mode in {"game", "replay"}:
            if self._animation is not None:
                return
            self.hovered_space_index = self.board_renderer.layout.hit_test(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.screen_mode in {"game", "replay"}:
            if self._animation is not None:
                return
            index = self.board_renderer.layout.hit_test(event.pos)
            if index is not None:
                self.controller.select_space(index)
                if self.screen_mode == "replay":
                    self._rebuild_replay_ui()
                else:
                    self._rebuild_game_ui()

        if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
            selected_text = self._dropdown_value_from_display(str(event.text))
            self._dropdown_values[event.ui_element] = selected_text
            if self.screen_mode == "setup":
                if event.ui_element == getattr(self, "setup_mode_dropdown", None):
                    self._capture_setup_inputs()
                    self.setup_game_mode = selected_text
                    self._build_setup_screen()
                    return
                setup_role_index = self._setup_role_dropdowns.get(event.ui_element)
                if setup_role_index is not None:
                    self._capture_setup_inputs()
                    self.player_roles[setup_role_index] = selected_text
                    self._build_setup_screen()
                    return
                setup_checkpoint_index = self._setup_checkpoint_dropdowns.get(event.ui_element)
                if setup_checkpoint_index is not None:
                    self.ai_checkpoint_paths[setup_checkpoint_index] = selected_text
                    return
            if self.screen_mode == "online_lobby":
                lobby_slot_index = self._host_lobby_slot_dropdowns.get(event.ui_element)
                if lobby_slot_index is not None:
                    self._set_host_lobby_slot_mode(lobby_slot_index, selected_text)
                    return
                lobby_ai_checkpoint_index = self._host_lobby_ai_checkpoint_dropdowns.get(event.ui_element)
                if lobby_ai_checkpoint_index is not None:
                    self.ai_checkpoint_paths[lobby_ai_checkpoint_index] = selected_text
                    return
            if self.screen_mode == "game" and self._debug_panel is not None and event.ui_element == self._debug_panel.player_select:
                self._debug_selected_player_name = str(event.text)
                self._rebuild_game_ui()
            return

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            self._handle_button_pressed(event.ui_element)

    def _handle_button_pressed(self, element: Any) -> None:
        if self._trade_prompt is not None:
            if self._trade_prompt.accept_button is not None and element == self._trade_prompt.accept_button:
                self._accept_trade_prompt()
                return
            if element == self._trade_prompt.cancel_button:
                if self._trade_prompt.reject_action is not None:
                    self._reject_trade_prompt()
                else:
                    self._close_trade_prompt()
                return
            if element == self._trade_prompt.submit_button:
                self._submit_trade_prompt()
                return

        if self._prompt is not None:
            if element == self._prompt.cancel_button:
                self._close_prompt()
                return
            if element == self._prompt.submit_button:
                self._submit_prompt()
                return

        command = self._ui_commands.get(element)
        if command is not None:
            self._handle_ui_command(*command)
            return

        host_lobby_ai_seat = self._host_lobby_ai_apply_buttons.get(element)
        if host_lobby_ai_seat is not None:
            self._apply_host_lobby_ai_settings(host_lobby_ai_seat)
            return

        if hasattr(self, "player_minus_button") and element == self.player_minus_button:
            self._capture_setup_inputs()
            self.player_count = max(2, self.player_count - 1)
            self._build_setup_screen()
            return
        if hasattr(self, "player_plus_button") and element == self.player_plus_button:
            self._capture_setup_inputs()
            self.player_count = min(6, self.player_count + 1)
            self._build_setup_screen()
            return
        if hasattr(self, "refresh_checkpoints_button") and element == self.refresh_checkpoints_button:
            self._capture_setup_inputs()
            self.ai_checkpoint_options = self._discover_checkpoint_options()
            self._build_setup_screen()
            return
        if hasattr(self, "start_button") and element == self.start_button:
            self._start_game()
            return
        if hasattr(self, "load_setup_button") and element == self.load_setup_button:
            self._open_path_prompt("load", "Load Saved Game", "savegame.json")
            return
        if hasattr(self, "save_button") and element == self.save_button:
            self._open_path_prompt("save", "Save Game", "savegame.json")
            return
        if hasattr(self, "load_button") and element == self.load_button:
            self._open_path_prompt("load", "Load Saved Game", "savegame.json")
            return

        action = self._action_buttons.get(element)
        if action is None:
            return

        if self._is_ai_turn_active():
            return

        if isinstance(action, tuple):
            command, value, _description = action
            if command == "submenu":
                self._property_submenu_mode = value
                self._property_submenu_page = 0
                self._rebuild_game_ui()
            elif command == "submenu_prev":
                self._property_submenu_page = max(0, self._property_submenu_page - 1)
                self._rebuild_game_ui()
            elif command == "submenu_next":
                self._property_submenu_page += 1
                self._rebuild_game_ui()
            elif command == "submenu_back":
                self._property_submenu_mode = None
                self._property_submenu_page = 0
                self._rebuild_game_ui()
            return

        if action.handler_name == "submit_auction_bid" and action.fixed_choice != "pass":
            self._open_bid_prompt(action)
            return
        if action.handler_name == "propose_trade_interactive":
            self._open_trade_prompt(action)
            return
        if action.handler_name == "counter_trade_interactive":
            self._open_counter_trade_prompt(action)
            return

        try:
            previous_state = self.controller.frontend_state
            interaction = self.controller.execute_action(action)
            self._property_submenu_mode = None
            self._property_submenu_page = 0
            if not self._start_action_animation(action, interaction, previous_state, self.controller.frontend_state):
                self._rebuild_game_ui()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._rebuild_game_ui()

    def _handle_ui_command(self, command: str, payload: Any) -> None:
        if command == "debug_tab":
            self._debug_active_tab = str(payload)
            self._rebuild_game_ui()
            return
        if command == "host_open_online_slot":
            self._host_open_online_slot(int(payload))
            return
        if command == "host_close_online_slot":
            self._host_close_online_slot(int(payload))
            return
        if command == "host_assign_ai_online_slot":
            self._host_assign_ai_online_slot(int(payload))
            return
        if command == "host_clear_online_slot":
            self._host_clear_online_slot(int(payload))
            return
        if command == "host_replace_disconnected_with_ai":
            self._host_replace_disconnected_with_ai(int(payload))
            return
        if command == "join_online_seat":
            self._join_online_seat(int(payload))
            return
        if command == "start_online_match":
            self._start_online_match()
            return
        if command == "back_to_setup":
            self._unregister_discovery_lobby()
            self._build_setup_screen()
            return
        if command == "trade_adjust":
            field_name, delta = payload
            self._adjust_trade_value(field_name, delta)
            return
        if command == "trade_toggle_property":
            side, property_name = payload
            self._toggle_trade_property(side, property_name)
            return
        if command == "trade_page":
            side, delta = payload
            self._change_trade_property_page(side, delta)
            return
        if command == "save_replay":
            self._open_text_prompt("save_replay", "Save Replay", "replay.json", None)
            return
        if command == "load_replay":
            self._open_text_prompt("load_replay", "Load Replay", "replay.json", None)
            return
        if command == "apply_debug_player":
            self._apply_debug_player_edits()
            return
        if command == "apply_debug_property":
            self._apply_debug_property_edits()
            return
        if command == "apply_debug_runtime":
            self._apply_debug_runtime_edits()
            return
        if command == "set_debug_roll":
            self._set_debug_next_roll()
            return
        if command == "clear_debug_roll":
            self._clear_debug_next_roll()
            return
        if command == "replay_prev":
            self._step_replay(-1)
            return
        if command == "replay_next":
            self._step_replay(1)
            return
        if command == "replay_toggle":
            if self._replay is not None:
                self._replay.is_playing = not self._replay.is_playing
                self._rebuild_replay_ui()
            return
        if command == "replay_exit":
            self._replay = None
            self.screen_mode = "game"
            self._rebuild_game_ui()
            return

    def _start_game(self) -> None:
        if self.setup_game_mode == LOCAL_PLAY_MODE:
            self._start_local_game()
            return
        if self.setup_game_mode == "host_online":
            self._create_online_lobby_from_setup()
            return
        if self.setup_game_mode == "join_online":
            self._join_online_lobby_from_setup()
            return

    def _start_local_game(self) -> None:
        try:
            self._ensure_local_backend_connection()
            self._capture_setup_inputs()
            player_names: list[str] = []
            player_roles: list[str] = []
            ai_player_setups: list[dict[str, Any]] = []
            for index, field in enumerate(self._setup_fields[: self.player_count]):
                value = field.entry.get_text().strip() or self.player_names[index]
                player_names.append(value)
                player_role = self._dropdown_value(field.role_dropdown, self.player_roles[index])
                player_roles.append(player_role)
                if player_role == AI_ROLE:
                    checkpoint_path = self._dropdown_value(field.checkpoint_dropdown, self.ai_checkpoint_paths[index]).strip()
                    cooldown_text = field.cooldown_entry.get_text().strip() or self.ai_cooldown_texts[index]
                    cooldown_seconds = float(cooldown_text)
                    if cooldown_seconds < 0:
                        raise ValueError("AI cooldown cannot be negative.")
                    self.ai_checkpoint_paths[index] = checkpoint_path
                    self.ai_cooldown_texts[index] = cooldown_text
                    ai_player_setups.append(
                        {
                            "player_name": value,
                            "checkpoint_path": checkpoint_path,
                            "action_cooldown_seconds": cooldown_seconds,
                        }
                    )
            if len(set(player_names)) != len(player_names):
                raise ValueError("Player names must be unique.")
            starting_cash = int(self.cash_entry.get_text().strip())
            animation_speed = float(self.animation_speed_entry.get_text().strip() or "0")
            if animation_speed < 0:
                raise ValueError("Animation speed cannot be negative.")
            self.animation_speed = animation_speed
            self.animation_speed_text = self.animation_speed_entry.get_text().strip() or "0"
            self.ai_checkpoint_path = ai_player_setups[0]["checkpoint_path"] if ai_player_setups else self._default_checkpoint_option()
            self.controller.start_game(
                player_names,
                starting_cash,
                player_roles=player_roles,
                ai_checkpoint_path=self.ai_checkpoint_path,
                ai_player_setups=ai_player_setups,
            )
            self.screen_mode = "game"
            self._replay = None
            self._property_submenu_mode = None
            self._property_submenu_page = 0
            self._pending_ai_step = None
            self._build_game_screen()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._build_setup_screen()

    def _create_online_lobby_from_setup(self) -> None:
        try:
            self._ensure_local_backend_connection()
            self._capture_setup_inputs()
            host_player_name = self.player_names[0].strip() or "Host"
            starting_cash = int(self.cash_entry.get_text().strip())
            self.controller.create_online_lobby(host_player_name, self.player_count, starting_cash)
            try:
                self._register_discovery_lobby()
            except Exception as exc:
                self.controller.message_history.append(f"Discovery registration failed: {exc}")
                self.controller.status_message = f"Lobby created, but discovery registration failed: {exc}"
            self.screen_mode = "online_lobby"
            self._pending_ai_step = None
            self._build_online_lobby_screen()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._build_setup_screen()

    def _join_online_lobby_from_setup(self) -> None:
        try:
            self._stop_managed_backend()
            if self.controller.client is not None:
                self.controller.client.close()
                self.controller.client = None
            self._capture_setup_inputs()
            remote_host = self.online_host_entry.get_text().strip()
            remote_port_text = self.online_port_entry.get_text().strip()
            session_code = self.online_session_code_entry.get_text().strip().upper()
            player_name = self.online_player_name_entry.get_text().strip() or self.online_join_player_name
            reconnect_token = "" if not hasattr(self, "online_reconnect_token_entry") else self.online_reconnect_token_entry.get_text().strip()
            discovery_host = getattr(self, "online_discovery_host", "") if not hasattr(self, "online_discovery_host_entry") else self.online_discovery_host_entry.get_text().strip()
            discovery_port = int(getattr(self, "online_discovery_port", "0") if not hasattr(self, "online_discovery_port_entry") else self.online_discovery_port_entry.get_text().strip())
            if reconnect_token:
                self.online_reconnect_token = reconnect_token
            if not remote_host or not remote_port_text:
                resolved_endpoint = self._resolve_lobby_endpoint(session_code, discovery_host, discovery_port)
                remote_host = resolved_endpoint[0]
                remote_port = resolved_endpoint[1]
            else:
                remote_port = int(remote_port_text)
            self.online_remote_host = remote_host
            self.online_remote_port = str(remote_port)
            self.online_session_code = session_code
            self.online_join_player_name = player_name
            self.online_discovery_host = discovery_host
            self.online_discovery_port = str(discovery_port)
            self.controller.connect_to_backend(remote_host, remote_port, owns_server=False)
            if reconnect_token:
                self.controller.reconnect_online_slot(reconnect_token, session_code=session_code)
            else:
                self.controller.refresh_online_session(session_code=session_code)
            self.screen_mode = "game" if self.controller.online_session is not None and self.controller.online_session.state == "in_game" and self.controller.frontend_state is not None else "online_waiting"
            self._pending_ai_step = None
            if self.screen_mode == "game":
                self._build_game_screen()
            else:
                self._build_online_waiting_screen()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._build_setup_screen()

    def _ensure_local_backend_connection(self) -> None:
        existing_client = self.controller.client
        if (
            existing_client is not None
            and existing_client.owns_server
            and not existing_client.is_closed
            and self._managed_backend_process is not None
            and self._managed_backend_process.is_alive()
        ):
            return
        if existing_client is not None and existing_client.is_closed:
            self.controller.client = None
        self._stop_managed_backend()
        host = "127.0.0.1"
        port = find_free_port()
        self._managed_backend_process = mp.Process(
            target=run_backend_process,
            args=(host, port, self.debug_mode),
            name="monopoly-backend",
        )
        self._managed_backend_process.start()
        logger.info("Started managed backend process on %s:%d.", host, port)
        self.controller.connect_to_backend(host, port, owns_server=True)
        self.online_remote_host = host
        self.online_remote_port = str(port)

    def _stop_managed_backend(self) -> None:
        backend_process = self._managed_backend_process
        self._managed_backend_process = None
        if backend_process is None:
            return
        backend_process.join(timeout=0.1)
        if backend_process.is_alive():
            logger.warning("Managed backend process did not stop gracefully; terminating it.")
            backend_process.terminate()
            backend_process.join(timeout=1)
        logger.info("Managed backend process stopped.")

    def _build_setup_screen(self) -> None:
        self.screen_mode = "setup"
        self._clear_elements()
        self._setup_role_dropdowns = {}
        self._setup_checkpoint_dropdowns = {}

        title = self._track(UILabel(pygame.Rect(60, 36, 520, 40), "Monopoly Game Night", manager=self.manager))
        title.set_text("Monopoly Game Night")

        self._track(UILabel(pygame.Rect(60, 84, 120, 28), "Play Mode", manager=self.manager))
        self.setup_mode_dropdown = self._create_dropdown(
            [LOCAL_PLAY_MODE, "host_online", "join_online"],
            self.setup_game_mode,
            pygame.Rect(180, 82, 180, 32),
            None,
        )
        self._track(UILabel(pygame.Rect(380, 84, 680, 28), self._setup_mode_description(), manager=self.manager))

        is_local_mode = self.setup_game_mode == LOCAL_PLAY_MODE
        is_host_online_mode = self.setup_game_mode == "host_online"
        is_join_online_mode = self.setup_game_mode == "join_online"

        self.player_minus_button = self._track(UIButton(pygame.Rect(60, 120, 36, 36), "-", manager=self.manager))
        self.player_count_label = self._track(UILabel(pygame.Rect(108, 120, 200, 36), f"Players: {self.player_count}", manager=self.manager))
        self.player_plus_button = self._track(UIButton(pygame.Rect(320, 120, 36, 36), "+", manager=self.manager))
        self.player_minus_button.visible = not is_join_online_mode
        self.player_count_label.visible = not is_join_online_mode
        self.player_plus_button.visible = not is_join_online_mode

        starting_money_label = self._track(UILabel(pygame.Rect(60, 176, 180, 28), "Starting Money", manager=self.manager))
        self.cash_entry = self._track(UITextEntryLine(pygame.Rect(200, 174, 120, 32), manager=self.manager))
        self.cash_entry.set_text(self.starting_cash)
        starting_money_label.visible = not is_join_online_mode
        self.cash_entry.visible = not is_join_online_mode

        animation_label = self._track(UILabel(pygame.Rect(360, 176, 180, 28), "Animation Pace", manager=self.manager))
        self.animation_speed_entry = self._track(UITextEntryLine(pygame.Rect(500, 174, 120, 32), manager=self.manager))
        self.animation_speed_entry.set_text(self.animation_speed_text)
        animation_hint = self._track(UILabel(pygame.Rect(640, 176, 460, 28), "0 disables animation; higher values play faster.", manager=self.manager))
        animation_label.visible = is_local_mode
        self.animation_speed_entry.visible = is_local_mode
        animation_hint.visible = is_local_mode

        self.refresh_checkpoints_button = self._track(UIButton(pygame.Rect(60, 220, 220, 36), "Refresh AI Lineup", manager=self.manager))
        checkpoint_summary = self._ai_option_summary()
        checkpoint_summary_label = self._track(UILabel(pygame.Rect(300, 224, 760, 28), f"Available AI opponents: {checkpoint_summary}", manager=self.manager))
        self.refresh_checkpoints_button.visible = is_local_mode
        checkpoint_summary_label.visible = is_local_mode

        self._setup_fields = []
        if is_local_mode:
            roster_player_label = self._track(UILabel(pygame.Rect(60, 268, 180, 28), "Player", manager=self.manager))
            roster_role_label = self._track(UILabel(pygame.Rect(280, 268, 120, 28), "Seat Type", manager=self.manager))
            show_ai_setup_columns = any(self._setup_player_uses_ai_settings(index) for index in range(self.player_count))
            checkpoint_header = self._track(UILabel(pygame.Rect(420, 268, 360, 28), "AI Opponent", manager=self.manager))
            checkpoint_header.visible = show_ai_setup_columns
            cooldown_header = self._track(UILabel(pygame.Rect(810, 268, 180, 28), "AI Pause (s)", manager=self.manager))
            cooldown_header.visible = show_ai_setup_columns
            roster_player_label.visible = True
            roster_role_label.visible = True

            for index in range(6):
                y = 304 + index * 52
                label = self._track(UILabel(pygame.Rect(60, y, 160, 28), f"Player {index + 1}", manager=self.manager))
                entry = self._track(UITextEntryLine(pygame.Rect(140, y, 120, 32), manager=self.manager))
                entry.set_text(self.player_names[index])
                role_dropdown = self._create_dropdown(sorted(PLAYER_ROLES), self.player_roles[index], pygame.Rect(280, y, 120, 32), None)
                enabled = index < self.player_count
                ai_fields_visible = enabled and self._setup_player_uses_ai_settings(index)
                checkpoint_dropdown: UIDropDownMenu | None = None
                cooldown_entry: UITextEntryLine | None = None
                if ai_fields_visible:
                    checkpoint_dropdown = self._create_dropdown(
                        self._checkpoint_dropdown_options(self.ai_checkpoint_paths[index]),
                        self.ai_checkpoint_paths[index],
                        pygame.Rect(420, y, 360, 32),
                        None,
                    )
                    cooldown_entry = self._track(UITextEntryLine(pygame.Rect(810, y, 120, 32), manager=self.manager))
                    cooldown_entry.set_text(self.ai_cooldown_texts[index])
                label.visible = enabled
                entry.visible = enabled
                role_dropdown.visible = enabled
                self._setup_role_dropdowns[role_dropdown] = index
                if checkpoint_dropdown is not None:
                    self._setup_checkpoint_dropdowns[checkpoint_dropdown] = index
                self._setup_fields.append(
                    SetupFieldState(
                        label=label,
                        entry=entry,
                        role_dropdown=role_dropdown,
                        checkpoint_dropdown=checkpoint_dropdown,
                        cooldown_entry=cooldown_entry,
                    )
                )

        host_hint = self._track(UILabel(pygame.Rect(60, 232, 900, 28), "Host creates the lobby now and manages slot open or close state there.", manager=self.manager))
        host_hint.visible = is_host_online_mode
        self.host_hint_label = host_hint

        discovery_host_label_rect = pygame.Rect(560, 284, 110, 28)
        discovery_host_entry_rect = pygame.Rect(670, 282, 180, 32)
        discovery_port_label_rect = pygame.Rect(860, 284, 110, 28)
        discovery_port_entry_rect = pygame.Rect(970, 282, 60, 32)
        if is_host_online_mode:
            discovery_host_label_rect = pygame.Rect(180, 306, 110, 28)
            discovery_host_entry_rect = pygame.Rect(300, 304, 240, 32)
            discovery_port_label_rect = pygame.Rect(570, 306, 110, 28)
            discovery_port_entry_rect = pygame.Rect(690, 304, 90, 32)

        self.online_host_label = self._track(UILabel(pygame.Rect(60, 232, 120, 28), "Host Address", manager=self.manager))
        self.online_host_entry = self._track(UITextEntryLine(pygame.Rect(180, 230, 180, 32), manager=self.manager))
        self.online_host_entry.set_text(self.online_remote_host)
        self.online_port_label = self._track(UILabel(pygame.Rect(395, 232, 60, 28), "Port", manager=self.manager))
        self.online_port_entry = self._track(UITextEntryLine(pygame.Rect(445, 230, 90, 32), manager=self.manager))
        self.online_port_entry.set_text(self.online_remote_port)
        self.online_session_code_label = self._track(UILabel(pygame.Rect(575, 232, 90, 28), "Lobby Code", manager=self.manager))
        self.online_session_code_entry = self._track(UITextEntryLine(pygame.Rect(665, 230, 120, 32), manager=self.manager))
        self.online_session_code_entry.set_text(self.online_session_code)
        self.online_player_name_label = self._track(UILabel(pygame.Rect(815, 232, 80, 28), "Your Name", manager=self.manager))
        self.online_player_name_entry = self._track(UITextEntryLine(pygame.Rect(895, 230, 135, 32), manager=self.manager))
        self.online_player_name_entry.set_text(self.online_join_player_name)
        self.online_reconnect_token_label = self._track(UILabel(pygame.Rect(60, 284, 140, 28), "Reconnect Token", manager=self.manager))
        self.online_reconnect_token_entry = self._track(UITextEntryLine(pygame.Rect(180, 282, 350, 32), manager=self.manager))
        self.online_reconnect_token_entry.set_text(self.online_reconnect_token)
        self.online_discovery_host_label = self._track(UILabel(discovery_host_label_rect, "Discovery Host", manager=self.manager))
        self.online_discovery_host_entry = self._track(UITextEntryLine(discovery_host_entry_rect, manager=self.manager))
        self.online_discovery_host_entry.set_text(self.online_discovery_host)
        self.online_discovery_port_label = self._track(UILabel(discovery_port_label_rect, "Discovery Port", manager=self.manager))
        self.online_discovery_port_entry = self._track(UITextEntryLine(discovery_port_entry_rect, manager=self.manager))
        self.online_discovery_port_entry.set_text(self.online_discovery_port)
        discovery_hint = self._track(UILabel(pygame.Rect(60, 332, 970, 28), "Leave host address blank to resolve the lobby code through the discovery service. Fill reconnect token to reclaim a disconnected seat.", manager=self.manager))
        for element in (
            self.online_host_label,
            self.online_host_entry,
            self.online_port_label,
            self.online_port_entry,
            self.online_session_code_label,
            self.online_session_code_entry,
            self.online_player_name_label,
            self.online_player_name_entry,
            self.online_reconnect_token_label,
            self.online_reconnect_token_entry,
            self.online_discovery_host_label,
            self.online_discovery_host_entry,
            self.online_discovery_port_label,
            self.online_discovery_port_entry,
            discovery_hint,
        ):
            element.visible = is_join_online_mode

        host_discovery_hint = self._track(UILabel(pygame.Rect(60, 268, 970, 28), "Optionally register this lobby with the discovery service so joiners only need the lobby code.", manager=self.manager))
        host_discovery_hint.visible = is_host_online_mode
        self.host_discovery_hint_label = host_discovery_hint
        for element in (
            self.online_discovery_host_label,
            self.online_discovery_host_entry,
            self.online_discovery_port_label,
            self.online_discovery_port_entry,
        ):
            element.visible = is_join_online_mode or is_host_online_mode

        self.start_button = self._track(UIButton(pygame.Rect(60, 620, 180, 42), self._setup_primary_button_label(), manager=self.manager))
        self.load_setup_button = self._track(UIButton(pygame.Rect(256, 620, 180, 42), "Load Game", manager=self.manager))
        self.load_setup_button.visible = is_local_mode

    def _build_online_lobby_screen(self) -> None:
        self._clear_elements()
        self._host_lobby_slot_dropdowns = {}
        self._host_lobby_ai_checkpoint_dropdowns = {}
        self._host_lobby_ai_speed_entries = {}
        self._host_lobby_ai_apply_buttons = {}
        self.screen_mode = "online_lobby"
        session = self.controller.online_session
        if session is None:
            self.controller.set_error("Online lobby is not available.")
            self._build_setup_screen()
            return

        self._track(UILabel(pygame.Rect(60, 36, 520, 40), "Host Online Lobby", manager=self.manager))
        self._track(UILabel(pygame.Rect(60, 84, 320, 28), f"Lobby code: {session.session_code}", manager=self.manager))
        self._track(UILabel(pygame.Rect(360, 84, 520, 28), f"Host endpoint: {self.controller.client.host}:{self.controller.client.port}", manager=self.manager))
        state_text = f"State: {session.state} | Starting cash: ${session.starting_cash} | Only the host can start."
        if session.state == "paused" and session.paused_reason == "player_disconnected":
            paused_seat_label = "unknown"
            if session.paused_seat_index is not None:
                paused_seat_label = str(session.paused_seat_index + 1)
            state_text = f"State: paused | Waiting for seat {paused_seat_label} to reconnect or be replaced with AI."
        self._track(UILabel(pygame.Rect(60, 122, 900, 28), state_text, manager=self.manager))

        seat_panel = self._track(UIPanel(pygame.Rect(60, 170, 980, 410), manager=self.manager))
        self._build_panel_title(seat_panel, 980, "Lobby Seats")
        for index, seat in enumerate(session.seats):
            row_y = 42 + index * 78
            self._track(UILabel(pygame.Rect(16, row_y, 420, 28), self._online_seat_summary(seat), manager=self.manager, container=seat_panel))
            if seat.is_host:
                continue
            if session.state == "lobby" and seat.status in {"open", "closed", "ai"}:
                self._track(UILabel(pygame.Rect(452, row_y, 90, 28), "Seat State", manager=self.manager, container=seat_panel))
                seat_mode_dropdown = self._create_dropdown(
                    ["open", "ai", "closed"],
                    seat.status,
                    pygame.Rect(540, row_y - 2, 160, 32),
                    seat_panel,
                    detached=True,
                )
                self._host_lobby_slot_dropdowns[seat_mode_dropdown] = seat.seat_index
                if seat.status == "ai":
                    self._track(UILabel(pygame.Rect(720, row_y, 220, 28), seat.player_name or f"AI Player {seat.seat_index + 1}", manager=self.manager, container=seat_panel))
                    checkpoint_selected = getattr(seat, "checkpoint_path", None) or self.ai_checkpoint_paths[seat.seat_index]
                    self.ai_checkpoint_paths[seat.seat_index] = checkpoint_selected
                    action_cooldown_seconds = getattr(seat, "action_cooldown_seconds", None)
                    cooldown_text = str(action_cooldown_seconds if action_cooldown_seconds is not None else self.ai_cooldown_texts[seat.seat_index])
                    self.ai_cooldown_texts[seat.seat_index] = cooldown_text
                    self._track(UILabel(pygame.Rect(452, row_y + 34, 86, 24), "Checkpoint", manager=self.manager, container=seat_panel))
                    checkpoint_dropdown = self._create_dropdown(
                        self._checkpoint_dropdown_options(checkpoint_selected),
                        checkpoint_selected,
                        pygame.Rect(540, row_y + 32, 220, 30),
                        seat_panel,
                        detached=True,
                    )
                    self._host_lobby_ai_checkpoint_dropdowns[checkpoint_dropdown] = seat.seat_index
                    self._track(UILabel(pygame.Rect(774, row_y + 34, 46, 24), "Speed", manager=self.manager, container=seat_panel))
                    speed_entry = self._track(UITextEntryLine(pygame.Rect(822, row_y + 32, 70, 30), manager=self.manager, container=seat_panel))
                    speed_entry.set_text(cooldown_text)
                    self._host_lobby_ai_speed_entries[seat.seat_index] = speed_entry
                    apply_button = self._track(UIButton(pygame.Rect(904, row_y + 31, 60, 30), "Apply", manager=self.manager, container=seat_panel))
                    self._host_lobby_ai_apply_buttons[apply_button] = seat.seat_index
            elif session.state == "lobby" and seat.status == "connected":
                self._track(UILabel(pygame.Rect(540, row_y, 220, 28), "Waiting for remote player", manager=self.manager, container=seat_panel))
            elif seat.status == "disconnected" and session.state == "paused":
                replace_button = self._track(UIButton(pygame.Rect(560, row_y - 4, 150, 32), "Take Over With AI", manager=self.manager, container=seat_panel))
                self._ui_commands[replace_button] = ("host_replace_disconnected_with_ai", seat.seat_index)
            elif seat.status in {"ai", "disconnected"}:
                clear_button = self._track(UIButton(pygame.Rect(560, row_y - 4, 110, 32), "Clear Seat", manager=self.manager, container=seat_panel))
                self._ui_commands[clear_button] = ("host_clear_online_slot", seat.seat_index)
                if seat.status == "disconnected":
                    close_button = self._track(UIButton(pygame.Rect(684, row_y - 4, 110, 32), "Close Slot", manager=self.manager, container=seat_panel))
                    self._ui_commands[close_button] = ("host_close_online_slot", seat.seat_index)

        start_online_button = self._track(UIButton(pygame.Rect(60, 580, 180, 42), "Start Online Match", manager=self.manager))
        back_button = self._track(UIButton(pygame.Rect(256, 580, 180, 42), "Back to Setup", manager=self.manager))
        self._ui_commands[start_online_button] = ("start_online_match", None)
        self._ui_commands[back_button] = ("back_to_setup", None)

    def _build_online_waiting_screen(self) -> None:
        self._clear_elements()
        self._host_lobby_slot_dropdowns = {}
        self.screen_mode = "online_waiting"
        session = self.controller.online_session
        if session is None:
            self.controller.set_error("Online lobby is not available.")
            self._build_setup_screen()
            return

        if session.state == "in_game" and self.controller.frontend_state is not None:
            self.screen_mode = "game"
            self._build_game_screen()
            return

        self._track(UILabel(pygame.Rect(60, 36, 520, 40), "Join Online Lobby", manager=self.manager))
        self._track(UILabel(pygame.Rect(60, 84, 420, 28), f"Connected to: {self.controller.client.host}:{self.controller.client.port}", manager=self.manager))
        self._track(UILabel(pygame.Rect(500, 84, 320, 28), f"Lobby code: {session.session_code}", manager=self.manager))
        self._track(UILabel(pygame.Rect(60, 122, 880, 28), self._online_waiting_status_text(), manager=self.manager))

        seat_panel = self._track(UIPanel(pygame.Rect(60, 170, 980, 380), manager=self.manager))
        self._build_panel_title(seat_panel, 980, "Available Seats")
        for index, seat in enumerate(session.seats):
            row_y = 42 + index * 54
            self._track(UILabel(pygame.Rect(16, row_y, 520, 28), self._online_seat_summary(seat), manager=self.manager, container=seat_panel))
            if seat.is_claimable and self.controller.session_token is None:
                join_button = self._track(UIButton(pygame.Rect(560, row_y - 4, 120, 32), "Join Seat", manager=self.manager, container=seat_panel))
                self._ui_commands[join_button] = ("join_online_seat", seat.seat_index)

        back_button = self._track(UIButton(pygame.Rect(60, 580, 180, 42), "Back to Setup", manager=self.manager))
        self._ui_commands[back_button] = ("back_to_setup", None)

    def _setup_mode_description(self) -> str:
        descriptions = {
            LOCAL_PLAY_MODE: "Local match on this machine. Mix humans and AI and pass the device between local human players.",
            "host_online": "Create a host-controlled online lobby and invite remote players.",
            "join_online": "Connect to a remote host by address, port, and lobby code.",
        }
        return descriptions.get(self.setup_game_mode, "")

    def _setup_primary_button_label(self) -> str:
        labels = {
            LOCAL_PLAY_MODE: "Start Match",
            "host_online": "Create Lobby",
            "join_online": "Connect",
        }
        if self.setup_game_mode == "join_online" and getattr(self, "online_reconnect_token", "").strip():
            return "Reconnect"
        return labels.get(self.setup_game_mode, "Start")

    def _online_seat_summary(self, seat: Any) -> str:
        name = seat.player_name or "Unassigned"
        role = getattr(seat, "player_role", HUMAN_ROLE)
        host_suffix = " | host" if getattr(seat, "is_host", False) else ""
        return f"Seat {seat.seat_index + 1}: {seat.status} | {name} | {role}{host_suffix}"

    def _online_waiting_status_text(self) -> str:
        session = self.controller.online_session
        if session is not None and session.state == "paused" and session.paused_reason == "player_disconnected":
            disconnected_name = None
            if session.paused_seat_index is not None and 0 <= session.paused_seat_index < len(session.seats):
                disconnected_name = session.seats[session.paused_seat_index].player_name
            if disconnected_name:
                return f"Match paused while {disconnected_name} reconnects or the host replaces that seat with AI."
            return "Match paused while a disconnected player reconnects or the host replaces that seat with AI."
        if self.controller.online_player_name is None:
            return "Pick any open seat that the host has made available, then wait for the host to start the match."
        reconnect_suffix = ""
        reconnect_token = getattr(self.controller, "reconnect_token", None)
        if reconnect_token:
            reconnect_suffix = f" Reconnect token: {reconnect_token}."
        return f"You are assigned to {self.controller.online_player_name}. Waiting for the host to start the match.{reconnect_suffix}"

    def _build_game_screen(self) -> None:
        self._clear_elements()
        self._rebuild_game_ui()

    def _rebuild_game_ui(self) -> None:
        self._clear_elements()
        state = self.controller.frontend_state
        if state is None:
            self._build_setup_screen()
            return
        debug_full_state = self.controller.get_debug_state() if self.debug_mode else None

        sidebar_x = self.board_rect.right + theme.BOARD_MARGIN
        sidebar_width = self.sidebar_width
        window_width, window_height = self.screen.get_size()

        header_text = (
            f"Round {state.game_view.turn_counter + 1} | Up now: {state.game_view.current_player_name or 'n/a'} | "
            f"Making the call: {state.active_turn_plan.player_name} | Stage: {self._friendly_phase_name(state.game_view.current_turn_phase)}"
        )
        self._track(UILabel(pygame.Rect(theme.BOARD_MARGIN, 12, 1120, 30), header_text, manager=self.manager))
        button_specs = [("save_button", "Save", None), ("load_button", "Load", None), (None, "Save Replay", ("save_replay", None)), (None, "Load Replay", ("load_replay", None))]
        button_width = 100
        button_gap = 10
        right_edge = window_width - 16
        first_button_left = right_edge
        for attribute_name, label, command in reversed(button_specs):
            right_edge -= button_width
            button = self._track(UIButton(pygame.Rect(right_edge, 12, button_width, 32), label, manager=self.manager))
            first_button_left = min(first_button_left, right_edge)
            if attribute_name is not None:
                setattr(self, attribute_name, button)
            if command is not None:
                self._ui_commands[button] = command
            right_edge -= button_gap
        debug_badge_bottom = 70
        if debug_full_state is not None:
            debug_badge_bottom = self._build_header_badges(debug_full_state, sidebar_x, sidebar_width, state.game_view.players)

        if self.debug_mode:
            actions_panel_top = self._build_debug_side_panels(state, debug_full_state, sidebar_x, sidebar_width, debug_badge_bottom + 14)
        else:
            self._build_standard_side_panels(state, sidebar_x, sidebar_width, window_height)
            self._maybe_open_pending_trade_prompt(state)
            return

        available_sidebar_height = max(120, window_height - theme.STATUS_BAR_HEIGHT - actions_panel_top - 20)
        if available_sidebar_height < 320:
            log_height = max(80, available_sidebar_height // 2)
            action_panel_height = max(80, available_sidebar_height - log_height - 10)
        else:
            log_height = max(140, min(220, available_sidebar_height // 2))
            action_panel_height = max(140, available_sidebar_height - log_height - 10)

        actions_panel = self._track(UIPanel(pygame.Rect(sidebar_x, actions_panel_top, sidebar_width, action_panel_height), manager=self.manager))
        self._action_buttons = {}
        display_actions = self._display_actions(state.active_turn_plan.legal_actions)
        ai_turn_active = self._is_ai_turn_active(state)
        if display_actions:
            button_height = 28
            button_spacing = 6
            max_buttons = max(1, (action_panel_height - 20) // (button_height + button_spacing))
            paged_actions = self._paginate_property_submenu_actions(display_actions, max_buttons)
            for index, action in enumerate(paged_actions):
                description = action.description if isinstance(action, LegalActionOption) else action[2]
                button = self._track(
                    UIButton(
                        pygame.Rect(10, 10 + index * (button_height + button_spacing), sidebar_width - 20, button_height),
                        description,
                        manager=self.manager,
                        container=actions_panel,
                    )
                )
                if ai_turn_active:
                    button.disable()
                self._action_buttons[button] = action
        else:
            message = "AI is playing. Human input is temporarily disabled." if ai_turn_active else "No legal actions available."
            self._track(UITextBox(message, pygame.Rect(10, 10, sidebar_width - 20, 40), manager=self.manager, container=actions_panel))

        log_top = actions_panel_top + action_panel_height + 10
        log_panel = self._track(UIPanel(pygame.Rect(sidebar_x, log_top, sidebar_width, log_height), manager=self.manager))
        self._track(UITextBox(self._log_html(log_height), pygame.Rect(10, 10, sidebar_width - 20, log_height - 20), manager=self.manager, container=log_panel))
        self._maybe_open_pending_trade_prompt(state)

    def _build_debug_side_panels(
        self,
        state: FrontendStateView,
        full_state: dict[str, Any] | None,
        sidebar_x: int,
        sidebar_width: int,
        top_y: int,
    ) -> int:
        if full_state is None:
            raise ValueError("Debug panels require debug state.")
        self._clear_debug_validation_state()
        runtime = full_state["runtime"]
        selected_space = state.board_spaces[self.controller.selected_space_index]
        player_names = [player.name for player in state.game_view.players]
        selected_player_name = self._debug_selected_player_name or state.game_view.current_player_name or player_names[0]
        if selected_player_name not in player_names:
            selected_player_name = player_names[0]
        self._debug_selected_player_name = selected_player_name
        selected_player = next(player for player in state.game_view.players if player.name == selected_player_name)
        queued_rolls = runtime.get("debug_next_rolls_by_player", {}).get(selected_player.name, [])
        queued_roll_text = "none" if not queued_rolls else ", ".join(f"{roll[0]} + {roll[1]}" for roll in queued_rolls)
        valid_tabs = ("player", "runtime", "property", "pending")
        if self._debug_active_tab not in valid_tabs:
            self._debug_active_tab = "player"

        gap = theme.PANEL_GAP
        tab_height = 34
        tab_width = max(120, (sidebar_width - gap * (len(valid_tabs) - 1)) // len(valid_tabs))
        for index, tab_name in enumerate(valid_tabs):
            tab_button = self._track(
                UIButton(
                    pygame.Rect(sidebar_x + index * (tab_width + gap), top_y, tab_width, tab_height),
                    tab_name.title(),
                    manager=self.manager,
                )
            )
            if tab_name == self._debug_active_tab:
                tab_button.disable()
            self._ui_commands[tab_button] = ("debug_tab", tab_name)

        panel_y = top_y + tab_height + 10
        panel_height = 412
        editor_panel = self._track(UIPanel(pygame.Rect(sidebar_x, panel_y, sidebar_width, panel_height), manager=self.manager))

        panel_state = DebugPanelState(
            selected_player_name=selected_player.name,
            player_select=None,
            player_role_dropdown=None,
            player_in_jail_dropdown=None,
            player_bankrupt_dropdown=None,
            player_cash_entry=None,
            player_position_entry=None,
            player_jail_cards_entry=None,
            player_jail_turns_entry=None,
            queued_roll_label=None,
            next_die_one_entry=None,
            next_die_two_entry=None,
            apply_player_button=None,
            set_next_roll_button=None,
            clear_next_roll_button=None,
            runtime_current_player_dropdown=None,
            runtime_turn_phase_dropdown=None,
            runtime_pending_action_dropdown=None,
            runtime_turn_counter_entry=None,
            runtime_houses_entry=None,
            runtime_hotels_entry=None,
            runtime_continuation_player_dropdown=None,
            runtime_continuation_doubles_entry=None,
            runtime_continuation_rolled_double_dropdown=None,
            runtime_auction_current_bid_entry=None,
            runtime_auction_bidder_index_entry=None,
            runtime_auction_winner_dropdown=None,
            runtime_trade_proposer_dropdown=None,
            runtime_trade_receiver_dropdown=None,
            runtime_trade_proposer_cash_entry=None,
            runtime_trade_receiver_cash_entry=None,
            runtime_trade_note_entry=None,
            apply_runtime_button=None,
            property_space_index=None,
            property_owner_dropdown=None,
            property_mortgaged_dropdown=None,
            property_building_dropdown=None,
            apply_property_button=None,
        )

        if self._debug_active_tab == "player":
            self._build_panel_title(editor_panel, sidebar_width, "Player Editor")
            self._populate_debug_player_panel(editor_panel, sidebar_width, selected_player, player_names, queued_roll_text, panel_state)
        elif self._debug_active_tab == "runtime":
            self._build_panel_title(editor_panel, sidebar_width, "Runtime Editor")
            self._populate_debug_runtime_panel(editor_panel, sidebar_width, full_state, player_names, panel_state)
        elif self._debug_active_tab == "property":
            self._build_panel_title(editor_panel, sidebar_width, "Property Editor")
            self._populate_debug_property_panel(editor_panel, sidebar_width, state, selected_space, player_names, panel_state)
        else:
            self._build_panel_title(editor_panel, sidebar_width, "Pending Editor")
            self._populate_debug_pending_panel(editor_panel, sidebar_width, state, full_state, player_names, panel_state)

        self._debug_panel = panel_state
        return panel_y + panel_height + gap

    def _populate_debug_player_panel(
        self,
        panel: UIPanel,
        panel_width: int,
        selected_player: PlayerView,
        player_names: list[str],
        queued_roll_text: str,
        panel_state: DebugPanelState,
    ) -> None:
        row_gap = 40
        left_x = 18
        right_x = panel_width // 2 + 18
        field_width = max(120, panel_width // 2 - 54)
        self._track(UILabel(pygame.Rect(left_x, 42, 70, 22), "Player", manager=self.manager, container=panel))
        panel_state.player_select = self._create_dropdown(player_names, selected_player.name, pygame.Rect(left_x + 74, 38, panel_width - left_x - 92, 30), panel, detached=True)
        self._track(UILabel(pygame.Rect(left_x, 42 + row_gap, 70, 22), "Cash", manager=self.manager, container=panel))
        panel_state.player_cash_entry = self._track(UITextEntryLine(pygame.Rect(left_x + 74, 40 + row_gap, field_width, 28), manager=self.manager, container=panel))
        panel_state.player_cash_entry.set_text(str(selected_player.cash))
        self._track(UILabel(pygame.Rect(right_x, 42 + row_gap, 56, 22), "Pos", manager=self.manager, container=panel))
        panel_state.player_position_entry = self._track(UITextEntryLine(pygame.Rect(right_x + 50, 40 + row_gap, field_width - 50, 28), manager=self.manager, container=panel))
        panel_state.player_position_entry.set_text(str(selected_player.position))
        self._track(UILabel(pygame.Rect(left_x, 42 + row_gap * 2, 70, 22), "Jail Cards", manager=self.manager, container=panel))
        panel_state.player_jail_cards_entry = self._track(UITextEntryLine(pygame.Rect(left_x + 74, 40 + row_gap * 2, field_width, 28), manager=self.manager, container=panel))
        panel_state.player_jail_cards_entry.set_text(str(selected_player.get_out_of_jail_cards))
        self._track(UILabel(pygame.Rect(right_x, 42 + row_gap * 2, 56, 22), "Turns", manager=self.manager, container=panel))
        panel_state.player_jail_turns_entry = self._track(UITextEntryLine(pygame.Rect(right_x + 50, 40 + row_gap * 2, field_width - 50, 28), manager=self.manager, container=panel))
        panel_state.player_jail_turns_entry.set_text(str(selected_player.jail_turns))
        self._track(UILabel(pygame.Rect(left_x, 42 + row_gap * 3, 70, 22), "Control", manager=self.manager, container=panel))
        panel_state.player_role_dropdown = self._create_dropdown(sorted(PLAYER_ROLES), selected_player.role, pygame.Rect(left_x + 74, 38 + row_gap * 3, field_width, 30), panel, detached=True)
        self._track(UILabel(pygame.Rect(right_x, 42 + row_gap * 3, 56, 22), "Jailed", manager=self.manager, container=panel))
        panel_state.player_in_jail_dropdown = self._create_dropdown(["False", "True"], "True" if selected_player.in_jail else "False", pygame.Rect(right_x + 50, 38 + row_gap * 3, field_width - 50, 30), panel, detached=True)
        self._track(UILabel(pygame.Rect(left_x, 42 + row_gap * 4, 70, 22), "Bankrupt", manager=self.manager, container=panel))
        panel_state.player_bankrupt_dropdown = self._create_dropdown(["False", "True"], "True" if selected_player.is_bankrupt else "False", pygame.Rect(left_x + 74, 38 + row_gap * 4, field_width, 30), panel, detached=True)
        panel_state.apply_player_button = self._track(UIButton(pygame.Rect(panel_width - 154, 38 + row_gap * 4, 136, 32), "Apply Player", manager=self.manager, container=panel))
        self._ui_commands[panel_state.apply_player_button] = ("apply_debug_player", None)
        panel_state.queued_roll_label = self._track(UILabel(pygame.Rect(left_x, 42 + row_gap * 5, panel_width - 36, 20), f"Queued: {queued_roll_text}", manager=self.manager, container=panel))
        self._track(UILabel(pygame.Rect(left_x, 42 + row_gap * 6, 70, 22), "Next Roll", manager=self.manager, container=panel))
        panel_state.next_die_one_entry = self._track(UITextEntryLine(pygame.Rect(left_x + 74, 40 + row_gap * 6, 48, 26), manager=self.manager, container=panel))
        panel_state.next_die_one_entry.set_text("1")
        panel_state.next_die_two_entry = self._track(UITextEntryLine(pygame.Rect(left_x + 128, 40 + row_gap * 6, 48, 26), manager=self.manager, container=panel))
        panel_state.next_die_two_entry.set_text("1")
        panel_state.set_next_roll_button = self._track(UIButton(pygame.Rect(left_x + 184, 38 + row_gap * 6, 62, 30), "Set", manager=self.manager, container=panel))
        panel_state.clear_next_roll_button = self._track(UIButton(pygame.Rect(left_x + 254, 38 + row_gap * 6, 68, 30), "Clear", manager=self.manager, container=panel))
        self._ui_commands[panel_state.set_next_roll_button] = ("set_debug_roll", None)
        self._ui_commands[panel_state.clear_next_roll_button] = ("clear_debug_roll", None)

    def _populate_debug_runtime_panel(
        self,
        panel: UIPanel,
        panel_width: int,
        full_state: dict[str, Any],
        player_names: list[str],
        panel_state: DebugPanelState,
    ) -> None:
        runtime = full_state["runtime"]
        current_player_name = str(runtime.get("current_player_name") or player_names[0])
        continuation = runtime.get("pending_turn_continuation")
        left_x = 18
        mid_x = panel_width // 2 + 12
        field_width = max(150, panel_width // 2 - 40)
        self._track(UILabel(pygame.Rect(left_x, 42, 82, 22), "Current", manager=self.manager, container=panel))
        panel_state.runtime_current_player_dropdown = self._create_dropdown(player_names, current_player_name, pygame.Rect(left_x + 86, 38, field_width, 30), panel, detached=True)
        self._register_debug_field("runtime.current_player", panel_state.runtime_current_player_dropdown)
        self._track(UILabel(pygame.Rect(mid_x, 42, 62, 22), "Phase", manager=self.manager, container=panel))
        panel_state.runtime_turn_phase_dropdown = self._create_dropdown([PRE_ROLL_PHASE, IN_TURN_PHASE, POST_ROLL_PHASE], str(runtime.get("current_turn_phase", PRE_ROLL_PHASE)), pygame.Rect(mid_x + 58, 38, field_width - 20, 30), panel, detached=True)
        self._register_debug_field("runtime.phase", panel_state.runtime_turn_phase_dropdown)
        self._track(UILabel(pygame.Rect(left_x, 82, 82, 22), "Turn", manager=self.manager, container=panel))
        panel_state.runtime_turn_counter_entry = self._track(UITextEntryLine(pygame.Rect(left_x + 86, 80, 90, 28), manager=self.manager, container=panel))
        panel_state.runtime_turn_counter_entry.set_text(str(runtime.get("turn_counter", 0)))
        self._register_debug_field("runtime.turn_counter", panel_state.runtime_turn_counter_entry)
        self._track(UILabel(pygame.Rect(mid_x, 82, 62, 22), "Pending", manager=self.manager, container=panel))
        panel_state.runtime_pending_action_dropdown = self._create_dropdown(["none", "purchase", "auction", "jail", "property", "trade"], self._current_pending_action_kind(runtime), pygame.Rect(mid_x + 58, 78, field_width - 20, 30), panel, detached=True)
        self._register_debug_field("runtime.pending_kind", panel_state.runtime_pending_action_dropdown)
        self._track(UILabel(pygame.Rect(left_x, 122, 82, 22), "Houses", manager=self.manager, container=panel))
        panel_state.runtime_houses_entry = self._track(UITextEntryLine(pygame.Rect(left_x + 86, 120, 90, 28), manager=self.manager, container=panel))
        panel_state.runtime_houses_entry.set_text(str(runtime.get("houses_remaining", 0)))
        self._register_debug_field("runtime.houses", panel_state.runtime_houses_entry)
        self._track(UILabel(pygame.Rect(mid_x, 122, 62, 22), "Hotels", manager=self.manager, container=panel))
        panel_state.runtime_hotels_entry = self._track(UITextEntryLine(pygame.Rect(mid_x + 58, 120, 90, 28), manager=self.manager, container=panel))
        panel_state.runtime_hotels_entry.set_text(str(runtime.get("hotels_remaining", 0)))
        self._register_debug_field("runtime.hotels", panel_state.runtime_hotels_entry)
        continuation_player = "none" if continuation is None else str(continuation.get("player_name") or "none")
        self._track(UILabel(pygame.Rect(left_x, 162, 82, 22), "Continue", manager=self.manager, container=panel))
        panel_state.runtime_continuation_player_dropdown = self._create_dropdown(["none", *player_names], continuation_player, pygame.Rect(left_x + 86, 158, field_width, 30), panel, detached=True)
        self._register_debug_field("runtime.continuation_player", panel_state.runtime_continuation_player_dropdown)
        self._track(UILabel(pygame.Rect(mid_x, 162, 62, 22), "Rolled", manager=self.manager, container=panel))
        panel_state.runtime_continuation_rolled_double_dropdown = self._create_dropdown(["False", "True"], "False" if continuation is None else ("True" if continuation.get("rolled_double") else "False"), pygame.Rect(mid_x + 58, 158, field_width - 20, 30), panel, detached=True)
        self._register_debug_field("runtime.continuation_rolled_double", panel_state.runtime_continuation_rolled_double_dropdown)
        self._track(UILabel(pygame.Rect(left_x, 202, 82, 22), "Doubles", manager=self.manager, container=panel))
        panel_state.runtime_continuation_doubles_entry = self._track(UITextEntryLine(pygame.Rect(left_x + 86, 200, 90, 28), manager=self.manager, container=panel))
        panel_state.runtime_continuation_doubles_entry.set_text(str(0 if continuation is None else continuation.get("doubles_in_row", 0)))
        self._register_debug_field("runtime.continuation_doubles", panel_state.runtime_continuation_doubles_entry)
        self._track(UITextBox(self._runtime_panel_html(full_state), pygame.Rect(18, 248, panel_width - 36, 76), manager=self.manager, container=panel))
        panel_state.apply_runtime_button = self._track(UIButton(pygame.Rect(panel_width - 150, 350, 132, 32), "Apply Runtime", manager=self.manager, container=panel))
        self._ui_commands[panel_state.apply_runtime_button] = ("apply_debug_runtime", None)

    def _populate_debug_property_panel(
        self,
        panel: UIPanel,
        panel_width: int,
        state: FrontendStateView,
        selected_space: Any,
        player_names: list[str],
        panel_state: DebugPanelState,
    ) -> None:
        self._track(UITextBox(self._selected_space_html(state), pygame.Rect(18, 42, panel_width - 36, 126), manager=self.manager, container=panel))
        if selected_space.price is None:
            self._track(UITextBox("Select an ownable space to edit owner, mortgage, or building state.", pygame.Rect(18, 182, panel_width - 36, 88), manager=self.manager, container=panel))
            return
        owner_options = ["bank", *player_names]
        self._track(UILabel(pygame.Rect(18, 188, 82, 24), "Owner", manager=self.manager, container=panel))
        panel_state.property_owner_dropdown = self._create_dropdown(owner_options, selected_space.owner_name or "bank", pygame.Rect(110, 184, panel_width - 128, 30), panel, detached=True)
        self._track(UILabel(pygame.Rect(18, 228, 82, 24), "Mortgage", manager=self.manager, container=panel))
        panel_state.property_mortgaged_dropdown = self._create_dropdown(["False", "True"], "True" if selected_space.mortgaged else "False", pygame.Rect(110, 224, 180, 30), panel, detached=True)
        if selected_space.building_count is not None:
            self._track(UILabel(pygame.Rect(306, 228, 82, 24), "Level", manager=self.manager, container=panel))
            panel_state.property_building_dropdown = self._create_dropdown([str(index) for index in range(6)], str(selected_space.building_count), pygame.Rect(368, 224, 120, 30), panel, detached=True)
        panel_state.property_space_index = selected_space.index
        panel_state.apply_property_button = self._track(UIButton(pygame.Rect(panel_width - 146, 350, 128, 32), "Apply Space", manager=self.manager, container=panel))
        self._ui_commands[panel_state.apply_property_button] = ("apply_debug_property", None)

    def _populate_debug_pending_panel(
        self,
        panel: UIPanel,
        panel_width: int,
        state: FrontendStateView,
        full_state: dict[str, Any],
        player_names: list[str],
        panel_state: DebugPanelState,
    ) -> None:
        self._track(UITextBox(self._turn_context_html(state), pygame.Rect(18, 42, panel_width - 36, 84), manager=self.manager, container=panel))
        self._track(UITextBox(self._pending_action_editor_html(full_state), pygame.Rect(18, 136, panel_width - 36, 76), manager=self.manager, container=panel))
        pending_controls = self._build_debug_pending_controls(panel, panel_width - 16, full_state, player_names, top_y=224)
        panel_state.runtime_auction_current_bid_entry = pending_controls["runtime_auction_current_bid_entry"]
        panel_state.runtime_auction_bidder_index_entry = pending_controls["runtime_auction_bidder_index_entry"]
        panel_state.runtime_auction_winner_dropdown = pending_controls["runtime_auction_winner_dropdown"]
        panel_state.runtime_trade_proposer_dropdown = pending_controls["runtime_trade_proposer_dropdown"]
        panel_state.runtime_trade_receiver_dropdown = pending_controls["runtime_trade_receiver_dropdown"]
        panel_state.runtime_trade_proposer_cash_entry = pending_controls["runtime_trade_proposer_cash_entry"]
        panel_state.runtime_trade_receiver_cash_entry = pending_controls["runtime_trade_receiver_cash_entry"]
        panel_state.runtime_trade_note_entry = pending_controls["runtime_trade_note_entry"]

    def _build_standard_side_panels(self, state: FrontendStateView, sidebar_x: int, sidebar_width: int, window_height: int) -> None:
        gap = theme.PANEL_GAP
        left_panel_width = max(240, int((sidebar_width - gap) * 0.38))
        right_panel_width = sidebar_width - gap - left_panel_width
        top_y = 70
        available_height = max(360, window_height - theme.STATUS_BAR_HEIGHT - top_y - 20)
        pending_trade_visible = self._has_pending_trade(state)
        selected_height = min(244, max(180, available_height // 3))
        trade_height = 0 if not pending_trade_visible else min(210, max(150, available_height // 4))
        left_gap_count = 1 if not pending_trade_visible else 2
        standings_height = max(150, available_height - selected_height - trade_height - gap * left_gap_count)
        moves_height = max(240, int((available_height - gap) * 0.56))
        story_height = max(140, available_height - moves_height - gap)
        left_bottom_y = top_y + selected_height + gap
        right_x = sidebar_x + left_panel_width + gap
        right_bottom_y = top_y + moves_height + gap

        selected_panel = self._track(UIPanel(pygame.Rect(sidebar_x, top_y, left_panel_width, selected_height), manager=self.manager))
        self._build_panel_title(selected_panel, left_panel_width, "Board Spotlight")
        self._track(UITextBox(self._selected_space_html(state), pygame.Rect(10, 34, left_panel_width - 20, selected_height - 44), manager=self.manager, container=selected_panel))

        standings_top = left_bottom_y
        if pending_trade_visible:
            trade_panel = self._track(UIPanel(pygame.Rect(sidebar_x, left_bottom_y, left_panel_width, trade_height), manager=self.manager))
            self._build_panel_title(trade_panel, left_panel_width, "Trade On Table")
            self._track(UITextBox(self._trade_panel_html(state), pygame.Rect(10, 34, left_panel_width - 20, trade_height - 44), manager=self.manager, container=trade_panel))
            standings_top = left_bottom_y + trade_height + gap

        standings_panel = self._track(UIPanel(pygame.Rect(sidebar_x, standings_top, left_panel_width, standings_height), manager=self.manager))
        self._build_panel_title(standings_panel, left_panel_width, "At the Table")
        self._track(UITextBox(self._player_cash_summary_html(state.game_view.players, state.game_view.current_player_name), pygame.Rect(10, 34, left_panel_width - 20, standings_height - 44), manager=self.manager, container=standings_panel))

        actions_panel = self._track(UIPanel(pygame.Rect(right_x, top_y, right_panel_width, moves_height), manager=self.manager))
        self._build_panel_title(actions_panel, right_panel_width, "Your Move")
        self._action_buttons = {}
        display_actions = self._display_actions(state.active_turn_plan.legal_actions)
        ai_turn_active = self._is_ai_turn_active(state)
        context_height = min(118, max(88, moves_height // 4))
        self._track(UITextBox(self._turn_context_html(state), pygame.Rect(10, 34, right_panel_width - 20, context_height), manager=self.manager, container=actions_panel))
        if display_actions:
            button_height = 28
            button_spacing = 6
            button_top = 46 + context_height
            max_buttons = max(1, (moves_height - button_top - 14) // (button_height + button_spacing))
            paged_actions = self._paginate_property_submenu_actions(display_actions, max_buttons)
            for index, action in enumerate(paged_actions):
                description = self._action_button_label(action) if isinstance(action, LegalActionOption) else action[2]
                button = self._track(
                    UIButton(
                        pygame.Rect(10, button_top + index * (button_height + button_spacing), right_panel_width - 20, button_height),
                        description,
                        manager=self.manager,
                        container=actions_panel,
                    )
                )
                if ai_turn_active:
                    button.disable()
                self._action_buttons[button] = action
        else:
            message = "<b>AI taking its turn</b><br>Human input wakes up again in a moment." if ai_turn_active else "<b>No move needed</b><br>There is nothing to decide right now."
            self._track(UITextBox(message, pygame.Rect(10, 46 + context_height, right_panel_width - 20, 54), manager=self.manager, container=actions_panel))

        log_panel = self._track(UIPanel(pygame.Rect(right_x, right_bottom_y, right_panel_width, story_height), manager=self.manager))
        self._build_panel_title(log_panel, right_panel_width, "Game Story")
        self._track(UITextBox(self._log_html(story_height), pygame.Rect(10, 34, right_panel_width - 20, story_height - 44), manager=self.manager, container=log_panel))

    def _build_panel_title(self, panel: UIPanel, panel_width: int, title: str) -> None:
        self._track(UILabel(pygame.Rect(12, 8, panel_width - 24, 20), title, manager=self.manager, container=panel))

    def _build_debug_pending_controls(self, context_panel: UIPanel, panel_width: int, full_state: dict[str, Any], player_names: list[str], *, top_y: int = 168) -> dict[str, Any]:
        controls: dict[str, Any] = {
            "runtime_auction_current_bid_entry": None,
            "runtime_auction_bidder_index_entry": None,
            "runtime_auction_winner_dropdown": None,
            "runtime_trade_proposer_dropdown": None,
            "runtime_trade_receiver_dropdown": None,
            "runtime_trade_proposer_cash_entry": None,
            "runtime_trade_receiver_cash_entry": None,
            "runtime_trade_note_entry": None,
        }
        runtime = full_state["runtime"]
        pending_kind = self._current_pending_action_kind(runtime)
        payload = self._pending_action_payload(runtime, pending_kind) if pending_kind != "none" else None

        if pending_kind == "auction" and isinstance(payload, dict):
            self._track(UILabel(pygame.Rect(18, top_y, 84, 22), "Current Bid", manager=self.manager, container=context_panel))
            current_bid_entry = self._track(UITextEntryLine(pygame.Rect(108, top_y - 2, 90, 28), manager=self.manager, container=context_panel))
            current_bid_entry.set_text(str(payload.get("current_bid", 0)))
            self._register_debug_field("runtime.auction_current_bid", current_bid_entry)

            self._track(UILabel(pygame.Rect(220, top_y, 52, 22), "Idx", manager=self.manager, container=context_panel))
            bidder_index_entry = self._track(UITextEntryLine(pygame.Rect(274, top_y - 2, 72, 28), manager=self.manager, container=context_panel))
            bidder_index_entry.set_text(str(payload.get("current_bidder_index", 0)))
            self._register_debug_field("runtime.auction_bidder_index", bidder_index_entry)

            winner_options = ["none", *[str(name) for name in payload.get("active_player_names", [])]]
            winner_selected = str(payload.get("current_winner_name") or "none")
            self._track(UILabel(pygame.Rect(18, top_y + 40, 84, 22), "Winner", manager=self.manager, container=context_panel))
            winner_dropdown = self._create_dropdown(winner_options or ["none", *player_names], winner_selected, pygame.Rect(108, top_y + 36, panel_width - 126, 30), context_panel, detached=True)
            self._register_debug_field("runtime.auction_winner", winner_dropdown)

            controls["runtime_auction_current_bid_entry"] = current_bid_entry
            controls["runtime_auction_bidder_index_entry"] = bidder_index_entry
            controls["runtime_auction_winner_dropdown"] = winner_dropdown
            return controls

        if pending_kind == "trade" and isinstance(payload, dict):
            proposer_name = str(payload.get("proposer_name") or player_names[0])
            receiver_name = str(payload.get("receiver_name") or player_names[0])
            self._track(UILabel(pygame.Rect(18, top_y, 84, 22), "Proposer", manager=self.manager, container=context_panel))
            proposer_dropdown = self._create_dropdown(player_names, proposer_name, pygame.Rect(108, top_y - 4, panel_width - 126, 30), context_panel, detached=True)
            self._register_debug_field("runtime.trade_proposer", proposer_dropdown)
            self._track(UILabel(pygame.Rect(18, top_y + 40, 84, 22), "Receiver", manager=self.manager, container=context_panel))
            receiver_dropdown = self._create_dropdown(player_names, receiver_name, pygame.Rect(108, top_y + 36, panel_width - 126, 30), context_panel, detached=True)
            self._register_debug_field("runtime.trade_receiver", receiver_dropdown)

            self._track(UILabel(pygame.Rect(18, top_y + 80, 84, 22), "Cash Out", manager=self.manager, container=context_panel))
            proposer_cash_entry = self._track(UITextEntryLine(pygame.Rect(108, top_y + 78, 80, 28), manager=self.manager, container=context_panel))
            proposer_cash_entry.set_text(str(payload.get("proposer_cash", 0)))
            self._register_debug_field("runtime.trade_proposer_cash", proposer_cash_entry)

            self._track(UILabel(pygame.Rect(220, top_y + 80, 62, 22), "Cash In", manager=self.manager, container=context_panel))
            receiver_cash_entry = self._track(UITextEntryLine(pygame.Rect(286, top_y + 78, 80, 28), manager=self.manager, container=context_panel))
            receiver_cash_entry.set_text(str(payload.get("receiver_cash", 0)))
            self._register_debug_field("runtime.trade_receiver_cash", receiver_cash_entry)

            self._track(UILabel(pygame.Rect(18, top_y + 120, 84, 22), "Note", manager=self.manager, container=context_panel))
            trade_note_entry = self._track(UITextEntryLine(pygame.Rect(108, top_y + 118, panel_width - 126, 28), manager=self.manager, container=context_panel))
            trade_note_entry.set_text(str(payload.get("note") or ""))
            self._register_debug_field("runtime.trade_note", trade_note_entry)

            controls["runtime_trade_proposer_dropdown"] = proposer_dropdown
            controls["runtime_trade_receiver_dropdown"] = receiver_dropdown
            controls["runtime_trade_proposer_cash_entry"] = proposer_cash_entry
            controls["runtime_trade_receiver_cash_entry"] = receiver_cash_entry
            controls["runtime_trade_note_entry"] = trade_note_entry
            return controls

        self._track(UITextBox("Switch the pending type to <b>auction</b> or <b>trade</b> to edit its fields directly.", pygame.Rect(18, top_y, panel_width - 36, 70), manager=self.manager, container=context_panel))
        return controls

    def _rebuild_replay_ui(self) -> None:
        self._clear_elements()
        if self._replay is None:
            self.screen_mode = "game"
            self._rebuild_game_ui()
            return
        frame = self._current_replay_frame()
        if frame is None:
            self.screen_mode = "game"
            self._replay = None
            self._rebuild_game_ui()
            return

        state = FrontendStateView.from_dict(frame["frontend_state"])
        sidebar_x = self.board_rect.right + theme.BOARD_MARGIN
        sidebar_width = self.sidebar_width
        window_width, window_height = self.screen.get_size()
        log_height = max(220, window_height - 500)
        frame_index = self._replay.current_index + 1
        header = f"Replay {frame_index}/{len(self._replay.frames)} | {frame.get('label', 'frame')}"
        self._track(UILabel(pygame.Rect(theme.BOARD_MARGIN, 12, 640, 30), header, manager=self.manager))

        controls = [("Prev", ("replay_prev", None)), ("Play" if not self._replay.is_playing else "Pause", ("replay_toggle", None)), ("Next", ("replay_next", None)), ("Exit Replay", ("replay_exit", None))]
        button_width = 96
        button_gap = 10
        right_edge = window_width - 16
        for label, command in reversed(controls):
            right_edge -= button_width
            button = self._track(UIButton(pygame.Rect(right_edge, 12, button_width, 32), label, manager=self.manager))
            self._ui_commands[button] = command
            right_edge -= button_gap

        selected_panel = self._track(UIPanel(pygame.Rect(sidebar_x, 70, sidebar_width, 170), manager=self.manager))
        self._build_panel_title(selected_panel, sidebar_width, "Replay Spotlight")
        self._track(UITextBox(self._selected_space_html(state), pygame.Rect(10, 34, sidebar_width - 20, 126), manager=self.manager, container=selected_panel))

        context_panel = self._track(UIPanel(pygame.Rect(sidebar_x, 250, sidebar_width, 170), manager=self.manager))
        self._build_panel_title(context_panel, sidebar_width, "Replay Round")
        replay_html = (
            f"<b>Status:</b> {escape(str(frame.get('status_message', '')))}<br>"
            f"<b>Selected space:</b> {frame.get('selected_space_index', 0)}<br>"
            f"<b>Up now:</b> {state.game_view.current_player_name or 'n/a'}<br>"
            f"<b>Stage:</b> {self._friendly_phase_name(state.game_view.current_turn_phase)}"
        )
        self._track(UITextBox(replay_html, pygame.Rect(10, 34, sidebar_width - 20, 126), manager=self.manager, container=context_panel))

        log_panel = self._track(UIPanel(pygame.Rect(sidebar_x, 430, sidebar_width, log_height), manager=self.manager))
        self._build_panel_title(log_panel, sidebar_width, "Replay Story")
        history = frame.get("message_history") if isinstance(frame.get("message_history"), list) else []
        self._track(UITextBox(self._log_html(log_height, history), pygame.Rect(10, 34, sidebar_width - 20, log_height - 44), manager=self.manager, container=log_panel))

    def _draw_frame(self) -> None:
        self.screen.fill(pygame.Color(theme.BACKGROUND_COLOR))
        if self.screen_mode in {"game", "replay"} and self._display_frontend_state() is not None:
            display_state = self._display_frontend_state()
            hidden_player_names, token_overrides = self._animated_token_render_data()
            self.board_renderer.render(
                self.screen,
                display_state,
                selected_space_index=self.controller.selected_space_index,
                hovered_space_index=self.hovered_space_index,
                hidden_player_names=hidden_player_names,
                token_overrides=token_overrides,
            )
            self._draw_sidebar_backdrop()
            self._draw_dice_overlay()
        else:
            self._draw_setup_background()
        self.manager.draw_ui(self.screen)
        self._draw_debug_validation_highlights()
        self._draw_status_bar()

    def _draw_sidebar_backdrop(self) -> None:
        window_width, window_height = self.screen.get_size()
        sidebar_rect = pygame.Rect(
            self.board_rect.right + 10,
            58,
            max(0, window_width - self.board_rect.right - 20),
            max(0, window_height - theme.STATUS_BAR_HEIGHT - 70),
        )
        if sidebar_rect.width <= 0 or sidebar_rect.height <= 0:
            return
        backdrop = pygame.Surface(sidebar_rect.size, pygame.SRCALPHA)
        backdrop.fill((0, 0, 0, 0))
        pygame.draw.rect(backdrop, (251, 244, 230, 216), backdrop.get_rect(), border_radius=26)
        pygame.draw.rect(backdrop, pygame.Color(theme.BOARD_LINE), backdrop.get_rect(), width=2, border_radius=26)
        self.screen.blit(backdrop, sidebar_rect.topleft)

    def _draw_setup_background(self) -> None:
        window_width, window_height = self.screen.get_size()
        paper = pygame.Surface((window_width - theme.SETUP_PANEL_MARGIN * 2, window_height - 120), pygame.SRCALPHA)
        paper.fill((252, 246, 235, 220))
        pygame.draw.rect(paper, pygame.Color(theme.BOARD_LINE), paper.get_rect(), width=3, border_radius=18)
        self.screen.blit(paper, (theme.SETUP_PANEL_MARGIN, 80))

    def _draw_status_bar(self) -> None:
        font = pygame.font.SysFont("segoeui", 18, bold=True)
        window_width, window_height = self.screen.get_size()
        bar_rect = pygame.Rect(0, window_height - theme.STATUS_BAR_HEIGHT, window_width, theme.STATUS_BAR_HEIGHT)
        pygame.draw.rect(self.screen, pygame.Color(theme.BOARD_LINE), bar_rect)
        status_color = theme.STATUS_ERROR if self.controller.last_error else theme.STATUS_OK
        text = font.render(self.controller.status_message, True, pygame.Color(status_color))
        self.screen.blit(text, (16, window_height - 27))

    def _draw_debug_validation_highlights(self) -> None:
        if not self._debug_invalid_elements:
            return
        for element in self._debug_invalid_elements:
            try:
                rect = element.get_abs_rect().inflate(4, 4)
            except Exception:
                continue
            pygame.draw.rect(self.screen, pygame.Color(theme.STATUS_ERROR), rect, width=2, border_radius=6)

    def _selected_space_html(self, state: FrontendStateView) -> str:
        space = state.board_spaces[self.controller.selected_space_index]
        lines = [
            f"<b>{space.name} (#{space.index})</b>",
            f"Kind: {self._friendly_space_type(space.space_type)}",
            f"Here now: {', '.join(space.occupant_names) if space.occupant_names else 'none'}",
            f"Owner: {space.owner_name or 'bank'}",
        ]
        if space.color_group:
            lines.append(f"Set: {space.color_group}")
        if space.price is not None:
            lines.append(f"Cost: ${space.price}")
        if space.house_cost is not None:
            lines.append(f"House cost: ${space.house_cost}")
        if space.building_count is not None:
            lines.append(f"Buildings: {space.building_count}")
        if space.mortgaged:
            lines.append("Mortgaged: yes")
        if space.notes:
            lines.append(space.notes)
        return "<br>".join(lines)

    def _turn_context_html(self, state: FrontendStateView) -> str:
        plan = state.active_turn_plan
        game_view = state.game_view
        lines = [
            f"<b>Up now:</b> {game_view.current_player_name or 'n/a'}",
            f"<b>Making the call:</b> {plan.player_name}",
            f"<b>Stage:</b> {self._friendly_phase_name(plan.turn_phase)}",
            f"<b>Bank supply:</b> {game_view.houses_remaining} houses, {game_view.hotels_remaining} hotels",
            f"<b>Decision waiting:</b> {'yes' if plan.has_pending_action else 'no'}",
        ]
        if plan.reason:
            lines.append(f"<b>Note:</b> {plan.reason}")
        if game_view.pending_action is not None:
            lines.append(f"<b>What happens now:</b> {game_view.pending_action.prompt}")
        return "<br>".join(lines)

    def _log_html(self, log_height: int, history: list[str] | None = None) -> str:
        max_lines = max(6, min(theme.MAX_VISIBLE_LOG_LINES, (log_height - 28) // 15))
        log_lines = (history if history is not None else self.controller.message_history)[-max_lines:] or ["The story starts here."]
        padding = max(0, max_lines - len(log_lines))
        lines = ["&nbsp;"] * padding + [escape(line) for line in log_lines]
        return "<br>".join(lines)

    def _pending_trade_summary_html(self, pending_action) -> str | None:
        if pending_action.action_type != "trade_decision" or pending_action.trade is None:
            return None
        trade = pending_action.trade
        proposer_gives = self._format_trade_side_summary(
            cash=trade.proposer_cash,
            property_names=trade.proposer_property_names,
            jail_cards=trade.proposer_jail_cards,
        )
        receiver_gives = self._format_trade_side_summary(
            cash=trade.receiver_cash,
            property_names=trade.receiver_property_names,
            jail_cards=trade.receiver_jail_cards,
        )
        lines = [
            f"<b>Trade on table:</b> {escape(trade.proposer_name)} -> {escape(trade.receiver_name)}",
            f"<b>{escape(trade.proposer_name)} gives:</b> {escape(proposer_gives)}",
            f"<b>{escape(trade.receiver_name)} gives:</b> {escape(receiver_gives)}",
        ]
        if trade.note:
            lines.append(f"<b>Trade note:</b> {escape(trade.note)}")
        return "<br>".join(lines)

    def _trade_panel_html(self, state: FrontendStateView) -> str:
        pending_action = state.game_view.pending_action
        if pending_action is None:
            return "No trade is waiting for a response."
        trade_html = self._pending_trade_summary_html(pending_action)
        return "No trade is waiting for a response." if trade_html is None else trade_html

    @staticmethod
    def _has_pending_trade(state: FrontendStateView) -> bool:
        pending_action = state.game_view.pending_action
        return pending_action is not None and pending_action.action_type == "trade_decision" and pending_action.trade is not None

    @staticmethod
    def _format_trade_side_summary(*, cash: int, property_names: tuple[str, ...], jail_cards: int) -> str:
        parts: list[str] = []
        if cash > 0:
            parts.append(f"${cash}")
        if property_names:
            parts.append(", ".join(property_names))
        if jail_cards > 0:
            card_label = "card" if jail_cards == 1 else "cards"
            parts.append(f"{jail_cards} jail {card_label}")
        return "nothing" if not parts else " + ".join(parts)

    def _is_ai_turn_active(self, state: FrontendStateView | None = None) -> bool:
        active_state = self.controller.frontend_state if state is None else state
        if active_state is None:
            return False
        return active_state.active_turn_plan.player_role == AI_ROLE

    def _open_bid_prompt(self, action: LegalActionOption) -> None:
        initial_value = str(action.min_bid or 1)
        self._open_text_prompt("bid", "Place Your Bid", initial_value, action)

    def _open_path_prompt(self, kind: str, title: str, initial_value: str) -> None:
        self._open_text_prompt(kind, title, initial_value, None)

    def _open_trade_prompt(self, action: LegalActionOption) -> None:
        self._open_trade_prompt_with_initial_offer(action)

    def _open_counter_trade_prompt(self, action: LegalActionOption) -> None:
        pending_trade = self._current_pending_trade_offer()
        if pending_trade is None:
            self.controller.set_error("There is no pending trade to counter.")
            self._rebuild_game_ui()
            return
        self._open_trade_prompt_with_initial_offer(
            action,
            initial_trade_offer={
                "proposer_name": action.actor_name,
                "receiver_name": action.target_player_name,
                "proposer_cash": int(pending_trade.get("receiver_cash", 0)),
                "receiver_cash": int(pending_trade.get("proposer_cash", 0)),
                "proposer_property_names": list(pending_trade.get("receiver_property_names", ())),
                "receiver_property_names": list(pending_trade.get("proposer_property_names", ())),
                "proposer_jail_cards": int(pending_trade.get("receiver_jail_cards", 0)),
                "receiver_jail_cards": int(pending_trade.get("proposer_jail_cards", 0)),
                "note": str(pending_trade.get("note", "")),
            },
            window_title="Respond To Trade Offer",
            submit_label="Send Counter Offer",
            cancel_label="Reject Trade",
            accept_action=self._legal_action_by_type("accept_trade"),
            reject_action=self._legal_action_by_type("reject_trade"),
        )

    def _open_pending_trade_response_prompt(self) -> None:
        pending_trade = self._current_pending_trade_offer()
        accept_action = self._legal_action_by_type("accept_trade")
        reject_action = self._legal_action_by_type("reject_trade")
        if pending_trade is None or (accept_action is None and reject_action is None):
            return
        counter_action = self._legal_action_by_type("counter_trade")
        response_action = counter_action or reject_action or accept_action
        if response_action is None:
            return
        self._open_trade_prompt_with_initial_offer(
            response_action,
            initial_trade_offer={
                "proposer_name": response_action.actor_name,
                "receiver_name": response_action.target_player_name,
                "proposer_cash": int(pending_trade.get("receiver_cash", 0)),
                "receiver_cash": int(pending_trade.get("proposer_cash", 0)),
                "proposer_property_names": list(pending_trade.get("receiver_property_names", ())),
                "receiver_property_names": list(pending_trade.get("proposer_property_names", ())),
                "proposer_jail_cards": int(pending_trade.get("receiver_jail_cards", 0)),
                "receiver_jail_cards": int(pending_trade.get("proposer_jail_cards", 0)),
                "note": str(pending_trade.get("note", "")),
            },
            window_title="Respond To Trade Offer",
            submit_label="Send Counter Offer" if counter_action is not None else "Counter Limit Reached",
            cancel_label="Reject Trade",
            accept_action=accept_action,
            reject_action=reject_action,
            submit_enabled=counter_action is not None,
        )

    def _open_trade_prompt_with_initial_offer(
        self,
        action: LegalActionOption,
        *,
        initial_trade_offer: dict[str, Any] | None = None,
        window_title: str = "Make a Trade Offer",
        submit_label: str = "Propose Trade",
        cancel_label: str = "Cancel",
        accept_action: LegalActionOption | None = None,
        reject_action: LegalActionOption | None = None,
        submit_enabled: bool = True,
    ) -> None:
        if self.controller.frontend_state is None:
            self.controller.set_error("Frontend state is not loaded.")
            self._rebuild_game_ui()
            return
        if action.target_player_name is None:
            self.controller.set_error("Trade action is missing a target player.")
            self._rebuild_game_ui()
            return
        if getattr(self, "_trade_prompt", None) is not None:
            self._close_trade_prompt()
        if getattr(self, "_prompt", None) is not None:
            self._close_prompt()

        proposer = self._player_by_name(action.actor_name)
        receiver = self._player_by_name(action.target_player_name)
        window = self._track(UIWindow(pygame.Rect(430, 120, 700, 600), manager=self.manager, window_display_title=window_title))

        left_x = 20
        right_x = 360
        self._track(UILabel(pygame.Rect(left_x, 18, 280, 24), f"{proposer.name} gives", manager=self.manager, container=window))
        self._track(UILabel(pygame.Rect(right_x, 18, 280, 24), f"{receiver.name} gives", manager=self.manager, container=window))

        proposer_info = self._track(
            UITextBox(
                self._trade_player_info_html(proposer),
                pygame.Rect(left_x, 48, 300, 120),
                manager=self.manager,
                container=window,
            )
        )
        receiver_info = self._track(
            UITextBox(
                self._trade_player_info_html(receiver),
                pygame.Rect(right_x, 48, 300, 120),
                manager=self.manager,
                container=window,
            )
        )
        _ = proposer_info, receiver_info

        self._track(UILabel(pygame.Rect(left_x, 184, 120, 22), "Cash", manager=self.manager, container=window))
        proposer_cash_label = self._track(UILabel(pygame.Rect(left_x + 110, 184, 86, 22), "$0", manager=self.manager, container=window))
        proposer_cash_minus = self._track(UIButton(pygame.Rect(left_x + 202, 180, 42, 30), "-", manager=self.manager, container=window))
        proposer_cash_plus = self._track(UIButton(pygame.Rect(left_x + 252, 180, 42, 30), "+", manager=self.manager, container=window))
        self._ui_commands[proposer_cash_minus] = ("trade_adjust", ("proposer_cash", -10))
        self._ui_commands[proposer_cash_plus] = ("trade_adjust", ("proposer_cash", 10))

        self._track(UILabel(pygame.Rect(right_x, 184, 120, 22), "Cash", manager=self.manager, container=window))
        receiver_cash_label = self._track(UILabel(pygame.Rect(right_x + 110, 184, 86, 22), "$0", manager=self.manager, container=window))
        receiver_cash_minus = self._track(UIButton(pygame.Rect(right_x + 202, 180, 42, 30), "-", manager=self.manager, container=window))
        receiver_cash_plus = self._track(UIButton(pygame.Rect(right_x + 252, 180, 42, 30), "+", manager=self.manager, container=window))
        self._ui_commands[receiver_cash_minus] = ("trade_adjust", ("receiver_cash", -10))
        self._ui_commands[receiver_cash_plus] = ("trade_adjust", ("receiver_cash", 10))

        self._track(UILabel(pygame.Rect(left_x, 224, 300, 22), "Properties", manager=self.manager, container=window))
        proposer_property_prev = self._track(UIButton(pygame.Rect(left_x, 248, 54, 28), "Prev", manager=self.manager, container=window))
        proposer_property_next = self._track(UIButton(pygame.Rect(left_x + 246, 248, 54, 28), "Next", manager=self.manager, container=window))
        self._ui_commands[proposer_property_prev] = ("trade_page", ("proposer", -1))
        self._ui_commands[proposer_property_next] = ("trade_page", ("proposer", 1))
        proposer_property_buttons: list[UIButton] = []
        for index in range(4):
            button = self._track(UIButton(pygame.Rect(left_x, 284 + index * 34, 300, 28), "", manager=self.manager, container=window))
            proposer_property_buttons.append(button)

        self._track(UILabel(pygame.Rect(right_x, 224, 300, 22), "Properties", manager=self.manager, container=window))
        receiver_property_prev = self._track(UIButton(pygame.Rect(right_x, 248, 54, 28), "Prev", manager=self.manager, container=window))
        receiver_property_next = self._track(UIButton(pygame.Rect(right_x + 246, 248, 54, 28), "Next", manager=self.manager, container=window))
        self._ui_commands[receiver_property_prev] = ("trade_page", ("receiver", -1))
        self._ui_commands[receiver_property_next] = ("trade_page", ("receiver", 1))
        receiver_property_buttons: list[UIButton] = []
        for index in range(4):
            button = self._track(UIButton(pygame.Rect(right_x, 284 + index * 34, 300, 28), "", manager=self.manager, container=window))
            receiver_property_buttons.append(button)

        self._track(UILabel(pygame.Rect(left_x, 432, 120, 22), "Jail cards", manager=self.manager, container=window))
        proposer_jail_cards_label = self._track(UILabel(pygame.Rect(left_x + 110, 432, 86, 22), "0", manager=self.manager, container=window))
        proposer_jail_minus = self._track(UIButton(pygame.Rect(left_x + 202, 428, 42, 30), "-", manager=self.manager, container=window))
        proposer_jail_plus = self._track(UIButton(pygame.Rect(left_x + 252, 428, 42, 30), "+", manager=self.manager, container=window))
        self._ui_commands[proposer_jail_minus] = ("trade_adjust", ("proposer_jail_cards", -1))
        self._ui_commands[proposer_jail_plus] = ("trade_adjust", ("proposer_jail_cards", 1))

        self._track(UILabel(pygame.Rect(right_x, 432, 120, 22), "Jail cards", manager=self.manager, container=window))
        receiver_jail_cards_label = self._track(UILabel(pygame.Rect(right_x + 110, 432, 86, 22), "0", manager=self.manager, container=window))
        receiver_jail_minus = self._track(UIButton(pygame.Rect(right_x + 202, 428, 42, 30), "-", manager=self.manager, container=window))
        receiver_jail_plus = self._track(UIButton(pygame.Rect(right_x + 252, 428, 42, 30), "+", manager=self.manager, container=window))
        self._ui_commands[receiver_jail_minus] = ("trade_adjust", ("receiver_jail_cards", -1))
        self._ui_commands[receiver_jail_plus] = ("trade_adjust", ("receiver_jail_cards", 1))

        self._track(UILabel(pygame.Rect(left_x, 478, 640, 22), "Note", manager=self.manager, container=window))
        note_entry = self._track(UITextEntryLine(pygame.Rect(left_x, 502, 640, 28), manager=self.manager, container=window))
        if initial_trade_offer is not None:
            note_entry.set_text(str(initial_trade_offer.get("note", "")))

        accept_button = None
        if accept_action is not None:
            accept_button = self._track(UIButton(pygame.Rect(200, 540, 140, 34), "Accept Trade", manager=self.manager, container=window))
        submit_button = self._track(UIButton(pygame.Rect(360, 540, 140, 34), submit_label, manager=self.manager, container=window))
        if not submit_enabled:
            submit_button.disable()
        cancel_button = self._track(UIButton(pygame.Rect(520, 540, 140, 34), cancel_label, manager=self.manager, container=window))

        proposer_selected_properties = set()
        receiver_selected_properties = set()
        proposer_cash_value = 0
        receiver_cash_value = 0
        proposer_jail_cards_value = 0
        receiver_jail_cards_value = 0
        if initial_trade_offer is not None:
            proposer_cash_value = int(initial_trade_offer.get("proposer_cash", 0))
            receiver_cash_value = int(initial_trade_offer.get("receiver_cash", 0))
            proposer_jail_cards_value = int(initial_trade_offer.get("proposer_jail_cards", 0))
            receiver_jail_cards_value = int(initial_trade_offer.get("receiver_jail_cards", 0))
            proposer_selected_properties = {str(name) for name in initial_trade_offer.get("proposer_property_names", ())}
            receiver_selected_properties = {str(name) for name in initial_trade_offer.get("receiver_property_names", ())}

        self._trade_prompt = TradePromptState(
            window=window,
            submit_button=submit_button,
            cancel_button=cancel_button,
            action=action,
            proposer=proposer,
            receiver=receiver,
            proposer_cash_label=proposer_cash_label,
            receiver_cash_label=receiver_cash_label,
            proposer_jail_cards_label=proposer_jail_cards_label,
            receiver_jail_cards_label=receiver_jail_cards_label,
            proposer_cash_value=proposer_cash_value,
            receiver_cash_value=receiver_cash_value,
            proposer_jail_cards_value=proposer_jail_cards_value,
            receiver_jail_cards_value=receiver_jail_cards_value,
            proposer_selected_properties=proposer_selected_properties,
            receiver_selected_properties=receiver_selected_properties,
            proposer_property_buttons=proposer_property_buttons,
            receiver_property_buttons=receiver_property_buttons,
            proposer_property_page=0,
            receiver_property_page=0,
            note_entry=note_entry,
            accept_button=accept_button,
            reject_action=reject_action,
            accept_action=accept_action,
        )
        self._refresh_trade_prompt_display()

    def _open_text_prompt(self, kind: str, title: str, initial_value: str, action: LegalActionOption | None) -> None:
        if self._prompt is not None:
            self._close_prompt()
        window = self._track(UIWindow(pygame.Rect(520, 280, 420, 180), manager=self.manager, window_display_title=title))
        entry = self._track(UITextEntryLine(pygame.Rect(20, 56, 360, 34), manager=self.manager, container=window))
        entry.set_text(initial_value)
        submit = self._track(UIButton(pygame.Rect(20, 108, 160, 34), "Confirm", manager=self.manager, container=window))
        cancel = self._track(UIButton(pygame.Rect(220, 108, 160, 34), "Cancel", manager=self.manager, container=window))
        self._prompt = TextPromptState(kind=kind, window=window, entry=entry, submit_button=submit, cancel_button=cancel, title=title, action=action)

    def _submit_prompt(self) -> None:
        if self._prompt is None:
            return
        value = self._prompt.entry.get_text().strip()
        try:
            if self._prompt.kind == "save":
                self.controller.save_game(value)
            elif self._prompt.kind == "load":
                self._ensure_local_backend_connection()
                self.controller.load_game(value)
                self.screen_mode = "game"
                self._replay = None
                self._animation = None
                self._property_submenu_mode = None
                self._property_submenu_page = 0
                self._pending_ai_step = None
            elif self._prompt.kind == "save_replay":
                self.controller.save_replay(value)
            elif self._prompt.kind == "load_replay":
                frames = self.controller.load_replay(value)
                self._replay = ReplayState(frames=frames)
                self.screen_mode = "replay"
                self._animation = None
                self._property_submenu_mode = None
                self._property_submenu_page = 0
                self._pending_ai_step = None
                first_frame = self._current_replay_frame()
                if first_frame is not None:
                    self.controller.select_space(int(first_frame.get("selected_space_index", 0)))
            elif self._prompt.kind == "bid":
                if self._prompt.action is None:
                    raise ValueError("Auction prompt is missing its action context.")
                previous_state = self.controller.frontend_state
                interaction = self.controller.execute_action(self._prompt.action, bid_amount=int(value))
                self._start_action_animation(self._prompt.action, interaction, previous_state, self.controller.frontend_state)
            else:
                raise ValueError(f"Unsupported prompt kind: {self._prompt.kind}")
            self._close_prompt()
            if self.screen_mode == "game":
                if self._animation is None:
                    self._rebuild_game_ui()
            elif self.screen_mode == "replay":
                self._rebuild_replay_ui()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._close_prompt()
            if self.screen_mode == "game":
                self._rebuild_game_ui()
            elif self.screen_mode == "replay":
                self._rebuild_replay_ui()
            else:
                self._build_setup_screen()

    def _submit_trade_prompt(self) -> None:
        if self._trade_prompt is None:
            return
        prompt = self._trade_prompt
        try:
            trade_offer = {
                "proposer_name": prompt.action.actor_name,
                "receiver_name": prompt.action.target_player_name,
                "proposer_cash": prompt.proposer_cash_value,
                "receiver_cash": prompt.receiver_cash_value,
                "proposer_property_names": sorted(prompt.proposer_selected_properties),
                "receiver_property_names": sorted(prompt.receiver_selected_properties),
                "proposer_jail_cards": prompt.proposer_jail_cards_value,
                "receiver_jail_cards": prompt.receiver_jail_cards_value,
                "note": prompt.note_entry.get_text().strip(),
            }
            self.controller.execute_action(prompt.action, trade_offer=trade_offer)
            self._close_trade_prompt()
            self._rebuild_game_ui()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._close_trade_prompt()
            self._rebuild_game_ui()

    def _update_online_session(self, delta_time: float) -> None:
        _ = delta_time

    def _accept_trade_prompt(self) -> None:
        if self._trade_prompt is None or self._trade_prompt.accept_action is None:
            return
        try:
            self.controller.execute_action(self._trade_prompt.accept_action)
            self._close_trade_prompt()
            self._rebuild_game_ui()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._close_trade_prompt()
            self._rebuild_game_ui()

    def _reject_trade_prompt(self) -> None:
        if self._trade_prompt is None or self._trade_prompt.reject_action is None:
            return
        try:
            self.controller.execute_action(self._trade_prompt.reject_action)
            self._close_trade_prompt()
            self._rebuild_game_ui()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._close_trade_prompt()
            self._rebuild_game_ui()

    def _close_prompt(self) -> None:
        if self._prompt is None:
            return
        self._prompt.window.kill()
        self._prompt = None

    def _close_trade_prompt(self) -> None:
        if self._trade_prompt is None:
            return
        self._trade_prompt.window.kill()
        self._trade_prompt = None

    def _maybe_open_pending_trade_prompt(self, state: FrontendStateView) -> None:
        pending_action = state.game_view.pending_action
        if pending_action is None or pending_action.action_type != "trade_decision":
            return
        if pending_action.player_role != HUMAN_ROLE:
            return
        if self._trade_prompt is not None:
            return
        self._open_pending_trade_response_prompt()

    def _legal_action_by_type(self, action_type: str) -> LegalActionOption | None:
        state = self.controller.frontend_state
        if state is None:
            return None
        return next((action for action in state.active_turn_plan.legal_actions if action.action_type == action_type), None)

    def _current_pending_trade_offer(self) -> dict[str, Any] | None:
        state = self.controller.frontend_state
        if state is None or state.game_view.pending_action is None or state.game_view.pending_action.trade is None:
            return None
        return state.game_view.pending_action.trade.to_dict()

    def _player_by_name(self, player_name: str) -> PlayerView:
        state = self.controller.frontend_state
        if state is None:
            raise ValueError("Frontend state is not loaded.")
        for player in state.game_view.players:
            if player.name == player_name:
                return player
        raise ValueError(f"Unknown player: {player_name}")

    def _trade_player_info_html(self, player: PlayerView) -> str:
        properties = ", ".join(escape(name) for name in player.properties) if player.properties else "none"
        return (
            f"<b>Cash:</b> ${player.cash}<br>"
            f"<b>Jail cards:</b> {player.get_out_of_jail_cards}<br>"
            f"<b>Properties:</b> {properties}"
        )

    def _create_dropdown(
        self,
        options: list[str],
        selected: str,
        rect: pygame.Rect,
        container: UIPanel | None,
        detached: bool = False,
    ) -> UIDropDownMenu:
        starting_option = selected if selected in options else options[0]
        display_options = [self._display_dropdown_option(option) for option in options]
        starting_display_option = self._display_dropdown_option(starting_option)
        dropdown_rect = rect
        dropdown_container = container
        if detached and container is not None:
            container_rect = container.get_abs_rect()
            dropdown_rect = pygame.Rect(container_rect.x + rect.x, container_rect.y + rect.y, rect.w, rect.h)
            dropdown_container = None
        dropdown = self._track(UIDropDownMenu(display_options, starting_display_option, dropdown_rect, manager=self.manager, container=dropdown_container))
        self._dropdown_values[dropdown] = starting_option
        return dropdown

    @staticmethod
    def _display_setup_mode(mode: str) -> str:
        labels = {
            LOCAL_PLAY_MODE: "Local Play",
            "host_online": "Host Online",
            "join_online": "Join Online",
        }
        return labels.get(mode, mode)

    @staticmethod
    def _setup_mode_value_from_display(label: str) -> str:
        values = {
            "Local Play": LOCAL_PLAY_MODE,
            "Host Online": "host_online",
            "Join Online": "join_online",
        }
        return values.get(label, label)

    @staticmethod
    def _scripted_ai_selection(variant_name: str) -> str:
        return f"{SCRIPTED_AI_SELECTION_PREFIX}{variant_name}"

    @staticmethod
    def _format_scripted_variant_label(variant_name: str) -> str:
        return f"Scripted: {variant_name.replace('_', ' ').title()}"

    def _display_dropdown_option(self, option: str) -> str:
        if option.startswith(SCRIPTED_AI_SELECTION_PREFIX):
            return self._format_scripted_variant_label(option.removeprefix(SCRIPTED_AI_SELECTION_PREFIX))
        return self._display_setup_mode(option)

    def _dropdown_value_from_display(self, label: str) -> str:
        setup_mode = self._setup_mode_value_from_display(label)
        if setup_mode != label:
            return setup_mode
        for variant_name in default_scripted_profiles():
            if label == self._format_scripted_variant_label(variant_name):
                return self._scripted_ai_selection(variant_name)
        return label

    def _ai_option_summary(self) -> str:
        scripted_count = sum(1 for option in self.ai_checkpoint_options if option.startswith(SCRIPTED_AI_SELECTION_PREFIX))
        checkpoint_labels = [Path(option).name for option in self.ai_checkpoint_options if not option.startswith(SCRIPTED_AI_SELECTION_PREFIX)]
        parts: list[str] = []
        if scripted_count:
            parts.append(f"{scripted_count} scripted variants")
        if checkpoint_labels:
            preview = ", ".join(checkpoint_labels[:2])
            if len(checkpoint_labels) > 2:
                preview = f"{preview}, +{len(checkpoint_labels) - 2} more checkpoints"
            parts.append(preview)
        return "; ".join(parts) if parts else self._display_dropdown_option(self._default_checkpoint_option())

    def _discover_checkpoint_options(self) -> list[str]:
        options = [self._scripted_ai_selection(variant_name) for variant_name in default_scripted_profiles()]
        checkpoint_directory = Path(DEFAULT_CHECKPOINT_DIRECTORY)
        if checkpoint_directory.exists():
            for checkpoint_path in sorted(checkpoint_directory.glob("*.pt")):
                options.append(checkpoint_path.as_posix())
        default_option = self._default_checkpoint_option()
        if default_option not in options:
            options.append(default_option)
        return options

    def _default_checkpoint_option(self) -> str:
        return f"{DEFAULT_CHECKPOINT_DIRECTORY}/{DEFAULT_CHECKPOINT_FILE_NAME}"

    def _checkpoint_dropdown_options(self, selected: str) -> list[str]:
        options = list(self.ai_checkpoint_options)
        if selected and selected not in options:
            options.insert(0, selected)
        return options or [self._default_checkpoint_option()]

    def _capture_setup_inputs(self) -> None:
        if self.screen_mode != "setup":
            return
        if hasattr(self, "setup_mode_dropdown"):
            self.setup_game_mode = self._dropdown_value(self.setup_mode_dropdown, self.setup_game_mode)
        if hasattr(self, "cash_entry"):
            self.starting_cash = self.cash_entry.get_text().strip() or self.starting_cash
        if hasattr(self, "animation_speed_entry"):
            self.animation_speed_text = self.animation_speed_entry.get_text().strip() or self.animation_speed_text
        if hasattr(self, "online_host_entry"):
            self.online_remote_host = self.online_host_entry.get_text().strip() or self.online_remote_host
        if hasattr(self, "online_port_entry"):
            self.online_remote_port = self.online_port_entry.get_text().strip() or self.online_remote_port
        if hasattr(self, "online_session_code_entry"):
            self.online_session_code = self.online_session_code_entry.get_text().strip().upper() or self.online_session_code
        if hasattr(self, "online_player_name_entry"):
            self.online_join_player_name = self.online_player_name_entry.get_text().strip() or self.online_join_player_name
        if hasattr(self, "online_reconnect_token_entry"):
            self.online_reconnect_token = self.online_reconnect_token_entry.get_text().strip() or self.online_reconnect_token
        if hasattr(self, "online_discovery_host_entry"):
            self.online_discovery_host = self.online_discovery_host_entry.get_text().strip() or self.online_discovery_host
        if hasattr(self, "online_discovery_port_entry"):
            self.online_discovery_port = self.online_discovery_port_entry.get_text().strip() or self.online_discovery_port
        for index, field in enumerate(self._setup_fields[: self.player_count]):
            self.player_names[index] = field.entry.get_text().strip() or self.player_names[index]
            self.player_roles[index] = self._dropdown_value(field.role_dropdown, self.player_roles[index])
            if self._setup_player_uses_ai_settings(index):
                self.ai_checkpoint_paths[index] = self._dropdown_value(field.checkpoint_dropdown, self.ai_checkpoint_paths[index])
                if field.cooldown_entry is not None:
                    self.ai_cooldown_texts[index] = field.cooldown_entry.get_text().strip() or self.ai_cooldown_texts[index]

    def _process_online_events(self) -> None:
        try:
            if not self.controller.drain_online_events():
                return
            self._apply_online_session_screen_update()
        except Exception as exc:
            self.controller.set_error(str(exc))
            if self.screen_mode == "online_lobby":
                self._build_online_lobby_screen()
            elif self.screen_mode == "online_waiting":
                self._build_online_waiting_screen()
            elif self.screen_mode == "game":
                self._build_game_screen()

    def _refresh_online_session_screen(self) -> None:
        try:
            session_code = None if self.controller.online_session is None else self.controller.online_session.session_code
            self.controller.refresh_online_session(session_code=session_code)
            self._apply_online_session_screen_update()
        except Exception as exc:
            self.controller.set_error(str(exc))
            if self.screen_mode == "online_lobby":
                self._build_online_lobby_screen()
            elif self.screen_mode == "online_waiting":
                self._build_online_waiting_screen()
            elif self.screen_mode == "game":
                self._build_game_screen()

    def _apply_online_session_screen_update(self) -> None:
        session = self.controller.online_session
        if self.screen_mode == "online_lobby":
            if session is not None and session.state == "in_game" and self.controller.frontend_state is not None:
                self.screen_mode = "game"
                self._build_game_screen()
            else:
                self._build_online_lobby_screen()
            return
        if self.screen_mode == "online_waiting":
            self._build_online_waiting_screen()
            return
        if self.screen_mode == "game":
            if session is not None and session.state == "paused":
                if self.controller.is_online_host:
                    self._build_online_lobby_screen()
                else:
                    self._build_online_waiting_screen()
            elif session is not None and session.state == "in_game" and self.controller.frontend_state is not None:
                self._build_game_screen()

    def _host_open_online_slot(self, seat_index: int) -> None:
        try:
            self.controller.open_online_slot(seat_index)
            self._build_online_lobby_screen()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._build_online_lobby_screen()

    def _host_close_online_slot(self, seat_index: int) -> None:
        try:
            self.controller.close_online_slot(seat_index)
            self._build_online_lobby_screen()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._build_online_lobby_screen()

    def _host_assign_ai_online_slot(self, seat_index: int) -> None:
        try:
            checkpoint_path = self.ai_checkpoint_paths[seat_index]
            action_cooldown_seconds = float(self.ai_cooldown_texts[seat_index])
            self.controller.assign_ai_to_online_slot(
                seat_index,
                player_name=f"AI Player {seat_index + 1}",
                checkpoint_path=checkpoint_path,
                action_cooldown_seconds=action_cooldown_seconds,
            )
            self._build_online_lobby_screen()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._build_online_lobby_screen()

    def _set_host_lobby_slot_mode(self, seat_index: int, mode: str) -> None:
        session = self.controller.online_session
        if session is None:
            self.controller.set_error("Online lobby is not available.")
            self._build_setup_screen()
            return
        seat = next((candidate for candidate in session.seats if candidate.seat_index == seat_index), None)
        if seat is None:
            self.controller.set_error(f"Seat {seat_index + 1} is not available.")
            self._build_online_lobby_screen()
            return
        try:
            if mode == seat.status:
                self._build_online_lobby_screen()
                return
            if mode == "open":
                if seat.status == "closed":
                    self.controller.open_online_slot(seat_index)
                elif seat.status == "ai":
                    self.controller.clear_online_slot(seat_index)
            elif mode == "ai":
                checkpoint_path = self.ai_checkpoint_paths[seat_index]
                action_cooldown_seconds = float(self.ai_cooldown_texts[seat_index])
                self.controller.assign_ai_to_online_slot(
                    seat_index,
                    player_name=f"AI Player {seat_index + 1}",
                    checkpoint_path=checkpoint_path,
                    action_cooldown_seconds=action_cooldown_seconds,
                )
            elif mode == "closed":
                if seat.status == "open":
                    self.controller.close_online_slot(seat_index)
                elif seat.status == "ai":
                    self.controller.clear_online_slot(seat_index)
                    self.controller.close_online_slot(seat_index)
            self._build_online_lobby_screen()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._build_online_lobby_screen()

    def _apply_host_lobby_ai_settings(self, seat_index: int) -> None:
        try:
            speed_entry = self._host_lobby_ai_speed_entries.get(seat_index)
            if speed_entry is not None:
                self.ai_cooldown_texts[seat_index] = speed_entry.get_text().strip() or self.ai_cooldown_texts[seat_index]
            action_cooldown_seconds = float(self.ai_cooldown_texts[seat_index])
            if action_cooldown_seconds < 0:
                raise ValueError("AI cooldown cannot be negative.")
            checkpoint_path = self.ai_checkpoint_paths[seat_index]
            self.controller.assign_ai_to_online_slot(
                seat_index,
                player_name=f"AI Player {seat_index + 1}",
                checkpoint_path=checkpoint_path,
                action_cooldown_seconds=action_cooldown_seconds,
            )
            self._build_online_lobby_screen()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._build_online_lobby_screen()

    def _host_clear_online_slot(self, seat_index: int) -> None:
        try:
            self.controller.clear_online_slot(seat_index)
            self._build_online_lobby_screen()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._build_online_lobby_screen()

    def _host_replace_disconnected_with_ai(self, seat_index: int) -> None:
        try:
            self.controller.replace_disconnected_online_slot_with_ai(seat_index)
            if self.controller.online_session is not None and self.controller.online_session.state == "in_game" and self.controller.frontend_state is not None:
                self.screen_mode = "game"
                self._build_game_screen()
            else:
                self._build_online_lobby_screen()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._build_online_lobby_screen()

    def _join_online_seat(self, seat_index: int) -> None:
        try:
            session_code = "" if self.controller.online_session is None else self.controller.online_session.session_code
            self.controller.claim_online_slot(session_code, seat_index, self.online_join_player_name)
            self._build_online_waiting_screen()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._build_online_waiting_screen()

    def _start_online_match(self) -> None:
        try:
            self.controller.start_online_game()
            self.screen_mode = "game"
            self._build_game_screen()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._build_online_lobby_screen()

    def _resolve_lobby_endpoint(self, session_code: str, discovery_host: str, discovery_port: int) -> tuple[str, int]:
        client = RendezvousClient(discovery_host, discovery_port)
        try:
            payload = client.resolve_lobby(session_code)
        finally:
            client.close()
        return str(payload["host"]), int(payload["port"])

    def _register_discovery_lobby(self) -> None:
        session = self.controller.online_session
        client = self.controller.client
        if session is None or client is None:
            return
        discovery_host = getattr(self, "online_discovery_host", "").strip()
        discovery_port_text = getattr(self, "online_discovery_port", "").strip()
        if not discovery_host or not discovery_port_text:
            return
        rendezvous_client = RendezvousClient(discovery_host, int(discovery_port_text))
        try:
            rendezvous_client.register_lobby(session.session_code, client.host, client.port)
        finally:
            rendezvous_client.close()

    def _unregister_discovery_lobby(self) -> None:
        session = self.controller.online_session
        if session is None or not self.controller.is_online_host:
            return
        discovery_host = getattr(self, "online_discovery_host", "").strip()
        discovery_port_text = getattr(self, "online_discovery_port", "").strip()
        if not discovery_host or not discovery_port_text:
            return
        rendezvous_client = RendezvousClient(discovery_host, int(discovery_port_text))
        try:
            rendezvous_client.unregister_lobby(session.session_code)
        except Exception:
            pass
        finally:
            rendezvous_client.close()

    def _setup_player_uses_ai_settings(self, index: int) -> bool:
        return index < self.player_count and self.player_roles[index] == AI_ROLE

    def _ai_cooldown_for_player(self, player_name: str) -> float:
        for index in range(self.player_count):
            if self.player_names[index] == player_name:
                try:
                    return max(0.0, float(self.ai_cooldown_texts[index]))
                except ValueError:
                    return 2.0
        return 2.0

    def _dropdown_value(self, dropdown: UIDropDownMenu | None, fallback: str) -> str:
        if dropdown is None:
            return fallback
        return self._dropdown_values.get(dropdown, fallback)

    def _apply_debug_player_edits(self) -> None:
        if self._debug_panel is None:
            return
        panel = self._debug_panel
        try:
            full_state = self.controller.get_debug_state()
            player_data = next(player for player in full_state["players"] if str(player["name"]) == panel.selected_player_name)
            player_data["cash"] = self._parse_int_range(panel.player_cash_entry.get_text(), "Cash", minimum=0)
            player_data["position"] = self._parse_int_range(panel.player_position_entry.get_text(), "Position", minimum=0, maximum=39)
            player_data["get_out_of_jail_cards"] = self._parse_int_range(panel.player_jail_cards_entry.get_text(), "Jail cards", minimum=0)
            player_data["jail_turns"] = self._parse_int_range(panel.player_jail_turns_entry.get_text(), "Jail turns", minimum=0)
            player_data["role"] = self._dropdown_value(panel.player_role_dropdown, str(player_data["role"]))
            player_data["in_jail"] = self._dropdown_value(panel.player_in_jail_dropdown, str(player_data["in_jail"])).lower() == "true"
            player_data["is_bankrupt"] = self._dropdown_value(panel.player_bankrupt_dropdown, str(player_data["is_bankrupt"])).lower() == "true"
            if not player_data["in_jail"]:
                player_data["jail_turns"] = 0
            self._normalize_debug_full_state(full_state)
            self.controller.apply_debug_state(full_state)
            self.controller.message_history.append(f"Updated player state for {panel.selected_player_name}.")
            self.controller.status_message = f"Updated player state for {panel.selected_player_name}."
            self.controller.last_error = None
            self._rebuild_game_ui()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._rebuild_game_ui()

    def _apply_debug_property_edits(self) -> None:
        if self._debug_panel is None or self._debug_panel.property_space_index is None:
            return
        panel = self._debug_panel
        try:
            full_state = self.controller.get_debug_state()
            property_data = next(space for space in full_state["board"]["properties"] if int(space["index"]) == panel.property_space_index)
            owner_value = self._dropdown_value(panel.property_owner_dropdown, str(property_data.get("owner_name") or "bank"))
            property_data["owner_name"] = None if owner_value == "bank" else owner_value
            property_data["mortgaged"] = self._dropdown_value(panel.property_mortgaged_dropdown, str(property_data.get("mortgaged", False))).lower() == "true"
            if panel.property_building_dropdown is not None:
                property_data["building_count"] = int(self._dropdown_value(panel.property_building_dropdown, str(property_data.get("building_count", 0))))
            self._normalize_debug_full_state(full_state)
            self.controller.apply_debug_state(full_state)
            self.controller.message_history.append(f"Updated space state for {property_data['name']}.")
            self.controller.status_message = f"Updated space state for {property_data['name']}."
            self.controller.last_error = None
            self._rebuild_game_ui()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._rebuild_game_ui()

    def _apply_debug_runtime_edits(self) -> None:
        if self._debug_panel is None:
            return
        panel = self._debug_panel
        self._clear_debug_validation_state()
        try:
            full_state = self.controller.get_debug_state()
            runtime = full_state.setdefault("runtime", {})
            parse_errors: dict[str, str] = {}

            def parse_entry(entry: UITextEntryLine, field_key: str, field_name: str, minimum: int | None = None, maximum: int | None = None) -> int | None:
                try:
                    return self._parse_int_range(entry.get_text(), field_name, minimum=minimum, maximum=maximum)
                except Exception as exc:
                    parse_errors[field_key] = str(exc)
                    return None

            runtime["current_player_name"] = self._dropdown_value(panel.runtime_current_player_dropdown, str(runtime.get("current_player_name") or panel.selected_player_name))
            runtime["current_turn_phase"] = self._dropdown_value(panel.runtime_turn_phase_dropdown, str(runtime.get("current_turn_phase") or PRE_ROLL_PHASE))
            turn_counter = parse_entry(panel.runtime_turn_counter_entry, "runtime.turn_counter", "Turn counter", minimum=0)
            houses_remaining = parse_entry(panel.runtime_houses_entry, "runtime.houses", "House bank", minimum=0)
            hotels_remaining = parse_entry(panel.runtime_hotels_entry, "runtime.hotels", "Hotel bank", minimum=0)
            if turn_counter is not None:
                runtime["turn_counter"] = turn_counter
            if houses_remaining is not None:
                runtime["houses_remaining"] = houses_remaining
            if hotels_remaining is not None:
                runtime["hotels_remaining"] = hotels_remaining

            continuation_player = self._dropdown_value(panel.runtime_continuation_player_dropdown, "none")
            if continuation_player == "none":
                runtime["pending_turn_continuation"] = None
            else:
                continuation_doubles = parse_entry(panel.runtime_continuation_doubles_entry, "runtime.continuation_doubles", "Continuation doubles", minimum=0)
                runtime["pending_turn_continuation"] = {
                    "player_name": continuation_player,
                    "doubles_in_row": 0 if continuation_doubles is None else continuation_doubles,
                    "rolled_double": self._dropdown_value(panel.runtime_continuation_rolled_double_dropdown, "False").lower() == "true",
                }

            pending_kind = self._dropdown_value(panel.runtime_pending_action_dropdown, self._current_pending_action_kind(runtime))
            existing_pending_kind = self._current_pending_action_kind(runtime)
            existing_pending_payloads = {
                "purchase": runtime.get("pending_purchase_decision"),
                "auction": runtime.get("pending_auction"),
                "jail": runtime.get("pending_jail_decision"),
                "property": runtime.get("pending_property_action"),
                "trade": runtime.get("pending_trade_decision"),
            }
            self._clear_pending_actions(runtime)
            if pending_kind != "none":
                if pending_kind == existing_pending_kind and existing_pending_payloads.get(pending_kind) is not None:
                    self._set_pending_action_payload(runtime, pending_kind, existing_pending_payloads[pending_kind])
                else:
                    self._set_pending_action_payload(runtime, pending_kind, self._build_debug_pending_state(full_state, pending_kind, runtime["current_player_name"]))
                self._apply_debug_pending_detail_edits(runtime, panel, parse_entry)

            if parse_errors:
                self._set_debug_validation_errors(parse_errors)
                self.controller.set_error("Runtime edit has invalid fields. Fix the highlighted inputs.")
                return

            validation_errors = self._validate_debug_runtime_state(full_state)
            if validation_errors:
                self._set_debug_validation_errors(validation_errors)
                self.controller.set_error("Runtime edit would create an inconsistent state. Fix the highlighted inputs.")
                return

            self._normalize_debug_full_state(full_state)
            self._clear_debug_validation_state()
            self.controller.apply_debug_state(full_state)
            self.controller.message_history.append("Updated runtime state.")
            self.controller.status_message = "Updated runtime state."
            self.controller.last_error = None
            self._rebuild_game_ui()
        except Exception as exc:
            self.controller.set_error(str(exc))

    def _apply_debug_pending_detail_edits(
        self,
        runtime: dict[str, Any],
        panel: DebugPanelState,
        parse_entry: Any,
    ) -> None:
        pending_kind = self._current_pending_action_kind(runtime)
        if pending_kind == "auction":
            auction = runtime.get("pending_auction")
            if not isinstance(auction, dict):
                return
            if panel.runtime_auction_current_bid_entry is not None:
                current_bid = parse_entry(panel.runtime_auction_current_bid_entry, "runtime.auction_current_bid", "Auction current bid", minimum=0)
                if current_bid is not None:
                    auction["current_bid"] = current_bid
            if panel.runtime_auction_bidder_index_entry is not None:
                bidder_index = parse_entry(panel.runtime_auction_bidder_index_entry, "runtime.auction_bidder_index", "Auction bidder index", minimum=0)
                if bidder_index is not None:
                    auction["current_bidder_index"] = bidder_index
            if panel.runtime_auction_winner_dropdown is not None:
                winner_name = self._dropdown_value(panel.runtime_auction_winner_dropdown, "none")
                auction["current_winner_name"] = None if winner_name == "none" else winner_name
            return

        if pending_kind == "trade":
            trade = runtime.get("pending_trade_decision")
            if not isinstance(trade, dict):
                return
            if panel.runtime_trade_proposer_dropdown is not None:
                trade["proposer_name"] = self._dropdown_value(panel.runtime_trade_proposer_dropdown, str(trade.get("proposer_name") or ""))
            if panel.runtime_trade_receiver_dropdown is not None:
                trade["receiver_name"] = self._dropdown_value(panel.runtime_trade_receiver_dropdown, str(trade.get("receiver_name") or ""))
            if panel.runtime_trade_proposer_cash_entry is not None:
                proposer_cash = parse_entry(panel.runtime_trade_proposer_cash_entry, "runtime.trade_proposer_cash", "Trade proposer cash", minimum=0)
                if proposer_cash is not None:
                    trade["proposer_cash"] = proposer_cash
            if panel.runtime_trade_receiver_cash_entry is not None:
                receiver_cash = parse_entry(panel.runtime_trade_receiver_cash_entry, "runtime.trade_receiver_cash", "Trade receiver cash", minimum=0)
                if receiver_cash is not None:
                    trade["receiver_cash"] = receiver_cash
            if panel.runtime_trade_note_entry is not None:
                trade["note"] = panel.runtime_trade_note_entry.get_text().strip()

    def _validate_debug_runtime_state(self, full_state: dict[str, Any]) -> dict[str, str]:
        errors: dict[str, str] = {}
        runtime = full_state.get("runtime", {})
        players = full_state.get("players", [])
        active_names = [str(player["name"]) for player in players if not bool(player.get("is_bankrupt", False))]
        player_cash = {str(player["name"]): int(player.get("cash", 0)) for player in players}

        current_player_name = str(runtime.get("current_player_name") or "")
        if current_player_name not in active_names:
            errors["runtime.current_player"] = "Current player must be a non-bankrupt player."

        houses_used, hotels_used = self._used_building_supply(full_state)
        expected_houses = BANK_HOUSES - houses_used
        expected_hotels = BANK_HOTELS - hotels_used
        houses_remaining = int(runtime.get("houses_remaining", 0))
        hotels_remaining = int(runtime.get("hotels_remaining", 0))
        if houses_remaining > BANK_HOUSES:
            errors["runtime.houses"] = f"House bank cannot exceed {BANK_HOUSES}."
        elif houses_remaining != expected_houses:
            errors["runtime.houses"] = f"House bank must match the board state: expected {expected_houses}."
        if hotels_remaining > BANK_HOTELS:
            errors["runtime.hotels"] = f"Hotel bank cannot exceed {BANK_HOTELS}."
        elif hotels_remaining != expected_hotels:
            errors["runtime.hotels"] = f"Hotel bank must match the board state: expected {expected_hotels}."

        continuation = runtime.get("pending_turn_continuation")
        if isinstance(continuation, dict):
            continuation_player = str(continuation.get("player_name") or "")
            if continuation_player not in active_names:
                errors["runtime.continuation_player"] = "Continuation player must be a non-bankrupt player."

        pending_kind = self._current_pending_action_kind(runtime)
        if pending_kind == "auction":
            auction = runtime.get("pending_auction")
            if isinstance(auction, dict):
                active_player_names = [str(name) for name in auction.get("active_player_names", [])]
                if not active_player_names:
                    errors["runtime.auction_bidder_index"] = "Auction must have at least one active bidder."
                bidder_index = int(auction.get("current_bidder_index", 0))
                if active_player_names and not 0 <= bidder_index < len(active_player_names):
                    errors["runtime.auction_bidder_index"] = f"Bidder index must be between 0 and {len(active_player_names) - 1}."
                current_winner = auction.get("current_winner_name")
                if current_winner is not None and str(current_winner) not in active_player_names:
                    errors["runtime.auction_winner"] = "Auction winner must be one of the active bidders."
                current_bid = int(auction.get("current_bid", 0))
                if current_winner is not None and current_bid <= 0:
                    errors["runtime.auction_current_bid"] = "A winning bidder requires a positive current bid."
        elif pending_kind == "trade":
            trade = runtime.get("pending_trade_decision")
            if isinstance(trade, dict):
                proposer = str(trade.get("proposer_name") or "")
                receiver = str(trade.get("receiver_name") or "")
                if proposer not in active_names:
                    errors["runtime.trade_proposer"] = "Trade proposer must be a non-bankrupt player."
                if receiver not in active_names:
                    errors["runtime.trade_receiver"] = "Trade receiver must be a non-bankrupt player."
                if proposer and receiver and proposer == receiver:
                    errors["runtime.trade_proposer"] = "Trade proposer and receiver must be different players."
                    errors["runtime.trade_receiver"] = "Trade proposer and receiver must be different players."
                proposer_cash = int(trade.get("proposer_cash", 0))
                receiver_cash = int(trade.get("receiver_cash", 0))
                if proposer in player_cash and proposer_cash > player_cash[proposer]:
                    errors["runtime.trade_proposer_cash"] = f"Trade proposer cash cannot exceed {player_cash[proposer]}."
                if receiver in player_cash and receiver_cash > player_cash[receiver]:
                    errors["runtime.trade_receiver_cash"] = f"Trade receiver cash cannot exceed {player_cash[receiver]}."

        return errors

    def _used_building_supply(self, full_state: dict[str, Any]) -> tuple[int, int]:
        houses_used = 0
        hotels_used = 0
        for property_data in full_state.get("board", {}).get("properties", []):
            building_count = int(property_data.get("building_count", 0) or 0)
            if building_count >= 5:
                hotels_used += 1
            else:
                houses_used += building_count
        return houses_used, hotels_used

    def _set_debug_next_roll(self) -> None:
        if self._debug_panel is None:
            return
        try:
            die_one = self._parse_int_range(self._debug_panel.next_die_one_entry.get_text(), "First die", minimum=1, maximum=6)
            die_two = self._parse_int_range(self._debug_panel.next_die_two_entry.get_text(), "Second die", minimum=1, maximum=6)
            full_state = self.controller.get_debug_state()
            runtime = full_state.setdefault("runtime", {})
            forced_rolls = runtime.setdefault("debug_next_rolls_by_player", {})
            forced_rolls[self._debug_panel.selected_player_name] = [[die_one, die_two]]
            self.controller.apply_debug_state(full_state)
            self.controller.message_history.append(f"Forced the next roll for {self._debug_panel.selected_player_name} to {die_one} and {die_two}.")
            self.controller.status_message = f"Forced the next roll for {self._debug_panel.selected_player_name}."
            self.controller.last_error = None
            self._rebuild_game_ui()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._rebuild_game_ui()

    def _clear_debug_next_roll(self) -> None:
        if self._debug_panel is None:
            return
        try:
            full_state = self.controller.get_debug_state()
            runtime = full_state.setdefault("runtime", {})
            forced_rolls = runtime.setdefault("debug_next_rolls_by_player", {})
            forced_rolls.pop(self._debug_panel.selected_player_name, None)
            self.controller.apply_debug_state(full_state)
            self.controller.message_history.append(f"Cleared forced next roll for {self._debug_panel.selected_player_name}.")
            self.controller.status_message = f"Cleared forced next roll for {self._debug_panel.selected_player_name}."
            self.controller.last_error = None
            self._rebuild_game_ui()
        except Exception as exc:
            self.controller.set_error(str(exc))
            self._rebuild_game_ui()

    def _normalize_debug_full_state(self, full_state: dict[str, Any]) -> None:
        players = full_state.get("players", [])
        active_names = [str(player["name"]) for player in players if not bool(player.get("is_bankrupt", False))]
        runtime = full_state.setdefault("runtime", {})
        current_player_name = runtime.get("current_player_name")
        if current_player_name not in active_names:
            runtime["current_player_name"] = active_names[0] if active_names else None

    def _runtime_panel_html(self, full_state: dict[str, Any]) -> str:
        runtime = full_state["runtime"]
        continuation = runtime.get("pending_turn_continuation")
        continuation_text = "none" if continuation is None else f"{continuation.get('player_name')} / d={continuation.get('doubles_in_row', 0)} / dbl={continuation.get('rolled_double', False)}"
        return (
            f"<b>Turn:</b> {runtime.get('turn_counter', 0) + 1}<br>"
            f"<b>Current:</b> {escape(str(runtime.get('current_player_name') or 'n/a'))}<br>"
            f"<b>Pending:</b> {escape(self._current_pending_action_kind(runtime))}<br>"
            f"<b>Continuation:</b> {escape(continuation_text)}"
        )

    def _build_header_badges(self, full_state: dict[str, Any], sidebar_x: int, sidebar_width: int, players: tuple[PlayerView, ...]) -> int:
        runtime = full_state["runtime"]
        pending_kind = self._current_pending_action_kind(runtime)
        current_player_name = str(runtime.get("current_player_name") or "n/a")
        queued_rolls = runtime.get("debug_next_rolls_by_player", {}).get(current_player_name, [])
        queued_text = "none" if not queued_rolls else ", ".join(f"{roll[0]}+{roll[1]}" for roll in queued_rolls)
        badge_html = [
            (
                "<b>Debug Runtime</b><br>"
                f"Current: {escape(current_player_name)}<br>"
                f"Pending: {escape(pending_kind)}<br>"
                f"Queue: {escape(queued_text)}"
            ),
            (
                "<b>Player Cash</b><br>"
                + "<br>".join(f"{escape(player.name)}: ${player.cash}" for player in players)
            ),
        ]
        badge_gap = 14
        badge_y = 54
        badge_height = 72
        badge_width = (sidebar_width - badge_gap) // 2
        for index, html in enumerate(badge_html):
            badge_x = sidebar_x + index * (badge_width + badge_gap)
            badge = self._track(UIPanel(pygame.Rect(badge_x, badge_y, badge_width, badge_height), manager=self.manager))
            self._track(UITextBox(html, pygame.Rect(10, 8, badge_width - 20, badge_height - 16), manager=self.manager, container=badge))
        return badge_y + badge_height

    def _player_cash_summary_html(self, players: tuple[PlayerView, ...], current_player_name: str | None) -> str:
        lines = ["<b>Money on Hand</b>"]
        for player in players:
            status_parts: list[str] = []
            if current_player_name is not None and player.name == current_player_name:
                status_parts.append("up now")
            if player.in_jail:
                status_parts.append("jail")
            if player.is_bankrupt:
                status_parts.append("bankrupt")
            suffix = "" if not status_parts else f" ({', '.join(status_parts)})"
            lines.append(f"<b>{escape(player.name)}</b>: ${player.cash}{escape(suffix)}")
        return "<br>".join(lines)

    def _player_cash_badge_text(self, players: tuple[PlayerView, ...]) -> str:
        return "Cash: " + " | ".join(f"{player.name} ${player.cash}" for player in players)

    def _pending_action_editor_html(self, full_state: dict[str, Any]) -> str:
        runtime = full_state["runtime"]
        pending_kind = self._current_pending_action_kind(runtime)
        if pending_kind == "none":
            return "<b>Pending Editor</b><br>No pending action is active."
        payload = self._pending_action_payload(runtime, pending_kind)
        if pending_kind == "auction" and isinstance(payload, dict):
            active_names = ", ".join(str(name) for name in payload.get("active_player_names", [])) or "none"
            return (
                f"<b>Pending:</b> auction<br>"
                f"<b>Property:</b> {escape(str(payload.get('property_name') or 'n/a'))}<br>"
                f"<b>Bid:</b> ${payload.get('current_bid', 0)} | <b>Winner:</b> {escape(str(payload.get('current_winner_name') or 'none'))}<br>"
                f"<b>Active:</b> {escape(active_names)}"
            )
        if pending_kind == "trade" and isinstance(payload, dict):
            return (
                f"<b>Pending:</b> trade<br>"
                f"<b>Route:</b> {escape(str(payload.get('proposer_name') or 'n/a'))} -> {escape(str(payload.get('receiver_name') or 'n/a'))}<br>"
                f"<b>Cash:</b> ${payload.get('proposer_cash', 0)} / ${payload.get('receiver_cash', 0)}<br>"
                f"<b>Note:</b> {escape(str(payload.get('note') or 'none'))}"
            )
        return f"<b>Pending:</b> {escape(pending_kind)}<br>{escape(str(payload))}"

    @staticmethod
    def _current_pending_action_kind(runtime: dict[str, Any]) -> str:
        pending_map = {
            "purchase": runtime.get("pending_purchase_decision"),
            "auction": runtime.get("pending_auction"),
            "jail": runtime.get("pending_jail_decision"),
            "property": runtime.get("pending_property_action"),
            "trade": runtime.get("pending_trade_decision"),
        }
        for kind, payload in pending_map.items():
            if payload is not None:
                return kind
        return "none"

    @staticmethod
    def _pending_action_payload(runtime: dict[str, Any], kind: str) -> Any:
        payload_map = {
            "purchase": runtime.get("pending_purchase_decision"),
            "auction": runtime.get("pending_auction"),
            "jail": runtime.get("pending_jail_decision"),
            "property": runtime.get("pending_property_action"),
            "trade": runtime.get("pending_trade_decision"),
        }
        return payload_map.get(kind)

    @staticmethod
    def _clear_pending_actions(runtime: dict[str, Any]) -> None:
        runtime["pending_purchase_decision"] = None
        runtime["pending_auction"] = None
        runtime["pending_jail_decision"] = None
        runtime["pending_property_action"] = None
        runtime["pending_trade_decision"] = None

    @staticmethod
    def _set_pending_action_payload(runtime: dict[str, Any], kind: str, payload: dict[str, Any]) -> None:
        field_map = {
            "purchase": "pending_purchase_decision",
            "auction": "pending_auction",
            "jail": "pending_jail_decision",
            "property": "pending_property_action",
            "trade": "pending_trade_decision",
        }
        runtime[field_map[kind]] = payload

    def _build_debug_pending_state(self, full_state: dict[str, Any], kind: str, current_player_name: str) -> dict[str, Any]:
        players = full_state["players"]
        current_player = next(player for player in players if str(player["name"]) == current_player_name)
        properties = full_state["board"]["properties"]
        selected_property = next((space for space in properties if int(space["index"]) == self.controller.selected_space_index), None)
        if selected_property is None and properties:
            selected_property = properties[0]

        if kind == "purchase":
            if selected_property is None:
                raise ValueError("No property is available for a purchase pending state.")
            return {
                "player_name": current_player_name,
                "property_index": int(selected_property["index"]),
                "property_name": str(selected_property["name"]),
                "price": int(selected_property.get("price", 0)),
            }
        if kind == "auction":
            if selected_property is None:
                raise ValueError("No property is available for an auction pending state.")
            active_names = [str(player["name"]) for player in players if not bool(player.get("is_bankrupt", False))]
            current_bidder_index = active_names.index(current_player_name) if current_player_name in active_names else 0
            return {
                "property_index": int(selected_property["index"]),
                "property_name": str(selected_property["name"]),
                "eligible_player_names": list(active_names),
                "active_player_names": list(active_names),
                "current_bid": 0,
                "current_winner_name": None,
                "current_bidder_index": current_bidder_index,
            }
        if kind == "jail":
            available_actions = ["roll"]
            if int(current_player.get("cash", 0)) >= JAIL_FINE:
                available_actions.append("pay_fine")
            if int(current_player.get("get_out_of_jail_cards", 0)) > 0:
                available_actions.append("use_card")
            return {
                "player_name": current_player_name,
                "available_actions": available_actions,
            }
        if kind == "property":
            owned_property = next((space for space in properties if str(space.get("owner_name") or "") == current_player_name), None)
            target_property = owned_property or selected_property
            if target_property is None:
                raise ValueError("No property is available for a property-action pending state.")
            return {
                "action_type": "mortgage",
                "player_name": current_player_name,
                "property_name": str(target_property["name"]),
                "property_index": int(target_property["index"]),
            }
        if kind == "trade":
            other_player = next((player for player in players if str(player["name"]) != current_player_name and not bool(player.get("is_bankrupt", False))), None)
            if other_player is None:
                raise ValueError("A trade pending state needs another non-bankrupt player.")
            return {
                "proposer_name": current_player_name,
                "receiver_name": str(other_player["name"]),
                "proposer_cash": 0,
                "receiver_cash": 0,
                "proposer_property_names": [],
                "receiver_property_names": [],
                "proposer_jail_cards": 0,
                "receiver_jail_cards": 0,
                "note": "Debug-created trade offer.",
            }
        raise ValueError(f"Unsupported pending action kind: {kind}")

    @staticmethod
    def _parse_int_range(raw_value: str, field_name: str, minimum: int | None = None, maximum: int | None = None) -> int:
        value = int(raw_value.strip())
        if minimum is not None and value < minimum:
            raise ValueError(f"{field_name} must be at least {minimum}.")
        if maximum is not None and value > maximum:
            raise ValueError(f"{field_name} must be at most {maximum}.")
        return value

    def _adjust_trade_value(self, field_name: str, delta: int) -> None:
        if self._trade_prompt is None:
            return
        prompt = self._trade_prompt
        if field_name == "proposer_cash":
            prompt.proposer_cash_value = min(prompt.proposer.cash, max(0, prompt.proposer_cash_value + delta))
        elif field_name == "receiver_cash":
            prompt.receiver_cash_value = min(prompt.receiver.cash, max(0, prompt.receiver_cash_value + delta))
        elif field_name == "proposer_jail_cards":
            prompt.proposer_jail_cards_value = min(prompt.proposer.get_out_of_jail_cards, max(0, prompt.proposer_jail_cards_value + delta))
        elif field_name == "receiver_jail_cards":
            prompt.receiver_jail_cards_value = min(prompt.receiver.get_out_of_jail_cards, max(0, prompt.receiver_jail_cards_value + delta))
        else:
            raise ValueError(f"Unsupported trade field: {field_name}")
        self._refresh_trade_prompt_display()

    def _toggle_trade_property(self, side: str, property_name: str) -> None:
        if self._trade_prompt is None:
            return
        prompt = self._trade_prompt
        if side == "proposer":
            selected = prompt.proposer_selected_properties
        elif side == "receiver":
            selected = prompt.receiver_selected_properties
        else:
            raise ValueError(f"Unsupported trade side: {side}")
        if property_name in selected:
            selected.remove(property_name)
        else:
            selected.add(property_name)
        self._refresh_trade_prompt_display()

    def _change_trade_property_page(self, side: str, delta: int) -> None:
        if self._trade_prompt is None:
            return
        prompt = self._trade_prompt
        if side == "proposer":
            total = len(prompt.proposer.properties)
            max_page = max(0, (total - 1) // len(prompt.proposer_property_buttons)) if prompt.proposer_property_buttons else 0
            prompt.proposer_property_page = max(0, min(max_page, prompt.proposer_property_page + delta))
        elif side == "receiver":
            total = len(prompt.receiver.properties)
            max_page = max(0, (total - 1) // len(prompt.receiver_property_buttons)) if prompt.receiver_property_buttons else 0
            prompt.receiver_property_page = max(0, min(max_page, prompt.receiver_property_page + delta))
        else:
            raise ValueError(f"Unsupported trade side: {side}")
        self._refresh_trade_prompt_display()

    def _refresh_trade_prompt_display(self) -> None:
        if self._trade_prompt is None:
            return
        prompt = self._trade_prompt
        prompt.proposer_cash_label.set_text(f"${prompt.proposer_cash_value}")
        prompt.receiver_cash_label.set_text(f"${prompt.receiver_cash_value}")
        prompt.proposer_jail_cards_label.set_text(str(prompt.proposer_jail_cards_value))
        prompt.receiver_jail_cards_label.set_text(str(prompt.receiver_jail_cards_value))
        self._refresh_trade_property_buttons(
            side="proposer",
            properties=prompt.proposer.properties,
            selected=prompt.proposer_selected_properties,
            buttons=prompt.proposer_property_buttons,
            page=prompt.proposer_property_page,
        )
        self._refresh_trade_property_buttons(
            side="receiver",
            properties=prompt.receiver.properties,
            selected=prompt.receiver_selected_properties,
            buttons=prompt.receiver_property_buttons,
            page=prompt.receiver_property_page,
        )

    def _refresh_trade_property_buttons(
        self,
        *,
        side: str,
        properties: tuple[str, ...],
        selected: set[str],
        buttons: list[UIButton],
        page: int,
    ) -> None:
        if not buttons:
            return
        page_size = len(buttons)
        start = page * page_size
        visible = properties[start : start + page_size]
        for index, button in enumerate(buttons):
            if index < len(visible):
                property_name = visible[index]
                prefix = "[x]" if property_name in selected else "[ ]"
                button.set_text(f"{prefix} {property_name}")
                self._ui_commands[button] = ("trade_toggle_property", (side, property_name))
            else:
                button.set_text("-")
                self._ui_commands.pop(button, None)

    @staticmethod
    def _parse_non_negative_int(raw_value: str, field_name: str) -> int:
        value = raw_value.strip() or "0"
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"{field_name} cannot be negative.")
        return parsed

    def _parse_jail_cards(self, raw_value: str, player: PlayerView, field_name: str) -> int:
        parsed = self._parse_non_negative_int(raw_value, field_name)
        if parsed > player.get_out_of_jail_cards:
            raise ValueError(f"{field_name} cannot exceed {player.get_out_of_jail_cards}.")
        return parsed

    @staticmethod
    def _parse_csv_names(raw_value: str) -> list[str]:
        if not raw_value.strip():
            return []
        names = [item.strip() for item in raw_value.split(",") if item.strip()]
        if len(set(names)) != len(names):
            raise ValueError("Trade property lists cannot contain duplicates.")
        return names

    def _parse_trade_properties(self, raw_value: str, player: PlayerView, field_name: str) -> list[str]:
        names = self._parse_csv_names(raw_value)
        unknown = [name for name in names if name not in player.properties]
        if unknown:
            raise ValueError(f"{field_name} contain unknown properties: {', '.join(unknown)}")
        return names

    def _track(self, element: Any) -> Any:
        self._elements.append(element)
        return element

    def _register_debug_field(self, field_key: str, element: Any) -> None:
        self._debug_field_elements[field_key] = element

    def _clear_debug_validation_state(self) -> None:
        self._debug_validation_messages.clear()
        self._debug_field_elements.clear()
        self._debug_invalid_elements = []

    def _set_debug_validation_errors(self, messages: dict[str, str]) -> None:
        self._debug_validation_messages = dict(messages)
        invalid_elements: list[Any] = []
        for field_key in messages:
            element = self._debug_field_elements.get(field_key)
            if element is not None and element not in invalid_elements:
                invalid_elements.append(element)
        self._debug_invalid_elements = invalid_elements

    def _clear_elements(self) -> None:
        self._action_buttons.clear()
        self._ui_commands.clear()
        self._dropdown_values.clear()
        self._host_lobby_slot_dropdowns.clear()
        self._host_lobby_ai_checkpoint_dropdowns.clear()
        self._host_lobby_ai_speed_entries.clear()
        self._host_lobby_ai_apply_buttons.clear()
        self._prompt = None
        self._trade_prompt = None
        self._debug_panel = None
        self._clear_debug_validation_state()
        for element in self._elements:
            try:
                element.kill()
            except Exception:
                pass
        self._elements = []

    def _recalculate_layout(self, width: int, height: int) -> None:
        width = max(theme.MIN_WINDOW_WIDTH, width)
        height = max(theme.MIN_WINDOW_HEIGHT, height)
        if self.debug_mode:
            available_debug_sidebar_width = width - theme.BOARD_MARGIN * 3 - theme.MIN_DEBUG_BOARD_SIZE
            self.sidebar_width = min(theme.DEBUG_SIDEBAR_MAX_WIDTH, max(theme.DEBUG_SIDEBAR_MIN_WIDTH, available_debug_sidebar_width))
            min_board_size = theme.MIN_DEBUG_BOARD_SIZE
        else:
            available_sidebar_width = width - theme.BOARD_MARGIN * 3 - theme.MIN_STANDARD_BOARD_SIZE
            self.sidebar_width = min(theme.SIDEBAR_MAX_WIDTH, max(theme.SIDEBAR_MIN_WIDTH, available_sidebar_width))
            min_board_size = theme.MIN_STANDARD_BOARD_SIZE
        board_height = height - theme.TOOLBAR_HEIGHT - theme.STATUS_BAR_HEIGHT - theme.BOARD_MARGIN * 2
        board_width = width - self.sidebar_width - theme.BOARD_MARGIN * 3
        board_size = max(min_board_size, min(board_height, board_width))
        self.board_rect = pygame.Rect(theme.BOARD_MARGIN, theme.TOOLBAR_HEIGHT + theme.BOARD_MARGIN, board_size, board_size)
        self.board_renderer.update_board_rect(self.board_rect)

    def _resize_window(self, width: int, height: int) -> None:
        width = max(theme.MIN_WINDOW_WIDTH, width)
        height = max(theme.MIN_WINDOW_HEIGHT, height)
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.manager.set_window_resolution((width, height))
        self._recalculate_layout(width, height)
        if self.screen_mode == "game":
            self._rebuild_game_ui()
        elif self.screen_mode == "replay":
            self._rebuild_replay_ui()
        elif self.screen_mode == "online_lobby":
            self._build_online_lobby_screen()
        elif self.screen_mode == "online_waiting":
            self._build_online_waiting_screen()
        else:
            self._capture_setup_inputs()
            self._build_setup_screen()

    def _display_actions(self, actions: tuple[LegalActionOption, ...]) -> list[Any]:
        submenu_action_map = {
            "build": "request_build",
            "sell_building": "request_sell_building",
            "mortgage": "request_mortgage",
            "unmortgage": "request_unmortgage",
        }
        submenu_labels = {
            "build": "Build Houses",
            "sell_building": "Sell Buildings",
            "mortgage": "Mortgage Properties",
            "unmortgage": "Redeem Mortgages",
        }

        if self._property_submenu_mode in submenu_action_map:
            selected_action_type = submenu_action_map[self._property_submenu_mode]
            submenu_actions = [action for action in actions if action.action_type == selected_action_type]
            if not submenu_actions:
                self._property_submenu_mode = None
                self._property_submenu_page = 0
            else:
                return submenu_actions

        display_actions: list[Any] = []
        for submenu_key, action_type in submenu_action_map.items():
            if any(action.action_type == action_type for action in actions):
                display_actions.append(("submenu", submenu_key, submenu_labels[submenu_key]))
        for action in actions:
            if action.action_type in submenu_action_map.values():
                continue
            display_actions.append(action)
        return display_actions

    def _paginate_property_submenu_actions(self, actions: list[Any], max_buttons: int) -> list[Any]:
        if self._property_submenu_mode is None or not actions:
            return actions[:max_buttons]

        if max_buttons <= 1:
            return [("submenu_back", None, "Back to moves")]

        if len(actions) + 1 <= max_buttons:
            self._property_submenu_page = 0
            return actions + [("submenu_back", None, "Back to moves")]

        action_slots = max(1, max_buttons - 3)
        total_pages = max(1, (len(actions) + action_slots - 1) // action_slots)
        current_page = min(self._property_submenu_page, total_pages - 1)
        self._property_submenu_page = current_page
        start_index = current_page * action_slots
        end_index = start_index + action_slots

        visible_actions = list(actions[start_index:end_index])
        if current_page > 0:
            visible_actions.append(("submenu_prev", None, f"Previous options ({current_page}/{total_pages})"))
        if current_page < total_pages - 1:
            visible_actions.append(("submenu_next", None, f"More options ({current_page + 2}/{total_pages})"))
        visible_actions.append(("submenu_back", None, "Back to moves"))
        return visible_actions[:max_buttons]

    @staticmethod
    def _friendly_phase_name(phase: str | None) -> str:
        if phase == PRE_ROLL_PHASE:
            return "Roll"
        if phase == IN_TURN_PHASE:
            return "Resolving move"
        if phase == POST_ROLL_PHASE:
            return "Wrap up"
        return "n/a"

    @staticmethod
    def _friendly_space_type(space_type: str | None) -> str:
        if not space_type:
            return "Unknown"
        return space_type.replace("_", " ").title()

    @staticmethod
    def _friendly_property_action_name(action_type: str) -> str:
        labels = {
            "build": "Build",
            "sell_building": "Sell building",
            "mortgage": "Mortgage",
            "unmortgage": "Redeem mortgage",
        }
        return labels.get(action_type, action_type.replace("_", " ").title())

    def _action_button_label(self, action: LegalActionOption) -> str:
        property_name = action.property_name or "property"
        if action.action_type == "start_turn":
            return "Roll the dice"
        if action.action_type == "end_turn":
            return "Finish turn"
        if action.action_type == "buy_property":
            return f"Buy {property_name}"
        if action.action_type == "decline_property":
            return f"Send {property_name} to auction"
        if action.action_type == "place_auction_bid":
            return f"Bid on {property_name}"
        if action.action_type == "pass_auction":
            return f"Pass on {property_name}"
        if action.action_type == "jail_roll":
            return "Roll for doubles"
        if action.action_type == "jail_pay_fine":
            return f"Pay ${JAIL_FINE} to leave jail"
        if action.action_type == "jail_use_card":
            return "Use Get Out of Jail Free card"
        if action.action_type == "confirm_property_action":
            if action.property_name:
                property_action_name = self._friendly_property_action_name(self._property_submenu_mode or "")
                if self._property_submenu_mode in {"build", "sell_building", "mortgage", "unmortgage"}:
                    return f"{property_action_name} {action.property_name}"
                return f"Manage {action.property_name}"
            return "Confirm property change"
        if action.action_type == "cancel_property_action":
            return "Keep things as they are"
        if action.action_type == "accept_trade":
            return f"Accept trade from {action.target_player_name or 'player'}"
        if action.action_type == "reject_trade":
            return f"Reject trade from {action.target_player_name or 'player'}"
        if action.action_type == "counter_trade":
            return f"Counter trade from {action.target_player_name or 'player'}"
        if action.action_type == "propose_trade":
            return f"Offer a trade to {action.target_player_name or 'player'}"
        if action.action_type == "request_build":
            return f"Build on {property_name}"
        if action.action_type == "request_sell_building":
            return f"Sell building on {property_name}"
        if action.action_type == "request_mortgage":
            return f"Mortgage {property_name}"
        if action.action_type == "request_unmortgage":
            return f"Redeem {property_name}"
        return action.description

    def _start_action_animation(
        self,
        action: LegalActionOption,
        interaction: Any,
        previous_state: FrontendStateView | None,
        final_state: FrontendStateView | None,
    ) -> bool:
        return self._start_actor_animation(action.actor_name, interaction, previous_state, final_state)

    def _start_actor_animation(
        self,
        actor_name: str,
        interaction: Any,
        previous_state: FrontendStateView | None,
        final_state: FrontendStateView | None,
    ) -> bool:
        if self.animation_speed <= 0 or previous_state is None or final_state is None:
            return False
        steps = self._extract_animation_segments(interaction.messages, actor_name, previous_state)
        if not steps:
            return False
        first_step = steps[0]
        initial_phase = "dice" if first_step.dice_values is not None else "move"
        self._animation = AnimationState(
            previous_state=previous_state,
            final_state=final_state,
            player_name=actor_name,
            steps=steps,
            speed=self.animation_speed,
            phase=initial_phase,
        )
        return True

    def _update_ai_turns(self, delta_time: float) -> None:
        if self.screen_mode != "game":
            self._pending_ai_step = None
            return
        if not self.controller.client.owns_server:
            self._pending_ai_step = None
            return
        if self._animation is not None or self._prompt is not None or self._trade_prompt is not None:
            return
        frontend_state = self.controller.frontend_state
        if frontend_state is None:
            self._pending_ai_step = None
            return
        active_turn_plan = frontend_state.active_turn_plan
        if active_turn_plan.player_role != AI_ROLE:
            self._pending_ai_step = None
            return

        if self._pending_ai_step is None or self._pending_ai_step.actor_name != active_turn_plan.player_name:
            self._pending_ai_step = AIStepState(
                actor_name=active_turn_plan.player_name,
                remaining_delay=self._ai_cooldown_for_player(active_turn_plan.player_name),
            )

        self._pending_ai_step.remaining_delay = max(0.0, self._pending_ai_step.remaining_delay - delta_time)
        if self._pending_ai_step.remaining_delay > 0:
            return

        try:
            previous_state = self.controller.frontend_state
            interaction = self.controller.step_ai()
            actor_name = self._pending_ai_step.actor_name
            self._pending_ai_step = None
            if not self._start_actor_animation(actor_name, interaction, previous_state, self.controller.frontend_state):
                self._rebuild_game_ui()
        except Exception as exc:
            self._pending_ai_step = None
            self.controller.set_error(str(exc))
            self._rebuild_game_ui()

    def _update_animation(self, delta_time: float) -> None:
        if self._animation is None:
            self._update_replay(delta_time)
            return
        animation = self._animation
        animation.elapsed += delta_time
        if animation.phase == "dice":
            if animation.elapsed >= animation.dice_duration:
                if len(animation.current_step.path) > 1:
                    animation.elapsed = 0.0
                    animation.phase = "move"
                else:
                    self._advance_animation_step()
        elif animation.phase == "move":
            if animation.elapsed >= animation.move_duration:
                self._advance_animation_step()
        if self._animation is None:
            self._rebuild_game_ui()
        self._update_replay(delta_time)

    def _advance_animation_step(self) -> None:
        if self._animation is None:
            return
        animation = self._animation
        animation.step_index += 1
        animation.elapsed = 0.0
        while animation.step_index < len(animation.steps):
            current_step = animation.current_step
            if current_step.dice_values is not None:
                animation.phase = "dice"
                return
            if len(current_step.path) > 1:
                animation.phase = "move"
                return
            animation.step_index += 1
        self._animation = None
        return

    def _update_replay(self, delta_time: float) -> None:
        if self._replay is None or not self._replay.is_playing:
            return
        self._replay.elapsed += delta_time
        frame_duration = 0.75 / max(0.25, self.animation_speed or 1.0)
        if self._replay.elapsed < frame_duration:
            return
        self._replay.elapsed = 0.0
        if self._replay.current_index >= len(self._replay.frames) - 1:
            self._replay.is_playing = False
            self._rebuild_replay_ui()
            return
        self._replay.current_index += 1
        frame = self._current_replay_frame()
        if frame is not None:
            self.controller.select_space(int(frame.get("selected_space_index", 0)))
        self._rebuild_replay_ui()

    def _display_frontend_state(self) -> FrontendStateView | None:
        if self.screen_mode == "replay":
            frame = self._current_replay_frame()
            if frame is None:
                return None
            return FrontendStateView.from_dict(frame["frontend_state"])
        if self._animation is not None:
            return self._animation.previous_state
        return self.controller.frontend_state

    def _current_replay_frame(self) -> dict[str, Any] | None:
        if self._replay is None or not self._replay.frames:
            return None
        index = max(0, min(self._replay.current_index, len(self._replay.frames) - 1))
        self._replay.current_index = index
        return self._replay.frames[index]

    def _step_replay(self, delta: int) -> None:
        if self._replay is None:
            return
        self._replay.is_playing = False
        self._replay.elapsed = 0.0
        self._replay.current_index = max(0, min(len(self._replay.frames) - 1, self._replay.current_index + delta))
        frame = self._current_replay_frame()
        if frame is not None:
            self.controller.select_space(int(frame.get("selected_space_index", 0)))
        self._rebuild_replay_ui()

    def _animated_token_render_data(self) -> tuple[set[str], dict[str, tuple[float, float]]]:
        if self._animation is None or self._animation.player_name is None:
            return set(), {}
        animation = self._animation
        current_step = animation.current_step
        if animation.phase != "move" or len(current_step.path) <= 1:
            return set(), {}
        hidden = {animation.player_name}
        duration = animation.move_duration
        if duration <= 0:
            return hidden, {}
        progress = min(1.0, animation.elapsed / duration)
        segments = len(current_step.path) - 1
        scaled = progress * segments
        segment_index = min(segments - 1, int(scaled))
        local_progress = scaled - segment_index
        start_index = current_step.path[segment_index]
        end_index = current_step.path[segment_index + 1]
        start_pos = self._space_center(start_index)
        end_pos = self._space_center(end_index)
        x = start_pos[0] + (end_pos[0] - start_pos[0]) * local_progress
        y = start_pos[1] + (end_pos[1] - start_pos[1]) * local_progress
        return hidden, {animation.player_name: (x, y)}

    def _draw_dice_overlay(self) -> None:
        if self._animation is None or self._animation.phase != "dice":
            return
        dice_values = self._current_dice_faces()
        if dice_values is None:
            return
        overlay = pygame.Surface((220, 100), pygame.SRCALPHA)
        overlay.fill((255, 249, 239, 220))
        pygame.draw.rect(overlay, pygame.Color("#4B4030"), overlay.get_rect(), width=2, border_radius=16)
        font = pygame.font.SysFont("georgia", 20, bold=True)
        label = font.render("Dice Roll", True, pygame.Color(theme.TEXT_PRIMARY))
        overlay.blit(label, (68, 10))
        for index, value in enumerate(dice_values):
            die_rect = pygame.Rect(28 + index * 96, 40, 64, 42)
            pygame.draw.rect(overlay, pygame.Color("#FFFFFF"), die_rect, border_radius=12)
            pygame.draw.rect(overlay, pygame.Color("#4B4030"), die_rect, width=2, border_radius=12)
            self._draw_die_pips(overlay, die_rect, value)
        self.screen.blit(overlay, overlay.get_rect(center=self.board_rect.center))

    def _draw_die_pips(self, surface: pygame.Surface, die_rect: pygame.Rect, value: int) -> None:
        pip_color = pygame.Color("#A44A3F")
        offsets = {
            1: ((0, 0),),
            2: ((-1, -1), (1, 1)),
            3: ((-1, -1), (0, 0), (1, 1)),
            4: ((-1, -1), (1, -1), (-1, 1), (1, 1)),
            5: ((-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)),
            6: ((-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)),
        }
        spacing_x = die_rect.width // 4
        spacing_y = die_rect.height // 4
        radius = max(3, min(die_rect.width, die_rect.height) // 10)
        for offset_x, offset_y in offsets[value]:
            center = (
                die_rect.centerx + offset_x * spacing_x,
                die_rect.centery + offset_y * spacing_y,
            )
            pygame.draw.circle(surface, pip_color, center, radius)

    def _current_dice_faces(self) -> tuple[int, int] | None:
        if self._animation is None or self._animation.current_step.dice_values is None:
            return None
        frame_index = int(self._animation.elapsed / 0.08)
        if self._animation.elapsed >= max(0.0, self._animation.dice_duration - 0.2):
            return self._animation.current_step.dice_values
        die_one, die_two = self._animation.current_step.dice_values
        random.seed(frame_index + die_one * 17 + die_two * 31)
        return random.randint(1, 6), random.randint(1, 6)

    def _space_center(self, index: int) -> tuple[float, float]:
        rect = self.board_renderer.layout.space_rect(index)
        return float(rect.centerx), float(rect.centery)

    def _extract_animation_segments(
        self,
        messages: tuple[str, ...],
        player_name: str,
        previous_state: FrontendStateView,
    ) -> list[AnimationSegment]:
        board_size = len(previous_state.board_spaces)
        current_position = next(
            player.position for player in previous_state.game_view.players if player.name == player_name
        )
        name_to_index = {space.name: space.index for space in previous_state.board_spaces}
        pending_dice: tuple[int, int] | None = None
        steps: list[AnimationSegment] = []
        move_pattern = re.compile(rf"^{re.escape(player_name)} moves from (\d+) to (\d+) \(.*\)\.$")

        for message in messages:
            dice_values = self._extract_dice_values_from_message(message, player_name)
            if dice_values is not None:
                if pending_dice is not None:
                    steps.append(AnimationSegment(dice_values=pending_dice, path=[current_position]))
                pending_dice = dice_values
                continue

            move_match = move_pattern.match(message)
            if move_match is not None:
                old_index = int(move_match.group(1))
                new_index = int(move_match.group(2))
                steps.append(AnimationSegment(dice_values=pending_dice, path=self._build_forward_path(old_index, new_index, board_size)))
                current_position = new_index
                pending_dice = None
                continue

            relocation_path = self._extract_relocation_path(message, player_name, current_position, name_to_index, board_size)
            if relocation_path is not None:
                path, current_position = relocation_path
                steps.append(AnimationSegment(dice_values=pending_dice, path=path))
                pending_dice = None

        if pending_dice is not None:
            steps.append(AnimationSegment(dice_values=pending_dice, path=[current_position]))

        return [step for step in steps if step.dice_values is not None or len(step.path) > 1]

    @staticmethod
    def _extract_dice_values_from_message(message: str, player_name: str) -> tuple[int, int] | None:
        patterns = (
            re.compile(rf"^{re.escape(player_name)} rolls (\d+) and (\d+) \(total \d+\)\.$"),
            re.compile(rf"^{re.escape(player_name)} attempts to roll doubles: (\d+) and (\d+)\.$"),
            re.compile(rf"^{re.escape(player_name)} rolls (\d+) and (\d+) for the utility card \(total \d+\)\.$"),
        )
        for pattern in patterns:
            match = pattern.match(message)
            if match is not None:
                return int(match.group(1)), int(match.group(2))
        return None

    @staticmethod
    def _build_forward_path(start_index: int, end_index: int, board_size: int) -> list[int]:
        path = [start_index]
        cursor = start_index
        while cursor != end_index:
            cursor = (cursor + 1) % board_size
            path.append(cursor)
        return path

    @staticmethod
    def _build_backward_path(start_index: int, end_index: int, board_size: int) -> list[int]:
        path = [start_index]
        cursor = start_index
        while cursor != end_index:
            cursor = (cursor - 1) % board_size
            path.append(cursor)
        return path

    @staticmethod
    def _direct_path(start_index: int, end_index: int) -> list[int]:
        if start_index == end_index:
            return [start_index]
        return [start_index, end_index]

    def _extract_relocation_path(
        self,
        message: str,
        player_name: str,
        current_position: int,
        name_to_index: dict[str, int],
        board_size: int,
    ) -> tuple[list[int], int] | None:
        relocation_rules: tuple[tuple[re.Pattern[str], Any], ...] = (
            (re.compile(rf"^{re.escape(player_name)} moves to Go and collects \$200\.$"), 0),
            (re.compile(rf"^{re.escape(player_name)} advances to Trafalgar Square\.$"), 24),
            (re.compile(rf"^{re.escape(player_name)} advances to Mayfair\.$"), 39),
            (re.compile(rf"^{re.escape(player_name)} advances to Pall Mall\.$"), 11),
            (re.compile(rf"^{re.escape(player_name)} advances to the nearest station: (.+)\.$"), "named_forward"),
            (re.compile(rf"^{re.escape(player_name)} advances to the nearest utility: (.+)\.$"), "named_forward"),
            (re.compile(rf"^{re.escape(player_name)} goes back three spaces\.$"), "back_three"),
            (re.compile(rf"^{re.escape(player_name)} takes a trip to King's Cross Station\.$"), 5),
            (re.compile(rf"^{re.escape(player_name)} goes directly to jail\.$"), "direct_jail"),
        )

        for pattern, rule in relocation_rules:
            match = pattern.match(message)
            if match is None:
                continue
            if rule == "named_forward":
                target_name = match.group(1)
                target_index = name_to_index.get(target_name)
                if target_index is None:
                    return None
                return self._build_forward_path(current_position, target_index, board_size), target_index
            if rule == "back_three":
                target_index = (current_position - 3) % board_size
                return self._build_backward_path(current_position, target_index, board_size), target_index
            if rule == "direct_jail":
                return self._direct_path(current_position, JAIL_INDEX), JAIL_INDEX
            target_index = int(rule)
            return self._build_forward_path(current_position, target_index, board_size), target_index
        return None


def run_gui_process(
    host: str | None = None,
    port: int | None = None,
    discovery_host: str | None = None,
    discovery_port: int | None = None,
    debug_mode: bool = False,
) -> None:
    log_path = configure_process_logging("frontend")
    logger.info("Frontend process starting. Log file: %s", log_path)
    try:
        MonopolyPygameApp(host, port, discovery_host=discovery_host, discovery_port=discovery_port, debug_mode=debug_mode).run()
    except Exception:
        logger.critical("Frontend process terminated unexpectedly.", exc_info=True)
        raise
    finally:
        logger.info("Frontend process stopped.")