from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock

from monopoly.api import LegalActionOption
from monopoly.constants import AI_ROLE, HUMAN_ROLE
from monopoly.dice import Dice
from monopoly.game import Game
from monopoly.gui.pygame_frontend.app import LOCAL_PLAY_MODE, MonopolyPygameApp
from monopoly.gui.pygame_frontend.controller import FrontendController


class PygameFrontendTests(unittest.TestCase):
    class _FakeEntry:
        def __init__(self, value: str) -> None:
            self._value = value

        def get_text(self) -> str:
            return self._value

    def _submenu_actions(self, count: int) -> list[LegalActionOption]:
        return [
            LegalActionOption(
                action_type="request_mortgage",
                actor_name="Alice",
                actor_role=HUMAN_ROLE,
                handler_name="request_property_action",
                description=f"Mortgage Lot {index + 1}",
                property_name=f"Lot {index + 1}",
            )
            for index in range(count)
        ]

    def test_property_submenu_paginates_with_navigation_controls(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app._property_submenu_mode = "mortgage"
        app._property_submenu_page = 0

        actions = self._submenu_actions(7)

        first_page = app._paginate_property_submenu_actions(actions, 6)
        self.assertEqual([action.description for action in first_page[:3]], ["Mortgage Lot 1", "Mortgage Lot 2", "Mortgage Lot 3"])
        self.assertEqual(("submenu_next", None, "More options (2/3)"), first_page[3])
        self.assertEqual(("submenu_back", None, "Back to moves"), first_page[4])

        app._property_submenu_page = 1
        middle_page = app._paginate_property_submenu_actions(actions, 6)
        self.assertEqual([action.description for action in middle_page[:3]], ["Mortgage Lot 4", "Mortgage Lot 5", "Mortgage Lot 6"])
        self.assertEqual(("submenu_prev", None, "Previous options (1/3)"), middle_page[3])
        self.assertEqual(("submenu_next", None, "More options (3/3)"), middle_page[4])
        self.assertEqual(("submenu_back", None, "Back to moves"), middle_page[5])

        app._property_submenu_page = 2
        last_page = app._paginate_property_submenu_actions(actions, 6)
        self.assertEqual([action.description for action in last_page[:1]], ["Mortgage Lot 7"])
        self.assertEqual(("submenu_prev", None, "Previous options (2/3)"), last_page[1])
        self.assertEqual(("submenu_back", None, "Back to moves"), last_page[2])

    def test_property_submenu_shows_all_actions_when_one_page_is_enough(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app._property_submenu_mode = "mortgage"
        app._property_submenu_page = 4

        actions = self._submenu_actions(2)

        visible_actions = app._paginate_property_submenu_actions(actions, 6)
        self.assertEqual([action.description for action in visible_actions[:2]], ["Mortgage Lot 1", "Mortgage Lot 2"])
        self.assertEqual(("submenu_back", None, "Back to moves"), visible_actions[2])
        self.assertEqual(0, app._property_submenu_page)

    def test_setup_player_uses_ai_settings_only_for_active_ai_rows(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.player_count = 3
        app.player_roles = [HUMAN_ROLE, AI_ROLE, HUMAN_ROLE, AI_ROLE, HUMAN_ROLE, HUMAN_ROLE]

        self.assertFalse(app._setup_player_uses_ai_settings(0))
        self.assertTrue(app._setup_player_uses_ai_settings(1))
        self.assertFalse(app._setup_player_uses_ai_settings(2))
        self.assertFalse(app._setup_player_uses_ai_settings(3))

    def test_discover_checkpoint_options_includes_all_scripted_variants(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)

        options = app._discover_checkpoint_options()

        self.assertIn("scripted:conservative_liquidity_manager", options)
        self.assertIn("scripted:auction_value_shark", options)
        self.assertIn("scripted:expansionist_builder", options)
        self.assertIn("scripted:monopoly_denial_disruptor", options)

    def test_dropdown_value_from_display_maps_scripted_labels(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)

        self.assertEqual("scripted:auction_value_shark", app._dropdown_value_from_display("Scripted: Auction Value Shark"))

    def test_confirm_property_action_uses_mortgage_label_in_mortgage_submenu(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app._property_submenu_mode = "mortgage"
        action = LegalActionOption(
            action_type="confirm_property_action",
            actor_name="Alice",
            actor_role=HUMAN_ROLE,
            handler_name="confirm_property_action",
            description="Confirm property change",
            property_name="Euston Road",
        )

        self.assertEqual("Mortgage Euston Road", app._action_button_label(action))

    def test_extract_animation_segments_sequences_double_rolls(self) -> None:
        game = Game(["Alice", "Bob"], dice=Dice(scripted_rolls=[]))
        game.players[0].position = 0
        previous_state = game.get_frontend_state()
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)

        segments = app._extract_animation_segments(
            (
                "Alice rolls 3 and 3 (total 6).",
                "Alice moves from 0 to 6 (The Angel, Islington).",
                "Alice rolled doubles and takes another turn.",
                "Alice rolls 2 and 2 (total 4).",
                "Alice moves from 6 to 10 (Jail).",
            ),
            "Alice",
            previous_state,
        )

        self.assertEqual([(3, 3), (2, 2)], [segment.dice_values for segment in segments])
        self.assertEqual([0, 1, 2, 3, 4, 5, 6], segments[0].path)
        self.assertEqual([6, 7, 8, 9, 10], segments[1].path)

    def test_extract_animation_segments_includes_card_relocation_path(self) -> None:
        game = Game(["Alice", "Bob"], dice=Dice(scripted_rolls=[]))
        game.players[0].position = 5
        previous_state = game.get_frontend_state()
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)

        segments = app._extract_animation_segments(
            (
                "Alice rolls 1 and 1 (total 2).",
                "Alice moves from 5 to 7 (Chance).",
                "Alice advances to Trafalgar Square.",
            ),
            "Alice",
            previous_state,
        )

        self.assertEqual((1, 1), segments[0].dice_values)
        self.assertEqual([5, 6, 7], segments[0].path)
        self.assertIsNone(segments[1].dice_values)
        self.assertEqual(7, segments[1].path[0])
        self.assertEqual(24, segments[1].path[-1])

    def test_pending_human_trade_opens_counter_prompt(self) -> None:
        game = Game(["Alice", "Bob"], player_roles={"Alice": HUMAN_ROLE, "Bob": HUMAN_ROLE})
        game.board.get_space(1).assign_owner(game.players[0])
        game.board.get_space(3).assign_owner(game.players[1])
        game.propose_trade_interactive(
            game.deserialize_trade_offer(
                {
                    "proposer_name": "Alice",
                    "receiver_name": "Bob",
                    "proposer_cash": 25,
                    "receiver_cash": 0,
                    "proposer_property_names": [],
                    "receiver_property_names": ["Whitechapel Road"],
                    "proposer_jail_cards": 0,
                    "receiver_jail_cards": 0,
                    "note": "Please trade",
                }
            )
        )

        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.controller = FrontendController(client=None)  # type: ignore[arg-type]
        app.controller.frontend_state = game.get_frontend_state()
        app._trade_prompt = None

        opened_actions: list[str] = []

        def capture():
            opened_actions.append("response_prompt")

        app._open_pending_trade_response_prompt = capture  # type: ignore[method-assign]

        app._maybe_open_pending_trade_prompt(app.controller.frontend_state)

        self.assertEqual(["response_prompt"], opened_actions)

    def test_pending_human_trade_still_opens_prompt_when_counter_is_unavailable(self) -> None:
        game = Game(["Alice", "Bob"], player_roles={"Alice": HUMAN_ROLE, "Bob": HUMAN_ROLE})
        game.board.get_space(1).assign_owner(game.players[0])
        game.board.get_space(3).assign_owner(game.players[1])
        game.execute_legal_action(
            next(action for action in game.get_turn_plan("Alice").legal_actions if action.action_type == "propose_trade"),
            trade_offer=game.deserialize_trade_offer(
                {
                    "proposer_name": "Alice",
                    "receiver_name": "Bob",
                    "proposer_cash": 25,
                    "receiver_cash": 0,
                    "proposer_property_names": [],
                    "receiver_property_names": ["Whitechapel Road"],
                    "proposer_jail_cards": 0,
                    "receiver_jail_cards": 0,
                    "note": "Please trade",
                }
            ),
        )
        game.execute_legal_action(
            next(action for action in game.get_turn_plan("Bob").legal_actions if action.action_type == "counter_trade"),
            trade_offer=game.deserialize_trade_offer(
                {
                    "proposer_name": "Bob",
                    "receiver_name": "Alice",
                    "proposer_cash": 0,
                    "receiver_cash": 25,
                    "proposer_property_names": ["Whitechapel Road"],
                    "receiver_property_names": ["Old Kent Road"],
                    "proposer_jail_cards": 0,
                    "receiver_jail_cards": 0,
                    "note": "One counter only",
                }
            ),
        )

        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.controller = FrontendController(client=None)  # type: ignore[arg-type]
        app.controller.frontend_state = game.get_frontend_state()
        app._trade_prompt = None

        opened = {"called": False}

        def capture():
            opened["called"] = True

        app._open_pending_trade_response_prompt = capture  # type: ignore[method-assign]

        app._maybe_open_pending_trade_prompt(app.controller.frontend_state)

        self.assertTrue(opened["called"])

    def test_pending_trade_summary_is_shown_in_dedicated_trade_panel(self) -> None:
        game = Game(["Alice", "Bob"], player_roles={"Alice": HUMAN_ROLE, "Bob": HUMAN_ROLE})
        game.board.get_space(3).assign_owner(game.players[1])
        game.propose_trade_interactive(
            game.deserialize_trade_offer(
                {
                    "proposer_name": "Alice",
                    "receiver_name": "Bob",
                    "proposer_cash": 25,
                    "receiver_cash": 0,
                    "proposer_property_names": [],
                    "receiver_property_names": ["Whitechapel Road"],
                    "proposer_jail_cards": 0,
                    "receiver_jail_cards": 0,
                    "note": "Need liquidity",
                }
            )
        )

        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.controller = FrontendController(client=None)  # type: ignore[arg-type]
        app.controller.frontend_state = game.get_frontend_state()
        app.controller.message_history = ["Alice offers a trade to Bob."]

        context_html = app._turn_context_html(app.controller.frontend_state)
        trade_html = app._trade_panel_html(app.controller.frontend_state)
        log_html = app._log_html(180)

        self.assertNotIn("Trade on table:", context_html)
        self.assertIn("Trade on table:", trade_html)
        self.assertIn("Alice gives:", trade_html)
        self.assertIn("Bob gives:", trade_html)
        self.assertIn("Need liquidity", trade_html)
        self.assertNotIn("Trade pending:", log_html)

    def test_setup_primary_button_label_changes_with_online_mode(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.online_reconnect_token = ""

        app.setup_game_mode = LOCAL_PLAY_MODE
        self.assertEqual("Start Match", app._setup_primary_button_label())
        self.assertEqual("Local Play", app._display_setup_mode(LOCAL_PLAY_MODE))

        app.setup_game_mode = "host_online"
        self.assertEqual("Create Lobby", app._setup_primary_button_label())

        app.setup_game_mode = "join_online"
        self.assertEqual("Connect", app._setup_primary_button_label())

        app.online_reconnect_token = "token-123"
        self.assertEqual("Reconnect", app._setup_primary_button_label())

    def test_start_game_in_host_online_mode_creates_lobby_screen(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.setup_game_mode = "host_online"
        app.player_names = ["Alice", "Player 2", "Player 3", "Player 4", "Player 5", "Player 6"]
        app.player_count = 3
        app.cash_entry = self._FakeEntry("1800")
        app.controller = MagicMock()
        app._capture_setup_inputs = lambda: None  # type: ignore[method-assign]
        app._ensure_local_backend_connection = MagicMock()
        app._build_online_lobby_screen = MagicMock()
        app._build_setup_screen = MagicMock()
        app._pending_ai_step = None

        app._start_game()

        app.controller.create_online_lobby.assert_called_once_with("Alice", 3, 1800)
        self.assertEqual("online_lobby", app.screen_mode)
        app._build_online_lobby_screen.assert_called_once_with()

    def test_local_start_game_clears_stale_online_controller_state(self) -> None:
        controller = FrontendController(client=MagicMock())
        controller.client.create_game.return_value = {
            "frontend_state": Game(["Alice", "Bob"]).get_serialized_frontend_state(),
        }
        controller.online_session = type("Session", (), {"state": "in_game"})()
        controller.session_token = "session-token"
        controller.reconnect_token = "reconnect-token"
        controller.online_player_name = "Alice"
        controller.online_seat_index = 0
        controller.is_online_host = True

        controller.start_game(["Alice", "Bob"], 1500, player_roles=[HUMAN_ROLE, HUMAN_ROLE])

        self.assertIsNone(controller.online_session)
        self.assertIsNone(controller.session_token)
        self.assertIsNone(controller.reconnect_token)
        self.assertIsNone(controller.online_player_name)
        self.assertIsNone(controller.online_seat_index)
        self.assertFalse(controller.is_online_host)

    def test_host_online_setup_separates_discovery_hint_and_fields(self) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

        app = MonopolyPygameApp(None, None)
        try:
            app.setup_game_mode = "host_online"
            app._build_setup_screen()

            self.assertTrue(app.host_discovery_hint_label.visible)
            self.assertTrue(app.online_discovery_host_label.visible)
            self.assertGreater(app.online_discovery_host_label.relative_rect.top, app.host_discovery_hint_label.relative_rect.bottom)
            self.assertGreater(app.online_discovery_port_label.relative_rect.top, app.host_discovery_hint_label.relative_rect.bottom)
        finally:
            app.controller.shutdown()
            app._stop_managed_backend()
            app.running = False
            import pygame

            pygame.quit()

    def test_local_setup_only_creates_ai_controls_for_ai_rows(self) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

        app = MonopolyPygameApp(None, None)
        try:
            app.setup_game_mode = LOCAL_PLAY_MODE
            app.player_count = 4
            app.player_roles = [HUMAN_ROLE, AI_ROLE, AI_ROLE, HUMAN_ROLE, HUMAN_ROLE, HUMAN_ROLE]

            app._build_setup_screen()

            self.assertIsNone(app._setup_fields[0].checkpoint_dropdown)
            self.assertIsNone(app._setup_fields[0].cooldown_entry)
            self.assertIsNotNone(app._setup_fields[1].checkpoint_dropdown)
            self.assertIsNotNone(app._setup_fields[1].cooldown_entry)
            self.assertIsNotNone(app._setup_fields[2].checkpoint_dropdown)
            self.assertIsNotNone(app._setup_fields[2].cooldown_entry)
            self.assertIsNone(app._setup_fields[3].checkpoint_dropdown)
            self.assertIsNone(app._setup_fields[3].cooldown_entry)
        finally:
            app.controller.shutdown()
            app._stop_managed_backend()
            app.running = False
            import pygame

            pygame.quit()

    def test_host_online_lobby_uses_seat_state_dropdowns_and_no_refresh_button(self) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

        app = MonopolyPygameApp(None, None)
        try:
            seat_host = type("Seat", (), {"seat_index": 0, "status": "host", "player_name": "Player 1", "player_role": HUMAN_ROLE, "is_host": True})()
            seat_open = type("Seat", (), {"seat_index": 1, "status": "open", "player_name": None, "player_role": HUMAN_ROLE, "is_host": False})()
            seat_ai = type("Seat", (), {"seat_index": 2, "status": "ai", "player_name": "AI Player 3", "player_role": AI_ROLE, "is_host": False})()
            app.controller = MagicMock()
            app.controller.online_session = type(
                "Session",
                (),
                {"session_code": "ABC123", "state": "lobby", "starting_cash": 1500, "seats": [seat_host, seat_open, seat_ai], "paused_reason": None, "paused_seat_index": None},
            )()
            app.controller.client = type("Client", (), {"host": "127.0.0.1", "port": 4567})()

            app._build_online_lobby_screen()

            self.assertEqual(2, len(app._host_lobby_slot_dropdowns))
            self.assertEqual(1, len(app._host_lobby_ai_checkpoint_dropdowns))
            self.assertIn(2, app._host_lobby_ai_speed_entries)
            self.assertEqual(1, len(app._host_lobby_ai_apply_buttons))
            self.assertFalse(any(command == "refresh_online_session" for command, _payload in app._ui_commands.values()))
        finally:
            app.controller.shutdown()
            app._stop_managed_backend()
            app.running = False
            import pygame

            pygame.quit()

    def test_update_online_session_is_noop_without_polling(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.screen_mode = "online_lobby"
        app._refresh_online_session_screen = MagicMock()

        app._update_online_session(5.0)

        app._refresh_online_session_screen.assert_not_called()

    def test_set_host_lobby_slot_mode_assigns_ai_with_saved_config(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.controller = MagicMock()
        app.controller.online_session = type(
            "Session",
            (),
            {"seats": [type("Seat", (), {"seat_index": 1, "status": "open"})()]},
        )()
        app.ai_checkpoint_paths = ["default.pt", "custom.pt", "default.pt", "default.pt", "default.pt", "default.pt"]
        app.ai_cooldown_texts = ["2.0", "0.5", "2.0", "2.0", "2.0", "2.0"]
        app._build_online_lobby_screen = MagicMock()
        app._build_setup_screen = MagicMock()

        app._set_host_lobby_slot_mode(1, "ai")

        app.controller.assign_ai_to_online_slot.assert_called_once_with(1, player_name="AI Player 2", checkpoint_path="custom.pt", action_cooldown_seconds=0.5)

    def test_apply_host_lobby_ai_settings_uses_inline_values(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.controller = MagicMock()
        app.ai_checkpoint_paths = ["default.pt", "custom.pt", "default.pt", "default.pt", "default.pt", "default.pt"]
        app.ai_cooldown_texts = ["2.0", "2.0", "2.0", "2.0", "2.0", "2.0"]
        app._host_lobby_ai_speed_entries = {1: self._FakeEntry("0.75")}  # type: ignore[assignment]
        app._build_online_lobby_screen = MagicMock()

        app._apply_host_lobby_ai_settings(1)

        app.controller.assign_ai_to_online_slot.assert_called_once_with(1, player_name="AI Player 2", checkpoint_path="custom.pt", action_cooldown_seconds=0.75)

    def test_ensure_local_backend_connection_recreates_closed_client(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.debug_mode = False
        app._managed_backend_process = MagicMock()
        app._managed_backend_process.is_alive.return_value = True
        app._stop_managed_backend = MagicMock()
        app.controller = MagicMock()
        app.controller.client = MagicMock()
        app.controller.client.owns_server = True
        app.controller.client.is_closed = True
        app.controller.connect_to_backend = MagicMock()

        fake_process = MagicMock()
        with unittest.mock.patch("monopoly.gui.pygame_frontend.app.find_free_port", return_value=4567), unittest.mock.patch(
            "monopoly.gui.pygame_frontend.app.mp.Process", return_value=fake_process
        ):
            app._ensure_local_backend_connection()

        app._stop_managed_backend.assert_called_once_with()
        self.assertIsNone(app.controller.client)
        fake_process.start.assert_called_once_with()
        app.controller.connect_to_backend.assert_called_once_with("127.0.0.1", 4567, owns_server=True)

    def test_start_game_in_join_online_mode_connects_to_remote_host(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.setup_game_mode = "join_online"
        app.online_host_entry = self._FakeEntry("203.0.113.10")
        app.online_port_entry = self._FakeEntry("4567")
        app.online_session_code_entry = self._FakeEntry("ABC123")
        app.online_player_name_entry = self._FakeEntry("Bob")
        app.online_reconnect_token_entry = self._FakeEntry("")
        app.online_discovery_host = "127.0.0.1"
        app.online_discovery_port = "47321"
        app.online_join_player_name = "Guest"
        app.controller = MagicMock()
        app._capture_setup_inputs = lambda: None  # type: ignore[method-assign]
        app._stop_managed_backend = MagicMock()
        app._build_online_waiting_screen = MagicMock()
        app._build_setup_screen = MagicMock()
        app._pending_ai_step = None

        app._start_game()

        app.controller.connect_to_backend.assert_called_once_with("203.0.113.10", 4567, owns_server=False)
        app.controller.refresh_online_session.assert_called_once_with(session_code="ABC123")
        self.assertEqual("online_waiting", app.screen_mode)
        app._build_online_waiting_screen.assert_called_once_with()

    def test_start_game_in_join_online_mode_can_resolve_discovery_endpoint(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.setup_game_mode = "join_online"
        app.online_host_entry = self._FakeEntry("")
        app.online_port_entry = self._FakeEntry("")
        app.online_session_code_entry = self._FakeEntry("ABC123")
        app.online_player_name_entry = self._FakeEntry("Bob")
        app.online_reconnect_token_entry = self._FakeEntry("")
        app.online_discovery_host_entry = self._FakeEntry("198.51.100.20")
        app.online_discovery_port_entry = self._FakeEntry("47321")
        app.online_join_player_name = "Guest"
        app.controller = MagicMock()
        app._capture_setup_inputs = lambda: None  # type: ignore[method-assign]
        app._stop_managed_backend = MagicMock()
        app._build_online_waiting_screen = MagicMock()
        app._build_setup_screen = MagicMock()
        app._resolve_lobby_endpoint = MagicMock(return_value=("203.0.113.10", 4567))
        app._pending_ai_step = None

        app._start_game()

        app._resolve_lobby_endpoint.assert_called_once_with("ABC123", "198.51.100.20", 47321)
        app.controller.connect_to_backend.assert_called_once_with("203.0.113.10", 4567, owns_server=False)

    def test_debug_mode_uses_tabbed_editor_workspace(self) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

        app = MonopolyPygameApp(None, None, debug_mode=True)
        try:
            game = Game(["Player 1", "Player 2"], player_roles=[HUMAN_ROLE, HUMAN_ROLE])
            controller = MagicMock()
            controller.frontend_state = game.get_frontend_state()
            controller.get_debug_state.return_value = game.serialize_full_state()
            controller.shutdown = MagicMock()
            app.controller = controller
            app.screen_mode = "game"

            app._rebuild_game_ui()

            self.assertEqual("player", app._debug_active_tab)
            self.assertIsNotNone(app._debug_panel)
            self.assertIsNotNone(app._debug_panel.player_select)
            self.assertIsNone(app._debug_panel.runtime_current_player_dropdown)

            app._handle_ui_command("debug_tab", "runtime")

            self.assertEqual("runtime", app._debug_active_tab)
            self.assertIsNotNone(app._debug_panel)
            self.assertIsNotNone(app._debug_panel.runtime_current_player_dropdown)
            self.assertIsNone(app._debug_panel.player_select)
            self.assertGreaterEqual(app.sidebar_width, 900)
        finally:
            app.controller.shutdown()
            app._stop_managed_backend()
            app.running = False
            import pygame

            pygame.quit()

    def test_start_game_in_join_online_mode_uses_reconnect_token(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.setup_game_mode = "join_online"
        app.online_host_entry = self._FakeEntry("203.0.113.10")
        app.online_port_entry = self._FakeEntry("4567")
        app.online_session_code_entry = self._FakeEntry("ABC123")
        app.online_player_name_entry = self._FakeEntry("Bob")
        app.online_reconnect_token_entry = self._FakeEntry("resume-token")
        app.online_discovery_host = "127.0.0.1"
        app.online_discovery_port = "47321"
        app.online_join_player_name = "Guest"
        app.controller = MagicMock()
        app.controller.online_session = type("Session", (), {"state": "paused"})()
        app._capture_setup_inputs = lambda: None  # type: ignore[method-assign]
        app._stop_managed_backend = MagicMock()
        app._build_online_waiting_screen = MagicMock()
        app._build_setup_screen = MagicMock()
        app._pending_ai_step = None

        app._start_game()

        app.controller.reconnect_online_slot.assert_called_once_with("resume-token", session_code="ABC123")

    def test_waiting_screen_transitions_into_game_when_host_starts(self) -> None:
        game = Game(["Alice", "Bob"], player_roles={"Alice": HUMAN_ROLE, "Bob": HUMAN_ROLE})
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.controller = FrontendController(client=None)  # type: ignore[arg-type]
        app.controller.online_session = type("Session", (), {"state": "in_game"})()
        app.controller.frontend_state = game.get_frontend_state()
        app._clear_elements = MagicMock()
        app._build_game_screen = MagicMock()

        app._build_online_waiting_screen()

        self.assertEqual("game", app.screen_mode)
        app._build_game_screen.assert_called_once_with()

    def test_update_ai_turns_is_disabled_for_remote_client(self) -> None:
        game = Game(["Alice", "Bob"], player_roles={"Alice": AI_ROLE, "Bob": HUMAN_ROLE})
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.screen_mode = "game"
        app.controller = FrontendController(client=MagicMock())
        app.controller.client.owns_server = False
        app.controller.frontend_state = game.get_frontend_state()
        app._animation = None
        app._prompt = None
        app._trade_prompt = None
        app._pending_ai_step = object()

        app._update_ai_turns(0.25)

        self.assertIsNone(app._pending_ai_step)

    def test_process_online_events_rebuilds_live_online_game(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.screen_mode = "game"
        app.controller = MagicMock()
        app.controller.online_session = type("Session", (), {"state": "in_game"})()
        app.controller.frontend_state = object()
        app.controller.drain_online_events.return_value = True
        app._build_game_screen = MagicMock()
        app._build_online_lobby_screen = MagicMock()
        app._build_online_waiting_screen = MagicMock()

        app._process_online_events()

        app._build_game_screen.assert_called_once_with()

    def test_process_online_events_rebuilds_waiting_room(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.screen_mode = "online_waiting"
        app.controller = MagicMock()
        app.controller.online_session = type("Session", (), {"state": "lobby"})()
        app.controller.drain_online_events.return_value = True
        app._build_online_waiting_screen = MagicMock()

        app._process_online_events()

        app._build_online_waiting_screen.assert_called_once_with()

    def test_refresh_online_session_moves_host_from_game_to_paused_lobby(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app.screen_mode = "game"
        app.controller = MagicMock()
        app.controller.frontend_state = object()
        app.controller.is_online_host = True
        app.controller.online_session = type("Session", (), {"session_code": "ABC123", "state": "paused"})()
        app.controller.refresh_online_session = MagicMock(side_effect=lambda session_code=None: None)
        app._build_online_lobby_screen = MagicMock()
        app._build_online_waiting_screen = MagicMock()
        app._build_game_screen = MagicMock()

        app._refresh_online_session_screen()

        app._build_online_lobby_screen.assert_called_once_with()
        app._build_online_waiting_screen.assert_not_called()
        app._build_game_screen.assert_not_called()

    def test_controller_drains_online_session_events(self) -> None:
        controller = FrontendController(client=MagicMock())
        controller.client.drain_events.return_value = [
            {
                "event": "online_session_updated",
                "payload": {
                    "online_session": {
                        "session_code": "ABC123",
                        "state": "lobby",
                        "host_player_name": "Alice",
                        "seat_count": 2,
                        "starting_cash": 1500,
                        "seats": [],
                        "paused_reason": None,
                        "paused_seat_index": None,
                    },
                },
            }
        ]

        changed = controller.drain_online_events()

        self.assertTrue(changed)
        self.assertEqual("ABC123", controller.online_session.session_code)

    def test_controller_online_game_requests_include_session_token(self) -> None:
        controller = FrontendController(client=MagicMock())
        controller.client.execute_action.return_value = {
            "interaction": {
                "messages": ["Alice rolls."],
                "game_view": Game(["Alice", "Bob"]).get_game_view().to_dict(),
                "turn_plan": Game(["Alice", "Bob"]).get_turn_plan().to_dict(),
            },
            "frontend_state": Game(["Alice", "Bob"]).get_serialized_frontend_state(),
        }
        controller.client.step_ai.return_value = {
            "actor_name": "Bot",
            "interaction": {
                "messages": ["Bot acts."],
                "game_view": Game(["Bot", "Bob"], player_roles={"Bot": AI_ROLE, "Bob": HUMAN_ROLE}).get_game_view().to_dict(),
                "turn_plan": Game(["Bot", "Bob"], player_roles={"Bot": AI_ROLE, "Bob": HUMAN_ROLE}).get_turn_plan().to_dict(),
            },
            "frontend_state": Game(["Bot", "Bob"], player_roles={"Bot": AI_ROLE, "Bob": HUMAN_ROLE}).get_serialized_frontend_state(),
        }
        controller.session_token = "session-token"
        action = Game(["Alice", "Bob"]).get_turn_plan().legal_actions[0]

        controller.execute_action(action)
        controller.step_ai()

        controller.client.execute_action.assert_called_once()
        self.assertEqual("session-token", controller.client.execute_action.call_args.kwargs["session_token"])
        controller.client.step_ai.assert_called_once_with(session_token="session-token")

    def test_load_prompt_ensures_local_backend_connection(self) -> None:
        app = MonopolyPygameApp.__new__(MonopolyPygameApp)
        app._prompt = type("Prompt", (), {"kind": "load", "entry": self._FakeEntry("savegame.json"), "action": None})()
        app.controller = MagicMock()
        app._ensure_local_backend_connection = MagicMock()
        app._close_prompt = MagicMock()
        app._rebuild_game_ui = MagicMock()
        app._replay = None
        app._animation = None
        app._property_submenu_mode = None
        app._property_submenu_page = 0
        app._pending_ai_step = None

        app._submit_prompt()

        app._ensure_local_backend_connection.assert_called_once_with()
        app.controller.load_game.assert_called_once_with("savegame.json")


if __name__ == "__main__":
    unittest.main()