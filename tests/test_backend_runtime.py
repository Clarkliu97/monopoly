from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from monopoly.agent import MonopolyActionSpace, ObservationEncoder, ParallelSelfPlayTrainer, PolicyConfig, TorchPolicyModel, TrainingConfig
from monopoly.gui.backend_process import BackendRuntime


class BackendRuntimeTests(unittest.TestCase):
    def _write_checkpoint(self, directory: str) -> str:
        encoder = ObservationEncoder()
        action_space = MonopolyActionSpace()
        trainer = ParallelSelfPlayTrainer(
            policy_model=TorchPolicyModel(encoder.observation_size, action_space.action_count, seed=5, hidden_sizes=(64, 64)),
            training_config=TrainingConfig(worker_count=1, episodes_per_worker=1, max_steps_per_episode=4, players_per_game=2),
            policy_config=PolicyConfig(seed=5, hidden_sizes=(64, 64), ppo_epochs=1, minibatch_size=8),
        )
        checkpoint_path = os.path.join(directory, "runtime.pt")
        trainer.save_checkpoint(checkpoint_path)
        return checkpoint_path

    def test_create_game_returns_serialized_frontend_state(self) -> None:
        runtime = BackendRuntime()

        response = runtime.handle_command(
            {
                "command": "create_game",
                "setup": {
                    "player_names": ["Alice", "Bob"],
                    "starting_cash": 1800,
                },
            }
        )

        self.assertTrue(response["ok"])
        self.assertEqual(1800, response["payload"]["frontend_state"]["game_view"]["starting_cash"])
        self.assertEqual("Alice", response["payload"]["frontend_state"]["active_turn_plan"]["player_name"])

    def test_create_game_accepts_ai_roles_and_loads_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = self._write_checkpoint(temp_dir)
            runtime = BackendRuntime(default_ai_checkpoint_path=checkpoint_path)

            response = runtime.handle_command(
                {
                    "command": "create_game",
                    "setup": {
                        "player_names": ["Alice", "Bob"],
                        "player_roles": ["human", "ai"],
                        "starting_cash": 1800,
                    },
                }
            )

        self.assertTrue(response["ok"])
        self.assertEqual("human", response["payload"]["frontend_state"]["game_view"]["players"][0]["role"])
        self.assertEqual("ai", response["payload"]["frontend_state"]["game_view"]["players"][1]["role"])
        self.assertEqual(Path(checkpoint_path).resolve(), Path(runtime.ai_checkpoint_path).resolve())

    def test_create_game_accepts_per_player_ai_setup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = self._write_checkpoint(temp_dir)
            runtime = BackendRuntime()

            response = runtime.handle_command(
                {
                    "command": "create_game",
                    "setup": {
                        "player_names": ["Alice", "Bob"],
                        "player_roles": ["human", "ai"],
                        "starting_cash": 1800,
                        "ai_player_setups": [
                            {
                                "player_name": "Bob",
                                "checkpoint_path": checkpoint_path,
                                "action_cooldown_seconds": 1.25,
                            }
                        ],
                    },
                }
            )

        self.assertTrue(response["ok"])
        self.assertEqual(Path(checkpoint_path).resolve(), Path(runtime.ai_checkpoint_paths_by_player["Bob"]).resolve())
        self.assertEqual(1.25, runtime.ai_cooldowns_by_player["Bob"])

    def test_create_game_accepts_scripted_ai_setup(self) -> None:
        runtime = BackendRuntime()

        response = runtime.handle_command(
            {
                "command": "create_game",
                "setup": {
                    "player_names": ["Alice", "Bob"],
                    "player_roles": ["ai", "human"],
                    "starting_cash": 1800,
                    "ai_player_setups": [
                        {
                            "player_name": "Alice",
                            "checkpoint_path": "scripted:auction_value_shark",
                            "action_cooldown_seconds": 0.25,
                        }
                    ],
                },
            }
        )

        self.assertTrue(response["ok"])
        self.assertEqual("scripted:auction_value_shark", runtime.ai_checkpoint_paths_by_player["Alice"])
        self.assertEqual(0.25, runtime.ai_cooldowns_by_player["Alice"])

        step_response = runtime.handle_command({"command": "step_ai"})

        self.assertTrue(step_response["ok"])
        self.assertEqual("Alice", step_response["payload"]["actor_name"])
        self.assertEqual(0.25, step_response["payload"]["cooldown_seconds"])

    def test_step_ai_executes_one_ai_action_and_returns_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = self._write_checkpoint(temp_dir)
            runtime = BackendRuntime()
            runtime.handle_command(
                {
                    "command": "create_game",
                    "setup": {
                        "player_names": ["Alice", "Bob"],
                        "player_roles": ["ai", "human"],
                        "starting_cash": 1800,
                        "ai_player_setups": [
                            {
                                "player_name": "Alice",
                                "checkpoint_path": checkpoint_path,
                                "action_cooldown_seconds": 0.5,
                            }
                        ],
                    },
                }
            )

            response = runtime.handle_command({"command": "step_ai"})

        self.assertTrue(response["ok"])
        self.assertEqual("Alice", response["payload"]["actor_name"])
        self.assertEqual(0.5, response["payload"]["cooldown_seconds"])
        self.assertIn(
            response["payload"]["frontend_state"]["active_turn_plan"]["turn_phase"],
            {"in_turn", "post_roll"},
        )

    def test_step_ai_uses_safe_fallback_after_turn_budget_is_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = self._write_checkpoint(temp_dir)
            runtime = BackendRuntime(max_ai_actions_per_turn=0)
            runtime.handle_command(
                {
                    "command": "create_game",
                    "setup": {
                        "player_names": ["Alice", "Bob"],
                        "player_roles": ["ai", "human"],
                        "starting_cash": 1800,
                        "ai_player_setups": [
                            {
                                "player_name": "Alice",
                                "checkpoint_path": checkpoint_path,
                                "action_cooldown_seconds": 0.5,
                            }
                        ],
                    },
                }
            )
            runtime.game.current_turn_phase = "post_roll"

            response = runtime.handle_command({"command": "step_ai"})

        self.assertTrue(response["ok"])
        self.assertEqual("Bob", response["payload"]["frontend_state"]["active_turn_plan"]["player_name"])

    def test_execute_action_runs_against_created_game(self) -> None:
        runtime = BackendRuntime()
        runtime.handle_command(
            {
                "command": "create_game",
                "setup": {
                    "player_names": ["Alice", "Bob"],
                    "starting_cash": 1500,
                },
            }
        )

        response = runtime.handle_command(
            {
                "command": "execute_action",
                "action": {
                    "action_type": "start_turn",
                    "actor_name": "Alice",
                    "actor_role": "human",
                    "handler_name": "start_turn_interactive",
                    "description": "Start Alice's interactive turn.",
                    "property_name": None,
                    "target_player_name": None,
                    "fixed_choice": None,
                    "min_bid": None,
                    "max_bid": None,
                },
            }
        )

        self.assertTrue(response["ok"])
        self.assertIn(
            response["payload"]["frontend_state"]["active_turn_plan"]["turn_phase"],
            {"in_turn", "post_roll"},
        )
        self.assertEqual(
            response["payload"]["interaction"]["game_view"]["current_turn_phase"],
            response["payload"]["frontend_state"]["active_turn_plan"]["turn_phase"],
        )

    def test_get_state_requires_created_game(self) -> None:
        runtime = BackendRuntime()

        response = runtime.handle_command({"command": "get_state"})

        self.assertFalse(response["ok"])
        self.assertIn("No game", response["error"])

    def test_create_online_lobby_returns_host_session_and_open_slots(self) -> None:
        runtime = BackendRuntime()

        response = runtime.handle_command(
            {
                "command": "create_online_lobby",
                "host_player_name": "Alice",
                "seat_count": 4,
                "starting_cash": 1800,
            }
        )

        self.assertTrue(response["ok"])
        self.assertTrue(response["payload"]["is_host"])
        self.assertEqual("Alice", response["payload"]["player_name"])
        session = response["payload"]["online_session"]
        self.assertEqual("lobby", session["state"])
        self.assertEqual(4, session["seat_count"])
        self.assertEqual("host", session["seats"][0]["status"])
        self.assertEqual("open", session["seats"][1]["status"])

    def test_online_lobby_rejects_closed_and_ai_slots_from_joining(self) -> None:
        runtime = BackendRuntime()
        create = runtime.handle_command(
            {
                "command": "create_online_lobby",
                "host_player_name": "Alice",
                "seat_count": 4,
                "starting_cash": 1800,
            }
        )
        host_token = create["payload"]["session_token"]
        session_code = create["payload"]["online_session"]["session_code"]

        close_response = runtime.handle_command(
            {
                "command": "close_online_slot",
                "session_token": host_token,
                "seat_index": 1,
            }
        )
        assign_ai_response = runtime.handle_command(
            {
                "command": "assign_ai_to_online_slot",
                "session_token": host_token,
                "seat_index": 2,
                "player_name": "Bot Seat",
            }
        )
        closed_join = runtime.handle_command(
            {
                "command": "claim_online_slot",
                "session_code": session_code,
                "seat_index": 1,
                "player_name": "Bob",
            }
        )
        ai_join = runtime.handle_command(
            {
                "command": "claim_online_slot",
                "session_code": session_code,
                "seat_index": 2,
                "player_name": "Carol",
            }
        )
        valid_join = runtime.handle_command(
            {
                "command": "claim_online_slot",
                "session_code": session_code,
                "seat_index": 3,
                "player_name": "Dave",
            }
        )

        self.assertTrue(close_response["ok"])
        self.assertTrue(assign_ai_response["ok"])
        self.assertFalse(closed_join["ok"])
        self.assertFalse(ai_join["ok"])
        self.assertTrue(valid_join["ok"])
        self.assertEqual("Dave", valid_join["payload"]["player_name"])

    def test_only_host_can_edit_slots_or_start_online_game(self) -> None:
        runtime = BackendRuntime()
        create = runtime.handle_command(
            {
                "command": "create_online_lobby",
                "host_player_name": "Alice",
                "seat_count": 3,
                "starting_cash": 1800,
            }
        )
        host_token = create["payload"]["session_token"]
        session_code = create["payload"]["online_session"]["session_code"]
        join = runtime.handle_command(
            {
                "command": "claim_online_slot",
                "session_code": session_code,
                "seat_index": 1,
                "player_name": "Bob",
            }
        )
        guest_token = join["payload"]["session_token"]

        guest_close = runtime.handle_command(
            {
                "command": "close_online_slot",
                "session_token": guest_token,
                "seat_index": 2,
            }
        )
        guest_start = runtime.handle_command(
            {
                "command": "start_online_game",
                "session_token": guest_token,
            }
        )
        host_start = runtime.handle_command(
            {
                "command": "start_online_game",
                "session_token": host_token,
            }
        )

        self.assertFalse(guest_close["ok"])
        self.assertFalse(guest_start["ok"])
        self.assertTrue(host_start["ok"])
        self.assertIn("frontend_state", host_start["payload"])
        self.assertEqual("in_game", host_start["payload"]["online_session"]["state"])

    def test_online_game_enforces_seat_owned_actions(self) -> None:
        runtime = BackendRuntime()
        create = runtime.handle_command(
            {
                "command": "create_online_lobby",
                "host_player_name": "Alice",
                "seat_count": 2,
                "starting_cash": 1500,
            }
        )
        host_token = create["payload"]["session_token"]
        session_code = create["payload"]["online_session"]["session_code"]
        guest = runtime.handle_command(
            {
                "command": "claim_online_slot",
                "session_code": session_code,
                "seat_index": 1,
                "player_name": "Bob",
            }
        )
        guest_token = guest["payload"]["session_token"]
        runtime.handle_command(
            {
                "command": "start_online_game",
                "session_token": host_token,
            }
        )

        illegal_guest_action = runtime.handle_command(
            {
                "command": "execute_action",
                "session_token": guest_token,
                "action": {
                    "action_type": "start_turn",
                    "actor_name": "Alice",
                    "actor_role": "human",
                    "handler_name": "start_turn_interactive",
                    "description": "Roll the dice for Alice.",
                    "property_name": None,
                    "target_player_name": None,
                    "fixed_choice": None,
                    "min_bid": None,
                    "max_bid": None,
                },
            }
        )
        legal_host_action = runtime.handle_command(
            {
                "command": "execute_action",
                "session_token": host_token,
                "action": {
                    "action_type": "start_turn",
                    "actor_name": "Alice",
                    "actor_role": "human",
                    "handler_name": "start_turn_interactive",
                    "description": "Roll the dice for Alice.",
                    "property_name": None,
                    "target_player_name": None,
                    "fixed_choice": None,
                    "min_bid": None,
                    "max_bid": None,
                },
            }
        )

        self.assertFalse(illegal_guest_action["ok"])
        self.assertTrue(legal_host_action["ok"])

    def test_disconnect_pauses_online_game_until_player_returns(self) -> None:
        runtime = BackendRuntime()
        create = runtime.handle_command(
            {
                "command": "create_online_lobby",
                "host_player_name": "Alice",
                "seat_count": 2,
                "starting_cash": 1500,
            }
        )
        host_token = create["payload"]["session_token"]
        session_code = create["payload"]["online_session"]["session_code"]
        guest = runtime.handle_command(
            {
                "command": "claim_online_slot",
                "session_code": session_code,
                "seat_index": 1,
                "player_name": "Bob",
            }
        )
        guest_token = guest["payload"]["session_token"]
        guest_reconnect_token = guest["payload"]["reconnect_token"]
        runtime.handle_command({"command": "start_online_game", "session_token": host_token})

        disconnect = runtime.handle_command(
            {
                "command": "disconnect_online_slot",
                "session_code": session_code,
                "session_token": guest_token,
            }
        )
        blocked_host_action = runtime.handle_command(
            {
                "command": "execute_action",
                "session_token": host_token,
                "action": {
                    "action_type": "start_turn",
                    "actor_name": "Alice",
                    "actor_role": "human",
                    "handler_name": "start_turn_interactive",
                    "description": "Roll the dice for Alice.",
                    "property_name": None,
                    "target_player_name": None,
                    "fixed_choice": None,
                    "min_bid": None,
                    "max_bid": None,
                },
            }
        )
        reconnect = runtime.handle_command(
            {
                "command": "reconnect_online_slot",
                "session_code": session_code,
                "reconnect_token": guest_reconnect_token,
            }
        )

        self.assertTrue(disconnect["ok"])
        self.assertEqual("paused", disconnect["payload"]["online_session"]["state"])
        self.assertEqual(1, disconnect["payload"]["online_session"]["paused_seat_index"])
        self.assertFalse(blocked_host_action["ok"])
        self.assertIn("paused", blocked_host_action["error"])
        self.assertTrue(reconnect["ok"])
        self.assertEqual("in_game", reconnect["payload"]["online_session"]["state"])
        self.assertIn("frontend_state", reconnect["payload"])

    def test_host_can_replace_disconnected_online_player_with_ai(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = self._write_checkpoint(temp_dir)
            runtime = BackendRuntime(default_ai_checkpoint_path=checkpoint_path)
            create = runtime.handle_command(
                {
                    "command": "create_online_lobby",
                    "host_player_name": "Alice",
                    "seat_count": 2,
                    "starting_cash": 1500,
                }
            )
            host_token = create["payload"]["session_token"]
            session_code = create["payload"]["online_session"]["session_code"]
            guest = runtime.handle_command(
                {
                    "command": "claim_online_slot",
                    "session_code": session_code,
                    "seat_index": 1,
                    "player_name": "Bob",
                }
            )
            runtime.handle_command({"command": "start_online_game", "session_token": host_token})
            runtime.handle_command(
                {
                    "command": "disconnect_online_slot",
                    "session_code": session_code,
                    "session_token": guest["payload"]["session_token"],
                }
            )

            replace = runtime.handle_command(
                {
                    "command": "replace_disconnected_online_slot_with_ai",
                    "session_token": host_token,
                    "seat_index": 1,
                    "checkpoint_path": checkpoint_path,
                    "action_cooldown_seconds": 0.0,
                }
            )

        self.assertTrue(replace["ok"])
        self.assertEqual("in_game", replace["payload"]["online_session"]["state"])
        self.assertEqual("ai", replace["payload"]["online_session"]["seats"][1]["status"])
        self.assertEqual("ai", runtime.game.players[1].role)
        self.assertIn("frontend_state", replace["payload"])

    def test_online_lobby_can_start_with_scripted_ai_slot(self) -> None:
        runtime = BackendRuntime()
        create = runtime.handle_command(
            {
                "command": "create_online_lobby",
                "host_player_name": "Alice",
                "seat_count": 2,
                "starting_cash": 1500,
            }
        )
        host_token = create["payload"]["session_token"]

        assign = runtime.handle_command(
            {
                "command": "assign_ai_to_online_slot",
                "session_token": host_token,
                "seat_index": 1,
                "player_name": "Bot Seat",
                "checkpoint_path": "scripted:expansionist_builder",
                "action_cooldown_seconds": 0.0,
            }
        )
        start = runtime.handle_command(
            {
                "command": "start_online_game",
                "session_token": host_token,
            }
        )

        self.assertTrue(assign["ok"])
        self.assertTrue(start["ok"])
        self.assertEqual("scripted:expansionist_builder", runtime.ai_checkpoint_paths_by_player["Bot Seat"])
        self.assertEqual("ai", start["payload"]["online_session"]["seats"][1]["status"])

    def test_save_and_load_commands_round_trip_runtime_game(self) -> None:
        runtime = BackendRuntime()
        runtime.handle_command(
            {
                "command": "create_game",
                "setup": {
                    "player_names": ["Alice", "Bob"],
                    "starting_cash": 1700,
                },
            }
        )
        expected_state = runtime.game.get_serialized_frontend_state()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            path = handle.name

        try:
            save_response = runtime.handle_command({"command": "save_game", "file_path": path})
            load_response = runtime.handle_command({"command": "load_game", "file_path": path})
        finally:
            os.unlink(path)

        self.assertTrue(save_response["ok"])
        self.assertTrue(load_response["ok"])
        self.assertEqual(expected_state, load_response["payload"]["frontend_state"])


if __name__ == "__main__":
    unittest.main()