from __future__ import annotations

import threading
import time
import unittest

from monopoly.gui.pygame_frontend.controller import BackendClient
from monopoly.gui.backend_process import BackendRuntime
from monopoly.gui.transport import SocketTransportClient, find_free_port, serve_socket_requests


class SocketTransportTests(unittest.TestCase):
    def test_socket_transport_round_trip_uses_same_command_contract(self) -> None:
        runtime = BackendRuntime()
        host = "127.0.0.1"
        port = find_free_port()
        server_thread = threading.Thread(
            target=serve_socket_requests,
            args=(host, port, runtime.handle_command),
            daemon=True,
        )
        server_thread.start()

        client = SocketTransportClient(host, port)
        try:
            create_response = client.request(
                {
                    "command": "create_game",
                    "setup": {"player_names": ["A", "B"], "starting_cash": 1600},
                }
            )
            state_response = client.request({"command": "get_state"})
            shutdown_response = client.request({"command": "shutdown"})
        finally:
            client.close()
            server_thread.join(timeout=2)

        self.assertTrue(create_response["ok"])
        self.assertEqual(1600, create_response["payload"]["frontend_state"]["game_view"]["starting_cash"])
        self.assertTrue(state_response["ok"])
        self.assertTrue(shutdown_response["ok"])

    def test_socket_transport_accepts_multiple_clients_on_same_server(self) -> None:
        runtime = BackendRuntime()
        host = "127.0.0.1"
        port = find_free_port()
        server_thread = threading.Thread(
            target=serve_socket_requests,
            args=(host, port, runtime.handle_command),
            daemon=True,
        )
        server_thread.start()

        client_one = SocketTransportClient(host, port)
        client_two = SocketTransportClient(host, port)
        try:
            lobby_response = client_one.request(
                {
                    "command": "create_online_lobby",
                    "host_player_name": "Host",
                    "seat_count": 3,
                    "starting_cash": 1600,
                }
            )
            session_code = lobby_response["payload"]["online_session"]["session_code"]
            join_response = client_two.request(
                {
                    "command": "claim_online_slot",
                    "session_code": session_code,
                    "seat_index": 1,
                    "player_name": "Remote",
                }
            )
            state_response = client_one.request(
                {
                    "command": "get_online_session",
                    "session_code": session_code,
                    "session_token": lobby_response["payload"]["session_token"],
                }
            )
            shutdown_response = client_one.request({"command": "shutdown"})
        finally:
            client_one.close()
            client_two.close()
            server_thread.join(timeout=2)

        self.assertTrue(lobby_response["ok"])
        self.assertTrue(join_response["ok"])
        seat_statuses = {seat["seat_index"]: seat["status"] for seat in state_response["payload"]["online_session"]["seats"]}
        self.assertEqual("connected", seat_statuses[1])
        self.assertTrue(shutdown_response["ok"])

    def test_socket_transport_broadcasts_online_session_events(self) -> None:
        runtime = BackendRuntime()
        host = "127.0.0.1"
        port = find_free_port()
        server_thread = threading.Thread(
            target=serve_socket_requests,
            args=(host, port, runtime.handle_command),
            daemon=True,
        )
        server_thread.start()

        host_client = SocketTransportClient(host, port)
        remote_client = SocketTransportClient(host, port)
        try:
            lobby_response = host_client.request(
                {
                    "command": "create_online_lobby",
                    "host_player_name": "Host",
                    "seat_count": 2,
                    "starting_cash": 1600,
                }
            )
            session_code = lobby_response["payload"]["online_session"]["session_code"]
            remote_client.request(
                {
                    "command": "claim_online_slot",
                    "session_code": session_code,
                    "seat_index": 1,
                    "player_name": "Remote",
                }
            )

            deadline = time.time() + 2.0
            events: list[dict[str, object]] = []
            while time.time() < deadline:
                events.extend(host_client.drain_events())
                if any(event.get("event") == "online_session_updated" for event in events):
                    break
                time.sleep(0.05)

            host_client.request({"command": "shutdown"})
        finally:
            host_client.close()
            remote_client.close()
            server_thread.join(timeout=2)

        self.assertTrue(any(event.get("event") == "online_session_updated" for event in events))

    def test_owned_backend_client_reconnects_after_transport_closes(self) -> None:
        runtime = BackendRuntime()
        host = "127.0.0.1"
        port = find_free_port()
        server_thread = threading.Thread(
            target=serve_socket_requests,
            args=(host, port, runtime.handle_command),
            daemon=True,
        )
        server_thread.start()

        client = BackendClient(host, port)
        try:
            create_payload = client.create_game(["A", "B"], 1500)
            self.assertEqual(1500, create_payload["frontend_state"]["game_view"]["starting_cash"])

            client.close()

            state_payload = client.get_state()
            shutdown_payload = client._request({"command": "shutdown"})
        finally:
            client.close()
            server_thread.join(timeout=2)

        self.assertIn("frontend_state", state_payload)
        self.assertTrue(shutdown_payload["shutting_down"])


if __name__ == "__main__":
    unittest.main()