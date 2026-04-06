from __future__ import annotations

import threading
import unittest

from monopoly.gui.rendezvous import RendezvousClient, RendezvousRuntime
from monopoly.gui.transport import find_free_port, serve_socket_requests


class RendezvousTests(unittest.TestCase):
    def test_rendezvous_resolves_registered_lobby(self) -> None:
        runtime = RendezvousRuntime()
        host = "127.0.0.1"
        port = find_free_port()
        server_thread = threading.Thread(
            target=serve_socket_requests,
            args=(host, port, runtime.handle_command),
            daemon=True,
        )
        server_thread.start()

        client = RendezvousClient(host, port)
        try:
            register_response = client.register_lobby("ABC123", "203.0.113.10", 4567)
            resolve_response = client.resolve_lobby("ABC123")
            shutdown_response = client._request({"command": "shutdown"})
        finally:
            client.close()
            server_thread.join(timeout=2)

        self.assertEqual("ABC123", register_response["session_code"])
        self.assertEqual("203.0.113.10", resolve_response["host"])
        self.assertEqual(4567, resolve_response["port"])
        self.assertTrue(shutdown_response["shutting_down"])


if __name__ == "__main__":
    unittest.main()