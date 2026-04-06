from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

from monopoly.gui.transport import SocketTransportClient, serve_socket_requests


DEFAULT_RENDEZVOUS_HOST = "127.0.0.1"
DEFAULT_RENDEZVOUS_PORT = 47321
DEFAULT_REGISTRATION_TTL_SECONDS = 900


@dataclass(slots=True)
class _LobbyRegistration:
    session_code: str
    host: str
    port: int
    expires_at: float


class RendezvousRuntime:
    def __init__(self) -> None:
        self._registrations: dict[str, _LobbyRegistration] = {}

    def handle_command(self, command: Mapping[str, Any]) -> dict[str, Any]:
        try:
            self._prune_expired_registrations()
            action = str(command["command"])
            if action == "register_lobby":
                return self._handle_register_lobby(command)
            if action == "resolve_lobby":
                return self._handle_resolve_lobby(command)
            if action == "unregister_lobby":
                return self._handle_unregister_lobby(command)
            if action == "shutdown":
                return {"ok": True, "payload": {"shutting_down": True}}
            raise ValueError(f"Unsupported rendezvous command: {action}")
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _handle_register_lobby(self, command: Mapping[str, Any]) -> dict[str, Any]:
        session_code = self._normalize_session_code(command.get("session_code"))
        host = str(command.get("host") or "").strip()
        if not host:
            raise ValueError("Lobby registration requires a host address.")
        port = int(command.get("port"))
        if not 1 <= port <= 65535:
            raise ValueError("Lobby registration requires a valid TCP port.")
        ttl_seconds = max(30, min(int(command.get("ttl_seconds", DEFAULT_REGISTRATION_TTL_SECONDS)), 3600))
        registration = _LobbyRegistration(
            session_code=session_code,
            host=host,
            port=port,
            expires_at=time.time() + ttl_seconds,
        )
        self._registrations[session_code] = registration
        return {
            "ok": True,
            "payload": {
                "session_code": registration.session_code,
                "host": registration.host,
                "port": registration.port,
                "ttl_seconds": ttl_seconds,
            },
        }

    def _handle_resolve_lobby(self, command: Mapping[str, Any]) -> dict[str, Any]:
        session_code = self._normalize_session_code(command.get("session_code"))
        registration = self._registrations.get(session_code)
        if registration is None:
            raise ValueError(f"No active lobby is registered for code {session_code}.")
        return {
            "ok": True,
            "payload": {
                "session_code": registration.session_code,
                "host": registration.host,
                "port": registration.port,
            },
        }

    def _handle_unregister_lobby(self, command: Mapping[str, Any]) -> dict[str, Any]:
        session_code = self._normalize_session_code(command.get("session_code"))
        self._registrations.pop(session_code, None)
        return {"ok": True, "payload": {"session_code": session_code}}

    def _prune_expired_registrations(self) -> None:
        now = time.time()
        expired_codes = [code for code, registration in self._registrations.items() if registration.expires_at <= now]
        for code in expired_codes:
            self._registrations.pop(code, None)

    @staticmethod
    def _normalize_session_code(value: Any) -> str:
        session_code = str(value or "").strip().upper()
        if not session_code or not session_code.isalnum() or len(session_code) > 24:
            raise ValueError("Lobby codes must be 1-24 alphanumeric characters.")
        return session_code


class RendezvousClient:
    def __init__(self, host: str, port: int) -> None:
        self._transport = SocketTransportClient(host, port)

    def register_lobby(self, session_code: str, host: str, port: int, *, ttl_seconds: int = DEFAULT_REGISTRATION_TTL_SECONDS) -> dict[str, Any]:
        return self._request(
            {
                "command": "register_lobby",
                "session_code": session_code,
                "host": host,
                "port": port,
                "ttl_seconds": ttl_seconds,
            }
        )

    def resolve_lobby(self, session_code: str) -> dict[str, Any]:
        return self._request({"command": "resolve_lobby", "session_code": session_code})

    def unregister_lobby(self, session_code: str) -> dict[str, Any]:
        return self._request({"command": "unregister_lobby", "session_code": session_code})

    def close(self) -> None:
        self._transport.close()

    def _request(self, command: dict[str, Any]) -> dict[str, Any]:
        response = self._transport.request(command)
        if not response.get("ok"):
            raise ValueError(response.get("error", "Unknown rendezvous error."))
        return response["payload"]


def run_rendezvous_process(host: str = DEFAULT_RENDEZVOUS_HOST, port: int = DEFAULT_RENDEZVOUS_PORT) -> None:
    runtime = RendezvousRuntime()
    serve_socket_requests(host, port, runtime.handle_command)