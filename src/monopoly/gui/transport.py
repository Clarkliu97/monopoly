from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import queue
import socket
import struct
import threading
import time
import uuid
from typing import Any, Callable, Mapping


TRANSPORT_PROTOCOL_VERSION = 2
MAX_MESSAGE_BYTES = 4 * 1024 * 1024


logger = logging.getLogger("monopoly.gui.transport")


@dataclass(slots=True)
class _SocketPeer:
    connection: socket.socket
    send_lock: threading.Lock


class _EventBroadcaster:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._peers: list[_SocketPeer] = []

    def register(self, peer: _SocketPeer) -> None:
        with self._lock:
            self._peers.append(peer)

    def unregister(self, peer: _SocketPeer) -> None:
        with self._lock:
            self._peers = [candidate for candidate in self._peers if candidate is not peer]

    def broadcast(self, event_name: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            peers = list(self._peers)
        for peer in peers:
            try:
                _send_enveloped_message(
                    peer,
                    {
                        "kind": "event",
                        "protocol_version": TRANSPORT_PROTOCOL_VERSION,
                        "event": event_name,
                        "payload": dict(payload),
                    },
                )
            except OSError:
                self.unregister(peer)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def serve_socket_requests(host: str, port: int, handler: Callable[[Mapping[str, Any]], dict[str, Any]]) -> None:
    stop_event = threading.Event()
    handler_lock = threading.Lock()
    client_threads: list[threading.Thread] = []
    broadcaster = _EventBroadcaster()
    handler_owner = getattr(handler, "__self__", None)
    if handler_owner is not None and hasattr(handler_owner, "set_event_publisher"):
        handler_owner.set_event_publisher(broadcaster.broadcast)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        server.settimeout(0.2)
        logger.info("Socket server listening on %s:%d.", host, port)
        while not stop_event.is_set():
            try:
                connection, _address = server.accept()
            except socket.timeout:
                continue
            logger.debug("Accepted socket client connection on %s:%d.", host, port)
            peer = _SocketPeer(connection=connection, send_lock=threading.Lock())
            broadcaster.register(peer)
            client_thread = threading.Thread(
                target=_serve_client_connection,
                args=(peer, handler, handler_lock, stop_event, broadcaster),
                daemon=True,
            )
            client_thread.start()
            client_threads.append(client_thread)

    if handler_owner is not None and hasattr(handler_owner, "set_event_publisher"):
        handler_owner.set_event_publisher(None)

    for client_thread in client_threads:
        client_thread.join(timeout=1)


def _serve_client_connection(
    peer: _SocketPeer,
    handler: Callable[[Mapping[str, Any]], dict[str, Any]],
    handler_lock: threading.Lock,
    stop_event: threading.Event,
    broadcaster: _EventBroadcaster,
) -> None:
    connection = peer.connection
    with connection:
        while not stop_event.is_set():
            try:
                envelope = _receive_json_message(connection)
            except (EOFError, ConnectionResetError, OSError):
                logger.debug("Socket client connection closed.")
                break
            _validate_envelope(envelope, expected_kind="request")
            command = envelope["payload"]
            with handler_lock:
                response = handler(command)
            _send_enveloped_message(
                peer,
                {
                    "kind": "response",
                    "protocol_version": TRANSPORT_PROTOCOL_VERSION,
                    "request_id": envelope["request_id"],
                    "payload": response,
                },
            )
            if command.get("command") == "shutdown":
                stop_event.set()
                break
    broadcaster.unregister(peer)


class SocketTransportClient:
    def __init__(self, host: str, port: int, timeout: float = 10.0) -> None:
        self._socket = self._connect(host, port, timeout)
        self._send_lock = threading.Lock()
        self._pending: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._closed = threading.Event()
        self._reader_thread = threading.Thread(target=self._read_messages, daemon=True)
        self._reader_thread.start()
        logger.info("Connected transport client to %s:%d.", host, port)

    def request(self, command: dict[str, Any]) -> dict[str, Any]:
        if self._closed.is_set():
            raise ConnectionError("Transport client is closed.")
        request_id = uuid.uuid4().hex
        response_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = response_queue
        try:
            _send_enveloped_message(
                _SocketPeer(connection=self._socket, send_lock=self._send_lock),
                {
                    "kind": "request",
                    "protocol_version": TRANSPORT_PROTOCOL_VERSION,
                    "request_id": request_id,
                    "payload": command,
                },
            )
            response = response_queue.get(timeout=10.0)
            return response["payload"]
        except queue.Empty as exc:
            raise TimeoutError("Timed out waiting for a transport response.") from exc
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    def drain_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    @property
    def is_closed(self) -> bool:
        return self._closed.is_set()

    def close(self) -> None:
        self._closed.set()
        try:
            self._socket.close()
        except OSError:
            pass
        self._fail_pending_requests(ConnectionError("Transport client is closed."))
        self._reader_thread.join(timeout=1)
        logger.info("Closed transport client connection.")

    @staticmethod
    def _connect(host: str, port: int, timeout: float) -> socket.socket:
        deadline = time.time() + timeout
        last_error: OSError | None = None
        while time.time() < deadline:
            try:
                return socket.create_connection((host, port), timeout=1.0)
            except OSError as exc:
                last_error = exc
                logger.debug("Transport connection attempt to %s:%d failed: %s", host, port, exc)
                time.sleep(0.1)
        raise ConnectionError(f"Could not connect to backend at {host}:{port}.") from last_error

    def _read_messages(self) -> None:
        try:
            while not self._closed.is_set():
                envelope = _receive_json_message(self._socket)
                kind = str(envelope.get("kind"))
                if kind == "response":
                    _validate_envelope(envelope, expected_kind="response")
                    request_id = str(envelope["request_id"])
                    with self._pending_lock:
                        response_queue = self._pending.get(request_id)
                    if response_queue is not None:
                        response_queue.put(envelope)
                elif kind == "event":
                    _validate_envelope(envelope, expected_kind="event")
                    self._events.put({"event": envelope["event"], "payload": envelope["payload"]})
                else:
                    raise ValueError(f"Unsupported transport message kind: {kind}")
        except (EOFError, OSError, ValueError) as exc:
            if not self._closed.is_set():
                logger.warning("Transport reader stopped because the connection closed unexpectedly: %s", exc)
                self._fail_pending_requests(ConnectionError("Transport connection closed unexpectedly.") if isinstance(exc, (EOFError, OSError)) else exc)
        finally:
            self._closed.set()

    def _fail_pending_requests(self, exc: Exception) -> None:
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for response_queue in pending:
            response_queue.put({"payload": {"ok": False, "error": str(exc)}})


def _send_enveloped_message(peer: _SocketPeer, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("Transport payload exceeds the maximum allowed size.")
    with peer.send_lock:
        peer.connection.sendall(struct.pack("!I", len(data)))
        peer.connection.sendall(data)


def _receive_json_message(connection: socket.socket) -> dict[str, Any]:
    header = _recv_exact(connection, 4)
    length = struct.unpack("!I", header)[0]
    if length > MAX_MESSAGE_BYTES:
        raise ValueError("Transport payload exceeds the maximum allowed size.")
    body = _recv_exact(connection, length)
    return json.loads(body.decode("utf-8"))


def _validate_envelope(envelope: Mapping[str, Any], *, expected_kind: str) -> None:
    if str(envelope.get("kind")) != expected_kind:
        raise ValueError(f"Expected a {expected_kind} envelope.")
    if int(envelope.get("protocol_version", -1)) != TRANSPORT_PROTOCOL_VERSION:
        raise ValueError("Transport protocol version is not supported.")
    if expected_kind in {"request", "response"} and "request_id" not in envelope:
        raise ValueError("Transport envelope is missing request_id.")
    if "payload" not in envelope:
        raise ValueError("Transport envelope is missing payload.")
    if expected_kind == "event" and "event" not in envelope:
        raise ValueError("Transport event envelope is missing the event name.")


def _recv_exact(connection: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        data = connection.recv(length - len(chunks))
        if not data:
            raise EOFError("Connection closed while receiving a message.")
        chunks.extend(data)
    return bytes(chunks)