"""Newline-delimited JSON messaging over TCP for the control plane.

Deliberately dependency-free: the control plane must keep working when the
data plane (NCCL) is wedged, so it never shares threads or sockets with it.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Callable

from resihp.utils.log import get_logger

log = get_logger(__name__)

MessageHandler = Callable[["Connection", dict], None]
CloseHandler = Callable[["Connection"], None]


class Connection:
    def __init__(self, sock: socket.socket, on_message: MessageHandler, on_close: CloseHandler | None = None,
                 name: str = ""):
        self.sock = sock
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self.meta: dict = {}
        self._on_message = on_message
        self._on_close = on_close
        self._wlock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name=f"rpc-{name}")
        self._reader.start()

    @property
    def closed(self) -> bool:
        return self._closed

    def send(self, msg: dict) -> bool:
        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        with self._wlock:
            if self._closed:
                return False
            try:
                self.sock.sendall(data)
                return True
            except OSError:
                pass
        self.close()
        return False

    def _read_loop(self) -> None:
        try:
            f = self.sock.makefile("rb")
            for line in f:
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("dropping malformed message")
                    continue
                try:
                    self._on_message(self, msg)
                except Exception:  # noqa: BLE001 - one bad handler must not kill the connection
                    log.exception("handler failed for %s", msg.get("type"))
        except (OSError, ValueError):
            pass
        finally:
            self.close()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
        if self._on_close is not None:
            try:
                self._on_close(self)
            except Exception:  # noqa: BLE001
                log.exception("close handler failed")


class Server:
    def __init__(self, host: str, port: int, on_message: MessageHandler,
                 on_connect: Callable[[Connection], None] | None = None,
                 on_close: CloseHandler | None = None):
        self._on_message = on_message
        self._on_connect = on_connect
        self._on_close = on_close
        self._sock = socket.create_server((host, port))
        self._sock.listen(1024)
        self.address: tuple[str, int] = self._sock.getsockname()[:2]
        self.connections: set[Connection] = set()
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(target=self._accept_loop, daemon=True, name="rpc-accept")
        self._thread.start()

    def _accept_loop(self) -> None:
        while not self._stopped:
            try:
                s, _ = self._sock.accept()
            except OSError:
                break
            with self._lock:
                # A connection accepted from the backlog while stop() runs must be refused, not
                # served: a deposed coordinator that keeps answering strands the agent on a
                # server that will never act, until its node lease expires.
                if self._stopped:
                    s.close()
                    break
                conn = Connection(s, self._on_message, self._closed)
                self.connections.add(conn)
            if self._on_connect is not None:
                self._on_connect(conn)

    def _closed(self, conn: Connection) -> None:
        with self._lock:
            self.connections.discard(conn)
        if self._on_close is not None:
            self._on_close(conn)

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
        try:
            self._sock.close()
        except OSError:
            pass
        with self._lock:
            conns, self.connections = list(self.connections), set()
        for c in conns:
            c.close()


def connect(addr: tuple[str, int], on_message: MessageHandler, on_close: CloseHandler | None = None,
            timeout: float = 10.0, retry_interval: float = 0.05, name: str = "") -> Connection:
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection(addr, timeout=min(2.0, timeout))
            s.settimeout(None)
            return Connection(s, on_message, on_close, name=name)
        except OSError as e:
            last = e
            time.sleep(retry_interval)
    raise TimeoutError(f"could not connect to {addr}: {last}")


def parse_addr(s: str) -> tuple[str, int]:
    host, port = s.rsplit(":", 1)
    return host, int(port)


def fmt_addr(addr: tuple[str, int]) -> str:
    return f"{addr[0]}:{addr[1]}"
