"""Bandwidth-contention generator used to emulate communication fail-slow on a link.

Run ``--serve`` on one host and the client on the other; the client streams data
for ``--seconds``. On InfiniBand clusters prefer ``ib_write_bw`` with the same
endpoints; this TCP version is the portable fallback.
"""

from __future__ import annotations

import argparse
import socket
import time

CHUNK = 4 << 20


def serve(port: int) -> None:
    with socket.create_server(("", port)) as srv:
        while True:
            conn, _ = srv.accept()
            with conn:
                while conn.recv(CHUNK):
                    pass


def flood(host: str, port: int, seconds: float) -> float:
    payload = b"\0" * CHUNK
    sent = 0
    deadline = time.monotonic() + seconds
    with socket.create_connection((host, port)) as s:
        while time.monotonic() < deadline:
            s.sendall(payload)
            sent += CHUNK
    return sent / seconds


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=29700)
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args(argv)
    if args.serve:
        serve(args.port)
    else:
        rate = flood(args.host, args.port, args.seconds)
        print(f"contention rate {rate / 1e9:.2f} GB/s")


if __name__ == "__main__":
    main()
