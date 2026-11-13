"""Helpers for multi-process gloo tests on localhost."""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import sys
import time
import traceback
from typing import Any, Callable


def loopback_iface() -> str:
    return "lo0" if sys.platform == "darwin" else "lo"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _entry(fn: Callable, rank: int, world: int, port: int, q: Any, args: tuple) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", loopback_iface())
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        q.put((rank, "ok", fn(rank, world, port, *args)))
    except BaseException as e:  # noqa: BLE001 - surface every worker failure to the test
        q.put((rank, "err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def spawn(fn: Callable, world: int, *args: Any, timeout: float = 120.0,
          expect_dead: set[int] | None = None) -> dict[int, Any]:
    """Run ``fn(rank, world, port, *args)`` in ``world`` processes; return {rank: result}.

    Ranks in ``expect_dead`` may exit without reporting (they are killed on purpose).
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=_entry, args=(fn, r, world, port, q, args), daemon=True) for r in range(world)]
    for p in procs:
        p.start()
    expect_dead = expect_dead or set()
    results: dict[int, Any] = {}
    errors: dict[int, str] = {}
    need = world - len(expect_dead)
    deadline = time.monotonic() + timeout
    while len(results) + len(errors) < need and time.monotonic() < deadline:
        try:
            rank, status, payload = q.get(timeout=0.5)
        except Exception:  # noqa: BLE001 - queue.Empty
            continue
        if status == "ok":
            results[rank] = payload
        else:
            errors[rank] = payload
    for p in procs:
        p.join(timeout=2)
        if p.is_alive():
            p.kill()
    if errors:
        raise AssertionError("worker failures:\n" + "\n".join(f"[rank {r}] {e}" for r, e in sorted(errors.items())))
    missing = set(range(world)) - set(results) - expect_dead
    if missing:
        raise AssertionError(f"ranks {sorted(missing)} did not report within {timeout}s")
    return results
