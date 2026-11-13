"""In-worker control client.

Runs on its own thread and never touches the data plane except through the
abort callbacks registered by the communicator manager: when a reconfiguration
notice arrives while the training thread is blocked inside a collective, the
callbacks abort the communicators so the training thread unwinds.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

from resihp.control import protocol as P
from resihp.utils.log import get_logger
from resihp.utils.rpc import Connection, connect, parse_addr

log = get_logger(__name__)


class ReconfigInterrupted(Exception):
    """A newer generation was announced while this worker was waiting on an older one."""


class JobFinished(Exception):
    """The coordinator declared the job complete."""


class WorkerClient:
    def __init__(self, agent_addr: str | None = None, worker_id: str | None = None, *, local_rank: int | None = None,
                 node_id: str | None = None, device: str | None = None, hb_interval: float = 0.25):
        self.agent_addr = parse_addr(agent_addr or os.environ["RESIHP_AGENT_ADDR"])
        self.worker_id = worker_id or os.environ["RESIHP_WORKER_ID"]
        self.local_rank = local_rank if local_rank is not None else int(os.environ.get("RESIHP_LOCAL_RANK", "0"))
        self.node_id = node_id or os.environ.get("RESIHP_NODE_ID", "node")
        self.device = device or os.environ.get("RESIHP_DEVICE", "cpu")
        self.hb_interval = hb_interval
        self._cv = threading.Condition()
        self._progress: dict = {"gen": 0, "iteration": -1, "op_seq": 0, "phase": "init", "in_collective": False}
        self._recovery: dict = {"committed_iteration": -1, "has_reduced_grads": False}
        self._notice_gen = 0
        self._notice: dict | None = None
        self._acked_gen = 0
        self._prepares: dict[int, dict] = {}
        self._commits: set[int] = set()
        self._in_control_wait = False
        self._abort_callbacks: list[Callable[[], None]] = []
        self._shutdown = False
        self.committed_generation = 0
        self._conn: Connection | None = None
        self._stopped = threading.Event()
        self._hb_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle
    def connect(self, timeout: float = 30.0) -> "WorkerClient":
        self._conn = connect(self.agent_addr, self._on_message, timeout=timeout, name=f"worker-{self.worker_id}")
        self._conn.send(P.msg(P.REGISTER, worker_id=self.worker_id, local_rank=self.local_rank, pid=os.getpid(),
                              device=self.device, node_id=self.node_id))
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name="worker-hb")
        self._hb_thread.start()
        return self

    def close(self) -> None:
        self._stopped.set()
        if self._conn is not None:
            self._conn.close()

    def _heartbeat_loop(self) -> None:
        while not self._stopped.is_set():
            with self._cv:
                prog = dict(self._progress)
            prog["t"] = time.time()
            if self._conn is not None:
                self._conn.send(P.msg(P.HEARTBEAT, progress=prog))
            time.sleep(self.hb_interval)

    # ------------------------------------------------------------------ training-thread API
    def set_progress(self, **kw) -> None:
        with self._cv:
            if "op_seq" not in kw:
                self._progress["op_seq"] = self._progress.get("op_seq", 0) + 1
            self._progress.update(kw)

    def set_recovery_info(self, **info) -> None:
        with self._cv:
            self._recovery.update(info)

    def add_abort_callback(self, fn: Callable[[], None]) -> None:
        self._abort_callbacks.append(fn)

    def reconfig_pending(self) -> bool:
        with self._cv:
            return self._notice_gen > self.committed_generation

    def immediate_abort_pending(self) -> bool:
        with self._cv:
            return self._notice_gen > self.committed_generation and bool(self._notice and self._notice.get("immediate"))

    def report(self, kind: str, **detail) -> None:
        if self._conn is not None:
            self._conn.send(P.msg(P.REPORT, kind=kind, **detail))

    def wait_prepare(self, timeout: float = 600.0) -> dict:
        """Ack any pending notice with the current recovery info, then block for the next PREPARE."""
        deadline = time.monotonic() + timeout
        with self._cv:
            self._in_control_wait = True
            try:
                while True:
                    if self._shutdown:
                        raise JobFinished()
                    self._ack_locked()
                    want = max(self._notice_gen, self.committed_generation + 1)
                    ready = [g for g in self._prepares if g >= want]
                    if ready:
                        gen = max(ready)
                        for g in [g for g in self._prepares if g < gen]:
                            del self._prepares[g]
                        return self._prepares[gen]
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"{self.worker_id}: no PREPARE for generation >= {want}")
                    self._cv.wait(min(0.05, remaining))
            finally:
                self._in_control_wait = False

    def finish(self, iteration: int, timeout: float = 600.0) -> str:
        """Report completion and wait: ``"shutdown"`` once the whole world is done, ``"reconfig"``
        if peers need this worker again (e.g. as a state source after a late failure)."""
        if self._conn is not None:
            self._conn.send(P.msg(P.REPORT, kind="done", iteration=iteration, gen=self.committed_generation))
        deadline = time.monotonic() + timeout
        with self._cv:
            self._in_control_wait = True
            try:
                while not self._shutdown:
                    if self._notice_gen > self.committed_generation:
                        return "reconfig"
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"{self.worker_id}: job completion not confirmed")
                    self._cv.wait(min(0.05, remaining))
                return "shutdown"
            finally:
                self._in_control_wait = False

    @property
    def shutdown(self) -> bool:
        return self._shutdown

    def send_ready(self, gen: int, **info) -> None:
        if self._conn is not None:
            self._conn.send(P.msg(P.READY, gen=gen, **info))

    def wait_commit(self, gen: int, timeout: float = 600.0) -> None:
        deadline = time.monotonic() + timeout
        with self._cv:
            self._in_control_wait = True
            try:
                while gen not in self._commits:
                    if self._notice_gen > gen:
                        raise ReconfigInterrupted(f"generation {self._notice_gen} announced while waiting for {gen}")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"{self.worker_id}: no COMMIT for generation {gen}")
                    self._cv.wait(min(0.05, remaining))
                self.committed_generation = max(self.committed_generation, gen)
                self._progress["gen"] = self.committed_generation
            finally:
                self._in_control_wait = False

    # ------------------------------------------------------------------ message handling
    def _ack_locked(self) -> None:
        if self._notice_gen > self._acked_gen and self._conn is not None:
            self._conn.send(P.msg(P.ABORTED, gen=self._notice_gen, **self._recovery))
            self._acked_gen = self._notice_gen

    def _on_message(self, conn: Connection, m: dict) -> None:
        kind = m.get("type")
        callbacks: list[Callable[[], None]] = []
        with self._cv:
            if kind == P.RECONFIG_NOTICE:
                gen = m["gen"]
                if gen > self._notice_gen:
                    self._notice_gen = gen
                    self._notice = m
                    log.info("notice for generation %d (immediate=%s, in_control_wait=%s)", gen,
                             m.get("immediate", True), self._in_control_wait)
                    if self._in_control_wait:
                        self._ack_locked()
                    elif m.get("immediate", True):
                        callbacks = list(self._abort_callbacks)
            elif kind == P.SELF_FENCE:
                callbacks = list(self._abort_callbacks)
            elif kind == P.PREPARE:
                if m["gen"] >= self._notice_gen:
                    self._prepares[m["gen"]] = m
            elif kind == P.COMMIT:
                self._commits.add(m["gen"])
            elif kind == P.SHUTDOWN:
                self._shutdown = True
            self._cv.notify_all()
        for fn in callbacks:
            try:
                fn()
            except Exception:  # noqa: BLE001 - abort is best effort; the training thread will time out otherwise
                log.exception("abort callback failed")
