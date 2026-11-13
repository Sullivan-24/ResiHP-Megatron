"""NodeAgent: one per node.

* spawns (or adopts) local workers and never tears down healthy workers when a
  peer dies — the opposite of torchrun's elastic agent;
* aggregates worker heartbeats over localhost and renews the node lease at the
  coordinator, so coordinator load scales with nodes, not GPUs;
* detects local deaths immediately (process exit, lost connection, stalled
  heartbeat, fatal GPU health event) and reports them;
* executes fencing (SIGKILL) on coordinator command;
* fails over between coordinator replicas and tells workers to self-fence when
  no coordinator is reachable for longer than the lease.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field

from resihp.control import protocol as P
from resihp.utils.log import get_logger
from resihp.utils.rpc import Connection, Server, connect, fmt_addr

log = get_logger(__name__)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


@dataclass
class LocalWorker:
    worker_id: str
    local_rank: int
    pid: int = 0
    device: str = "cpu"
    proc: subprocess.Popen | None = None
    conn: Connection | None = None
    last_hb: float = field(default_factory=time.monotonic)
    progress: dict = field(default_factory=dict)
    dead: bool = False


class NodeAgent:
    def __init__(self, node_id: str, coordinator_addrs: list[tuple[str, int]], *, host: str = "127.0.0.1",
                 port: int = 0, hb_interval: float = 0.25, worker_timeout: float = 2.0,
                 self_fence_after: float = 5.0, health_probe=None, health_every: int = 8):
        self.node_id = node_id
        self.coordinator_addrs = list(coordinator_addrs)
        self.hb_interval = hb_interval
        self.worker_timeout = worker_timeout
        self.self_fence_after = self_fence_after
        self.health_probe = health_probe
        self.health_every = health_every
        self.workers: dict[str, LocalWorker] = {}
        self._lock = threading.RLock()
        self._coord: Connection | None = None
        self._coord_lost_at: float | None = None
        self._fenced_notice_sent = False
        self._stopped = threading.Event()
        self.server = Server(host, port, self._on_worker_message, on_close=self._on_worker_close)
        self._threads: list[threading.Thread] = []

    @property
    def address(self) -> tuple[str, int]:
        return self.server.address

    def start(self) -> "NodeAgent":
        self._connect_coordinator()
        for target, name in ((self._heartbeat_loop, "agent-hb"), (self._monitor_loop, "agent-monitor")):
            t = threading.Thread(target=target, daemon=True, name=f"{name}-{self.node_id}")
            t.start()
            self._threads.append(t)
        return self

    def stop(self, kill_workers: bool = False) -> None:
        self._stopped.set()
        if kill_workers:
            for w in list(self.workers.values()):
                self._kill(w)
        if self._coord is not None:
            self._coord.close()
        self.server.stop()

    # ------------------------------------------------------------------ workers
    def spawn(self, cmd: list[str], nproc: int, env: dict[str, str] | None = None,
              devices: list[str] | None = None) -> list[LocalWorker]:
        out = []
        for i in range(nproc):
            wid = f"{self.node_id}g{i}"
            dev = devices[i] if devices else "cpu"
            wenv = dict(os.environ)
            wenv.update(env or {})
            wenv.update({"RESIHP_AGENT_ADDR": fmt_addr(self.address), "RESIHP_WORKER_ID": wid,
                         "RESIHP_NODE_ID": self.node_id, "RESIHP_LOCAL_RANK": str(i), "RESIHP_DEVICE": dev})
            if dev.startswith("cuda:"):
                wenv["CUDA_VISIBLE_DEVICES"] = dev.split(":", 1)[1]
            proc = subprocess.Popen(cmd, env=wenv)
            w = LocalWorker(worker_id=wid, local_rank=i, pid=proc.pid, device=dev, proc=proc)
            with self._lock:
                self.workers[wid] = w
            out.append(w)
        return out

    def _on_worker_message(self, conn: Connection, m: dict) -> None:
        kind = m.get("type")
        now = time.monotonic()
        if kind == P.REGISTER:
            wid = m["worker_id"]
            with self._lock:
                w = self.workers.get(wid)
                if w is None or w.dead:
                    w = LocalWorker(worker_id=wid, local_rank=m.get("local_rank", 0))
                    self.workers[wid] = w
                w.pid = m.get("pid", w.pid)
                w.device = m.get("device", w.device)
                w.conn = conn
                w.last_hb = now
                conn.meta["worker_id"] = wid
            self._to_coord(P.msg(P.WORKER_REGISTER, worker_id=wid, node_id=self.node_id, local_rank=w.local_rank,
                                 pid=w.pid, device=w.device))
            return
        wid = conn.meta.get("worker_id")
        if wid is None:
            return
        with self._lock:
            w = self.workers.get(wid)
            if w is None or w.dead:
                return
            w.last_hb = now
            if kind == P.HEARTBEAT:
                w.progress = m.get("progress", {})
                return
        self._to_coord(P.msg(P.WORKER_MSG, worker_id=wid, msg=m))

    def _on_worker_close(self, conn: Connection) -> None:
        wid = conn.meta.get("worker_id")
        if wid is not None and not self._stopped.is_set():
            with self._lock:
                w = self.workers.get(wid)
                if w is not None and w.conn is conn:
                    self._mark_dead(w, "control connection lost")

    def _mark_dead(self, w: LocalWorker, reason: str) -> None:
        if w.dead:
            return
        w.dead = True
        log.warning("worker %s dead: %s", w.worker_id, reason)
        self._to_coord(P.msg(P.WORKER_FAILED, worker_id=w.worker_id, node_id=self.node_id, reason=reason))

    def _kill(self, w: LocalWorker) -> None:
        if w.pid > 0:
            try:
                os.kill(w.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if w.proc is not None:
            try:
                w.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _monitor_loop(self) -> None:
        polls = 0
        while not self._stopped.is_set():
            now = time.monotonic()
            with self._lock:
                for w in list(self.workers.values()):
                    if w.dead:
                        continue
                    if w.proc is not None and w.proc.poll() is not None:
                        self._mark_dead(w, f"exited with {w.proc.returncode}")
                    elif w.proc is None and w.pid and not pid_alive(w.pid):
                        self._mark_dead(w, "process gone")
                    elif w.conn is not None and now - w.last_hb > self.worker_timeout:
                        self._mark_dead(w, f"heartbeat stalled {now - w.last_hb:.1f}s (hang)")
            polls += 1
            if self.health_probe is not None and polls % self.health_every == 0:
                for ev in self.health_probe.poll():
                    if ev.fatal:
                        with self._lock:
                            for w in self.workers.values():
                                if not w.dead and ev.matches(w.device):
                                    self._mark_dead(w, f"health: {ev.describe()}")
            time.sleep(self.hb_interval / 2)

    # ------------------------------------------------------------------ coordinator link
    def _to_coord(self, message: dict) -> None:
        conn = self._coord
        if conn is None or not conn.send(message):
            log.warning("coordinator unreachable; dropped %s", message.get("type"))

    def _connect_coordinator(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while not self._stopped.is_set() and time.monotonic() < deadline:
            for addr in self.coordinator_addrs:
                try:
                    conn = connect(addr, self._on_coord_message, self._on_coord_close, timeout=0.5,
                                   name=f"agent-{self.node_id}")
                except TimeoutError:
                    continue
                with self._lock:
                    self._coord = conn
                    self._coord_lost_at = None
                    self._fenced_notice_sent = False
                    workers = [{"worker_id": w.worker_id, "node_id": self.node_id, "local_rank": w.local_rank,
                                "pid": w.pid, "device": w.device}
                               for w in self.workers.values() if not w.dead and w.conn is not None]
                conn.send(P.msg(P.AGENT_REGISTER, node_id=self.node_id, workers=workers))
                return True
            time.sleep(0.1)
        return False

    def _on_coord_close(self, conn: Connection) -> None:
        with self._lock:
            if self._coord is conn:
                self._coord = None
                self._coord_lost_at = time.monotonic()

    def _on_coord_message(self, conn: Connection, m: dict) -> None:
        kind = m.get("type")
        if kind == P.KILL:
            with self._lock:
                w = self.workers.get(m["worker_id"])
            if w is not None:
                log.warning("fencing %s: %s", w.worker_id, m.get("reason"))
                w.dead = True
                self._kill(w)
        elif kind == P.TO_WORKER:
            with self._lock:
                w = self.workers.get(m["worker_id"])
            if w is not None and not w.dead and w.conn is not None:
                w.conn.send(m["msg"])

    def _heartbeat_loop(self) -> None:
        while not self._stopped.is_set():
            if self._coord is None:
                lost = self._coord_lost_at or time.monotonic()
                if not self._fenced_notice_sent and time.monotonic() - lost > self.self_fence_after:
                    with self._lock:
                        for w in self.workers.values():
                            if not w.dead and w.conn is not None:
                                w.conn.send(P.msg(P.SELF_FENCE, reason="coordinator unreachable"))
                    self._fenced_notice_sent = True
                self._connect_coordinator(timeout=self.hb_interval)
            else:
                with self._lock:
                    progress = {w.worker_id: w.progress for w in self.workers.values() if not w.dead}
                self._to_coord(P.msg(P.NODE_HEARTBEAT, node_id=self.node_id, progress=progress))
            time.sleep(self.hb_interval)
