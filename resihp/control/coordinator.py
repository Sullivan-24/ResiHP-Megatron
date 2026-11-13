"""Coordinator: the single membership authority.

Responsibilities
* node leases (agents heartbeat on behalf of their workers) and hang detection;
* failure declaration + fencing (the dead/hung worker is SIGKILLed by its agent and
  every message from an older generation is ignored);
* generation-numbered reconfiguration transactions:

    RECONFIG_NOTICE(g) -> ABORTED(g) from every live worker -> planner -> PREPARE(g)
    -> READY(g) from every member of the new world -> COMMIT(g)

  A failure observed at any point restarts the transaction with g+1, so a crash
  during reconfiguration is handled exactly like any other crash.
* hosting the TCPStore that data-plane communicators rendezvous on, so the store
  never lives on a training rank that can die.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from resihp.control import protocol as P
from resihp.control.kvstore import InMemoryKV, KVStore
from resihp.control.membership import Membership, WorkerRecord, WorkerState
from resihp.utils.log import get_logger
from resihp.utils.rpc import Connection, Server

log = get_logger(__name__)


@dataclass
class ClusterView:
    generation: int
    alive: list[WorkerRecord]
    failed: list[str]
    previous_world: list[str]
    previous_payload: dict | None
    recovery: dict[str, dict]
    reason: str
    reports: dict[str, list[dict]] = field(default_factory=dict)


PlannerFn = Callable[[ClusterView], tuple[list[str], dict]]


def keep_all_planner(view: ClusterView) -> tuple[list[str], dict]:
    world = [w.worker_id for w in view.alive]
    return world, {"world_size": len(world)}


class TransactionRestart(Exception):
    pass


class Coordinator:
    def __init__(self, *, host: str = "127.0.0.1", port: int = 0, kv: KVStore | None = None,
                 planner: PlannerFn | None = None, expected_workers: int | None = None, min_workers: int = 1,
                 start_grace: float = 5.0, lease_timeout: float = 3.0, hang_timeout: float = 60.0,
                 abort_timeout: float = 15.0, ready_timeout: float = 180.0, host_store: bool = True,
                 store_port: int = 0, namespace: str = "resihp", coordinator_id: str = "coord-0",
                 leader_ttl: float | None = None, tick: float = 0.05,
                 report_handler: Callable[[str, dict], None] | None = None):
        self.host = host
        self._port = port
        self.kv: KVStore = kv if kv is not None else InMemoryKV()
        self.planner = planner or keep_all_planner
        self.expected_workers = expected_workers
        self.min_workers = min_workers
        self.start_grace = start_grace
        self.lease_timeout = lease_timeout
        self.hang_timeout = hang_timeout
        self.abort_timeout = abort_timeout
        self.ready_timeout = ready_timeout
        self.host_store = host_store
        self.store_port = store_port
        self.ns = namespace
        self.coordinator_id = coordinator_id
        self.leader_ttl = leader_ttl
        self.tick = tick
        self.report_handler = report_handler

        self.membership = Membership()
        self.generation = 0
        self.committed_generation = 0
        self.world: list[str] = []
        self.payload: dict | None = None
        self.phase = "waiting"
        self.history: list[dict] = []

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._agents: dict[str, Connection] = {}
        self._node_seen: dict[str, float] = {}
        self._pending_failures: set[str] = set()
        self._failed_since_commit: list[str] = []
        self._planned_reason: str | None = None
        self._force_immediate = False
        self._joiners: set[str] = set()
        self._open_notice: dict | None = None
        self._start_requested = False
        self._first_register: float | None = None
        self._prepared_any = False
        self._aborted: dict[int, dict[str, dict]] = {}
        self._ready: dict[int, set[str]] = {}
        self._build_failed: dict[int, set[str]] = {}
        self._done: set[str] = set()
        self._reports: dict[str, list[dict]] = {}
        self._stopped = threading.Event()
        self._is_leader = threading.Event()
        self._leader_lease: int | None = None
        self.server: Server | None = None
        self.store = None
        self.store_addr: tuple[str, int] | None = None
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------ lifecycle
    @property
    def address(self) -> tuple[str, int]:
        assert self.server is not None
        return self.server.address

    @property
    def is_leader(self) -> bool:
        return self._is_leader.is_set()

    def start(self, block_until_leader: bool = True) -> "Coordinator":
        if self.leader_ttl is None:
            self._become_leader()
        else:
            t = threading.Thread(target=self._election_loop, daemon=True, name="coord-election")
            t.start()
            self._threads.append(t)
            if block_until_leader:
                self._is_leader.wait()
        return self

    def stop(self) -> None:
        self._stopped.set()
        self._is_leader.clear()
        with self._cv:
            self._cv.notify_all()
        if self.server is not None:
            self.server.stop()
        if self._leader_lease is not None:
            try:
                self.kv.revoke(self._leader_lease)
            except Exception:  # noqa: BLE001
                pass
        self.store = None

    def _become_leader(self) -> None:
        self._restore_from_kv()
        if self.host_store:
            import torch.distributed as dist  # data-plane rendezvous only; control plane stays torch-free

            self.store = dist.TCPStore(self.host, self.store_port, is_master=True, wait_for_workers=False)
            self.store_addr = (self.host, self.store.port)
        self.server = Server(self.host, self._port, self._on_message, on_close=self._on_conn_close)
        self._is_leader.set()
        for target, name in ((self._monitor_loop, "coord-monitor"), (self._reconfig_loop, "coord-reconfig")):
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)
        log.warning("%s is leader at %s", self.coordinator_id, self.server.address)

    def _election_loop(self) -> None:
        key = f"{self.ns}/leader"
        while not self._stopped.is_set():
            if self._leader_lease is None:
                lease = self.kv.grant_lease(self.leader_ttl)
                if self.kv.cas(key, None, self.coordinator_id, lease=lease):
                    self._leader_lease = lease
                    self._become_leader()
                else:
                    self.kv.revoke(lease)
            elif not self.kv.keepalive(self._leader_lease):
                log.error("%s lost leadership; stepping down", self.coordinator_id)
                self.stop()
                return
            time.sleep(self.leader_ttl / 4)

    # ------------------------------------------------------------------ persistence
    def _persist(self) -> bool:
        state = {
            "generation": self.generation,
            "committed_generation": self.committed_generation,
            "world": self.world,
            "payload": self.payload,
            "members": {wid: {"node_id": w.node_id, "local_rank": w.local_rank, "device": w.device,
                              "state": w.state.value, "incarnation": w.incarnation, "pid": w.pid,
                              "health": w.health}
                        for wid, w in self.membership.workers.items()},
        }
        return self._kv_put(f"{self.ns}/state", json.dumps(state))

    def _kv_put(self, key: str, value: str) -> bool:
        """Write control state, fenced on still holding the leader key (a deposed leader must not
        clobber its successor's state)."""
        if self.leader_ttl is None:
            self.kv.put(key, value)
            return True
        if not self._stopped.is_set() and self.kv.put_if(key, value, f"{self.ns}/leader", self.coordinator_id):
            return True
        if not self._stopped.is_set():
            log.error("%s no longer holds the leader key; stepping down", self.coordinator_id)
            self._stopped.set()
            self._is_leader.clear()
            threading.Thread(target=self.stop, daemon=True).start()
        return False

    def _restore_from_kv(self) -> None:
        raw = self.kv.get(f"{self.ns}/state")
        if not raw:
            return
        state = json.loads(raw)
        self.generation = state["generation"]
        self.committed_generation = state["committed_generation"]
        self.world = state["world"]
        self.payload = state["payload"]
        now = time.monotonic()
        # Agents need a moment to discover the new leader; do not mistake that for node loss.
        grace = now + 2 * self.lease_timeout
        for wid, m in state["members"].items():
            rec = WorkerRecord(worker_id=wid, node_id=m["node_id"], local_rank=m["local_rank"], device=m["device"],
                               pid=m.get("pid", 0), incarnation=m["incarnation"], state=WorkerState(m["state"]),
                               health=dict(m.get("health", {})))
            rec.progress_changed_at = grace
            self.membership.workers[wid] = rec
            if rec.alive:
                self._node_seen[rec.node_id] = grace
        self._prepared_any = self.committed_generation > 0
        self.phase = "running" if self.committed_generation > 0 else "waiting"
        if self.generation > self.committed_generation and self.committed_generation > 0:
            # The previous leader died mid-transaction; workers may be blocked in PREPARE.
            self._planned_reason = "failover"
            self._force_immediate = True
        log.warning("%s restored generation %d (committed %d)", self.coordinator_id, self.generation,
                    self.committed_generation)

    # ------------------------------------------------------------------ messaging
    def _send_to_worker(self, worker_id: str, message: dict) -> bool:
        w = self.membership.workers.get(worker_id)
        if w is None:
            return False
        conn = self._agents.get(w.node_id)
        return conn is not None and conn.send(P.msg(P.TO_WORKER, worker_id=worker_id, msg=message))

    def _broadcast(self, worker_ids, message: dict) -> None:
        for wid in worker_ids:
            self._send_to_worker(wid, message)

    def _on_conn_close(self, conn: Connection) -> None:
        node = conn.meta.get("node_id")
        with self._lock:
            if node and self._agents.get(node) is conn:
                del self._agents[node]

    def _on_message(self, conn: Connection, m: dict) -> None:
        kind = m.get("type")
        now = time.monotonic()
        with self._cv:
            if kind == P.AGENT_REGISTER:
                node = m["node_id"]
                conn.meta["node_id"] = node
                self._agents[node] = conn
                self._node_seen[node] = now
                for rec in m.get("workers", []):
                    self._register_worker(rec, now)
                conn.send(P.msg(P.COORDINATOR_HELLO, coordinator_id=self.coordinator_id,
                                generation=self.generation, committed_generation=self.committed_generation))
                if self._open_notice is not None:
                    acked = self._aborted.get(self._open_notice["gen"], {})
                    for w in self.membership.on_node(node):
                        if w.alive and w.worker_id not in acked:
                            self._send_to_worker(w.worker_id, self._open_notice)
            elif kind == P.WORKER_REGISTER:
                self._node_seen[m["node_id"]] = now
                self._register_worker(m, now)
            elif kind == P.NODE_HEARTBEAT:
                node = m["node_id"]
                if node in self._agents:
                    self._node_seen[node] = now
                for wid, prog in m.get("progress", {}).items():
                    self.membership.update_progress(wid, prog, now)
            elif kind == P.WORKER_FAILED:
                self._declare_failed([m["worker_id"]], m.get("reason", "reported by agent"))
            elif kind == P.WORKER_MSG:
                self._on_worker_message(m["worker_id"], m["msg"])
            self._cv.notify_all()

    def _register_worker(self, rec: dict, now: float) -> None:
        wid = rec["worker_id"]
        old = self.membership.workers.get(wid)
        if old is not None and old.alive and old.pid == rec.get("pid", 0):
            return
        if self._first_register is None:
            self._first_register = now
        self.membership.register(WorkerRecord(worker_id=wid, node_id=rec["node_id"], local_rank=rec.get("local_rank", 0),
                                              device=rec.get("device", "cpu"), pid=rec.get("pid", 0)), now)
        if self.committed_generation > 0 and wid not in self.world:
            self._planned_reason = self._planned_reason or f"join:{wid}"
            self._joiners.add(wid)
        if self._open_notice is not None:
            # Registered after the notice went out but before planning: join this transaction.
            self._send_to_worker(wid, self._open_notice)
        log.warning("registered %s on %s", wid, rec["node_id"])

    def _on_worker_message(self, worker_id: str, inner: dict) -> None:
        w = self.membership.workers.get(worker_id)
        if w is None or not w.alive:
            return
        kind = inner.get("type")
        gen = inner.get("gen", 0)
        if kind == P.ABORTED and gen == self.generation:
            self._aborted.setdefault(gen, {})[worker_id] = inner
            w.committed_iteration = inner.get("committed_iteration", -1)
            w.has_reduced_grads = bool(inner.get("has_reduced_grads", False))
        elif kind == P.READY and gen == self.generation:
            self._ready.setdefault(gen, set()).add(worker_id)
        elif kind == P.REPORT:
            if inner.get("kind") == "timing":
                if self.report_handler is not None:
                    self.report_handler(worker_id, inner)
                return
            self._reports.setdefault(worker_id, []).append(inner)
            if inner.get("kind") == "build_failed" and inner.get("gen") == self.generation:
                self._build_failed.setdefault(self.generation, set()).add(worker_id)
            if inner.get("kind") == "done" and self.phase == "running":
                self._done.add(worker_id)
                if set(self.world) <= self._done:
                    self._finish_job()
            if inner.get("kind") == "fatal":
                self._declare_failed([worker_id], inner.get("detail", "fatal self-report"))
            elif inner.get("kind") == "replan" and self.committed_generation > 0:
                self._planned_reason = self._planned_reason or f"replan:{inner.get('detail', '')}"

    # ------------------------------------------------------------------ failure handling
    def declare_failed(self, worker_ids, reason: str) -> None:
        with self._cv:
            self._declare_failed(worker_ids, reason)
            self._cv.notify_all()

    def _finish_job(self) -> None:
        self.phase = "finished"
        self.history.append({"t": time.monotonic(), "event": "finished", "generation": self.committed_generation})
        self._persist()
        self._broadcast(self.membership.alive_ids(), P.msg(P.SHUTDOWN, gen=self.committed_generation))
        log.warning("job finished at generation %d", self.committed_generation)

    def _declare_failed(self, worker_ids, reason: str) -> None:
        if self.phase == "finished":
            self.membership.mark(worker_ids, WorkerState.FAILED)
            return
        newly = self.membership.mark(worker_ids, WorkerState.FAILED)
        if not newly:
            return
        log.warning("declaring failed %s (%s)", newly, reason)
        self.history.append({"t": time.monotonic(), "event": "failure", "workers": newly, "reason": reason})
        for wid in newly:
            w = self.membership.workers[wid]
            conn = self._agents.get(w.node_id)
            if conn is not None:
                conn.send(P.msg(P.KILL, worker_id=wid, reason=reason))
        self._pending_failures.update(newly)
        self._failed_since_commit.extend(newly)

    def _mark(self, phase: str, generation: int) -> None:
        """Timestamped transaction phase (MTTR breakdown in ResiBench)."""
        self.history.append({"t": time.monotonic(), "event": phase, "generation": generation})

    def request_replan(self, reason: str) -> None:
        with self._cv:
            self._planned_reason = reason
            self._cv.notify_all()

    def set_health(self, worker_id: str, **health) -> None:
        with self._cv:
            w = self.membership.workers.get(worker_id)
            if w is not None:
                w.health.update(health)

    def _monitor_loop(self) -> None:
        while not self._stopped.is_set():
            now = time.monotonic()
            with self._cv:
                for node, seen in list(self._node_seen.items()):
                    if now - seen > self.lease_timeout:
                        victims = [w.worker_id for w in self.membership.on_node(node) if w.alive]
                        if victims:
                            self._declare_failed(victims, f"node {node} lease expired")
                        del self._node_seen[node]
                if self.phase == "running":
                    self._check_hang(now)
                if self.committed_generation == 0 and not self._start_requested and self.phase == "waiting":
                    n = len(self.membership.alive())
                    enough = self.expected_workers is not None and n >= self.expected_workers
                    grace = (self.expected_workers is None and self._first_register is not None and n >= self.min_workers
                             and now - self._first_register >= self.start_grace)
                    if enough or grace:
                        self._start_requested = True
                self._cv.notify_all()
            time.sleep(self.tick)

    def _check_hang(self, now: float) -> None:
        alive = [w for w in self.membership.alive() if w.worker_id in self.world]
        if not alive or any(now - w.progress_changed_at <= self.hang_timeout for w in alive):
            return
        culprits = [w for w in alive if not w.progress.get("in_collective", False)]
        if not culprits or len(culprits) == len(alive):
            key = lambda w: (w.progress.get("iteration", -1), w.progress.get("op_seq", -1))  # noqa: E731
            lowest = min(key(w) for w in alive)
            culprits = [w for w in alive if key(w) == lowest]
        if len(culprits) < len(alive):
            self._declare_failed([w.worker_id for w in culprits], "hang (no progress while peers wait)")
        else:
            for w in alive:
                w.progress_changed_at = now
            log.error("global stall with no identifiable culprit; extending deadline")

    # ------------------------------------------------------------------ transactions
    def _reconfig_loop(self) -> None:
        while not self._stopped.is_set():
            with self._cv:
                while not self._stopped.is_set() and not (
                        self._pending_failures or self._start_requested or
                        (self._planned_reason and self.committed_generation > 0)):
                    self._cv.wait(self.tick)
                if self._stopped.is_set() or self.phase == "finished":
                    return
                reason = ("failure" if self._pending_failures else
                          "start" if self.committed_generation == 0 else self._planned_reason or "replan")
                immediate = bool(self._pending_failures) or self.committed_generation == 0 or self._force_immediate
                self._planned_reason = None
                self._force_immediate = False
                self._start_requested = False
            try:
                self._transaction(reason, immediate)
            except Exception:  # noqa: BLE001 - never let the authority die silently
                log.exception("transaction failed; will retry")
                time.sleep(self.tick)

    def _wait(self, pred: Callable[[], bool], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._cv:
            while not pred():
                if self._pending_failures:
                    raise TransactionRestart()
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stopped.is_set():
                    return False
                self._cv.wait(min(self.tick, remaining))
            return True

    def _transaction(self, reason: str, immediate: bool) -> None:
        while not self._stopped.is_set():
            with self._cv:
                self._pending_failures.clear()
                self._done.clear()
                self.generation += 1
                gen = self.generation
                self.phase = "reconfiguring"
                participants = self.membership.alive_ids()
                if not self._persist():
                    return
            try:
                recovery: dict[str, dict] = {}
                if reason == "failover":
                    # Agents are still discovering the new leader; nodes that never come back are
                    # declared failed by the lease monitor and restart this transaction.
                    self._wait(lambda: {w.node_id for w in self.membership.alive()} <= set(self._agents),
                               2 * self.lease_timeout)
                if self._prepared_any:
                    notice = P.msg(P.RECONFIG_NOTICE, gen=gen, failed=list(self._failed_since_commit),
                                   immediate=immediate, reason=reason)
                    with self._cv:
                        self._open_notice = notice
                        self._mark("notice", gen)
                    self._broadcast(participants, notice)

                    def all_aborted() -> bool:
                        alive = set(self.membership.alive_ids())
                        return alive <= set(self._aborted.get(gen, {}))

                    if not self._wait(all_aborted, self.abort_timeout):
                        with self._cv:
                            self._open_notice = None
                            slow = sorted(set(self.membership.alive_ids()) - set(self._aborted.get(gen, {})))
                            self._declare_failed(slow, "no ABORTED ack within timeout")
                        raise TransactionRestart()

                with self._cv:
                    self._open_notice = None
                    self._mark("aborted", gen)
                    recovery = dict(self._aborted.get(gen, {}))
                    alive = [w for w in self.membership.alive() if not self._prepared_any or w.worker_id in recovery]
                    self._joiners -= {w.worker_id for w in alive}
                    if not self._joiners and (self._planned_reason or "").startswith("join:"):
                        self._planned_reason = None
                    view = ClusterView(generation=gen, alive=alive, failed=list(self._failed_since_commit),
                                       previous_world=list(self.world), previous_payload=self.payload,
                                       recovery=recovery, reason=reason, reports=dict(self._reports))
                world, payload = self.planner(view)
                if not world:
                    raise RuntimeError("planner returned an empty world")
                with self._cv:
                    self._mark("planned", gen)
                    if not (self._persist() and
                            self._kv_put(f"{self.ns}/plan/{gen}", json.dumps({"world": world, "payload": payload}))):
                        return
                    targets = self.membership.alive_ids()
                    self._prepared_any = True
                prepare = P.msg(P.PREPARE, gen=gen, world=world, payload=payload, reason=reason,
                                store_addr=list(self.store_addr) if self.store_addr else None,
                                store_prefix=f"{self.ns}/gen{gen}/", recovery=recovery)
                self._broadcast(targets, prepare)

                def all_ready() -> bool:
                    return set(world) <= self._ready.get(gen, set())

                if not self._wait(all_ready, self.ready_timeout):
                    with self._cv:
                        laggards = set(world) - self._ready.get(gen, set())
                        # Ranks that reported a failed build were blocked by a peer; the peers that
                        # neither became READY nor reported are the culprits.
                        culprits = laggards - self._build_failed.get(gen, set()) or laggards
                        self._declare_failed(sorted(culprits), "no READY within timeout")
                    raise TransactionRestart()

                with self._cv:
                    if self._stopped.is_set():
                        return
                    self._mark("ready", gen)
                    self.committed_generation = gen
                    self.world = list(world)
                    self.payload = payload
                    self.phase = "running"
                    self._failed_since_commit = []
                    self._reports = {}
                    now = time.monotonic()
                    for w in self.membership.alive():
                        w.progress_changed_at = now
                    if not self._persist():
                        return
                    self.history.append({"t": now, "event": "commit", "generation": gen, "world": list(world),
                                         "reason": reason})
                    targets = self.membership.alive_ids()
                self._broadcast(targets, P.msg(P.COMMIT, gen=gen))
                log.warning("committed generation %d (%d workers, reason=%s)", gen, len(world), reason)
                return
            except TransactionRestart:
                with self._cv:
                    self._open_notice = None
                immediate = True
                reason = "failure"
                continue

    # ------------------------------------------------------------------ introspection
    def wait_for_commit(self, generation: int, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._cv:
            while self.committed_generation < generation:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(min(0.05, remaining))
        return True
