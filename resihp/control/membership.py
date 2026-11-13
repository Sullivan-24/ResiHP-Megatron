"""Membership bookkeeping shared by the coordinator and its planners."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class WorkerState(str, Enum):
    ALIVE = "alive"
    FAILED = "failed"
    FENCED = "fenced"


@dataclass
class WorkerRecord:
    worker_id: str
    node_id: str
    local_rank: int
    device: str = "cpu"
    pid: int = 0
    incarnation: int = 0
    state: WorkerState = WorkerState.ALIVE
    progress: dict = field(default_factory=dict)
    progress_changed_at: float = 0.0
    committed_iteration: int = -1
    has_reduced_grads: bool = False
    health: dict = field(default_factory=dict)

    @property
    def alive(self) -> bool:
        return self.state == WorkerState.ALIVE

    def sort_key(self) -> tuple:
        return (self.node_id, self.local_rank, self.worker_id)


class Membership:
    def __init__(self) -> None:
        self.workers: dict[str, WorkerRecord] = {}

    def register(self, rec: WorkerRecord, now: float) -> WorkerRecord:
        old = self.workers.get(rec.worker_id)
        if old is not None:
            rec.incarnation = old.incarnation + (0 if old.alive and old.pid == rec.pid else 1)
        rec.progress_changed_at = now
        self.workers[rec.worker_id] = rec
        return rec

    def alive(self) -> list[WorkerRecord]:
        return sorted((w for w in self.workers.values() if w.alive), key=WorkerRecord.sort_key)

    def alive_ids(self) -> list[str]:
        return [w.worker_id for w in self.alive()]

    def on_node(self, node_id: str) -> list[WorkerRecord]:
        return [w for w in self.workers.values() if w.node_id == node_id]

    def mark(self, worker_ids, state: WorkerState) -> list[str]:
        changed = []
        for wid in worker_ids:
            w = self.workers.get(wid)
            if w is not None and w.alive:
                w.state = state
                changed.append(wid)
        return changed

    def update_progress(self, worker_id: str, progress: dict, now: float) -> None:
        w = self.workers.get(worker_id)
        if w is None or not w.alive:
            return
        keys = ("gen", "iteration", "op_seq")
        if any(progress.get(k) != w.progress.get(k) for k in keys):
            w.progress_changed_at = now
        w.progress = progress
        if "p" in progress:
            w.health["p"] = progress["p"]
