"""Iteration-level transactions.

The optimizer step is the commit point. Reduced gradients are retained until the
step commits, which enables *roll-forward* recovery without any per-step
consensus round: survivors report ``(committed_iteration, has_reduced_grads)``
and the planner moves everyone to the newest committed iteration — ranks that
already hold the reduced gradients step locally (Adam is deterministic), the rest
copy state from a peer that committed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import torch

from resihp.state.rng_data import DataCursor, capture_rng_state, restore_rng_state


class TxnPhase(str, Enum):
    COMPUTE = "compute"
    REDUCED = "reduced"
    STEPPING = "stepping"
    COMMITTED = "committed"
    ABORTED = "aborted"


@dataclass
class IterationTransaction:
    iteration: int
    rng: dict
    cursor: dict
    phase: TxnPhase = TxnPhase.COMPUTE
    events: list[str] = field(default_factory=list)
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)


class TransactionManager:
    """``buffers`` returns the live non-trainable tensors an iteration mutates (FP8 amax history,
    MoE router token counts / expert bias); they are snapshotted at ``begin`` and restored
    whenever the iteration is rolled back."""

    def __init__(self, cursor: DataCursor | None = None, committed_iteration: int = -1,
                 buffers: Callable[[], dict[str, torch.Tensor]] | None = None):
        self.cursor = cursor or DataCursor(committed_iteration + 1)
        self.committed_iteration = committed_iteration
        self.current: IterationTransaction | None = None
        self.aborts = 0
        self.buffers = buffers

    def track_buffers(self, buffers: Callable[[], dict[str, torch.Tensor]] | None) -> None:
        self.buffers = buffers

    def _live_buffers(self) -> dict[str, torch.Tensor]:
        return self.buffers() if self.buffers is not None else {}

    def _restore_buffers(self) -> None:
        if self.current is None or not self.current.buffers:
            return
        with torch.no_grad():
            for k, t in self._live_buffers().items():
                if k in self.current.buffers:
                    t.copy_(self.current.buffers[k])

    def begin(self, iteration: int) -> IterationTransaction:
        if iteration != self.committed_iteration + 1:
            raise ValueError(f"iteration {iteration} does not follow committed {self.committed_iteration}")
        self.cursor.iteration = iteration
        snap = {k: t.detach().clone() for k, t in self._live_buffers().items()}
        self.current = IterationTransaction(iteration, capture_rng_state(), self.cursor.snapshot(), buffers=snap)
        return self.current

    def on_reduced(self) -> None:
        assert self.current is not None and self.current.phase == TxnPhase.COMPUTE
        self.current.phase = TxnPhase.REDUCED

    def on_step_begin(self) -> None:
        assert self.current is not None and self.current.phase == TxnPhase.REDUCED
        self.current.phase = TxnPhase.STEPPING

    def on_commit(self) -> None:
        assert self.current is not None and self.current.phase == TxnPhase.STEPPING
        self.current.phase = TxnPhase.COMMITTED
        self.committed_iteration = self.current.iteration
        self.cursor.iteration = self.committed_iteration + 1
        self.current = None

    @property
    def has_reduced_grads(self) -> bool:
        return self.current is not None and self.current.phase == TxnPhase.REDUCED

    def abort(self) -> dict:
        """Roll RNG and the data cursor back to the start of the in-flight iteration. An iteration
        that already reduced its gradients keeps its buffers: it may still be rolled forward."""
        info = self.recovery_info()
        if self.current is not None:
            restore_rng_state(self.current.rng)
            self.cursor.restore(self.current.cursor)
            if self.current.phase != TxnPhase.REDUCED:
                self.current.phase = TxnPhase.ABORTED
                self._restore_buffers()
            self.aborts += 1
        return info

    def discard(self) -> None:
        """Drop the in-flight iteration entirely (its gradients will not be used)."""
        if self.current is not None:
            self.cursor.restore(self.current.cursor)
            self._restore_buffers()
        self.current = None

    def adopt(self, iteration: int) -> None:
        """State was replaced (peer copy, snapshot or checkpoint) by the state at ``iteration``."""
        self.committed_iteration = iteration
        self.cursor.iteration = iteration + 1
        self.current = None

    def recovery_info(self) -> dict:
        return {"committed_iteration": self.committed_iteration, "has_reduced_grads": self.has_reduced_grads,
                "inflight_iteration": self.current.iteration if self.current else None}


def roll_forward_plan(recovery: dict[str, dict], world: list[str]) -> dict:
    """Decide the recovery target from ABORTED reports of the survivors in ``world``.

    Returns ``{"target_iteration", "local_step": [...], "needs_state": [...], "up_to_date": [...]}``.
    """
    infos = {w: recovery.get(w, {"committed_iteration": -1, "has_reduced_grads": False}) for w in world}
    target = max((i.get("committed_iteration", -1) for i in infos.values()), default=-1)
    local_step, needs_state, up_to_date = [], [], []
    for w in world:
        info = infos[w]
        c = info.get("committed_iteration", -1)
        if c == target:
            up_to_date.append(w)
        elif c == target - 1 and info.get("has_reduced_grads") and info.get("inflight_iteration") == target:
            local_step.append(w)
        else:
            needs_state.append(w)
    return {"target_iteration": target, "local_step": local_step, "needs_state": needs_state, "up_to_date": up_to_date}
