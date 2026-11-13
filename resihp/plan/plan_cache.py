"""Precomputed plans for the failures that are most likely next.

After every committed plan, a background thread computes the replan for each
single-node loss (and, optionally, each single-device loss). On a matching failure
the coordinator gets its plan in O(1), taking planning off the fail-stop critical
path; anything else falls back to synchronous planning.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from resihp.plan.hetero_plan import HeteroParallelPlan
from resihp.plan.topology import ClusterTopology
from resihp.utils.log import get_logger

log = get_logger(__name__)


class PlanCache:
    def __init__(self, replan: Callable[[HeteroParallelPlan, ClusterTopology], HeteroParallelPlan],
                 single_device: bool = False):
        self.replan = replan
        self.single_device = single_device
        self._lock = threading.Lock()
        self._entries: dict[frozenset[str], HeteroParallelPlan] = {}
        self._base: tuple[int, frozenset[str]] | None = None
        self._thread: threading.Thread | None = None
        self.hits = 0
        self.misses = 0
        self.build_seconds = 0.0

    def scenarios(self, plan: HeteroParallelPlan, topology: ClusterTopology) -> list[frozenset[str]]:
        used = set(plan.world)
        out = []
        for node, devs in topology.nodes().items():
            failed = frozenset(d.id for d in devs if d.id in used)
            if failed:
                out.append(failed)
        if self.single_device:
            out.extend(frozenset([d]) for d in plan.world)
        return out

    def warm(self, plan: HeteroParallelPlan, topology: ClusterTopology, blocking: bool = False) -> None:
        base = (plan.generation, frozenset(topology.ids()))
        with self._lock:
            self._base = base
            self._entries = {}

        def work() -> None:
            t0 = time.monotonic()
            for failed in self.scenarios(plan, topology):
                with self._lock:
                    if self._base != base:
                        return
                try:
                    p = self.replan(plan, topology.without(failed))
                except Exception:  # noqa: BLE001 - infeasible scenarios just miss the cache
                    continue
                with self._lock:
                    if self._base == base:
                        self._entries[failed] = p
            self.build_seconds = time.monotonic() - t0

        if blocking:
            work()
        else:
            self._thread = threading.Thread(target=work, daemon=True, name="plan-cache")
            self._thread.start()

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def lookup(self, base_generation: int, topology_ids: set[str], failed: set[str]) -> HeteroParallelPlan | None:
        with self._lock:
            ok = self._base is not None and self._base[0] == base_generation and \
                set(self._base[1]) - set(failed) == set(topology_ids)
            p = self._entries.get(frozenset(failed)) if ok else None
        if p is None:
            self.misses += 1
        else:
            self.hits += 1
        return p
