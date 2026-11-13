"""Trace-driven simulation of one long pre-training job under different resilience policies.

The cluster is ``D`` replicas x ``S`` pipeline stages x ``TP``-GPU groups (Megatron rank
order, TP innermost). Progress is measured in *healthy seconds* (wall time the healthy job
would need), so ETTR = progress / wall time. A policy decides, per event, the downtime,
the work rolled back, and the throughput of the degraded cluster until hardware returns.

Throughput of a degraded cluster (fraction of healthy):

* each (replica, stage) group has a capacity ``c`` in [0, 1] (shrunk TP, slow members, or 0);
* policies that keep the global batch move *whole micro-batches*: a stage's time is the
  optimal integer assignment ``T_s = min T s.t. sum_r floor(T c_r) >= M`` (one extra
  micro-batch costs ``1 / (M / D)`` on the replicas that receive it, whatever ``D`` is), its
  effective capacity ``C_s = M / T_s``; without layer repartition the job runs at
  ``min_s C_s / D``, with perfect repartition at ``sum_s C_s / (S D)``; ResiHP achieves a
  blend ``beta`` of the two;
* restart-based policies re-form ``D'`` complete replicas and keep the batch, running at
  ``(M / D) / ceil(M / D')``; ``torchft`` drops replicas and lets the global batch shrink
  (different training semantics), ``torchft-bp`` keeps it;
* undetected slowdowns stall every replica (synchronous DP).

ResiHP's ``beta`` and overhead factor are fitted to the real planner + pipeline simulator at
small scale (``resihp.sim.calibrate``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from resihp.plan.model_spec import ModelProfile
from resihp.sim.traces import Event


@dataclass
class JobConfig:
    gpus: int
    tp: int = 8
    pp: int = 16
    gpus_per_node: int = 8
    iteration_s: float = 12.0
    microbatches_per_replica: int = 16
    min_tp: int = 4
    tp_degrees: tuple[int, ...] = (1, 2, 4, 8)
    """TP degrees the model's head / KV-group / FFN sizes divide into (8 KV groups: 1, 2, 4, 8)."""
    spare_nodes: int = 0
    profile: ModelProfile = field(default_factory=lambda: ModelProfile(alpha=1.0, beta=0.0))

    @property
    def dp(self) -> int:
        return self.gpus // (self.tp * self.pp)

    @property
    def nodes(self) -> int:
        return self.gpus // self.gpus_per_node

    @property
    def microbatches(self) -> int:
        return self.microbatches_per_replica * self.dp

    def group_of(self, gpu: int) -> tuple[int, int]:
        r = (gpu // self.tp) % self.dp
        s = gpu // (self.tp * self.dp)
        return r, s

    def nodes_per_replica(self) -> int:
        return self.tp * self.pp // self.gpus_per_node


@dataclass
class PolicyParams:
    name: str
    detect_s: float
    reconfig_s: float
    lost_iterations: float = 0.5
    restart_s: float = 0.0
    checkpoint_interval_s: float = 1800.0
    job_restart_s: float = 600.0
    job_rollback_s: float | None = None
    repair_s: float = 0.0
    tp_shrink: bool = False
    migrate: bool = False
    beta: float = 0.0
    overhead: float = 1.0
    slow_detect_s: float | None = None
    slow_mitigation: str = "none"  # none | migrate | exclude_node
    node_granular: bool = False
    replica_granular: bool = False
    restart_on_fail: bool = False
    keep_batch: bool = True
    ideal: bool = False


def ideal_params() -> PolicyParams:
    """Upper bound: no downtime, no rollback, throughput = surviving GPU fraction, slowdowns ignored."""
    return PolicyParams("Ideal", detect_s=0.0, reconfig_s=0.0, lost_iterations=0.0, job_restart_s=0.0,
                        job_rollback_s=0.0, ideal=True)


def resihp_params(job: JobConfig, *, comm_rebuild_s: float = 10.0, migrate_s: float = 5.0, snapshot_iters: int = 10,
                  beta: float = 0.5, overhead: float = 1.0, slow_detect_iters: float = 3.0) -> PolicyParams:
    return PolicyParams("ResiHP", detect_s=3.0, reconfig_s=comm_rebuild_s + migrate_s + job.iteration_s,
                        lost_iterations=0.5, job_restart_s=120.0, job_rollback_s=snapshot_iters / 2 * job.iteration_s,
                        repair_s=comm_rebuild_s + migrate_s, tp_shrink=True, migrate=True, beta=beta, overhead=overhead,
                        slow_detect_s=slow_detect_iters * job.iteration_s, slow_mitigation="migrate")


def baseline_params(job: JobConfig, *, restart_s: float = 300.0, checkpoint_interval_s: float = 1200.0) -> list[PolicyParams]:
    """Defaults put checkpoint-restart near Llama 3's reported >90% effective training time on
    16K GPUs (fail-stop only); ``restart_s`` is swept in the sensitivity study."""
    return [
        PolicyParams("torchft", detect_s=10.0, reconfig_s=15.0, job_restart_s=restart_s, repair_s=15.0,
                     replica_granular=True, checkpoint_interval_s=checkpoint_interval_s, keep_batch=False),
        PolicyParams("torchft-bp", detect_s=10.0, reconfig_s=15.0, job_restart_s=restart_s, repair_s=15.0,
                     replica_granular=True, checkpoint_interval_s=checkpoint_interval_s),
        PolicyParams("NVRx", detect_s=60.0, reconfig_s=0.0, restart_s=restart_s, job_restart_s=restart_s,
                     checkpoint_interval_s=checkpoint_interval_s, restart_on_fail=True, replica_granular=True,
                     slow_detect_s=300.0, slow_mitigation="exclude_node"),
        PolicyParams("ReCycle", detect_s=10.0, reconfig_s=15.0, job_restart_s=restart_s, repair_s=15.0, migrate=True,
                     checkpoint_interval_s=checkpoint_interval_s),
        PolicyParams("Oobleck", detect_s=10.0, reconfig_s=60.0, job_restart_s=restart_s, repair_s=60.0,
                     node_granular=True, checkpoint_interval_s=checkpoint_interval_s),
        PolicyParams("Greyhound", detect_s=60.0, reconfig_s=0.0, restart_s=restart_s, job_restart_s=restart_s,
                     checkpoint_interval_s=checkpoint_interval_s, restart_on_fail=True, replica_granular=True,
                     migrate=True, slow_detect_s=10 * job.iteration_s, slow_mitigation="migrate"),
    ]


@dataclass
class SimResult:
    policy: str
    gpus: int
    horizon_s: float
    progress_s: float
    downtime_s: float
    lost_s: float
    degraded_loss_s: float
    events: int
    mttr_s: list[float]

    @property
    def ettr(self) -> float:
        return self.progress_s / self.horizon_s

    def to_json(self) -> dict:
        n = len(self.mttr_s)
        return {"policy": self.policy, "gpus": self.gpus, "ettr": self.ettr,
                "downtime_frac": self.downtime_s / self.horizon_s, "lost_frac": self.lost_s / self.horizon_s,
                "degraded_frac": self.degraded_loss_s / self.horizon_s, "events": self.events,
                "mttr_mean_s": sum(self.mttr_s) / n if n else 0.0,
                "mttr_p99_s": sorted(self.mttr_s)[int(0.99 * (n - 1))] if n else 0.0}


class ClusterState:
    def __init__(self, job: JobConfig, p: PolicyParams):
        self.job, self.p = job, p
        D, S = job.dp, job.pp
        self.lost: dict[int, float] = {}
        self.slow: dict[int, tuple[float, float, bool]] = {}  # gpu -> (factor, until, mitigated)
        self.c = [[1.0] * S for _ in range(D)]
        self.stage_cap = [float(D)] * S
        self.spares = 0
        self.spares_free = job.spare_nodes
        self.covered: set[int] = set()
        self.failed: dict[int, float] = {}

    def fail(self, gpus, until: float) -> tuple[list[int], int]:
        """Record failed GPUs; nodes that become bad take a free spare if there is one (their
        GPUs then count as healthy). Returns (GPUs that now degrade capacity, spares used)."""
        gpn = self.job.gpus_per_node
        used = 0
        for g in gpus:
            self.failed[g] = max(self.failed.get(g, 0.0), until)
        degrading = []
        for node in sorted({g // gpn for g in gpus}):
            if node in self.covered:
                continue
            if self.spares_free > 0:
                self.spares_free -= 1
                self.covered.add(node)
                used += 1
                for g in range(node * gpn, (node + 1) * gpn):
                    self.lost.pop(g, None)
            else:
                degrading += [g for g in gpus if g // gpn == node and g not in self.lost]
        for g in degrading:
            self.lost[g] = self.failed[g]
        return degrading, used

    def repair(self, gpus, now: float) -> list[int]:
        gpn = self.job.gpus_per_node
        back = [g for g in gpus if g in self.failed and self.failed[g] <= now + 1e-6]
        for g in back:
            del self.failed[g]
            self.lost.pop(g, None)
        for node in {g // gpn for g in back}:
            if node in self.covered and not any(h // gpn == node for h in self.failed):
                self.covered.discard(node)
                self.spares_free += 1
        return back

    def stage_capacity(self, s: int) -> float:
        """Effective capacity (healthy-group units) of stage ``s`` under whole-micro-batch moves."""
        return integer_capacity([self.c[r][s] for r in range(self.job.dp)], self.job.microbatches)

    def group_gpus(self, r: int, s: int) -> range:
        base = (s * self.job.dp + r) * self.job.tp
        return range(base, base + self.job.tp)

    def _group_capacity(self, r: int, s: int) -> float:
        """Relative capacity of a group once the planner knows about its state: lost members
        either shrink TP (Eq. 4 over the survivors, down to ``min_tp`` for memory) or evict
        the group; a known-slow member caps the group at its speed unless TP excludes it."""
        job, p = self.job, self.p
        speeds = sorted((self._speed(g) for g in self.group_gpus(r, s) if g not in self.lost), reverse=True)
        lost = job.tp - len(speeds)
        if (lost and not p.tp_shrink) or not speeds:
            return 0.0
        full = job.profile.group_speed(job.tp, 1.0)
        if not p.tp_shrink:
            return job.profile.group_speed(job.tp, speeds[-1]) / full
        return max((job.profile.group_speed(k, speeds[k - 1]) / full for k in job.tp_degrees
                    if job.min_tp <= k <= len(speeds)), default=0.0)

    def _speed(self, g: int) -> float:
        """Speed as seen by the planner: an undetected slowdown is not visible yet (it slows the
        whole synchronous job instead, see ``throughput``)."""
        f = self.slow.get(g)
        return 1.0 / f[0] if f is not None and f[2] else 1.0

    def refresh(self, gpus) -> None:
        for r, s in {self.job.group_of(g) for g in gpus}:
            self.c[r][s] = self._group_capacity(r, s)
            self.stage_cap[s] = self.stage_capacity(s)

    def throughput(self) -> float:
        job, p = self.job, self.p
        D, S, M = job.dp, job.pp, job.microbatches
        if p.ideal:
            return (job.gpus - len(self.lost)) / job.gpus
        unmitigated = [f for f, _, m in self.slow.values() if not m]
        sync = 1.0 / max(unmitigated) if unmitigated else 1.0
        if p.node_granular:
            bad = {g // job.gpus_per_node for g in self.lost}
            usable = max(0, job.nodes - max(0, len(bad) - self.spares))
            return 0.95 * usable / job.nodes * sync if bad else sync
        if p.replica_granular:
            bad_gpus = set(self.lost)
            if p.slow_mitigation == "exclude_node":
                bad_gpus |= {g for g, (_, _, m) in self.slow.items() if m}
            if p.restart_on_fail:
                # Each restart re-forms as many complete replicas as healthy + spare nodes allow.
                bad_nodes = {g // job.gpus_per_node for g in bad_gpus}
                up = min(D, (job.nodes - len(bad_nodes) + self.spares) // job.nodes_per_replica())
            else:
                down = {job.group_of(g)[0] for g in bad_gpus}
                up = D - max(0, len(down) - self.spares)
            if up <= 0:
                return 0.0
            caps = [1.0] * up
            if p.slow_mitigation == "migrate":
                slow_replicas: dict[int, float] = {}
                for g, (f, _, m) in self.slow.items():
                    r = job.group_of(g)[0]
                    if m and r < up:
                        slow_replicas[r] = min(slow_replicas.get(r, 1.0), 1.0 / f)
                for r, x in slow_replicas.items():
                    caps[r] = x
            if not p.keep_batch:
                return sum(caps) / D * sync
            return integer_capacity(caps, M) / D * sync
        if not p.migrate:
            return min(min(row) for row in self.c) * sync
        bottleneck = min(self.stage_cap) / D
        spread = sum(self.stage_cap) / (S * D)
        ideal = (1 - p.beta) * bottleneck + p.beta * spread
        return max(0.0, 1.0 - p.overhead * (1.0 - ideal)) * sync


def integer_capacity(caps: list[float], microbatches: int) -> float:
    """``M / T*`` with ``T* = min T s.t. sum floor(T c) >= M``: the throughput of groups with
    capacities ``caps`` that split ``M`` whole micro-batches optimally."""
    live = [c for c in caps if c > 0]
    if not live:
        return 0.0
    if all(c == 1.0 for c in live):
        return microbatches / math.ceil(microbatches / len(live))
    lo, hi = 0.0, microbatches / max(live)
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(math.floor(mid * c + 1e-9) for c in live) >= microbatches:
            hi = mid
        else:
            lo = mid
    return microbatches / hi


def simulate(events: list[Event], job: JobConfig, p: PolicyParams, horizon_s: float) -> SimResult:
    st = ClusterState(job, p)
    t = 0.0
    progress = downtime = lost_work = degraded = 0.0
    busy_until = 0.0
    last_ckpt_progress = 0.0
    next_ckpt = p.checkpoint_interval_s
    mttr: list[float] = []
    pending: list[tuple[float, str, object]] = []

    def advance(to: float) -> None:
        nonlocal t, progress, degraded, next_ckpt, last_ckpt_progress
        while t < to:
            nxt = min(to, next_ckpt)
            run_from = max(t, busy_until)
            if nxt > run_from:
                thr = st.throughput()
                progress += thr * (nxt - run_from)
                degraded += (1.0 - thr) * (nxt - run_from)
            t = nxt
            if t >= next_ckpt:
                last_ckpt_progress = progress
                next_ckpt += p.checkpoint_interval_s

    def outage(cost: float, rollback: float) -> None:
        nonlocal busy_until, progress, lost_work, downtime
        start = max(t, busy_until)
        busy_until = start + cost
        downtime += cost
        rb = min(rollback, progress)
        progress -= rb
        lost_work += rb
        mttr.append(cost)

    def restart_rollback() -> float:
        return progress - last_ckpt_progress

    queue = sorted([(e.t, "event", e) for e in events if e.t < horizon_s], key=lambda x: x[0])
    import heapq

    heap = [(x[0], i, x[1], x[2]) for i, x in enumerate(queue)]
    heapq.heapify(heap)
    seq = len(heap)
    while heap:
        when, _, kind, obj = heapq.heappop(heap)
        if when >= horizon_s:
            break
        advance(when)
        if kind == "event":
            e: Event = obj
            if e.kind == "fail" and e.domain == "job":
                rb = p.job_rollback_s if p.job_rollback_s is not None else restart_rollback()
                outage(p.detect_s + p.job_restart_s, rb)
            elif e.kind == "fail":
                degrading, spares_used = st.fail(e.targets, when + e.duration_s)
                st.refresh(e.targets)
                if p.restart_on_fail:
                    outage(p.detect_s + p.restart_s, restart_rollback())
                else:
                    # Online policies integrate a spare with one more (planned) reconfiguration.
                    extra = p.repair_s if spares_used else 0.0
                    outage(p.detect_s + p.reconfig_s + extra, p.lost_iterations * job.iteration_s)
                heapq.heappush(heap, (when + e.duration_s, seq, "repair", tuple(e.targets)))
                seq += 1
            elif e.kind == "slow":
                g = e.targets[0]
                st.slow[g] = (e.factor, when + e.duration_s, False)
                st.refresh([g])
                if p.slow_detect_s is not None and p.slow_detect_s < e.duration_s:
                    heapq.heappush(heap, (when + p.slow_detect_s, seq, "slow_detected", (g, e.factor)))
                    seq += 1
                heapq.heappush(heap, (when + e.duration_s, seq, "slow_end", g))
                seq += 1
            elif e.kind == "planned":
                outage(p.job_restart_s if p.restart_on_fail else p.reconfig_s, 0.0)
        elif kind == "repair":
            degraded_before = any(g in st.lost for g in obj)
            back = st.repair(obj, when)
            if back:
                st.refresh(back)
                if degraded_before and p.repair_s and not p.restart_on_fail:
                    outage(p.repair_s, 0.0)
        elif kind == "slow_detected":
            g, factor = obj
            if g in st.slow:
                f, until, _ = st.slow[g]
                st.slow[g] = (f, until, True)
                st.refresh([g])
                if p.slow_mitigation == "exclude_node":
                    outage(p.detect_s + p.restart_s, restart_rollback())
                elif p.slow_mitigation == "migrate":
                    outage(p.reconfig_s if not p.restart_on_fail else 0.0, 0.0)
        elif kind == "slow_end":
            if obj in st.slow:
                del st.slow[obj]
                st.refresh([obj])
    advance(horizon_s)
    return SimResult(p.name, job.gpus, horizon_s, progress, downtime, lost_work, degraded, len(mttr), mttr)
