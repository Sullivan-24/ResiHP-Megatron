"""Progress-aware stage-level micro-batch migration across DP replicas (ResiHP section 6.3, Algorithm 1).

Each candidate move takes the last pending forward of the slowest replica at a
stage and hands it to the fastest replica at that stage; the move is simulated and
kept only if the makespan improves and the destination's in-flight activations stay
within its memory cap. Fail-stop eviction is the special case where a replica's
stage is unavailable: all of its micro-batches start out migrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

from resihp.plan.cost_model import CostModel
from resihp.plan.hetero_plan import HeteroParallelPlan, StageGroup
from resihp.plan.simulator import SimResult, schedule_with_fallback

Key = tuple[int, int]


@dataclass
class MigrationResult:
    assignment: tuple[tuple[int, ...], ...]
    makespan: float
    moves: int
    sim: SimResult
    initial_makespan: float


def simulate(plan: HeteroParallelPlan, cm: CostModel, split_backward: bool = False) -> SimResult:
    return schedule_with_fallback(plan, cm.cost, cm.comm, split_backward=split_backward, sync_time=cm.sync_time)


def initial_assignment(groups: list[StageGroup], num_stages: int, num_microbatches: int,
                       speed: Callable[[Key], float], mode: str = "speed") -> tuple[tuple[int, ...], ...]:
    """Default executors: each replica owns a contiguous block of micro-batches.

    ``equal`` gives every replica the same share (the paper's starting point);
    ``speed`` weights shares by each replica's slowest available stage.
    """
    avail = {(g.replica, g.stage) for g in groups}
    replicas = sorted({g.replica for g in groups})
    # A replica that lost some stages keeps executing its surviving stages; only the work
    # of the missing stages is evicted to peers (stage-level, not replica-level, exclusion).
    base_pool = replicas
    if mode == "equal":
        weights = {r: 1.0 for r in base_pool}
    else:
        weights = {r: min(speed((r, s)) for s in range(num_stages) if (r, s) in avail) for r in base_pool}
    total = sum(weights.values())
    quotas = [num_microbatches * weights[r] / total for r in base_pool]
    counts = [int(q) for q in quotas]
    for i in sorted(range(len(base_pool)), key=lambda i: (-(quotas[i] - counts[i]), i))[: num_microbatches - sum(counts)]:
        counts[i] += 1
    base: list[int] = []
    for r, c in zip(base_pool, counts):
        base.extend([r] * c)
    rows = []
    for s in range(num_stages):
        here = [r for r in replicas if (r, s) in avail]
        row = list(base)
        load = {r: row.count(r) for r in here}
        for m, r in enumerate(row):
            if r not in load:
                dst = min(here, key=lambda x: ((load[x] + 1) / max(speed((x, s)), 1e-12), x))
                row[m] = dst
                load[dst] += 1
        rows.append(tuple(row))
    return tuple(rows)


def _apply(cur: HeteroParallelPlan, kind: str, s: int, m: int, slow: int, fast: int) -> HeteroParallelPlan:
    rows = [list(r) for r in cur.assignment]
    if kind == "stage":
        for t in range(s, cur.num_stages, cur.num_physical_stages):
            rows[t][m] = fast
    else:
        for t in range(cur.num_stages):
            if rows[t][m] == slow and cur.group(fast, t) is not None:
                rows[t][m] = fast
    return replace(cur, assignment=tuple(tuple(r) for r in rows))


def progress_aware_migration(plan: HeteroParallelPlan, cm: CostModel, *, mem_cap: dict[Key, float] | None = None,
                             delta: float = 0.0, max_rounds: int = 32, split_backward: bool = False) -> MigrationResult:
    """Steepest-descent over two move types, each simulated before it is accepted:

    * stage move (Algorithm 1): one stage of one micro-batch, slowest -> fastest replica;
    * pipeline move: the same micro-batch on every stage where the fast replica exists,
      which escapes local optima when the bottleneck is spread over several stages.
    """
    def fits(res: SimResult) -> bool:
        return mem_cap is None or all(v <= mem_cap.get(k, math.inf) for k, v in res.peak_inflight.items())

    speed = cm.group_speed
    given = simulate(plan, cm, split_backward)
    start = [(given.makespan, "given", plan)]
    for mode in ("equal", "speed"):
        rows = initial_assignment(list(plan.stage_groups), plan.num_stages, plan.num_microbatches, speed, mode)
        cand = replace(plan, assignment=rows)
        res = simulate(cand, cm, split_backward)
        if not res.deadlock and fits(res):
            start.append((res.makespan, mode, cand))
    _, _, cur = min(start, key=lambda x: (x[0], x[1]))
    sim = simulate(cur, cm, split_backward)
    initial = given.makespan
    moves = 0
    M = plan.num_microbatches
    for _ in range(max_rounds):
        cands = []
        for s in range(cur.num_physical_stages):
            gs = cur.groups_at(s)
            if len(gs) < 2:
                continue
            done = {g.replica: sim.stage_forward_done.get((g.replica, s), 0.0) for g in gs}
            slow = max(done, key=lambda r: (done[r], -r))
            fast = min(done, key=lambda r: (done[r], r))
            if done[slow] - done[fast] <= delta or slow == fast:
                continue
            fwd_order = [e[1] for e in sim.order[(slow, s)] if e[0] == "F" and (len(e) == 2 or e[2] == s)]
            if len(fwd_order) <= 1:
                continue
            for kind in ("stage", "pipeline"):
                c = (kind, s if kind == "stage" else -1, fwd_order[-1], slow, fast)
                if c not in cands:
                    cands.append(c)
        best: tuple[HeteroParallelPlan, SimResult] | None = None
        for kind, s, m, slow, fast in cands:
            trial = _apply(cur, kind, s, m, slow, fast)
            if trial.assignment == cur.assignment:
                continue
            tsim = simulate(trial, cm, split_backward)
            if tsim.deadlock:
                continue
            if mem_cap is not None and any(v > mem_cap.get(k, math.inf) for k, v in tsim.peak_inflight.items()):
                continue
            bar = best[1].makespan if best else sim.makespan
            if tsim.makespan < bar - 1e-12:
                best = (trial, tsim)
        if best is None:
            break
        cur, sim = best
        moves += 1
    assert all(len(r) == M for r in cur.assignment)
    return MigrationResult(cur.assignment, sim.makespan, moves, sim, initial)
