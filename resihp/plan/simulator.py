"""List-scheduling pipeline simulator shared by the planner and the fail-slow detector.

Given a plan (groups, layer partition, micro-batch assignment) and a cost model, it
produces a concrete per-group compute order (1F1B-style with an in-flight limit,
optional zero-bubble W filling), timings, per-group in-flight activation peaks, and
the makespan. The order is converted to IR by ``ir.compute_order_to_ir``; the
per-direction FIFO discipline required for NCCL is enforced while scheduling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from resihp.plan.hetero_plan import HeteroParallelPlan

Key = tuple[int, int]
CostFn = Callable[[Key, str, int], float]
CommFn = Callable[[Key, Key, str], float]


@dataclass
class SimOp:
    key: Key
    op: str
    mb: int
    start: float
    end: float
    stage: int | None = None


@dataclass
class SimResult:
    order: dict[Key, list[tuple[str, int]]]
    ops: list[SimOp]
    makespan: float
    finish: dict[Key, float]
    busy: dict[Key, float]
    peak_inflight: dict[Key, int]
    deadlock: bool = False
    stage_forward_done: dict[Key, float] = field(default_factory=dict)


def default_inflight_limit(plan: HeteroParallelPlan) -> dict[Key, float]:
    """Micro-batches a physical slot may have started but not finished backward for.

    1F1B: the warm-up depth ``P - p``, scaled by the group's share of its stage. Interleaved: chunks of a slot share one
    FIFO channel per peer, so a per-slot limit could leave a blocked chunk-0 message at
    the head of a channel in front of a chunk-1 message of an in-flight micro-batch.
    Only the pipeline entry (data-driven, never a channel head) is limited, to rank 0's
    interleaved warm-up depth in micro-batches, which bounds every slot's in-flight work.
    """
    out: dict[Key, float] = {}
    P, v = plan.num_physical_stages, plan.vpp
    for g in plan.stage_groups:
        r, p = plan.physical_key(g.replica, g.stage)
        if v == 1:
            # A group that executes k times the average share of its stage (stage-level migration)
            # needs k times the warm-up depth to stay busy; its micro-batches are proportionally
            # shorter after repartition, and the planner's memory check bounds the result.
            share = len(plan.microbatches_of(r, p)) * plan.num_replicas / plan.num_microbatches
            out[(r, p)] = max(1, math.ceil((P - p) * max(1.0, share)))
        else:
            out[(r, p)] = math.ceil(((P - 1) * 2 + (v - 1) * P) / v) + 1 if p == 0 else math.inf
    return out


def megatron_interleaved_order(plan: HeteroParallelPlan, split_backward: bool = False) -> dict[Key, list[tuple]] | None:
    """Megatron's interleaved 1F1B order per physical slot, over the micro-batches the slot
    executes: forwards in groups of ``P`` micro-batches chunk by chunk, backwards with the
    chunk order reversed, warm-up depth ``(P - p - 1) * 2 + (v - 1) * P``. With a split
    backward each W directly follows its B. ``None`` when chunks of a slot execute
    different micro-batches."""
    P, v = plan.num_physical_stages, plan.vpp
    out: dict[Key, list[tuple]] = {}
    for g in plan.stage_groups:
        r, p = plan.physical_key(g.replica, g.stage)
        if (r, p) in out:
            continue
        mbs = plan.microbatches_of(r, p)
        if any(plan.group(r, p + c * P) is None or plan.microbatches_of(r, p + c * P) != mbs for c in range(v)):
            return None
        n = len(mbs)

        def chunk_op(k: int, backward: bool) -> tuple[int, int]:
            base = (k // (P * v)) * P
            size = min(P, n - base)
            c, i = divmod(k - base * v, size)
            return mbs[base + i], (v - 1 - c if backward else c) * P + p

        total = n * v
        warm = min(total, (P - p - 1) * 2 + (v - 1) * P)
        seq: list[tuple] = []

        def back(i: int) -> None:
            mb, s = chunk_op(i, True)
            seq.append(("B", mb, s))
            if split_backward:
                seq.append(("W", mb, s))

        for k in range(warm):
            seq.append(("F",) + chunk_op(k, False))
        for i in range(total - warm):
            seq.append(("F",) + chunk_op(warm + i, False))
            back(i)
        for i in range(total - warm, total):
            back(i)
        out[(r, p)] = seq
    return out


def list_schedule(plan: HeteroParallelPlan, cost: CostFn, comm: CommFn, *, split_backward: bool = False,
                  inflight_limit: dict[Key, int] | None = None, sync_time: Callable[[Key], float] | None = None,
                  optimizer_time: float = 0.0, fixed: dict[Key, list[tuple]] | None = None) -> SimResult:
    """Greedy earliest-start list scheduling of F/B/(W) over physical slots.

    Units are (replica, virtual stage); with interleaving several units share one
    physical slot. A message channel is a physical (src, dst, kind) triple shared by
    all chunks and consumed strictly in FIFO order (the NCCL-safety condition), and
    the in-flight limit counts micro-batches entering the slot's first chunk, so later
    chunks of an in-flight micro-batch can always proceed. With ``fixed`` every slot
    executes the given ``(op, mb, stage)`` sequence and only timing is computed.
    """
    pos = {k: 0 for k in fixed} if fixed is not None else None
    S, M, P = plan.num_stages, plan.num_microbatches, plan.num_physical_stages
    vpp = plan.vpp > 1
    units = [(g.replica, g.stage) for g in plan.stage_groups]
    phys = {u: plan.physical_key(*u) for u in units}
    keys = sorted(set(phys.values()), key=units.index if not vpp else None)
    chunks = {k: sorted(u[1] for u in units if phys[u] == k) for k in keys}
    limit = inflight_limit if inflight_limit is not None else default_inflight_limit(plan)
    assigned = {u: [m for m in range(M) if plan.executor(u[1], m) == u[0]] for u in units}
    f_left = {u: list(v) for u, v in assigned.items()}
    f_done: dict[Key, dict[int, float]] = {u: {} for u in units}
    b_done: dict[Key, dict[int, float]] = {u: {} for u in units}
    w_left: dict[Key, list[tuple[int, int]]] = {k: [] for k in keys}
    free = {k: 0.0 for k in keys}
    busy = {k: 0.0 for k in keys}
    inflight = {k: 0 for k in keys}
    peak = {k: 0 for k in keys}
    order: dict[Key, list[tuple]] = {k: [] for k in keys}
    ops: list[SimOp] = []
    # FIFO channels: (src slot, dst slot, kind) -> [(mb, dst stage, arrival)] in send order, plus a consume pointer.
    chan: dict[tuple[Key, Key, str], list[tuple[int, int, float]]] = {}
    head: dict[tuple[Key, Key, str], int] = {}
    total = sum(len(v) for v in assigned.values()) * (3 if split_backward else 2)
    scheduled = 0

    def heads(dst: Key, kind: str) -> list[tuple[int, int, float, tuple]]:
        out = []
        for ch, msgs in chan.items():
            if ch[1] == dst and ch[2] == kind and head[ch] < len(msgs):
                mb, s, t = msgs[head[ch]]
                out.append((mb, s, t, ch))
        return out

    while scheduled < total:
        best = None
        for k in keys:
            r = k[0]
            first = chunks[k][0]
            cands = []
            open_ = inflight[k] < limit.get(k, math.inf)
            if 0 in chunks[k] and f_left[(r, 0)] and open_:
                cands.append((free[k], 1, f_left[(r, 0)][0], 0, "F", None))
            for mb, s, t, ch in heads(k, "act"):
                if s != first or open_:
                    cands.append((max(free[k], t), 1, mb, s, "F", ch))
            if S - 1 in chunks[k]:
                u = (r, S - 1)
                pend = sorted(m for m in f_done[u] if m not in b_done[u])
                if pend:
                    cands.append((max(free[k], f_done[u][pend[0]]), 0, pend[0], S - 1, "B", None))
            for mb, s, t, ch in heads(k, "grad"):
                cands.append((max(free[k], t, f_done[(r, s)][mb]), 0, mb, s, "B", ch))
            if split_backward and w_left[k]:
                if fixed is None:
                    s, m = w_left[k][0]
                    cands.append((max(free[k], b_done[(r, s)][m]), 2, m, s, "W", None))
                else:
                    cands += [(max(free[k], b_done[(r, s)][m]), 2, m, s, "W", None) for s, m in w_left[k]]
            if fixed is not None:
                want = fixed[k][pos[k]] if pos[k] < len(fixed[k]) else None
                cands = [c for c in cands if want is not None and (c[4], c[2], c[3]) == tuple(want)]
            if not cands:
                continue
            c = min(cands, key=lambda c: (c[0], c[1], c[2], c[3]))
            if best is None or (c[0], c[1], k) < (best[1][0], best[1][1], best[0]):
                best = (k, c)
        if best is None:
            return SimResult(order, ops, math.inf, dict(free), busy, peak, deadlock=True)
        k, (start, _, mb, s, op, ch) = best
        u = (k[0], s)
        dur = cost(u, op, mb)
        end = start + dur
        free[k] = end
        busy[k] += dur
        order[k].append((op, mb, s) if vpp else (op, mb))
        ops.append(SimOp(k, op, mb, start, end, s if vpp else None))
        scheduled += 1
        if pos is not None:
            pos[k] += 1
        if ch is not None:
            head[ch] += 1
        if op == "F":
            if s == 0:
                f_left[u].remove(mb)
            f_done[u][mb] = end
            if s == chunks[k][0]:
                inflight[k] += 1
                peak[k] = max(peak[k], inflight[k])
            if s < S - 1:
                dst = (plan.executor(s + 1, mb), s + 1)
                c2 = (k, phys[dst], "act")
                chan.setdefault(c2, []).append((mb, s + 1, end + comm(u, dst, "act")))
                head.setdefault(c2, 0)
        elif op == "B":
            b_done[u][mb] = end
            if s == chunks[k][0]:
                inflight[k] -= 1
            if split_backward:
                w_left[k].append((s, mb))
            if s > 0:
                dst = (plan.executor(s - 1, mb), s - 1)
                c2 = (k, phys[dst], "grad")
                chan.setdefault(c2, []).append((mb, s - 1, end + comm(u, dst, "grad")))
                head.setdefault(c2, 0)
        else:
            w_left[k].remove((s, mb))

    finish = {k: free[k] + (sum(sync_time((k[0], s)) for s in chunks[k]) if sync_time else 0.0) for k in keys}
    fwd_done = {k: max((t for s in chunks[k] for t in f_done[(k[0], s)].values()), default=0.0) for k in keys}
    return SimResult(order, ops, max(finish.values(), default=0.0) + optimizer_time, finish, busy, peak,
                     stage_forward_done=fwd_done)


def schedule_with_fallback(plan: HeteroParallelPlan, cost: CostFn, comm: CommFn, **kw) -> SimResult:
    """1F1B-limited schedule (for interleaved plans also Megatron's static order, whichever is
    faster); falls back to unlimited in-flight (always deadlock-free) if needed."""
    unlimited = {plan.physical_key(g.replica, g.stage): math.inf for g in plan.stage_groups}
    res = list_schedule(plan, cost, comm, **kw)
    if plan.vpp > 1:
        fixed = megatron_interleaved_order(plan, kw.get("split_backward", False))
        if fixed is not None:
            alt = list_schedule(plan, cost, comm, fixed=fixed, **{**kw, "inflight_limit": unlimited})
            if not alt.deadlock and (res.deadlock or alt.makespan < res.makespan):
                res = alt
    if not res.deadlock:
        return res
    return list_schedule(plan, cost, comm, **{**kw, "inflight_limit": unlimited})
