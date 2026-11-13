"""Replica-level planner: the whole-replica exclusion baseline (what torchft offers),
but with the global batch preserved by redistributing the lost replica's micro-batches.
"""

from __future__ import annotations

from resihp.control.coordinator import ClusterView
from resihp.state.transaction import roll_forward_plan


def weighted_assignment(workers: list[str], num_microbatches: int, weights: dict[str, float]) -> dict[str, list[int]]:
    """Contiguous micro-batch ranges proportional to speed (largest-remainder rounding)."""
    total = sum(max(weights.get(w, 1.0), 1e-9) for w in workers)
    quotas = [num_microbatches * max(weights.get(w, 1.0), 1e-9) / total for w in workers]
    counts = [int(q) for q in quotas]
    for i in sorted(range(len(workers)), key=lambda i: (-(quotas[i] - counts[i]), i))[: num_microbatches - sum(counts)]:
        counts[i] += 1
    out, start = {}, 0
    for w, c in zip(workers, counts):
        out[w] = list(range(start, start + c))
        start += c
    return out


def assign_sources(needs: list[str], providers: list[str], node_of: dict[str, str]) -> dict[str, str]:
    """Pick a provider per receiver: same node first, then round-robin to spread the load."""
    if not providers:
        return {}
    load = {p: 0 for p in providers}
    out = {}
    for w in needs:
        local = [p for p in providers if node_of.get(p) == node_of.get(w)]
        pool = local or providers
        src = min(pool, key=lambda p: (load[p], providers.index(p)))
        load[src] += 1
        out[w] = src
    return out


def make_dp_planner(num_microbatches: int, min_speed: float = 0.0):
    def planner(view: ClusterView) -> tuple[list[str], dict]:
        alive = view.alive
        world = [w.worker_id for w in alive if w.health.get("p", 1.0) >= min_speed] or [w.worker_id for w in alive]
        weights = {w.worker_id: float(w.health.get("p", 1.0)) for w in alive}
        node_of = {w.worker_id: w.node_id for w in alive}
        rec = roll_forward_plan(view.recovery, world)
        restore = None
        if rec["target_iteration"] >= 0 and not rec["up_to_date"] and not rec["local_step"]:
            restore = "checkpoint"
        rec["sources"] = assign_sources(rec["needs_state"], rec["up_to_date"] + rec["local_step"], node_of)
        rec["restore"] = restore
        payload = {"kind": "dp", "num_microbatches": num_microbatches,
                   "assignment": weighted_assignment(world, num_microbatches, weights), "recovery": rec}
        return world, payload

    return planner
