"""Glue between the coordinator's PlannerFn interface and the progressive HeteroPlanner.

The PREPARE payload carries the new plan, its validated IR schedule, the previous plan
(so every worker can derive the same state-transfer plan locally) and the roll-forward
recovery decision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from resihp.control.coordinator import ClusterView
from resihp.plan.hetero_plan import HeteroParallelPlan
from resihp.plan.ir import schedule_to_json
from resihp.plan.plan_cache import PlanCache
from resihp.plan.planner import HeteroPlanner, PlanInfeasible, PlannerConfig
from resihp.plan.reconfig_cost import estimate_reconfig_cost, gpt_layer_specs
from resihp.plan.topology import ClusterTopology, Device
from resihp.state.transaction import roll_forward_plan
from resihp.utils.log import get_logger

log = get_logger(__name__)


def topology_from_view(view: ClusterView) -> ClusterTopology:
    return ClusterTopology(tuple(
        Device(w.worker_id, w.node_id, w.local_rank, str(w.health.get("rack", "r0")), float(w.health.get("p", 1.0)))
        for w in view.alive))


def make_hetero_planner(cfg: PlannerConfig, use_cache: bool = True):
    planner = HeteroPlanner(cfg)
    cache = PlanCache(lambda p, t: planner.replan(p, t)) if use_cache else None

    def plan_fn(view: ClusterView) -> tuple[list[str], dict]:
        t0 = time.monotonic()
        topo = topology_from_view(view)
        prev = None
        if view.previous_payload and view.previous_payload.get("kind") == "hetero":
            prev = HeteroParallelPlan.from_json(view.previous_payload["plan"])
        restore = None
        hit = False
        if prev is None:
            plan = planner.initial_plan(topo, generation=view.generation)
        else:
            failed = set(prev.world) - set(topo.ids())
            cached = cache.lookup(prev.generation, set(topo.ids()), failed) if cache and failed else None
            if cached is not None and not any(w.health.get("p", 1.0) < 1.0 for w in view.alive):
                plan = replace(cached, generation=view.generation)
                hit = True
            else:
                plan = None
        if prev is not None and plan is None:
            try:
                plan = planner.replan(prev, topo, generation=view.generation)
            except PlanInfeasible as e:
                # Not enough devices to keep every stage alive with the current replica count:
                # rebuild with fewer replicas; state comes from snapshots/checkpoints.
                log.error("replan infeasible (%s); rebuilding with fewer replicas", e)
                slots = sum(len(v) // cfg.tp for v in topo.nodes().values())
                replicas = max(1, slots // cfg.num_stages)
                small = HeteroPlanner(replace(cfg, num_replicas=replicas))
                plan = small.initial_plan(topo, generation=view.generation)
                restore = "snapshot_or_checkpoint"
        schedule, sim = planner.schedule(plan, topo)
        world = list(plan.world)
        rec = roll_forward_plan(view.recovery, world)
        rec["restore"] = restore
        payload = {"kind": "hetero", "plan": plan.to_json(), "schedule": schedule_to_json(schedule),
                   "previous_plan": prev.to_json() if prev else None, "recovery": rec,
                   "predicted_iteration_s": sim.makespan, "nodes": {d.id: d.node for d in topo.devices},
                   "planning_s": time.monotonic() - t0, "plan_cache_hit": hit}
        if cache is not None:
            cache.warm(plan, topo)
        return world, payload

    plan_fn.cache = cache
    return plan_fn


@dataclass(frozen=True)
class WhatIf:
    current_s: float
    mitigated_s: float
    reconfig_s: float
    bytes_moved: int


def make_what_if(cfg: PlannerConfig, **cost_kw):
    """Price a confirmed slowdown for the ski-rental policy: predicted iteration time if we keep
    the committed plan, if we replan around the slow devices, and the one-off reconfiguration cost.
    Times are in the profile's units, so the profile must be calibrated to wall-clock seconds."""
    planner = HeteroPlanner(cfg)
    specs = gpt_layer_specs(cfg.model)

    def what_if(payload: dict, speeds: dict[str, float]) -> WhatIf:
        prev = HeteroParallelPlan.from_json(payload["plan"])
        nodes: dict[str, str] = payload["nodes"]
        index: dict[str, int] = {}
        devices = []
        for d, n in nodes.items():
            devices.append(Device(d, n, index.setdefault(n, 0), "r0", float(speeds.get(d, 1.0))))
            index[n] += 1
        topo = ClusterTopology(tuple(devices))
        _, sim = planner.schedule(prev, topo)
        new = planner.replan(prev, topo)
        est = estimate_reconfig_cost(prev, new, specs, node_of=nodes, **cost_kw)
        return WhatIf(sim.makespan, new.meta["predicted_iteration_s"], est.seconds, est.bytes_moved)

    return what_if
