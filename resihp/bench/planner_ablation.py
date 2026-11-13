"""Planner ablation: predicted throughput after canonical degradations (healthy = 1).

Variants: full ResiHP; no TP shrink (evict weakened groups); no layer repartition; no
micro-batch migration; and whole-replica exclusion with the global batch preserved
(torchft-style replica drop, every stage of the failed replica evicted).
"""

from __future__ import annotations

from dataclasses import replace

from resihp.plan.planner import HeteroPlanner, PlanInfeasible, PlannerConfig
from resihp.plan.topology import ClusterTopology
from resihp.sim.calibrate import MODEL, PROFILE

VARIANTS = {"ResiHP": {}, "no TP shrink": {"enable_tp_shrink": False}, "no repartition": {"enable_repartition": False},
            "no migration": {"enable_migration": False}}


def run(dps=(2, 4), tp: int = 8, pp: int = 4, mb_per_replica: int = 16) -> list[dict]:
    rows = []
    for dp in dps:
        base = PlannerConfig(model=MODEL, profile=PROFILE, num_stages=pp, num_replicas=dp, tp=tp,
                             num_microbatches=mb_per_replica * dp, mem_capacity=80e9, migration_rounds=32)
        topo = ClusterTopology.uniform(tp * pp * dp // 8, 8, nodes_per_rack=2)
        plan = HeteroPlanner(base).initial_plan(topo)
        healthy = plan.meta["predicted_iteration_s"]
        g = plan.group(0, 1).devices
        replica0 = {d for x in plan.stage_groups if x.replica == 0 for d in x.devices}
        scenarios = {"1 GPU lost": topo.without({g[0]}), "2 GPUs lost (2 replicas)":
                     topo.without({g[0], plan.group(1, 1).devices[0]}), "node lost": topo.without(set(g)),
                     "1 GPU 2x slow": topo.with_speeds({g[0]: 0.5})}
        for name, t in scenarios.items():
            row = {"gpus": tp * pp * dp, "dp": dp, "scenario": name}
            for v, kw in VARIANTS.items():
                try:
                    new = HeteroPlanner(replace(base, **kw)).replan(plan, t)
                    row[v] = healthy / new.meta["predicted_iteration_s"]
                except PlanInfeasible:
                    row[v] = None
            if name != "1 GPU 2x slow":
                lost = set(topo.ids()) - set(t.ids())
                try:
                    drop = HeteroPlanner(base).replan(plan, topo.without(replica0 | lost))
                    row["replica drop"] = healthy / drop.meta["predicted_iteration_s"]
                except PlanInfeasible:
                    row["replica drop"] = None
            else:
                row["replica drop"] = 0.5
            rows.append(row)
    return rows


def to_markdown(rows: list[dict]) -> str:
    cols = [*VARIANTS, "replica drop"]
    lines = ["| GPUs | scenario | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for r in rows:
        cells = ["stops" if r[c] is None else f"{r[c]:.3f}" for c in cols]
        lines.append(f"| {r['gpus']} | {r['scenario']} | " + " | ".join(cells) + " |")
    lines.append("\n`replica drop` for a slow GPU: synchronous DP waits for the slow replica (0.5 at 2x).")
    return "\n".join(lines)
