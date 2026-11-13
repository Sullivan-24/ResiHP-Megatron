"""Calibrate the cluster simulator's ResiHP throughput model against the real planner.

The planner (TP subgroup choice, layer repartition, progress-aware migration) and the
pipeline simulator it shares with the detector predict the degraded iteration time of a
concrete plan. At a scale where that is cheap (TP8 x PP4 x DP4 = 128 GPUs) we measure
three canonical degradations and fit the analytic model's repartition blend ``beta`` and
overhead factor, which ``resihp.sim.cluster`` then applies at 1K-100K GPUs.
"""

from __future__ import annotations

import math

from resihp.plan.model_spec import ModelProfile, ModelSpec
from resihp.plan.planner import HeteroPlanner, PlannerConfig
from resihp.plan.tp_select import feasible_degrees
from resihp.plan.topology import ClusterTopology
from resihp.sim.cluster import ClusterState, JobConfig, resihp_params

MODEL = ModelSpec(num_layers=40, hidden=5120, ffn=13824, num_heads=40, num_kv_heads=40, vocab=32000, seq_len=4096)
PROFILE = ModelProfile(alpha=3.0e-6, beta=1.0e-10)
MB_PER_REPLICA = 16


def measured(tp: int = 8, pp: int = 4, dp: int = 4, microbatches: int = 32) -> dict[str, float]:
    cfg = PlannerConfig(model=MODEL, profile=PROFILE, num_stages=pp, num_replicas=dp, tp=tp,
                        num_microbatches=microbatches, mem_capacity=80e9, migration_rounds=24)
    pl = HeteroPlanner(cfg)
    topo = ClusterTopology.uniform(tp * pp * dp // 8, 8, nodes_per_rack=2)
    plan = pl.initial_plan(topo)
    healthy = plan.meta["predicted_iteration_s"]
    group = plan.group(0, 1).devices
    out = {}
    for name, t in (("lose_gpu", topo.without({group[0]})), ("lose_group", topo.without(set(group))),
                    ("slow_gpu_2x", topo.with_speeds({group[0]: 0.5}))):
        out[name] = healthy / pl.replan(plan, t).meta["predicted_iteration_s"]
    return out


def model(job: JobConfig, beta: float, overhead: float) -> dict[str, float]:
    p = resihp_params(job, beta=beta, overhead=overhead)
    group = list(ClusterState(job, p).group_gpus(0, 1))
    out = {}
    for name in ("lose_gpu", "lose_group", "slow_gpu_2x"):
        st = ClusterState(job, p)
        if name == "lose_gpu":
            st.lost[group[0]] = math.inf
        elif name == "lose_group":
            st.lost.update({g: math.inf for g in group})
        else:
            st.slow[group[0]] = (2.0, math.inf, True)
        st.refresh(group)
        out[name] = st.throughput()
    return out


def calibrate(tp: int = 8, pp: int = 4, dps: tuple[int, ...] = (2, 4, 8)) -> dict:
    """Joint fit over several DP sizes: a single size cannot separate the 1/D (bottleneck) and
    1/(S D) (repartitioned) scaling of the deficit, which is exactly what extrapolation needs."""
    degrees = tuple(feasible_degrees(MODEL, tp))
    meas = {dp: measured(tp, pp, dp, microbatches=MB_PER_REPLICA * dp) for dp in dps}
    jobs = {dp: JobConfig(gpus=tp * pp * dp, tp=tp, pp=pp, min_tp=1, tp_degrees=degrees, profile=PROFILE,
                          microbatches_per_replica=MB_PER_REPLICA) for dp in dps}
    best = None
    for b in range(21):
        for o in range(5, 61):
            beta, overhead = b / 20, o / 20
            preds = {dp: model(jobs[dp], beta, overhead) for dp in dps}
            err = sum((preds[dp][k] - meas[dp][k]) ** 2 for dp in dps for k in meas[dp])
            if best is None or err < best[0]:
                best = (err, beta, overhead, preds)
    err, beta, overhead, preds = best
    n = sum(len(m) for m in meas.values())
    return {"beta": beta, "overhead": overhead, "rmse": math.sqrt(err / n),
            "measured": {str(dp): m for dp, m in meas.items()}, "model": {str(dp): p for dp, p in preds.items()},
            "scale": {"tp": tp, "pp": pp, "dp": list(dps)}, "tp_degrees": list(degrees)}
