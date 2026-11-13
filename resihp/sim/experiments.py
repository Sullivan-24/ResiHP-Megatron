"""Cluster-scale experiments: ETTR across scales and policies, loss breakdown, sensitivity.

The job is a 405B-class dense model trained with TP8 x PP16 (128 GPUs per replica), 16
micro-batches per replica (Llama 3: 16M-token global batch of 8K sequences on 16K GPUs),
on the Llama 3 failure trace scaled to the cluster size.
"""

from __future__ import annotations

import statistics
from dataclasses import replace

from resihp.sim.calibrate import PROFILE
from resihp.sim.cluster import JobConfig, baseline_params, ideal_params, resihp_params, simulate
from resihp.sim.traces import TraceConfig, generate, summarize

DEFAULT_SCALES = (1024, 4096, 16384, 32768, 65536)


def job_for(gpus: int, **kw) -> JobConfig:
    return JobConfig(gpus=gpus, tp=8, pp=16, iteration_s=10.0, microbatches_per_replica=16, min_tp=4,
                     tp_degrees=(1, 2, 4, 8), profile=PROFILE, **kw)


def policies(job: JobConfig, calib: dict, *, restart_s: float = 300.0, checkpoint_interval_s: float = 1200.0):
    return [ideal_params(), resihp_params(job, beta=calib["beta"], overhead=calib["overhead"]),
            *baseline_params(job, restart_s=restart_s, checkpoint_interval_s=checkpoint_interval_s)]


def run_cell(gpus: int, days: float, seeds: int, calib: dict, trace_kw=None, policy_kw=None,
             spare_frac: float = 0.0) -> dict:
    trace_kw, policy_kw = trace_kw or {}, policy_kw or {}
    job = job_for(gpus, spare_nodes=int(round(spare_frac * gpus / 8)))
    rows: dict[str, list[dict]] = {}
    traces = []
    for seed in range(seeds):
        cfg = TraceConfig(gpus=gpus, days=days, seed=seed, **trace_kw)
        events = generate(cfg)
        traces.append(summarize(events, cfg))
        for p in policies(job, calib, **policy_kw):
            rows.setdefault(p.name, []).append(simulate(events, job, p, days * 86400.0).to_json())
    out = {"gpus": gpus, "days": days, "seeds": seeds, "trace": traces[0] if traces else {},
           "fail_stop_mean": statistics.fmean(t["fail_stop"] for t in traces), "policies": {}}
    for name, rs in rows.items():
        agg = {k: statistics.fmean(r[k] for r in rs) for k in rs[0] if isinstance(rs[0][k], (int, float))}
        agg["ettr_std"] = statistics.pstdev(r["ettr"] for r in rs)
        out["policies"][name] = agg
    return out


def scale_sweep(calib: dict, scales=DEFAULT_SCALES, days: float = 30.0, seeds: int = 5) -> list[dict]:
    return [run_cell(g, days, seeds, calib) for g in scales]


def sensitivity(calib: dict, gpus: int = 16384, days: float = 30.0, seeds: int = 3) -> dict:
    base = TraceConfig(gpus=gpus)
    out = {"restart_s": {}, "gpu_repair_h": {}, "fail_rate_x": {}, "slow_rate_x": {}}
    for r in (300.0, 600.0, 1200.0, 1800.0):
        out["restart_s"][str(int(r))] = run_cell(gpus, days, seeds, calib, policy_kw={"restart_s": r})["policies"]
    for h in (6.0, 24.0, 72.0):
        repair = dict(base.repair_median_s, gpu=h * 3600.0)
        out["gpu_repair_h"][str(int(h))] = run_cell(gpus, days, seeds, calib,
                                                    trace_kw={"repair_median_s": repair})["policies"]
    for x in (0.5, 1.0, 2.0, 4.0):
        out["fail_rate_x"][str(x)] = run_cell(gpus, days, seeds, calib, trace_kw={"rate_scale": x})["policies"]
    for x in (0.0, 1.0, 4.0):
        out["slow_rate_x"][str(x)] = run_cell(gpus, days, seeds, calib,
                                              trace_kw={"slow_per_gpu_day": base.slow_per_gpu_day * x})["policies"]
    return out


def scale_table(cells: list[dict]) -> str:
    names = list(cells[0]["policies"])
    head = "| policy | " + " | ".join(f"{c['gpus']:,} GPUs" for c in cells) + " |"
    lines = [head, "|---|" + "---|" * len(cells)]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(f"{c['policies'][n]['ettr'] * 100:.1f}%" for c in cells) + " |")
    lines.append("| fail-stop events | " + " | ".join(f"{c['fail_stop_mean']:.0f}" for c in cells) + " |")
    return "\n".join(lines)


def breakdown_table(cell: dict) -> str:
    lines = [f"| policy ({cell['gpus']:,} GPUs, {cell['days']:.0f} days) | ETTR | downtime | rollback | degraded throughput | "
             "mean MTTR (s) | p99 MTTR (s) |", "|---|---|---|---|---|---|---|"]
    for n, a in cell["policies"].items():
        lines.append(f"| {n} | {a['ettr'] * 100:.2f}% | {a['downtime_frac'] * 100:.2f}% | {a['lost_frac'] * 100:.2f}% | "
                     f"{a['degraded_frac'] * 100:.2f}% | {a['mttr_mean_s']:.0f} | {a['mttr_p99_s']:.0f} |")
    return "\n".join(lines)


def sensitivity_table(sens: dict) -> str:
    out = []
    for knob, cells in sens.items():
        names = list(next(iter(cells.values())))
        out.append(f"\n**{knob}**\n")
        out.append("| policy | " + " | ".join(cells) + " |")
        out.append("|---|" + "---|" * len(cells))
        for n in names:
            out.append(f"| {n} | " + " | ".join(f"{cells[k][n]['ettr'] * 100:.1f}%" for k in cells) + " |")
    return "\n".join(out)
