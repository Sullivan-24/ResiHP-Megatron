"""Fail-slow detection accuracy under packed variable-length data (cf. ResiHP Table 5).

Compares Greyhound-style raw-series detection, ResiHP-as-published (raw series +
predicted-time filter), residual-series detection, and the per-group service on
synthetic traces with long-document bursts, for several slowdown factors.
"""

from __future__ import annotations

import statistics

from resihp.detect.synthetic import SyntheticJob, evaluate_detector, generate_trace

MODES = ("greyhound", "paper", "residual", "group")


def run(seeds: int = 20, iterations: int = 120, slow_at: int = 70, factors=(1.3, 1.6, 2.0), horizon: int = 10) -> dict:
    job = SyntheticJob()
    out = {"healthy": {m: 0 for m in MODES}, "slow": {}, "seeds": seeds}
    for s in range(seeds):
        trace = generate_trace(job, iterations, seed=1000 + s)
        for m in MODES:
            out["healthy"][m] += evaluate_detector(job, trace, m)["validations"]
    for f in factors:
        stats = {m: {"detected": 0, "false_alarms": 0, "latency": [], "localized": 0} for m in MODES}
        for s in range(seeds):
            trace = generate_trace(job, iterations, seed=2000 + s, slow_at=slow_at, factor=f)
            for m in MODES:
                res = evaluate_detector(job, trace, m)
                det = res["detections"]
                pre = [d for d in det if d < slow_at]
                post = [d for d in det if slow_at <= d <= slow_at + horizon]
                stats[m]["false_alarms"] += len(pre)
                if post:
                    stats[m]["detected"] += 1
                    stats[m]["latency"].append(post[0] - slow_at)
                    if m == "group":
                        first = res["groups"][det.index(post[0])]
                        stats[m]["localized"] += int(first == [(0, 1)])
        out["slow"][str(f)] = stats
    return out


def to_markdown(res: dict) -> str:
    seeds = res["seeds"]
    lines = ["| detector | false alarms (healthy traces) | slowdown | detected | false alarms before onset | "
             "mean latency (iters) | localized |", "|---|---|---|---|---|---|---|"]
    for f, stats in res["slow"].items():
        for m in MODES:
            st = stats[m]
            lat = f"{statistics.fmean(st['latency']):.1f}" if st["latency"] else "-"
            loc = f"{st['localized']}/{st['detected']}" if m == "group" else "n/a"
            lines.append(f"| {m} | {res['healthy'][m]} | {f}x | {st['detected']}/{seeds} | {st['false_alarms']} | "
                         f"{lat} | {loc} |")
    return "\n".join(lines)
