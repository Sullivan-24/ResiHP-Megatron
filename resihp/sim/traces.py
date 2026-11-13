"""Failure traces calibrated to published production statistics.

Fail-stop rates come from Llama 3 405B pre-training (Meta, "The Llama 3 Herd of Models",
2024, Table 5): 419 unexpected interruptions in a 54-day window on 16,384 H100 GPUs,
plus 47 planned ones. Every root-cause category is mapped to a failure domain:

* ``gpu``: one GPU is lost until its node is serviced;
* ``node``: all GPUs of a node (host, NIC, PSU, chassis, ...);
* ``network``: the nodes behind one cable/switch domain;
* ``job``: no hardware is lost but every process must restart (software bugs,
  dependencies, NCCL watchdog timeouts).

Counts scale linearly with the number of GPUs (per-GPU Poisson rates). The paper does not
report repair times or fail-slow statistics; those defaults are assumptions, exposed as
parameters, and swept in the sensitivity study.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

LLAMA3_GPUS = 16384
LLAMA3_DAYS = 54.0
LLAMA3_PLANNED = 47
LLAMA3_UNEXPECTED: dict[str, tuple[int, str]] = {
    "faulty_gpu": (148, "gpu"),
    "gpu_hbm3_memory": (72, "gpu"),
    "software_bug": (54, "job"),
    "network_switch_cable": (35, "network"),
    "host_maintenance": (32, "node"),
    "gpu_sram_memory": (19, "gpu"),
    "gpu_system_processor": (17, "gpu"),
    "nic": (7, "node"),
    "nccl_watchdog_timeout": (7, "job"),
    "silent_data_corruption": (6, "gpu"),
    "gpu_thermal_interface_sensor": (6, "gpu"),
    "ssd": (3, "node"),
    "power_supply": (3, "node"),
    "server_chassis": (2, "node"),
    "io_expansion_board": (2, "node"),
    "dependency": (2, "job"),
    "cpu": (2, "node"),
    "system_memory": (2, "node"),
}


@dataclass(frozen=True)
class Event:
    t: float
    kind: str  # "fail" | "slow" | "planned"
    domain: str  # "gpu" | "node" | "network" | "job"
    category: str
    targets: tuple[int, ...]  # GPU indices affected (empty for job-wide events)
    duration_s: float  # repair time (fail) or slowdown duration (slow)
    factor: float = 1.0  # slowdown factor of a slow event


@dataclass
class TraceConfig:
    gpus: int
    days: float = LLAMA3_DAYS
    gpus_per_node: int = 8
    rate_scale: float = 1.0
    network_domain_nodes: int = 2
    repair_median_s: dict[str, float] = field(default_factory=lambda: {"gpu": 24 * 3600.0, "node": 6 * 3600.0,
                                                                         "network": 2 * 3600.0, "job": 0.0})
    repair_sigma: float = 1.0
    slow_per_gpu_day: float = 2 * sum(c for c, d in LLAMA3_UNEXPECTED.values() if d == "gpu") / (LLAMA3_GPUS * LLAMA3_DAYS)
    slow_factor_median: float = 1.6
    slow_factor_sigma: float = 0.35
    slow_duration_median_s: float = 1800.0
    slow_duration_sigma: float = 1.5
    include_planned: bool = False
    seed: int = 0

    @property
    def nodes(self) -> int:
        return self.gpus // self.gpus_per_node


def per_gpu_day_rates() -> dict[str, float]:
    return {cat: count / (LLAMA3_GPUS * LLAMA3_DAYS) for cat, (count, _) in LLAMA3_UNEXPECTED.items()}


def expected_failures(gpus: int, days: float) -> float:
    return sum(per_gpu_day_rates().values()) * gpus * days


def _lognormal(rng: random.Random, median: float, sigma: float) -> float:
    return median * math.exp(rng.gauss(0.0, sigma)) if median > 0 else 0.0


def generate(cfg: TraceConfig) -> list[Event]:
    rng = random.Random(cfg.seed)
    horizon = cfg.days * 86400.0
    gpn = cfg.gpus_per_node
    out: list[Event] = []

    def poisson_times(rate_per_s: float) -> list[float]:
        t, ts = 0.0, []
        if rate_per_s <= 0:
            return ts
        while True:
            t += rng.expovariate(rate_per_s)
            if t >= horizon:
                return ts
            ts.append(t)

    for cat, rate in per_gpu_day_rates().items():
        domain = LLAMA3_UNEXPECTED[cat][1]
        for t in poisson_times(rate * cfg.rate_scale * cfg.gpus / 86400.0):
            if domain == "gpu":
                targets = (rng.randrange(cfg.gpus),)
            elif domain == "node":
                n = rng.randrange(cfg.nodes)
                targets = tuple(range(n * gpn, (n + 1) * gpn))
            elif domain == "network":
                k = max(1, min(cfg.network_domain_nodes, cfg.nodes))
                first = rng.randrange(cfg.nodes - k + 1)
                targets = tuple(range(first * gpn, (first + k) * gpn))
            else:
                targets = ()
            out.append(Event(t, "fail", domain, cat, targets,
                             _lognormal(rng, cfg.repair_median_s[domain], cfg.repair_sigma)))
    for t in poisson_times(cfg.slow_per_gpu_day * cfg.gpus / 86400.0):
        factor = min(4.0, max(1.1, _lognormal(rng, cfg.slow_factor_median, cfg.slow_factor_sigma)))
        out.append(Event(t, "slow", "gpu", "fail_slow", (rng.randrange(cfg.gpus),),
                         _lognormal(rng, cfg.slow_duration_median_s, cfg.slow_duration_sigma), factor))
    if cfg.include_planned:
        for t in poisson_times(LLAMA3_PLANNED / (LLAMA3_GPUS * LLAMA3_DAYS) * cfg.gpus / 86400.0):
            out.append(Event(t, "planned", "job", "planned_maintenance", (), 0.0))
    out.sort(key=lambda e: e.t)
    return out


def summarize(events: list[Event], cfg: TraceConfig) -> dict:
    fails = [e for e in events if e.kind == "fail"]
    by_domain: dict[str, int] = {}
    for e in fails:
        by_domain[e.domain] = by_domain.get(e.domain, 0) + 1
    hours = cfg.days * 24.0
    return {"gpus": cfg.gpus, "days": cfg.days, "fail_stop": len(fails), "fail_slow": sum(e.kind == "slow" for e in events),
            "by_domain": by_domain, "mtbf_h": hours / len(fails) if fails else math.inf,
            "expected_fail_stop": expected_failures(cfg.gpus, cfg.days) * cfg.rate_scale}
