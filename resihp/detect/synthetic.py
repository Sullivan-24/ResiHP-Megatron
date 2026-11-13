"""Synthetic training traces with packed variable-length documents, for detector evaluation.

Iteration time comes from the same pipeline simulator the planner uses, driven by
per-micro-batch work ``alpha * N + beta * sum(l_i^2)``, multiplicative noise, and an
optional injected slowdown of one (replica, stage) group.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from resihp.plan.hetero_plan import HeteroParallelPlan, StageGroup
from resihp.plan.simulator import list_schedule


@dataclass
class SyntheticJob:
    stages: int = 4
    replicas: int = 2
    microbatches: int = 8
    token_budget: int = 32768
    doc_log_mean: float = 6.0
    doc_log_sigma: float = 0.9
    long_doc_prob: float = 0.0
    alpha: float = 3.0e-6
    beta: float = 2.0e-10
    layers_per_stage: int = 8
    bwd_ratio: float = 2.0
    noise: float = 0.015
    p2p_seconds: float = 0.002
    burst_prob: float = 0.04
    burst_len_mean: float = 8.0
    burst_long_doc_prob: float = 0.02

    def plan(self) -> HeteroParallelPlan:
        groups, world = [], []
        for s in range(self.stages):
            for r in range(self.replicas):
                d = f"r{r}s{s}"
                world.append(d)
                groups.append(StageGroup(r, s, (d,), (s, s + 1)))
        rows = tuple(tuple(m * self.replicas // self.microbatches for m in range(self.microbatches))
                     for _ in range(self.stages))
        return HeteroParallelPlan(1, tuple(world), self.stages, self.stages, self.replicas, tuple(groups),
                                  self.microbatches, rows)

    def sample_workload(self, rng: np.random.Generator, long_doc_prob: float | None = None) -> list[list[int]]:
        p_long = self.long_doc_prob if long_doc_prob is None else long_doc_prob
        out = []
        for _ in range(self.microbatches):
            docs, left = [], self.token_budget
            while left > 0:
                if rng.random() < p_long:
                    l = int(rng.integers(self.token_budget // 2, self.token_budget + 1))
                else:
                    l = int(min(max(1, rng.lognormal(self.doc_log_mean, self.doc_log_sigma)), self.token_budget))
                l = min(l, left)
                docs.append(l)
                left -= l
            out.append(docs)
        return out

    def chunk_work(self, docs: list[int], alpha: float | None = None, beta: float | None = None) -> float:
        a = self.alpha if alpha is None else alpha
        b = self.beta if beta is None else beta
        n = sum(docs)
        return self.layers_per_stage * (a * n + b * sum(x * x for x in docs))

    def iteration_time(self, workload: list[list[int]], speeds: dict[tuple[int, int], float] | None = None,
                       rng: np.random.Generator | None = None, alpha: float | None = None,
                       beta: float | None = None, chunk_fn=None) -> tuple[float, dict]:
        """``chunk_fn(docs)`` overrides the forward-chunk cost (the detector passes its predictor)."""
        speeds = speeds or {}
        plan = self.plan()
        fwd = [chunk_fn(d) if chunk_fn is not None else self.chunk_work(d, alpha, beta) for d in workload]
        jitter = {}

        def cost(key, op, mb):
            base = fwd[mb] / speeds.get(key, 1.0)
            if rng is not None:
                j = jitter.setdefault((key, op, mb), float(rng.normal(1.0, self.noise)))
                base *= max(j, 0.5)
            return base if op == "F" else base * self.bwd_ratio

        res = list_schedule(plan, cost, lambda a, b, k: self.p2p_seconds)
        chunks = {}
        for op in res.ops:
            if op.op == "F":
                chunks.setdefault(op.key, []).append((op.mb, op.end - op.start))
        return res.makespan, chunks


def evaluate_detector(job: SyntheticJob, trace: list[dict], mode: str, **detector_kw) -> dict:
    """Run one detector over a trace with an online, gated RLS predictor feeding the simulator.

    ``group`` is the runtime's DetectorService view: one residual detector per (replica, stage)
    group on its own forward-chunk times, which also localizes the slow group.
    """
    from resihp.detect.failslow import FailSlowDetector
    from resihp.detect.predictor import RLSPredictor

    if mode == "group":
        return _evaluate_group(job, trace, **detector_kw)
    det = FailSlowDetector(mode, **detector_kw)
    rls = RLSPredictor(scales=(float(job.token_budget), float(job.token_budget) ** 2, 1.0))
    found = []
    for rec in trace:
        pred = None
        if rls.n >= 16:
            pred, _ = job.iteration_time(
                rec["workload"], chunk_fn=lambda d: max(rls.predict(float(sum(d)), float(sum(x * x for x in d))), 0.0))
        d = det.observe(rec["iteration"], rec["observed"], pred)
        if d is not None:
            found.append(d.iteration)
        for items in rec["chunks"].values():
            for mb, sec in items:
                docs = rec["workload"][mb]
                rls.update(float(sum(docs)), float(sum(x * x for x in docs)), sec)
    return {"validations": det.validations, "detections": found, "filtered": det.filtered, "rls": rls}


def _evaluate_group(job: SyntheticJob, trace: list[dict], **detector_kw) -> dict:
    from resihp.detect.failslow import FailSlowDetector
    from resihp.detect.predictor import RLSPredictor

    rls = RLSPredictor(scales=(float(job.token_budget), float(job.token_budget) ** 2, 1.0))
    dets: dict = {}
    found, groups = [], []
    for rec in trace:
        hit_groups = []
        for key, items in sorted(rec["chunks"].items()):
            feats = [(float(sum(rec["workload"][mb])), float(sum(x * x for x in rec["workload"][mb])), sec)
                     for mb, sec in items]
            observed = sum(sec for *_, sec in feats)
            pred = sum(max(rls.predict(n, s), 1e-12) for n, s, _ in feats) if rls.n >= 16 else None
            det = dets.setdefault(key, FailSlowDetector("residual", **detector_kw))
            if det.observe(rec["iteration"], observed, pred) is not None:
                hit_groups.append(key)
        if hit_groups:
            found.append(rec["iteration"])
            groups.append(hit_groups)
        for items in rec["chunks"].values():
            for mb, sec in items:
                docs = rec["workload"][mb]
                rls.update(float(sum(docs)), float(sum(x * x for x in docs)), sec)
    return {"validations": sum(d.validations for d in dets.values()), "detections": found, "groups": groups,
            "filtered": 0, "rls": rls}


def generate_trace(job: SyntheticJob, iterations: int, *, seed: int = 0, slow_at: int | None = None,
                   slow_group: tuple[int, int] = (0, 1), factor: float = 2.0) -> list[dict]:
    """Data arrives in sticky regimes: occasional bursts of long documents (e.g. a run of long
    code files) shift iteration time for several iterations without any hardware fault --
    exactly what fools detectors that only look at the raw iteration-time series."""
    rng = np.random.default_rng(seed)
    out = []
    burst_left = 0
    for it in range(iterations):
        if burst_left == 0 and rng.random() < job.burst_prob:
            burst_left = 1 + int(rng.geometric(1.0 / job.burst_len_mean))
        in_burst = burst_left > 0
        wl = job.sample_workload(rng, job.burst_long_doc_prob if in_burst else None)
        burst_left = max(0, burst_left - 1)
        speeds = {slow_group: 1.0 / factor} if slow_at is not None and it >= slow_at else {}
        t, chunks = job.iteration_time(wl, speeds, rng)
        out.append({"iteration": it, "observed": t, "workload": wl, "chunks": chunks,
                    "slow": bool(speeds), "burst": in_burst})
    return out
