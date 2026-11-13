import math

import pytest

from resihp.sim.calibrate import PROFILE
from resihp.sim.cluster import (ClusterState, JobConfig, baseline_params, ideal_params, integer_capacity, resihp_params,
                                simulate)
from resihp.sim.traces import LLAMA3_UNEXPECTED, Event, TraceConfig, expected_failures, generate


def test_trace_matches_llama3_statistics():
    assert sum(c for c, _ in LLAMA3_UNEXPECTED.values()) == 419
    assert expected_failures(16384, 54) == pytest.approx(419)
    cfg = TraceConfig(gpus=16384, days=54, slow_per_gpu_day=0.0, seed=3)
    events = generate(cfg)
    assert abs(len(events) - 419) < 4 * math.sqrt(419)
    for e in events:
        assert len(e.targets) == {"gpu": 1, "node": 8, "network": 16, "job": 0}[e.domain]
    assert generate(cfg) == events


def test_integer_capacity():
    assert integer_capacity([1.0] * 4, 16) == 4
    assert integer_capacity([1.0] * 3, 16) == pytest.approx(16 / 6)
    assert integer_capacity([1.0, 1.0, 0.5], 10) == pytest.approx(2.5)
    assert integer_capacity([0.0, 0.0], 8) == 0.0


def small_job(**kw):
    return JobConfig(gpus=256, tp=8, pp=4, iteration_s=10.0, microbatches_per_replica=16, profile=PROFILE, **kw)


def test_single_gpu_loss_costs_resihp_less_than_replica_drop():
    job = small_job()
    lost = {0: math.inf}
    thr = {}
    for p in [resihp_params(job), *baseline_params(job)]:
        st = ClusterState(job, p)
        st.lost.update(lost)
        st.refresh(lost)
        thr[p.name] = st.throughput()
    assert thr["ResiHP"] > thr["torchft-bp"] and thr["ResiHP"] > thr["ReCycle"] - 1e-9
    assert thr["torchft"] == pytest.approx(7 / 8)


def test_simulation_accounting():
    job = small_job()
    horizon = 86400.0
    for p in [ideal_params(), resihp_params(job), *baseline_params(job)]:
        assert simulate([], job, p, horizon).ettr == pytest.approx(1.0)
    fail = [Event(3700.0, "fail", "node", "host_maintenance", tuple(range(8)), 7200.0)]
    res = {p.name: simulate(fail, job, p, horizon) for p in [resihp_params(job), *baseline_params(job)]}
    assert res["ResiHP"].ettr > res["torchft-bp"].ettr > res["NVRx"].ettr
    assert res["NVRx"].lost_s > 0 and res["ResiHP"].lost_s <= job.iteration_s
    spared = simulate(fail, small_job(spare_nodes=1), baseline_params(job)[1], horizon)
    assert spared.degraded_loss_s < res["torchft-bp"].degraded_loss_s
