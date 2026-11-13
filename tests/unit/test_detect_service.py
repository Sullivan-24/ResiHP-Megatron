import random

from resihp.detect.service import DetectorService
from resihp.plan.coordinator_adapter import WhatIf, make_what_if
from resihp.plan.model_spec import ModelProfile, ModelSpec
from resihp.plan.planner import HeteroPlanner, PlannerConfig
from resihp.plan.topology import ClusterTopology


class FakeCoord:
    payload = {}

    def __init__(self):
        self.health, self.replans = {}, []

    def set_health(self, d, **h):
        self.health.setdefault(d, {}).update(h)

    def request_replan(self, reason):
        self.replans.append(reason)


def drive(svc, slow_from=40, iterations=80, factor=1.6):
    rng = random.Random(0)
    groups = [("a0", "a1"), ("b0", "b1")]
    for it in range(iterations):
        for g in groups:
            samples = []
            for _ in range(4):
                n = 4096
                s = sum(rng.randint(100, 2000) ** 2 for _ in range(4))
                sec = (2e-6 * n + 1e-10 * s + 1e-3) * rng.lognormvariate(0, 0.01)
                if g == groups[1] and it >= slow_from:
                    sec *= factor
                samples.append((n, s, sec))
            svc._process(g[0], {"kind": "timing", "devices": list(g), "layers": 1, "tp": 1, "samples": samples,
                                "iteration": it, "gen": 1})


def service(what_if):
    svc = DetectorService(FakeCoord(), warmup=6, min_increase=0.3, what_if=what_if)
    svc.stop()
    return svc


def test_rent_until_excess_covers_reconfiguration_cost():
    svc = service(lambda payload, speeds: WhatIf(1.6, 1.1, 3.0, 0))
    drive(svc)
    acted = [e for e in svc.events if e["act"]]
    assert len(acted) == 1 and svc.coord.replans == ["failslow:b0,b1"]
    first = svc.events[0]["iteration"]
    assert first >= 40 and acted[0]["iteration"] == first + 5
    assert svc.coord.health["b0"]["p"] < 0.75


def test_unmitigable_slowdown_never_reconfigures():
    svc = service(lambda payload, speeds: WhatIf(1.6, 1.6, 3.0, 0))
    drive(svc)
    assert svc.events and not any(e["act"] for e in svc.events)
    assert svc.coord.replans == []


def test_what_if_prices_slow_device():
    model = ModelSpec(num_layers=40, hidden=5120, ffn=13824, num_heads=40, num_kv_heads=40, vocab=32000, seq_len=4096)
    cfg = PlannerConfig(model=model, profile=ModelProfile(alpha=3.0e-6, beta=1.0e-10), num_stages=4, num_replicas=2,
                        tp=4, num_microbatches=16, mem_capacity=80e9, migration_rounds=24)
    topo = ClusterTopology.uniform(4, 8, nodes_per_rack=2)
    plan = HeteroPlanner(cfg).initial_plan(topo)
    payload = {"plan": plan.to_json(), "nodes": {d.id: d.node for d in topo.devices}}
    healthy = make_what_if(cfg)(payload, {})
    victim = plan.group(1, 2).devices[0]
    slow = make_what_if(cfg, rebuild_s=1.0)(payload, {victim: 0.5})
    assert abs(healthy.current_s - plan.meta["predicted_iteration_s"]) < 1e-9
    assert slow.current_s > slow.mitigated_s > healthy.current_s
    assert slow.reconfig_s > 1.0 and slow.bytes_moved > 0
