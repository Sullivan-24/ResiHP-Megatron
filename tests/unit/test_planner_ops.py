import time
from dataclasses import replace

from resihp.plan.model_spec import ModelProfile, ModelSpec
from resihp.plan.plan_cache import PlanCache
from resihp.plan.planner import HeteroPlanner, PlannerConfig
from resihp.plan.reconfig_cost import estimate_reconfig_cost, gpt_layer_specs
from resihp.plan.topology import ClusterTopology

MODEL = ModelSpec(num_layers=40, hidden=5120, ffn=13824, num_heads=40, num_kv_heads=40, vocab=32000, seq_len=4096)
PROFILE = ModelProfile(alpha=3.0e-6, beta=1.0e-10)


def planner(**kw):
    cfg = PlannerConfig(model=MODEL, profile=PROFILE, num_stages=4, num_replicas=2, tp=4, num_microbatches=16,
                        mem_capacity=80e9, migration_rounds=24)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return HeteroPlanner(cfg)


def test_plan_cache_hits_single_node_failures():
    pl = planner()
    topo = ClusterTopology.uniform(4, 8, nodes_per_rack=2)
    plan = pl.initial_plan(topo)
    cache = PlanCache(lambda p, t: pl.replan(p, t))
    cache.warm(plan, topo, blocking=True)
    lost = {d.id for d in topo.devices if d.node == "n2"}
    t0 = time.perf_counter()
    hit = cache.lookup(plan.generation, set(topo.ids()) - lost, lost)
    lookup_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    fresh = pl.replan(plan, topo.without(lost))
    replan_s = time.perf_counter() - t0
    assert hit is not None and hit.to_json() == fresh.to_json()
    assert lookup_s < replan_s / 10
    assert cache.lookup(plan.generation, set(topo.ids()) - {"n0g0"}, {"n0g0"}) is None
    assert cache.lookup(plan.generation + 1, set(topo.ids()) - lost, lost) is None
    assert cache.hits == 1 and cache.misses == 2


def test_reconfiguration_cost_reflects_moved_state():
    pl = planner()
    topo = ClusterTopology.uniform(4, 8, nodes_per_rack=2)
    plan = pl.initial_plan(topo)
    specs = gpt_layer_specs(MODEL)
    nodes = {d.id: d.node for d in topo.devices}
    same = estimate_reconfig_cost(plan, plan, specs, node_of=nodes, rebuild_s=1.0)
    assert same.bytes_moved == 0 and same.seconds == 1.0
    victim = plan.group(0, 1).devices[1]
    new = pl.replan(plan, topo.without({victim}))
    est = estimate_reconfig_cost(plan, new, specs, node_of=nodes, rebuild_s=1.0)
    assert est.bytes_moved > 0 and est.seconds > 1.0
    assert est.max_device_bytes <= est.bytes_moved


def test_rejoin_restores_missing_group_and_grow_adds_replica():
    pl = planner()
    topo = ClusterTopology.uniform(4, 8, nodes_per_rack=1)
    plan = pl.initial_plan(topo)
    lost = {d.id for d in topo.devices if d.node == "n1"}
    degraded = pl.replan(plan, topo.without(lost))
    missing = [(r, s) for r in range(2) for s in range(4) if degraded.group(r, s) is None]
    assert missing
    bigger = ClusterTopology.uniform(5, 8, nodes_per_rack=1)
    healed = pl.replan(degraded, bigger.without(lost))
    assert all(healed.group(r, s) is not None for r in range(2) for s in range(4))
    assert {d for r, s in missing for d in healed.group(r, s).devices} <= {d.id for d in bigger.devices if d.node == "n4"}
    assert healed.meta["predicted_iteration_s"] < degraded.meta["predicted_iteration_s"]
    grow = planner(max_replicas=3)
    big = ClusterTopology.uniform(6, 8, nodes_per_rack=1)
    p2 = grow.initial_plan(big)
    grown = grow.replan(p2, big)
    assert grown.num_replicas == 3 and len(grown.world) == 48
    assert grown.meta["predicted_iteration_s"] < p2.meta["predicted_iteration_s"]
