from dataclasses import replace

import pytest

from resihp.plan.model_spec import ModelProfile, ModelSpec
from resihp.plan.planner import HeteroPlanner, PlanInfeasible, PlannerConfig
from resihp.plan.topology import ClusterTopology

LLAMA13B = ModelSpec(num_layers=40, hidden=5120, ffn=13824, num_heads=40, num_kv_heads=40, vocab=32000, seq_len=4096)
# ~0.62 GFLOP forward per token per layer and 4*l^2*h attention FLOPs at ~200 TFLOP/s.
PROFILE = ModelProfile(alpha=3.0e-6, beta=1.0e-10, last_stage_extra_layers=1.0)


def planner(**kw):
    cfg = PlannerConfig(model=LLAMA13B, profile=PROFILE, num_stages=4, num_replicas=2, tp=4, num_microbatches=16,
                        mem_capacity=80e9, migration_rounds=48)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return HeteroPlanner(cfg)


def topo():
    return ClusterTopology.uniform(4, 8, nodes_per_rack=2)


def test_initial_plan_is_valid_and_schedulable():
    pl = planner()
    t = topo()
    plan = pl.initial_plan(t)
    counts = plan.meta["layer_counts"]
    assert len(plan.world) == 32 and sum(counts) == 40 and counts[-1] <= counts[0]
    ir, sim = pl.schedule(plan, t)
    assert not sim.deadlock and plan.meta["predicted_iteration_s"] > 0


def test_single_gpu_failure_is_no_worse_than_any_ablation():
    pl = planner()
    t = topo()
    plan = pl.initial_plan(t)
    victim = plan.group(0, 1).devices[1]
    t2 = t.without({victim})
    new = pl.replan(plan, t2)
    g = new.group(0, 1)
    # Keep the group at TP2 or evict its stage work to the peer replica: whichever the joint
    # decision predicts faster; either way stage 1 sheds layers and replica 1 takes its micro-batches.
    assert g is None or (g.tp == 2 and victim not in g.devices)
    counts = new.meta["layer_counts"]
    assert counts[1] < max(counts), "the weakened stage gets fewer layers"
    assert any(new.assignment[1][m] == 1 and plan.assignment[1][m] == 0 for m in range(16)), "stage-level migration"
    pl.schedule(new, t2)
    resihp = new.meta["predicted_iteration_s"]
    for kw in ({"enable_tp_shrink": False}, {"enable_repartition": False}, {"enable_migration": False}):
        assert resihp <= planner(**kw).replan(plan, t2).meta["predicted_iteration_s"] + 1e-12, kw
    assert resihp < planner(enable_repartition=False).replan(plan, t2).meta["predicted_iteration_s"] * 0.95


def test_tp_shrink_keeps_training_where_exclusion_cannot():
    pl = planner()
    t = topo()
    plan = pl.initial_plan(t)
    victims = {plan.group(0, 1).devices[1], plan.group(1, 1).devices[2]}
    t2 = t.without(victims)
    new = pl.replan(plan, t2)
    assert all(new.group(r, 1) is not None and new.group(r, 1).tp == 2 for r in range(2))
    pl.schedule(new, t2)
    with pytest.raises(PlanInfeasible):
        planner(enable_tp_shrink=False).replan(plan, t2)
    single = planner(num_replicas=1)
    one = single.initial_plan(ClusterTopology.uniform(2, 8))
    lost = one.group(0, 2).devices[0]
    shrunk = single.replan(one, ClusterTopology.uniform(2, 8).without({lost}))
    assert shrunk.group(0, 2).tp == 2
    with pytest.raises(PlanInfeasible):
        planner(num_replicas=1, enable_tp_shrink=False).replan(one, ClusterTopology.uniform(2, 8).without({lost}))


def test_node_failure_evicts_stage_work_to_surviving_replicas():
    pl = planner()
    t = topo()
    plan = pl.initial_plan(t)
    lost = {d.id for d in t.devices if d.node == "n1"}
    new = pl.replan(plan, t.without(lost))
    new.validate()
    assert not set(new.world) & lost
    missing = [(r, s) for r in range(2) for s in range(4) if new.group(r, s) is None]
    assert missing, "spread placement put two groups on n1"
    for r, s in missing:
        assert r not in new.assignment[s]
    pl.schedule(new, t.without(lost))


def test_whole_group_rebuilt_on_spare_node():
    pl = planner()
    t = ClusterTopology.uniform(5, 8, nodes_per_rack=1)
    plan = pl.initial_plan(t)
    spare = {d.id for d in t.devices if d.id not in plan.world}
    assert len(spare) == 8
    victims = set(plan.group(1, 2).devices)
    new = pl.replan(plan, t.without(victims))
    g = new.group(1, 2)
    assert g is not None and g.tp == 4 and set(g.devices) <= spare


def test_fail_slow_device_and_infeasible_case():
    pl = planner()
    t = topo()
    plan = pl.initial_plan(t)
    slow = plan.group(1, 2).devices[0]
    t2 = t.with_speeds({slow: 0.3})
    new = pl.replan(plan, t2)
    g = new.group(1, 2)
    assert (g is None or slow not in g.devices) and slow in new.standby
    assert new.meta["predicted_iteration_s"] < pl.replan(plan, t2).meta["unmigrated_iteration_s"] * 1.0001
    tiny = ClusterTopology.uniform(1, 4)
    pl2 = HeteroPlanner(replace(pl.cfg, num_replicas=1, num_stages=1))
    p1 = pl2.initial_plan(tiny)
    with pytest.raises(PlanInfeasible):
        pl2.replan(p1, tiny.without(set(p1.world)))
