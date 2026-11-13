"""Interleaved (VPP) and zero-bubble schedules in the plan, IR, simulator and planner."""

import pytest

from resihp.plan.hetero_plan import HeteroParallelPlan, StageGroup, interleave
from resihp.plan.ir import compute_order_to_ir, schedule_from_json, schedule_to_json, validate
from resihp.plan.model_spec import ModelProfile, ModelSpec
from resihp.plan.planner import HeteroPlanner, PlannerConfig
from resihp.plan.simulator import schedule_with_fallback
from resihp.plan.topology import ClusterTopology


def pipeline(P=4, layers=16, M=8, replicas=1):
    groups, n = [], 0
    for r in range(replicas):
        for p in range(P):
            groups.append(StageGroup(r, p, (f"r{r}p{p}",), (p * layers // P, (p + 1) * layers // P)))
            n += 1
    world = tuple(sorted(g.devices[0] for g in groups))
    groups = [StageGroup(g.replica, g.stage, g.devices, g.layers) for g in groups]
    assign = tuple(tuple(m * replicas // M for m in range(M)) for _ in range(P))
    return HeteroParallelPlan(1, world, layers, P, replicas, tuple(groups), M, assign)


def run(plan, zb=False):
    def cost(key, op, mb):
        g = plan.group(*key)
        f = g.layers[1] - g.layers[0]
        return {"F": f, "B": f if zb else 2 * f, "W": f}[op]

    res = schedule_with_fallback(plan, cost, lambda a, b, kind: 0.0, split_backward=zb)
    assert not res.deadlock
    ir = compute_order_to_ir(plan, res.order)
    validate(plan, ir, split_backward=zb)
    assert schedule_from_json(schedule_to_json(ir)) == ir
    return res


def test_interleave_round_trip_and_validation():
    plan = pipeline()
    v = interleave(plan, 2)
    v.validate()
    assert v.num_stages == 8 and v.num_physical_stages == 4
    assert [g.layers for g in sorted(v.groups_of_device("r0p1"), key=lambda g: g.stage)] == [(2, 4), (10, 12)]
    assert v.physical().layer_partition() == plan.layer_partition()
    assert HeteroParallelPlan.from_json(v.to_json()) == v
    with pytest.raises(ValueError):
        interleave(pipeline(layers=4), 2)


@pytest.mark.parametrize("zb", [False, True])
def test_interleaving_shrinks_the_bubble(zb):
    flat, inter = run(pipeline(), zb), run(interleave(pipeline(), 2), zb)
    work = sum(op.end - op.start for op in flat.ops) / 4
    assert inter.makespan < flat.makespan
    assert inter.makespan - work < 0.75 * (flat.makespan - work)


def test_zero_bubble_beats_1f1b():
    assert run(pipeline(), zb=True).makespan < run(pipeline(), zb=False).makespan


def test_planner_interleaved_initial_and_replan():
    model = ModelSpec(num_layers=32, hidden=4096, ffn=11008, num_heads=32, num_kv_heads=32, vocab=32000, seq_len=4096)
    cfg = PlannerConfig(model=model, profile=ModelProfile(alpha=3.0e-6, beta=1.0e-10), num_stages=4, num_replicas=2,
                        tp=2, num_microbatches=8, mem_capacity=80e9, migration_rounds=8, vpp=2)
    pl = HeteroPlanner(cfg)
    topo = ClusterTopology.uniform(2, 8)
    plan = pl.initial_plan(topo)
    assert plan.vpp == 2 and plan.num_stages == 8
    pl.schedule(plan, topo)
    victim = plan.group(1, 2).devices[0]
    new = pl.replan(plan, topo.without({victim}))
    assert new.vpp == 2 and victim not in new.world
    pl.schedule(new, topo.without({victim}))
    flat = HeteroPlanner(PlannerConfig(**{**cfg.__dict__, "vpp": 1})).initial_plan(topo)
    assert plan.meta["predicted_iteration_s"] < flat.meta["predicted_iteration_s"]
