import math
from dataclasses import replace

import pytest

from resihp.plan.cost_model import CostModel, Workload
from resihp.plan.dp_migration import initial_assignment, progress_aware_migration, simulate
from resihp.plan.hetero_plan import HeteroParallelPlan, StageGroup, dpshard_group_name
from resihp.plan.ir import Instr, Op, ScheduleError, compute_order_to_ir, stream_simulation, validate
from resihp.plan.layer_partition import counts_to_ranges, partition_layers
from resihp.plan.model_spec import ModelProfile, ModelSpec
from resihp.plan.placement import place_groups, stage_loss_exposure
from resihp.plan.simulator import list_schedule
from resihp.plan.topology import ClusterTopology
from resihp.plan.tp_select import feasible_degrees, min_tp_for_memory, select_subgroup


def make_plan(tps=((2, 2), (2, 2)), layers=8, M=8, assignment=None):
    """tps[r][s] = TP degree of replica r at stage s; devices numbered in rank order."""
    R, S = len(tps), len(tps[0])
    groups, world, i = [], [], 0
    per = layers // S
    for s in range(S):
        for r in range(R):
            devs = tuple(f"d{i + j}" for j in range(tps[r][s]))
            i += tps[r][s]
            world.extend(devs)
            groups.append(StageGroup(r, s, devs, (s * per, (s + 1) * per)))
    if assignment is None:
        assignment = tuple(tuple(m * R // M for m in range(M)) for _ in range(S))
    return HeteroParallelPlan(1, tuple(world), layers, S, R, tuple(groups), M, assignment)


def test_plan_validation_and_json_roundtrip():
    p = make_plan()
    p.validate()
    assert HeteroParallelPlan.from_json(p.to_json()) == p
    bad = replace(p, assignment=((0,) * 8, (5,) * 8))
    with pytest.raises(ValueError):
        bad.validate()
    with pytest.raises(ValueError):
        replace(p, world=p.world + ("ghost",)).validate()


def test_uniform_tp_dp_groups_match_megatron():
    p = make_plan(tps=((2, 2), (2, 2)))
    holders = p.dpshard_holder_sets(0, (24, 8))
    assert holders == [(0, 2), (1, 3)]


def test_heterogeneous_tp_shard_aligned_groups():
    p = make_plan(tps=((4,), (2,)))
    assert p.dpshard_holder_sets(0, (24,)) == [(0, 4), (1, 4), (2, 5), (3, 5)]
    p3 = make_plan(tps=((4,), (3,)))
    assert p3.dpshard_holder_sets(0, (24,)) == [(0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6)]
    specs = {s.name for s in p3.group_specs((24,))}
    assert "tp/r0/s0" in specs and dpshard_group_name(0, (1, 5)) in specs


def cost_fixed(f=1.0, b=2.0):
    return lambda key, op, mb: {"F": f, "B": b, "W": 0.5}[op]


def test_1f1b_matches_closed_form_and_ir_validates():
    S, M = 4, 8
    p = make_plan(tps=((1,) * S,), layers=8, M=M)
    res = list_schedule(p, cost_fixed(), lambda a, b, k: 0.0)
    assert not res.deadlock
    assert res.makespan == pytest.approx((M + S - 1) * 3.0)
    assert max(res.peak_inflight.values()) <= S
    ir = compute_order_to_ir(p, res.order)
    validate(p, ir)
    fixed = stream_simulation(ir, duration=lambda k, ins: {Op.F: 1.0, Op.B: 2.0}.get(ins.op, 0.0))
    assert fixed.complete and fixed.makespan == pytest.approx(res.makespan)


def test_zero_bubble_split_backward_schedules_w():
    p = make_plan(tps=((1, 1, 1),), layers=6, M=6)
    res = list_schedule(p, cost_fixed(1.0, 1.0), lambda a, b, k: 0.0, split_backward=True)
    ir = compute_order_to_ir(p, res.order)
    validate(p, ir, split_backward=True)
    assert sum(1 for i in ir[(0, 2)] if i.op == Op.W) == 6


def test_validator_catches_fifo_violation_and_deadlock():
    p = make_plan(tps=((1, 1),), layers=4, M=2)
    res = list_schedule(p, cost_fixed(), lambda a, b, k: 0.0)
    ir = compute_order_to_ir(p, res.order)
    s0 = list(ir[(0, 0)])
    sends = [i for i, ins in enumerate(s0) if ins.op == Op.SEND_ACT]
    s0[sends[0]], s0[sends[1]] = s0[sends[1]], s0[sends[0]]
    with pytest.raises(ScheduleError):
        validate(p, {**ir, (0, 0): s0})
    dead = {(0, 0): [Instr(Op.RECV_GRAD, 0, (0, 1)), Instr(Op.SEND_ACT, 0, (0, 1))],
            (0, 1): [Instr(Op.RECV_ACT, 1, (0, 0)), Instr(Op.SEND_GRAD, 0, (0, 0))]}
    assert not stream_simulation(dead).complete


def test_tp_selection_formula():
    model = ModelSpec(num_layers=32, hidden=4096, ffn=12288, num_heads=48, num_kv_heads=12, vocab=32000, seq_len=4096)
    assert feasible_degrees(model, 8) == [1, 2, 3, 4, 6]
    assert feasible_degrees(model, 8, power_of_two_only=True) == [1, 2, 4]
    choice = select_subgroup([("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 0.3)], [1, 2, 4])
    assert choice.tp == 2 and choice.score == pytest.approx(2.0) and set(choice.standby) == {"c", "d"}
    assert select_subgroup([("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 0.9)], [1, 2, 4]).tp == 4
    three = select_subgroup([("a", 1.0), ("b", 1.0), ("c", 1.0)], feasible_degrees(model, 3))
    assert three.tp == 3, "non-power-of-two TP keeps all three surviving GPUs busy"
    assert min_tp_for_memory(model, 32, 80e9, inflight=1) in (1, 2)
    assert min_tp_for_memory(model, 32, 1e9, inflight=1) is None


def test_layer_partition_paper_example_and_caps():
    assert partition_layers(12, [1.0, 0.4, 1.0]) == [5, 2, 5]
    assert partition_layers(12, [1.0, 1.0, 1.0]) == [4, 4, 4]
    assert partition_layers(12, [1.0, 1.0, 1.0], max_layers=[3, 6, 6]) == [3, 5, 4]
    counts = partition_layers(16, [1.0] * 4, last_extra=2.0)
    assert counts[-1] < counts[0]
    assert counts_to_ranges([2, 3]) == [(0, 2), (2, 5)]
    with pytest.raises(ValueError):
        partition_layers(2, [1, 1, 1])


def _cm(plan, speeds):
    model = ModelSpec(num_layers=plan.num_layers, hidden=1024, ffn=4096, num_heads=16, num_kv_heads=16, vocab=1000,
                      seq_len=1024)
    prof = ModelProfile(alpha=1e-6, beta=0.0, last_stage_extra_layers=0.0)
    return CostModel(plan, model, prof, Workload.uniform(plan.num_microbatches, 1024),
                     speeds, {d: d for d in plan.world})


def test_progress_aware_migration_mitigates_fail_slow():
    p = make_plan(tps=((1, 1, 1), (1, 1, 1), (1, 1, 1)), layers=6, M=12)
    speeds = {d: 1.0 for d in p.world}
    slow = p.group(1, 1).devices[0]
    speeds[slow] = 0.5
    cm = _cm(p, speeds)
    res = progress_aware_migration(p, cm, max_rounds=32)
    assert res.makespan < res.initial_makespan * 0.95
    moved = [m for m in range(12) if res.assignment[1][m] != p.assignment[1][m]]
    assert moved and all(p.assignment[1][m] == 1 for m in moved)
    validate(replace(p, assignment=res.assignment), compute_order_to_ir(replace(p, assignment=res.assignment),
                                                                        res.sim.order))


def test_fail_stop_eviction_and_memory_caps():
    full = make_plan(tps=((1, 1, 1), (1, 1, 1), (1, 1, 1)), layers=6, M=12)
    groups = tuple(g for g in full.stage_groups if not (g.replica == 1 and g.stage == 2))
    world = tuple(d for d in full.world if d in {x for g in groups for x in g.devices})
    assign = initial_assignment(list(groups), 3, 12, lambda k: 1.0)
    assert 1 not in assign[2] and assign[0].count(1) == 4
    p = replace(full, stage_groups=groups, world=world, assignment=assign)
    p.validate()
    cm = _cm(p, {d: 1.0 for d in world})
    caps = {(g.replica, g.stage): 4 for g in groups}
    res = progress_aware_migration(p, cm, mem_cap=caps)
    final = replace(p, assignment=res.assignment)
    sim = simulate(final, cm)
    assert all(sim.peak_inflight[k] <= caps[k] for k in caps)
    validate(final, compute_order_to_ir(final, sim.order))


def test_spread_placement_avoids_single_node_stage_loss():
    topo = ClusterTopology.uniform(4, 8, nodes_per_rack=2)
    spread = place_groups(topo, num_replicas=2, num_stages=4, tp=4, strategy="spread")
    mega = place_groups(topo, num_replicas=2, num_stages=4, tp=4, strategy="megatron")
    assert stage_loss_exposure(spread, topo) == 0.0
    assert stage_loss_exposure(mega, topo) == 1.0
    assert len({d for v in spread.values() for d in v}) == 32
