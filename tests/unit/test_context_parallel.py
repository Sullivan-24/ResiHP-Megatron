"""Context parallelism in the plan, group derivation, state holdings and planner."""

from resihp.plan.hetero_plan import HeteroParallelPlan, StageGroup, cp_group_name, tp_group_name
from resihp.plan.model_spec import ModelProfile, ModelSpec
from resihp.plan.planner import HeteroPlanner, PlannerConfig
from resihp.plan.topology import ClusterTopology
from resihp.runtime.megatron_runtime import dp_group_name, megatron_group_specs, tpcp_group_name
from resihp.runtime.reference_model import RefModelConfig
from resihp.state.layer_registry import holdings, reference_layer_of, reference_specs


def cp_plan():
    ids = tuple(f"w{i}" for i in range(8))
    groups = (StageGroup(0, 0, ids[0:4], (0, 2), 2), StageGroup(1, 0, ids[4:6], (0, 2), 2),
              StageGroup(0, 1, ids[6:8], (2, 4), 2))
    plan = HeteroParallelPlan(1, ids, 4, 2, 2, groups, 8, ((0, 0, 0, 0, 1, 1, 1, 1), (0,) * 8))
    plan.validate()
    return plan


def test_group_derivation_with_cp():
    plan = cp_plan()
    g = plan.group(0, 0)
    assert g.tp == 2 and g.slice(1) == ("w2", "w3") and g.cp_peers(1) == ("w1", "w3")
    specs = {s.name: s.ranks for s in plan.group_specs((8, 24))}
    assert specs[tp_group_name(0, 0, 0)] == (0, 1) and specs[tp_group_name(0, 0, 1)] == (2, 3)
    assert specs[cp_group_name(0, 0, 0)] == (0, 2) and specs[cp_group_name(1, 0, 0)] == (4, 5)
    # Stage 0: four TP partitions (2 replicas x 2 CP slices); shard 0 of TP2 overlaps both TP1 slices.
    holders = plan.dpshard_holder_sets(0, (8,))
    assert (0, 2, 4, 5) in holders and (1, 3, 4, 5) in holders
    assert plan.dpshard_holder_sets(1, (8,)) == [(6, 7)]
    meg = {s.name: s for s in megatron_group_specs(plan, (8, 24))}
    assert meg[tpcp_group_name(0, 0)].ranks == (0, 1, 2, 3)
    assert meg[tp_group_name(1, 0, 1)].ranks == (5,) and meg[tp_group_name(1, 0, 1)].keep_singleton
    assert meg[dp_group_name(1, 0, 0)].ranks == (6,)
    assert HeteroParallelPlan.from_json(plan.to_json()) == plan


def test_holdings_replicate_tp_partitions_across_cp_slices():
    plan = cp_plan()
    h = holdings(plan, reference_specs(RefModelConfig()), reference_layer_of)
    assert sorted(h["layers.0.fc1.weight"]) == [("w0", (0, 12)), ("w1", (12, 24)), ("w2", (0, 12)),
                                                ("w3", (12, 24)), ("w4", (0, 24)), ("w5", (0, 24))]


def test_planner_keeps_cp_and_shrinks_tp():
    model = ModelSpec(num_layers=32, hidden=4096, ffn=11008, num_heads=32, num_kv_heads=8, vocab=32000, seq_len=32768)
    cfg = PlannerConfig(model=model, profile=ModelProfile(alpha=3.0e-6, beta=1.0e-10), num_stages=2, num_replicas=2,
                        tp=2, num_microbatches=8, migration_rounds=8, cp=2)
    pl = HeteroPlanner(cfg)
    topo = ClusterTopology.uniform(2, 8)
    plan = pl.initial_plan(topo)
    assert all(g.cp == 2 and len(g.devices) == 4 for g in plan.stage_groups)
    victim = plan.group(0, 1).devices[1]
    new = pl.replan(plan, topo.without({victim}))
    g = new.group(0, 1)
    assert victim not in new.world and all(x.cp == 2 for x in new.stage_groups)
    assert g is None or (g.tp in (1, 2) and len(g.devices) == 2 * g.tp)
    pl.schedule(new, topo.without({victim}))
