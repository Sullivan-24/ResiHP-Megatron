from resihp.control.coordinator import ClusterView
from resihp.control.membership import WorkerRecord
from resihp.plan.coordinator_adapter import make_hetero_planner
from resihp.plan.hetero_plan import HeteroParallelPlan
from resihp.plan.ir import schedule_from_json, validate
from resihp.plan.model_spec import ModelProfile, ModelSpec
from resihp.plan.planner import PlannerConfig

MODEL = ModelSpec(num_layers=8, hidden=1024, ffn=4096, num_heads=16, num_kv_heads=16, vocab=1000, seq_len=512)
PROFILE = ModelProfile(alpha=3e-6, beta=1e-10)


def records(nodes=3, gpus=4, dead=()):
    return [WorkerRecord(f"n{n}g{g}", f"n{n}", g) for n in range(nodes) for g in range(gpus) if f"n{n}g{g}" not in dead]


def test_initial_then_failure_replan_payloads():
    fn = make_hetero_planner(PlannerConfig(model=MODEL, profile=PROFILE, num_stages=2, num_replicas=2, tp=2,
                                           num_microbatches=8))
    world, payload = fn(ClusterView(1, records(), [], [], None, {}, "start"))
    plan = HeteroParallelPlan.from_json(payload["plan"])
    assert plan.generation == 1 and len(world) == 8 and len(plan.standby) == 4
    validate(plan, schedule_from_json(payload["schedule"]))
    victim = plan.group(0, 1).devices[0]
    rec = {w: {"committed_iteration": 5} for w in world if w != victim}
    world2, payload2 = fn(ClusterView(2, records(dead={victim}), [victim], world, payload, rec, "failure"))
    plan2 = HeteroParallelPlan.from_json(payload2["plan"])
    assert victim not in world2 and plan2.generation == 2
    assert payload2["previous_plan"] == payload["plan"]
    assert payload2["recovery"]["target_iteration"] == 5
    validate(plan2, schedule_from_json(payload2["schedule"]))
    g = plan2.group(0, 1)
    assert g is not None and victim not in g.devices
