"""Named planner instances for ``resihp-run --planner module:attr``."""

from resihp.plan.coordinator_adapter import make_hetero_planner
from resihp.plan.dp_planner import make_dp_planner
from resihp.plan.model_spec import ModelProfile, ModelSpec
from resihp.plan.planner import PlannerConfig

dp_8mb = make_dp_planner(num_microbatches=8)

# Reference model of examples/hetero_demo.py (RefModelConfig defaults) on 8 devices: TP2 x PP2 x DP2.
HETERO_8 = PlannerConfig(
    model=ModelSpec(num_layers=4, hidden=8, ffn=24, num_heads=12, num_kv_heads=12, vocab=16, seq_len=24,
                    micro_batch_size=2),
    profile=ModelProfile(alpha=1e-6, beta=1e-9, last_stage_extra_layers=0.0),
    num_stages=2, num_replicas=2, tp=2, num_microbatches=8, sequence_parallel=True, migration_rounds=16)

hetero_8gpu = make_hetero_planner(HETERO_8)
