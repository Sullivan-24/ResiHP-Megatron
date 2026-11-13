"""Coordinator planner for the 8-GPU Megatron demo: nominal TP2 x PP2 x DP2.

Used as ``resihp-run local --planner examples.megatron.planner_8gpu:gpt_8gpu``. The model and
cost profile must match the arguments given to ``resilient_pretrain.py``; the planner only
needs shapes (to size TP groups and memory) and a relative cost model (to repartition layers
and migrate micro-batches).
"""

from resihp.plan.coordinator_adapter import make_hetero_planner
from resihp.plan.model_spec import ModelProfile, ModelSpec
from resihp.plan.planner import PlannerConfig

MODEL = ModelSpec(num_layers=8, hidden=512, ffn=2048, num_heads=8, num_kv_heads=8, vocab=1024,
                  seq_len=256, micro_batch_size=1)
PROFILE = ModelProfile(alpha=3.0e-6, beta=1.0e-10, last_stage_extra_layers=1.0)
CONFIG = PlannerConfig(model=MODEL, profile=PROFILE, num_stages=2, num_replicas=2, tp=2,
                       num_microbatches=8, mem_capacity=90e9, migration_rounds=16,
                       sequence_parallel=False)

gpt_8gpu = make_hetero_planner(CONFIG)
