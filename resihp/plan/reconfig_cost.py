"""Reconfiguration cost: what a plan transition costs before it pays off.

The estimate replays the slice-level transfer plan between two plans (the same one
the workers will execute), so it accounts for local reuse, striping across replicas
and locality, and adds the communicator rebuild time. It is the ``buy`` price for
the ski-rental policy and the tie-breaker between candidate plans.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from resihp.plan.hetero_plan import HeteroParallelPlan
from resihp.plan.model_spec import ModelSpec
from resihp.state.layer_registry import TensorSpec, holdings
from resihp.state.transfer_plan import plan_transfers


@dataclass(frozen=True)
class ReconfigEstimate:
    seconds: float
    bytes_moved: int
    max_device_bytes: int
    rebuild_s: float


def gpt_layer_specs(model: ModelSpec) -> dict[str, TensorSpec]:
    """Per-layer tensors of a Megatron GPT layer with their TP partition dims."""
    h, f = model.hidden, model.ffn
    head = h // model.num_heads
    qkv = h + 2 * model.num_kv_heads * head
    fc1 = (2 if model.swiglu else 1) * f
    out = {}
    for l in range(model.num_layers):
        for name, shape, dim in (("attn.qkv.weight", (qkv, h), 0), ("attn.proj.weight", (h, h), 1),
                                 ("mlp.fc1.weight", (fc1, h), 0), ("mlp.fc2.weight", (h, f), 1),
                                 ("ln1.weight", (h,), None), ("ln2.weight", (h,), None)):
            key = f"layers.{l}.{name}"
            out[key] = TensorSpec(key, shape, dim)
    return out


def layer_of_key(key: str) -> int:
    return int(key.split(".")[1])


def estimate_reconfig_cost(old: HeteroParallelPlan | None, new: HeteroParallelPlan, specs: dict[str, TensorSpec], *,
                           bytes_per_elem: int = 14, inter_bw: float = 25e9, intra_bw: float = 300e9,
                           rebuild_s: float = 2.0, node_of: dict[str, str] | None = None) -> ReconfigEstimate:
    """``bytes_per_elem`` defaults to bf16 weights + fp32 master + two fp32 Adam moments."""
    if old is None:
        return ReconfigEstimate(rebuild_s, 0, 0, rebuild_s)
    node_of = node_of or {}
    valid = set(old.world) & set(new.world)
    tp = plan_transfers(holdings(old, specs, layer_of_key), holdings(new, specs, layer_of_key),
                        valid_sources=valid, node_of=node_of)
    per_dev_time: dict[str, float] = defaultdict(float)
    per_dev_bytes: dict[str, int] = defaultdict(int)
    total = 0
    for t in tp.transfers:
        n = 1
        for d in specs[t.key].local_shape(t.interval):
            n *= d
        b = n * bytes_per_elem
        same = node_of.get(t.src) is not None and node_of.get(t.src) == node_of.get(t.dst)
        secs = b / (intra_bw if same else inter_bw)
        for dev in (t.src, t.dst):
            per_dev_time[dev] += secs
            per_dev_bytes[dev] += b
        total += b
    transfer_s = max(per_dev_time.values(), default=0.0)
    return ReconfigEstimate(rebuild_s + transfer_s, total, max(per_dev_bytes.values(), default=0), rebuild_s)
