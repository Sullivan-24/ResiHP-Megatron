"""Model shape, memory model and compute-cost profile used by the planner and the detector."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    num_layers: int
    hidden: int
    ffn: int
    num_heads: int
    num_kv_heads: int
    vocab: int
    seq_len: int
    micro_batch_size: int = 1
    param_bytes: int = 2
    grad_bytes: int = 4
    optim_bytes: int = 12
    swiglu: bool = True

    def layer_params(self) -> int:
        head_dim = self.hidden // self.num_heads
        attn = self.hidden * self.hidden * 2 + 2 * self.hidden * self.num_kv_heads * head_dim
        mlp = (3 if self.swiglu else 2) * self.hidden * self.ffn
        return attn + mlp + 2 * self.hidden

    def embedding_params(self) -> int:
        return self.vocab * self.hidden

    def tp_feasible(self, k: int) -> bool:
        return k >= 1 and self.num_heads % k == 0 and self.num_kv_heads % k == 0 and self.ffn % k == 0

    def static_bytes(self, layers: int, tp: int, dp_shard: int = 1, extra_params: int = 0) -> float:
        """Weights + fp32 grads + (distributed) optimizer state on one GPU."""
        p = (layers * self.layer_params() + extra_params) / tp
        return p * (self.param_bytes + self.grad_bytes) + p * self.optim_bytes / dp_shard

    def activation_bytes(self, layers: int, tp: int, sequence_parallel: bool = True) -> float:
        """Stored activations per in-flight micro-batch (Korthikanti et al., flash attention, no recompute)."""
        sbh = self.seq_len * self.micro_batch_size * self.hidden
        per_layer = sbh * 34 / (tp if sequence_parallel else 1)
        return layers * per_layer

    def boundary_bytes(self) -> int:
        """Hidden-state tensor crossing a pipeline boundary (bf16)."""
        return self.seq_len * self.micro_batch_size * self.hidden * 2


@dataclass(frozen=True)
class ModelProfile:
    """Per-layer cost at TP=1 on a healthy device: ``T_fwd = alpha * N + beta * sum(l_i^2)``.

    ``alpha``/``beta`` are what the online RLS predictor estimates; the planner uses
    the same model so prediction and planning agree.
    """

    alpha: float
    beta: float
    bwd_ratio: float = 2.0
    w_fraction: float = 0.5
    tp_overhead: float = 0.04
    p2p_bandwidth: float = 25e9
    p2p_latency: float = 20e-6
    intra_node_bandwidth: float = 300e9
    first_stage_extra_layers: float = 0.0
    last_stage_extra_layers: float = 1.0
    grad_sync_overlap: float = 0.8

    def tp_efficiency(self, k: int) -> float:
        return 1.0 / (1.0 + self.tp_overhead * (k - 1))

    def fwd_layer_time(self, tokens: float, sum_sq: float, tp: int, speed: float) -> float:
        return (self.alpha * tokens + self.beta * sum_sq) / (tp * self.tp_efficiency(tp) * max(speed, 1e-9))

    def group_speed(self, tp: int, min_speed: float) -> float:
        """Relative throughput of a TP group (``k * min p`` from the paper, times TP efficiency)."""
        return tp * self.tp_efficiency(tp) * min_speed

    def p2p_time(self, nbytes: float, same_node: bool = False) -> float:
        bw = self.intra_node_bandwidth if same_node else self.p2p_bandwidth
        return self.p2p_latency + nbytes / bw
