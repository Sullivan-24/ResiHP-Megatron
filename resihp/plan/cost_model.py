"""Cost functions binding a plan to the model profile, the workload and device speeds."""

from __future__ import annotations

from dataclasses import dataclass

from resihp.plan.hetero_plan import HeteroParallelPlan
from resihp.plan.model_spec import ModelProfile, ModelSpec

Key = tuple[int, int]


@dataclass(frozen=True)
class Workload:
    """Per micro-batch token count ``N`` and attention work ``sum(l_i^2)`` (packed documents)."""

    tokens: tuple[float, ...]
    sum_sq: tuple[float, ...]

    @staticmethod
    def uniform(num_microbatches: int, seq_len: int, micro_batch_size: int = 1) -> "Workload":
        n = float(seq_len * micro_batch_size)
        return Workload((n,) * num_microbatches, (micro_batch_size * float(seq_len) ** 2,) * num_microbatches)

    @staticmethod
    def from_doc_lengths(docs: list[list[int]]) -> "Workload":
        return Workload(tuple(float(sum(d)) for d in docs), tuple(float(sum(x * x for x in d)) for d in docs))


@dataclass
class CostModel:
    plan: HeteroParallelPlan
    model: ModelSpec
    profile: ModelProfile
    workload: Workload
    speeds: dict[str, float]
    node_of: dict[str, str]
    split_backward: bool = False

    def __post_init__(self) -> None:
        self._layers = {s: (b - a) for s, (a, b) in enumerate(self.plan.layer_partition())}
        S = self.plan.num_stages
        self._extra = {s: (self.profile.first_stage_extra_layers if s == 0 else 0.0)
                       + (self.profile.last_stage_extra_layers if s == S - 1 else 0.0) for s in range(S)}

    def group_speed(self, key: Key) -> float:
        g = self.plan.group(*key)
        return self.profile.group_speed(g.tp, min(self.speeds.get(d, 1.0) for d in g.devices)) * g.cp

    def fwd(self, key: Key, mb: int) -> float:
        s = key[1]
        layers = self._layers[s] + self._extra[s]
        work = self.profile.alpha * self.workload.tokens[mb] + self.profile.beta * self.workload.sum_sq[mb]
        return layers * work / max(self.group_speed(key), 1e-12)

    def cost(self, key: Key, op: str, mb: int) -> float:
        f = self.fwd(key, mb)
        b = f * self.profile.bwd_ratio
        if op == "F":
            return f
        if op == "B":
            return b * (1 - self.profile.w_fraction) if self.split_backward else b
        return b * self.profile.w_fraction

    def comm(self, src: Key, dst: Key, kind: str) -> float:
        gs, gd = self.plan.group(*src), self.plan.group(*dst)
        same = {self.node_of.get(d) for d in gs.devices} == {self.node_of.get(d) for d in gd.devices}
        nbytes = self.model.boundary_bytes()
        t = self.profile.p2p_time(nbytes, same_node=same)
        if gs.tp != gd.tp and not self.plan.sequence_parallel:
            t += nbytes / self.profile.intra_node_bandwidth
        return t

    def sync_time(self, key: Key) -> float:
        g = self.plan.group(*key)
        d = len(self.plan.groups_at(key[1]))
        if d < 2:
            return 0.0
        nbytes = self._layers[key[1]] * self.model.layer_params() / g.tp * self.model.grad_bytes
        full = 2 * (d - 1) / d * nbytes / self.profile.p2p_bandwidth
        return full * (1.0 - self.profile.grad_sync_overlap)
