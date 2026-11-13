"""Reference MoE block and a dropless expert-parallel runtime over arbitrary placements.

``y = x + sum_k g_k(x) * E_{e_k}(x)`` with top-k routing (softmax over the selected
logits) and two-layer GELU experts. Parameters are a pure function of ``(seed, key)``
so every placement sees the same model. The EP runtime dispatches token copies with
an autograd-aware all-to-all over the EP group, runs the local expert copies, returns
the outputs with the reverse all-to-all, and sums each expert's gradient over all of
its holders. Dropless routing plus per-token loss over the global token count make
any placement, with or without replicated experts, equivalent to the single-process
reference up to reduction order.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from resihp.plan.expert_placement import ExpertPlacement, expert_dp_groups, smooth_weighted_round_robin
from resihp.state.layer_registry import FULL, Holdings, TensorSpec
from resihp.state.rng_data import generator
from resihp.utils.intervals import even_split


@dataclass(frozen=True)
class MoEConfig:
    hidden: int = 8
    ffn: int = 12
    num_experts: int = 6
    top_k: int = 2
    tokens: int = 48
    seed: int = 4321
    dtype: torch.dtype = torch.float64
    bias_update_rate: float = 0.0
    """Aux-loss-free balancing (DeepSeek-V3): a non-trainable per-expert bias added to the logits
    for expert *selection* only, moved by ``rate * sign(mean load - load)`` every iteration."""


def moe_specs(cfg: MoEConfig) -> dict[str, TensorSpec]:
    h, f = cfg.hidden, cfg.ffn
    out = {"router.weight": TensorSpec("router.weight", (cfg.num_experts, h), None)}
    if cfg.bias_update_rate:
        out["router.expert_bias"] = TensorSpec("router.expert_bias", (cfg.num_experts,), None, trainable=False)
    for e in range(cfg.num_experts):
        out[f"experts.{e}.fc1.weight"] = TensorSpec(f"experts.{e}.fc1.weight", (f, h), None)
        out[f"experts.{e}.fc2.weight"] = TensorSpec(f"experts.{e}.fc2.weight", (h, f), None)
    return out


def expert_of(key: str) -> int | None:
    return int(key.split(".")[1]) if key.startswith("experts.") else None


def init_param(cfg: MoEConfig, spec: TensorSpec) -> torch.Tensor:
    if not spec.trainable:
        return torch.zeros(spec.global_shape, dtype=cfg.dtype)
    g = generator(cfg.seed, zlib.crc32(spec.key.encode()))
    return torch.randn(spec.global_shape, generator=g, dtype=cfg.dtype) / spec.global_shape[-1] ** 0.5


def update_expert_bias(cfg: MoEConfig, bias: torch.Tensor, loads: torch.Tensor) -> None:
    with torch.no_grad():
        bias.add_(cfg.bias_update_rate * torch.sign(loads.mean() - loads))


def batch(cfg: MoEConfig, iteration: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(cfg.tokens, cfg.hidden, generator=generator(cfg.seed, iteration, 5), dtype=cfg.dtype)
    teacher = torch.randn(cfg.hidden, cfg.hidden, generator=generator(cfg.seed, 3), dtype=cfg.dtype)
    return x, torch.tanh(x @ teacher)


def route(cfg: MoEConfig, x: torch.Tensor, router: torch.Tensor,
          bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    logits = x @ router.t()
    topi = (logits if bias is None else logits + bias).topk(cfg.top_k, dim=-1).indices
    return topi, torch.softmax(logits.gather(-1, topi), dim=-1)


def expert_ffn(x: torch.Tensor, fc1: torch.Tensor, fc2: torch.Tensor) -> torch.Tensor:
    return F.linear(F.gelu(F.linear(x, fc1)), fc2)


def token_loss(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((out - y) ** 2).mean(dim=-1).sum()


def reference_forward(cfg: MoEConfig, params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    topi, gates = route(cfg, x, params["router.weight"], params.get("router.expert_bias"))
    if "router.expert_bias" in params:
        update_expert_bias(cfg, params["router.expert_bias"],
                           torch.bincount(topi.reshape(-1), minlength=cfg.num_experts).to(cfg.dtype))
    combined = torch.zeros_like(x)
    for e in range(cfg.num_experts):
        rows, slot = (topi == e).nonzero(as_tuple=True)
        if rows.numel():
            y = expert_ffn(x[rows], params[f"experts.{e}.fc1.weight"], params[f"experts.{e}.fc2.weight"])
            combined = combined.index_add(0, rows, gates[rows, slot].unsqueeze(1) * y)
    return x + combined


def reference_train(cfg: MoEConfig, iterations: int, lr: float = 1e-2):
    from resihp.runtime.optim import Adam

    specs = moe_specs(cfg)
    params = {k: torch.nn.Parameter(init_param(cfg, s)) for k, s in specs.items() if s.trainable}
    buffers = {k: init_param(cfg, s) for k, s in specs.items() if not s.trainable}
    opt = Adam(params, lr=lr)
    losses = []
    for it in range(iterations):
        opt.zero_grad()
        x, y = batch(cfg, it)
        loss = token_loss(reference_forward(cfg, {**params, **buffers}, x), y)
        (loss / cfg.tokens).backward()
        opt.step()
        losses.append(loss.item() / cfg.tokens)
    return losses, {**{k: v.detach().clone() for k, v in params.items()}, **{k: v.clone() for k, v in buffers.items()}}


def moe_holdings(placement: ExpertPlacement, specs: dict[str, TensorSpec]) -> Holdings:
    devices = sorted(d for g in placement.groups for d in g)
    out: Holdings = {}
    for key in specs:
        e = expert_of(key)
        holders = devices if e is None else placement.holders(e)
        out[key] = [(d, FULL) for d in holders]
    return out


def edp_group_name(holders: tuple[str, ...]) -> str:
    return "edp/" + ",".join(holders)


def moe_group_specs(placement: ExpertPlacement, world: list[str]):
    from resihp.comm.group_spec import GroupSpec

    rank = {d: i for i, d in enumerate(world)}
    specs = [GroupSpec("dp", tuple(range(len(world))), kind="dp")]
    for i, g in enumerate(placement.groups):
        specs.append(GroupSpec(f"ep/{i}", tuple(sorted(rank[d] for d in g)), kind="ep"))
    for holders in expert_dp_groups(placement):
        if len(holders) > 1:
            specs.append(GroupSpec(edp_group_name(holders), tuple(sorted(rank[d] for d in holders)), kind="edp"))
    return specs


def _all_to_all(out: torch.Tensor, inp: torch.Tensor, out_splits, in_splits, group, wait) -> torch.Tensor:
    if group is None:
        out.copy_(inp)
    else:
        wait(dist.all_to_all_single(out, inp.contiguous(), out_splits, in_splits, group=group, async_op=True))
    return out


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group, wait):
        ctx.meta = (out_splits, in_splits, group, wait)
        return _all_to_all(x.new_empty((sum(out_splits),) + tuple(x.shape[1:])), x, out_splits, in_splits, group, wait)

    @staticmethod
    def backward(ctx, g):
        out_splits, in_splits, group, wait = ctx.meta
        grad = g.new_empty((sum(in_splits),) + tuple(g.shape[1:]))
        return _all_to_all(grad, g, in_splits, out_splits, group, wait), None, None, None, None


class EPRuntime:
    """One MoE block on one device: router (replicated) + the expert copies the placement gives it."""

    def __init__(self, cfg: MoEConfig, placement: ExpertPlacement, me: str, world: list[str], comms, mgr,
                 params: dict[str, torch.nn.Parameter], speeds: dict[str, float] | None = None, timeout: float = 60.0,
                 buffers: dict[str, torch.Tensor] | None = None):
        self.cfg, self.placement, self.me, self.world = cfg, placement, me, list(world)
        self.comms, self.mgr, self.timeout = comms, mgr, timeout
        self.speeds = speeds or {}
        self.gidx = placement.group_of(me)
        self.ep = sorted(placement.groups[self.gidx], key=self.world.index)
        self.params = params
        self.buffers = buffers or {}
        self.local_experts = placement.slots[me]
        self.wait = lambda w: mgr.guarded_wait(w, timeout=timeout, what="moe")

    def token_range(self) -> tuple[int, int]:
        return even_split(self.cfg.tokens, len(self.world))[self.world.index(self.me)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg, E = self.cfg, self.cfg.num_experts
        topi, gates = route(cfg, x, self.params["router.weight"], self.buffers.get("router.expert_bias"))
        self.tokens_per_expert = torch.bincount(topi.reshape(-1), minlength=E).to(torch.float64)
        dest: dict[tuple[int, int], str] = {}
        for e in range(E):
            pairs = [tuple(p) for p in (topi == e).nonzero().tolist()]
            if pairs:
                picks = smooth_weighted_round_robin(self.placement.group_holders(self.gidx, e), self.speeds, len(pairs))
                dest.update(zip(pairs, picks))
        order: list[tuple[int, int]] = []
        counts = torch.zeros(len(self.ep), E, dtype=torch.int64)
        for di, d in enumerate(self.ep):
            for e in range(E):
                rows = sorted(p for p, h in dest.items() if h == d and int(topi[p]) == e)
                order += rows
                counts[di, e] = len(rows)
        idx_i = torch.tensor([i for i, _ in order], dtype=torch.long)
        idx_j = torch.tensor([j for _, j in order], dtype=torch.long)
        group = self.comms.group(f"ep/{self.gidx}")
        recv_counts = _all_to_all(torch.empty_like(counts), counts, None, None, group, self.wait)
        in_splits = counts.sum(1).tolist()
        out_splits = recv_counts.sum(1).tolist()
        recv = _AllToAll.apply(x.index_select(0, idx_i), out_splits, in_splits, group, self.wait)
        outs, off = [], 0
        for src in range(len(self.ep)):
            for e in range(E):
                c = int(recv_counts[src, e])
                if c:
                    if e not in self.local_experts:
                        raise RuntimeError(f"{self.me} received tokens for expert {e} it does not host")
                    outs.append(expert_ffn(recv[off:off + c], self.params[f"experts.{e}.fc1.weight"],
                                           self.params[f"experts.{e}.fc2.weight"]))
                    off += c
        y = torch.cat(outs) if outs else recv.new_zeros((0, cfg.hidden))
        back = _AllToAll.apply(y, in_splits, out_splits, group, self.wait)
        combined = torch.zeros_like(x).index_add(0, idx_i, gates[idx_i, idx_j].unsqueeze(1) * back)
        return x + combined

    def global_loads(self) -> list[float]:
        """Tokens routed to each expert in the last forward, summed over all data-parallel ranks."""
        loads = self.tokens_per_expert.clone()
        group = self.comms.group("dp")
        if group is not None:
            self.wait(dist.all_reduce(loads, group=group, async_op=True))
        return loads.tolist()

    def finish_iteration(self) -> None:
        """Aux-loss-free bias update from the global loads (Megatron's ``_update_router_expert_bias``)."""
        bias = self.buffers.get("router.expert_bias")
        if bias is not None:
            update_expert_bias(self.cfg, bias, torch.tensor(self.global_loads(), dtype=bias.dtype))

    def sync_grads(self) -> None:
        def flat(keys):
            return torch.cat([(self.params[k].grad if self.params[k].grad is not None
                               else torch.zeros_like(self.params[k])).reshape(-1) for k in keys])

        def unflat(buf, keys):
            off = 0
            for k in keys:
                p = self.params[k]
                p.grad = buf[off:off + p.numel()].view_as(p).clone()
                off += p.numel()

        jobs = [("dp", ["router.weight"])]
        for holders, experts in sorted(expert_dp_groups(self.placement).items()):
            if self.me in holders and len(holders) > 1:
                keys = [f"experts.{e}.{n}" for e in experts for n in ("fc1.weight", "fc2.weight")]
                jobs.append((edp_group_name(holders), keys))
        for name, keys in jobs:
            group = self.comms.group(name)
            if group is None:
                continue
            buf = flat(keys)
            self.wait(dist.all_reduce(buf, group=group, async_op=True))
            unflat(buf, keys)


def local_keys(placement: ExpertPlacement, me: str, specs: dict[str, TensorSpec]) -> list[str]:
    return [k for k in specs if expert_of(k) is None or expert_of(k) in placement.slots[me]]
