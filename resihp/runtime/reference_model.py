"""Executable reference model for CPU tests and as the specification of runtime semantics.

A stack of pre-LN MLP blocks, ``x + fc2(gelu(fc1(LN(x))))``, laid out exactly like
Megatron tensor parallelism: fc1 is column-parallel (partitioned on dim 0), fc2 is
row-parallel (partitioned on dim 1), LN and fc2.bias are replicated. With sequence
parallelism the token dimension is sharded between blocks (all-gather before fc1,
reduce-scatter after fc2), which is what makes heterogeneous-TP pipeline
boundaries non-trivial.

Global parameters are a pure function of ``(seed, layer)``, so every TP degree and
every pipeline partition sees the same model; any distributed execution can be
checked against ``reference_train`` bit-for-bit (up to reduction order).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from resihp.state.rng_data import IndexAddressableSampler, derive_seed, generator
from resihp.utils.intervals import even_split

PARAM_DIMS = {"ln.weight": None, "ln.bias": None, "fc1.weight": 0, "fc1.bias": 0, "fc2.weight": 1, "fc2.bias": None}
SP_PARTIAL = {"ln.weight", "ln.bias", "fc2.bias"}


@dataclass(frozen=True)
class RefModelConfig:
    hidden: int = 8
    ffn: int = 24
    num_layers: int = 4
    seq_min: int = 2
    seq_max: int = 9
    token_multiple: int = 12
    seed: int = 1234
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    """Where the model and activations live. Parameters and data are always *generated* on CPU
    from counter-based seeds and then moved, so every device sees the same model and batch."""
    amax_history: int = 0
    """FP8-style delayed scaling: each block scales its fc1 input by ``1 / (1 + 0.1 * max(history))``
    and appends the iteration's global amax to the history (0 disables)."""


def full_layer_params(cfg: RefModelConfig, layer: int) -> dict[str, torch.Tensor]:
    g = generator(cfg.seed, layer, 7)
    h, f = cfg.hidden, cfg.ffn
    return {
        "ln.weight": 1.0 + 0.1 * torch.randn(h, generator=g, dtype=cfg.dtype),
        "ln.bias": 0.1 * torch.randn(h, generator=g, dtype=cfg.dtype),
        "fc1.weight": torch.randn(f, h, generator=g, dtype=cfg.dtype) / h ** 0.5,
        "fc1.bias": 0.1 * torch.randn(f, generator=g, dtype=cfg.dtype),
        "fc2.weight": torch.randn(h, f, generator=g, dtype=cfg.dtype) / f ** 0.5,
        "fc2.bias": 0.1 * torch.randn(h, generator=g, dtype=cfg.dtype),
    }


def shard(t: torch.Tensor, dim: int | None, tp_rank: int, tp_size: int) -> torch.Tensor:
    if dim is None:
        return t.clone()
    s, e = even_split(t.shape[dim], tp_size)[tp_rank]
    return t.narrow(dim, s, e - s).clone()


def sample(cfg: RefModelConfig, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = generator(cfg.seed, index, 11)
    length = cfg.seq_min + int(torch.randint(0, cfg.seq_max - cfg.seq_min + 1, (1,), generator=g).item())
    x = torch.randn(length, cfg.hidden, generator=g, dtype=cfg.dtype)
    teacher = torch.randn(cfg.hidden, cfg.hidden, generator=generator(cfg.seed, 3), dtype=cfg.dtype)
    y = torch.tanh(x @ teacher)
    return x, y


@dataclass
class MicroBatch:
    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    doc_lengths: list[int]

    @property
    def num_tokens(self) -> int:
        return int(self.mask.sum().item())


def microbatch(cfg: RefModelConfig, sampler: IndexAddressableSampler, iteration: int, mb: int) -> MicroBatch:
    """Packed micro-batch: documents concatenated along tokens, padded to ``token_multiple``."""
    xs, ys, lens = [], [], []
    for idx in sampler.microbatch_indices(iteration, mb):
        x, y = sample(cfg, idx)
        xs.append(x)
        ys.append(y)
        lens.append(x.shape[0])
    x, y = torch.cat(xs), torch.cat(ys)
    real = x.shape[0]
    total = -(-real // cfg.token_multiple) * cfg.token_multiple
    pad = total - real
    mask = torch.cat([torch.ones(real, dtype=cfg.dtype), torch.zeros(pad, dtype=cfg.dtype)])
    x = torch.cat([x, torch.zeros(pad, cfg.hidden, dtype=cfg.dtype)])
    y = torch.cat([y, torch.zeros(pad, cfg.hidden, dtype=cfg.dtype)])
    return MicroBatch(x, y, mask, lens)


def global_tokens(cfg: RefModelConfig, sampler: IndexAddressableSampler, iteration: int) -> int:
    return sum(sample(cfg, i)[0].shape[0] for i in sampler.batch_indices(iteration))


def token_loss(out: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((out - y) ** 2).mean(dim=-1) * mask).sum()


# ---------------------------------------------------------------------- tensor parallel plumbing
Waiter = Callable[[object], None]


class TPContext:
    def __init__(self, group=None, rank: int = 0, size: int = 1, sequence_parallel: bool = False,
                 waiter: Waiter | None = None):
        self.group = group
        self.rank = rank
        self.size = size
        self.sequence_parallel = sequence_parallel and size > 1
        self.waiter = waiter or (lambda w: w.wait())

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        if self.size > 1:
            self.waiter(dist.all_reduce(t, group=self.group, async_op=True))
        return t

    def gather_tokens(self, shard_: torch.Tensor) -> torch.Tensor:
        if self.size == 1:
            return shard_
        parts = [torch.empty_like(shard_) for _ in range(self.size)]
        self.waiter(dist.all_gather(parts, shard_.contiguous(), group=self.group, async_op=True))
        return torch.cat(parts)

    def scatter_tokens(self, full: torch.Tensor) -> torch.Tensor:
        if self.size == 1:
            return full
        s, e = even_split(full.shape[0], self.size)[self.rank]
        return full.narrow(0, s, e - s).contiguous()

    def reduce_scatter_tokens(self, full: torch.Tensor) -> torch.Tensor:
        summed = self.all_reduce(full.contiguous().clone())
        return self.scatter_tokens(summed)


class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp):
        ctx.tp = tp
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return ctx.tp.all_reduce(g.contiguous().clone()), None


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp):
        return tp.all_reduce(x.contiguous().clone())

    @staticmethod
    def backward(ctx, g):
        return g, None


class _GatherFromSP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp):
        ctx.tp = tp
        return tp.gather_tokens(x)

    @staticmethod
    def backward(ctx, g):
        return ctx.tp.reduce_scatter_tokens(g), None


class _ReduceScatterToSP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp):
        ctx.tp = tp
        return tp.reduce_scatter_tokens(x)

    @staticmethod
    def backward(ctx, g):
        return ctx.tp.gather_tokens(g.contiguous()), None


class RefBlock(torch.nn.Module):
    def __init__(self, cfg: RefModelConfig, layer: int, tp: TPContext):
        super().__init__()
        self.layer = layer
        self.tp = tp
        full = full_layer_params(cfg, layer)
        self.params = torch.nn.ParameterDict(
            {k.replace(".", "_"): torch.nn.Parameter(shard(v, PARAM_DIMS[k], tp.rank, tp.size).to(cfg.device))
             for k, v in full.items()})
        self.fp8 = cfg.amax_history > 0
        if self.fp8:
            self.register_buffer("amax_history", torch.ones(cfg.amax_history, dtype=cfg.dtype, device=cfg.device))
            self.register_buffer("pending_amax", torch.zeros((), dtype=cfg.dtype, device=cfg.device))

    def p(self, name: str) -> torch.nn.Parameter:
        return self.params[name.replace(".", "_")]

    def finish_iteration(self, amax: torch.Tensor) -> None:
        """Delayed-scaling update with the iteration's global amax (reduced over every rank)."""
        with torch.no_grad():
            self.amax_history.copy_(torch.cat([amax.reshape(1), self.amax_history[:-1]]))
            self.pending_amax.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.layer_norm(x, (x.shape[-1],), self.p("ln.weight"), self.p("ln.bias"))
        if self.tp.sequence_parallel:
            h = _GatherFromSP.apply(h, self.tp)
        elif self.tp.size > 1:
            h = _CopyToTP.apply(h, self.tp)
        if self.fp8:
            h = h / (1.0 + 0.1 * self.amax_history.max())
            with torch.no_grad():
                self.pending_amax.copy_(torch.maximum(self.pending_amax, h.detach().abs().max()))
        a = F.gelu(F.linear(h, self.p("fc1.weight"), self.p("fc1.bias")))
        o = F.linear(a, self.p("fc2.weight"))
        if self.tp.sequence_parallel:
            o = _ReduceScatterToSP.apply(o, self.tp)
        elif self.tp.size > 1:
            o = _ReduceFromTP.apply(o, self.tp)
        return x + o + self.p("fc2.bias")


class RefStage(torch.nn.Module):
    """Layers ``[start, end)`` of the reference model on one TP rank."""

    def __init__(self, cfg: RefModelConfig, start: int, end: int, tp: TPContext | None = None):
        super().__init__()
        self.cfg = cfg
        self.start, self.end = start, end
        self.tp = tp or TPContext()
        self.blocks = torch.nn.ModuleList(RefBlock(cfg, l, self.tp) for l in range(start, end))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x

    def named_state(self) -> dict[str, torch.nn.Parameter]:
        """Stable keys independent of stage boundaries: ``layers.<global layer>.<param>``."""
        return {f"layers.{b.layer}.{k}": b.p(k) for b in self.blocks for k in PARAM_DIMS}

    def named_buffer_state(self) -> dict[str, torch.Tensor]:
        """Persistent non-trainable state that migrates with its layer (FP8 amax history)."""
        return {f"layers.{b.layer}.amax_history": b.amax_history for b in self.blocks if b.fp8}

    def transaction_buffers(self) -> dict[str, torch.Tensor]:
        """Everything an aborted iteration may have mutated and a replay must not see."""
        out = dict(self.named_buffer_state())
        out.update({f"layers.{b.layer}.pending_amax": b.pending_amax for b in self.blocks if b.fp8})
        return out

    @staticmethod
    def param_dim(key: str) -> int | None:
        return PARAM_DIMS[key.split(".", 2)[2]]

    def sp_partial(self, key: str) -> bool:
        return self.tp.sequence_parallel and key.split(".", 2)[2] in SP_PARTIAL


class ChunkedStage(torch.nn.Module):
    """The model chunks of one device under interleaved pipelining (virtual stage -> RefStage),
    exposing the same state interface as a single ``RefStage``."""

    def __init__(self, chunks: dict[int, RefStage]):
        super().__init__()
        self.chunks = dict(sorted(chunks.items()))
        self._chunks = torch.nn.ModuleDict({str(s): m for s, m in self.chunks.items()})
        self.tp = next(iter(self.chunks.values())).tp

    @property
    def blocks(self) -> list[RefBlock]:
        return [b for m in self.chunks.values() for b in m.blocks]

    def named_state(self) -> dict[str, torch.nn.Parameter]:
        return {k: v for m in self.chunks.values() for k, v in m.named_state().items()}

    def named_buffer_state(self) -> dict[str, torch.Tensor]:
        return {k: v for m in self.chunks.values() for k, v in m.named_buffer_state().items()}

    def transaction_buffers(self) -> dict[str, torch.Tensor]:
        return {k: v for m in self.chunks.values() for k, v in m.transaction_buffers().items()}

    param_dim = staticmethod(RefStage.param_dim)

    def sp_partial(self, key: str) -> bool:
        return self.tp.sequence_parallel and key.split(".", 2)[2] in SP_PARTIAL


def reference_train(cfg: RefModelConfig, sampler: IndexAddressableSampler, iterations: int, lr: float = 1e-2,
                    weight_decay: float = 0.0) -> tuple[list[float], dict[str, torch.Tensor]]:
    """Single-process ground truth: full model, full global batch, per-token loss over global tokens."""
    from resihp.runtime.optim import Adam

    model = RefStage(cfg, 0, cfg.num_layers)
    params = model.named_state()
    opt = Adam(params, lr=lr, weight_decay=weight_decay)
    losses = []
    for it in range(iterations):
        opt.zero_grad()
        total = global_tokens(cfg, sampler, it)
        loss_sum = 0.0
        for mb in range(sampler.num_microbatches):
            m = microbatch(cfg, sampler, it, mb)
            l = token_loss(model(m.x), m.y, m.mask)
            (l / total).backward()
            loss_sum += l.item()
        for b in model.blocks:
            if b.fp8:
                b.finish_iteration(b.pending_amax.clone())
        opt.step()
        losses.append(loss_sum / total)
    state = {k: v.detach().clone() for k, v in params.items()}
    state.update({k: v.clone() for k, v in model.named_buffer_state().items()})
    return losses, state


def init_seed(cfg: RefModelConfig) -> int:
    return derive_seed(cfg.seed, 99)
