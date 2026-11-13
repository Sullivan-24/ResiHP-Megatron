"""Shard-aligned gradient reduction across DP replicas whose TP degrees differ.

For a stage, each replica splits a parameter dimension of length ``n`` evenly by its
own TP degree. On the common refinement of those splits every interval is held by
exactly one rank per replica; those holders all-reduce exactly that slice. With
uniform TP this is Megatron's DP all-reduce; with TP 4 vs 2 (or 4 vs 3) it is the
reduction the InternEvo prototype got wrong by pairing ranks by TP index.

TP-replicated parameters are reduced through *responsibility* slices (rank ``t`` of a
TP-``k`` replica reduces ``split(numel, k)[t]``) and re-assembled with one all-gather in
the TP group; under sequence parallelism their gradients are partial and are first
summed inside the TP group, as Megatron does for LayerNorm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from resihp.plan.hetero_plan import HeteroParallelPlan, dpshard_group_name
from resihp.utils.intervals import even_split, locate, refine


@dataclass(frozen=True)
class ParamShard:
    key: str
    dim: int | None
    full_size: int
    sp_partial: bool = False


class HeteroGradSync:
    def __init__(self, plan: HeteroParallelPlan, comms, mgr, stage: int, replica: int, params: dict[str, torch.nn.Parameter],
                 shards: dict[str, ParamShard], tp_group=None, timeout: float = 120.0):
        self.plan = plan
        self.comms = comms
        self.mgr = mgr
        self.stage = stage
        self.replica = replica
        self.params = params
        self.shards = shards
        self.tp_group = tp_group
        self.timeout = timeout
        g = plan.group(replica, stage)
        self.my_rank = comms.rank
        me = plan.world[self.my_rank]
        self.tp = g.tp
        self.tp_rank = g.slice(g.slice_of(me)).index(me)
        self.peers = plan.tp_slices(stage)
        self._layout = self._build_layout()

    def _build_layout(self) -> dict[str, list[tuple[str, tuple[int, int], tuple[int, int]]]]:
        """group name -> [(param key, global interval, local interval)] in deterministic order."""
        layout: dict[str, list] = {}
        for key in sorted(self.params):
            info = self.shards[key]
            n = info.full_size
            parts = [even_split(n, len(sl)) for sl in self.peers]
            mine = even_split(n, self.tp)[self.tp_rank]
            # Partitioned tensors start at this rank's shard; replicated ones hold the whole parameter.
            base = mine[0] if info.dim is not None else 0
            for iv in refine(parts):
                if not (mine[0] <= iv[0] and iv[1] <= mine[1]):
                    continue
                holders = tuple(sorted(self.plan.rank(sl[locate(p, iv)]) for sl, p in zip(self.peers, parts)))
                local = (iv[0] - base, iv[1] - base)
                layout.setdefault(dpshard_group_name(self.stage, holders), []).append((key, iv, local))
        return dict(sorted(layout.items()))

    def _flat_view(self, key: str) -> torch.Tensor:
        """The tensor whose dim-0 slices the layout refers to (partitioned dim moved first, or flattened)."""
        g = self.params[key].grad
        info = self.shards[key]
        if info.dim is None:
            return g.reshape(-1)
        return g.movedim(info.dim, 0)

    def sync(self) -> None:
        for key, p in self.params.items():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        if self.tp > 1:
            partial = [k for k in sorted(self.params) if self.shards[k].sp_partial]
            if partial:
                flat = torch.cat([self.params[k].grad.reshape(-1) for k in partial])
                self.mgr.guarded_wait(dist.all_reduce(flat, group=self.tp_group, async_op=True), timeout=self.timeout,
                                      what="tp partial-grad all-reduce")
                off = 0
                for k in partial:
                    n = self.params[k].grad.numel()
                    self.params[k].grad.copy_(flat[off:off + n].view_as(self.params[k].grad))
                    off += n
        for name, entries in self._layout.items():
            pg = self.comms.groups.get(name)
            if pg is None:
                continue
            views = [self._flat_view(k).narrow(0, loc[0], loc[1] - loc[0]) for k, _, loc in entries]
            flat = torch.cat([v.reshape(-1) for v in views])
            self.mgr.guarded_wait(dist.all_reduce(flat, group=pg, async_op=True), timeout=self.timeout,
                                  what=f"grad all-reduce {name}")
            off = 0
            for v in views:
                v.copy_(flat[off:off + v.numel()].view_as(v))
                off += v.numel()
        if self.tp > 1:
            self._regather_replicated()

    def _regather_replicated(self) -> None:
        """Replicated params: only this rank's responsibility slice was reduced; all-gather the rest."""
        keys = [k for k in sorted(self.params) if self.shards[k].dim is None]
        for k in keys:
            flat = self.params[k].grad.reshape(-1)
            parts = even_split(flat.numel(), self.tp)
            width = max(e - s for s, e in parts)
            s, e = parts[self.tp_rank]
            mine = torch.zeros(width, dtype=flat.dtype, device=flat.device)
            mine[: e - s].copy_(flat[s:e])
            out = [torch.empty_like(mine) for _ in range(self.tp)]
            self.mgr.guarded_wait(dist.all_gather(out, mine, group=self.tp_group, async_op=True), timeout=self.timeout,
                                  what="replicated grad all-gather")
            flat.copy_(torch.cat([o[: pe - ps] for o, (ps, pe) in zip(out, parts)]))

    def holder_groups(self) -> list[str]:
        return list(self._layout)
