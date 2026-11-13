"""Replica-level data-parallel runtime (every worker holds the whole model)."""

from __future__ import annotations

import torch
import torch.distributed as dist

from resihp.comm import GenerationComms, ResilientCommManager
from resihp.runtime.optim import Adam
from resihp.runtime.reference_model import RefModelConfig, RefStage, global_tokens, microbatch, token_loss
from resihp.state.rng_data import IndexAddressableSampler


class DataParallelRuntime:
    def __init__(self, cfg: RefModelConfig, sampler: IndexAddressableSampler, model: RefStage,
                 comms: GenerationComms, mgr: ResilientCommManager, assignment: list[int], *,
                 hook=None, progress=None, timeout: float = 120.0):
        self.cfg = cfg
        self.sampler = sampler
        self.model = model
        self.comms = comms
        self.mgr = mgr
        self.assignment = assignment
        self.hook = hook or (lambda it, phase: None)
        self.progress = progress or (lambda **kw: None)
        self.timeout = timeout

    def run_iteration(self, iteration: int) -> float:
        params = list(self.model.named_state().items())
        for _, p in params:
            p.grad = None
        local = 0.0
        for k, mb in enumerate(self.assignment):
            self.hook(iteration, "forward")
            self.progress(iteration=iteration, phase=f"mb{mb}", in_collective=False)
            m = microbatch(self.cfg, self.sampler, iteration, mb)
            loss = token_loss(self.model(m.x), m.y, m.mask)
            loss.backward()
            local += loss.item()
        self.hook(iteration, "reduce")
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for _, p in params]
                         + [torch.tensor([local], dtype=self.cfg.dtype)])
        if self.comms.world_size > 1:
            self.progress(iteration=iteration, phase="grad_allreduce", in_collective=True)
            self.mgr.guarded_wait(dist.all_reduce(flat, async_op=True), timeout=self.timeout, what="grad all-reduce")
        total = global_tokens(self.cfg, self.sampler, iteration)
        off = 0
        for _, p in params:
            n = p.numel()
            p.grad = (flat[off:off + n].view_as(p) / total).clone()
            off += n
        self.hook(iteration, "step")
        return flat[-1].item() / total


def full_state(opt: Adam, committed_iteration: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    return opt.state_tensors(), torch.tensor([committed_iteration, opt.step_count], dtype=torch.int64)


def sync_full_state(mgr: ResilientCommManager, comms: GenerationComms, my_id: str, sources: dict[str, str],
                    opt: Adam, committed_iteration: int, timeout: float = 120.0) -> int | None:
    """Copy model + optimizer state from each receiver's source; returns the adopted iteration if received."""
    tensors, meta = full_state(opt, committed_iteration)
    keys = sorted(tensors)
    works = []
    received = None
    recv_meta = torch.zeros(2, dtype=torch.int64)
    recv_buf: dict[str, torch.Tensor] = {}
    for dst, src in sorted(sources.items()):
        if my_id == src:
            peer = comms.rank_of(dst)
            works.append(dist.isend(meta, peer, tag=0))
            works.extend(dist.isend(tensors[k].contiguous(), peer, tag=i + 1) for i, k in enumerate(keys))
        if my_id == dst:
            peer = comms.rank_of(src)
            recv_buf = {k: torch.empty_like(tensors[k]) for k in keys}
            works.append(dist.irecv(recv_meta, peer, tag=0))
            works.extend(dist.irecv(recv_buf[k], peer, tag=i + 1) for i, k in enumerate(keys))
            received = True
    for w in works:
        mgr.guarded_wait(w, timeout=timeout, what="state sync")
    if received:
        opt.load_state_tensors(recv_buf, int(recv_meta[1].item()))
        return int(recv_meta[0].item())
    return None
