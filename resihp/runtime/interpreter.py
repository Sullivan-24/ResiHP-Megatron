"""Worker-side interpreter of the pipeline IR (one stream per (replica, stage) group).

Compute goes through a ``StageExecutor`` (the reference model on CPU, Megatron's
``forward_step``/``backward_step`` on GPU); communication goes through ``P2PEngine``
with heterogeneous-TP routing. Sends are posted non-blocking and waited at the end of
the iteration, which is what the IR validator assumes.
"""

from __future__ import annotations

from typing import Protocol

import torch

from resihp.comm.hetero_p2p import P2PEngine, local_rows
from resihp.plan.hetero_plan import HeteroParallelPlan
from resihp.plan.ir import Instr, Op
from resihp.runtime.reference_model import MicroBatch, RefModelConfig, RefStage, microbatch, token_loss
from resihp.state.rng_data import IndexAddressableSampler
from resihp.utils.intervals import even_split


class StageExecutor(Protocol):
    """``stage`` is the virtual stage (model chunk) of the instruction."""

    def begin_iteration(self, iteration: int) -> None: ...
    def n_rows(self, mb: int) -> int: ...
    def forward(self, mb: int, inp: torch.Tensor | None, stage: int) -> torch.Tensor | None: ...
    def backward(self, mb: int, grad: torch.Tensor | None, stage: int) -> torch.Tensor | None: ...
    def weight_backward(self, mb: int, stage: int) -> None: ...


class RefStageExecutor:
    def __init__(self, cfg: RefModelConfig, sampler: IndexAddressableSampler, module, plan: HeteroParallelPlan,
                 replica: int, stage: int, rank: int, weight_grad_store=None, timer=None):
        self.cfg = cfg
        self.sampler = sampler
        self.module = module
        self.modules: dict[int, RefStage] = getattr(module, "chunks", None) or {stage: module}
        self.plan = plan
        self.group = plan.group(replica, stage)
        self.cp_index = self.group.slice_of(plan.world[rank])
        self.ranks = plan.slice_ranks(self.group, self.cp_index)
        self.rank = rank
        self.num_stages = plan.num_stages
        self.sp = plan.sequence_parallel and self.group.tp > 1
        self.iteration = -1
        self.saved: dict[int, tuple] = {}
        self.loss_sum = 0.0
        self._mb_cache: dict[int, MicroBatch] = {}
        self.wgs = weight_grad_store
        self.timer = timer
        self.fwd_docs: dict[int, list[int]] = {}

    def begin_iteration(self, iteration: int) -> None:
        self.iteration = iteration
        self.saved.clear()
        self._mb_cache.clear()
        self.loss_sum = 0.0
        self.fwd_docs = {}

    def timing_samples(self) -> list[tuple[float, float, float]]:
        """(tokens, sum of squared document lengths, forward seconds over all local chunks) per micro-batch."""
        if self.timer is None:
            return []
        seconds: dict[int, float] = {}
        for rec in self.timer.collect():
            if rec.op == "F" and rec.iteration == self.iteration and rec.mb in self.fwd_docs:
                seconds[rec.mb] = seconds.get(rec.mb, 0.0) + rec.seconds
        return [(float(sum(self.fwd_docs[mb])), float(sum(d * d for d in self.fwd_docs[mb])), sec)
                for mb, sec in seconds.items()]

    def _timed(self, op: str, mb: int):
        return self.timer.time(self.iteration, op, mb) if self.timer is not None else _nullcontext()

    def _mb(self, mb: int) -> MicroBatch:
        if mb not in self._mb_cache:
            m = microbatch(self.cfg, self.sampler, self.iteration, mb)
            if self.cfg.device != "cpu":
                m = MicroBatch(m.x.to(self.cfg.device), m.y.to(self.cfg.device), m.mask.to(self.cfg.device),
                               m.doc_lengths)
            self._mb_cache[mb] = m
        return self._mb_cache[mb]

    def _chunk(self, mb: int) -> tuple[int, int]:
        """Token rows of this rank's context-parallel chunk of the micro-batch."""
        return even_split(self._mb(mb).x.shape[0], self.group.cp)[self.cp_index]

    def n_rows(self, mb: int) -> int:
        a, b = self._chunk(mb)
        return b - a

    def _rows(self, mb: int) -> tuple[int, int]:
        a, b = self._chunk(mb)
        if not self.sp:
            return a, b
        s, e = local_rows(b - a, self.ranks, self.rank)
        return a + s, a + e

    def forward(self, mb: int, inp: torch.Tensor | None, stage: int) -> torch.Tensor | None:
        first, last = stage == 0, stage == self.num_stages - 1
        s, e = self._rows(mb)
        if first:
            x = self._mb(mb).x[s:e].clone()
        else:
            x = inp.detach().requires_grad_(True)
        with self._timed("F", mb):
            out = self.modules[stage](x)
            if last:
                m = self._mb(mb)
                loss = token_loss(out, m.y[s:e], m.mask[s:e])
        self.fwd_docs[mb] = self._mb(mb).doc_lengths
        if last:
            self.loss_sum += loss.item()
            self.saved[(stage, mb)] = (x, loss)
            return None
        self.saved[(stage, mb)] = (x, out)
        return out.detach()

    def backward(self, mb: int, grad: torch.Tensor | None, stage: int) -> torch.Tensor | None:
        first, last = stage == 0, stage == self.num_stages - 1
        x, out = self.saved.pop((stage, mb))
        if self.wgs is not None and hasattr(self.wgs, "backward_input"):
            with self._timed("B", mb):
                return self.wgs.backward_input((stage, mb), out, None if last else grad, None if first else x)
        ctx = self.wgs.deferring((stage, mb)) if self.wgs is not None else _nullcontext()
        with ctx, self._timed("B", mb):
            if last:
                out.backward()
            else:
                torch.autograd.backward(out, grad)
        return None if first else x.grad

    def weight_backward(self, mb: int, stage: int) -> None:
        if self.wgs is not None:
            self.wgs.compute((stage, mb))


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def message_tag(mb: int, kind: str, dst_stage: int) -> int:
    return (mb * 2 + (0 if kind == "act" else 1)) * 256 + dst_stage


class Interpreter:
    def __init__(self, plan: HeteroParallelPlan, stream: list[Instr], executor: StageExecutor, p2p: P2PEngine,
                 replica: int, stage: int, row_shape: tuple[int, ...], dtype: torch.dtype, tp_group=None,
                 progress=None, hook=None):
        self.plan = plan
        self.stream = stream
        self.executor = executor
        self.p2p = p2p
        self.key = (replica, stage)
        self.group = plan.group(replica, stage)
        # Activations stay inside a context-parallel chunk: slice i only talks to slice i.
        self.cp_index = self.group.slice_of(plan.world[p2p.rank])
        self.ranks = plan.slice_ranks(self.group, self.cp_index)
        self.row_shape = row_shape
        self.dtype = dtype
        self.tp_group = tp_group
        self.sp = plan.sequence_parallel
        self.progress = progress or (lambda **kw: None)
        self.hook = hook or (lambda it, phase: None)

    def peer_ranks(self, peer: tuple[int, int]) -> tuple[int, ...]:
        return self.plan.slice_ranks(self.plan.group(*peer), self.cp_index)

    def run(self, iteration: int) -> None:
        ex = self.executor
        ex.begin_iteration(iteration)
        acts: dict[int, torch.Tensor] = {}
        grads: dict[int, torch.Tensor] = {}
        outs: dict[int, torch.Tensor] = {}
        igrads: dict[int, torch.Tensor] = {}
        sends: list = []
        for i, ins in enumerate(self.stream):
            self.progress(iteration=iteration, phase=f"{ins.op.value}{ins.mb}", op_seq=i,
                          in_collective=ins.op in (Op.RECV_ACT, Op.RECV_GRAD))
            mb = ins.mb
            s = ins.stage_on(self.key)
            k = (s, mb)
            if ins.op == Op.RECV_ACT:
                src = self.peer_ranks(ins.peer)
                acts[k] = self.p2p.recv(ex.n_rows(mb), self.row_shape, self.dtype, src, self.ranks,
                                        self.sp, message_tag(mb, "act", s), self.tp_group)
            elif ins.op == Op.F:
                self.hook(iteration, "forward")
                out = ex.forward(mb, acts.pop(k, None), s)
                if out is not None:
                    outs[k] = out
            elif ins.op == Op.SEND_ACT:
                sends += self.p2p.send(outs.pop(k), self.ranks, self.peer_ranks(ins.peer), ex.n_rows(mb), self.sp,
                                       message_tag(mb, "act", ins.peer[1]))
            elif ins.op == Op.RECV_GRAD:
                src = self.peer_ranks(ins.peer)
                grads[k] = self.p2p.recv(ex.n_rows(mb), self.row_shape, self.dtype, src, self.ranks,
                                         self.sp, message_tag(mb, "grad", s), self.tp_group)
            elif ins.op == Op.B:
                self.hook(iteration, "backward")
                g = ex.backward(mb, grads.pop(k, None), s)
                if g is not None:
                    igrads[k] = g
            elif ins.op == Op.SEND_GRAD:
                sends += self.p2p.send(igrads.pop(k), self.ranks, self.peer_ranks(ins.peer), ex.n_rows(mb), self.sp,
                                       message_tag(mb, "grad", ins.peer[1]))
            elif ins.op == Op.W:
                ex.weight_backward(mb, s)
        self.p2p.wait_sends(sends)
