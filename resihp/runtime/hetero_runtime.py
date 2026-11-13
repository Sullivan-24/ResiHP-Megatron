"""Dynamic hybrid-parallel runtime for one generation: heterogeneous TP, uneven PP, stage-level migration."""

from __future__ import annotations

import torch
import torch.distributed as dist

from resihp.comm import GenerationComms, GroupSpec, ResilientCommManager
from resihp.comm.hetero_p2p import P2PEngine, boundary_routes, channel_name
from resihp.plan.hetero_plan import HeteroParallelPlan
from resihp.plan.ir import Op, Schedule
from resihp.runtime.hetero_grad_sync import HeteroGradSync, ParamShard
from resihp.runtime.interpreter import Interpreter, RefStageExecutor
from resihp.runtime.reference_model import ChunkedStage, RefModelConfig, RefStage, TPContext, global_tokens
from resihp.runtime.weight_grad_store import AutogradWeightGradStore
from resihp.state.rng_data import IndexAddressableSampler


def reference_shards(cfg: RefModelConfig, stage_module: RefStage) -> dict[str, ParamShard]:
    out = {}
    full = {"ln.weight": cfg.hidden, "ln.bias": cfg.hidden, "fc1.weight": cfg.ffn, "fc1.bias": cfg.ffn,
            "fc2.weight": cfg.ffn, "fc2.bias": cfg.hidden}
    for key in stage_module.named_state():
        local = key.split(".", 2)[2]
        out[key] = ParamShard(key, RefStage.param_dim(key), full[local], stage_module.sp_partial(key))
    return out


def reference_partition_sizes(cfg: RefModelConfig) -> tuple[int, ...]:
    return tuple(sorted({cfg.hidden, cfg.ffn}))


def channel_specs(plan: HeteroParallelPlan, schedule: Schedule, n_rows_max: int,
                  row_counts=None) -> list[GroupSpec]:
    """One 2-rank group per communicating (src -> dst) rank pair and direction (NCCL safety).

    Routes depend on the row count, so the union over every count a micro-batch can have
    (``row_counts``, default 1..``n_rows_max``; a fixed sequence length for Megatron) is created.
    """
    counts = list(row_counts) if row_counts is not None else range(1, n_rows_max + 1)
    pairs: set[tuple[int, int]] = set()
    for (r, s), stream in schedule.items():
        src = plan.group(r, s)
        for ins in stream:
            if ins.op not in (Op.SEND_ACT, Op.SEND_GRAD):
                continue
            dst = plan.group(*ins.peer)
            for i in range(src.cp):
                for n in counts:
                    for route in boundary_routes(plan.slice_ranks(src, i), plan.slice_ranks(dst, i), n):
                        pairs.add((route.src, route.dst))
    return [GroupSpec(channel_name(a, b), tuple(sorted((a, b))), "p2p") for a, b in sorted(pairs)]


def channel_map(comms: GenerationComms) -> dict[tuple[int, int], object]:
    out = {}
    for name in comms.specs:
        if name.startswith("p2p/") and comms.is_member(name):
            a, b = (int(x) for x in name[4:].split(">"))
            out[(a, b)] = comms.group(name)
    return out


class HeteroRuntime:
    def __init__(self, cfg: RefModelConfig, sampler: IndexAddressableSampler, plan: HeteroParallelPlan, schedule: Schedule,
                 comms: GenerationComms, mgr: ResilientCommManager, module: RefStage, *, use_channels: bool = False,
                 progress=None, hook=None, timeout: float = 120.0, weight_grad_store=None, timer=None,
                 reporter=None):
        self.cfg = cfg
        self.sampler = sampler
        self.plan = plan
        self.comms = comms
        self.mgr = mgr
        self.module = module
        me = comms.world[comms.rank]
        g = plan.group_of_device(me)
        self.replica, self.stage = g.replica, g.stage
        self.group = g
        tp_group = comms.group(plan.tp_name(g, me)) if g.tp > 1 else None
        self.tp_group = tp_group
        self.timeout = timeout
        p2p = P2PEngine(comms.rank, mgr, channels=channel_map(comms) if use_channels else None, timeout=timeout,
                        device=cfg.device)
        if weight_grad_store is None and any(ins.op == Op.W for ins in schedule[(g.replica, g.stage)]):
            weight_grad_store = AutogradWeightGradStore(lambda: list(module.parameters()))
        self.executor = RefStageExecutor(cfg, sampler, module, plan, g.replica, g.stage, comms.rank,
                                         weight_grad_store=weight_grad_store, timer=timer)
        self.reporter = reporter
        self.timer = timer
        self.interp = Interpreter(plan, schedule[(g.replica, g.stage)], self.executor, p2p, g.replica, g.stage,
                                  (cfg.hidden,), cfg.dtype, tp_group=tp_group, progress=progress, hook=hook)
        self.params = module.named_state()
        self.syncs = [HeteroGradSync(plan, comms, mgr, s, g.replica, m.named_state(), reference_shards(cfg, m),
                                     tp_group=tp_group, timeout=timeout)
                      for s, m in self.executor.modules.items()]
        self.local_layers = sum(m.end - m.start for m in self.executor.modules.values())

    def run_iteration(self, iteration: int) -> float:
        for p in self.params.values():
            p.grad = None
        self.interp.run(iteration)
        for sync in self.syncs:
            sync.sync()
        self._finish_delayed_scaling()
        self.interp.hook(iteration, "loss")
        total = global_tokens(self.cfg, self.sampler, iteration)
        for p in self.params.values():
            p.grad.div_(total)
        last = self.plan.num_stages - 1 in self.executor.modules
        share = self.executor.loss_sum if last else 0.0
        if last and not (self.plan.sequence_parallel and self.group.tp > 1):
            share /= self.group.tp
        t = torch.tensor([share], dtype=self.cfg.dtype, device=self.cfg.device)
        self.mgr.guarded_wait(dist.all_reduce(t, async_op=True), timeout=self.timeout, what="loss all-reduce")
        samples = self.executor.timing_samples()
        if self.reporter is not None and self.plan.ranks(self.group)[0] == self.comms.rank:
            self.reporter("timing", iteration=iteration, gen=self.plan.generation, devices=list(self.group.devices),
                          layers=self.local_layers, tp=self.group.tp, cp=self.group.cp, samples=samples)
        return t.item() / total


    def _finish_delayed_scaling(self) -> None:
        """Global amax per layer (max over every rank's micro-batches, like TE's amax reduction
        group), then the history update; ranks without the layer contribute 0."""
        if not self.cfg.amax_history:
            return
        blocks = [b for m in self.executor.modules.values() for b in m.blocks]
        amax = torch.zeros(self.cfg.num_layers, dtype=self.cfg.dtype, device=self.cfg.device)
        for b in blocks:
            amax[b.layer] = b.pending_amax
        self.mgr.guarded_wait(dist.all_reduce(amax, op=dist.ReduceOp.MAX, async_op=True), timeout=self.timeout,
                              what="amax all-reduce")
        for b in blocks:
            b.finish_iteration(amax[b.layer])


def build_stage_module(cfg: RefModelConfig, plan: HeteroParallelPlan, comms: GenerationComms,
                       mgr: ResilientCommManager) -> RefStage | ChunkedStage:
    """This device's layers: one ``RefStage``, or a ``ChunkedStage`` holding every local chunk."""
    me = comms.world[comms.rank]
    gs = plan.groups_of_device(me)
    g = gs[0]
    tp = TPContext(group=comms.group(plan.tp_name(g, me)) if g.tp > 1 else None,
                   rank=g.slice(g.slice_of(me)).index(me), size=g.tp, sequence_parallel=plan.sequence_parallel,
                   waiter=lambda w: mgr.guarded_wait(w, what="tp collective"))
    if len(gs) == 1:
        return RefStage(cfg, g.layers[0], g.layers[1], tp)
    return ChunkedStage({c.stage: RefStage(cfg, c.layers[0], c.layers[1], tp) for c in gs})
