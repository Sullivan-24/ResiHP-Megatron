"""Megatron Core binding of the ResiHP runtime (GPU path).

Megatron is imported lazily so the rest of ResiHP stays importable on CPU-only
hosts; every Megatron symbol used here is pinned by ``resihp.compat.megatron_contract``.

Per generation, on each rank:
1. ``megatron_group_specs`` adds the size-1 groups Megatron expects to the plan's TP and
   shard-aligned DP groups;
2. ``build_pg_collection`` maps them onto a ``ProcessGroupCollection``. The pipeline
   dimension is driven by ResiHP's interpreter, so Megatron sees a size-1 PP group and an
   explicit ``resihp_layer_range`` (patch 0001) plus ``pre_process``/``post_process``;
3. ``wrap_model`` uses native DDP (uniform TP across replicas), Megatron's Nonuniform-TP DDP
   (single reduced degree, NTP-compatible layout), or DDP with a local-only DP group followed
   by ``HeteroGradSync`` on ``main_grad`` (any heterogeneous layout);
4. ``MegatronStageExecutor`` runs Megatron's ``forward_step``/``backward_step`` for the IR
   interpreter, and ``make_forward_backward_func`` exposes the whole thing to Megatron's
   training loop through ``register_forward_backward_func`` (patch 0005).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.distributed as dist

from resihp.comm import GenerationComms, GroupSpec
from resihp.comm.hetero_p2p import P2PEngine
from resihp.plan.hetero_plan import HeteroParallelPlan, cp_group_name, dpshard_group_name
from resihp.plan.ir import Op, Schedule
from resihp.runtime.hetero_grad_sync import HeteroGradSync, ParamShard
from resihp.runtime.hetero_runtime import channel_map
from resihp.runtime.interpreter import Interpreter
from resihp.runtime.weight_grad_store import MegatronWeightGradStore


def singleton_name(kind: str, rank: int) -> str:
    return f"{kind}/self/{rank}"


def tpcp_group_name(replica: int, stage: int) -> str:
    return f"tpcp/r{replica}/s{stage}"


def dp_group_name(stage: int, cp_slice: int, tp_rank: int) -> str:
    return f"dp/s{stage}/c{cp_slice}/t{tp_rank}"


def megatron_group_specs(plan: HeteroParallelPlan, partition_sizes: tuple[int, ...]) -> list[GroupSpec]:
    """``plan.group_specs`` plus what Megatron's ProcessGroupCollection needs: size-1 TP/CP
    groups, the TP x CP group of a stage group, and (with CP) DP groups without CP."""
    specs = {s.name: s for s in plan.group_specs(partition_sizes)}
    for g in plan.stage_groups:
        for i in range(g.cp):
            if g.tp == 1:
                name = plan.tp_name(g, g.slice(i)[0])
                specs[name] = GroupSpec(name, plan.slice_ranks(g, i), "tp", keep_singleton=True)
        if g.cp > 1:
            name = tpcp_group_name(g.replica, g.stage)
            specs[name] = GroupSpec(name, plan.ranks(g), "tpcp")
            if stage_is_uniform(plan, g.stage):
                for i in range(g.cp):
                    for j in range(g.tp):
                        ranks = tuple(sorted(plan.rank(h.slice(i)[j]) for h in plan.groups_at(g.stage)))
                        name = dp_group_name(g.stage, i, j)
                        specs[name] = GroupSpec(name, ranks, "dp", keep_singleton=True)
    for r in range(len(plan.world)):
        name = singleton_name("self", r)
        specs[name] = GroupSpec(name, (r,), "self", keep_singleton=True)
    return list(specs.values())


def stage_is_uniform(plan: HeteroParallelPlan, stage: int) -> bool:
    return len({g.tp for g in plan.groups_at(stage)}) == 1


def uniform_dp_group_name(plan: HeteroParallelPlan, stage: int, tp_rank: int) -> str | None:
    """Megatron's ``dp_cp`` group: the holders of TP shard ``tp_rank`` in every replica and CP slice."""
    slices = plan.tp_slices(stage)
    if len(slices) < 2:
        return None
    return dpshard_group_name(stage, tuple(sorted(plan.rank(sl[tp_rank]) for sl in slices)))


def build_pg_collection(plan: HeteroParallelPlan, comms: GenerationComms):
    from megatron.core.process_groups_config import ProcessGroupCollection

    me = comms.world[comms.rank]
    g = plan.group_of_device(me)
    cp_rank = g.slice_of(me)
    tp_rank = g.slice(cp_rank).index(me)
    selfg = comms.group(singleton_name("self", comms.rank))
    tp = comms.group(plan.tp_name(g, me))
    cp = comms.group(cp_group_name(g.replica, g.stage, tp_rank)) if g.cp > 1 else selfg
    tp_cp = comms.group(tpcp_group_name(g.replica, g.stage)) if g.cp > 1 else tp
    dp_cp = dp = selfg
    if stage_is_uniform(plan, g.stage):
        name = uniform_dp_group_name(plan, g.stage, tp_rank)
        if name is not None and comms.group(name) is not None:
            dp_cp = dp = comms.group(name)
        if g.cp > 1:
            dp = comms.group(dp_group_name(g.stage, cp_rank, tp_rank))
    pgc = ProcessGroupCollection()
    for attr, pg in (("tp", tp), ("pp", selfg), ("mp", tp), ("embd", selfg), ("pos_embd", selfg), ("cp", cp),
                     ("tp_cp", tp_cp), ("hcp", [cp]), ("ep", selfg), ("expt_tp", tp), ("tp_ep", tp), ("tp_ep_pp", tp),
                     ("tp_ep_pp_with_egtp_remat", tp), ("tp_dp_cp", tp_cp), ("dp", dp), ("dp_gtp_remat", dp),
                     ("dp_cp", dp_cp), ("dp_cp_gtp_remat", dp_cp), ("expt_dp", dp_cp), ("expt_dp_gtp_remat", dp_cp),
                     ("intra_dp_cp", dp_cp), ("intra_expt_dp", dp_cp), ("inter_dist_opt", selfg)):
        try:
            setattr(pgc, attr, pg)
        except AttributeError:
            pass
    return pgc


def stage_layout(plan: HeteroParallelPlan, comms: GenerationComms) -> dict:
    me = comms.world[comms.rank]
    g = plan.group_of_device(me)
    cp_rank = g.slice_of(me)
    return {"pre_process": g.stage == 0, "post_process": g.stage == plan.num_stages - 1,
            "resihp_layer_range": g.layers, "tp_size": g.tp, "tp_rank": g.slice(cp_rank).index(me),
            "cp_size": g.cp, "cp_rank": cp_rank, "replica": g.replica, "stage": g.stage}


def ntp_config_for_stage(plan: HeteroParallelPlan, stage: int, tp_base: int):
    """NonuniformTPConfig when the stage fits NTP's model (one reduced degree, healthy = tp_base)."""
    from megatron.core.distributed.nonuniform_tp import NonuniformTPConfig

    degrees = {g.replica: g.tp for g in plan.groups_at(stage)}
    reduced = {r: k for r, k in degrees.items() if k != tp_base}
    if not reduced or len(set(reduced.values())) != 1:
        return None
    k = next(iter(reduced.values()))
    non_active = {(r, 0, stage): list(range(k, tp_base)) for r in reduced}
    return NonuniformTPConfig(tp_base=tp_base, tp_spares=tp_base - k, num_reduced_tp_dp_ranks=len(reduced),
                              non_active_ranks_per_dp=non_active)


def megatron_param_shards(model: torch.nn.Module, tp_size: int) -> dict[str, ParamShard]:
    """Partition metadata from Megatron's tensor-parallel attributes."""
    out = {}
    for name, p in model.named_parameters():
        if getattr(p, "tensor_model_parallel", False) and tp_size > 1:
            dim = int(getattr(p, "partition_dim", 0))
            out[name] = ParamShard(name, dim, p.shape[dim] * tp_size, False)
        else:
            sp_partial = bool(getattr(p, "sequence_parallel", False)) and tp_size > 1
            out[name] = ParamShard(name, None, p.numel(), sp_partial)
    return out


@dataclass
class MegatronModelBundle:
    model: torch.nn.Module
    ddp: torch.nn.Module
    grad_sync: Callable[[], None]
    mode: str
    shards: dict[str, ParamShard] = field(default_factory=dict)


def wrap_model(model: torch.nn.Module, config, ddp_config, plan: HeteroParallelPlan, comms: GenerationComms, mgr,
               pg_collection, tp_base: int, prefer_ntp: bool = True) -> MegatronModelBundle:
    """``grad_sync`` leaves unscaled, fully reduced gradients in ``main_grad`` (requires
    ``calculate_per_token_loss`` so DDP applies no 1/dp factor)."""
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads

    layout = stage_layout(plan, comms)
    stage = layout["stage"]
    if stage_is_uniform(plan, stage):
        ddp = DistributedDataParallel(config=config, ddp_config=ddp_config, module=model, pg_collection=pg_collection)
        return MegatronModelBundle(model, ddp, lambda: finalize_model_grads([ddp], num_tokens=None,
                                                                            pg_collection=pg_collection), "native")
    ntp_cfg = ntp_config_for_stage(plan, stage, tp_base) if prefer_ntp else None
    if ntp_cfg is not None:
        from megatron.core.distributed.nonuniform_tp import NonuniformTPDistributedDataParallel, ntp_init

        for module in model.modules():
            if module.__class__.__name__ == "TransformerLayer":
                ntp_init(module, ntp_cfg)
        ddp = NonuniformTPDistributedDataParallel(config=config, ddp_config=ddp_config, module=model,
                                                  pg_collection=pg_collection, ntp_config=ntp_cfg)
        return MegatronModelBundle(model, ddp, lambda: finalize_model_grads([ddp], num_tokens=None,
                                                                            pg_collection=pg_collection), "ntp")
    ddp = DistributedDataParallel(config=config, ddp_config=ddp_config, module=model, pg_collection=pg_collection)
    shards = megatron_param_shards(model, layout["tp_size"])
    params = {n: p for n, p in model.named_parameters()}
    sync = HeteroGradSync(plan, comms, mgr, stage, layout["replica"], params, shards,
                          tp_group=comms.group(plan.tp_name(plan.group(layout["replica"], stage), comms.world[comms.rank])))

    def grad_sync() -> None:
        ddp.finish_grad_sync()
        for p in params.values():
            if hasattr(p, "main_grad"):
                p.grad = p.main_grad
        sync.sync()

    return MegatronModelBundle(model, ddp, grad_sync, "resihp", shards)


class MegatronStageExecutor:
    """Adapts Megatron's forward_step/backward_step to the IR interpreter.

    ``batch_for(iteration, mb)`` returns an iterator yielding the batch of global
    micro-batch ``mb`` (index-addressable data), so migrated micro-batches load the
    right samples on whichever replica executes them.
    """

    def __init__(self, model, forward_step_func, batch_for: Callable[[int, int], object], config, plan, layout: dict,
                 num_microbatches: int, seq_rows: int, weight_grad_store=None):
        self.wgs = weight_grad_store
        self.model = model
        self.forward_step_func = forward_step_func
        self.batch_for = batch_for
        self.config = config
        self.plan = plan
        self.layout = layout
        self.num_microbatches = num_microbatches
        self.seq_rows = seq_rows
        self.iteration = -1
        self.saved: dict[int, tuple] = {}
        self.forward_data_store: list = []
        self.num_tokens = torch.zeros(1, dtype=torch.int64)

    def begin_iteration(self, iteration: int) -> None:
        self.iteration = iteration
        self.saved.clear()
        self.forward_data_store = []
        self.num_tokens = torch.zeros(1, dtype=torch.int64, device=torch.cuda.current_device())

    def n_rows(self, mb: int) -> int:
        return self.seq_rows

    def forward(self, mb: int, inp, stage: int):
        from megatron.core.pipeline_parallel.schedules import forward_step

        last = stage == self.plan.num_stages - 1
        if inp is not None:
            inp = inp.detach().requires_grad_(True)
        model = self.model[stage] if isinstance(self.model, dict) else self.model
        out, ntok = forward_step(self.forward_step_func, self.batch_for(self.iteration, mb), model,
                                 self.num_microbatches, inp, self.forward_data_store, self.config, 1,
                                 current_microbatch=mb, is_last_stage=last)
        self.num_tokens += ntok
        self.saved[(stage, mb)] = (inp, out)
        return None if last else out.detach()

    def backward(self, mb: int, grad, stage: int):
        from megatron.core.pipeline_parallel.schedules import backward_step

        inp, out = self.saved.pop((stage, mb))
        if self.wgs is None:
            return backward_step(inp, out, grad, self.config)
        with self.wgs.deferring((stage, mb)):
            return backward_step(inp, out, grad, self.config)

    def weight_backward(self, mb: int, stage: int) -> None:
        if self.wgs is not None:
            self.wgs.compute((stage, mb))


def make_forward_backward_func(plan: HeteroParallelPlan, schedule: Schedule, comms: GenerationComms, mgr,
                               bundle: MegatronModelBundle, batch_for, hidden: int, dtype: torch.dtype,
                               use_channels: bool = True):
    """A drop-in for Megatron's forward_backward funcs driven by the ResiHP interpreter."""
    layout = stage_layout(plan, comms)
    p2p = P2PEngine(comms.rank, mgr, channels=channel_map(comms) if use_channels else None)
    stream = schedule[(layout["replica"], layout["stage"])]
    wgs = None
    if any(ins.op == Op.W for ins in stream):
        cfg = bundle.model.config
        if not (getattr(cfg, "delay_wgrad_compute", False) and getattr(cfg, "external_wgrad_schedule", False)):
            raise ValueError("a zero-bubble schedule needs delay_wgrad_compute=True and external_wgrad_schedule=True "
                             "(patch 0006) with Transformer Engine linear layers")
        wgs = MegatronWeightGradStore({layout["stage"]: bundle.model})

    def forward_backward_func(*, forward_step_func, data_iterator, model, num_microbatches, seq_length,
                              micro_batch_size, decoder_seq_length=None, forward_only=False,
                              collect_non_loss_data=False, first_val_step=None, adjust_tensor_shapes_fn=None,
                              p2p_communicator=None, pg_collection=None, force_all_reduce=False, iteration=0):
        executor = MegatronStageExecutor(bundle.ddp, forward_step_func, batch_for, bundle.model.config, plan, layout,
                                         plan.num_microbatches, seq_length, weight_grad_store=wgs)
        group = plan.group(layout["replica"], layout["stage"])
        interp = Interpreter(plan, stream, executor, p2p, layout["replica"], layout["stage"], (micro_batch_size, hidden),
                             dtype, tp_group=comms.group(plan.tp_name(group, comms.world[comms.rank])))
        interp.run(iteration)
        if forward_only:
            return executor.forward_data_store
        bundle.grad_sync()
        # Global token count: every TP rank of a last-stage group counted the same tokens.
        share = executor.num_tokens.double() / layout["tp_size"] if layout["post_process"] else \
            torch.zeros(1, dtype=torch.float64, device=executor.num_tokens.device)
        dist.all_reduce(share)
        total = share.clamp(min=1.0)
        for p in bundle.model.parameters():
            g = getattr(p, "main_grad", None)
            if g is not None:
                g.div_(total.to(g.dtype))
        return executor.forward_data_store

    return forward_backward_func
