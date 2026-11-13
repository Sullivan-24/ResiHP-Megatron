"""Binding of Megatron-Core models to the resilient loop (GPU path).

Per generation the binding builds this device's model chunk for its (replica, stage) with
an explicit layer range (patch 0001) and a plan-derived ProcessGroupCollection, wraps it
(native DDP, Nonuniform TP, or ResiHP's shard-aligned sync), builds Megatron's optimizer,
migrates model + optimizer state in model space from the previous generation's holders,
and registers the interpreter-driven ``forward_backward_func`` (patch 0005), so
Megatron's ``train_step`` machinery runs heterogeneous, failure-adapted plans unchanged.

State sources for migration:

* optimizer not DP-sharded: every survivor reads its live TP-local state; zero rollback;
* DistributedOptimizer: a rank holds 1/DP of its shard and the old DP communicator is
  aborted by the failure, so model-space state is exported over the live DP group every
  ``snapshot_interval`` iterations to host memory and serves as the source (rollback of
  at most ``snapshot_interval`` iterations, the T1 tier).

Status: written against the pinned Megatron commit and checked statically
(``resihp.compat.megatron_contract``); not yet executed on GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from resihp.comm import GenerationComms, GroupSpec, ResilientCommManager
from resihp.control.worker_client import WorkerClient
from resihp.plan.hetero_plan import HeteroParallelPlan
from resihp.plan.ir import schedule_from_json
from resihp.runtime.hetero_runtime import channel_specs
from resihp.runtime.megatron_runtime import (build_pg_collection, make_forward_backward_func, megatron_group_specs,
                                             stage_layout, uniform_dp_group_name, wrap_model)
from resihp.state.layer_registry import TensorSpec, device_layout, holdings
from resihp.state.megatron_adapter import export_tp_shard_state, import_tp_shard_state
from resihp.state.transaction import TransactionManager
from resihp.state.transfer_engine import execute_transfers
from resihp.state.transfer_plan import plan_transfers

KINDS = ("param", "exp_avg", "exp_avg_sq")


@dataclass
class MegatronJob:
    """Model-specific hooks, mirroring ``pretrain_gpt.py``."""

    build_config: Callable[[dict], Any]  # stage layout -> TransformerConfig (resihp_layer_range, TP/CP sizes)
    model_provider: Callable[..., torch.nn.Module]  # (config, pre_process, post_process, pg_collection) -> chunk
    forward_step_func: Callable  # forward_step(data_iterator, model)
    batch_for: Callable[[int, int], Any]  # (iteration, global micro-batch) -> data iterator
    ddp_config: Any
    optimizer_config: Any
    state_key: Callable[[str, dict], str]  # (Megatron param name, layout) -> stable key ("layers.<L>.<...>")
    tensor_specs: dict[str, TensorSpec]  # stable key -> global shape and TP dim
    layer_of: Callable[[str], int]  # stable key -> global layer (embeddings: 0, output head: num_layers - 1)
    partition_sizes: tuple[int, ...]
    hidden: int
    seq_length: int
    micro_batch_size: int
    tp_base: int
    dtype: torch.dtype = torch.bfloat16


@dataclass
class _Generation:
    plan: HeteroParallelPlan
    comms: GenerationComms
    layout: dict
    bundle: Any
    optimizer: Any
    names: dict[str, str]  # stable key -> parameter name
    fbf: Callable | None = None


class MegatronBinding:
    def __init__(self, job: MegatronJob, mgr: ResilientCommManager, client: WorkerClient, *,
                 use_channels: bool = True, snapshot_interval: int = 10, timeout: float = 600.0):
        self.job = job
        self.mgr = mgr
        self.client = client
        self.use_channels = use_channels
        self.snapshot_interval = snapshot_interval
        self.timeout = timeout
        self.me = client.worker_id
        self.committed: _Generation | None = None
        self.pending: _Generation | None = None
        self.snapshot: tuple[int, dict[str, dict[str, torch.Tensor]]] | None = None

    # ------------------------------------------------------------------ RuntimeBinding
    def group_specs(self, world: list[str], payload: dict) -> list[GroupSpec]:
        plan = HeteroParallelPlan.from_json(payload["plan"])
        specs = megatron_group_specs(plan, self.job.partition_sizes)
        if self.use_channels:
            g = plan.stage_groups[0]
            specs += channel_specs(plan, schedule_from_json(payload["schedule"]), 0,
                                   row_counts=[self.job.seq_length // g.cp])
        return specs

    def _distopt(self) -> bool:
        return bool(getattr(self.job.ddp_config, "use_distributed_optimizer", False))

    def _model_space_state(self, gen: _Generation) -> dict[str, dict[str, torch.Tensor]]:
        dp_group = None
        if self._distopt():
            name = uniform_dp_group_name(gen.plan, gen.layout["stage"], gen.layout["tp_rank"])
            dp_group = gen.comms.group(name) if name else None
        by_name = export_tp_shard_state(gen.optimizer, gen.bundle.model, dp_group)
        keys = {n: k for k, n in gen.names.items()}
        return {kind: {keys[n]: t for n, t in per.items() if n in keys} for kind, per in by_name.items()}

    def recover(self, comms: GenerationComms, payload: dict, txm: TransactionManager) -> None:
        from megatron.core.optimizer import get_megatron_optimizer

        job = self.job
        plan = HeteroParallelPlan.from_json(payload["plan"])
        rec = payload["recovery"]
        target = rec["target_iteration"]
        if self.committed is not None and self.me in rec["local_step"] and txm.has_reduced_grads:
            self.committed.optimizer.step()
            txm.adopt(target)
        else:
            txm.discard()
        layout = stage_layout(plan, comms)
        pgc = build_pg_collection(plan, comms)
        config = job.build_config(layout)
        model = job.model_provider(config=config, pre_process=layout["pre_process"],
                                   post_process=layout["post_process"], pg_collection=pgc)
        bundle = wrap_model(model, config, job.ddp_config, plan, comms, self.mgr, pgc, job.tp_base)
        optimizer = get_megatron_optimizer(job.optimizer_config, [bundle.ddp], pg_collection=pgc)
        names = {job.state_key(n, layout): n for n, _ in model.named_parameters()}
        gen = _Generation(plan, comms, layout, bundle, optimizer, names)

        prev = payload.get("previous_plan")
        if prev is not None and target >= 0:
            old_plan = HeteroParallelPlan.from_json(prev)
            new_h = holdings(plan, job.tensor_specs, job.layer_of)
            old_h = holdings(old_plan, job.tensor_specs, job.layer_of)
            if self._distopt():
                source_it = self.snapshot[0] if self.snapshot else -1
                old_state = self.snapshot[1] if self.snapshot else {}
            else:
                source_it = target
                old_state = self._model_space_state(self.committed) if self.committed is not None else {}
            valid = set(rec["up_to_date"]) | set(rec["local_step"])
            tplan = plan_transfers(old_h, new_h, valid_sources=valid, node_of=payload.get("nodes", {}))
            if tplan.lost:
                raise RuntimeError(f"{len(tplan.lost)} state slices have no surviving holder")
            mine = self.me in valid and bool(old_state)
            new_state = execute_transfers(tplan, self.me, comms, self.mgr, specs=job.tensor_specs,
                                          old_layout=device_layout(old_h, self.me) if mine else {},
                                          old_state=old_state if mine else {},
                                          new_layout=device_layout(new_h, self.me), kinds=KINDS,
                                          dtype=torch.float32, device=f"cuda:{torch.cuda.current_device()}",
                                          timeout=self.timeout)
            import_tp_shard_state(optimizer, model, {kind: {names[k]: t for k, t in per.items() if k in names}
                                                     for kind, per in new_state.items()})
            txm.adopt(source_it)
        self.pending = gen

    def build_runtime(self, comms: GenerationComms, payload: dict):
        from megatron.core.pipeline_parallel.schedules import register_forward_backward_func

        gen, job = self.pending, self.job
        gen.fbf = make_forward_backward_func(gen.plan, schedule_from_json(payload["schedule"]), comms, self.mgr,
                                             gen.bundle, job.batch_for, job.hidden, job.dtype, self.use_channels)
        register_forward_backward_func(gen.fbf)
        return _MegatronRuntime(gen, job)

    def on_generation_commit(self) -> None:
        self.committed, self.pending = self.pending, None

    def optimizer_step(self) -> None:
        self.committed.optimizer.step()

    def discard_grads(self) -> None:
        self.committed.optimizer.zero_grad()

    def after_commit(self, iteration: int, comms: GenerationComms) -> None:
        if self._distopt() and (iteration + 1) % self.snapshot_interval == 0:
            state = self._model_space_state(self.committed)
            self.snapshot = (iteration, {k: {key: t.detach().to("cpu", non_blocking=True) for key, t in per.items()}
                                         for k, per in state.items()})

    def try_resume(self, txm: TransactionManager) -> None:
        pass


class _MegatronRuntime:
    def __init__(self, gen: _Generation, job: MegatronJob):
        self.gen, self.job = gen, job

    def run_iteration(self, iteration: int) -> float:
        gen, job = self.gen, self.job
        gen.bundle.ddp.zero_grad_buffer()
        gen.optimizer.zero_grad()
        store = gen.fbf(forward_step_func=job.forward_step_func, data_iterator=None, model=[gen.bundle.ddp],
                        num_microbatches=gen.plan.num_microbatches, seq_length=job.seq_length,
                        micro_batch_size=job.micro_batch_size, iteration=iteration)
        losses = [float(v) for d in store if isinstance(d, dict) for v in d.values() if torch.is_tensor(v) or
                  isinstance(v, (int, float))]
        return sum(losses) / len(losses) if losses else 0.0
