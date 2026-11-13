"""Binding of the dynamic hybrid-parallel runtime (reference model) to the resilient loop.

Across generations the state of the committed plan is migrated to the new plan's
layout (TP resharding, PP layer moves, replica recovery) with the slice-level
transfer engine. The committed state is kept until the new generation commits, so a
reconfiguration interrupted by another failure restarts from intact state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from resihp.comm import GenerationComms, GroupSpec, ResilientCommManager
from resihp.control.worker_client import WorkerClient
from resihp.plan.hetero_plan import HeteroParallelPlan
from resihp.plan.ir import Schedule, schedule_from_json
from resihp.runtime.hetero_runtime import HeteroRuntime, build_stage_module, channel_specs, reference_partition_sizes
from resihp.runtime.optim import Adam
from resihp.runtime.reference_model import RefModelConfig, RefStage
from resihp.state.checkpoint import SliceCheckpointer, SliceMeta
from resihp.state.layer_registry import device_layout, holdings, reference_layer_of, reference_specs
from resihp.state.rng_data import IndexAddressableSampler
from resihp.state.transaction import TransactionManager
from resihp.state.transfer_engine import LocalState, execute_transfers
from resihp.state.transfer_plan import plan_transfers

KINDS = ("param", "exp_avg", "exp_avg_sq")


@dataclass
class ActiveState:
    plan: HeteroParallelPlan
    schedule: Schedule
    module: RefStage
    opt: Adam
    layout: dict[str, tuple[int, int]]

    def local_state(self) -> LocalState:
        """Parameters with their Adam moments; persistent buffers only have a value."""
        params = self.module.named_state()
        return {"param": {**{k: p.data for k, p in params.items()}, **self.module.named_buffer_state()},
                "exp_avg": {k: self.opt.state[k]["exp_avg"] for k in params},
                "exp_avg_sq": {k: self.opt.state[k]["exp_avg_sq"] for k in params}}


class HeteroBinding:
    def __init__(self, cfg: RefModelConfig, sampler: IndexAddressableSampler, mgr: ResilientCommManager,
                 client: WorkerClient, *, lr: float = 1e-2, use_channels: bool = False, injector=None,
                 timeout: float = 120.0, chunk: int | None = None, checkpointer: SliceCheckpointer | None = None,
                 checkpoint_interval: int = 0, report_timing: bool = True, iter_sleep: float = 0.0):
        self.cfg = cfg
        self.sampler = sampler
        self.mgr = mgr
        self.client = client
        self.lr = lr
        self.use_channels = use_channels
        self.injector = injector
        self.timeout = timeout
        self.chunk = chunk
        self.checkpointer = checkpointer
        self.checkpoint_interval = checkpoint_interval
        self.report_timing = report_timing
        self.iter_sleep = iter_sleep
        self.specs = reference_specs(cfg)
        self.committed: ActiveState | None = None
        self.pending: ActiveState | None = None
        self.last_transfer: dict = {}
        self.me = client.worker_id

    def _rows_max(self) -> int:
        m = self.cfg.token_multiple
        return -(-self.sampler.micro_batch_size * self.cfg.seq_max // m) * m

    def group_specs(self, world: list[str], payload: dict) -> list[GroupSpec]:
        plan = HeteroParallelPlan.from_json(payload["plan"])
        specs = plan.group_specs(reference_partition_sizes(self.cfg))
        if self.use_channels:
            specs += channel_specs(plan, schedule_from_json(payload["schedule"]), self._rows_max())
        return specs

    def recover(self, comms: GenerationComms, payload: dict, txm: TransactionManager) -> None:
        plan = HeteroParallelPlan.from_json(payload["plan"])
        schedule = schedule_from_json(payload["schedule"])
        rec = payload["recovery"]
        target = rec["target_iteration"]
        if self.committed is not None and self.me in rec["local_step"] and txm.has_reduced_grads:
            self.committed.opt.step()
            txm.adopt(target)
        else:
            # Roll back buffers of an abandoned iteration before this state serves as a source.
            txm.discard()
        module = build_stage_module(self.cfg, plan, comms, self.mgr)
        opt = Adam(module.named_state(), lr=self.lr)
        new_h = holdings(plan, self.specs, reference_layer_of)
        layout = device_layout(new_h, self.me)
        prev = payload.get("previous_plan")
        if rec.get("restore"):
            it = self._restore_from_checkpoint(module, opt, layout)
            txm.adopt(it)
        elif prev is not None and target >= 0:
            old_plan = HeteroParallelPlan.from_json(prev)
            old_h = holdings(old_plan, self.specs, reference_layer_of)
            valid = set(rec["up_to_date"]) | set(rec["local_step"])
            nodes = payload.get("nodes", {})
            tplan = plan_transfers(old_h, new_h, valid_sources=valid, node_of=nodes, chunk=self.chunk)
            if tplan.lost:
                raise RuntimeError(f"{len(tplan.lost)} state slices have no surviving holder")
            mine_valid = self.committed is not None and self.me in valid
            old_layout = device_layout(old_h, self.me) if mine_valid else {}
            old_state = self.committed.local_state() if mine_valid else {}
            new_state = execute_transfers(tplan, self.me, comms, self.mgr, specs=self.specs, old_layout=old_layout,
                                          old_state=old_state, new_layout=layout, kinds=KINDS, dtype=self.cfg.dtype,
                                          device=self.cfg.device, timeout=self.timeout)
            self._load(module, opt, new_state, step_count=target + 1)
            txm.adopt(target)
            self.last_transfer = {"transfers": len(tplan.transfers), "reuse": len(tplan.reuse),
                                  "received": sum(1 for t in tplan.transfers if t.dst == self.me)}
        txm.discard()
        self.pending = ActiveState(plan, schedule, module, opt, layout)

    def _load(self, module: RefStage, opt: Adam, state: LocalState, step_count: int) -> None:
        with torch.no_grad():
            for k, p in module.named_state().items():
                p.data.copy_(state["param"][k])
                opt.state[k]["exp_avg"].copy_(state["exp_avg"][k])
                opt.state[k]["exp_avg_sq"].copy_(state["exp_avg_sq"][k])
            for k, b in module.named_buffer_state().items():
                b.copy_(state["param"][k])
        opt.step_count = step_count

    def buffers(self) -> dict[str, torch.Tensor]:
        a = self.committed
        return a.module.transaction_buffers() if a is not None else {}

    def _kinds(self, key: str) -> tuple[str, ...]:
        return KINDS if self.specs[key].trainable else KINDS[:1]

    def build_runtime(self, comms: GenerationComms, payload: dict) -> HeteroRuntime:
        from resihp.detect.timing import ChunkTimer

        a = self.pending
        hook = self.injector.on_hook if self.injector is not None else None
        timer = ChunkTimer(use_cuda=False, injector=self.injector) if self.report_timing else None
        return HeteroRuntime(self.cfg, self.sampler, a.plan, a.schedule, comms, self.mgr, a.module,
                             use_channels=self.use_channels, progress=self.client.set_progress, hook=hook,
                             timeout=self.timeout, timer=timer,
                             reporter=self.client.report if self.report_timing else None)

    def on_generation_commit(self) -> None:
        self.committed, self.pending = self.pending, None

    def optimizer_step(self) -> None:
        self.committed.opt.step()

    def discard_grads(self) -> None:
        self.committed.opt.zero_grad()

    def after_commit(self, iteration: int, comms: GenerationComms) -> None:
        if self.iter_sleep:
            import time

            time.sleep(self.iter_sleep)
        if not (self.checkpointer and self.checkpoint_interval and (iteration + 1) % self.checkpoint_interval == 0):
            return
        a = self.committed
        state = a.local_state()
        slices = [(SliceMeta(k, self.specs[k].global_shape, self.specs[k].dim, iv),
                   {kind: state[kind][k] for kind in self._kinds(k)}) for k, iv in a.layout.items()]
        self.checkpointer.save(iteration, self.me, slices, extra={"step_count": a.opt.step_count}, blocking=True)

    def _restore_from_checkpoint(self, module: RefStage, opt: Adam, layout: dict[str, tuple[int, int]]) -> int:
        if self.checkpointer is None:
            raise RuntimeError("restore requested but no checkpointer configured")
        required = {k: (s.global_shape, s.dim) for k, s in self.specs.items()}
        it = self.checkpointer.latest_complete(required)
        if it is None:
            raise RuntimeError("no complete checkpoint to restore from")
        state: LocalState = {kind: {} for kind in KINDS}
        for trainable in (True, False):
            wanted = [SliceMeta(k, self.specs[k].global_shape, self.specs[k].dim, iv) for k, iv in layout.items()
                      if self.specs[k].trainable == trainable]
            kinds = KINDS if trainable else KINDS[:1]
            for k, v in self.checkpointer.load(it, wanted, kinds=kinds).items() if wanted else ():
                for kind in kinds:
                    state[kind][k] = v[kind]
        self._load(module, opt, state, step_count=it + 1)
        return it

    def try_resume(self, txm: TransactionManager) -> None:
        pass
