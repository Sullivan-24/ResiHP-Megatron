"""Binding of the replica-level DP runtime to the resilient loop (reference model + Adam)."""

from __future__ import annotations

from resihp.comm import GenerationComms, GroupSpec, ResilientCommManager
from resihp.control.worker_client import WorkerClient
from resihp.runtime.dp_runtime import DataParallelRuntime, sync_full_state
from resihp.runtime.optim import Adam
from resihp.runtime.reference_model import RefModelConfig, RefStage
from resihp.state.checkpoint import SliceCheckpointer, SliceMeta
from resihp.state.redundancy import HostSnapshotManager, exchange_snapshot_with_buddy
from resihp.state.rng_data import IndexAddressableSampler
from resihp.state.transaction import TransactionManager


class DPBinding:
    def __init__(self, cfg: RefModelConfig, sampler: IndexAddressableSampler, model: RefStage, opt: Adam,
                 mgr: ResilientCommManager, client: WorkerClient, *, injector=None,
                 snapshots: HostSnapshotManager | None = None, replicate_snapshots: bool = True,
                 checkpointer: SliceCheckpointer | None = None, checkpoint_interval: int = 0, timeout: float = 120.0):
        self.cfg = cfg
        self.sampler = sampler
        self.model = model
        self.opt = opt
        self.mgr = mgr
        self.client = client
        self.injector = injector
        self.snapshots = snapshots
        self.replicate_snapshots = replicate_snapshots
        self.checkpointer = checkpointer
        self.checkpoint_interval = checkpoint_interval
        self.timeout = timeout
        self.me = client.worker_id

    def group_specs(self, world: list[str], payload: dict) -> list[GroupSpec]:
        return []

    def build_runtime(self, comms: GenerationComms, payload: dict) -> DataParallelRuntime:
        hook = self.injector.on_hook if self.injector is not None else None
        return DataParallelRuntime(self.cfg, self.sampler, self.model, comms, self.mgr, payload["assignment"][self.me],
                                   hook=hook, progress=self.client.set_progress, timeout=self.timeout)

    def recover(self, comms: GenerationComms, payload: dict, txm: TransactionManager) -> None:
        rec = payload["recovery"]
        if rec.get("restore") == "checkpoint":
            it = self._load_checkpoint()
            if it is None:
                raise RuntimeError("planner requested checkpoint restore but no complete checkpoint exists")
            txm.adopt(it)
        else:
            if self.me in rec["local_step"] and txm.has_reduced_grads:
                self.opt.step()
                txm.adopt(rec["target_iteration"])
            adopted = sync_full_state(self.mgr, comms, self.me, rec["sources"], self.opt, txm.committed_iteration,
                                      timeout=self.timeout)
            if adopted is not None:
                txm.adopt(adopted)
        txm.discard()
        self.opt.zero_grad()

    def optimizer_step(self) -> None:
        if self.snapshots is not None:
            self.snapshots.fence()
        self.opt.step()

    def discard_grads(self) -> None:
        self.opt.zero_grad()

    def _required(self) -> dict:
        return {k: (tuple(v.shape), None) for k, v in self.opt.state_tensors().items()}

    def after_commit(self, iteration: int, comms: GenerationComms) -> None:
        if self.snapshots is not None and self.snapshots.due(iteration):
            state = self.opt.state_tensors()
            self.snapshots.snapshot(iteration, state, blocking=True)
            if self.replicate_snapshots and comms.world_size > 1:
                w = list(comms.world)
                r = comms.rank
                it, held = exchange_snapshot_with_buddy(self.mgr, comms, self.snapshots.latest(),
                                                        send_to=w[(r + 1) % len(w)], recv_from=w[(r - 1) % len(w)],
                                                        template=state, timeout=self.timeout)
                self.snapshots.hold_for(w[(r - 1) % len(w)], it, held)
        if (self.checkpointer is not None and self.checkpoint_interval and (iteration + 1) % self.checkpoint_interval == 0
                and comms.rank == 0):
            slices = [(SliceMeta(k, tuple(v.shape), None, (0, 1)), {"value": v}) for k, v in self.opt.state_tensors().items()]
            self.checkpointer.save(iteration, self.me, slices, extra={"step_count": self.opt.step_count}, blocking=True)

    def _load_checkpoint(self) -> int | None:
        if self.checkpointer is None:
            return None
        it = self.checkpointer.latest_complete(self._required())
        if it is None:
            return None
        wanted = [SliceMeta(k, shape, None, (0, 1)) for k, (shape, _) in self._required().items()]
        loaded = self.checkpointer.load(it, wanted, kinds=("value",))
        import torch

        extra = torch.load(next(self.checkpointer._dir(it).glob("*.pt")), weights_only=False)["extra"]
        self.opt.load_state_tensors({k: v["value"] for k, v in loaded.items()}, extra["step_count"])
        return it

    def try_resume(self, txm: TransactionManager) -> None:
        it = self._load_checkpoint()
        if it is not None:
            txm.adopt(it)
