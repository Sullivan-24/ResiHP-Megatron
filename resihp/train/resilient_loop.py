"""Framework-agnostic resilient training loop.

    begin txn -> run iteration (abortable) -> reduced -> optimizer step (commit)
    on any communication failure / reconfiguration notice:
        abort txn (roll back RNG + data cursor) -> wait PREPARE(g) -> rebuild comms(g)
        -> recover state (roll-forward / peer copy / snapshot / checkpoint) -> READY(g)
        -> COMMIT(g) -> replay the aborted iteration with the same global batch

Survivor processes never restart. The runtime is supplied by ``RuntimeBinding`` so
the same loop drives the replica-level DP runtime, the heterogeneous hybrid
parallel runtime, and the Megatron adapter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from resihp.comm import CommAborted, CommTimeout, GenerationBuildError, GenerationComms, ResilientCommManager
from resihp.control.worker_client import JobFinished, ReconfigInterrupted, WorkerClient
from resihp.runtime.generation_hooks import GenerationHooks
from resihp.state.transaction import TransactionManager
from resihp.utils.log import get_logger

log = get_logger(__name__)

RECOVERABLE = (CommAborted, CommTimeout, GenerationBuildError, RuntimeError, ReconfigInterrupted)


class Runtime(Protocol):
    def run_iteration(self, iteration: int) -> float: ...


class RuntimeBinding(Protocol):
    """Glue between the loop and a concrete model/optimizer/runtime."""

    def group_specs(self, world: list[str], payload: dict) -> list: ...
    def build_runtime(self, comms: GenerationComms, payload: dict) -> Runtime: ...
    def recover(self, comms: GenerationComms, payload: dict, txm: TransactionManager) -> None: ...
    def optimizer_step(self) -> None: ...
    def discard_grads(self) -> None: ...
    def after_commit(self, iteration: int, comms: GenerationComms) -> None: ...
    def try_resume(self, txm: TransactionManager) -> None: ...


@dataclass
class TrainResult:
    losses: dict[int, float] = field(default_factory=dict)
    generations: list[int] = field(default_factory=list)
    reconfigurations: list[dict] = field(default_factory=list)
    standby: bool = False


class ResilientTrainer:
    def __init__(self, client: WorkerClient, mgr: ResilientCommManager, binding: RuntimeBinding, *,
                 txm: TransactionManager | None = None, prepare_timeout: float = 600.0,
                 hooks: GenerationHooks | None = None):
        self.client = client
        self.mgr = mgr
        self.binding = binding
        self.txm = txm or TransactionManager()
        self.prepare_timeout = prepare_timeout
        self.hooks = hooks or GenerationHooks()
        self.comms: GenerationComms | None = None
        self.runtime: Runtime | None = None
        self.result = TrainResult()
        client.add_abort_callback(mgr.abort)
        if hasattr(binding, "buffers"):
            self.txm.track_buffers(binding.buffers)

    def run(self, num_iterations: int) -> TrainResult:
        self.binding.try_resume(self.txm)
        try:
            self._reconfigure()
            while True:
                self._train(num_iterations)
                status = self.client.finish(self.txm.committed_iteration, timeout=self.prepare_timeout)
                if status == "shutdown":
                    break
                self._reconfigure()
        except JobFinished:
            pass
        return self.result

    def _train(self, num_iterations: int) -> None:
        while self.txm.committed_iteration + 1 < num_iterations:
            if self.runtime is None:
                self._reconfigure()
                continue
            it = self.txm.committed_iteration + 1
            self.txm.begin(it)
            try:
                if self.client.immediate_abort_pending():
                    raise CommAborted("reconfiguration announced before iteration start")
                loss = self.runtime.run_iteration(it)
                self.txm.on_reduced()
                self.txm.on_step_begin()
                self.binding.optimizer_step()
                self.txm.on_commit()
                self.result.losses[it] = loss
                self.client.set_progress(iteration=it, phase="committed", in_collective=False)
                self.client.set_recovery_info(**self.txm.recovery_info())
                self.binding.after_commit(it, self.comms)
                self.hooks.after_iteration(it)
                if self.client.reconfig_pending():
                    self._reconfigure()
            except RECOVERABLE as e:
                log.warning("iteration %d interrupted: %s", it, e)
                self.txm.abort()
                self._reconfigure()

    def _reconfigure(self) -> None:
        t0 = time.monotonic()
        while True:
            # Rebuilding communicators blocks the GIL for seconds; tell the agent not to apply its
            # training-time hang timeout until this worker is computing again.
            self.client.set_progress(phase="rebuild", in_collective=False)
            self.client.set_recovery_info(**self.txm.recovery_info())
            prep = self.client.wait_prepare(timeout=self.prepare_timeout)
            gen, world, payload = prep["gen"], prep["world"], prep["payload"]
            me = self.client.worker_id
            if self.comms is not None:
                self.hooks.teardown()
            if me not in world:
                self.mgr.destroy()
                self.comms, self.runtime = None, None
                self.client.send_ready(gen, standby=True)
                try:
                    self.client.wait_commit(gen, timeout=self.prepare_timeout)
                except ReconfigInterrupted:
                    continue
                self.result.standby = True
                self.result.generations.append(gen)
                continue
            try:
                specs = self.binding.group_specs(world, payload)
                store_addr = tuple(prep["store_addr"]) if prep.get("store_addr") else None
                self.comms = None
                comms = self.mgr.build(gen, world, me, specs, store_addr=store_addr, store_prefix=prep["store_prefix"])
                self.binding.recover(comms, payload, self.txm)
                self.runtime = self.binding.build_runtime(comms, payload)
                self.hooks.rebuild(comms, payload)
                self.comms = comms
                self.client.send_ready(gen)
                self.client.wait_commit(gen, timeout=self.prepare_timeout)
                on_commit = getattr(self.binding, "on_generation_commit", None)
                if on_commit is not None:
                    on_commit()
            except RECOVERABLE as e:
                log.warning("generation %d build/recovery failed: %s", gen, e)
                self.hooks.teardown()
                self.client.report("build_failed", gen=gen, suspects=list(getattr(e, "suspects", ())),
                                   detail=str(e)[:200])
                self.runtime = None
                continue
            self.result.generations.append(gen)
            self.result.standby = False
            self.result.reconfigurations.append({"generation": gen, "world": list(world), "seconds": time.monotonic() - t0,
                                                 "committed_iteration": self.txm.committed_iteration,
                                                 "build_seconds": comms.build_seconds, "path": comms.path})
            self.client.set_recovery_info(**self.txm.recovery_info())
            return
