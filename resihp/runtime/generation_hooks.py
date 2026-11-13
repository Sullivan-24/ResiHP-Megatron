"""Per-generation rebuild of resources bound to one communicator layout.

Transformer Engine's TP-overlap userbuffers are allocated for one TP group and size,
and captured CUDA graphs bake in communicator handles, buffer addresses and the layer
set. Both must be released *before* the old generation's communicators are destroyed
and re-created *after* the new generation's model exists (before READY), or the new
generation replays stale pointers. ``ResilientTrainer`` drives the hooks:

    teardown()                 before ResilientCommManager.build() destroys generation g
    rebuild(comms, payload)    after the binding built generation g+1's runtime
    after_iteration(i)         after every committed iteration (deferred CUDA graph capture)

Teardown is best effort (a failed release must not block recovery); a failed rebuild
raises, which reports ``build_failed`` and retries the generation like any build error.
"""

from __future__ import annotations

from typing import Callable, Protocol

from resihp.plan.hetero_plan import HeteroParallelPlan
from resihp.utils.log import get_logger

log = get_logger(__name__)


class GenerationHook(Protocol):
    def teardown(self) -> None: ...
    def rebuild(self, comms, payload: dict) -> None: ...


class GenerationHooks:
    def __init__(self, hooks=()):
        self.hooks = list(hooks)

    def register(self, hook) -> None:
        self.hooks.append(hook)

    def teardown(self) -> list[str]:
        errors = []
        for h in reversed(self.hooks):
            try:
                h.teardown()
            except Exception as e:  # noqa: BLE001 - see module docstring
                log.warning("teardown of %s failed: %s", type(h).__name__, e)
                errors.append(f"{type(h).__name__}: {e}")
        return errors

    def rebuild(self, comms, payload: dict) -> None:
        for h in self.hooks:
            h.rebuild(comms, payload)

    def after_iteration(self, iteration: int) -> None:
        for h in self.hooks:
            fn = getattr(h, "after_iteration", None)
            if fn is not None:
                fn(iteration)


def uniform_contiguous_tp(plan: HeteroParallelPlan) -> int | None:
    """TE bootstraps userbuffers over contiguous, aligned, equal-size TP blocks of the world;
    return that size when every TP group of the plan has this shape, else ``None``."""
    tps = {g.tp for g in plan.stage_groups}
    if len(tps) != 1:
        return None
    tp = tps.pop()
    for g in plan.stage_groups:
        for i in range(g.cp):
            ranks = plan.slice_ranks(g, i)
            if ranks != tuple(range(ranks[0], ranks[0] + tp)) or ranks[0] % tp:
                return None
    return tp


class TEUserBufferHook:
    """TP comm-overlap userbuffers for the new generation's TP layout.

    When the layout is not uniform and rank-contiguous (heterogeneous TP after a
    failure), the generation runs without TP overlap: ``enabled`` is False and the model
    builder must construct the chunk with ``tp_comm_overlap=False``.
    """

    def __init__(self, shape_fn: Callable[[int], list[int]], *, dtype=None, use_fp8: bool = False, ub_cfgs=None,
                 bootstrap_backend: str | None = None, te_base=None):
        self.shape_fn = shape_fn
        self.dtype = dtype
        self.use_fp8 = use_fp8
        self.ub_cfgs = ub_cfgs
        self.bootstrap_backend = bootstrap_backend
        self._te = te_base
        self.active = False
        self.enabled = False

    @property
    def te(self):
        if self._te is None:
            from transformer_engine.pytorch.module import base as te_base

            self._te = te_base
        return self._te

    def teardown(self) -> None:
        if self.active:
            self.te.destroy_ub()
            self.active = False

    def rebuild(self, comms, payload: dict) -> None:
        tp = uniform_contiguous_tp(HeteroParallelPlan.from_json(payload["plan"]))
        self.enabled = tp is not None and tp > 1
        if not self.enabled:
            return
        self.te.initialize_ub(shape=self.shape_fn(tp), tp_size=tp, use_fp8=self.use_fp8, dtype=self.dtype,
                              ub_cfgs=self.ub_cfgs, bootstrap_backend=self.bootstrap_backend)
        self.active = True


class CudaGraphHook:
    """Megatron CUDA graphs: ``TECudaGraphHelper`` (``cuda_graph_impl="transformer_engine"``) or the
    module-level ``delete_cuda_graphs`` of the local implementation. Graphs are deleted with the
    old generation and re-captured after ``warmup`` eager iterations of the new one (the
    warm-up also absorbs the first iteration's allocator growth)."""

    def __init__(self, impl: str, helper_factory: Callable[[], object] | None = None, warmup: int = 1,
                 cuda_graphs=None):
        if impl not in ("transformer_engine", "local"):
            raise ValueError(f"unsupported cuda_graph_impl {impl!r}")
        self.impl = impl
        self.helper_factory = helper_factory
        self.warmup = warmup
        self._cg = cuda_graphs
        self.helper = None
        self.eager_left = 0
        self.captures = 0

    @property
    def cg(self):
        if self._cg is None:
            from megatron.core.transformer import cuda_graphs

            self._cg = cuda_graphs
        return self._cg

    def teardown(self) -> None:
        if self.impl == "local":
            self.cg.delete_cuda_graphs()
        elif self.helper is not None and self.helper.graphs_created():
            self.helper.delete_cuda_graphs()
        self.helper = None

    def rebuild(self, comms, payload: dict) -> None:
        if self.impl == "transformer_engine":
            self.helper = self.helper_factory()
        self.eager_left = self.warmup

    def after_iteration(self, iteration: int) -> None:
        if self.eager_left <= 0:
            return
        self.eager_left -= 1
        if self.eager_left == 0 and self.helper is not None:
            self.helper.create_cudagraphs()
            self.helper.cuda_graph_set_manual_hooks()
            self.captures += 1
