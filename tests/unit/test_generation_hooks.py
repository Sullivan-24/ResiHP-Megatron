"""Per-generation rebuild hooks: decisions, ordering inside the resilient loop, failure handling."""

import pytest

from resihp.plan.hetero_plan import HeteroParallelPlan, StageGroup
from resihp.runtime.generation_hooks import CudaGraphHook, GenerationHooks, TEUserBufferHook, uniform_contiguous_tp
from resihp.train.resilient_loop import ResilientTrainer


def plan(groups):
    world = tuple(sorted({d for g in groups for d in g.devices}, key=lambda d: int(d[1:])))
    stages = max(g.stage for g in groups) + 1
    return HeteroParallelPlan(1, world, 4, stages, 2, tuple(groups), 2, tuple((0, 1) for _ in range(stages)))


UNIFORM = plan([StageGroup(0, 0, ("w0", "w1"), (0, 2)), StageGroup(1, 0, ("w2", "w3"), (0, 2)),
                StageGroup(0, 1, ("w4", "w5"), (2, 4)), StageGroup(1, 1, ("w6", "w7"), (2, 4))])
SHRUNK = plan([StageGroup(0, 0, ("w0", "w1"), (0, 2)), StageGroup(1, 0, ("w2",), (0, 2)),
               StageGroup(0, 1, ("w3", "w4"), (2, 4)), StageGroup(1, 1, ("w5", "w6"), (2, 4))])


class FakeTE:
    def __init__(self):
        self.calls = []

    def initialize_ub(self, **kw):
        self.calls.append(("init", kw["tp_size"], tuple(kw["shape"])))

    def destroy_ub(self):
        self.calls.append(("destroy",))


class FakeHelper:
    def __init__(self, log):
        self.log, self.created = log, False

    def graphs_created(self):
        return self.created

    def create_cudagraphs(self):
        self.created = True
        self.log.append("capture")

    def cuda_graph_set_manual_hooks(self):
        self.log.append("hooks")

    def delete_cuda_graphs(self):
        self.created = False
        self.log.append("delete")


def test_userbuffers_follow_the_tp_layout():
    assert uniform_contiguous_tp(UNIFORM) == 2 and uniform_contiguous_tp(SHRUNK) is None
    te = FakeTE()
    hook = TEUserBufferHook(lambda tp: [4096, 1024], te_base=te)
    hook.rebuild(None, {"plan": UNIFORM.to_json()})
    assert hook.enabled and te.calls == [("init", 2, (4096, 1024))]
    hook.teardown()
    hook.rebuild(None, {"plan": SHRUNK.to_json()})
    assert not hook.enabled and not hook.active
    hook.teardown()
    assert te.calls == [("init", 2, (4096, 1024)), ("destroy",)]


def test_cuda_graphs_are_deleted_and_recaptured_after_warmup():
    log = []
    hook = CudaGraphHook("transformer_engine", helper_factory=lambda: FakeHelper(log), warmup=2)
    hook.rebuild(None, {})
    hook.after_iteration(0)
    assert log == []
    hook.after_iteration(1)
    assert log == ["capture", "hooks"]
    hook.after_iteration(2)
    hook.teardown()
    assert log[-1] == "delete" and hook.helper is None
    with pytest.raises(ValueError):
        CudaGraphHook("full_iteration")


class Recorder:
    def __init__(self, log, fail_rebuild_at=None):
        self.log, self.fail_at, self.rebuilds = log, fail_rebuild_at, 0

    def teardown(self):
        self.log.append("teardown")

    def rebuild(self, comms, payload):
        self.rebuilds += 1
        if self.rebuilds == self.fail_at:
            raise RuntimeError("userbuffer bootstrap failed")
        self.log.append(f"rebuild:{comms}")

    def after_iteration(self, it):
        self.log.append(f"it{it}")


class FakeClient:
    worker_id = "w0"

    def __init__(self, log, preps, reconfig_after):
        self.log, self.preps, self.reconfig_after = log, list(preps), reconfig_after
        self.it = -1

    def add_abort_callback(self, fn): pass
    def set_recovery_info(self, **kw): pass
    def set_progress(self, **kw): self.it = kw.get("iteration", self.it)
    def immediate_abort_pending(self): return False
    def reconfig_pending(self): return self.it == self.reconfig_after and len(self.preps) > 0
    def wait_prepare(self, timeout): return self.preps.pop(0)
    def send_ready(self, gen, **kw): self.log.append(f"ready:{gen}")
    def wait_commit(self, gen, timeout): pass
    def report(self, kind, **kw): self.log.append(f"report:{kind}")
    def finish(self, it, timeout): return "shutdown"


class FakeComms:
    build_seconds, path = 0.0, "fake"

    def __init__(self, gen):
        self.gen = gen

    def __str__(self):
        return f"comms{self.gen}"


class FakeMgr:
    def __init__(self, log):
        self.log = log

    def build(self, gen, world, me, specs, **kw):
        self.log.append(f"build:{gen}")
        return FakeComms(gen)

    def destroy(self): self.log.append("destroy")
    def abort(self): pass


class FakeBinding:
    def group_specs(self, world, payload): return []
    def recover(self, comms, payload, txm): pass
    def build_runtime(self, comms, payload): return self
    def run_iteration(self, it): return 0.0
    def optimizer_step(self): pass
    def after_commit(self, it, comms): pass
    def try_resume(self, txm): pass


def prep(gen):
    return {"gen": gen, "world": ["w0"], "payload": {}, "store_addr": None, "store_prefix": f"g{gen}/"}


@pytest.mark.parametrize("fail_at", [None, 2])
def test_trainer_tears_down_before_rebuilding_and_retries_failed_rebuilds(fail_at):
    log = []
    preps = [prep(1), prep(2)] + ([prep(3)] if fail_at else [])
    trainer = ResilientTrainer(FakeClient(log, preps, reconfig_after=0), FakeMgr(log), FakeBinding(),
                               hooks=GenerationHooks([Recorder(log, fail_at)]))
    trainer.run(2)
    if fail_at is None:
        assert log == ["build:1", "rebuild:comms1", "ready:1", "it0", "teardown", "build:2", "rebuild:comms2",
                       "ready:2", "it1"]
    else:
        assert log == ["build:1", "rebuild:comms1", "ready:1", "it0", "teardown", "build:2", "teardown",
                       "report:build_failed", "build:3", "rebuild:comms3", "ready:3", "it1"]
