"""Roll-forward state copy: a lagging (or newly joined) replica receives model + optimizer state."""

import pytest
import torch.distributed as dist

from tests.dist_utils import spawn

pytestmark = pytest.mark.dist


@pytest.fixture
def store():
    srv = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    yield ("127.0.0.1", srv.port)
    del srv


def _sync(rank, world, port, store_addr):
    import torch

    from resihp.comm import ResilientCommManager
    from resihp.runtime.dp_runtime import sync_full_state
    from resihp.runtime.optim import Adam
    from resihp.runtime.reference_model import RefModelConfig, RefStage

    ids = ["w0", "w1", "w2"]
    mgr = ResilientCommManager("gloo", timeout_s=20)
    comms = mgr.build(1, ids, ids[rank], [], store_addr=store_addr, store_prefix="sync/gen1/")
    model = RefStage(RefModelConfig(), 0, 2)
    opt = Adam(model.named_state(), lr=0.1)
    if rank < 2:
        for _ in range(3):
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            opt.step()
    adopted = sync_full_state(mgr, comms, ids[rank], {"w2": "w0"}, opt, committed_iteration=2 if rank < 2 else -1)
    out = {k: v.clone() for k, v in opt.state_tensors().items()}
    mgr.destroy()
    return {"adopted": adopted, "step": opt.step_count, "state": out}


def test_lagging_replica_receives_full_state(store):
    out = spawn(_sync, 3, store, timeout=90)
    assert out[2]["adopted"] == 2 and out[2]["step"] == 3
    assert out[0]["adopted"] is None
    for k, v in out[0]["state"].items():
        assert (out[2]["state"][k] == v).all(), k
