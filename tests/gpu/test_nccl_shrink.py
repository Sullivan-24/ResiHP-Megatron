"""NCCL fast path: survivors-only ncclCommShrink + ncclCommSplit-derived subgroups (needs >= 4 GPUs)."""

import os
import time

import pytest
import torch
import torch.distributed as dist

from tests.dist_utils import spawn

pytestmark = pytest.mark.gpu


@pytest.fixture
def store():
    srv = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    yield ("127.0.0.1", srv.port)
    del srv


def _shrink_after_death(rank, world, port, store_addr):
    from resihp.comm import GroupSpec, ResilientCommManager

    torch.cuda.set_device(rank)
    mgr = ResilientCommManager("nccl", device=f"cuda:{rank}", timeout_s=30)
    ids = [f"w{i}" for i in range(world)]
    specs = [GroupSpec("tp/0", (0, 1), "tp"), GroupSpec("tp/1", (2, 3), "tp"),
             GroupSpec("dp/0", (0, 2), "dp"), GroupSpec("dp/1", (1, 3), "dp")]
    mgr.build(1, ids, ids[rank], specs, store_addr=store_addr, store_prefix="nccl/gen1/")
    if rank == 3:
        os._exit(3)
    time.sleep(1.0)
    mgr.abort()
    survivors = ids[:3]
    t0 = time.monotonic()
    comms = mgr.build(2, survivors, ids[rank], [GroupSpec("tp/0", (0, 1), "tp")],
                      store_addr=store_addr, store_prefix="nccl/gen2/")
    rebuild = time.monotonic() - t0
    t = torch.ones(1, device="cuda")
    dist.all_reduce(t)
    torch.cuda.synchronize()
    mgr.destroy()
    return {"path": comms.path, "sum": t.item(), "rebuild_s": rebuild}


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
def test_shrink_path_after_rank_death(store):
    out = spawn(_shrink_after_death, 4, store, timeout=180, expect_dead={3})
    caps = __import__("resihp.comm", fromlist=["ResilientCommManager"]).ResilientCommManager.capabilities()
    expected_path = "shrink" if caps["shrink_group"] and tuple(caps["nccl_version"] or (0,)) >= (2, 27, 3) else "fresh"
    for res in out.values():
        assert res["sum"] == 3.0
        assert res["path"] == expected_path
