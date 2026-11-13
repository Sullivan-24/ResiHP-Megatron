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


def _shrink_after_death(rank, world, port, store_addr, abort_first):
    from resihp.comm import GroupSpec, ResilientCommManager

    torch.cuda.set_device(rank)
    mgr = ResilientCommManager("nccl", device=f"cuda:{rank}", timeout_s=30)
    ids = [f"w{i}" for i in range(world)]
    specs = [GroupSpec("tp/0", (0, 1), "tp"), GroupSpec("tp/1", (2, 3), "tp"),
             GroupSpec("dp/0", (0, 2), "dp"), GroupSpec("dp/1", (1, 3), "dp")]
    comms1 = mgr.build(1, ids, ids[rank], specs, store_addr=store_addr,
                       store_prefix=f"nccl{int(abort_first)}/gen1/")
    if rank == 3:
        os._exit(3)
    time.sleep(1.0)
    if abort_first:
        mgr.abort()
    survivors = ids[:3]
    t0 = time.monotonic()
    comms = mgr.build(2, survivors, ids[rank], [GroupSpec("tp/0", (0, 1), "tp")],
                      store_addr=store_addr, store_prefix=f"nccl{int(abort_first)}/gen2/")
    rebuild = time.monotonic() - t0
    t = torch.ones(1, device="cuda")
    dist.all_reduce(t)
    torch.cuda.synchronize()
    tp = comms.group("tp/0")
    if tp is not None:
        g = torch.ones(1, device="cuda")
        dist.all_reduce(g, group=tp)
        torch.cuda.synchronize()
        tp_sum = g.item()
    else:
        tp_sum = None
    subgroup_path = mgr.subgroup_path()
    mgr.destroy()
    del comms1
    return {"path": comms.path, "sum": t.item(), "tp_sum": tp_sum, "rebuild_s": rebuild,
            "subgroup_path": subgroup_path}


# A failure notice makes the worker abort its communicators to unblock the training thread, and an
# aborted NCCL communicator can no longer be shrunk; the fast path is for planned reconfigurations.
@pytest.mark.parametrize("abort_first,expected", [(False, "shrink"), (True, "fresh")])
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
def test_rebuild_after_rank_death(store, abort_first, expected):
    caps = __import__("resihp.comm", fromlist=["ResilientCommManager"]).ResilientCommManager.capabilities()
    if not (caps["shrink_group"] and tuple(caps["nccl_version"] or (0,)) >= (2, 27, 3)):
        expected = "fresh"
    out = spawn(_shrink_after_death, 4, store, abort_first, timeout=180, expect_dead={3})
    for res in out.values():
        assert res["sum"] == 3.0, res
        assert res["tp_sum"] in (2.0, None), res
        assert res["path"] == expected, res
        assert res["subgroup_path"] == "split", "NCCL subgroups must come from ncclCommSplit"
