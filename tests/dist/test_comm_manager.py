"""Multi-process tests of generation rebuilds on gloo (the fresh-init path; NCCL shrink is under tests/gpu)."""

import os
import threading
import time

import pytest
import torch.distributed as dist

from tests.dist_utils import spawn

pytestmark = pytest.mark.dist

SPECS4 = [
    ("tp/r0", (0, 1), "tp"), ("tp/r1", (2, 3), "tp"),
    ("dp/t0", (0, 2), "dp"), ("dp/t1", (1, 3), "dp"),
    ("dpshard/a", (0, 2), "dpshard"), ("dpshard/b", (0, 3), "dpshard"), ("dpshard/c", (1, 3), "dpshard"),
]


def _specs(raw):
    from resihp.comm import GroupSpec

    return [GroupSpec(n, r, k) for n, r, k in raw]


@pytest.fixture
def store():
    srv = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    yield ("127.0.0.1", srv.port)
    del srv


def _build_and_check(rank, world, port, store_addr, mode):
    import torch

    from resihp.comm import ResilientCommManager

    mgr = ResilientCommManager("gloo", timeout_s=20, subgroup_mode=mode)
    ids = [f"w{i}" for i in range(world)]
    comms = mgr.build(1, ids, ids[rank], _specs(SPECS4), store_addr=store_addr, store_prefix=f"t-{mode}/gen1/")
    sums = {}
    for name in sorted(comms.specs):
        pg = comms.group(name)
        if pg is None:
            continue
        t = torch.tensor([float(rank)])
        mgr.guarded_wait(dist.all_reduce(t, group=pg, async_op=True), timeout=10)
        sums[name] = t.item()
    mgr.destroy()
    return sums


@pytest.mark.parametrize("mode", ["split", "local"])
def test_build_validates_and_groups_are_correct(store, mode):
    out = spawn(_build_and_check, 4, store, mode, timeout=90)
    expected = {n: float(sum(r)) for n, r, _ in SPECS4}
    for rank, sums in out.items():
        for name, ranks, _ in SPECS4:
            if rank in ranks:
                assert sums[name] == expected[name], (rank, name)
            else:
                assert name not in sums


def _survive_peer_death(rank, world, port, store_addr):
    import torch

    from resihp.comm import CommTimeout, ResilientCommManager

    mgr = ResilientCommManager("gloo", timeout_s=10, validate_timeout_s=10)
    ids = [f"w{i}" for i in range(world)]
    mgr.build(1, ids, ids[rank], _specs(SPECS4), store_addr=store_addr, store_prefix="death/gen1/")
    if rank == 3:
        os._exit(17)
    t0 = time.monotonic()
    err = None
    try:
        for _ in range(3):
            t = torch.ones(1)
            mgr.guarded_wait(dist.all_reduce(t, async_op=True), timeout=12)
            time.sleep(0.2)
    except (RuntimeError, CommTimeout) as e:
        err = type(e).__name__
    detect = time.monotonic() - t0
    survivors = ids[:3]
    specs = _specs([("tp/r0", (0, 1), "tp"), ("dp/t0", (0, 2), "dp")])
    comms = mgr.build(2, survivors, ids[rank], specs, store_addr=store_addr, store_prefix="death/gen2/")
    t = torch.ones(1)
    mgr.guarded_wait(dist.all_reduce(t, async_op=True), timeout=10)
    mgr.destroy()
    return {"err": err, "detect_s": detect, "world_sum": t.item(), "path": comms.path, "rank": comms.rank}


def test_survivors_rebuild_after_peer_death(store):
    out = spawn(_survive_peer_death, 4, store, timeout=120, expect_dead={3})
    assert sorted(out) == [0, 1, 2]
    for r, res in out.items():
        assert res["err"] is not None, "the dead peer must surface as an error, not a hang"
        assert res["world_sum"] == 3.0 and res["path"] == "fresh" and res["rank"] == r


def _abort_unblocks(rank, world, port, store_addr):
    import torch

    from resihp.comm import CommAborted, ResilientCommManager

    mgr = ResilientCommManager("gloo", timeout_s=30)
    ids = ["a", "b"]
    mgr.build(1, ids, ids[rank], [], store_addr=store_addr, store_prefix="abort/gen1/")
    result = {}
    if rank == 0:
        threading.Timer(0.5, mgr.abort).start()
        t0 = time.monotonic()
        try:
            mgr.guarded_wait(dist.all_reduce(torch.ones(1), async_op=True), timeout=60)
        except CommAborted:
            result["abort_latency"] = time.monotonic() - t0
    else:
        time.sleep(1.5)
    comms = mgr.build(2, ids, ids[rank], [], store_addr=store_addr, store_prefix="abort/gen2/")
    t = torch.ones(1)
    mgr.guarded_wait(dist.all_reduce(t, async_op=True), timeout=10)
    result["gen2_sum"] = t.item()
    result["gen"] = comms.generation
    mgr.destroy()
    return result


def test_abort_unblocks_a_stuck_collective_then_rebuild(store):
    out = spawn(_abort_unblocks, 2, store, timeout=90)
    assert out[0]["abort_latency"] < 2.0
    assert out[0]["gen2_sum"] == out[1]["gen2_sum"] == 2.0
