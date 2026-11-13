import pytest
import torch.distributed as dist

from tests.dist_utils import spawn

pytestmark = pytest.mark.dist


@pytest.fixture
def store():
    srv = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    yield ("127.0.0.1", srv.port)
    del srv


def _probe(rank, world, port, store_addr):
    from resihp.comm import ResilientCommManager
    from resihp.detect.validation import edge_rounds, gemm_tflops, probe_links, ring_edges

    ids = [f"w{i}" for i in range(world)]
    mgr = ResilientCommManager("gloo", timeout_s=30)
    mgr.build(1, ids, ids[rank], [], store_addr=store_addr, store_prefix="probe/gen1/")
    rounds = edge_rounds(ring_edges(list(range(world))))
    bw = probe_links(rank, rounds, mgr, nbytes=1 << 20, repeats=2)
    tf = gemm_tflops(n=256, iters=3, device="cpu")
    mgr.destroy()
    return {"bw": bw, "rounds": len(rounds), "tflops": tf}


def test_ring_link_probe_covers_every_link(store):
    out = spawn(_probe, 5, store, timeout=120)
    links = set()
    for rank, res in out.items():
        assert res["rounds"] == 3 and res["tflops"] > 0
        for (a, b), gbps in res["bw"].items():
            assert rank in (a, b) and gbps > 0
            links.add((a, b))
    assert len(links) == 5
