import pytest

from tests.dist_utils import spawn

pytestmark = pytest.mark.dist


def _allreduce(rank, world, port):
    import torch
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    t = torch.tensor([float(rank + 1)])
    dist.all_reduce(t)
    dist.destroy_process_group()
    return t.item()


def test_gloo_allreduce_on_localhost():
    out = spawn(_allreduce, 3, timeout=60)
    assert out == {0: 6.0, 1: 6.0, 2: 6.0}
