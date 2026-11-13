"""State redundancy tiers.

T0  live replicas: model parameters exist on every DP replica of a stage; optimizer
    state is replicated across distributed-optimizer instances
    (``num_distributed_optimizer_instances >= 2``). Zero rollback.
T1  host-memory snapshots every ``k`` iterations, replicated to a buddy on another
    node (Gemini/ByteRobust style). Rollback <= k iterations; covers losing every
    live copy of a shard (e.g. all DP replicas of a stage).
T2  persistent sliced checkpoints that reshard on load (see ``checkpoint.py``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import torch


def distopt_twin_holders(dp_members: list[str], num_instances: int) -> dict[str, list[str]]:
    """Holders of each distributed-optimizer shard under Megatron's partial DistOpt layout.

    The DP group is cut into ``num_instances`` contiguous instances; shard ``j`` lives on
    position ``j`` of every instance.
    """
    d = len(dp_members)
    if num_instances < 1 or d % num_instances:
        raise ValueError(f"DP size {d} not divisible by {num_instances} instances")
    per = d // num_instances
    return {f"shard{j}": [dp_members[i * per + j] for i in range(num_instances)] for j in range(per)}


@dataclass
class HolderMap:
    holders: dict[str, list[str]]

    def lost(self, failed: set[str]) -> list[str]:
        return sorted(k for k, hs in self.holders.items() if all(h in failed for h in hs))

    def survivors(self, key: str, failed: set[str]) -> list[str]:
        return [h for h in self.holders[key] if h not in failed]

    def node_diverse(self, node_of: Callable[[str], str]) -> bool:
        """True when every shard with >1 holder has holders on more than one node."""
        return all(len({node_of(h) for h in hs}) > 1 for hs in self.holders.values() if len(hs) > 1)


class HostSnapshotManager:
    """Asynchronous host-memory snapshots of a flat ``name -> tensor`` state dict."""

    def __init__(self, interval: int, keep: int = 1):
        self.interval = interval
        self.keep = keep
        self._snaps: list[tuple[int, dict[str, torch.Tensor]]] = []
        self._held: dict[str, tuple[int, dict[str, torch.Tensor]]] = {}
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._copy_done = None

    def due(self, iteration: int) -> bool:
        return self.interval > 0 and (iteration + 1) % self.interval == 0

    def snapshot(self, iteration: int, state: dict[str, torch.Tensor], blocking: bool = False) -> None:
        """Capture ``state`` as of now.

        CPU tensors are cloned synchronously. CUDA tensors are copied to pinned host
        memory on a side stream; ``fence()`` must run before the next in-place update
        so the copy reads the snapshot-time values.
        """
        self.wait()
        side = None
        if any(v.is_cuda for v in state.values()):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
        staged: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if v.is_cuda:
                with torch.cuda.stream(side):
                    buf = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
                    buf.copy_(v.detach(), non_blocking=True)
                staged[k] = buf
            else:
                staged[k] = v.detach().clone()
        if side is not None:
            self._copy_done = torch.cuda.Event()
            self._copy_done.record(side)

        def finish():
            if side is not None:
                side.synchronize()
            with self._lock:
                self._snaps.append((iteration, staged))
                self._snaps = self._snaps[-self.keep:]

        if blocking or side is None:
            finish()
        else:
            self._thread = threading.Thread(target=finish, daemon=True, name="snapshot")
            self._thread.start()

    def fence(self) -> None:
        """Order the next in-place update after any in-flight snapshot copy (GPU-side wait)."""
        if self._copy_done is not None:
            torch.cuda.current_stream().wait_event(self._copy_done)
            self._copy_done = None

    def maybe_snapshot(self, iteration: int, state_fn: Callable[[], dict[str, torch.Tensor]]) -> bool:
        if not self.due(iteration):
            return False
        self.snapshot(iteration, state_fn(), blocking=True)
        return True

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def latest(self) -> tuple[int, dict[str, torch.Tensor]] | None:
        self.wait()
        with self._lock:
            return self._snaps[-1] if self._snaps else None

    def at(self, iteration: int) -> dict[str, torch.Tensor] | None:
        self.wait()
        with self._lock:
            for it, s in self._snaps:
                if it == iteration:
                    return s
        return None

    def hold_for(self, owner: str, iteration: int, state: dict[str, torch.Tensor]) -> None:
        with self._lock:
            self._held[owner] = (iteration, state)

    def held(self, owner: str) -> tuple[int, dict[str, torch.Tensor]] | None:
        with self._lock:
            return self._held.get(owner)


def exchange_snapshot_with_buddy(mgr, comms, mine: tuple[int, dict[str, torch.Tensor]], send_to: str,
                                 recv_from: str, template: dict[str, torch.Tensor], group=None,
                                 timeout: float = 60.0) -> tuple[int, dict[str, torch.Tensor]]:
    """Ring replication: send my snapshot to ``send_to`` and keep ``recv_from``'s.

    ``template`` gives the shapes of ``recv_from``'s snapshot (identical keys for DP
    peers; supplied by the state registry otherwise).
    """
    import torch.distributed as dist

    it, state = mine
    dst, src = comms.rank_of(send_to), comms.rank_of(recv_from)
    header = torch.tensor([it], dtype=torch.int64)
    rheader = torch.zeros(1, dtype=torch.int64)
    works = [dist.isend(header, dst, group=group, tag=0), dist.irecv(rheader, src, group=group, tag=0)]
    recv = {k: torch.empty_like(v, device="cpu") for k, v in sorted(template.items())}
    for i, k in enumerate(sorted(state)):
        works.append(dist.isend(state[k].contiguous(), dst, group=group, tag=i + 1))
    for i, k in enumerate(sorted(recv)):
        works.append(dist.irecv(recv[k], src, group=group, tag=i + 1))
    for w in works:
        mgr.guarded_wait(w, timeout=timeout, what="snapshot replication")
    return int(rheader.item()), recv
