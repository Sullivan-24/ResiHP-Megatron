"""Pipeline-boundary transfers between TP groups of arbitrary (different) sizes.

Interval-overlap routing on the token dimension:

* sequence parallel: sender rank ``i`` owns token rows ``split(n, k_src)[i]`` and receiver
  rank ``j`` needs ``split(n, k_dst)[j]``; each overlap is sent point to point and no
  collective is needed at the receiver;
* replicated activations (no SP): every sender holds the full tensor, so sender ``i`` is
  only *responsible* for ``split(n, k_src)[i]``; each element crosses the network once
  and receivers reassemble the full tensor with one all-gather inside their own
  (NVLink) TP group. This generalizes the paper's scatter/all-gather rule to any pair
  of TP degrees, including non powers of two.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from resihp.utils.intervals import even_split, overlaps


@dataclass(frozen=True)
class Route:
    src: int
    dst: int
    rows: tuple[int, int]


def boundary_routes(src_ranks: tuple[int, ...], dst_ranks: tuple[int, ...], n_rows: int) -> list[Route]:
    src_parts = even_split(n_rows, len(src_ranks))
    dst_parts = even_split(n_rows, len(dst_ranks))
    return [Route(src_ranks[i], dst_ranks[j], iv) for i, j, iv in overlaps(src_parts, dst_parts)]


def local_rows(n_rows: int, group_ranks: tuple[int, ...], rank: int) -> tuple[int, int]:
    return even_split(n_rows, len(group_ranks))[group_ranks.index(rank)]


def channel_name(src: int, dst: int) -> str:
    return f"p2p/{src}>{dst}"


class P2PEngine:
    """Executes boundary transfers. ``channels`` maps ``(src, dst)`` to a dedicated 2-rank
    group per direction (required on NCCL); without it the world group plus tags is used."""

    def __init__(self, rank: int, mgr, *, channels: dict[tuple[int, int], object] | None = None,
                 timeout: float = 120.0, device: str = "cpu"):
        self.rank = rank
        self.mgr = mgr
        self.channels = channels
        self.timeout = timeout
        self.device = device

    def _group(self, src: int, dst: int):
        if self.channels is None:
            return None
        return self.channels[(src, dst)]

    def send(self, local: torch.Tensor, src_ranks: tuple[int, ...], dst_ranks: tuple[int, ...], n_rows: int,
             sequence_parallel: bool, tag: int) -> list:
        """Non-blocking; ``local`` is this rank's shard (SP) or the full tensor. Returns works to wait later."""
        works = []
        mine = local_rows(n_rows, src_ranks, self.rank)
        offset = mine[0] if sequence_parallel else 0
        for i, route in enumerate(boundary_routes(src_ranks, dst_ranks, n_rows)):
            if route.src != self.rank:
                continue
            piece = local.narrow(0, route.rows[0] - offset, route.rows[1] - route.rows[0]).contiguous()
            works.append((dist.isend(piece, route.dst, group=self._group(route.src, route.dst),
                                     tag=self._tag(tag, i)), piece))
        return works

    def recv(self, n_rows: int, row_shape: tuple[int, ...], dtype: torch.dtype, src_ranks: tuple[int, ...],
             dst_ranks: tuple[int, ...], sequence_parallel: bool, tag: int, tp_group=None) -> torch.Tensor:
        mine = local_rows(n_rows, dst_ranks, self.rank)
        buf = torch.empty((mine[1] - mine[0],) + tuple(row_shape), dtype=dtype, device=self.device)
        pending = []
        for i, route in enumerate(boundary_routes(src_ranks, dst_ranks, n_rows)):
            if route.dst != self.rank:
                continue
            piece = torch.empty((route.rows[1] - route.rows[0],) + tuple(row_shape), dtype=dtype,
                                device=self.device)
            pending.append((dist.irecv(piece, route.src, group=self._group(route.src, route.dst),
                                       tag=self._tag(tag, i)), piece, route.rows))
        for work, piece, rows in pending:
            self.mgr.guarded_wait(work, timeout=self.timeout, what="boundary recv")
            buf.narrow(0, rows[0] - mine[0], rows[1] - rows[0]).copy_(piece)
        if sequence_parallel or len(dst_ranks) == 1:
            return buf
        return self._gather_rows(buf, n_rows, dst_ranks, tp_group)

    def _gather_rows(self, mine: torch.Tensor, n_rows: int, ranks: tuple[int, ...], group) -> torch.Tensor:
        parts = even_split(n_rows, len(ranks))
        width = max(e - s for s, e in parts)
        padded = torch.zeros((width,) + tuple(mine.shape[1:]), dtype=mine.dtype, device=mine.device)
        padded[: mine.shape[0]].copy_(mine)
        out = [torch.empty_like(padded) for _ in ranks]
        self.mgr.guarded_wait(dist.all_gather(out, padded, group=group, async_op=True), timeout=self.timeout,
                              what="boundary all-gather")
        return torch.cat([o[: e - s] for o, (s, e) in zip(out, parts)])

    def wait_sends(self, works: list) -> None:
        for work, _piece in works:
            self.mgr.guarded_wait(work, timeout=self.timeout, what="boundary send")

    @staticmethod
    def _tag(base: int, index: int) -> int:
        return (base * 4096 + index) % (2 ** 31 - 1)
