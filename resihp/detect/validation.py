"""Greyhound-style validation: pinpoint a slow GPU or link after a confirmed change point.

* compute: a short GEMM benchmark per suspect device against its healthy reference;
* links: P2P bandwidth probes scheduled in O(max degree) rounds of disjoint pairs
  (2 rounds for an even ring, 3 for an odd one), so the whole fabric is checked in a
  constant number of steps regardless of scale.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist


def gemm_tflops(n: int = 2048, iters: int = 10, device: str | None = None, dtype: torch.dtype | None = None) -> float:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = dtype or (torch.bfloat16 if dev.type == "cuda" else torch.float32)
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(2):
        a @ b
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return 2 * n ** 3 * iters / (time.perf_counter() - t0) / 1e12


def slowdown_vs_reference(measured: float, reference: float) -> float:
    """>1 means slower than the device's healthy reference."""
    return reference / max(measured, 1e-12)


def edge_rounds(edges: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Greedy edge coloring: each round is a set of disjoint links probed concurrently."""
    rounds: list[list[tuple[int, int]]] = []
    busy: list[set[int]] = []
    for a, b in sorted(edges, key=lambda e: (min(e), max(e))):
        for rnd, used in zip(rounds, busy):
            if a not in used and b not in used:
                rnd.append((a, b))
                used.update((a, b))
                break
        else:
            rounds.append([(a, b)])
            busy.append({a, b})
    return rounds


def ring_edges(order: list[int]) -> list[tuple[int, int]]:
    n = len(order)
    if n < 2:
        return []
    if n == 2:
        return [(order[0], order[1])]
    return [(order[i], order[(i + 1) % n]) for i in range(n)]


def probe_links(rank: int, rounds: list[list[tuple[int, int]]], mgr, nbytes: int = 8 << 20, repeats: int = 3,
                timeout: float = 60.0) -> dict[tuple[int, int], float]:
    """Measure GB/s on each link this rank belongs to (rounds must be identical on every rank)."""
    buf = torch.empty(nbytes // 4, dtype=torch.float32)
    out = {}
    for rnd in rounds:
        for a, b in rnd:
            if rank not in (a, b):
                continue
            peer = b if rank == a else a
            t0 = time.perf_counter()
            for i in range(repeats):
                if rank == a:
                    mgr.guarded_wait(dist.isend(buf, peer, tag=7000 + i), timeout=timeout, what="link probe")
                    mgr.guarded_wait(dist.irecv(buf, peer, tag=8000 + i), timeout=timeout, what="link probe")
                else:
                    mgr.guarded_wait(dist.irecv(buf, peer, tag=7000 + i), timeout=timeout, what="link probe")
                    mgr.guarded_wait(dist.isend(buf, peer, tag=8000 + i), timeout=timeout, what="link probe")
            elapsed = time.perf_counter() - t0
            out[(a, b)] = 2 * repeats * nbytes / elapsed / 1e9
    return out
