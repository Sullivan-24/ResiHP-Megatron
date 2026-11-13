"""Half-open interval arithmetic for sharded tensors.

Heterogeneous-TP P2P, shard-aligned gradient reduction across replicas with
different TP degrees, and optimizer-state resharding all reduce to computing
overlaps between two (or more) partitions of the same index range.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

Interval = tuple[int, int]


def even_split(n: int, k: int) -> list[Interval]:
    """Split ``[0, n)`` into ``k`` contiguous parts; the first ``n % k`` parts get one extra element."""
    if k <= 0:
        raise ValueError(f"number of parts must be positive, got {k}")
    if n < 0:
        raise ValueError(f"length must be non-negative, got {n}")
    base, rem = divmod(n, k)
    out: list[Interval] = []
    start = 0
    for i in range(k):
        end = start + base + (1 if i < rem else 0)
        out.append((start, end))
        start = end
    return out


def length(iv: Interval) -> int:
    return iv[1] - iv[0]


def intersect(a: Interval, b: Interval) -> Interval | None:
    start, end = max(a[0], b[0]), min(a[1], b[1])
    return (start, end) if start < end else None


def contains(outer: Interval, inner: Interval) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def overlaps(src: Sequence[Interval], dst: Sequence[Interval]) -> list[tuple[int, int, Interval]]:
    """All ``(src_index, dst_index, overlap)`` triples with a non-empty overlap, ordered by position."""
    out = []
    for i, a in enumerate(src):
        for j, b in enumerate(dst):
            o = intersect(a, b)
            if o is not None:
                out.append((i, j, o))
    out.sort(key=lambda t: (t[2][0], t[0], t[1]))
    return out


def refine(partitions: Iterable[Sequence[Interval]]) -> list[Interval]:
    """Coarsest partition that refines every input partition (all inputs must cover the same range)."""
    points: set[int] = set()
    spans = set()
    for part in partitions:
        if not part:
            continue
        spans.add((part[0][0], part[-1][1]))
        for s, e in part:
            points.add(s)
            points.add(e)
    if len(spans) > 1:
        raise ValueError(f"partitions cover different ranges: {sorted(spans)}")
    pts = sorted(points)
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1) if pts[i] < pts[i + 1]]


def locate(partition: Sequence[Interval], iv: Interval) -> int:
    """Index of the part that fully contains ``iv``; raises if ``iv`` straddles a boundary."""
    for i, part in enumerate(partition):
        if contains(part, iv):
            return i
    raise ValueError(f"{iv} is not contained in a single part of {list(partition)}")


def narrow(t: torch.Tensor, dim: int, iv: Interval, offset: int = 0) -> torch.Tensor:
    """View of ``t`` covering global interval ``iv`` along ``dim`` when ``t`` starts at global ``offset``."""
    return t.narrow(dim, iv[0] - offset, iv[1] - iv[0])


def is_partition(parts: Sequence[Interval], n: int) -> bool:
    if not parts:
        return n == 0
    if parts[0][0] != 0 or parts[-1][1] != n:
        return False
    return all(parts[i][1] == parts[i + 1][0] for i in range(len(parts) - 1))
