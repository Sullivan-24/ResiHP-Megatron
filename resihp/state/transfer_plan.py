"""Slice-level state-transfer planning between two plans.

For every interval a device must hold in the new plan:
* parts it already holds (and that are at the recovery iteration) are reused in place;
* the rest is cut on the boundaries of the surviving holders and each piece gets the
  closest, least-loaded source (same device > same node > same rack > other), so
  large moves are striped across DP replicas;
* pieces without a valid holder are reported as lost (fallback to T1/T2).

All ranks compute the identical plan from identical inputs; no coordinator round trip.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from resihp.state.layer_registry import Holdings, TensorSpec
from resihp.utils.intervals import intersect


@dataclass(frozen=True)
class SliceTransfer:
    key: str
    src: str
    dst: str
    interval: tuple[int, int]


@dataclass
class TransferPlan:
    transfers: list[SliceTransfer] = field(default_factory=list)
    reuse: list[tuple[str, str, tuple[int, int]]] = field(default_factory=list)
    lost: list[tuple[str, tuple[int, int]]] = field(default_factory=list)
    load: dict[str, int] = field(default_factory=dict)

    def bytes_moved(self, specs: dict[str, TensorSpec], bytes_per_elem: int, kinds: int) -> int:
        total = 0
        for t in self.transfers:
            s = specs[t.key]
            n = 1
            for d in s.local_shape(t.interval):
                n *= d
            total += n * bytes_per_elem * (kinds if s.trainable else 1)
        return total


def _subtract(iv: tuple[int, int], cuts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = [iv]
    for c in cuts:
        nxt = []
        for a in out:
            o = intersect(a, c)
            if o is None:
                nxt.append(a)
                continue
            if a[0] < o[0]:
                nxt.append((a[0], o[0]))
            if o[1] < a[1]:
                nxt.append((o[1], a[1]))
        out = nxt
    return out


def plan_transfers(old: Holdings, new: Holdings, *, valid_sources: set[str], node_of: dict[str, str],
                   rack_of: dict[str, str] | None = None, chunk: int | None = None) -> TransferPlan:
    rack_of = rack_of or {}
    plan = TransferPlan()
    load: dict[str, int] = defaultdict(int)

    def locality(src: str, dst: str) -> int:
        if node_of.get(src) == node_of.get(dst):
            return 0
        if rack_of.get(src) is not None and rack_of.get(src) == rack_of.get(dst):
            return 1
        return 2

    for key in sorted(new):
        old_h = [(d, iv) for d, iv in old.get(key, []) if d in valid_sources]
        for dst, want in sorted(new[key]):
            own = [iv for d, iv in old_h if d == dst]
            for iv in own:
                o = intersect(iv, want)
                if o is not None:
                    plan.reuse.append((dst, key, o))
            for rest in _subtract(want, own):
                holders = [(d, iv) for d, iv in old_h if d != dst and intersect(iv, rest) is not None]
                pts = {rest[0], rest[1]}
                for _, iv in holders:
                    pts.update(x for x in iv if rest[0] < x < rest[1])
                pts = sorted(pts)
                for piece in zip(pts[:-1], pts[1:]):
                    cands = sorted({d for d, iv in holders if iv[0] <= piece[0] and piece[1] <= iv[1]})
                    if not cands:
                        plan.lost.append((key, piece))
                        continue
                    bounds = [piece]
                    if chunk and piece[1] - piece[0] > chunk:
                        bounds = [(s, min(s + chunk, piece[1])) for s in range(piece[0], piece[1], chunk)]
                    for b in bounds:
                        src = min(cands, key=lambda d: (locality(d, dst), load[d], d))
                        load[src] += b[1] - b[0]
                        plan.transfers.append(SliceTransfer(key, src, dst, b))
    plan.transfers.sort(key=lambda t: (t.key, t.interval, t.src, t.dst))
    plan.load = dict(load)
    return plan
