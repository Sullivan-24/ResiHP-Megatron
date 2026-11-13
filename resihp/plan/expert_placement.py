"""Expert placement for resilient expert parallelism (Lazarus / EPLB style).

Every EP group hosts each logical expert at least once; spare slots replicate the
hottest experts. Physical copies are packed by (load / device speed) with the
constraint that copies of one expert land on distinct devices, and incumbents are
kept when that costs little, so a reconfiguration moves as little expert state as
possible. An expert's gradient is summed over all of its holders (replicas inside
an EP group and copies in other EP groups), so the expert data-parallel groups are
derived from the placement rather than from a fixed rank layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from resihp.plan.planner import PlanInfeasible


@dataclass(frozen=True)
class ExpertPlacement:
    num_experts: int
    groups: tuple[tuple[str, ...], ...]
    slots: dict[str, tuple[int, ...]]

    def holders(self, expert: int) -> list[str]:
        return sorted(d for g in self.groups for d in g if expert in self.slots[d])

    def group_holders(self, group: int, expert: int) -> list[str]:
        return [d for d in self.groups[group] if expert in self.slots[d]]

    def group_of(self, device: str) -> int:
        return next(i for i, g in enumerate(self.groups) if device in g)

    def validate(self) -> None:
        for i, g in enumerate(self.groups):
            hosted = {e for d in g for e in self.slots[d]}
            if hosted != set(range(self.num_experts)):
                raise ValueError(f"EP group {i} misses experts {set(range(self.num_experts)) - hosted}")
            for d in g:
                if len(set(self.slots[d])) != len(self.slots[d]):
                    raise ValueError(f"{d} hosts duplicate copies")

    def to_json(self) -> dict:
        return {"num_experts": self.num_experts, "groups": [list(g) for g in self.groups],
                "slots": {d: list(v) for d, v in self.slots.items()}}

    @staticmethod
    def from_json(d: dict) -> "ExpertPlacement":
        return ExpertPlacement(d["num_experts"], tuple(tuple(g) for g in d["groups"]),
                               {k: tuple(v) for k, v in d["slots"].items()})


def replica_counts(loads: list[float], total_slots: int, max_copies: int | None = None) -> list[int]:
    """Every expert once; each extra slot goes to the expert with the highest load per copy
    (copies of one expert never exceed ``max_copies``, the number of devices that can host it)."""
    if total_slots < len(loads):
        raise PlanInfeasible(f"{total_slots} slots cannot host {len(loads)} experts")
    cap = max_copies or total_slots
    counts = [1] * len(loads)
    for _ in range(total_slots - len(loads)):
        open_ = [i for i in range(len(loads)) if counts[i] < cap]
        if not open_:
            break
        e = max(open_, key=lambda i: (loads[i] / counts[i], -i))
        counts[e] += 1
    return counts


def group_time(slots: dict[str, tuple[int, ...]], loads: list[float], speeds: dict[str, float]) -> float:
    """Expert-compute time of the slowest device when each expert's tokens are split across its
    holders in proportion to speed (so every holder of ``e`` spends ``load_e / sum(speed)``)."""
    holders: dict[int, float] = {}
    for d, es in slots.items():
        for e in es:
            holders[e] = holders.get(e, 0.0) + max(speeds.get(d, 1.0), 1e-9)
    return max((sum(loads[e] / holders[e] for e in es) for es in slots.values()), default=0.0)


def _pack(devices, speed, loads, counts, slots_per_device, prev, stickiness):
    copies = sorted(((counts[e], loads[e] / counts[e], e) for e in range(len(loads)) for _ in range(counts[e])),
                    key=lambda x: (-x[0], -x[1], x[2]))
    total = sum(w for _, w, _ in copies)
    share = {d: total * speed[d] / sum(speed.values()) for d in devices}
    hosted: dict[str, list[int]] = {d: [] for d in devices}
    work = {d: 0.0 for d in devices}
    for _, w, e in copies:
        cands = [d for d in devices if len(hosted[d]) < slots_per_device and e not in hosted[d]]
        if not cands:
            return None
        cost = {d: (work[d] + w) / speed[d] for d in cands}
        keep = [d for d in cands if e in prev.get(d, ()) and work[d] + w <= share[d] * (1 + stickiness) + 1e-12]
        pick = min(keep or cands, key=lambda d: (cost[d], d))
        hosted[pick].append(e)
        work[pick] += w
    return {d: tuple(sorted(v)) for d, v in hosted.items()}


def place_group(devices: list[str], speeds: dict[str, float], loads: list[float], slots_per_device: int,
                prev: dict[str, tuple[int, ...]] | None = None, stickiness: float = 0.1) -> dict[str, tuple[int, ...]]:
    """Search over the number of replica slots to use: EPLB replica counts for each budget,
    most-constrained-first packing (incumbents kept while within ``stickiness`` of their
    speed share, otherwise least load/speed), scored by the exact speed-proportional time;
    ties go to fewer moved copies, then fewer copies."""
    prev = prev or {}
    speed = {d: max(speeds.get(d, 1.0), 1e-9) for d in devices}
    E = len(loads)
    best = None
    for budget in range(E, len(devices) * slots_per_device + 1):
        counts = replica_counts(loads, budget, max_copies=len(devices))
        slots = _pack(devices, speed, loads, counts, slots_per_device, prev, stickiness)
        if slots is None:
            continue
        moved = sum(1 for d, es in slots.items() for e in es if e not in prev.get(d, ()))
        key = (round(group_time(slots, loads, speed), 12), moved, budget)
        if best is None or key < best[0]:
            best = (key, slots)
    if best is None:
        raise PlanInfeasible(f"cannot place {E} experts on {len(devices)} devices x {slots_per_device} slots")
    return best[1]


def plan_expert_placement(ep_groups: list[list[str]], num_experts: int, *, speeds: dict[str, float] | None = None,
                          loads: list[float] | None = None, prev: ExpertPlacement | None = None,
                          max_slots: int | None = None, slots_per_device: int | None = None) -> ExpertPlacement:
    speeds = speeds or {}
    loads = loads or [1.0] * num_experts
    prev_slots = prev.slots if prev else {}
    slots: dict[str, tuple[int, ...]] = {}
    for g in ep_groups:
        need = slots_per_device or math.ceil(num_experts / len(g))
        if max_slots is not None and len(g) * max_slots < num_experts:
            raise PlanInfeasible(f"EP group of {len(g)} devices x {max_slots} slots cannot host {num_experts} experts")
        if max_slots is not None:
            need = min(need, max_slots)
        slots.update(place_group(list(g), speeds, loads, need, prev_slots))
    out = ExpertPlacement(num_experts, tuple(tuple(g) for g in ep_groups), slots)
    out.validate()
    return out


def expert_dp_groups(placement: ExpertPlacement) -> dict[tuple[str, ...], list[int]]:
    """Experts with identical holder sets share one gradient all-reduce group."""
    out: dict[tuple[str, ...], list[int]] = {}
    for e in range(placement.num_experts):
        out.setdefault(tuple(placement.holders(e)), []).append(e)
    return out


def smooth_weighted_round_robin(holders: list[str], speeds: dict[str, float], n: int) -> list[str]:
    """Deterministic speed-proportional assignment of ``n`` tokens to an expert's replicas."""
    if len(holders) == 1:
        return holders * n
    weights = {h: max(1, round(10 * speeds.get(h, 1.0))) for h in holders}
    total = sum(weights.values())
    current = {h: 0 for h in holders}
    out = []
    for _ in range(n):
        for h in holders:
            current[h] += weights[h]
        pick = max(holders, key=lambda h: (current[h], h))
        current[pick] -= total
        out.append(pick)
    return out


def moved_experts(old: ExpertPlacement, new: ExpertPlacement) -> int:
    return sum(1 for d, es in new.slots.items() for e in es if e not in old.slots.get(d, ()))
