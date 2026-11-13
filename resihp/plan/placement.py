"""Initial placement of (replica, stage) TP groups onto NVLink domains.

``spread`` interleaves nodes across racks and assigns stage-major, so the DP
replicas of each stage land on different nodes (and racks when possible): a single
node failure then never removes every live copy of a stage, which keeps recovery on
the zero-rollback T0 path. Megatron's default rank order (tp fastest, then dp, then
pp) packs replicas of a stage onto the same node.
"""

from __future__ import annotations

from resihp.plan.topology import ClusterTopology

Key = tuple[int, int]


def _slots(topology: ClusterTopology, tp: int) -> dict[str, list[tuple[str, ...]]]:
    out = {}
    for node, devs in topology.nodes().items():
        ids = [d.id for d in devs]
        out[node] = [tuple(ids[i:i + tp]) for i in range(0, len(ids) - tp + 1, tp)]
    return out


def place_groups(topology: ClusterTopology, num_replicas: int, num_stages: int, tp: int,
                 strategy: str = "spread") -> dict[Key, tuple[str, ...]]:
    slots = _slots(topology, tp)
    need = num_replicas * num_stages
    if sum(len(v) for v in slots.values()) < need:
        raise ValueError(f"need {need} TP{tp} slots, cluster has {sum(len(v) for v in slots.values())}")
    keys = [(r, s) for s in range(num_stages) for r in range(num_replicas)]
    out: dict[Key, tuple[str, ...]] = {}
    if strategy == "megatron":
        flat = [s for node in sorted(slots) for s in slots[node]]
        for key, slot in zip(keys, flat):
            out[key] = slot
        return out
    if strategy != "spread":
        raise ValueError(strategy)
    rack_of = {d.node: d.rack for d in topology.devices}
    by_rack: dict[str, list[str]] = {}
    for node in sorted(slots):
        by_rack.setdefault(rack_of[node], []).append(node)
    order = []
    depth = max(len(v) for v in by_rack.values())
    for i in range(depth):
        for rack in sorted(by_rack):
            if i < len(by_rack[rack]):
                order.append(by_rack[rack][i])
    used = {n: 0 for n in order}
    cursor = 0
    for key in keys:
        for _ in range(len(order)):
            node = order[cursor % len(order)]
            cursor += 1
            if used[node] < len(slots[node]):
                out[key] = slots[node][used[node]]
                used[node] += 1
                break
    return out


def stage_loss_exposure(placement: dict[Key, tuple[str, ...]], topology: ClusterTopology) -> float:
    """Fraction of single-node failures that remove every replica of at least one stage."""
    node_of = {d.id: d.node for d in topology.devices}
    nodes = sorted({node_of[d] for devs in placement.values() for d in devs})
    stages = sorted({s for _, s in placement})
    bad = 0
    for n in nodes:
        for s in stages:
            reps = [devs for (r, st), devs in placement.items() if st == s]
            if all(any(node_of[d] == n for d in devs) for devs in reps):
                bad += 1
                break
    return bad / len(nodes) if nodes else 0.0
