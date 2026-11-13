"""Cluster topology: devices, nodes (NVLink domains) and racks (failure domains)."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Device:
    id: str
    node: str
    local_rank: int
    rack: str = "r0"
    speed: float = 1.0


@dataclass(frozen=True)
class ClusterTopology:
    devices: tuple[Device, ...]

    @staticmethod
    def uniform(num_nodes: int, gpus_per_node: int, nodes_per_rack: int | None = None) -> "ClusterTopology":
        devs = []
        for n in range(num_nodes):
            rack = f"r{n // nodes_per_rack}" if nodes_per_rack else "r0"
            for g in range(gpus_per_node):
                devs.append(Device(f"n{n}g{g}", f"n{n}", g, rack))
        return ClusterTopology(tuple(devs))

    def by_id(self) -> dict[str, Device]:
        return {d.id: d for d in self.devices}

    def nodes(self) -> dict[str, list[Device]]:
        out: dict[str, list[Device]] = {}
        for d in sorted(self.devices, key=lambda d: (d.node, d.local_rank)):
            out.setdefault(d.node, []).append(d)
        return out

    def node_of(self, dev_id: str) -> str:
        return self.by_id()[dev_id].node

    def speed(self, dev_id: str) -> float:
        return self.by_id()[dev_id].speed

    def without(self, failed) -> "ClusterTopology":
        failed = set(failed)
        return ClusterTopology(tuple(d for d in self.devices if d.id not in failed))

    def with_speeds(self, speeds: dict[str, float]) -> "ClusterTopology":
        return ClusterTopology(tuple(replace(d, speed=speeds.get(d.id, d.speed)) for d in self.devices))

    def ids(self) -> list[str]:
        return [d.id for d in self.devices]
