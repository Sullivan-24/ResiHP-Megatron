"""HeteroParallelPlan: an explicit, generation-scoped description of dynamic hybrid parallelism.

Unlike Megatron's rectangular rank grid, every (replica, stage) group lists its own
devices, TP degree and layer range, and ``assignment[stage][mb]`` names the replica
that executes micro-batch ``mb`` at that stage. This single table expresses plain
DP, whole-replica exclusion, and ResiHP's stage-level micro-batch migration.

With interleaved pipelining (``vpp > 1``) stages are *virtual*: virtual stage ``s``
runs on physical stage ``s % (num_stages // vpp)``, and the groups of one physical
slot (same replica, same physical stage) list identical devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from resihp.comm.group_spec import GroupSpec
from resihp.utils.intervals import even_split, locate, refine


@dataclass(frozen=True)
class StageGroup:
    """``devices`` are TP-major (Megatron order): CP slice ``i`` is the TP group
    ``devices[i * tp:(i + 1) * tp]``; the CP group of TP rank ``j`` is ``devices[j::tp]``."""

    replica: int
    stage: int
    devices: tuple[str, ...]
    layers: tuple[int, int]
    cp: int = 1

    @property
    def tp(self) -> int:
        return len(self.devices) // self.cp

    def slice(self, i: int) -> tuple[str, ...]:
        return self.devices[i * self.tp:(i + 1) * self.tp]

    def cp_peers(self, j: int) -> tuple[str, ...]:
        return self.devices[j::self.tp]

    def slice_of(self, device: str) -> int:
        return self.devices.index(device) // self.tp

    def to_json(self) -> dict:
        out = {"replica": self.replica, "stage": self.stage, "devices": list(self.devices), "layers": list(self.layers)}
        if self.cp > 1:
            out["cp"] = self.cp
        return out

    @staticmethod
    def from_json(d: dict) -> "StageGroup":
        return StageGroup(d["replica"], d["stage"], tuple(d["devices"]), tuple(d["layers"]), d.get("cp", 1))


def tp_group_name(replica: int, stage: int, cp_slice: int | None = None) -> str:
    return f"tp/r{replica}/s{stage}" if cp_slice is None else f"tp/r{replica}/s{stage}/c{cp_slice}"


def cp_group_name(replica: int, stage: int, tp_rank: int) -> str:
    return f"cp/r{replica}/s{stage}/t{tp_rank}"


def dpshard_group_name(stage: int, ranks: tuple[int, ...]) -> str:
    return f"dpshard/s{stage}/" + "-".join(map(str, ranks))


@dataclass(frozen=True)
class HeteroParallelPlan:
    generation: int
    world: tuple[str, ...]
    num_layers: int
    num_stages: int
    num_replicas: int
    stage_groups: tuple[StageGroup, ...]
    num_microbatches: int
    assignment: tuple[tuple[int, ...], ...]
    standby: tuple[str, ...] = ()
    sequence_parallel: bool = False
    meta: dict = field(default_factory=dict, compare=False)
    vpp: int = 1

    # ------------------------------------------------------------------ lookups
    def group(self, replica: int, stage: int) -> StageGroup | None:
        for g in self.stage_groups:
            if g.replica == replica and g.stage == stage:
                return g
        return None

    @property
    def num_physical_stages(self) -> int:
        return self.num_stages // self.vpp

    def physical_key(self, replica: int, stage: int) -> tuple[int, int]:
        return replica, stage % self.num_physical_stages

    def chunk_stages(self, replica: int, physical_stage: int) -> list[int]:
        P = self.num_physical_stages
        return [s for s in range(physical_stage, self.num_stages, P) if self.group(replica, s) is not None]

    def groups_of_device(self, device: str) -> list[StageGroup]:
        return sorted((g for g in self.stage_groups if device in g.devices), key=lambda g: g.stage)

    def groups_at(self, stage: int) -> list[StageGroup]:
        return sorted((g for g in self.stage_groups if g.stage == stage), key=lambda g: g.replica)

    def rank(self, device: str) -> int:
        return self.world.index(device)

    def ranks(self, g: StageGroup) -> tuple[int, ...]:
        return tuple(self.rank(d) for d in g.devices)

    def slice_ranks(self, g: StageGroup, i: int) -> tuple[int, ...]:
        return tuple(self.rank(d) for d in g.slice(i))

    def tp_name(self, g: StageGroup, device: str) -> str:
        """Name of the TP group ``device`` belongs to in ``g`` (its CP slice)."""
        return tp_group_name(g.replica, g.stage) if g.cp == 1 else tp_group_name(g.replica, g.stage, g.slice_of(device))

    def group_of_device(self, device: str) -> StageGroup | None:
        """The device's group; with interleaving, its group for the first local chunk."""
        gs = self.groups_of_device(device)
        return gs[0] if gs else None

    def executor(self, stage: int, mb: int) -> int:
        return self.assignment[stage][mb]

    def layer_partition(self) -> list[tuple[int, int]]:
        return [self.groups_at(s)[0].layers for s in range(self.num_stages)]

    def microbatches_of(self, replica: int, stage: int) -> list[int]:
        return [m for m in range(self.num_microbatches) if self.assignment[stage][m] == replica]

    def physical(self) -> "HeteroParallelPlan":
        """The planning view of an interleaved plan: one stage per physical slot, holding the
        total layer count of its chunks (layer ranges are nominal), chunk-0 executors."""
        if self.vpp == 1:
            return self
        P = self.num_physical_stages
        counts = [sum(self.groups_at(s)[0].layers[1] - self.groups_at(s)[0].layers[0]
                      for s in range(p, self.num_stages, P)) for p in range(P)]
        starts = [sum(counts[:p]) for p in range(P + 1)]
        groups = tuple(StageGroup(g.replica, g.stage, g.devices, (starts[g.stage], starts[g.stage + 1]), g.cp)
                       for g in self.stage_groups if g.stage < P)
        return replace(self, num_stages=P, stage_groups=groups, assignment=self.assignment[:P], vpp=1)

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        if self.vpp < 1 or self.num_stages % self.vpp:
            raise ValueError(f"{self.num_stages} stages cannot be split into {self.vpp} chunks per physical stage")
        slot_of: dict[str, tuple[int, int]] = {}
        slot_devices: dict[tuple[int, int], tuple[str, ...]] = {}
        world = set(self.world)
        for g in self.stage_groups:
            if not g.devices:
                raise ValueError(f"empty group r{g.replica}s{g.stage}")
            slot = self.physical_key(g.replica, g.stage)
            if slot_devices.setdefault(slot, g.devices) != g.devices:
                raise ValueError(f"chunks of physical slot {slot} must use identical devices")
            if len(set(g.devices)) != len(g.devices):
                raise ValueError(f"group r{g.replica}s{g.stage} lists a device twice")
            if g.cp < 1 or len(g.devices) % g.cp:
                raise ValueError(f"group r{g.replica}s{g.stage}: {len(g.devices)} devices do not split into cp={g.cp}")
            if g.cp != self.stage_groups[0].cp:
                raise ValueError("context parallelism must be uniform (every stage splits the same sequence)")
            for d in g.devices:
                if slot_of.setdefault(d, slot) != slot:
                    raise ValueError(f"device {d} used twice")
                if d not in world:
                    raise ValueError(f"device {d} not in world")
            ranks = self.ranks(g)
            if list(ranks) != sorted(ranks):
                raise ValueError(f"group r{g.replica}s{g.stage} devices must be in ascending rank order")
        seen = set(slot_of)
        if seen != world:
            raise ValueError(f"world contains devices outside any group: {sorted(world - seen)}")
        if set(self.standby) & world:
            raise ValueError("standby devices cannot be in the world")
        for s in range(self.num_stages):
            gs = self.groups_at(s)
            if not gs:
                raise ValueError(f"stage {s} has no available group")
            if len({g.layers for g in gs}) != 1:
                raise ValueError(f"stage {s} layer ranges differ across replicas")
        bounds = self.layer_partition()
        if bounds[0][0] != 0 or bounds[-1][1] != self.num_layers or any(
                bounds[i][1] != bounds[i + 1][0] for i in range(len(bounds) - 1)):
            raise ValueError(f"layer partition {bounds} does not tile [0, {self.num_layers})")
        if len(self.assignment) != self.num_stages:
            raise ValueError("assignment must have one row per stage")
        for s, row in enumerate(self.assignment):
            if len(row) != self.num_microbatches:
                raise ValueError(f"assignment row {s} has {len(row)} entries")
            for m, r in enumerate(row):
                if self.group(r, s) is None:
                    raise ValueError(f"micro-batch {m} stage {s} assigned to unavailable replica {r}")

    # ------------------------------------------------------------------ process groups
    def group_specs(self, partition_sizes: tuple[int, ...]) -> list[GroupSpec]:
        """TP groups plus shard-aligned DP gradient-reduction groups.

        For each stage, every parameter dimension of length ``n`` is split evenly by each
        replica's TP degree; on the common refinement each interval is held by exactly
        one rank per replica, and those holders form one reduction group. With uniform TP
        this degenerates to Megatron's DP groups.
        """
        specs: dict[str, GroupSpec] = {}
        for g in self.stage_groups:
            for i in range(g.cp):
                if g.tp > 1:
                    name = tp_group_name(g.replica, g.stage, None if g.cp == 1 else i)
                    specs[name] = GroupSpec(name, tuple(sorted(self.slice_ranks(g, i))), "tp")
            if g.cp > 1:
                for j in range(g.tp):
                    name = cp_group_name(g.replica, g.stage, j)
                    specs[name] = GroupSpec(name, tuple(sorted(self.rank(d) for d in g.cp_peers(j))), "cp")
        for s in range(self.num_stages):
            for holders in self.dpshard_holder_sets(s, partition_sizes):
                if len(holders) > 1:
                    name = dpshard_group_name(s, holders)
                    specs[name] = GroupSpec(name, holders, "dpshard")
        return list(specs.values())

    def tp_slices(self, stage: int) -> list[tuple[str, ...]]:
        """Every TP partition of a stage: one per (replica, CP slice); each holds a full copy of
        the stage's parameters, so gradients are reduced across all of them (DP x CP)."""
        return [g.slice(i) for g in self.groups_at(stage) for i in range(g.cp)]

    def dpshard_holder_sets(self, stage: int, partition_sizes: tuple[int, ...]) -> list[tuple[int, ...]]:
        slices = self.tp_slices(stage)
        out: list[tuple[int, ...]] = []
        for n in sorted(set(partition_sizes)):
            parts = [even_split(n, len(sl)) for sl in slices]
            for iv in refine([[p for p in part] for part in parts]):
                holders = tuple(sorted(self.rank(sl[locate(part, iv)]) for sl, part in zip(slices, parts)))
                if holders not in out:
                    out.append(holders)
        return out

    # ------------------------------------------------------------------ serialization
    def to_json(self) -> dict:
        out = {"generation": self.generation, "world": list(self.world), "num_layers": self.num_layers,
               "num_stages": self.num_stages, "num_replicas": self.num_replicas,
               "stage_groups": [g.to_json() for g in self.stage_groups], "num_microbatches": self.num_microbatches,
               "assignment": [list(r) for r in self.assignment], "standby": list(self.standby),
               "sequence_parallel": self.sequence_parallel, "meta": self.meta}
        if self.vpp > 1:
            out["vpp"] = self.vpp
        return out

    @staticmethod
    def from_json(d: dict) -> "HeteroParallelPlan":
        return HeteroParallelPlan(
            generation=d["generation"], world=tuple(d["world"]), num_layers=d["num_layers"],
            num_stages=d["num_stages"], num_replicas=d["num_replicas"],
            stage_groups=tuple(StageGroup.from_json(g) for g in d["stage_groups"]),
            num_microbatches=d["num_microbatches"], assignment=tuple(tuple(r) for r in d["assignment"]),
            standby=tuple(d.get("standby", ())), sequence_parallel=d.get("sequence_parallel", False),
            meta=d.get("meta", {}), vpp=d.get("vpp", 1))


def interleave(plan: HeteroParallelPlan, vpp: int) -> HeteroParallelPlan:
    """Split every physical stage's layers into ``vpp`` chunks; chunk ``c`` of physical stage
    ``p`` becomes virtual stage ``c * P + p`` (Megatron's interleaved layout)."""
    if vpp == 1:
        return plan
    P = plan.num_stages
    counts = [plan.groups_at(p)[0].layers[1] - plan.groups_at(p)[0].layers[0] for p in range(P)]
    vcounts = [0] * (P * vpp)
    for p, n in enumerate(counts):
        for c, (a, b) in enumerate(even_split(n, vpp)):
            vcounts[c * P + p] = b - a
    if min(vcounts) < 1:
        raise ValueError(f"layer counts {counts} cannot be split into {vpp} chunks per stage")
    starts = [sum(vcounts[:s]) for s in range(P * vpp + 1)]
    groups = tuple(StageGroup(g.replica, c * P + g.stage, g.devices,
                              (starts[c * P + g.stage], starts[c * P + g.stage + 1]), g.cp)
                   for c in range(vpp) for g in plan.stage_groups)
    return replace(plan, num_stages=P * vpp, stage_groups=groups,
                   assignment=tuple(plan.assignment[s % P] for s in range(P * vpp)), vpp=vpp)
