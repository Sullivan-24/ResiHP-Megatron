"""Progressive TP -> PP -> DP planner (ResiHP section 6) producing HeteroParallelPlans and IR schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from resihp.plan.cost_model import CostModel, Workload
from resihp.plan.dp_migration import initial_assignment, progress_aware_migration, simulate
from resihp.plan.hetero_plan import HeteroParallelPlan, StageGroup, interleave
from resihp.plan.ir import Schedule, compute_order_to_ir, validate
from resihp.plan.layer_partition import counts_to_ranges, partition_layers
from resihp.plan.model_spec import ModelProfile, ModelSpec
from resihp.plan.placement import place_groups
from resihp.plan.simulator import SimResult
from resihp.plan.topology import ClusterTopology
from resihp.plan.tp_select import feasible_degrees, max_layers_for_memory, min_tp_for_memory, select_subgroup


class PlanInfeasible(RuntimeError):
    pass


@dataclass
class PlannerConfig:
    model: ModelSpec
    profile: ModelProfile
    num_stages: int
    num_replicas: int
    tp: int
    num_microbatches: int
    mem_capacity: float = math.inf
    dp_shard: int = 1
    power_of_two_only: bool = False
    sequence_parallel: bool = True
    split_backward: bool = False
    migration_rounds: int = 32
    delta: float = 0.0
    placement: str = "spread"
    workload: Workload | None = None
    enable_tp_shrink: bool = True
    enable_repartition: bool = True
    enable_migration: bool = True
    max_replicas: int | None = None
    vpp: int = 1
    cp: int = 1


class HeteroPlanner:
    def __init__(self, cfg: PlannerConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ helpers
    def workload(self) -> Workload:
        c = self.cfg
        return c.workload or Workload.uniform(c.num_microbatches, c.model.seq_len, c.model.micro_batch_size)

    def _extra(self, s: int) -> float:
        p = self.cfg.profile
        return (p.first_stage_extra_layers if s == 0 else 0.0) + (p.last_stage_extra_layers if s == self.cfg.num_stages - 1 else 0.0)

    def _extra_params(self, s: int) -> int:
        m = self.cfg.model
        return (m.embedding_params() if s == 0 else 0) + (m.embedding_params() if s == self.cfg.num_stages - 1 else 0)

    def _inflight(self, s: int) -> int:
        """In-flight micro-batches for memory sizing; context parallelism splits each one's
        activations over ``cp`` devices."""
        P, v = self.cfg.num_stages, self.cfg.vpp
        n = max(1, P - s) if v == 1 else math.ceil(((P - s - 1) * 2 + (v - 1) * P) / v) + 1
        return max(1, math.ceil(n / self.cfg.cp))

    def _speed(self, devices, speed: dict[str, float]) -> float:
        c = self.cfg
        return c.profile.group_speed(len(devices) // c.cp, min(speed.get(d, 1.0) for d in devices)) * c.cp

    def _choose(self, cands: list[tuple[str, float]], max_tp: int, k_min: int, incumbents=None):
        c = self.cfg
        degrees = feasible_degrees(c.model, min(len(cands) // c.cp, max_tp), k_min, c.power_of_two_only)
        return select_subgroup(cands, degrees, c.profile, incumbents=incumbents, cp=c.cp)

    def cost_model(self, plan: HeteroParallelPlan, topology: ClusterTopology) -> CostModel:
        return CostModel(plan, self.cfg.model, self.cfg.profile, self.workload(),
                         {d.id: d.speed for d in topology.devices}, {d.id: d.node for d in topology.devices},
                         split_backward=self.cfg.split_backward)

    def mem_caps(self, plan: HeteroParallelPlan) -> dict[tuple[int, int], float]:
        c = self.cfg
        if math.isinf(c.mem_capacity):
            return {}
        static: dict[tuple[int, int], float] = {}
        act: dict[tuple[int, int], float] = {}
        S = plan.num_stages
        for g in plan.stage_groups:
            key = plan.physical_key(g.replica, g.stage)
            layers = g.layers[1] - g.layers[0]
            extra = (c.model.embedding_params() if g.stage == 0 else 0) + (c.model.embedding_params() if g.stage == S - 1 else 0)
            static[key] = static.get(key, 0.0) + c.model.static_bytes(layers, g.tp, c.dp_shard, extra)
            act[key] = act.get(key, 0.0) + c.model.activation_bytes(layers, g.tp, c.sequence_parallel) / g.cp
        return {k: math.floor((c.mem_capacity - static[k]) / act[k]) if act[k] > 0 else math.inf for k in static}

    def _partition(self, capacities: list[float], max_layers: list[int]) -> list[int]:
        c = self.cfg
        if not c.enable_repartition:
            base, rem = divmod(c.model.num_layers, c.num_stages)
            return [base + (1 if s < rem else 0) for s in range(c.num_stages)]
        return partition_layers(c.model.num_layers, capacities, first_extra=c.profile.first_stage_extra_layers,
                                last_extra=c.profile.last_stage_extra_layers, max_layers=max_layers, min_layers=c.vpp)

    def _assemble(self, generation: int, members: dict[tuple[int, int], tuple[str, ...]], counts: list[int],
                  world_order: list[str], standby: list[str], topology: ClusterTopology) -> HeteroParallelPlan:
        c = self.cfg
        used = {d for devs in members.values() for d in devs}
        world = [d for d in world_order if d in used] + sorted(used - set(world_order))
        rank = {d: i for i, d in enumerate(world)}
        ranges = counts_to_ranges(counts)
        groups = tuple(StageGroup(r, s, tuple(sorted(devs, key=rank.__getitem__)), ranges[s], c.cp)
                       for (r, s), devs in sorted(members.items(), key=lambda kv: (kv[0][1], kv[0][0])))
        speed = {d.id: d.speed for d in topology.devices}

        def gspeed(key):
            return self._speed(members[key], speed)

        assignment = initial_assignment(list(groups), c.num_stages, c.num_microbatches, gspeed)
        num_replicas = max(r for r, _ in members) + 1
        plan = HeteroParallelPlan(generation=generation, world=tuple(world), num_layers=c.model.num_layers,
                                  num_stages=c.num_stages, num_replicas=num_replicas, stage_groups=groups,
                                  num_microbatches=c.num_microbatches, assignment=assignment,
                                  standby=tuple(sorted(standby)), sequence_parallel=c.sequence_parallel)
        plan = interleave(plan, c.vpp)
        plan.validate()
        cm = self.cost_model(plan, topology)
        if c.enable_migration:
            res = progress_aware_migration(plan, cm, mem_cap=self.mem_caps(plan) or None, delta=c.delta,
                                           max_rounds=c.migration_rounds, split_backward=c.split_backward)
            plan = replace(plan, assignment=res.assignment)
            makespan, initial = res.makespan, res.initial_makespan
        else:
            makespan = initial = simulate(plan, cm, c.split_backward).makespan
        plan = replace(plan, meta={"predicted_iteration_s": makespan, "unmigrated_iteration_s": initial,
                                   "layer_counts": counts})
        plan.validate()
        return plan

    # ------------------------------------------------------------------ entry points
    def initial_plan(self, topology: ClusterTopology, generation: int = 1) -> HeteroParallelPlan:
        c = self.cfg
        members = place_groups(topology, c.num_replicas, c.num_stages, c.tp * c.cp, c.placement)
        cap = [c.num_replicas * c.profile.group_speed(c.tp, 1.0) * c.cp] * c.num_stages
        max_layers = [max_layers_for_memory(c.model, c.tp, c.mem_capacity, self._inflight(s), dp_shard=c.dp_shard,
                                            sequence_parallel=c.sequence_parallel, extra_params=self._extra_params(s))
                      for s in range(c.num_stages)]
        counts = self._partition(cap, max_layers)
        used = {d for devs in members.values() for d in devs}
        standby = [d for d in topology.ids() if d not in used]
        return self._assemble(generation, members, counts, topology.ids(), standby, topology)

    def replan(self, prev: HeteroParallelPlan, topology: ClusterTopology, generation: int | None = None) -> HeteroParallelPlan:
        c = self.cfg
        prev = prev.physical()
        alive = {d.id: d for d in topology.devices}
        node_of = {d.id: d.node for d in topology.devices}
        in_groups = {d for g in prev.stage_groups for d in g.devices}
        pool: dict[str, list[str]] = {}
        for d in topology.devices:
            if d.id in prev.standby or d.id not in in_groups:
                pool.setdefault(d.node, []).append(d.id)
        members: dict[tuple[int, int], tuple[str, ...]] = {}
        prev_counts = [b - a for a, b in prev.layer_partition()]
        for g in sorted(prev.stage_groups, key=lambda g: (g.stage, g.replica)):
            survivors = [d for d in g.devices if d in alive]
            if survivors:
                node = sorted({node_of[d] for d in survivors})[0]
            else:
                # Whole group lost: rebuild it on the node with the most free devices (spare/rejoined node).
                node = max((n for n in pool if pool[n]), key=lambda n: (len(pool[n]), n), default=None)
            extra = pool.get(node, []) if node else []
            cands = [(d, alive[d].speed) for d in survivors + extra]
            if not cands:
                continue
            if not c.enable_tp_shrink:
                if len(survivors) == len(g.devices):
                    members[(g.replica, g.stage)] = tuple(survivors)
                continue
            k_min = min_tp_for_memory(c.model, prev_counts[g.stage], c.mem_capacity, self._inflight(g.stage),
                                      dp_shard=c.dp_shard, sequence_parallel=c.sequence_parallel,
                                      extra_params=self._extra_params(g.stage)) or c.tp + 1
            choice = self._choose(cands, c.tp, k_min, incumbents=set(survivors))
            if choice is None:
                continue
            members[(g.replica, g.stage)] = choice.devices
            if node:
                pool[node] = [d for d in pool.get(node, []) + survivors if d not in choice.devices]
        self._fill_from_pool(members, pool, alive, prev_counts)
        for s in range(c.num_stages):
            if not any(key[1] == s for key in members):
                raise PlanInfeasible(f"no TP group can host stage {s}")
        gen = prev.generation + 1 if generation is None else generation
        best = self._plan_from_members(members, prev, topology, gen)
        # Joint decision the per-group rule (Eq. 4) cannot make: keeping a weakened group may
        # slow the uniform layer partition more than evicting its work to peer replicas.
        weakened = [k for k, devs in members.items()
                    if len(devs) < c.tp * c.cp or min(alive[d].speed for d in devs) < 1.0]
        for key in weakened:
            if sum(1 for k in members if k[1] == key[1]) < 2:
                continue
            variant = {k: v for k, v in members.items() if k != key}
            try:
                cand = self._plan_from_members(variant, prev, topology, gen)
            except ValueError:
                continue
            if cand.meta["predicted_iteration_s"] < best.meta["predicted_iteration_s"] - 1e-12:
                best = cand
        return best

    def _fill_from_pool(self, members: dict[tuple[int, int], tuple[str, ...]], pool: dict[str, list[str]],
                        alive: dict, prev_counts: list[int]) -> None:
        """Rejoin/grow: rebuild missing (replica, stage) groups of the nominal layout from free
        devices (replacement or repaired nodes), then add whole replicas up to ``max_replicas``."""
        c = self.cfg
        target = max(c.num_replicas, c.max_replicas or 0)

        def take(stage: int) -> tuple[str, ...] | None:
            for node in sorted(pool, key=lambda n: (-len(pool[n]), n)):
                cands = [(d, alive[d].speed) for d in pool[node]]
                if len(cands) < 1:
                    continue
                k_min = min_tp_for_memory(c.model, prev_counts[stage], c.mem_capacity, self._inflight(stage),
                                          dp_shard=c.dp_shard, sequence_parallel=c.sequence_parallel,
                                          extra_params=self._extra_params(stage)) or c.tp + 1
                choice = self._choose(cands, c.tp, k_min)
                if choice is not None and choice.tp == c.tp:
                    pool[node] = [d for d in pool[node] if d not in choice.devices]
                    return choice.devices
            return None

        for r in range(target):
            missing = [s for s in range(c.num_stages) if (r, s) not in members]
            if not missing:
                continue
            if r >= c.num_replicas and len(missing) < c.num_stages:
                continue
            free_slots = sum(len(v) // (c.tp * c.cp) for v in pool.values())
            if r >= c.num_replicas and free_slots < c.num_stages:
                break
            for s in missing:
                devs = take(s)
                if devs is not None:
                    members[(r, s)] = devs

    def _plan_from_members(self, members: dict[tuple[int, int], tuple[str, ...]], prev: HeteroParallelPlan,
                           topology: ClusterTopology, gen: int) -> HeteroParallelPlan:
        c = self.cfg
        alive = topology.by_id()
        cap = []
        max_layers = []
        speed = {d: alive[d].speed for devs in members.values() for d in devs}
        for s in range(c.num_stages):
            keys = [k for k in members if k[1] == s]
            cap.append(sum(self._speed(members[k], speed) for k in keys))
            max_layers.append(min(max_layers_for_memory(c.model, len(members[k]) // c.cp, c.mem_capacity, self._inflight(s),
                                                        dp_shard=c.dp_shard, sequence_parallel=c.sequence_parallel,
                                                        extra_params=self._extra_params(s)) for k in keys))
        counts = self._partition(cap, max_layers)
        used = {d for devs in members.values() for d in devs}
        standby = [d for d in topology.ids() if d not in used]
        return self._assemble(gen, members, counts, list(prev.world), standby, topology)

    def schedule(self, plan: HeteroParallelPlan, topology: ClusterTopology) -> tuple[Schedule, SimResult]:
        sim = simulate(plan, self.cost_model(plan, topology), self.cfg.split_backward)
        ir = compute_order_to_ir(plan, sim.order)
        validate(plan, ir, self.cfg.split_backward)
        return ir, sim
