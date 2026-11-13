"""Selective device exclusion inside a TP group (ResiHP section 6.1), without the power-of-two restriction."""

from __future__ import annotations

import math
from dataclasses import dataclass

from resihp.plan.model_spec import ModelProfile, ModelSpec


@dataclass(frozen=True)
class TPChoice:
    devices: tuple[str, ...]
    tp: int
    score: float
    standby: tuple[str, ...]


def feasible_degrees(model: ModelSpec | None, max_k: int, min_k: int = 1, power_of_two_only: bool = False) -> list[int]:
    out = []
    for k in range(max(1, min_k), max_k + 1):
        if power_of_two_only and k & (k - 1):
            continue
        if model is not None and not model.tp_feasible(k):
            continue
        out.append(k)
    return out


def min_tp_for_memory(model: ModelSpec, layers: int, capacity_bytes: float, inflight: int, *, dp_shard: int = 1,
                      sequence_parallel: bool = True, extra_params: int = 0, max_k: int = 64) -> int | None:
    if math.isinf(capacity_bytes):
        return 1
    for k in range(1, max_k + 1):
        if not model.tp_feasible(k):
            continue
        need = model.static_bytes(layers, k, dp_shard, extra_params) + inflight * model.activation_bytes(
            layers, k, sequence_parallel)
        if need <= capacity_bytes:
            return k
    return None


def max_layers_for_memory(model: ModelSpec, tp: int, capacity_bytes: float, inflight: int, *, dp_shard: int = 1,
                          sequence_parallel: bool = True, extra_params: int = 0) -> int:
    if math.isinf(capacity_bytes):
        return model.num_layers
    lo = 0
    for layers in range(1, model.num_layers + 1):
        need = model.static_bytes(layers, tp, dp_shard, extra_params) + inflight * model.activation_bytes(
            layers, tp, sequence_parallel)
        if need > capacity_bytes:
            break
        lo = layers
    return lo


def select_subgroup(candidates: list[tuple[str, float]], degrees: list[int],
                    profile: ModelProfile | None = None, incumbents: set[str] | None = None,
                    cp: int = 1) -> TPChoice | None:
    """argmax_k k * min_{i in S_k} p_i over top-k candidates by normalized throughput (Eq. 4).

    With a profile, the score also includes measured TP scaling efficiency. Ties go to
    the larger group (more memory headroom); among equally fast devices, current members
    are kept so the transition moves as little state as possible. With context
    parallelism a TP degree ``k`` takes ``k * cp`` devices (``cp`` TP slices).
    """
    inc = incumbents or set()
    ordered = sorted(candidates, key=lambda c: (-c[1], c[0] not in inc, c[0]))
    best: TPChoice | None = None
    for k in sorted(degrees):
        if k * cp > len(ordered) or k < 1:
            continue
        chosen = ordered[:k * cp]
        p_min = min(p for _, p in chosen)
        if p_min <= 0:
            continue
        score = (profile.group_speed(k, p_min) if profile else k * p_min) * cp
        if best is None or score >= best.score - 1e-12:
            best = TPChoice(tuple(d for d, _ in chosen), k, score, tuple(d for d, _ in ordered[k * cp:]))
    return best
