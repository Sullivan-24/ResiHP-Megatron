"""Layer repartition across pipeline stages (ResiHP section 6.2).

Exact min-max contiguous partition by dynamic programming over (stage, layer
prefix), with per-stage capacities and memory caps. Capacities are aggregated
over the DP replicas of a stage because residual cross-replica imbalance is
removed afterwards by stage-level micro-batch migration (section 6.3).
"""

from __future__ import annotations

import math


def partition_layers(num_layers: int, capacities: list[float], *, layer_cost: list[float] | None = None,
                     first_extra: float = 0.0, last_extra: float = 0.0, max_layers: list[int] | None = None,
                     min_layers: int = 1) -> list[int]:
    S = len(capacities)
    if S == 0 or num_layers < S * min_layers:
        raise ValueError(f"cannot place {num_layers} layers on {S} stages with >= {min_layers} each")
    cost = layer_cost or [1.0] * num_layers
    prefix = [0.0]
    for c in cost:
        prefix.append(prefix[-1] + c)
    caps = [max(c, 1e-12) for c in capacities]
    maxl = max_layers or [num_layers] * S

    def t(s: int, i: int, j: int) -> float:
        extra = (first_extra if s == 0 else 0.0) + (last_extra if s == S - 1 else 0.0)
        return (prefix[j] - prefix[i] + extra) / caps[s]

    INF = (math.inf, math.inf)
    best = [[INF] * (num_layers + 1) for _ in range(S + 1)]
    back = [[-1] * (num_layers + 1) for _ in range(S + 1)]
    best[0][0] = (0.0, 0.0)
    for s in range(S):
        for j in range(num_layers + 1):
            for i in range(j - min_layers, -1, -1):
                if j - i > maxl[s]:
                    break
                prev = best[s][i]
                if prev[0] == math.inf:
                    continue
                ts = t(s, i, j)
                cand = (max(prev[0], ts), prev[1] + ts * ts)
                if cand < best[s + 1][j]:
                    best[s + 1][j] = cand
                    back[s + 1][j] = i
    if best[S][num_layers][0] == math.inf:
        raise ValueError("no feasible partition under the memory caps")
    counts = []
    j = num_layers
    for s in range(S, 0, -1):
        i = back[s][j]
        counts.append(j - i)
        j = i
    return counts[::-1]


def counts_to_ranges(counts: list[int]) -> list[tuple[int, int]]:
    out, start = [], 0
    for c in counts:
        out.append((start, start + c))
        start += c
    return out
