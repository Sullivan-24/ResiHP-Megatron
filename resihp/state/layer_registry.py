"""Stable tensor identities and plan-derived ownership.

Every tensor is named by a key that does not depend on stage boundaries
(``layers.<global layer>.<param>``), with its global shape and TP partition dim.
Given a plan, ``holdings`` says which device holds which interval of every key, so
the old and new plans of a reconfiguration fully determine the state to move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from resihp.plan.hetero_plan import HeteroParallelPlan
from resihp.utils.intervals import even_split

FULL = (0, 1)


@dataclass(frozen=True)
class TensorSpec:
    key: str
    global_shape: tuple[int, ...]
    dim: int | None
    trainable: bool = True
    """Non-trainable tensors (FP8 amax history, router expert bias) carry no optimizer state."""

    @property
    def extent(self) -> int:
        """Length of the partitioned dimension; replicated tensors use a pseudo-extent of 1."""
        return self.global_shape[self.dim] if self.dim is not None else 1

    def local_shape(self, interval: tuple[int, int]) -> tuple[int, ...]:
        if self.dim is None:
            return self.global_shape
        shape = list(self.global_shape)
        shape[self.dim] = interval[1] - interval[0]
        return tuple(shape)


Holdings = dict[str, list[tuple[str, tuple[int, int]]]]


def holdings(plan: HeteroParallelPlan, specs: dict[str, TensorSpec], layer_of: Callable[[str], int]) -> Holdings:
    out: Holdings = {}
    for g in plan.stage_groups:
        lo, hi = g.layers
        for key, spec in specs.items():
            if not lo <= layer_of(key) < hi:
                continue
            parts = even_split(spec.extent, g.tp) if spec.dim is not None else [FULL] * g.tp
            for i in range(g.cp):
                for dev, iv in zip(g.slice(i), parts):
                    out.setdefault(key, []).append((dev, iv))
    for key in out:
        out[key].sort()
    return out


def device_layout(h: Holdings, device: str) -> dict[str, tuple[int, int]]:
    return {key: iv for key, hs in h.items() for d, iv in hs if d == device}


def reference_specs(cfg) -> dict[str, TensorSpec]:
    from resihp.runtime.reference_model import PARAM_DIMS, full_layer_params

    out = {}
    for layer in range(cfg.num_layers):
        for name, t in full_layer_params(cfg, layer).items():
            key = f"layers.{layer}.{name}"
            out[key] = TensorSpec(key, tuple(t.shape), PARAM_DIMS[name])
        if cfg.amax_history:
            key = f"layers.{layer}.amax_history"
            out[key] = TensorSpec(key, (cfg.amax_history,), None, trainable=False)
    return out


def reference_layer_of(key: str) -> int:
    return int(key.split(".")[1])
