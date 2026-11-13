"""Process-group specifications derived from a plan.

Every rank derives the same ordered list of specs from the same plan, which is
what makes group creation deadlock-free: creation calls happen in an identical,
plan-determined sequence and ranks outside a group never block on it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroupSpec:
    name: str
    ranks: tuple[int, ...]
    kind: str = "custom"
    keep_singleton: bool = False

    def __post_init__(self) -> None:
        if list(self.ranks) != sorted(set(self.ranks)):
            raise ValueError(f"group {self.name}: ranks must be unique and ascending, got {self.ranks}")

    @property
    def materialized(self) -> bool:
        """Whether a ProcessGroup object is created (frameworks such as Megatron need size-1 groups)."""
        return len(self.ranks) > 1 or self.keep_singleton


def canonical_order(specs) -> list[GroupSpec]:
    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate group names")
    return sorted(specs, key=lambda s: (s.kind, s.name))


def split_layers(specs) -> list[list[GroupSpec]]:
    """Greedy layering of multi-rank groups so that groups within a layer are disjoint.

    Each layer becomes one ``split_group`` call that every rank enters; groups with
    an identical rank set are created once and shared.
    """
    layers: list[list[GroupSpec]] = []
    used: list[set[int]] = []
    seen: set[tuple[int, ...]] = set()
    for spec in canonical_order(specs):
        if not spec.materialized or spec.ranks in seen:
            continue
        seen.add(spec.ranks)
        members = set(spec.ranks)
        for layer, taken in zip(layers, used):
            if not taken & members:
                layer.append(spec)
                taken |= members
                break
        else:
            layers.append([spec])
            used.append(set(members))
    return layers
