"""Bridge between Megatron's distributed optimizer and ResiHP's model-space state slices (GPU path).

Megatron shards each parameter's fp32 master copy and Adam moments across the DP
group as contiguous ranges of the flattened TP-local parameter. Migration works in
model space: ``export_tp_shard_state`` assembles full TP-local tensors per parameter
(all-gather of the DP-owned ranges), the transfer engine moves/reshards them, and
``import_tp_shard_state`` writes back each rank's range in the new layout.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _owned_range(optimizer, param) -> tuple[int, int]:
    r = optimizer._get_model_param_range_map(param)["param"]
    return int(r.start), int(r.end)


def export_tp_shard_state(optimizer, model: torch.nn.Module, dp_group) -> dict[str, dict[str, torch.Tensor]]:
    """{kind: {param name: full TP-local tensor}} with kinds 'param' (fp32 master) and the optimizer moments."""
    world = dist.get_world_size(dp_group) if dp_group is not None else 1
    out: dict[str, dict[str, torch.Tensor]] = {}
    for name, p in model.named_parameters():
        lo, hi = _owned_range(optimizer, p)
        local = optimizer._get_main_param_and_optimizer_states(p)
        bounds = torch.tensor([lo, hi], dtype=torch.int64, device=p.device)
        all_bounds = [torch.empty_like(bounds) for _ in range(world)]
        dist.all_gather(all_bounds, bounds, group=dp_group)
        width = max(int(b[1] - b[0]) for b in all_bounds)
        for kind, t in local.items():
            if not isinstance(t, torch.Tensor) or t.dim() == 0:
                continue
            flat = torch.zeros(width, dtype=t.dtype, device=t.device)
            flat[: hi - lo].copy_(t.reshape(-1))
            parts = [torch.empty_like(flat) for _ in range(world)]
            dist.all_gather(parts, flat, group=dp_group)
            full = torch.empty(p.numel(), dtype=t.dtype, device=t.device)
            for b, part in zip(all_bounds, parts):
                s, e = int(b[0]), int(b[1])
                full[s:e].copy_(part[: e - s])
            out.setdefault(kind, {})[name] = full.view(p.shape)
    return out


def import_tp_shard_state(optimizer, model: torch.nn.Module, state: dict[str, dict[str, torch.Tensor]]) -> None:
    for name, p in model.named_parameters():
        lo, hi = _owned_range(optimizer, p)
        tensors = {kind: per[name].reshape(-1)[lo:hi].clone() for kind, per in state.items() if name in per}
        optimizer._set_main_param_and_optimizer_states(p, tensors)
        with torch.no_grad():
            p.data.copy_(state["param"][name].to(p.dtype))
