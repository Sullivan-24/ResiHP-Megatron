"""Executes a TransferPlan for model *and* optimizer state over the new generation's communicators.

Every kind (parameter, Adam moments, ...) follows the same interval routing, so TP
resharding, PP layer moves and DP replica recovery are one mechanism.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from resihp.state.layer_registry import TensorSpec
from resihp.state.transfer_plan import TransferPlan

LocalState = dict[str, dict[str, torch.Tensor]]


def _view(t: torch.Tensor, spec: TensorSpec, have: tuple[int, int], want: tuple[int, int]) -> torch.Tensor:
    if spec.dim is None:
        return t
    return t.narrow(spec.dim, want[0] - have[0], want[1] - want[0])


def execute_transfers(plan: TransferPlan, me: str, comms, mgr, *, specs: dict[str, TensorSpec],
                      old_layout: dict[str, tuple[int, int]], old_state: LocalState,
                      new_layout: dict[str, tuple[int, int]], kinds: tuple[str, ...], dtype: torch.dtype,
                      device: str = "cpu", timeout: float = 300.0) -> LocalState:
    """Returns this device's state for ``new_layout``; ``old_state`` is left untouched.
    Non-trainable tensors only carry the first kind (their value)."""
    def kinds_of(key: str) -> tuple[str, ...]:
        return kinds if specs[key].trainable else kinds[:1]

    new: LocalState = {k: {key: torch.empty(specs[key].local_shape(iv), dtype=dtype, device=device)
                           for key, iv in new_layout.items() if k in kinds_of(key)} for k in kinds}
    for dst, key, iv in plan.reuse:
        if dst != me:
            continue
        for k in kinds_of(key):
            _view(new[k][key], specs[key], new_layout[key], iv).copy_(
                _view(old_state[k][key], specs[key], old_layout[key], iv))
    nk = len(kinds)
    ops: list = []
    pending: list[tuple[torch.Tensor, str, str, tuple[int, int]]] = []
    for i, tr in enumerate(plan.transfers):
        spec = specs[tr.key]
        for j, k in enumerate(kinds_of(tr.key)):
            tag = (i * nk + j) % (2 ** 31 - 1)
            if tr.src == me:
                piece = _view(old_state[k][tr.key], spec, old_layout[tr.key], tr.interval).contiguous()
                ops.append((dist.isend(piece, comms.rank_of(tr.dst), tag=tag), piece))
            if tr.dst == me:
                buf = torch.empty(spec.local_shape(tr.interval), dtype=dtype, device=device)
                ops.append((dist.irecv(buf, comms.rank_of(tr.src), tag=tag), buf))
                pending.append((buf, k, tr.key, tr.interval))
    for work, _ in ops:
        mgr.guarded_wait(work, timeout=timeout, what="state transfer")
    for buf, k, key, iv in pending:
        _view(new[k][key], specs[key], new_layout[key], iv).copy_(buf)
    return new
