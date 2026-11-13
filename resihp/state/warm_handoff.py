"""Warm handoff: move layer ownership without stopping training (planned / fail-slow repartition).

1. ``prestage`` at a commit boundary: copy parameters and optimizer state of the keys
   that change owner to their destinations (the copies are one step stale soon after).
2. ``lockstep`` for the next iteration: sources forward the *reduced* gradients of those
   keys and destinations apply the identical optimizer update, so destination state is
   bit-identical to source state.
3. Switch to the new plan at the following boundary; no state is on the critical path.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from resihp.state.layer_registry import TensorSpec
from resihp.state.transfer_engine import LocalState, execute_transfers
from resihp.state.transfer_plan import TransferPlan


class WarmHandoff:
    def __init__(self, plan: TransferPlan, me: str, comms, mgr, specs: dict[str, TensorSpec],
                 old_layout: dict[str, tuple[int, int]], incoming_layout: dict[str, tuple[int, int]],
                 kinds: tuple[str, ...] = ("param", "exp_avg", "exp_avg_sq"), timeout: float = 300.0):
        self.plan = TransferPlan(transfers=list(plan.transfers))
        self.me = me
        self.comms = comms
        self.mgr = mgr
        self.specs = specs
        self.old_layout = old_layout
        self.incoming_layout = incoming_layout
        self.kinds = kinds
        self.timeout = timeout
        self.staged: LocalState | None = None

    def prestage(self, old_state: LocalState, dtype: torch.dtype) -> LocalState:
        self.staged = execute_transfers(self.plan, self.me, self.comms, self.mgr, specs=self.specs,
                                        old_layout=self.old_layout, old_state=old_state,
                                        new_layout=self.incoming_layout, kinds=self.kinds, dtype=dtype,
                                        timeout=self.timeout)
        return self.staged

    def forward_grads(self, reduced_grads: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
        """Ship this iteration's reduced gradients along the prestage routes; returns received grads."""
        got = execute_transfers(self.plan, self.me, self.comms, self.mgr, specs=self.specs,
                                old_layout=self.old_layout, old_state={"grad": reduced_grads},
                                new_layout=self.incoming_layout, kinds=("grad",), dtype=dtype, timeout=self.timeout)
        return got["grad"]

    def lockstep_update(self, grads: dict[str, torch.Tensor], step_count: int, lr: float, betas=(0.9, 0.999),
                        eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        """Apply exactly the update ``resihp.runtime.optim.Adam`` applies at ``step_count``."""
        import math

        assert self.staged is not None
        b1, b2 = betas
        bc1 = 1 - b1 ** step_count
        bc2_sqrt = math.sqrt(1 - b2 ** step_count)
        with torch.no_grad():
            for key, g in grads.items():
                p = self.staged["param"][key]
                if weight_decay:
                    g = g.add(p, alpha=weight_decay)
                m, v = self.staged["exp_avg"][key], self.staged["exp_avg_sq"][key]
                m.lerp_(g, 1 - b1)
                v.mul_(b2).addcmul_(g, g.conj(), value=1 - b2)
                denom = (v.sqrt() / bc2_sqrt).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)
