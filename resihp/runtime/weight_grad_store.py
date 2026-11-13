"""Zero-bubble B/W split for the IR's ``W`` instruction.

B produces only the input gradient, which unblocks the previous stage; W later produces
the parameter gradients of the same micro-batch and can be scheduled into bubbles.

* ``AutogradWeightGradStore`` works for any PyTorch module: B runs
  ``autograd.grad`` w.r.t. the stage input with the graph retained, W runs it w.r.t.
  the parameters. W re-traverses the graph between the output and the parameters,
  so it is the semantic reference (used by the CPU runtime), not the fast path.
* ``MegatronWeightGradStore`` drives Megatron's own split: with
  ``config.delay_wgrad_compute=True`` (plus ``external_wgrad_schedule``, patch 0006) the
  Transformer Engine linear layers queue their weight-gradient GEMMs during backward
  and ``backward_dw()`` runs them. The queues are FIFO per module, so W must follow B
  order per model chunk, which the simulator's W FIFO guarantees and this store checks.
"""

from __future__ import annotations

import contextlib
from typing import Callable

import torch


class AutogradWeightGradStore:
    def __init__(self, params: Callable[[], list[torch.Tensor]]):
        self.params = params
        self._pending: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}

    def backward_input(self, mb: int, out: torch.Tensor, grad: torch.Tensor | None,
                       inp: torch.Tensor | None) -> torch.Tensor | None:
        self._pending[mb] = (out, grad)
        if inp is None:
            return None
        (dx,) = torch.autograd.grad(out, [inp], grad_outputs=grad, retain_graph=True)
        return dx

    def compute(self, mb: int) -> None:
        out, grad = self._pending.pop(mb)
        params = [p for p in self.params() if p.requires_grad]
        grads = torch.autograd.grad(out, params, grad_outputs=grad, allow_unused=True)
        with torch.no_grad():
            for p, g in zip(params, grads):
                if g is None:
                    continue
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad.add_(g)

    def pending(self) -> list[int]:
        return sorted(self._pending)

    def clear(self) -> None:
        self._pending.clear()


class MegatronWeightGradStore:
    """``chunks`` maps a virtual stage to its model chunk; keys are ``(stage, mb)``."""

    def __init__(self, chunks: dict[int, torch.nn.Module]):
        self.modules = {s: [m for m in chunk.modules() if hasattr(m, "backward_dw")] for s, chunk in chunks.items()}
        self._order: dict[int, list[int]] = {s: [] for s in chunks}

    @contextlib.contextmanager
    def deferring(self, key: tuple[int, int]):
        yield
        self._order[key[0]].append(key[1])

    def compute(self, key: tuple[int, int]) -> None:
        stage, mb = key
        queue = self._order[stage]
        if not queue or queue[0] != mb:
            raise RuntimeError(f"W({mb}) of stage {stage} out of B order {queue}: wgrad queues are FIFO")
        queue.pop(0)
        for m in self.modules[stage]:
            m.backward_dw()

    def pending(self) -> list[tuple[int, int]]:
        return [(s, mb) for s, q in self._order.items() for mb in q]

    def clear(self) -> None:
        for q in self._order.values():
            q.clear()
