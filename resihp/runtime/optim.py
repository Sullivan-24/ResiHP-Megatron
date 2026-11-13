"""Adam(W) with explicit, name-keyed state.

Bit-for-bit the update of ``torch.optim.Adam(foreach=False)``; the state lives in
plain tensors keyed by parameter name so it can be sliced, shipped and resharded
by the state-migration engine.
"""

from __future__ import annotations

import math

import torch


class Adam:
    def __init__(self, params: dict[str, torch.nn.Parameter], lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.params = dict(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.state = {n: {"exp_avg": torch.zeros_like(p, memory_format=torch.preserve_format),
                          "exp_avg_sq": torch.zeros_like(p, memory_format=torch.preserve_format)}
                      for n, p in self.params.items()}

    def zero_grad(self) -> None:
        for p in self.params.values():
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        b1, b2 = self.betas
        bc1 = 1 - b1 ** self.step_count
        bc2_sqrt = math.sqrt(1 - b2 ** self.step_count)
        for n, p in self.params.items():
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            if self.weight_decay:
                g = g.add(p, alpha=self.weight_decay)
            s = self.state[n]
            s["exp_avg"].lerp_(g, 1 - b1)
            s["exp_avg_sq"].mul_(b2).addcmul_(g, g.conj(), value=1 - b2)
            denom = (s["exp_avg_sq"].sqrt() / bc2_sqrt).add_(self.eps)
            p.addcdiv_(s["exp_avg"], denom, value=-self.lr / bc1)

    def state_tensors(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for n, p in self.params.items():
            out[f"param/{n}"] = p.data
            out[f"exp_avg/{n}"] = self.state[n]["exp_avg"]
            out[f"exp_avg_sq/{n}"] = self.state[n]["exp_avg_sq"]
        return out

    def load_state_tensors(self, tensors: dict[str, torch.Tensor], step_count: int) -> None:
        with torch.no_grad():
            for n, p in self.params.items():
                p.data.copy_(tensors[f"param/{n}"])
                self.state[n]["exp_avg"].copy_(tensors[f"exp_avg/{n}"])
                self.state[n]["exp_avg_sq"].copy_(tensors[f"exp_avg_sq/{n}"])
        self.step_count = step_count
