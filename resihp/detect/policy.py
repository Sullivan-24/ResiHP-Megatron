"""When to act on a confirmed fail-slow: ski rental with hysteresis.

Renting = keep paying the per-iteration slowdown; buying = pay the reconfiguration
cost once. Acting when the accumulated excess time reaches the reconfiguration cost
is 2-competitive against an oracle that knows how long the slowdown lasts.
Hysteresis (minimum gain, minimum dwell time between actions) prevents oscillation
on noisy or transient slowdowns.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    act: bool
    reason: str
    accumulated: float = 0.0


class SkiRentalPolicy:
    def __init__(self, min_gain: float = 0.03, min_dwell: int = 50):
        self.min_gain = min_gain
        self.min_dwell = min_dwell
        self.accumulated = 0.0
        self.last_action = -10 ** 9

    def observe(self, iteration: int, current_iter_s: float, best_iter_s: float, reconfig_cost_s: float) -> Decision:
        gain = (current_iter_s - best_iter_s) / max(current_iter_s, 1e-12)
        if gain < self.min_gain:
            self.accumulated = 0.0
            return Decision(False, f"gain {gain:.3f} below {self.min_gain}")
        self.accumulated += current_iter_s - best_iter_s
        if iteration - self.last_action < self.min_dwell:
            return Decision(False, "dwell time not elapsed", self.accumulated)
        if self.accumulated >= reconfig_cost_s:
            acc = self.accumulated
            self.accumulated = 0.0
            self.last_action = iteration
            return Decision(True, f"excess {acc:.2f}s >= reconfiguration cost {reconfig_cost_s:.2f}s", acc)
        return Decision(False, "renting", self.accumulated)
