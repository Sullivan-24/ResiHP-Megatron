"""Online workload-aware execution-time predictor.

Per micro-batch and per stage: ``T = alpha * N + beta * sum(l_i^2) + gamma`` where ``N``
is the packed token count and ``l_i`` the document lengths (attention is
block-diagonal under packing). Estimated by recursive least squares with a
forgetting factor, so there is no fixed warm-up window and slow drift is tracked.

Updates are *gated*: a sample whose residual exceeds ``gate`` is not learned from,
otherwise the predictor would adapt to a fail-slow and hide it.
"""

from __future__ import annotations

import math

import numpy as np


class RLSPredictor:
    def __init__(self, scales: tuple[float, ...] = (1e4, 1e8, 1.0), forgetting: float = 0.995, delta: float = 1e4,
                 gate: float = 0.15, warmup: int = 8):
        self.scales = np.asarray(scales, dtype=np.float64)
        self.lam = forgetting
        self.theta = np.zeros(len(scales))
        self.P = np.eye(len(scales)) * delta
        self.gate = gate
        self.warmup = warmup
        self.n = 0
        self.rejected = 0

    def features(self, tokens: float, sum_sq: float) -> np.ndarray:
        return np.array([tokens, sum_sq, 1.0]) / self.scales

    def predict(self, tokens: float, sum_sq: float) -> float:
        return float(self.theta @ self.features(tokens, sum_sq))

    def update(self, tokens: float, sum_sq: float, seconds: float, force: bool = False) -> bool:
        x = self.features(tokens, sum_sq)
        pred = float(self.theta @ x)
        if not force and self.n >= self.warmup and pred > 0 and abs(math.log(max(seconds, 1e-12) / pred)) > self.gate:
            self.rejected += 1
            return False
        Px = self.P @ x
        k = Px / (self.lam + x @ Px)
        self.theta = self.theta + k * (seconds - pred)
        self.P = (self.P - np.outer(k, Px)) / self.lam
        self.n += 1
        return True

    def coefficients(self) -> tuple[float, float, float]:
        a, b, c = self.theta / self.scales
        return float(a), float(b), float(c)


def fit_least_squares(samples: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Batch fit of (alpha, beta, gamma) from (tokens, sum_sq, seconds) samples (warm-start/profiling)."""
    X = np.array([[t, s, 1.0] for t, s, _ in samples])
    y = np.array([sec for _, _, sec in samples])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])
