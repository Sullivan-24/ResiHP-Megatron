"""Fail-slow detection with workload awareness.

Modes (all share the same BOCD core, so comparisons are apples to apples):

* ``greyhound``: change points on the raw iteration-time series; every confirmed
  increase triggers the (expensive) validation phase.
* ``paper``: ResiHP as published -- change points on the raw series, then a check
  against the predicted healthy time (simulator + T = alpha*N + beta*sum l^2); if the
  observation is within ``validate_ratio`` of the prediction the point is removed
  from the series and validation is skipped.
* ``residual``: change points on the *residual* series log(T_obs / T_pred), which
  removes workload-induced fluctuation before detection instead of after it.

``localize`` turns per-group chunk residuals into normalized throughputs p_i that
feed the planner; validation then pinpoints the GPU inside a slow TP group.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from resihp.detect.bocd import BOCD
from resihp.detect.predictor import RLSPredictor


@dataclass
class Detection:
    iteration: int
    mode: str
    magnitude: float
    suspects: dict = field(default_factory=dict)


class FailSlowDetector:
    def __init__(self, mode: str = "residual", *, threshold: float = 0.9, window: int = 3, min_increase: float = 0.08,
                 validate_ratio: float = 1.25, warmup: int = 10, sigma0: float = 0.03, hazard: float = 1 / 250,
                 cooldown: int = 10, calibrate: bool = True, sigma_floor: float = 0.005, prior_strength: float = 10.0):
        if mode not in ("greyhound", "paper", "residual"):
            raise ValueError(mode)
        self.mode = mode
        self.threshold = threshold
        self.window = window
        self.min_increase = min_increase
        self.validate_ratio = validate_ratio
        self.warmup = warmup
        self.cooldown = cooldown
        self.hazard = hazard
        self.calibrate = calibrate
        self.sigma_floor = sigma_floor
        self.prior_strength = prior_strength
        self.bocd = BOCD(hazard=hazard, sigma0=sigma0)
        self.xs: list[float] = []
        self._raw_base: list[float] = []
        self.validations = 0
        self.filtered = 0
        self.detections: list[Detection] = []
        self._last = -10 ** 9

    def _value(self, observed: float, predicted: float | None) -> float | None:
        if self.mode == "residual":
            if predicted is None or predicted <= 0:
                return None
            return math.log(observed / predicted)
        if len(self._raw_base) < self.warmup:
            self._raw_base.append(observed)
        return math.log(observed / statistics.median(self._raw_base))

    def observe(self, iteration: int, observed: float, predicted: float | None = None) -> Detection | None:
        x = self._value(observed, predicted)
        if x is None:
            return None
        before = self.bocd.snapshot()
        self.bocd.update(x)
        self.xs.append(x)
        if self.calibrate and len(self.xs) == self.warmup:
            # Noise prior from the observed warm-up spread, then replay the warm-up samples.
            mean = statistics.fmean(self.xs)
            sigma = max(statistics.pstdev(self.xs), self.sigma_floor)
            self.bocd = BOCD(hazard=self.hazard, mu0=mean, alpha0=self.prior_strength, sigma0=sigma)
            for v in self.xs:
                self.bocd.update(v)
        if len(self.xs) <= self.warmup or iteration - self._last <= self.cooldown:
            return None
        if self.bocd.prob_recent_change(self.window) <= self.threshold:
            return None
        # Only the points after the most likely change point measure the new level.
        since = min(max(1, self.bocd.map_run_length() + 1), self.window)
        recent = self.xs[-since:]
        prior = self.xs[-(since + 4 * self.window):-since] or self.xs[:-since]
        inc = math.exp(statistics.fmean(recent) - statistics.fmean(prior)) - 1.0
        if inc < self.min_increase:
            return None
        if self.mode == "paper" and predicted is not None and observed <= self.validate_ratio * predicted:
            self.bocd = before
            self.xs.pop()
            self.filtered += 1
            return None
        self.validations += 1
        self._last = iteration
        det = Detection(iteration, self.mode, 1.0 + inc)
        self.detections.append(det)
        self.bocd.reset()
        return det


def localize(group_samples: dict[object, list[tuple[float, float, float]]], predictor: RLSPredictor,
             speed_factor: dict[object, float] | None = None, threshold: float = 1.1) -> dict[object, float]:
    """Median observed/expected chunk time per group; groups above ``threshold`` are suspects.

    ``speed_factor`` normalizes groups with different TP degrees to the stage-level predictor.
    Returns {group: p} with p = expected/observed (the planner's normalized throughput).
    """
    out = {}
    for g, samples in group_samples.items():
        f = (speed_factor or {}).get(g, 1.0)
        ratios = [sec / max(predictor.predict(n, s) / f, 1e-12) for n, s, sec in samples]
        if not ratios:
            continue
        r = statistics.median(ratios)
        if r > threshold:
            out[g] = 1.0 / r
    return out
