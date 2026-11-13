"""Bayesian online change-point detection (Adams & MacKay, 2007).

Normal observations with unknown mean and variance (Normal-Inverse-Gamma prior,
Student-t predictive), constant hazard, run length truncated at ``rmax`` so each
update is O(rmax). Everything is in log space.
"""

from __future__ import annotations

import copy
import math

import numpy as np


def _logsumexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.sum(np.exp(a - m))))


def _student_t_logpdf(x: float, df: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (x - loc) / scale
    lg = np.vectorize(math.lgamma)
    return (lg((df + 1) / 2) - lg(df / 2) - 0.5 * np.log(df * math.pi) - np.log(scale)
            - (df + 1) / 2 * np.log1p(z * z / df))


class BOCD:
    def __init__(self, hazard: float = 1 / 250, mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 2.0,
                 sigma0: float = 0.03, rmax: int = 400):
        self.log_h = math.log(hazard)
        self.log_1mh = math.log1p(-hazard)
        self.prior = (mu0, kappa0, alpha0, alpha0 * sigma0 ** 2)
        self.rmax = rmax
        self.reset()

    def reset(self) -> None:
        mu0, k0, a0, b0 = self.prior
        self.log_r = np.array([0.0])
        self.mu = np.array([mu0])
        self.kappa = np.array([k0])
        self.alpha = np.array([a0])
        self.beta = np.array([b0])
        self.t = 0

    def snapshot(self) -> "BOCD":
        return copy.deepcopy(self)

    def update(self, x: float) -> np.ndarray:
        """Feed one observation; returns the run-length posterior P(r_t = r)."""
        df = 2 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        logpred = _student_t_logpdf(x, df, self.mu, scale)
        growth = self.log_r + logpred + self.log_1mh
        cp = _logsumexp(self.log_r + logpred + self.log_h)
        log_r = np.concatenate([[cp], growth])
        log_r -= _logsumexp(log_r)
        mu0, k0, a0, b0 = self.prior
        mu_new = (self.kappa * self.mu + x) / (self.kappa + 1)
        beta_new = self.beta + self.kappa * (x - self.mu) ** 2 / (2 * (self.kappa + 1))
        self.mu = np.concatenate([[mu0], mu_new])
        self.kappa = np.concatenate([[k0], self.kappa + 1])
        self.alpha = np.concatenate([[a0], self.alpha + 0.5])
        self.beta = np.concatenate([[b0], beta_new])
        if len(log_r) > self.rmax:
            log_r = log_r[: self.rmax]
            log_r -= _logsumexp(log_r)
            self.mu, self.kappa, self.alpha, self.beta = (v[: self.rmax] for v in (self.mu, self.kappa, self.alpha, self.beta))
        self.log_r = log_r
        self.t += 1
        return np.exp(log_r)

    def prob_recent_change(self, window: int) -> float:
        return float(np.exp(_logsumexp(self.log_r[: window + 1])))

    def map_run_length(self) -> int:
        return int(np.argmax(self.log_r))

    def segment_mean(self) -> float:
        """Posterior mean of the current segment under the MAP run length."""
        return float(self.mu[self.map_run_length()])
