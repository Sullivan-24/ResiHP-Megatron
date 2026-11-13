import math

import numpy as np
import pytest

from resihp.detect.bocd import BOCD
from resihp.detect.failslow import localize
from resihp.detect.policy import SkiRentalPolicy
from resihp.detect.predictor import RLSPredictor, fit_least_squares
from resihp.detect.synthetic import SyntheticJob, evaluate_detector, generate_trace
from resihp.detect.validation import edge_rounds, ring_edges


def test_rls_recovers_coefficients_and_gates_slow_samples():
    rng = np.random.default_rng(0)
    p = RLSPredictor(scales=(1e4, 1e8, 1.0))
    samples = []
    for _ in range(400):
        n = rng.uniform(4e3, 3.2e4)
        s = rng.uniform(1e6, n * n)
        y = (3e-6 * n + 1.2e-10 * s) * rng.normal(1.0, 0.01)
        p.update(n, s, y)
        samples.append((n, s, y))
    a, b, _ = p.coefficients()
    assert a == pytest.approx(3e-6, rel=0.05) and b == pytest.approx(1.2e-10, rel=0.1)
    fa, fb, _ = fit_least_squares(samples)
    assert fa == pytest.approx(3e-6, rel=0.05)
    before = p.coefficients()
    for _ in range(50):
        assert not p.update(1e4, 1e7, 2 * (3e-6 * 1e4 + 1.2e-10 * 1e7))
    assert p.coefficients() == before and p.rejected == 50


def test_bocd_detects_shift_quickly_and_is_quiet_on_noise():
    rng = np.random.default_rng(1)
    b = BOCD(sigma0=0.02)
    probs = [b.prob_recent_change(3) if b.update(x) is not None else 0 for x in rng.normal(0, 0.02, 200)]
    assert max(probs[20:]) < 0.9
    hits = []
    for i, x in enumerate(rng.normal(0.3, 0.02, 10)):
        b.update(x)
        if b.prob_recent_change(3) > 0.9:
            hits.append(i)
    assert hits and hits[0] <= 2


def test_workload_aware_detection_beats_raw_on_variable_length_data():
    job = SyntheticJob()
    raw_fa = ours_fa = paper_fa = 0
    for seed in (10, 11, 12):
        healthy = generate_trace(job, 120, seed=seed)
        raw_fa += evaluate_detector(job, healthy, "greyhound")["validations"]
        paper_fa += evaluate_detector(job, healthy, "paper")["validations"]
        ours_fa += evaluate_detector(job, healthy, "residual")["validations"]
    assert raw_fa > 0, "long-document bursts fool raw iteration-time detection"
    assert ours_fa == 0 and paper_fa <= raw_fa
    slow = generate_trace(job, 120, seed=10, slow_at=70, factor=1.6)
    hit = evaluate_detector(job, slow, "residual")["detections"]
    assert hit and 70 <= hit[0] <= 74, hit


def test_localization_and_policy():
    p = RLSPredictor(scales=(1.0, 1.0, 1.0), warmup=0)
    for n in range(1, 50):
        p.update(float(n), 0.0, 2.0 * n, force=True)
    samples = {"g0": [(10.0, 0.0, 20.0), (20.0, 0.0, 40.5)], "g1": [(10.0, 0.0, 41.0), (20.0, 0.0, 79.0)]}
    sus = localize(samples, p)
    assert list(sus) == ["g1"] and sus["g1"] == pytest.approx(0.5, rel=0.05)
    pol = SkiRentalPolicy(min_gain=0.05, min_dwell=0)
    assert not pol.observe(0, 1.02, 1.0, 10).act
    acts = [pol.observe(i, 1.5, 1.0, 2.0).act for i in range(1, 6)]
    assert acts == [False, False, False, True, False]


def test_probe_rounds_are_disjoint_and_constant_depth():
    for n, depth in ((8, 2), (7, 3), (2, 1)):
        edges = ring_edges(list(range(n)))
        rounds = edge_rounds(edges)
        assert len(rounds) == depth
        assert sorted(e for r in rounds for e in r) == sorted(edges)
        for r in rounds:
            nodes = [x for e in r for x in e]
            assert len(nodes) == len(set(nodes))
