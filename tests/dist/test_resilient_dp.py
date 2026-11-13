"""End-to-end checkpoint-free recovery with real processes, the control plane, and kill -9.

Survivors never restart, the global batch is preserved after each exclusion, and
the loss trajectory and final parameters must match single-process training.
"""

import json
import sys
import time
from pathlib import Path

import pytest
import torch

from resihp.control.agent import NodeAgent
from resihp.control.coordinator import Coordinator
from resihp.plan.dp_planner import make_dp_planner
from resihp.runtime.reference_model import RefModelConfig, reference_train
from resihp.state.rng_data import IndexAddressableSampler

pytestmark = pytest.mark.dist

ROOT = Path(__file__).resolve().parents[2]
DEMO = str(ROOT / "examples" / "cpu_resilient_demo.py")


def run_job(out: Path, faults: str, iterations: int, extra: list[str] = (), nnodes: int = 2, nproc: int = 2,
            timeout: float = 240.0):
    coord = Coordinator(planner=make_dp_planner(8), expected_workers=nnodes * nproc, lease_timeout=2.0,
                        hang_timeout=60.0, abort_timeout=20.0, ready_timeout=60.0).start()
    env = {"RESIHP_FAULTS": faults, "PYTHONPATH": str(ROOT), "RESIHP_LOG_LEVEL": "ERROR"}
    agents = []
    try:
        for n in range(nnodes):
            ag = NodeAgent(f"n{n}", [coord.address], hb_interval=0.1, worker_timeout=5.0).start()
            ag.spawn([sys.executable, DEMO, "--iterations", str(iterations), "--out", str(out), "--timeout", "10", *extra], nproc, env=env)
            agents.append(ag)
        deadline = time.monotonic() + timeout
        procs = [w.proc for ag in agents for w in ag.workers.values()]
        while any(p.poll() is None for p in procs):
            assert time.monotonic() < deadline, "job did not finish"
            time.sleep(0.2)
        codes = {w.worker_id: w.proc.returncode for ag in agents for w in ag.workers.values()}
        return codes, list(coord.history)
    finally:
        for ag in agents:
            ag.stop(kill_workers=True)
        coord.stop()


def load(out: Path, worker: str):
    res = json.loads((out / f"{worker}.json").read_text())
    params = torch.load(out / f"{worker}.pt")
    return res, params


def assert_matches_reference(res, params, ref_losses, ref_params, first_iteration=0):
    losses = {int(k): v for k, v in res["losses"].items()}
    for it, v in losses.items():
        assert it >= first_iteration
        assert v == pytest.approx(ref_losses[it], rel=1e-10, abs=1e-12), it
    for k, v in ref_params.items():
        torch.testing.assert_close(params[k], v, rtol=1e-10, atol=1e-12)


def test_two_kills_checkpoint_free(tmp_path):
    iterations = 8
    codes, history = run_job(tmp_path, "kill@iter=3:target=n1g1:phase=reduce;kill@iter=6:target=n0g0:phase=forward",
                             iterations)
    assert codes["n1g1"] == -9 and codes["n0g0"] == -9
    survivors = [w for w, c in codes.items() if c == 0]
    assert sorted(survivors) == ["n0g1", "n1g0"]
    cfg = RefModelConfig()
    ref_losses, ref_params = reference_train(cfg, IndexAddressableSampler(256, 16, 2, seed=7), iterations)
    results = {w: load(tmp_path, w) for w in survivors}
    for w, (res, params) in results.items():
        assert sorted(int(k) for k in res["losses"]) == list(range(iterations))
        assert len(res["generations"]) >= 3
        assert_matches_reference(res, params, ref_losses, ref_params)
    a, b = (results[w][1] for w in survivors)
    assert all(torch.equal(a[k], b[k]) for k in a), "replicas must stay bitwise identical"
    commits = [h for h in history if h["event"] == "commit"]
    assert [len(c["world"]) for c in commits] == [4, 3, 2]


def test_auto_resume_from_resharding_checkpoint_and_snapshot_replication(tmp_path):
    ckpt = tmp_path / "ckpt"
    first = tmp_path / "first"
    codes, _ = run_job(first, "", 4, extra=["--ckpt-dir", str(ckpt), "--ckpt-interval", "2",
                                             "--snapshot-interval", "2"])
    assert all(c == 0 for c in codes.values())
    for w in codes:
        res, _ = load(first, w)
        assert len(res["snapshots_held"]) == 1, "each worker keeps its ring predecessor's snapshot"
    second = tmp_path / "second"
    codes, _ = run_job(second, "", 8, extra=["--ckpt-dir", str(ckpt), "--ckpt-interval", "2"])
    assert all(c == 0 for c in codes.values())
    ref_losses, ref_params = reference_train(RefModelConfig(), IndexAddressableSampler(256, 16, 2, seed=7), 8)
    for w in codes:
        res, params = load(second, w)
        assert sorted(int(k) for k in res["losses"]) == [4, 5, 6, 7]
        assert_matches_reference(res, params, ref_losses, ref_params, first_iteration=4)
