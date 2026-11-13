"""End-to-end fail-slow handling: one worker becomes 6x slower; the workload-aware detector
flags exactly that device, the policy triggers a planned replan, the planner re-balances
around it, and training semantics are unchanged."""

import json
import sys
import time
from pathlib import Path

import pytest
import torch

from resihp.control.agent import NodeAgent
from resihp.control.coordinator import Coordinator
from resihp.detect.service import DetectorService
from resihp.plan.coordinator_adapter import make_hetero_planner
from resihp.plan.hetero_plan import HeteroParallelPlan
from resihp.plan.model_spec import ModelProfile, ModelSpec
from resihp.plan.planner import PlannerConfig
from resihp.runtime.reference_model import RefModelConfig, reference_train
from resihp.state.layer_registry import reference_specs
from resihp.state.rng_data import IndexAddressableSampler

pytestmark = pytest.mark.dist

ROOT = Path(__file__).resolve().parents[2]
DEMO = str(ROOT / "examples" / "hetero_demo.py")

PLANNER = PlannerConfig(
    model=ModelSpec(num_layers=4, hidden=8, ffn=24, num_heads=12, num_kv_heads=12, vocab=16, seq_len=24,
                    micro_batch_size=2),
    profile=ModelProfile(alpha=1e-6, beta=1e-9, last_stage_extra_layers=0.0),
    num_stages=2, num_replicas=3, tp=1, num_microbatches=12, sequence_parallel=False, migration_rounds=16)

SLOW, SLOW_AT, ITERS = "n0g1", 10, 22


def test_failslow_detected_localized_and_mitigated(tmp_path):
    coord = Coordinator(planner=make_hetero_planner(PLANNER), expected_workers=6, lease_timeout=2.0,
                        hang_timeout=120.0, abort_timeout=60.0, ready_timeout=120.0)
    svc = DetectorService(coord, warmup=6, min_increase=0.5)
    coord.report_handler = svc.handle
    coord.start()
    env = {"RESIHP_FAULTS": f"slow@iter={SLOW_AT}:target={SLOW}:factor=6", "PYTHONPATH": str(ROOT),
           "RESIHP_LOG_LEVEL": "ERROR"}
    agents = []
    try:
        for n in range(2):
            ag = NodeAgent(f"n{n}", [coord.address], hb_interval=0.1, worker_timeout=15.0).start()
            ag.spawn([sys.executable, DEMO, "--iterations", str(ITERS), "--out", str(tmp_path), "--gbs", "24",
                      "--timeout", "20"], 3, env=env)
            agents.append(ag)
        procs = [w.proc for ag in agents for w in ag.workers.values()]
        deadline = time.monotonic() + 240
        while any(p.poll() is None for p in procs):
            assert time.monotonic() < deadline, "job did not finish"
            time.sleep(0.2)
        codes = {w.worker_id: w.proc.returncode for ag in agents for w in ag.workers.values()}
        payload, history, events = coord.payload, list(coord.history), list(svc.events)
    finally:
        for ag in agents:
            ag.stop(kill_workers=True)
        svc.stop()
        coord.stop()

    assert all(c == 0 for c in codes.values()), codes
    acted = [e for e in events if e["act"]]
    assert acted, events
    assert all(e["iteration"] >= SLOW_AT for e in events), f"false alarm before the slowdown: {events}"
    assert acted[0]["devices"] == [SLOW] and acted[0]["iteration"] <= SLOW_AT + 6 and acted[0]["ratio"] > 1.5
    assert any(h["event"] == "commit" and h["reason"].startswith("failslow") for h in history)
    final = HeteroParallelPlan.from_json(payload["plan"])
    g = final.group_of_device(SLOW)
    load = len(final.microbatches_of(g.replica, g.stage)) if g else 0
    assert g is None or load < PLANNER.num_microbatches // PLANNER.num_replicas, "slow device must shed work"

    ref_losses, ref_params = reference_train(RefModelConfig(), IndexAddressableSampler(256, 24, 2, seed=7), ITERS)
    specs = reference_specs(RefModelConfig())
    for w in final.world:
        info = json.loads((tmp_path / f"{w}.json").read_text())
        for it, v in info["losses"].items():
            assert v == pytest.approx(ref_losses[int(it)], rel=1e-9)
        shards = torch.load(tmp_path / f"{w}.pt")
        for key, iv in info["layout"].items():
            spec = specs[key]
            want = ref_params[key] if spec.dim is None else ref_params[key].narrow(spec.dim, iv[0], iv[1] - iv[0])
            torch.testing.assert_close(shards[key], want, rtol=1e-9, atol=1e-11)
