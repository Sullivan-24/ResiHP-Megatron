"""End-to-end: a GPU in a TP group is killed; the coordinator replans (TP shrink / repartition /
migration), survivors migrate parameters + Adam state live, and training continues with the
exact semantics of single-process training."""

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from resihp.control.agent import NodeAgent
from resihp.control.coordinator import Coordinator
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
    num_stages=2, num_replicas=2, tp=2, num_microbatches=8, sequence_parallel=True, migration_rounds=16)


def run(out: Path, faults: str, iterations: int, extra=(), timeout=240.0, planner_cfg=PLANNER):
    coord = Coordinator(planner=make_hetero_planner(planner_cfg), expected_workers=8, lease_timeout=2.0,
                        hang_timeout=120.0, abort_timeout=30.0, ready_timeout=120.0).start()
    env = {"RESIHP_FAULTS": faults, "PYTHONPATH": str(ROOT), "RESIHP_LOG_LEVEL": "ERROR"}
    agents = []
    try:
        for n in range(2):
            ag = NodeAgent(f"n{n}", [coord.address], hb_interval=0.1, worker_timeout=10.0).start()
            ag.spawn([sys.executable, DEMO, "--iterations", str(iterations), "--out", str(out), "--timeout", "10", *extra], 4, env=env)
            agents.append(ag)
        procs = [w.proc for ag in agents for w in ag.workers.values()]
        deadline = time.monotonic() + timeout
        while any(p.poll() is None for p in procs):
            assert time.monotonic() < deadline, "job did not finish"
            time.sleep(0.2)
        codes = {w.worker_id: w.proc.returncode for ag in agents for w in ag.workers.values()}
        return codes, coord.payload, list(coord.history)
    finally:
        for ag in agents:
            ag.stop(kill_workers=True)
        coord.stop()


@pytest.mark.parametrize("channels,vpp,cp", [(False, 1, 1), (True, 1, 1), (True, 2, 1), (True, 1, 2)])
def test_kill_in_tp_group_replans_and_migrates_state(tmp_path, channels, vpp, cp):
    iterations = 6
    extra = ["--channels"] if channels else []
    cfg = replace(PLANNER, vpp=vpp, cp=cp, tp=PLANNER.tp // cp)
    codes, payload, history = run(tmp_path, "kill@iter=2:target=n0g3:phase=forward", iterations, extra,
                                  planner_cfg=cfg)
    assert codes["n0g3"] == -9
    assert all(c == 0 for w, c in codes.items() if w != "n0g3"), codes
    final = HeteroParallelPlan.from_json(payload["plan"])
    assert "n0g3" not in final.world and final.generation >= 2 and final.vpp == vpp
    assert all(g.cp == cp for g in final.stage_groups)
    check_final_state(tmp_path, final, history, iterations, RefModelConfig())


@pytest.mark.parametrize("phase", ["forward", "loss"])
def test_fp8_amax_history_migrates_and_rolls_back(tmp_path, phase):
    """``loss`` kills the victim after every rank updated its amax history for the in-flight
    iteration: survivors must roll the history back or the replay would apply it twice."""
    iterations = 6
    codes, payload, history = run(tmp_path, f"kill@iter=2:target=n0g3:phase={phase}", iterations,
                                  ["--amax-history", "4"])
    assert codes["n0g3"] == -9 and all(c == 0 for w, c in codes.items() if w != "n0g3"), codes
    final = HeteroParallelPlan.from_json(payload["plan"])
    check_final_state(tmp_path, final, history, iterations, RefModelConfig(amax_history=4))


def check_final_state(tmp_path, final, history, iterations, cfg):
    ref_losses, ref_params = reference_train(cfg, IndexAddressableSampler(256, 16, 2, seed=7), iterations)
    specs = reference_specs(cfg)
    assembled: dict[str, torch.Tensor] = {}
    covered: dict[str, list] = {}
    migrated = 0
    for w in final.world:
        info = json.loads((tmp_path / f"{w}.json").read_text())
        assert info["generation"] == final.generation
        for it, v in info["losses"].items():
            assert v == pytest.approx(ref_losses[int(it)], rel=1e-9)
        migrated += info["last_transfer"].get("received", 0)
        shards = torch.load(tmp_path / f"{w}.pt")
        for key, iv in info["layout"].items():
            spec = specs[key]
            want = ref_params[key] if spec.dim is None else ref_params[key].narrow(spec.dim, iv[0], iv[1] - iv[0])
            torch.testing.assert_close(shards[key], want, rtol=1e-9, atol=1e-11)
            covered.setdefault(key, []).append(tuple(iv))
    assert set(covered) == set(specs), "every tensor is held somewhere in the final plan"
    assert migrated > 0, "the reconfiguration moved state"
    old_layers = [h for h in history if h["event"] == "commit"]
    assert len(old_layers) >= 2
