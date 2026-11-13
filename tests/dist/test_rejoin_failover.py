"""Rejoin of a replacement node, and coordinator failover through the KV lease, with real processes."""

import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

from resihp.control.agent import NodeAgent
from resihp.control.coordinator import Coordinator
from resihp.control.kvstore import InMemoryKV
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
HELPER = str(ROOT / "tests" / "helpers" / "control_worker.py")
PLANNER = PlannerConfig(
    model=ModelSpec(num_layers=4, hidden=8, ffn=24, num_heads=12, num_kv_heads=12, vocab=16, seq_len=24,
                    micro_batch_size=2),
    profile=ModelProfile(alpha=1e-6, beta=1e-9, last_stage_extra_layers=0.0),
    num_stages=2, num_replicas=2, tp=2, num_microbatches=8, sequence_parallel=True, migration_rounds=16)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(pred, timeout, what):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out: {what}")


def check_against_reference(out: Path, final: HeteroParallelPlan, iterations: int):
    ref_losses, ref_params = reference_train(RefModelConfig(), IndexAddressableSampler(256, 16, 2, seed=7), iterations)
    specs = reference_specs(RefModelConfig())
    covered = set()
    for w in final.world:
        info = json.loads((out / f"{w}.json").read_text())
        for it, v in info["losses"].items():
            assert v == pytest.approx(ref_losses[int(it)], rel=1e-9)
        shards = torch.load(out / f"{w}.pt")
        for key, iv in info["layout"].items():
            spec = specs[key]
            want = ref_params[key] if spec.dim is None else ref_params[key].narrow(spec.dim, iv[0], iv[1] - iv[0])
            torch.testing.assert_close(shards[key], want, rtol=1e-9, atol=1e-11)
            covered.add(key)
    assert covered == set(specs)


def spawn_args(out, iterations, sleep):
    return [sys.executable, DEMO, "--iterations", str(iterations), "--out", str(out), "--timeout", "10",
            "--iter-sleep", str(sleep)]


def test_replacement_node_rejoins_and_receives_state(tmp_path):
    iterations = 40
    env = {"RESIHP_FAULTS": "kill@iter=2:target=n0g3:phase=forward", "PYTHONPATH": str(ROOT),
           "RESIHP_LOG_LEVEL": "ERROR"}
    coord = Coordinator(planner=make_hetero_planner(PLANNER), expected_workers=8, lease_timeout=2.0,
                        hang_timeout=120.0, abort_timeout=30.0, ready_timeout=120.0).start()
    agents = []
    try:
        for n in range(2):
            ag = NodeAgent(f"n{n}", [coord.address], hb_interval=0.1, worker_timeout=10.0).start()
            ag.spawn(spawn_args(tmp_path, iterations, 0.05), 4, env=env)
            agents.append(ag)
        wait_for(lambda: coord.committed_generation >= 2, 60, "failure handled")
        degraded = HeteroParallelPlan.from_json(coord.payload["plan"])
        assert "n0g3" not in degraded.world
        spare = NodeAgent("n2", [coord.address], hb_interval=0.1, worker_timeout=10.0).start()
        spare.spawn(spawn_args(tmp_path, iterations, 0.05), 4, env={**env, "RESIHP_FAULTS": ""})
        agents.append(spare)
        wait_for(lambda: any(h["event"] == "commit" and h["reason"].startswith("join") for h in coord.history),
                 60, "join replan")
        procs = [w.proc for ag in agents for w in ag.workers.values()]
        wait_for(lambda: all(p.poll() is not None for p in procs), 240, "job end")
        final = HeteroParallelPlan.from_json(coord.payload["plan"])
    finally:
        for ag in agents:
            ag.stop(kill_workers=True)
        coord.stop()
    joined = [d for d in final.world if d.startswith("n2")]
    assert len(joined) == 2 and all(final.group(r, s) is not None for r in range(2) for s in range(2))
    for d in joined:
        info = json.loads((tmp_path / f"{d}.json").read_text())
        assert info["last_transfer"].get("received", 0) > 0, "the rejoined worker got its state from peers"
    check_against_reference(tmp_path, final, iterations)


def test_coordinator_failover_keeps_training_and_handles_later_failures(tmp_path):
    iterations = 40
    kv = InMemoryKV()
    ports = [free_port(), free_port()]
    planner = make_hetero_planner(PLANNER)
    leader = Coordinator(port=ports[0], kv=kv, planner=planner, expected_workers=8, lease_timeout=2.0,
                         hang_timeout=120.0, abort_timeout=30.0, ready_timeout=120.0, coordinator_id="c0",
                         leader_ttl=0.6).start()
    standby = Coordinator(port=ports[1], kv=kv, planner=planner, expected_workers=8, lease_timeout=2.0,
                          hang_timeout=120.0, abort_timeout=30.0, ready_timeout=120.0, coordinator_id="c1",
                          leader_ttl=0.6).start(block_until_leader=False)
    addrs = [("127.0.0.1", p) for p in ports]
    env = {"RESIHP_FAULTS": "kill@iter=25:target=n1g2:phase=forward", "PYTHONPATH": str(ROOT),
           "RESIHP_LOG_LEVEL": "ERROR"}
    agents = []
    try:
        for n in range(2):
            ag = NodeAgent(f"n{n}", addrs, hb_interval=0.1, worker_timeout=10.0, self_fence_after=10.0).start()
            ag.spawn(spawn_args(tmp_path, iterations, 0.08), 4, env=env)
            agents.append(ag)
        wait_for(lambda: leader.committed_generation >= 1, 60, "start")
        time.sleep(1.0)
        leader.stop()
        wait_for(lambda: standby.is_leader, 10, "standby takes over")
        assert standby.committed_generation == 1
        wait_for(lambda: standby.committed_generation >= 2, 120, "post-failover failure handled by the new leader")
        procs = [w.proc for ag in agents for w in ag.workers.values()]
        wait_for(lambda: all(p.poll() is not None for p in procs), 240, "job end")
        final = HeteroParallelPlan.from_json(standby.payload["plan"])
        history = list(standby.history)
    finally:
        for ag in agents:
            ag.stop(kill_workers=True)
        standby.stop()
    assert "n1g2" not in final.world
    assert any(h["event"] == "failure" and h["workers"] == ["n1g2"] for h in history)
    assert any(h["event"] == "finished" for h in history)
    check_against_reference(tmp_path, final, iterations)


def test_leader_dies_mid_transaction_and_successor_completes_it(tmp_path):
    kv = InMemoryKV()
    ports = [free_port(), free_port()]
    release = threading.Event()

    def everyone(view):
        return sorted(w.worker_id for w in view.alive), {"kind": "test"}

    def stuck(view):
        if view.reason.startswith("replan:stuck"):
            release.wait(30)
        return everyone(view)

    common = dict(kv=kv, expected_workers=4, lease_timeout=2.0, leader_ttl=0.6)
    leader = Coordinator(port=ports[0], planner=stuck, coordinator_id="c0", **common).start()
    standby = Coordinator(port=ports[1], planner=everyone, coordinator_id="c1", **common).start(block_until_leader=False)
    addrs = [("127.0.0.1", p) for p in ports]
    log = tmp_path / "events.jsonl"
    env = {"TEST_EVENT_LOG": str(log), "TEST_TRIGGER_DIR": str(tmp_path), "PYTHONPATH": str(ROOT)}
    agents = []
    try:
        for n in range(2):
            ag = NodeAgent(f"n{n}", addrs, hb_interval=0.1, worker_timeout=10.0, self_fence_after=10.0).start()
            ag.spawn([sys.executable, HELPER], 2, env=env)
            agents.append(ag)
        wait_for(lambda: leader.committed_generation >= 1, 30, "start")
        leader.request_replan("replan:stuck")
        wait_for(lambda: leader.generation == 2 and len(leader._aborted.get(2, {})) == 4, 30, "workers parked in gen 2")
        leader.stop()
        wait_for(lambda: standby.committed_generation >= 3, 30, "successor completes the interrupted reconfiguration")
        release.set()
        time.sleep(0.5)
        state = json.loads(kv.get("resihp/state"))
        history = list(standby.history)
    finally:
        release.set()
        for ag in agents:
            ag.stop(kill_workers=True)
        standby.stop()
        leader.stop()
    assert history[-1]["event"] == "commit" and history[-1]["reason"] == "failover"
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert {e["worker"] for e in events if e["gen"] == 3} == {"n0g0", "n0g1", "n1g0", "n1g1"}
    assert not any(e["gen"] == 2 for e in events)
    assert state["committed_generation"] == 3, "the deposed leader must not overwrite its successor's state"
