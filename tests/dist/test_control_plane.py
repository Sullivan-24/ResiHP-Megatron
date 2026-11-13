"""Chaos test of the control plane with real worker processes.

3 nodes x 2 workers. Faults, in order: SIGKILL, SIGSTOP (hang without heartbeat),
whole-node loss (agent and workers gone; only the lease reveals it), and a
main-thread hang with a live heartbeat thread (progress-based detection).
Every fault must be detected, fenced, and followed by a committed generation that
excludes exactly the failed workers.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from resihp.control.agent import NodeAgent
from resihp.control.coordinator import Coordinator

pytestmark = pytest.mark.dist

HELPER = str(Path(__file__).resolve().parents[1] / "helpers" / "control_worker.py")
ROOT = str(Path(__file__).resolve().parents[2])


def _wait(pred, timeout, what):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def test_failure_detection_fencing_and_generations(tmp_path):
    events = tmp_path / "events.jsonl"
    triggers = tmp_path / "triggers"
    triggers.mkdir()
    coord = Coordinator(expected_workers=6, lease_timeout=1.0, hang_timeout=1.5, abort_timeout=5.0,
                        ready_timeout=10.0, host_store=False).start()
    agents = {}
    env = {"TEST_EVENT_LOG": str(events), "TEST_TRIGGER_DIR": str(triggers),
           "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    try:
        for n in range(3):
            ag = NodeAgent(f"n{n}", [coord.address], hb_interval=0.1, worker_timeout=1.0).start()
            ag.spawn([sys.executable, HELPER], 2, env=env)
            agents[f"n{n}"] = ag

        assert coord.wait_for_commit(1, timeout=30)
        assert sorted(coord.world) == ["n0g0", "n0g1", "n1g0", "n1g1", "n2g0", "n2g1"]

        def committed_without(*gone):
            return lambda: coord.phase == "running" and not set(gone) & set(coord.world) and \
                coord.committed_generation > 1

        # 1. fail-stop: SIGKILL
        t0 = time.monotonic()
        os.kill(agents["n2"].workers["n2g1"].pid, signal.SIGKILL)
        _wait(committed_without("n2g1"), 15, "commit after SIGKILL")
        kill_latency = time.monotonic() - t0
        assert len(coord.world) == 5

        # 2. hang without heartbeats: SIGSTOP -> agent heartbeat timeout -> fenced (SIGKILL)
        stopped = agents["n2"].workers["n2g0"].pid
        os.kill(stopped, signal.SIGSTOP)
        _wait(committed_without("n2g0", "n2g1"), 15, "commit after SIGSTOP")
        _wait(lambda: agents["n2"].workers["n2g0"].proc.poll() is not None, 5, "fenced process to exit")

        # 3. node loss: agent and workers vanish; only the lease expiry can tell
        lost = agents.pop("n1")
        lost.stop()
        for w in lost.workers.values():
            os.kill(w.pid, signal.SIGKILL)
        _wait(committed_without("n1g0", "n1g1"), 15, "commit after node loss")
        assert any("lease expired" in h.get("reason", "") for h in coord.history if h["event"] == "failure")

        # 4. main thread hung while heartbeats continue: progress-based detection picks the culprit
        (triggers / "stall_n0g0").touch()
        (triggers / "wait").touch()
        _wait(lambda: coord.world == ["n0g1"] and coord.phase == "running", 20, "commit after silent hang")
        hang = [h for h in coord.history if h["event"] == "failure" and h["workers"] == ["n0g0"]]
        assert hang and "hang" in hang[0]["reason"]

        records = [json.loads(l) for l in events.read_text().splitlines()]
        last = {}
        for r in records:
            last[r["worker"]] = r
        assert last["n0g1"]["world"] == ["n0g1"] and last["n0g1"]["gen"] == coord.committed_generation
        gens = [h["generation"] for h in coord.history if h["event"] == "commit"]
        assert gens == sorted(gens) and len(gens) >= 5
        assert kill_latency < 5.0
    finally:
        for ag in agents.values():
            ag.stop(kill_workers=True)
        coord.stop()
