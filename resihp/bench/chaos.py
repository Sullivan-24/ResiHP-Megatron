"""Real-process chaos benchmark on CPU (8 workers, gloo): MTTR breakdown per fault type.

Faults are injected from outside the workers at a recorded time ``t0`` on the
coordinator's clock: ``kill`` (SIGKILL), ``hang`` (SIGSTOP: process alive, heartbeats
and progress frozen) and ``slow`` (50% SIGSTOP/SIGCONT duty cycle, i.e. a 2x fail-slow).
The coordinator's transaction history splits recovery into detection, abort, planning,
generation build + state migration, and commit. Every trial also checks that the final
parameters equal the single-process reference (the semantics guarantee).

The CPU harness bounds absolute numbers (gloo, process start-up, Python); the breakdown,
not the totals, is what transfers to GPUs.
"""

from __future__ import annotations

import json
import os
import signal
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[2]
DEMO = str(ROOT / "examples" / "hetero_demo.py")
PLANNER = PlannerConfig(
    model=ModelSpec(num_layers=4, hidden=8, ffn=24, num_heads=12, num_kv_heads=12, vocab=16, seq_len=24,
                    micro_batch_size=2),
    profile=ModelProfile(alpha=1e-6, beta=1e-9, last_stage_extra_layers=0.0),
    num_stages=2, num_replicas=2, tp=2, num_microbatches=8, sequence_parallel=True, migration_rounds=16)
FAULTS = ("kill", "hang", "slow")


def _wait(pred, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise TimeoutError(what)
        time.sleep(0.02)


def _duty_cycle(pid: int, stop: threading.Event, period: float = 0.01) -> None:
    while not stop.is_set():
        try:
            os.kill(pid, signal.SIGSTOP)
            time.sleep(period / 2)
            os.kill(pid, signal.SIGCONT)
            time.sleep(period / 2)
        except ProcessLookupError:
            return


def verify(out: Path, final: HeteroParallelPlan, iterations: int) -> bool:
    cfg = RefModelConfig()
    ref_losses, ref_params = reference_train(cfg, IndexAddressableSampler(256, 16, 2, seed=7), iterations)
    specs = reference_specs(cfg)
    for w in final.world:
        info = json.loads((out / f"{w}.json").read_text())
        if any(abs(v - ref_losses[int(it)]) > 1e-9 * abs(ref_losses[int(it)]) for it, v in info["losses"].items()):
            return False
        shards = torch.load(out / f"{w}.pt")
        for key, iv in info["layout"].items():
            spec = specs[key]
            want = ref_params[key] if spec.dim is None else ref_params[key].narrow(spec.dim, iv[0], iv[1] - iv[0])
            if not torch.allclose(shards[key], want, rtol=1e-9, atol=1e-11):
                return False
    return True


def run_trial(fault: str, iterations: int = 80, inject_after: int | None = None, iter_sleep: float = 0.05) -> dict:
    """``inject_after`` defaults to iteration 8, or 20 for ``slow``: the detector calibrates its
    noise prior during warm-up, so a slowdown that starts inside warm-up is partly absorbed."""
    from resihp.detect.service import DetectorService

    inject_after = inject_after if inject_after is not None else (20 if fault == "slow" else 8)
    out = Path(tempfile.mkdtemp(prefix=f"resihp-chaos-{fault}-"))
    # The hang timeout must exceed the communication timeout (10 s here, like NCCL's in
    # production): gloo cannot abort an op already blocked on a frozen peer, so survivors may
    # spend up to that long before they resume, and that must not look like a second hang.
    coord = Coordinator(planner=make_hetero_planner(PLANNER), expected_workers=8, lease_timeout=2.0,
                        hang_timeout=30.0, abort_timeout=30.0, ready_timeout=120.0)
    svc = DetectorService(coord, warmup=6, min_increase=0.5) if fault == "slow" else None
    if svc is not None:
        coord.report_handler = svc.handle
    coord.start()
    env = {"PYTHONPATH": str(ROOT), "RESIHP_LOG_LEVEL": "ERROR"}
    agents, stop = [], threading.Event()
    try:
        for n in range(2):
            ag = NodeAgent(f"n{n}", [coord.address], hb_interval=0.1, worker_timeout=3.0).start()
            ag.spawn([sys.executable, DEMO, "--iterations", str(iterations), "--out", str(out), "--timeout", "10",
                      "--iter-sleep", str(iter_sleep)], 4, env=env)
            agents.append(ag)
        _wait(lambda: coord.committed_generation >= 1, 60, "start")
        _wait(lambda: min((w.progress.get("iteration", -1) for w in coord.membership.alive()), default=-1)
              >= inject_after, 120, "warm-up")
        victim = agents[0].workers["n0g3"].proc.pid
        at = max(w.progress.get("iteration", -1) for w in coord.membership.alive())
        t0 = time.monotonic()
        if fault == "kill":
            os.kill(victim, signal.SIGKILL)
        elif fault == "hang":
            os.kill(victim, signal.SIGSTOP)
        else:
            threading.Thread(target=_duty_cycle, args=(victim, stop), daemon=True).start()
        procs = [w.proc for ag in agents for w in ag.workers.values()]
        _wait(lambda: coord.committed_generation >= 2 or coord.phase == "finished", 180, "recovery commit")
        recovered = coord.committed_generation >= 2
        t_commit = time.monotonic()

        def resumed() -> bool:
            live = [w for w in coord.membership.alive() if w.worker_id in coord.world]
            return bool(live) and min(w.progress.get("iteration", -1) for w in live) > at + 1

        if recovered:
            _wait(lambda: resumed() or coord.phase == "finished", 120, "training resumed")
        t_resume = time.monotonic()
        stop.set()
        _wait(lambda: all(p.poll() is not None for p in procs), 300, "job end")
        history = list(coord.history)
        final = HeteroParallelPlan.from_json(coord.payload["plan"])
    finally:
        stop.set()
        for ag in agents:
            ag.stop(kill_workers=True)
        if svc is not None:
            svc.stop()
        coord.stop()

    def first(event: str, gen: int | None = None) -> float | None:
        for h in history:
            if h["event"] == event and h["t"] >= t0 and (gen is None or h.get("generation") == gen):
                return h["t"]
        return None

    if not recovered:
        return {"fault": fault, "recovered": False, "correct": verify(out, final, iterations)}
    gen = next(h["generation"] for h in history if h["event"] == "commit" and h["t"] >= t0)
    detected = first("failure") if fault != "slow" else first("notice", gen)
    phases = {"detect_s": detected - t0, "abort_s": first("aborted", gen) - first("notice", gen),
              "plan_s": first("planned", gen) - first("aborted", gen),
              "build_and_migrate_s": first("ready", gen) - first("planned", gen),
              "commit_s": t_commit - first("ready", gen), "total_s": t_commit - t0,
              "resume_s": t_resume - t_commit}
    return {"fault": fault, "recovered": True, **phases, "generations": gen, "final_world": len(final.world),
            "correct": verify(out, final, iterations)}


def run(trials: int = 3, faults=FAULTS) -> dict:
    out: dict[str, dict] = {}
    for f in faults:
        rows = [run_trial(f) for _ in range(trials)]
        ok = [r for r in rows if r["recovered"]]
        keys = [k for k in ok[0] if k.endswith("_s")] if ok else []
        out[f] = {"trials": rows, "recovered": len(ok), "median": {k: statistics.median(r[k] for r in ok) for k in keys},
                  "all_correct": all(r["correct"] for r in rows)}
    return out


def to_markdown(res: dict) -> str:
    lines = ["| fault | recovered | detect | abort | plan | build + migrate | commit | fault -> commit | "
             "commit -> 2 iters | semantics |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f, r in res.items():
        m, n = r["median"], len(r["trials"])
        sem = f"{'exact' if r['all_correct'] else 'MISMATCH'}"
        if not m:
            lines.append(f"| {f} | 0/{n} | - | - | - | - | - | - | - | {sem} |")
            continue
        lines.append(f"| {f} | {r['recovered']}/{n} | {m['detect_s']:.2f}s | {m['abort_s']:.2f}s | {m['plan_s']:.3f}s | "
                     f"{m['build_and_migrate_s']:.2f}s | {m['commit_s']:.3f}s | {m['total_s']:.2f}s | "
                     f"{m['resume_s']:.2f}s | {sem} |")
    return "\n".join(lines)
