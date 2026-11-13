# ResiHP-Megatron

Resilient LLM pre-training on Megatron-Core with **dynamic hybrid parallelism**: when GPUs,
nodes or links fail — or quietly become slow — ResiHP re-plans tensor, pipeline and data
parallelism *online* (TP degree per group, layer partition, which replica runs which
micro-batch at which stage), migrates parameters, optimizer state and buffers into the new
layout, and resumes. Surviving processes never restart, and the training trajectory is the
one the failure-free job would have produced: same global batch, same per-token loss
normalisation, same data order and RNG streams.

Implementation of *ResiHP: Taming LLM Training Failures with Dynamic Hybrid Parallelism*
(arXiv 2605.06374), rebuilt as a Megatron-Core plugin with a full control plane, live state
migration and workload-aware fail-slow detection.

## Status

| area | state |
|---|---|
| control plane, communicator rebuild, transactions, state migration, planner, detector | implemented, tested with real processes on CPU (gloo) |
| TP (any degree) / SP / uneven PP / VPP / zero-bubble / CP / MoE-EP / FP8-style buffers | implemented; every combination tested **bit-for-bit** against single-process training across live reconfigurations (float64 reference model) |
| coordinator HA (etcd lease, fenced writes, mid-transaction failover), rejoin/grow, plan cache | implemented, tested |
| Megatron-Core adapters (+ 6 upstreamable patches) | implemented; API contract (45 symbols) and patch application checked statically; **not yet run on GPUs** |
| GPU-scale evaluation (>= 1K GPUs, MFU parity, >= 100B-token convergence) | not done here; plan and tooling in [docs/evaluation.md](docs/evaluation.md) |

## Highlights

* **Fail-stop without restarts.** Generation-numbered reconfiguration transactions, fencing,
  roll-forward of iterations whose gradients were already reduced, and `shrink_group` /
  `split_group` communicator rebuilds (fresh-init fallback). A `kill -9` in the 8-process
  heterogeneous-runtime chaos benchmark is detected, re-planned, migrated and committed in
  **~70 ms** (on gloo, ring collectives that were in flight still wait for the communication
  timeout before survivors resume; NCCL abort removes that on GPUs).
* **Joint TP x PP x DP re-planning.** TP subgroups of any degree the model divides into,
  min-max layer repartition, progress-aware micro-batch migration, keep-or-evict decided
  jointly, plans validated as NCCL-safe instruction schedules. When both replicas of a stage
  lose a GPU, exclusion-based designs must stop; ResiHP keeps training at ~0.91 throughput.
* **Slice-level live migration** of parameters + Adam moments + buffers for TP resharding,
  PP layer moves, replica recovery, rejoin/grow and expert re-placement; old state kept until
  commit; warm handoff for planned moves.
* **Fail-slow detection that ignores data-dependent load** (residual BOCD on predicted vs
  observed per-group timings): 0 false alarms where raw-series detection (Greyhound-style)
  raised 35, 20/20 slowdowns detected and localised; mitigation priced by the planner.
* **Coverage**: VPP, zero-bubble B/W split, context parallelism, MoE expert parallelism with
  replica-aware expert placement, FP8 amax history and router expert bias (migrated and rolled
  back), per-generation rebuild of TE userbuffers and CUDA graphs.

## Results

Numbers produced by `resihp-bench` in this repository (CPU and simulation; see
[docs/evaluation.md](docs/evaluation.md) for methodology and threats to validity).

**Chaos on real processes** (8 workers, medians of 3 trials, every trial bit-identical to the
reference afterwards; `results/chaos/chaos.md`):

| fault | detect | plan | build + migrate | fault -> commit |
|---|---|---|---|---|
| `kill -9` | 0.01 s | 0.006 s | 0.03 s | 0.07 s |
| SIGSTOP hang | 2.98 s (heartbeat timeout) | 0.009 s | 0.04 s | 3.06 s |
| 2x fail-slow | 0.18 s | 0.005 s | 0.05 s | 0.31 s |

**Cluster scale** (trace-driven with Llama 3's failure statistics, 405B-class job TP8 x PP16,
30 days; policy models of published systems; `results/sim/cluster.md`). ETTR, fail-stop only:

| GPUs | ResiHP | torchft | ReCycle | Oobleck | NVRx |
|---|---|---|---|---|---|
| 16K, no spares | **97.3%** | 92.5% | 92.6% | 92.5% | 86.4% |
| 16K, 2% spares | **99.4%** | 98.4% | 98.4% | 97.7% | 91.8% |
| 64K, 2% spares | **97.6%** | 93.8% | 93.8% | 91.2% | 70.8% |

With (assumed) fail-slow rates added, ResiHP stays at 98.0% at 16K (2% spares) while
replica-level systems drop to ~78% and Greyhound to 88.7%; see the sensitivity study before
quoting these. Planner ablation: `results/planner/ablation.md`; detection:
`results/detection/detection.md`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.10"   # or a CUDA build
pip install -e ".[dev]"

pytest -q -m "not gpu"                 # unit + multi-process gloo tests (~10 min)

# an 8-worker job on 2 simulated nodes; kill one worker at iteration 3 and watch it recover
# (--timeout bounds how long gloo, which cannot abort an in-flight op, waits on the dead peer)
RESIHP_FAULTS="kill@iter=3:target=n0g3:phase=forward" \
  resihp-run local --nnodes 2 --nproc-per-node 4 --planner resihp.train.demo_planners:dp_8mb -- \
  python examples/cpu_resilient_demo.py --iterations 20 --timeout 10 --out /tmp/resihp-demo

resihp-bench chaos        # MTTR breakdown on real processes
resihp-bench all --quick  # every benchmark, small configuration
```

Megatron (GPU):

```bash
bash scripts/setup_megatron.sh                       # pinned commit + patches/megatron/*.patch
export PYTHONPATH=$PWD/third_party/Megatron-LM
python -m resihp.compat.megatron_contract third_party/Megatron-LM
pytest -q -m gpu                                     # NCCL shrink/split/abort fast path
# examples/megatron/resilient_pretrain.py: experimental, see docs/megatron_integration.md
```

## Layout

```
resihp/control   coordinator, node agent, worker client, KV/etcd, launcher (resihp-run)
resihp/comm      per-generation process groups (shrink/split, fresh init), heterogeneous P2P
resihp/plan      HeteroParallelPlan, IR + validator, simulator, planner, plan cache, cost model, expert placement
resihp/runtime   interpreter, reference model, MoE runtime, grad sync, ZB stores, Megatron adapters, generation hooks
resihp/state     transactions, sampler, transfer planning/engine, redundancy tiers, checkpoints, DistOpt bridge
resihp/detect    timing, NCCL profiler plugin, RLS, BOCD, validation probes, policy, detector service
resihp/train     resilient loop and bindings (DP, heterogeneous, Megatron)
resihp/sim       failure traces, cluster simulator, calibration, experiments
resihp/bench     ResiBench (resihp-bench)
patches/megatron upstreamable Megatron-LM patches (see its README)
docs/            architecture, recovery protocol, Megatron integration, evaluation, positioning
```

## Documentation

* [Architecture](docs/architecture.md) · [Recovery protocol](docs/recovery_protocol.md) ·
  [Megatron integration](docs/megatron_integration.md) · [Evaluation](docs/evaluation.md) ·
  [Positioning and paper revision](docs/positioning.md) · [Patches](patches/megatron/README.md)

## Known limitations

* The Megatron path is statically checked only; GPU validation, MFU parity and convergence
  runs are outstanding.
* Micro-batches move whole; receiving replicas pay up to `1 / (micro-batches per replica)`
  extra work, mostly recovered by layer repartition (sample-level migration is future work).
* With DistributedOptimizer, migration sources are model-space host snapshots taken every `k`
  steps (rollback <= `k`); zero-rollback live migration needs a non-sharded optimizer or the
  twin-instance (T0) export path, which is designed but not implemented.
* MoE expert re-placement runs in ResiHP's reference EP runtime; Megatron's token dispatchers
  assume equal local expert counts.
* GPU-level localisation inside a TP > 1 group relies on validation probes; the CPU harness
  bounds absolute MTTR by gloo (operations cannot be aborted, so a frozen peer releases its
  partners only at the communication timeout).

## License

Apache-2.0.
