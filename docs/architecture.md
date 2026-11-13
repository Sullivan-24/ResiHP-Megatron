# Architecture

ResiHP-Megatron keeps a pre-training job running through fail-stop failures (dead GPUs,
nodes, links, hung processes) and fail-slow GPUs by re-planning the *hybrid parallel*
layout on the fly — TP degree per group, layer partition across stages, and which replica
executes which micro-batch at which stage — and migrating model and optimizer state into
the new layout, while surviving processes keep running and the global batch, loss
normalisation and RNG streams stay exactly those of the failure-free run.

```
              +------------------- control plane (CPU only) --------------------+
              | Coordinator (leader, etcd lease) <-> NodeAgent per node        |
              |   generations, fencing, transactions, planner, detector service |
              +----------------^-------------------------------^----------------+
                 heartbeats,   |  NOTICE / PREPARE / COMMIT      | timing reports
                 progress      |                                 |
   +-------------------------- worker process (one per GPU) ---------------------------+
   | WorkerClient -> ResilientTrainer: begin txn -> run iteration -> commit (opt step) |
   |   on notice/failure: abort txn -> rebuild comms -> migrate state -> READY         |
   |   ResilientCommManager (shrink/split, abortable waits)   GenerationHooks (UB, CG)  |
   |   Runtime binding: reference model (CPU) | Megatron-Core (GPU)                      |
   |   IR interpreter: F/B/W + heterogeneous P2P (TP/SP/CP aware), HeteroGradSync       |
   +-----------------------------------------------------------------------------------+
```

## Packages

| package | responsibility |
|---|---|
| `resihp.control` | `Coordinator` (generation-numbered reconfiguration transactions, leases, fencing, hang detection, etcd-backed HA with fenced writes), `NodeAgent` (spawns/monitors/kills workers, never kills healthy ones), `WorkerClient`, launcher `resihp-run` |
| `resihp.comm` | `ResilientCommManager`: per-generation process groups via `shrink_group(SHRINK_ABORT)` + `split_group` (NCCL >= 2.27) or a fresh `PrefixStore` init; canonical group order; validation all-reduce; abortable waits |
| `resihp.plan` | `HeteroParallelPlan` (explicit groups, TP x CP slices, virtual stages for VPP, per-stage micro-batch executors), IR + validator, list-scheduling simulator shared with the detector, progressive planner (TP subgroup choice, layer DP, progress-aware migration, joint keep-or-evict), plan cache, reconfiguration cost model, expert placement (MoE) |
| `resihp.runtime` | IR interpreter, heterogeneous P2P routing, shard-aligned gradient reduction, zero-bubble W stores, reference model (TP/SP/CP/VPP/FP8-style), MoE expert-parallel runtime, Megatron adapters, per-generation hooks |
| `resihp.state` | iteration transactions (RNG, data cursor, buffers), index-addressable sampler, slice-level transfer planning and execution for parameters + Adam moments + buffers, redundancy tiers (T0 twins, T1 host snapshots + buddy copies, T2 resharding checkpoints), DistOpt export/import |
| `resihp.detect` | chunk timing, NCCL profiler plugin (v5 ABI), gated RLS workload predictor, residual BOCD, localization, Greyhound-style validation probes, ski-rental policy priced by the planner (what-if) |
| `resihp.train` | `ResilientTrainer` and bindings (replica DP, heterogeneous runtime, Megatron) |
| `resihp.sim`, `resihp.bench` | failure traces, cluster simulator with baselines, ResiBench |

## One reconfiguration, end to end

1. **Detect.** A worker dies (agent sees the exit or a lost connection), stops making
   progress (heartbeat or progress-counter hang detection), its node lease expires, a
   health probe reports a fatal XID, or the detector service confirms a slowdown.
2. **Fence + notice.** The coordinator kills the culprit through its agent, bumps the
   generation and sends `RECONFIG_NOTICE(g)`. Survivors abort their communicators (NCCL
   abort / abortable waits), roll back the in-flight iteration (RNG, data cursor, mutated
   buffers) unless its gradients were already reduced, and answer `ABORTED(g)` with their
   recovery info.
3. **Plan.** The planner sees survivors, speeds and the previous plan. A plan-cache hit
   returns a precomputed plan; otherwise it chooses TP subgroups (Eq. 4 extended to
   non-power-of-two degrees and CP), repartitions layers (min-max DP on aggregated stage
   capacity), migrates micro-batches between replicas per stage (Algorithm 1 + pipeline
   moves) and decides keep-or-evict for weakened groups jointly; the result is validated
   as an IR schedule (completeness, FIFO per channel, deadlock freedom).
4. **Prepare.** Every worker builds generation `g`'s communicators, derives the identical
   slice-level transfer plan from the old and new plans, streams parameters, Adam moments
   and buffers from the closest valid holders (or rolls forward locally if it holds the
   reduced gradients), rebuilds its model chunk(s) and per-generation resources, and
   reports `READY(g)`.
5. **Commit.** `COMMIT(g)`; the aborted iteration is replayed with the same global batch.
   A failure at any point restarts the transaction with `g + 1`; old state is freed only
   after commit.
