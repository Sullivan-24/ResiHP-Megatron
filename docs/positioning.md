# Positioning and paper revision

## What ResiHP claims (and can back with this artifact)

1. **Online, failure- and straggler-aware reshaping of all three parallel dimensions,
   planned jointly.** TP subgroup choice (any degree the model divides into, CP-aware),
   layer repartition across stages and per-stage micro-batch migration across replicas are
   decided together (including keep-or-evict of weakened groups), validated as an
   NCCL-safe instruction schedule, and executed without restarting survivors.
2. **Exact training semantics across reconfigurations.** Global batch, per-token loss
   normalisation, sampler order, RNG streams and stateful buffers are preserved; the
   artifact checks bit-for-bit equality with failure-free training (float64 reference)
   for heterogeneous TP x uneven PP x migration x SP x CP x VPP x ZB, MoE re-placement and
   FP8-style buffers, including a kill in the middle of an iteration.
3. **Live state migration at slice granularity** for parameters, optimizer moments and
   buffers, covering TP resharding, PP layer moves, replica recovery, rejoin/grow and
   expert re-placement with one mechanism; old state kept until commit.
4. **Workload-aware fail-slow detection** (residual BOCD on predicted vs observed chunk
   times) that does not alarm on long-document bursts, priced by the same planner and
   simulator that mitigate it (ski-rental on predicted gain vs reconfiguration cost).
5. **A transactional control plane** (generations, fencing, roll-forward, late joiners,
   leader failover with fenced writes) that never kills healthy workers.

## What it should *not* claim

* "First heterogeneous / non-uniform TP": Nonuniform TP (2025, now in Megatron-Core as
  `nonuniform_tp.py`) and Malleus (2025) already run reduced or uneven TP. ResiHP's TP part
  is better described as *online, failure- and straggler-aware TP reshaping that is
  SP/CP/DistOpt compatible and jointly planned with PP and DP*; it reuses Nonuniform TP's
  DDP when a stage fits that model.
* Large end-to-end speedups from fail-stop handling alone when spare nodes are plentiful:
  the cluster simulation shows the gap to online replica-level systems is then about one
  ETTR point (MTTR and rollback); the large gaps appear without spares and with fail-slows.
* Anything about GPU-scale MTTR, MFU or convergence before those runs exist
  (`docs/evaluation.md`).

## Related work to cite and how ResiHP differs

| work | what it does | difference |
|---|---|---|
| Oobleck (SOSP'23) | pipeline templates, re-instantiation from replicas | node-granular, no TP reshaping, pipelines restart; ResiHP keeps survivors and works at GPU granularity |
| ReCycle (SOSP'24) | reroutes a failed stage's micro-batches to peer replicas using bubbles | no TP shrink (a single dead GPU removes its whole TP group), no repartition, no fail-slow |
| Malleus (2025) | straggler-resilient hybrid parallelism via malleable data/model partitioning | targets stragglers and replans with restarts between plans; ResiHP handles fail-stop + fail-slow online with exact semantics |
| Nonuniform TP (2025) | keeps degraded TP groups training at reduced degree | one reduced degree per stage, no PP repartition or DP migration; ResiHP generalises the layout and plans it jointly |
| ElasWave (2025) | elastic-native hybrid-parallel training | evaluate side by side once an artifact is available |
| torchft (2025) | replica-group fault tolerance with quorum, per-step healing | replica granularity: a single GPU failure idles a full model-parallel replica; global batch varies |
| NVRx / checkpoint restart | in-job restart, local checkpoints, straggler detection | restarts every rank, rolls back to the last checkpoint |
| Greyhound (ATC'25) | BOCD-based fail-slow detection and mitigation | raw-series detection alarms on data-dependent load changes (35 false alarms on our healthy traces vs 0) |
| Gemini (SOSP'23), ByteRobust (2025) | in-memory checkpoints, robust large-scale training stacks | ResiHP's T1 tier uses the same idea; complementary |
| Lazarus (2024), DeepSeek EPLB | adaptive expert placement / replication | ResiHP applies placement on failure with migration of expert state and exact routing semantics |
| Zero Bubble PP (ICLR'24) | B/W split | ResiHP schedules W in failure-adapted heterogeneous pipelines |
| TrainMover, FlashRecovery (2025) | fast recovery runtimes | compare MTTR breakdowns on the same GPU testbed |

## Suggested revisions to the paper

* Retitle the TP contribution as above and add Nonuniform TP / Malleus to the TP background.
* Replace "drop the failed replica" baselines with batch-preserving ones (ResiHP keeps the
  batch; `torchft` shrinks it) and report both.
* Report the micro-batch-granularity limit honestly: moving whole micro-batches costs up to
  `1 / (M / D)` on receiving replicas; layer repartition recovers most of it (planner
  ablation), sample-level migration is future work.
* Separate fail-stop-only and fail-stop + fail-slow results; fail-slow rates are not public
  for Llama 3 and must be presented as a sensitivity study.
