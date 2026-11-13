# Recovery protocol

## Generations and transactions

The coordinator owns a monotonically increasing *generation*. Every reconfiguration is a
transaction for generation `g`:

```
RECONFIG_NOTICE(g)  ->  ABORTED(g) from every live worker  ->  planner
  ->  PREPARE(g, world, plan, schedule, recovery)  ->  READY(g) from every member  ->  COMMIT(g)
```

* Messages carry `g`; anything for an older generation is ignored (fencing of stale actors).
* A failure detected while a transaction is open restarts it with `g + 1` in *immediate* mode.
* Workers that time out on `ABORTED` or `READY` are declared failed and killed by their agent.
  Ranks that report `build_failed` point at the peers that never became ready.
* Planned reconfigurations (fail-slow mitigation, rejoin, grow) use a non-immediate notice:
  workers finish the current iteration first.

## Iteration transactions and roll-forward

The optimizer step is the commit point of an iteration. Each worker snapshots RNG state, the
data cursor and the buffers an iteration mutates (FP8 amax history, MoE router token counts /
expert bias) at `begin`. On abort, a worker that had not reduced its gradients rolls all of them
back. A worker that had (`REDUCED`) keeps them: if the planner's roll-forward decision lists it
in `local_step`, it applies the optimizer step locally (Adam is deterministic) and becomes a
state source at the new iteration; otherwise it discards the iteration *before* serving as a
migration source, so no source ever mixes post-iteration buffers with pre-iteration parameters.
(`tests/dist/test_resilient_hetero.py::test_fp8_amax_history_migrates_and_rolls_back` kills a
worker after the amax update; a mutation that removes the rollback makes it fail.)

## State migration

Tensors have stable keys (`layers.<global layer>.<param>`, embeddings, experts) with global
shape and TP dimension. From the old and new plans every rank computes the same
`TransferPlan`: reused local slices, slices to receive (cut at holder boundaries, closest and
least-loaded source, optionally striped), and lost slices (fallback to host snapshots or the
resharding checkpoint). Parameters, Adam moments and buffers follow the same routing; the old
state is kept until `COMMIT`.

## Membership changes during a transaction

* A worker that registers while the abort phase is open receives the open notice immediately
  and joins the transaction; one that registers later is left to a follow-up join transaction.
* An agent that reconnects (e.g. after leader failover) is re-sent the open notice for its
  workers that have not acknowledged it.

## Coordinator high availability

Coordinators compete for a leader key with a lease (`InMemoryKV` for tests, etcd in
production). The leader persists generation, committed generation, world, payload and
membership (including pids and health) after every step. All its writes are *fenced*:
`put_if(state, leader_key == me)` (an etcd transaction), so a deposed leader that wakes up from
a pause cannot overwrite its successor's state; on a failed fenced write it steps down. A new
leader restores the state, gives every node a reconnect grace of two lease periods, and if the
previous leader died mid-transaction (`generation > committed_generation`) it first waits for
the agents to reconnect and then restarts the transaction in immediate mode.

## Detection

| signal | mechanism | default |
|---|---|---|
| process exit / crash | agent polls the child, connection loss | immediate |
| hang (process alive) | worker heartbeats stop -> agent; progress counters stall while peers wait in collectives -> coordinator | 3 s / 60 s |
| node loss | node lease (agent heartbeats) | 3 s |
| hardware | NVML fatal XIDs / ECC, all IB ports down | probe interval |
| fail-slow | per-group residual BOCD on workload-normalised timings, localisation, ski-rental decision priced by the planner | 1-3 iterations |

The hang timeout must exceed the communication timeout: a peer blocked on a frozen rank only
unblocks when its own operation times out (NCCL abort helps on GPUs; on the gloo CPU harness
operations cannot be interrupted at all).
