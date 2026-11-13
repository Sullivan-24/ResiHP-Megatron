# Megatron-LM patch set

Applied by `scripts/setup_megatron.sh` on top of the commit in `MEGATRON_PIN`, in file
order. Each patch is self-contained, keeps default behaviour unchanged, and is written to
be proposed upstream on its own. `python -m resihp.compat.megatron_contract third_party/Megatron-LM`
checks every symbol ResiHP relies on (base + patched contract) after a rebase.

| patch | upstream-PR title | why ResiHP needs it |
|---|---|---|
| 0001 | Explicit per-stage layer range in `TransformerConfig` | heterogeneous plans repartition layers unevenly across stages |
| 0002 | `destroy_model_parallel(destroy_groups=...)` | release NCCL sub-communicators when a generation ends |
| 0003 | Eager NCCL initialisation (`device_id`) | bounded, failure-detectable communicator creation |
| 0004 | Explicit previous/next peers in `P2PCommunicator` | pipeline neighbours are no longer `rank +/- 1` |
| 0005 | `register_forward_backward_func` | run an external pipeline schedule through the stock training loop |
| 0006 | `external_wgrad_schedule` for `delay_wgrad_compute` | zero-bubble W instructions on dense models |

## 0001 — Explicit layer range per pipeline stage

**Motivation.** `get_num_layers_to_build` / `get_transformer_layer_offset` derive a stage's
layers from a uniform split (plus first/last-stage adjustments). Failure-adapted plans give
stages different layer counts that are only known at run time.

**Change.** New `TransformerConfig.resihp_layer_range: Optional[Tuple[int, int]] = None`. When
set, the layer count and offset come from it; otherwise the existing logic runs unchanged.

## 0002 — Let `destroy_model_parallel` destroy the groups it created

**Motivation.** Today it only resets the module-level handles; the NCCL communicators stay
alive until process exit. A process that survives many reconfigurations accumulates
communicators (and their memory) from every generation.

**Change.** `destroy_model_parallel(destroy_groups: bool = False)`; with `True`, every group
it created is passed to `torch.distributed.destroy_process_group`. Default unchanged.

## 0003 — Eager NCCL initialisation

**Motivation.** With lazy init, a dead peer is discovered at the first collective, deep in
the training step. Passing `device_id` to `init_process_group` creates the communicator
eagerly, so a failure surfaces at a well-defined, timeout-guarded point and enables
`ncclCommSplit` for sub-groups (no extra bootstrap).

**Change.** `MEGATRON_EAGER_NCCL_INIT=1` makes `_initialize_distributed` pass
`device_id=torch.device("cuda", local_rank)`. Default unchanged.

## 0004 — Explicit peers in `P2PCommunicator`

**Motivation.** The communicator computes neighbours from the pipeline group's rank order.
With heterogeneous TP per stage and stage-level micro-batch migration, a stage may talk to
a different replica's next stage.

**Change.** Optional `prev_rank` / `next_rank` constructor arguments; when omitted, the
existing computation is used.

## 0005 — Pluggable forward/backward schedule

**Motivation.** Alternative pipeline schedules currently require forking `train_step` or
monkey-patching `get_forward_backward_func`.

**Change.** `register_forward_backward_func(func)` installs a function returned by
`get_forward_backward_func` (`None` restores the default). ResiHP registers its instruction
interpreter per generation; the rest of the training loop (optimizer, logging,
checkpointing) is untouched.

## 0006 — Zero-bubble weight gradients without MoE overlap

**Motivation.** `TransformerConfig.__post_init__` asserts
`overlap_moe_expert_parallel_comm` whenever `delay_wgrad_compute` is set, because the only
in-tree consumer of deferred weight gradients is the MoE A2A-overlap schedule. An external
schedule that issues W itself (zero-bubble pipelines) is rejected for dense models although
the Transformer Engine linear layers already implement `backward_dw()`.

**Change.** New `ModelParallelConfig.external_wgrad_schedule: bool = False`; the assertion
becomes `overlap_moe_expert_parallel_comm or external_wgrad_schedule`. Default unchanged.

---

## Upstream PyTorch issue (draft)

**`new_group(use_local_synchronization=True)` gives overlapping groups rank-dependent names
and deadlocks** (observed on torch 2.15).

`ProcessGroup` names are salted with the per-process counter `_world.group_count`. With
`use_local_synchronization=True`, a rank that is *not* a member returns from `new_group`
before the counter is incremented, while members increment it. After creating a group that
excludes some ranks, the next group's name differs between ranks that were and were not in
the previous one, so members of an overlapping group look for different store keys and
block forever.

*Repro.* 4 ranks; create `[0, 1]` then `[1, 2]`, both with `use_local_synchronization=True`:
rank 1 has seen two groups, rank 2 only one, and the second group hangs.

*Workaround used by ResiHP* (`resihp/comm/comm_manager.py`): create groups from
`split_group` in disjoint layers (default), or have every rank call `new_group` for every
group in a canonical order and bump the counter on non-members.
