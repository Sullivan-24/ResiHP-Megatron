# Megatron-LM integration

ResiHP is a plugin: it extends Megatron-Core through a small, upstreamable patch set
(`patches/megatron/`, see its README) on top of a pinned `dev` commit (`MEGATRON_PIN`).

## Pieces

| ResiHP | Megatron API it drives |
|---|---|
| `megatron_group_specs`, `build_pg_collection` | `ProcessGroupCollection` built per generation from the plan (TP per CP slice, CP, TP x CP, DP, DP x CP, singleton groups) instead of the global `parallel_state` |
| `stage_layout` + patch 0001 | `TransformerConfig.resihp_layer_range`: a chunk with the plan's (uneven) layer range |
| `wrap_model` | native `DistributedDataParallel` when a stage is uniform across replicas; `NonuniformTPDistributedDataParallel` when it fits Nonuniform TP's model (one reduced degree); otherwise DDP + ResiHP's shard-aligned gradient reduction (`HeteroGradSync`) |
| `make_forward_backward_func` + patch 0005 | the instruction interpreter registered as the forward/backward function, calling Megatron's `forward_step` / `backward_step` per instruction |
| `MegatronWeightGradStore` + patch 0006 | zero-bubble W instructions via `backward_dw()` of Transformer Engine linear layers (`delay_wgrad_compute` + `external_wgrad_schedule`) |
| `export_tp_shard_state` / `import_tp_shard_state` | `DistributedOptimizer` main params and moments in model space for migration |
| `GenerationHooks` (`TEUserBufferHook`, `CudaGraphHook`) | `te.module.base.initialize_ub/destroy_ub`; `TECudaGraphHelper.create_cudagraphs/delete_cuda_graphs` or `delete_cuda_graphs()` |
| `MegatronBinding`, `examples/megatron/resilient_pretrain.py` | model provider (`gpt_builder`), `pretrain_gpt.forward_step`, `get_megatron_optimizer`, GPT datasets read by global micro-batch index |

## Coverage

* **Tensor/sequence parallelism.** Any TP degree the model's heads, KV groups and FFN divide into
  (not only powers of two), different per (replica, stage); activations crossing a TP-degree
  boundary are re-sharded by interval overlap (SP) or scatter + all-gather (no SP).
* **Pipeline parallelism.** Uneven layer partitions, stage-level micro-batch migration,
  1F1B, interleaved (VPP; Megatron's static order or the list scheduler, whichever the
  simulator predicts faster) and zero-bubble (B/W split) schedules, all validated for NCCL
  safety (per-direction FIFO on shared channels, deadlock freedom).
* **Context parallelism.** Uniform CP degree; activations stay within a CP chunk, gradients are
  reduced over DP x CP; TP shrinks inside each CP slice.
* **MoE / expert parallelism.** Expert re-placement after EP-rank loss with replicas of hot
  experts (EPLB-style counts, packing by load / speed, incumbents kept), expert-DP groups
  derived from the placement, expert parameters + Adam state migrated, aux-loss-free router
  bias carried as a buffer. Executed by ResiHP's reference EP runtime; mapping onto Megatron's
  token dispatchers with uneven local expert counts is future work.
* **FP8.** Amax history / scales are buffers: they migrate with their layer and roll back with
  aborted iterations.

## Verification status

| level | what is checked | where |
|---|---|---|
| semantics | every parallelism feature above, on the reference model, bit-for-bit (float64) against single-process training, including across live reconfigurations | `tests/dist/*` (gloo, CPU) |
| API contract | 45 Megatron symbols (+ patched ones) with their argument / field names | `resihp/compat/megatron_contract.py`, CI |
| patches | apply cleanly on a fresh checkout of the pin, in order | `scripts/setup_megatron.sh` |
| GPU | NCCL shrink/split/abort fast path | `tests/gpu/test_nccl_shrink.py` (self-hosted runner) |
| **not yet** | Megatron GPT end-to-end on GPUs, MFU parity with stock Megatron, convergence at scale | `examples/megatron/resilient_pretrain.py`, `docs/evaluation.md` |

Megatron Core cannot be imported without CUDA/Triton, which is why the contract is checked
statically and the Megatron path has not run in this repository's CI yet.
