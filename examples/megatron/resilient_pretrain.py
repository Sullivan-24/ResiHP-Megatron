"""Resilient Megatron GPT pre-training driven by ResiHP.

Status: written against the pinned Megatron commit (see MEGATRON_PIN) and checked by
``python -m resihp.compat.megatron_contract``; it has not been executed on GPUs yet.

    bash scripts/setup_megatron.sh && export PYTHONPATH=$PWD/third_party/Megatron-LM
    resihp-run local --nnodes 1 --nproc-per-node 8 --planner my_planners:llama3_8b -- \
        python examples/megatron/resilient_pretrain.py <usual pretrain_gpt.py arguments> \
        --use-distributed-optimizer --calculate-per-token-loss --bf16

where ``my_planners.llama3_8b = resihp.plan.coordinator_adapter.make_hetero_planner(PlannerConfig(...))``
describes the model, the nominal TP/PP/DP layout and the cost profile.

Differences from ``pretrain_gpt.py``: the parallel layout comes from the coordinator's
HeteroParallelPlan (not from --tensor/--pipeline-model-parallel-size), process groups are
rebuilt per generation through ResilientCommManager, and the data is read by global
micro-batch index so migrated micro-batches load the same samples on any replica.
"""

from __future__ import annotations

import os
import re

import torch


def stable_key(name: str, layout: dict) -> str:
    """Megatron parameter name -> plan-independent key (global layer index)."""
    m = re.match(r"(?:module\.)*decoder\.layers\.(\d+)\.(.+)", name)
    if m:
        return f"layers.{layout['resihp_layer_range'][0] + int(m.group(1))}.{m.group(2)}"
    return re.sub(r"^(?:module\.)+", "", name)


def gpt_tensor_specs(args):
    """Global shapes and TP partition dims of a Transformer Engine GPT (SwiGLU, GQA, no bias)."""
    from resihp.state.layer_registry import TensorSpec

    h, f = args.hidden_size, args.ffn_hidden_size
    kv = args.num_query_groups or args.num_attention_heads
    qkv = (args.num_attention_heads + 2 * kv) * args.kv_channels
    vocab = args.padded_vocab_size
    out = {}

    def add(key, shape, dim):
        out[key] = TensorSpec(key, shape, dim)

    for layer in range(args.num_layers):
        p = f"layers.{layer}."
        add(p + "self_attention.linear_qkv.layer_norm_weight", (h,), None)
        add(p + "self_attention.linear_qkv.weight", (qkv, h), 0)
        add(p + "self_attention.linear_proj.weight", (h, args.num_attention_heads * args.kv_channels), 1)
        add(p + "mlp.linear_fc1.layer_norm_weight", (h,), None)
        add(p + "mlp.linear_fc1.weight", (2 * f if args.swiglu else f, h), 0)
        add(p + "mlp.linear_fc2.weight", (h, f), 1)
    add("embedding.word_embeddings.weight", (vocab, h), 0)
    add("decoder.final_layernorm.weight", (h,), None)
    if not args.untie_embeddings_and_output_weights:
        return out
    add("output_layer.weight", (vocab, h), 0)
    return out


def layer_of(num_layers: int):
    def f(key: str) -> int:
        if key.startswith("layers."):
            return int(key.split(".")[1])
        return 0 if key.startswith("embedding.") else num_layers - 1

    return f


def main() -> None:
    from gpt_builders import gpt_builder
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig
    from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
    from megatron.training.global_vars import set_args
    from pretrain_gpt import forward_step, train_valid_test_datasets_provider

    from resihp.comm import ResilientCommManager
    from resihp.control.worker_client import WorkerClient
    from resihp.state.rng_data import IndexAddressableSampler
    from resihp.train.megatron_binding import MegatronBinding, MegatronJob
    from resihp.train.resilient_loop import ResilientTrainer

    args = parse_and_validate_args()
    set_args(args)
    torch.cuda.set_device(int(os.environ.get("RESIHP_LOCAL_RANK", "0")))
    train_ds, _, _ = train_valid_test_datasets_provider([args.train_iters * args.global_batch_size, 0, 0])
    sampler = IndexAddressableSampler(len(train_ds), args.global_batch_size, args.micro_batch_size, seed=args.seed)

    def batch_for(iteration: int, mb: int):
        samples = [train_ds[i] for i in sampler.microbatch_indices(iteration, mb)]
        batch = {k: torch.stack([torch.as_tensor(s[k]) for s in samples]).cuda(non_blocking=True) for k in samples[0]}
        return iter([batch])

    def build_config(layout: dict):
        config = core_transformer_config_from_args(args)
        config.tensor_model_parallel_size = layout["tp_size"]
        config.context_parallel_size = layout["cp_size"]
        config.resihp_layer_range = tuple(layout["resihp_layer_range"])
        return config

    def model_provider(config, pre_process, post_process, pg_collection):
        return gpt_builder(args, pre_process, post_process, config=config, pg_collection=pg_collection).cuda()

    job = MegatronJob(
        build_config=build_config, model_provider=model_provider, forward_step_func=forward_step,
        batch_for=batch_for,
        ddp_config=DistributedDataParallelConfig(use_distributed_optimizer=args.use_distributed_optimizer,
                                                 overlap_grad_reduce=False),
        optimizer_config=OptimizerConfig(lr=args.lr, min_lr=args.min_lr, weight_decay=args.weight_decay,
                                         adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2, bf16=args.bf16,
                                         use_distributed_optimizer=args.use_distributed_optimizer,
                                         clip_grad=args.clip_grad),
        state_key=stable_key, tensor_specs=gpt_tensor_specs(args), layer_of=layer_of(args.num_layers),
        partition_sizes=tuple(sorted({args.hidden_size, args.ffn_hidden_size, args.padded_vocab_size})),
        hidden=args.hidden_size, seq_length=args.seq_length, micro_batch_size=args.micro_batch_size,
        tp_base=args.tensor_model_parallel_size, seed=args.seed)
    client = WorkerClient().connect()
    mgr = ResilientCommManager("nccl", device=f"cuda:{torch.cuda.current_device()}", timeout_s=600)
    ResilientTrainer(client, mgr, MegatronBinding(job, mgr, client)).run(args.train_iters)
    mgr.destroy()
    client.close()


if __name__ == "__main__":
    main()
