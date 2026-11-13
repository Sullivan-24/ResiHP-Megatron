"""Static API contract between ResiHP's Megatron adapters and the pinned Megatron-LM commit.

Megatron Core cannot be imported without CUDA/Triton, so the contract is checked by
parsing the source with ``ast``. Run it after bumping ``MEGATRON_PIN``::

    python -m resihp.compat.megatron_contract third_party/Megatron-LM
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Requirement:
    path: str
    symbol: str
    args: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    used_by: str = ""


CONTRACT: tuple[Requirement, ...] = (
    Requirement("megatron/core/process_groups_config.py", "ProcessGroupCollection",
                fields=("tp", "pp", "mp", "embd", "cp", "tp_cp", "ep", "expt_tp", "dp", "dp_cp", "expt_dp",
                        "inter_dist_opt"),
                used_by="resihp.runtime.megatron_runtime.build_pg_collection"),
    Requirement("megatron/core/pipeline_parallel/p2p_communication.py", "P2PCommunicator.__init__",
                args=("pp_group", "config"), used_by="resihp.runtime.megatron_runtime"),
    Requirement("megatron/core/pipeline_parallel/schedules.py", "get_forward_backward_func",
                args=("pp_size", "vp_size"), used_by="patches/megatron/0005"),
    Requirement("megatron/core/pipeline_parallel/schedules.py", "forward_step",
                args=("forward_step_func", "data_iterator", "model", "num_microbatches", "input_tensor",
                      "forward_data_store", "config", "cp_group_size"),
                used_by="resihp.runtime.megatron_runtime.MegatronStageExecutor"),
    Requirement("megatron/core/pipeline_parallel/schedules.py", "backward_step",
                args=("input_tensor", "output_tensor", "output_tensor_grad", "config"),
                used_by="resihp.runtime.megatron_runtime.MegatronStageExecutor"),
    Requirement("megatron/core/distributed/nonuniform_tp.py", "NonuniformTPConfig",
                fields=("tp_base", "tp_spares", "num_reduced_tp_dp_ranks", "non_active_ranks_per_dp"),
                used_by="resihp.runtime.megatron_runtime.ntp_config_for_stage"),
    Requirement("megatron/core/distributed/nonuniform_tp.py", "NonuniformTPDistributedDataParallel.__init__",
                args=("config", "ddp_config", "module", "pg_collection", "ntp_config")),
    Requirement("megatron/core/distributed/nonuniform_tp.py", "initialize_nonuniform_tp_process_groups",
                args=("ntp_config",)),
    Requirement("megatron/core/distributed/nonuniform_tp.py", "ntp_init", args=("layer", "ntp_config")),
    Requirement("megatron/core/distributed/nonuniform_tp.py", "ntp_map", args=("module", "ntp_config", "num_shards")),
    Requirement("megatron/core/optimizer/distrib_optimizer.py", "DistributedOptimizer._get_main_param_and_optimizer_states",
                args=("model_param",), used_by="resihp.state.megatron_adapter"),
    Requirement("megatron/core/optimizer/distrib_optimizer.py", "DistributedOptimizer._set_main_param_and_optimizer_states",
                args=("model_param", "tensors"), used_by="resihp.state.megatron_adapter"),
    Requirement("megatron/core/optimizer/distrib_optimizer.py", "DistributedOptimizer._get_model_param_range_map",
                args=("param",), used_by="resihp.state.megatron_adapter"),
    Requirement("megatron/core/resharding/planner.py", "build_local_reshard_plan",
                args=("src_module", "dst_module", "group"), used_by="resihp.state.megatron_adapter"),
    Requirement("megatron/core/resharding/execution.py", "execute_reshard_plan",
                args=("plan", "src_module", "dst_module", "service", "group")),
    Requirement("megatron/core/transformer/transformer_block.py", "get_num_layers_to_build",
                used_by="patches/megatron/0004"),
    Requirement("megatron/core/transformer/transformer_layer.py", "get_transformer_layer_offset",
                used_by="patches/megatron/0004"),
    Requirement("megatron/core/distributed/finalize_model_grads.py", "finalize_model_grads",
                args=("model", "num_tokens", "pg_collection")),
    Requirement("megatron/core/distributed/distributed_data_parallel_config.py", "DistributedDataParallelConfig",
                fields=("num_distributed_optimizer_instances", "overlap_grad_reduce", "use_distributed_optimizer")),
    Requirement("megatron/core/packed_seq_params.py", "PackedSeqParams", fields=("cu_seqlens_q", "qkv_format")),
    Requirement("megatron/core/parallel_state.py", "initialize_model_parallel"),
    Requirement("megatron/core/parallel_state.py", "destroy_model_parallel", used_by="patches/megatron/0002"),
    Requirement("megatron/training/training.py", "train_step", used_by="examples/megatron/resilient_pretrain.py"),
    Requirement("megatron/core/optimizer/__init__.py", "get_megatron_optimizer",
                args=("config", "model_chunks", "pg_collection"), used_by="resihp.train.megatron_binding"),
    Requirement("megatron/core/distributed/distributed_data_parallel.py", "DistributedDataParallel.zero_grad_buffer",
                used_by="resihp.train.megatron_binding"),
    Requirement("gpt_builders.py", "gpt_builder", args=("args", "pre_process", "post_process", "config", "pg_collection"),
                used_by="examples/megatron/resilient_pretrain.py"),
    Requirement("pretrain_gpt.py", "forward_step", args=("data_iterator", "model"),
                used_by="examples/megatron/resilient_pretrain.py"),
    Requirement("pretrain_gpt.py", "get_batch", args=("data_iterator", "pg_collection")),
    Requirement("megatron/training/training.py", "setup_model_and_optimizer", used_by="patches/megatron/0001"),
    Requirement("megatron/training/initialize.py", "_initialize_distributed", used_by="patches/megatron/0003"),
    Requirement("megatron/core/pipeline_parallel/schedules.py", "forward_backward_pipelining_without_interleaving",
                args=("forward_step_func", "data_iterator", "model", "num_microbatches", "seq_length",
                      "micro_batch_size", "forward_only", "pg_collection"),
                used_by="resihp.runtime.megatron_runtime.make_forward_backward_func (signature mirrored)"),
    Requirement("megatron/core/distributed/distributed_data_parallel.py", "DistributedDataParallel.__init__",
                args=("config", "ddp_config", "module", "pg_collection")),
    Requirement("megatron/core/distributed/distributed_data_parallel.py", "DistributedDataParallel.finish_grad_sync"),
    # Zero-bubble W split (resihp.runtime.weight_grad_store.MegatronWeightGradStore).
    Requirement("megatron/core/model_parallel_config.py", "ModelParallelConfig",
                fields=("delay_wgrad_compute", "tp_comm_overlap"), used_by="resihp.runtime.megatron_runtime"),
    Requirement("megatron/core/extensions/transformer_engine.py", "TELinear.backward_dw",
                used_by="resihp.runtime.weight_grad_store.MegatronWeightGradStore"),
    Requirement("megatron/core/extensions/transformer_engine.py", "TELayerNormColumnParallelLinear.backward_dw",
                used_by="resihp.runtime.weight_grad_store.MegatronWeightGradStore"),
    # Per-generation rebuild of communicator-bound resources (resihp.runtime.generation_hooks).
    Requirement("megatron/core/transformer/cuda_graphs.py", "delete_cuda_graphs",
                used_by="resihp.runtime.generation_hooks.CudaGraphHook (local impl)"),
    Requirement("megatron/core/transformer/cuda_graphs.py", "TECudaGraphHelper.graphs_created"),
    Requirement("megatron/core/transformer/cuda_graphs.py", "TECudaGraphHelper.create_cudagraphs"),
    Requirement("megatron/core/transformer/cuda_graphs.py", "TECudaGraphHelper.delete_cuda_graphs"),
    Requirement("megatron/core/transformer/cuda_graphs.py", "TECudaGraphHelper.cuda_graph_set_manual_hooks"),
    Requirement("megatron/core/transformer/transformer_config.py", "TransformerConfig", fields=("cuda_graph_impl",)),
    Requirement("megatron/training/initialize.py", "_initialize_tp_communicators",
                args=("config", "sequence_length", "micro_batch_size", "tp_comm_overlap_cfg"),
                used_by="resihp.runtime.generation_hooks.TEUserBufferHook"),
    # Buffers that migrate with layers and roll back with iterations.
    Requirement("megatron/core/fp8_utils.py", "correct_amax_history_if_needed", args=("model",)),
    Requirement("megatron/core/distributed/finalize_model_grads.py", "_update_router_expert_bias",
                args=("model", "config", "tp_dp_cp_group")),
)

# Symbols introduced by patches/megatron; checked only on a patched checkout.
PATCHED_CONTRACT: tuple[Requirement, ...] = (
    Requirement("megatron/core/transformer/transformer_config.py", "TransformerConfig", fields=("resihp_layer_range",),
                used_by="patches/megatron/0001"),
    Requirement("megatron/core/parallel_state.py", "destroy_model_parallel", args=("destroy_groups",),
                used_by="patches/megatron/0002"),
    Requirement("megatron/core/pipeline_parallel/p2p_communication.py", "P2PCommunicator.__init__",
                args=("prev_rank", "next_rank"), used_by="patches/megatron/0004"),
    Requirement("megatron/core/pipeline_parallel/schedules.py", "register_forward_backward_func", args=("func",),
                used_by="patches/megatron/0005"),
    Requirement("megatron/core/model_parallel_config.py", "ModelParallelConfig", fields=("external_wgrad_schedule",),
                used_by="patches/megatron/0006"),
)


def is_patched(megatron_root: str | Path) -> bool:
    cfg = Path(megatron_root) / "megatron/core/transformer/transformer_config.py"
    return cfg.exists() and "resihp_layer_range" in cfg.read_text()


@lru_cache(maxsize=None)
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _find(body: list[ast.stmt], name: str) -> ast.AST | None:
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _arg_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    a = fn.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    if a.vararg:
        names.add("*" + a.vararg.arg)
    if a.kwarg:
        names.add("**" + a.kwarg.arg)
    return names


def _class_fields(cls: ast.ClassDef) -> set[str]:
    out = set()
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return out


def check(megatron_root: str | Path, contract: tuple[Requirement, ...] = CONTRACT) -> list[str]:
    root = Path(megatron_root)
    problems = []
    for req in contract:
        path = root / req.path
        if not path.exists():
            problems.append(f"{req.path}: file missing")
            continue
        node: ast.AST | None = None
        body = _parse(path).body
        for part in req.symbol.split("."):
            node = _find(body, part)
            if node is None:
                break
            body = node.body
        if node is None:
            problems.append(f"{req.path}: {req.symbol} not found")
            continue
        if req.args:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                problems.append(f"{req.path}: {req.symbol} is not a function")
                continue
            names = _arg_names(node)
            if not any(n.startswith("**") for n in names):
                missing = [a for a in req.args if a not in names]
                if missing:
                    problems.append(f"{req.path}: {req.symbol} lacks args {missing}")
        if req.fields:
            if not isinstance(node, ast.ClassDef):
                problems.append(f"{req.path}: {req.symbol} is not a class")
                continue
            missing = [f for f in req.fields if f not in _class_fields(node)]
            if missing:
                problems.append(f"{req.path}: {req.symbol} lacks fields {missing}")
    return problems


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = argv[0] if argv else "third_party/Megatron-LM"
    problems = check(root)
    for p in problems:
        print("CONTRACT VIOLATION:", p)
    print(f"{len(CONTRACT) - len(problems)}/{len(CONTRACT)} requirements satisfied")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
