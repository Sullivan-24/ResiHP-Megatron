#!/usr/bin/env bash
# Baseline (no ResiHP) throughput/loss reproduction on the pinned Megatron-LM commit.
# Every ResiHP number is reported relative to this script's output on the same hardware.
#
#   MODEL=llama3_8b TP=4 PP=2 NNODES=1 bash examples/megatron/baseline.sh
#   python -m resihp.bench.megatron_log results/baseline_llama3_8b.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MEGATRON="${MEGATRON:-${ROOT}/third_party/Megatron-LM}"
MODEL="${MODEL:-llama3_8b}"
TP="${TP:-4}"
PP="${PP:-2}"
CP="${CP:-1}"
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
SEQ="${SEQ:-8192}"
MBS="${MBS:-1}"
GBS="${GBS:-128}"
ITERS="${ITERS:-200}"
LOG_DIR="${LOG_DIR:-${ROOT}/results}"
mkdir -p "${LOG_DIR}"

case "${MODEL}" in
  llama3_8b)
    MODEL_ARGS=(--num-layers 32 --hidden-size 4096 --ffn-hidden-size 14336 --num-attention-heads 32
                --group-query-attention --num-query-groups 8 --kv-channels 128 --vocab-size 128256) ;;
  llama3_70b)
    MODEL_ARGS=(--num-layers 80 --hidden-size 8192 --ffn-hidden-size 28672 --num-attention-heads 64
                --group-query-attention --num-query-groups 8 --kv-channels 128 --vocab-size 128256) ;;
  qwen25_32b)
    MODEL_ARGS=(--num-layers 64 --hidden-size 5120 --ffn-hidden-size 27648 --num-attention-heads 40
                --group-query-attention --num-query-groups 8 --kv-channels 128 --vocab-size 152064
                --add-qkv-bias) ;;
  *) echo "unknown MODEL=${MODEL}" >&2; exit 1 ;;
esac

COMMON=(
  --use-mcore-models
  --seq-length "${SEQ}" --max-position-embeddings "${SEQ}"
  --position-embedding-type rope --rotary-base 1000000 --rotary-percent 1.0
  --attention-dropout 0.0 --hidden-dropout 0.0
  --swiglu --normalization RMSNorm --untie-embeddings-and-output-weights --disable-bias-linear
  --micro-batch-size "${MBS}" --global-batch-size "${GBS}" --train-iters "${ITERS}"
  --lr 1.5e-4 --min-lr 1.5e-5 --lr-decay-style cosine --lr-warmup-iters 20
  --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95
  --bf16 --calculate-per-token-loss
  --tensor-model-parallel-size "${TP}" --pipeline-model-parallel-size "${PP}"
  --context-parallel-size "${CP}" --sequence-parallel
  --use-distributed-optimizer --overlap-grad-reduce
  --mock-data --tokenizer-type NullTokenizer --split 99,1,0
  --log-interval 1 --log-throughput --eval-iters 0
)

export CUDA_DEVICE_MAX_CONNECTIONS=1
LOG="${LOG_DIR}/baseline_${MODEL}_tp${TP}_pp${PP}_n${NNODES}.log"
cd "${MEGATRON}"
torchrun --nproc_per_node "${GPUS_PER_NODE}" --nnodes "${NNODES}" --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" \
  pretrain_gpt.py "${MODEL_ARGS[@]}" "${COMMON[@]}" "$@" 2>&1 | tee "${LOG}"

python -m resihp.bench.megatron_log "${LOG}" > "${LOG%.log}.json"
echo "summary written to ${LOG%.log}.json"
