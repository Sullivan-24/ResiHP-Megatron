#!/usr/bin/env bash
# Single-node chaos smoke test used by .github/workflows/gpu.yml.
#
# 1. NCCL fast path on this node: shrink_group(SHRINK_ABORT) + split_group, abort of a
#    stuck collective, and a full rebuild after a member is killed (pytest -m gpu).
# 2. Control plane + runtime chaos with real processes: kill -9, SIGSTOP hang and a 2x
#    fail-slow on 8 workers; every trial must finish with parameters bit-identical to
#    the single-process reference (CPU harness, independent of GPU count).
# 3. Static contract of the Megatron adapters against the pinned, patched checkout.
#
# The Megatron GPU end-to-end run (examples/megatron/resilient_pretrain.py) is opt-in
# with RESIHP_MEGATRON_E2E=1 until it has been validated on hardware.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
OUT="${RESIHP_SMOKE_OUT:-results/smoke}"
mkdir -p "${OUT}"

echo "== NCCL shrink/split/abort fast path"
pytest -q -m gpu tests/gpu

echo "== chaos: kill -9, hang, fail-slow (exactness checked per trial)"
resihp-bench --out "${OUT}" chaos --trials 1 --faults kill,hang,slow
grep -q MISMATCH "${OUT}/chaos/chaos.md" && { echo "semantic mismatch after recovery" >&2; exit 1; }

echo "== Megatron adapter contract"
python -m resihp.compat.megatron_contract third_party/Megatron-LM

if [[ "${RESIHP_MEGATRON_E2E:-0}" == "1" ]]; then
  echo "== Megatron GPT end-to-end with an injected kill -9 (experimental)"
  NPROC="$(nvidia-smi -L | wc -l)"
  RESIHP_FAULTS="kill@iter=20:target=n0g$((NPROC - 1)):phase=forward" \
    resihp-run local --nnodes 1 --nproc-per-node "${NPROC}" --planner "${RESIHP_PLANNER:?module:function}" -- \
      python examples/megatron/resilient_pretrain.py ${RESIHP_MEGATRON_ARGS:?set RESIHP_MEGATRON_ARGS}
fi
echo "chaos smoke passed"
