#!/usr/bin/env bash
# Fetch the pinned Megatron-LM commit into third_party/ and apply the ResiHP patch set.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN_FILE="${ROOT}/MEGATRON_PIN"
DEST="${ROOT}/third_party/Megatron-LM"

repo="$(grep '^repo=' "${PIN_FILE}" | cut -d= -f2-)"
commit="$(grep '^commit=' "${PIN_FILE}" | cut -d= -f2-)"

if [[ ! -d "${DEST}/.git" ]]; then
  mkdir -p "$(dirname "${DEST}")"
  git init -q "${DEST}"
  git -C "${DEST}" remote add origin "${repo}"
fi

if ! git -C "${DEST}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
  git -C "${DEST}" fetch --depth 1 origin "${commit}"
fi
git -C "${DEST}" checkout -q --detach "${commit}"

APPLY="${RESIHP_APPLY_PATCHES:-1}"
if [[ "${APPLY}" == "1" ]]; then
  for p in "${ROOT}"/patches/megatron/*.patch; do
    [[ -e "${p}" ]] || continue
    if git -C "${DEST}" apply --check "${p}" 2>/dev/null; then
      git -C "${DEST}" apply "${p}"
      echo "applied $(basename "${p}")"
    elif git -C "${DEST}" apply --reverse --check "${p}" 2>/dev/null; then
      echo "already applied $(basename "${p}")"
    else
      echo "FAILED to apply $(basename "${p}")" >&2
      exit 1
    fi
  done
fi

echo "Megatron-LM ready at ${DEST} (${commit})"
echo "export PYTHONPATH=${DEST}:\${PYTHONPATH:-}"
