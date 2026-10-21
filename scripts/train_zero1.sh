#!/usr/bin/env bash
# Single-node training; eight GPUs by default.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

NPROC_PER_NODE=8
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  NPROC_PER_NODE="$1"
  shift
fi
if ((NPROC_PER_NODE < 1 || NPROC_PER_NODE > 8)); then
  echo "Usage: bash scripts/train_zero1.sh [1-8, default: 8] [task=robotwin|robodojo] [hydra_overrides...]" >&2
  exit 1
fi

TASK_NAME=robotwin
for arg in "$@"; do
  case "${arg}" in task=*) TASK_NAME="${arg#task=}" ;; esac
done
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

exec accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_machines 1 \
  --num_processes "${NPROC_PER_NODE}" \
  --main_process_port "${MASTER_PORT:-29500}" \
  scripts/train.py \
  "output_dir=./runs/${TASK_NAME}/${RUN_ID}" \
  "wandb.name=${TASK_NAME}" \
  "$@"
