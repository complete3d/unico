#!/usr/bin/env bash

if [ "${DEBUG:-0}" = "1" ]; then
  set -x
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NGPUS=$(nvidia-smi --list-gpus | wc -l)
else
  NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c)
  NGPUS=$((NGPUS + 1))
fi

PORT="${MASTER_PORT:-29500}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  PORT="$1"
  shift 1
fi

echo "Using $NGPUS GPUs"

UNICO_USE_HELPER_SCRIPT=1 torchrun --master_port="${PORT}" --nproc_per_node="${NGPUS}" main.py "$@"

# Usage:
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_ddp.sh
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_ddp.sh 29500
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_ddp.sh 29500 exp_name=my_exp runtime.seed=1117
# ./scripts/train_ddp.sh  # Uses all GPUs and default port 29500
