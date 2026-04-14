#!/usr/bin/env bash

if [ "${DEBUG:-0}" = "1" ]; then
  set -x
fi

export LOCAL_RANK=0

UNICO_USE_HELPER_SCRIPT=1 python3 main.py "$@"

# Usage:
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_dp.sh
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_dp.sh exp_name=my_exp runtime.seed=1117
