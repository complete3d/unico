#!/usr/bin/env bash

set -x
PY_ARGS=${@:1}

UNICO_USE_HELPER_SCRIPT=1 python3 inference.py ${PY_ARGS}
