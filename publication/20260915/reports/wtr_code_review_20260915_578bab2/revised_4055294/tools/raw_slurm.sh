#!/usr/bin/env bash
# Scheduler allocation is required; use one independent GPU, never a login-node forward.
set -euo pipefail
root=$1
python_bin=$2
shift 2
cd "$root"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
exec "$python_bin" tools/raw_run.py --resources runtime/resources.json --protocol runtime/protocol.json "$@"
