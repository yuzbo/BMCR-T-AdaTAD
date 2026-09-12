#!/usr/bin/env bash
set -eo pipefail
source /etc/profile
module load cuda/11.8
module load miniforge3/24.11
set -u
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Launch through tools/ds3_dispatch.py from the project root}"
cd "$PROJECT_ROOT"
export H65_RESOURCE_ROOT="$PROJECT_ROOT/resources"
export DS3_RUNS_DIR="$PROJECT_ROOT/ds3_20260912/runs"
export TORCH_HOME="$PROJECT_ROOT/cache/torch"
export MPLCONFIGDIR="$PROJECT_ROOT/cache/matplotlib"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/upstream${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
mkdir -p "$TORCH_HOME" "$MPLCONFIGDIR" "$DS3_RUNS_DIR"
exec /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python -u "$@"
