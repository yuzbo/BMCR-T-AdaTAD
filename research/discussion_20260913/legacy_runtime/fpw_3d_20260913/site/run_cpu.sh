#!/usr/bin/env bash
set -eo pipefail
source /etc/profile
module load cuda/11.8
module load miniforge3/24.11
set -u
PROJECT_ROOT="$(pwd)"
export H65_RESOURCE_ROOT="$PROJECT_ROOT/resources"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/upstream${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="$PROJECT_ROOT/cache/matplotlib"
exec /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python -u "$@"
