#!/usr/bin/env bash
set -eo pipefail
source /etc/profile
module load cuda/11.8
module load miniforge3/24.11
set -u
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the FPW source root}"
cd "$PROJECT_ROOT"
export H65_RESOURCE_ROOT="$PROJECT_ROOT/resources"
export H65_RUNS_DIR="$PROJECT_ROOT/research/frame/runs"
export H65_SOURCE_REVISION="$(<"$PROJECT_ROOT/source_revision.txt")"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/upstream${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$PROJECT_ROOT/cache/torch" MPLCONFIGDIR="$PROJECT_ROOT/cache/matplotlib"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
mkdir -p "$TORCH_HOME" "$MPLCONFIGDIR" "$H65_RUNS_DIR"
exec /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python -u "$@"
