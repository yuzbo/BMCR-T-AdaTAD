#!/usr/bin/env bash
# Submit this script through your own Slurm allocation. Site account/partition/QOS
# and environment activation are supplied by the operator, not embedded here.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/upstream${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export TORCH_HOME="$PWD/cache/torch"
export MPLCONFIGDIR="$PWD/cache/matplotlib"
exec "${H65_PYTHON:-python}" -u "$@"
