#!/bin/bash
source /etc/profile
set -euo pipefail
cd /data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/paper_20260913
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/upstream${PYTHONPATH:+:$PYTHONPATH}"
export H65_SOURCE_REVISION="$(< source_revision.txt)"
exec /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python -u "$@"
