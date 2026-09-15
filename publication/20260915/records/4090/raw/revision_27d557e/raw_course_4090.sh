#!/usr/bin/env bash
set -euo pipefail
cd /data/run01/sczc063/yuzibo/wtr_raw_v1_20260915/revision_27d557e
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
python_bin=/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python
nvidia-smi --query-gpu=name,uuid --format=csv
"$python_bin" tools/raw_run.py --resources runtime/resources.json --protocol runtime/protocol.json --stage gate --output results/r2_six
"$python_bin" tools/raw_run.py --resources runtime/resources.json --protocol runtime/protocol.json --stage mini-bank --gate-receipt results/r2_six/passed.json --output results/mini_bank
"$python_bin" tools/raw_train.py --bank results/mini_bank --protocol runtime/protocol.json --output results/mini_diagnostics --analyze-only
