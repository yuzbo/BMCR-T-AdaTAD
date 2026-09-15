#!/bin/bash
# CPU video preparation; the site's GPU partition requires one GPU reservation.
source /etc/profile
set -euo pipefail
cd /data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/paper_20260913
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
exec /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python -u tools/paper_data/prepare_anet_archives.py --ready-output research/paper/assets/anet_ready.json
