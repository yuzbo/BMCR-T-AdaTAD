#!/bin/bash
source /etc/profile
set -euo pipefail
cd /XYAIFS00/HDD_POOL/pxyai/pxyai_0057/yzb/wtr_fasttrack_20260915
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
exec /HOME/pxyai/pxyai_0057/HDD_POOL/yzb/geosparse_tad_20260907/envs/opentad/bin/python -u "$@"
