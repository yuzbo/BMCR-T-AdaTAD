#!/usr/bin/env bash
# TEMPLATE: review partition/account/time at your site before submitting.
#SBATCH --partition=gpu
#SBATCH --account=gpugpu
#SBATCH --qos=normal
#SBATCH --job-name=h65-ds3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
set -euo pipefail
: "${BUNDLE:?Set absolute bundle directory}"
: "${MANIFEST:?Set reviewed launch manifest}"
: "${H65_DS3_ALLOW_GPU_RUN:?Explicit authorization variable is required}"
[[ "$H65_DS3_ALLOW_GPU_RUN" == 1 ]] || exit 2
ENV_ROOT=/data/run01/sczc063/yuzibo/conda_envs/opentad
[[ -f "$ENV_ROOT/bin/activate" ]] || { echo 'Existing OpenTAD environment missing' >&2; exit 2; }
source "$ENV_ROOT/bin/activate"
# Respect the scheduler's CUDA_VISIBLE_DEVICES. Do not force physical GPU1 here.
python "$BUNDLE/bin/run_experiment.py" "$MANIFEST" --execute
