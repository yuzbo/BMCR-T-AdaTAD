#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
python -m pytest -q "$HERE/tests"
python -m compileall -q "$HERE/wtr" "$HERE/dispatch_entry.py"
for MOD in probe forecast fit_router transfer_router make_configs snapshot_metrics; do
  python -m "wtr.$MOD" --help >/dev/null
done
printf 'CPU contracts and CLI parsing passed. CUDA/OpenTAD integration is NOT covered.\n'
