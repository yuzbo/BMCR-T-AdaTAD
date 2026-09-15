"""Single-shot module launcher for the EXISTING site's python run_job.sh.

This is NOT a dispatcher and never submits a job. Place its absolute path in a
stage's args only after the existing deployment owner approves registration.
"""
import runpy
import sys

allowed={'probe','forecast','fit_router','make_configs','snapshot_metrics','transfer_router'}
if len(sys.argv)<2 or sys.argv[1] not in allowed:
    raise SystemExit('Usage: dispatch_entry.py {'+','.join(sorted(allowed))+'} [args...]')
module='wtr.'+sys.argv[1];sys.argv=[module,*sys.argv[2:]]
runpy.run_module(module,run_name='__main__')
