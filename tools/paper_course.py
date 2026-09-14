"""One allocation for technical verification followed immediately by the full course."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from h65.paper.runtime import json_write,read_config


def main(args):
    cfg=read_config(args.config);start=time.perf_counter();audit=ROOT/args.preflight_output
    if cfg.get('wtr_fasttrack'):
        from h65.paper.fasttrack import admission
        resources=json.loads((ROOT/'research/paper/resources.local.json').read_text())
        admission(cfg,resources)
    if not (audit/'completed.json').exists():
        result=subprocess.run([sys.executable,'-u',str(ROOT/'tools/paper_train.py'),'--config',args.config,'--preflight','--output',str(audit)])
        if result.returncode:return result.returncode
    verified=json.loads((audit/'completed.json').read_text())
    if verified.get('real_task_updates')!=2 or not verified.get('no_gt_inference'):raise RuntimeError('Incomplete integrated GPU check')
    if cfg.get('wtr_fasttrack'):
        if verified['config']!=cfg or not verified.get('fresh_instance_strict_reload'):
            raise RuntimeError('Fast-Track course requires matching fresh-instance reload preflight')
        if verified['wtr_fasttrack_science_sha']!=(ROOT/'WTR_FASTTRACK_SCIENCE_SHA').read_text().strip():
            raise RuntimeError('Preflight is from another scientific revision')
    if cfg['recipe']=='graph_tad_v1':
        if verified['config']!=cfg:raise RuntimeError('Graph preflight configuration mismatch')
        gradients=json.loads((audit/'graph_gradient_contract.json').read_text())
        if gradients['relation_gradient_l1']<=0 or not gradients['changed_groups']:raise RuntimeError('Graph relation learning was not verified')
    elapsed=time.perf_counter()-start
    json_write(ROOT/'research/paper/runs'/cfg['id']/'integrated_preflight.json',dict(audit=str(audit),seconds_in_this_allocation=elapsed,
        audit_job_id=verified['slurm_job_id'],course_job_id=os.environ['SLURM_JOB_ID'],preflight_updates_discarded=True))
    remaining=max(.25,args.slice_hours-elapsed/3600)
    os.execv(sys.executable,[sys.executable,'-u',str(ROOT/'tools/paper_train.py'),'--config',args.config,'--resume','--slice-hours',str(remaining)])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--preflight-output',required=True)
    p.add_argument('--slice-hours',type=float,default=10);raise SystemExit(main(p.parse_args()))
