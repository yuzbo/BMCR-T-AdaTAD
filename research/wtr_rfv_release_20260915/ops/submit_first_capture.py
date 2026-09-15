"""Submit two distinct reviewed RFV capture jobs; preserve receipts on the server."""
import json
from pathlib import Path
import subprocess
from deploy_probe import SITES,SSH_CONFIG,OUT

SCIENCE='8341588a51ae4247693aaeee3a52816598000c05'


def submit(site):
    cfg=SITES[site];code=cfg['root']+'/versions/'+SCIENCE
    results=cfg['root']+'/results/'+SCIENCE[:7]
    role='local_cf_and_mini0' if site=='4090' else 'mini1'
    script='''import json,os,subprocess,sys,time
from pathlib import Path
root=Path(ROOT);code=Path(CODE);result=Path(RESULTS);role=ROLE;python=PYTHON
sys.path[:0]=[str(code),str(code/'upstream')]
from h65.paper.runtime import json_write
receipt=root/'receipts'/SCIENCE/(role+'.json')
if receipt.exists():
 print(json.dumps(dict(reused_submission=json.loads(receipt.read_text()))));raise SystemExit(0)
(root/'slurm').mkdir(parents=True,exist_ok=True)
result.mkdir(parents=True,exist_ok=True)
commands=[python+' -u tools/rfv_run.py smoke --config configs/rfv/T-V.json --resources runtime/resources.json --protocol runtime/protocol.json --output '+str(result/('smoke_'+SITE))]
common=python+' -u tools/rfv_run.py '
if SITE=='4090':
 commands.append(common+'local_cf --config configs/rfv/T-U.json --resources runtime/resources.json --protocol runtime/protocol.json --output '+str(result/'T_LOCAL_CF_G0A'))
shard=0 if SITE=='4090' else 1
commands.append(common+'bank --config configs/rfv/T-U.json --resources runtime/resources.json --protocol runtime/protocol.json --cohort mini --shard '+str(shard)+' --shards 2 --output '+str(result/('mini_c40_shard'+str(shard))))
job=root/('run_'+role+'.sh')
job.write_text('#!/bin/bash\\nsource /etc/profile\\nset -euo pipefail\\ncd '+str(code)+'\\nexport OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 H65_SOURCE_REVISION='+SCIENCE+'\\n'+'\\n'.join(commands)+'\\n')
if SITE=='a100':options=['--partition=a100x','--gres=gpu:a100:1']
else:
 from tools.fidelity_dispatch import nodes_for_submission
 placement=nodes_for_submission()
 if placement is None:raise RuntimeError('No verified 4090 placement')
 eligible,excluded=placement
 options=['--partition=gpu','--qos=gpugpu','--gres=gpu:1']+(['--exclude='+','.join(excluded)] if excluded else [])
command=['sbatch','--parsable',*options,'--nodes=1','--ntasks=1','--cpus-per-task=6','--time=06:00:00',
 '--job-name=rfv-'+role,'--output='+str(root/'slurm/%j.log'),str(job)]
submission=subprocess.run(command,capture_output=True,text=True)
if submission.returncode:raise RuntimeError(submission.stdout+submission.stderr)
job_id=int(submission.stdout.strip().split(';')[0])
saved=dict(job_id=job_id,role=role,site=SITE,science_sha=SCIENCE,code=str(code),results=str(result),
 commands=commands,sbatch=command,submitted_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),scope='GPU technical smoke and real CF data; no scientific PASS inferred')
json_write(receipt,saved)
print(json.dumps(saved))
'''
    values={'ROOT':cfg['root'],'CODE':code,'RESULTS':results,'ROLE':role,'PYTHON':cfg['python'],'SITE':site,'SCIENCE':SCIENCE}
    header='\n'.join(key+'='+repr(value) for key,value in values.items())+'\n'
    # Exactly one submission attempt. If transport is uncertain, inspect the
    # remote receipt before rerunning; receipt presence returns the original ID.
    options=['-F',SSH_CONFIG,'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3']
    proc=subprocess.run(['ssh',*options,cfg['host'],cfg['python']+' -'],input=(header+script).encode(),capture_output=True)
    (OUT/('submit_'+site+'.txt')).write_bytes(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError(site+': inspect submission transcript/remote receipt before retrying')
    receipt=json.loads(proc.stdout)
    (OUT/('submission_'+site+'.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    import sys
    for site in sys.argv[1:] or ['4090','a100']:submit(site)
