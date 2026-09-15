"""Submit one independent A100 D/S diagnostic; do not touch the four courses."""
import json
import subprocess
from deploy_probe import SITES,SSH_CONFIG,OUT
site=SITES['a100'];REV='72459727f61454194f0d84865f31a16ed4e10448'
body=r'''import json,subprocess,shlex,time
from pathlib import Path
root=Path(ROOT);code=root/'versions'/REV;out=root/'results'/REV[:7]/'DS_DIAGNOSTIC'
receipt=root/'receipts'/('ds_diagnostic_'+REV[:7]+'.json')
if receipt.exists():print(receipt.read_text());raise SystemExit(0)
out.mkdir(parents=True,exist_ok=True);(root/'slurm').mkdir(exist_ok=True)
command=[PYTHON,str(code/'tools/rfv_ds_diagnostic.py'),
 '--checkpoint',str(root/'assets/ds_epoch40/wtr_d_v_s42_raw40.pth'),
 '--checkpoint',str(root/'assets/ds_epoch40/wtr_s_v_s42_raw40.pth'),
 '--resources',str(code/'runtime/resources.json'),'--output',str(out)]
script=root/('ds_diagnostic_'+REV[:7]+'.sh')
script.write_text('#!/bin/bash\nsource /etc/profile\nset -euo pipefail\ncd '+shlex.quote(str(code))+
 '\nexport OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 H65_SOURCE_REVISION='+REV+'\n'+shlex.join(command)+'\n')
job_name='rfv-ds-diagnostic-'+REV[:7]
queue=subprocess.run(['squeue','--me','-h','-o','%i|%j'],capture_output=True,text=True,check=True).stdout
if any(line.endswith('|'+job_name) for line in queue.splitlines()):raise RuntimeError('Same diagnostic already queued; inspect its original receipt')
args=['sbatch','--parsable','--partition=a100x','--gres=gpu:a100:1','--nodes=1','--ntasks=1','--cpus-per-task=6',
 '--time=02:00:00','--job-name='+job_name,'--output='+str(root/'slurm/%j.log'),str(script)]
completed=subprocess.run(args,capture_output=True,text=True,check=True)
job_id=int(completed.stdout.strip().split(';')[0])
saved=dict(job_id=job_id,source_revision=REV,site='a100',original_science='40552945ad5d56b7404f833cd86f998e8558e8b1',
 submitted_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),command=command,sbatch=args,output=str(out),
 log=str(root/'slurm'/(str(job_id)+'.log')),scope='two frozen raw40 models; no optimizer step or old-course change')
receipt.write_text(json.dumps(saved,indent=2));(out/'run_receipt.json').write_text(json.dumps(saved,indent=2))
print(json.dumps(saved))
'''
header='\n'.join(key+'='+repr(value) for key,value in dict(ROOT=site['root'],PYTHON=site['python'],REV=REV).items())+'\n'
compile(header+body,'ds_slurm_submission','exec')
options=['-F',SSH_CONFIG,'-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3']
result=subprocess.run(['ssh',*options,site['host'],site['python']+' -'],input=(header+body).encode(),capture_output=True)
(OUT/'DS_SUBMISSION_TRANSCRIPT.txt').write_bytes(result.stdout+result.stderr)
if result.returncode:raise RuntimeError('Inspect remote receipt before repeating an uncertain submission: '+result.stderr.decode(errors='replace'))
saved=json.loads(result.stdout);(OUT/'DS_DIAGNOSTIC_LAUNCH.json').write_text(json.dumps(saved,indent=2),encoding='utf8')
print(json.dumps(saved))
