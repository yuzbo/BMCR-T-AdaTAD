"""Add the first Core pair to the existing 4090 owner, preserving retired stages."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root=Path('/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/paper_20260913')
core=Path('/data/run01/sczc063/yuzibo/wtr_fasttrack_20260915')
exp=root/'research/paper'
stages=json.loads((core/'research/wtr_fasttrack/deployment_stages.json').read_text())
if len(stages)!=2 or any(not k.startswith('wtr_train_wtr_d_') for k in stages):
    raise RuntimeError('Unexpected first-wave stage registration')
path=exp/'deployment.json';state=json.loads(path.read_text());pid=state['controller_pid']
process=Path('/proc')/str(pid)
if process.exists():
    cmd=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if str(root/'tools/paper_dispatch.py') not in cmd:raise RuntimeError('Unexpected owner')
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        if not process.exists():break
        time.sleep(.1)
    else:raise RuntimeError('Owner did not stop')
state=json.loads(path.read_text());planpath=exp/'plan.json';plan=json.loads(planpath.read_text())
for key,stage in stages.items():
    stage['source_revision']=(core/'WTR_FASTTRACK_SCIENCE_SHA').read_text().strip()
    if key in state['stages'] and state['stages'][key].get('job_id'):
        continue
    state['stages'][key]=stage;plan['stages'][key]=stage
for target,value in [(path,state),(planpath,plan)]:
    temp=target.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(target)
log=(exp/'dispatcher_fasttrack_retirement.log').open('a')
owner=subprocess.Popen([sys.executable,'-u',str(root/'tools/paper_dispatch.py'),'--submit','--max-live','7'],
    cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
log.close()
receipt=dict(owner_pid=owner.pid,stages=list(stages),same_owner_code=True,legacy_inheritance=False,
             science_sha=(core/'WTR_FASTTRACK_SCIENCE_SHA').read_text().strip(),inline_training_evaluation=True)
(core/'research/wtr_fasttrack/launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
