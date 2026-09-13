"""Apply the user's seed-42-only decision to owned unfinished training jobs."""
import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'research/paper'
state=json.loads((EXP/'deployment.json').read_text());pid=state['controller_pid']
proc=Path(f'/proc/{pid}/cmdline')
if proc.exists():
    if b'paper_dispatch.py' not in proc.read_bytes():raise RuntimeError('Unexpected controller PID')
    os.kill(pid,signal.SIGTERM)
    for _ in range(25):
        if not proc.exists():break
        time.sleep(.2)
    else:raise RuntimeError('Controller did not stop')
queue=subprocess.run(['squeue','-u',os.environ['USER'],'-h','-o','%i|%T'],text=True,capture_output=True,check=True)
live=dict(x.split('|',1) for x in queue.stdout.splitlines() if '|' in x)
legacy_path=ROOT.parent/'fpw_3d_20260913/research/frame/deployment.json';legacy=json.loads(legacy_path.read_text())
jobs=[]
for name,stage in state['stages'].items():
    if str(stage.get('job_id')) in live:jobs.append(dict(program='paper',stage=name,job_id=stage['job_id'],state=live[str(stage['job_id'])]))
for name,stage in legacy['stages'].items():
    if stage['kind']=='train' and str(stage.get('job_id')) in live:jobs.append(dict(program='legacy_train',stage=name,job_id=stage['job_id'],state=live[str(stage['job_id'])]))
if jobs:subprocess.run(['scancel',*[str(x['job_id']) for x in jobs]],check=True)
archive=EXP/'seed42_transition';archive.mkdir(exist_ok=True)
(archive/'previous_deployment.json').write_text(json.dumps(state,indent=2)+'\n')
(archive/'previous_plan.json').write_text((EXP/'plan.json').read_text())
(archive/'legacy_before.json').write_text(json.dumps(legacy,indent=2)+'\n')
for stage in legacy['stages'].values():
    if stage['kind']=='train' and stage.get('status')!='COMPLETED':
        stage['status']='FAILED';stage['failure_note']='Superseded by user seed42-only paper courses; do not resubmit old-seed training'
        stage['cancelled_by_seed_policy']=True;stage.pop('job_id',None)
legacy['training_superseded_by_seed42']=True
legacy_path.write_text(json.dumps(legacy,indent=2)+'\n')
record=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),old_controller_pid=pid,cancelled_jobs=jobs,
    required_seed=42,completed_results_preserved=True,anet_preparation_preserved=True,bmcr80_completed_course_evaluations_preserved=True)
(archive/'cancellation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
