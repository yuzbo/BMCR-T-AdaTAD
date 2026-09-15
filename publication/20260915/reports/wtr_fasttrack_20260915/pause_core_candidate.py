"""Preserve the reviewed candidate before replacing its scientific split protocol."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--site',choices=['4090','a100'],required=True);a=p.parse_args()
if a.site=='4090':
    core=Path('/data/run01/sczc063/yuzibo/wtr_fasttrack_20260915')
    owner_root=Path('/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/paper_20260913')
else:
    core=Path('/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/wtr_fasttrack_20260915')
    owner_root=core
core=core.resolve();owner_root=owner_root.resolve();exp=owner_root/'research/paper'
statefile=exp/'deployment.json';state=json.loads(statefile.read_text());pid=state['controller_pid']
argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
if not any(Path(arg).resolve()==owner_root/'tools/paper_dispatch.py' for arg in argv if arg.endswith('paper_dispatch.py')):
    raise RuntimeError('Unexpected owner process')
os.kill(pid,signal.SIGTERM)
for _ in range(50):
    if not Path('/proc',str(pid)).exists():break
    time.sleep(.1)
else:raise RuntimeError('Owner did not exit')
state=json.loads(statefile.read_text());planfile=exp/'plan.json';plan=json.loads(planfile.read_text())
targets=[]
for name,stage in state['stages'].items():
    if not name.startswith('wtr_train_'):continue
    if stage.get('job_id'):
        targets.append(str(stage['job_id']))
        subprocess.run(['scancel','--signal=USR1','--batch',str(stage['job_id'])],capture_output=True)
    stage.update(cancelled_by_user=True,status='CANCELLED',cancellation_reason='Cross-review protocol correction; replace candidate, retain artifacts')
    stage['review_candidate_job_id']=stage.pop('job_id',None)
    if name in plan['stages']:plan['stages'][name].update(cancelled_by_user=True,status='CANCELLED')
for path,value in [(statefile,state),(planfile,plan)]:
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
print(json.dumps({'owner_paused':pid,'candidate_jobs':targets}),flush=True)
time.sleep(35)
if targets:subprocess.run(['scancel',*targets],capture_output=True)
time.sleep(3)
archive=core/'research/paper/review_candidates'/('578bab2_'+time.strftime('%Y%m%d_%H%M%S'))
archive.mkdir(parents=True,exist_ok=False)
for run in (core/'research/paper/runs').glob('*wtr_*'):
    source=run.resolve();target=(archive/run.name).resolve()
    if not source.is_relative_to(core) or not target.is_relative_to(core):raise RuntimeError('Archive leaves the Core directory')
    run.rename(target)
(archive/'review_status.json').write_text(json.dumps(dict(status='candidate_superseded_before_scientific_use',
    reason='Value supervision used all 200 training videos; future router-label holdout must stay out of fitting',
    source_revision='578bab2a97431a908764fa0ab30a53c47fdfa336',jobs=targets,files_deleted=False),indent=2)+'\n')
print(json.dumps({'archive':str(archive),'owner_left_paused':True}),flush=True)
