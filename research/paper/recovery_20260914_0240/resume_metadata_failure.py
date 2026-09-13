"""Retry only the eight diagnosed pre-update metadata failures with seed42."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'research/paper';OUT=EXP/'recovery_20260914_0240'
OUT.mkdir(exist_ok=True);state=json.loads((EXP/'deployment.json').read_text());pid=state['controller_pid']
proc=Path(f'/proc/{pid}/cmdline')
if proc.exists():
    assert b'paper_dispatch.py' in proc.read_bytes()
    os.kill(pid,signal.SIGTERM)
    for _ in range(25):
        if not proc.exists():break
        time.sleep(.2)
    else:raise RuntimeError('Controller has not stopped')
(OUT/'deployment_before_metadata_retry.json').write_text(json.dumps(state,indent=2)+'\n')
ids={1288635,1288636,1288637,1288638,1288639,1288640,1288641,1288642};retried=[]
for name,stage in state['stages'].items():
    jid=stage.get('job_id')
    if jid not in ids:continue
    assert stage.get('status')=='FAILED' and stage['config_id'].endswith('_seed42')
    log=(EXP/'slurm'/f'{jid}.log').read_text()
    assert "dict() got multiple values for keyword argument 'source_revision'" in log
    (OUT/f'{jid}_failure.log').write_text(log)
    stage.setdefault('diagnosed_failures',[]).append(dict(job_id=jid,reason='metadata source_revision duplicate before training updates',fix_revision='a8930de649e55ed9d1e8709c4cc895f4049f109e'))
    for key in ('job_id','exit_code','scheduler_state','failure'):stage.pop(key,None)
    stage['status']='WAITING';retried.append(name)
assert len(retried)==8
(EXP/'deployment.json').write_text(json.dumps(state,indent=2)+'\n')
(ROOT/'source_revision.txt').write_text('a8930de649e55ed9d1e8709c4cc895f4049f109e\n')
with (EXP/'slurm/controller_metadata_fix.log').open('a') as log:
    child=subprocess.Popen([sys.executable,'-u','tools/paper_dispatch.py','--submit','--max-live','10','--legacy-receipt',str(ROOT.parent/'fpw_3d_20260913/research/frame/deployment.json'),'--inherit-legacy'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(old_pid=pid,new_pid=child.pid,retried_stages=retried,seed=42,same_courses=True,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
(OUT/'metadata_retry.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
