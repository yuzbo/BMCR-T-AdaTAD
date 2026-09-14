"""Requeue the diagnosed PBD-B preflight with bounded-memory support statistics."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
BASE=Path('/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910');OLD=BASE/'paper_20260913';NEW=BASE/'support_review_20260914'
EXP=OLD/'research/paper';OUT=NEW/'research/paper/review_5485/monitor_20260914_1403';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(NEW));from h65.paper.runtime import json_write
receipt=OUT/'pbd_b_memory_retry.json'
if receipt.exists():print(receipt.read_text());raise SystemExit(0)
revision=(NEW/'source_revision.txt').read_text().strip()
assert revision=='5b88b3cc4dfc216df29a659892caf7d6860036b1'
state=json.loads((EXP/'deployment.json').read_text());name='train_review5485_p01_b_seed42';target=state['stages'][name]
assert target['status']=='FAILED' and target['job_id']==1289577
assert not (NEW/'research/paper/runs/review5485_p01_b_seed42/latest.pth').exists()
pid=state['controller_pid'];proc=Path(f'/proc/{pid}/cmdline')
if proc.exists():
    assert b'paper_dispatch.py' in proc.read_bytes();os.kill(pid,signal.SIGTERM)
    for _ in range(25):
        if not proc.exists():break
        time.sleep(.2)
    else:raise RuntimeError('Previous controller did not stop')
state=json.loads((EXP/'deployment.json').read_text());plan=json.loads((EXP/'plan.json').read_text())
json_write(OUT/'before_memory_retry_deployment.json',state)
queue=dict(line.split('|') for line in subprocess.check_output(['squeue','-u','sczc063','-h','-o','%i|%T'],text=True).splitlines())
deferred=[]
for key,row in state['stages'].items():
    job=row.get('job_id')
    if key!=name and queue.get(str(job))=='PENDING':
        entry=dict(job_id=job,stage=key,reason='P0 PBD-B technical recovery before queued continuations and supplementary work')
        deferred.append(entry);row.setdefault('priority_deferrals',[]).append(entry);row.pop('job_id');row['status']='WAITING'
if deferred:subprocess.run(['scancel','--state=PENDING',*[str(x['job_id']) for x in deferred]],check=True)
target=state['stages'][name]
target.setdefault('diagnosed_failures',[]).append(dict(job_id=1289577,reason='OOM in support-state cosine diagnostics during preflight',
    fix='Per-pack FP32 statistics; recomputed masked NMSE gradient; diagnostic attention detached from unused loss graph',fix_revision=revision,
    validation='FP32/BF16 value, gradient, invalid-mask and target-stop-gradient CPU tests passed'))
for key in ('job_id','scheduler_state','exit_code','failure','submission_error'):target.pop(key,None)
target.update(status='WAITING',priority=-1,source_revision=revision)
plan['stages'][name].update(priority=-1,source_revision=revision)
json_write(EXP/'plan.json',plan);json_write(EXP/'deployment.json',state)
with (EXP/'slurm/controller_support_memory_fix.log').open('a') as log:
    child=subprocess.Popen([sys.executable,'-u',str(OLD/'tools/paper_dispatch.py'),'--submit','--max-live','10','--legacy-receipt',
        str(BASE/'fpw_3d_20260913/research/frame/deployment.json'),'--inherit-legacy'],cwd=OLD,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),old_controller=pid,new_controller=child.pid,target=name,revision=revision,
            deferred_pending=deferred,running_jobs_preserved=True,seed=42,course_epochs=80,scientific_loss_definition_unchanged=True)
json_write(receipt,record);print(json.dumps(record))
