"""One-time recovery of the nine diagnosed failed preflights and P0 registration."""
import json, os, signal, subprocess, sys, time
from pathlib import Path

base=Path('/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910')
old=base/'paper_20260913';new=base/'support_review_20260914'
exp=old/'research/paper';out=new/'research/paper/review_5485'
check=exp/'runs/preflight_strict_inference_s'
proof=json.loads((check/'completed.json').read_text())
probe=json.loads((check/'inference_contract.json').read_text())
assert proof['real_task_updates']==2 and proof['no_gt_inference'] and proof['strict_state_reload']
assert probe['passed'] and probe['repeat_close'] and probe['atol']==1e-6 and probe['rtol']==1e-5
state_path=exp/'deployment.json';state=json.loads(state_path.read_text())
pid=state['controller_pid'];proc=Path(f'/proc/{pid}/cmdline')
if proc.exists():
    assert b'paper_dispatch.py' in proc.read_bytes()
    os.kill(pid,signal.SIGTERM)
    for _ in range(25):
        if not proc.exists():break
        time.sleep(.2)
    else:raise RuntimeError('Old controller did not stop')
state=json.loads(state_path.read_text())
(out/'reconnect_before_retry.json').write_text(json.dumps(state,indent=2)+'\n')
targets=[
 'preflight_anet_s_point_full_seed42','preflight_anet_b_point_full_seed42',
 'train_thumos_s_point_full_seed42','train_thumos_b_point_full_seed42',
 'train_thumos_s_point_full_finetune_seed42','train_thumos_s_point_mae_pretrained_seed42',
 'train_thumos_internvideo_mq_point_full_seed42',
 'train_thumos_s_tadtr_full_seed42','train_thumos_b_tadtr_full_seed42']
retried=[]
for name in targets:
    row=state['stages'][name]
    if row.get('status')!='FAILED':continue
    assert row['kind']=='preflight' or row.get('contains_preflight')
    assert not (exp/row['done']).exists()
    if row['kind']=='train':assert not (exp/'runs'/row['config_id']/'latest.pth').exists()
    record=dict(stage=name,job_id=row.get('job_id'),fix_revision=(old/'source_revision.txt').read_text().strip(),
                reason='single-rank TadTR distributed initialization' if 'tadtr' in name else 'nondeterministic CUDA inference contract; strict deterministic repeat/control now passed',
                verification_job=proof['slurm_job_id'])
    row.setdefault('diagnosed_failures',[]).append(record);retried.append(record)
    for key in ('job_id','scheduler_state','exit_code','failure','submission_error'):row.pop(key,None)
    row['status']='WAITING'
tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
(out/'technical_retries.json').write_text(json.dumps(retried,indent=2)+'\n')
subprocess.run([sys.executable,str(new/'tools/paper_review_register.py'),'--owner-root',str(old),'--register'],cwd=new,check=True)
print(json.dumps(dict(retried=len(retried),verification_job=proof['slurm_job_id'])))
