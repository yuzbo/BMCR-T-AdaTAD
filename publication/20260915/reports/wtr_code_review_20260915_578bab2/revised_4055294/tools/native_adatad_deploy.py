"""One-time addition and priority handoff inside the existing owned controller."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from h65.paper.runtime import json_write
from tools.native_adatad_plan import build


def main():
    owner=ROOT.parent/'paper_20260913';exp=owner/'research/paper';out=ROOT/'research/paper/native_adatad'
    receipt=out/'registration.json'
    if receipt.exists():
        print(receipt.read_text());return
    validation=json.loads((out/'cpu_validation.json').read_text())
    if not validation.get('passed') or validation.get('test_windows')!=792:
        raise RuntimeError('Native CPU contract validation missing')
    addon=build(ROOT)
    for stage in addon['stages'].values():
        if stage.get('args') and not Path(stage['args'][0]).exists():raise RuntimeError('Missing deployed entry')
    state=json.loads((exp/'deployment.json').read_text());pid=state['controller_pid']
    command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
    if not any('paper_dispatch.py' in x for x in command):raise RuntimeError('Unexpected owner controller')
    json_write(out/'owner_before_registration.json',state)
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        if not Path(f'/proc/{pid}').exists():break
        time.sleep(.1)
    deferred=[];failure=None
    try:
        with (exp/'dispatcher.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=json.loads((exp/'deployment.json').read_text())
            plan=json.loads((exp/'plan.json').read_text())
            for name,stage in addon['stages'].items():
                if name in state['stages'] or name in plan['stages']:
                    raise RuntimeError('Native stage already registered without receipt: '+name)
                state['stages'][name]=dict(stage,status='WAITING')
                plan['stages'][name]=stage
            # Only unstarted supplementary jobs yield queue slots. Running
            # courses, V1 continuations and the shared data CPU step are untouched.
            for name in ('eval_public_adatad_s_retest','preflight_anet_b_point_full_seed42'):
                stage=state['stages'].get(name,{});jid=stage.get('job_id')
                if jid is None:continue
                query=subprocess.run(['squeue','-j',str(jid),'-h','-o','%T'],text=True,capture_output=True,check=True)
                if query.stdout.strip()!='PENDING':continue
                subprocess.run(['scancel',str(jid)],check=True)
                item=dict(job_id=jid,stage=name,reason='User-requested native AdaTAD uniform baseline receives the next evaluation slots; unstarted job deferred')
                deferred.append(item);stage.setdefault('priority_deferrals',[]).append(item)
                stage.pop('job_id');stage['status']='WAITING'
            plan['native_adatad_extension']=dict(configs=4,training_courses=2,stages=10,seed=42)
            state['native_adatad_extension']=plan['native_adatad_extension']
            json_write(exp/'plan.json',plan);json_write(exp/'deployment.json',state)
            json_write(out/'plan.json',addon)
    except BaseException as error:
        failure=error
    finally:
        log=(exp/'slurm/controller_review_priority.log').open('a')
        child=subprocess.Popen(command,cwd=owner,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close()
    if failure is not None:raise failure
    result=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),previous_controller=pid,controller_pid=child.pid,
        owner=str(owner),runtime=str(ROOT),source_revision=(ROOT/'source_revision.txt').read_text().strip(),
        configurations=4,new_training_courses=2,total_training_courses=77,new_stages=10,total_stages=len(state['stages']),
        native_stage_ids=list(addon['stages']),deferred_pending_jobs=deferred,running_jobs_cancelled=0,
        queue_policy='Existing single controller, max_live10/max_train8; K384 direct S/B priority -3, native dense -2, adaptation -1',
        gpu_preflight='Two discarded GT updates before zero-shot eval; adaptation waits only for its own technical receipt',
        no_performance_gate=True)
    json_write(receipt,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
