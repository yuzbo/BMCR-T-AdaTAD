"""Register graph courses in the existing paper controller; no second scheduler."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from h65.paper.runtime import json_write
from tools.graph_plan import build


def main():
    owner=ROOT.parent/'paper_20260913';exp=owner/'research/paper';out=ROOT/'research/paper/graph'
    receipt=out/'registration.json'
    if receipt.exists():print(receipt.read_text());return
    validation=json.loads((out/'assets_validation.json').read_text())
    cpu=json.loads((out/'cpu_validation.json').read_text())
    if not validation.get('passed') or cpu.get('tests_passed')!=10:raise RuntimeError('Graph CPU validation is not complete')
    addon=build(ROOT)
    for stage in addon['stages'].values():
        if not Path(stage['args'][0]).exists():raise RuntimeError('Missing final graph stage entry: '+stage['args'][0])
    state=json.loads((exp/'deployment.json').read_text());pid=state['controller_pid']
    command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
    if not any('paper_dispatch.py' in item for item in command):raise RuntimeError('Unexpected owner process')
    json_write(out/'owner_before_registration.json',state)
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        if not Path(f'/proc/{pid}').exists():break
        time.sleep(.1)
    deferred=[];failure=None
    try:
        with (exp/'dispatcher.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=json.loads((exp/'deployment.json').read_text());plan=json.loads((exp/'plan.json').read_text())
            for name,stage in addon['stages'].items():
                if name in state['stages'] or name in plan['stages']:raise RuntimeError('Graph stage already exists without a registration receipt')
                state['stages'][name]=dict(stage,status='WAITING');plan['stages'][name]=stage
            # Move only our two unstarted V1 continuation requests behind the new
            # S/B technical checks. Their checkpoints and 80-epoch courses remain.
            for name in ('train_thumos_s_point_full_seed42','train_thumos_b_point_full_seed42'):
                stage=state['stages'][name];jid=stage.get('job_id')
                if jid is None:continue
                status=subprocess.run(['squeue','-j',str(jid),'-h','-o','%T'],text=True,capture_output=True,check=True)
                if status.stdout.strip()!='PENDING':continue
                checkpoint=Path(stage.get('resume_checkpoint',owner/'research/paper/runs'/stage['config_id']/'latest.pth'))
                if not checkpoint.is_absolute():checkpoint=owner/checkpoint
                if not checkpoint.exists():raise RuntimeError('Continuation checkpoint is missing')
                subprocess.run(['scancel','--state=PENDING',str(jid)],check=True)
                after=subprocess.run(['squeue','-j',str(jid),'-h','-o','%T'],text=True,capture_output=True,check=True)
                if after.stdout.strip():continue
                event=dict(job_id=jid,stage=name,reason='Latest user request: graph S/B technical checks before unstarted saved V1 continuations; no training results discarded')
                deferred.append(event);stage.setdefault('priority_deferrals',[]).append(event)
                stage.pop('job_id');stage['status']='WAITING'
            extension=dict(configurations=7,main80=4,mechanism40=3,stages=len(addon['stages']),seed=42,runtime=str(ROOT))
            plan['graph_extension']=extension;state['graph_extension']=extension
            json_write(exp/'plan.json',plan);json_write(exp/'deployment.json',state)
    except BaseException as error:failure=error
    finally:
        log=(exp/'slurm/controller_review_priority.log').open('a')
        child=subprocess.Popen(command,cwd=owner,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close()
    if failure is not None:raise failure
    result=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),previous_controller=pid,controller_pid=child.pid,
        owner=str(owner),runtime=str(ROOT),source_revision=(ROOT/'source_revision.txt').read_text().strip(),
        new_configurations=7,total_configurations=86,total_training_courses=84,new_stages=len(addon['stages']),total_stages=len(state['stages']),
        graph_stage_ids=list(addon['stages']),deferred_pending_jobs=deferred,running_jobs_cancelled=0,
        performance_gates=False,queue_limits=dict(max_live=10,max_train=8,account_jobs=16))
    json_write(receipt,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
