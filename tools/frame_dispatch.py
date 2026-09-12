"""Parallel FPW launch: only technical preflight dependencies, never mAP gating."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));EXP=ROOT/'research/frame';STATE=EXP/'deployment.json'
command=nodes_for_submission=None


def save(state):
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(STATE)


def completed(stage):
    path=EXP/stage['done']
    if not path.exists():return False
    data=json.loads(path.read_text())
    if stage['kind']=='audit':
        if not data.get('weights_loaded') or data.get('real_task_updates')!=2 or not data.get('teacher_frozen'):raise ValueError('GPU audit incomplete')
    elif stage['kind']=='train':
        if data['successful_updates']!=data['config']['epochs']*100:raise ValueError('Training update budget incomplete')
    elif stage['kind']=='eval':
        if data['test_videos']!=211 or data['test_windows']!=792 or not data.get('gflops'):raise ValueError('Full test / actual compute record incomplete')
    return True


def tick(state,max_live):
    if state.get('cancelled_by_user'):return True
    queue=command('squeue','-u',os.environ['USER'],'-h','-o','%i|%T')
    if queue.returncode:state['queue_error']=queue.stderr;save(state);return False
    rows=dict(line.split('|',1) for line in queue.stdout.splitlines() if '|' in line);stages=state['stages']
    for name,stage in stages.items():
        try:
            if completed(stage):stage['status']='COMPLETED';continue
        except (KeyError,ValueError,FileNotFoundError,json.JSONDecodeError) as error:
            stage['status']='FAILED';stage['failure_note']=str(error);continue
        job=stage.get('job_id')
        if str(job) in rows:stage['status']=rows[str(job)]
        elif job:
            answer=command('sacct','-j',str(job),'-X','-n','-P','--format=JobIDRaw,State,ExitCode')
            found=next((x.split('|') for x in answer.stdout.splitlines() if x.split('|')[0]==str(job)),None)
            if found:stage.update(status='FAILED' if found[1].startswith('COMPLETED') else found[1].split()[0],exit_code=found[2])
        elif stage.get('status')!='FAILED':stage['status']='WAITING'
    live=sum(str(stage.get('job_id')) in rows for stage in stages.values())
    train_live=sum(str(s.get('job_id')) in rows and s['kind']=='train' for s in stages.values())
    priority=['audit_recovery_s','audit_recovery_b','audit_s','audit_b','eval_R01_interpolate_s','eval_R01_interpolate_b',
              'train_R03_cross_s','train_D02_amod50_s','train_S02_token48_s','train_J01_joint_s',
              'train_R03_cross_b','train_R02_tcn_s','train_R04_feature_only_s','train_D02_amod125_s']
    # Ready early milestones are mixed with independent training; no cross-route quality gate.
    early=[name for name,s in stages.items() if s['kind']=='eval' and s.get('epoch',99)<=5]
    priority=priority[:4]+early+priority[4:]+[name for name in stages if name not in priority and name not in early]
    if live<max_live and len(rows)<16:
        for name in priority:
            stage=stages[name]
            if stage.get('job_id') or stage.get('status') in ('COMPLETED','FAILED'):continue
            if stage['kind']=='train' and train_live>=max(1,max_live-2):continue
            if not all(stages[d].get('status')=='COMPLETED' for d in stage.get('dependencies',[])):continue
            if not all((EXP/p).exists() for p in stage.get('requires',[])):continue
            placement=nodes_for_submission()
            if placement is None:state['resource_note']='No verified4090 placement available';break
            eligible,excluded=placement;exclude=['--exclude='+','.join(excluded)] if excluded else []
            limit='12:00:00' if stage['kind']=='train' else '03:00:00'
            result=command('sbatch','--parsable','--partition=gpu','--qos=gpugpu','--nodes=1','--ntasks=1','--gres=gpu:1',
                 '--cpus-per-task=6','--time='+limit,*exclude,'--job-name=fpw-'+name,'--output='+str(EXP/'slurm/%j.log'),
                 str(EXP/'site/run_job.sh'),*stage['args'])
            if result.returncode==0:
                stage.update(job_id=int(result.stdout.strip().split(';')[0]),status='PENDING',eligible_nodes=eligible)
                stage.setdefault('attempts',[]).append(stage['job_id']);print(f'{name}: submitted {stage["job_id"]}',flush=True)
            else:stage['submission_error']=result.stderr
            break
    state.update(updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),account_jobs=len(rows),max_live_jobs=max_live,
                 latency_is_decision_gate=False,selection_objectives=['actual GFLOPs','best average mAP'])
    save(state)
    return all(stage.get('status')=='COMPLETED' for stage in stages.values())


def main():
    global command,nodes_for_submission
    p=argparse.ArgumentParser();p.add_argument('--plan',default=str(EXP/'plan.json'));p.add_argument('--emit-only',action='store_true')
    p.add_argument('--submit',action='store_true');p.add_argument('--max-live',type=int,default=8);p.add_argument('--once',action='store_true');args=p.parse_args()
    plan=json.loads(Path(args.plan).read_text())
    if args.emit_only or not args.submit:
        print(json.dumps(dict(stages=len(plan['stages']),max_live=args.max_live,submit=False,parallel_routes=True)));return
    import fcntl
    from tools.fidelity_dispatch import command,nodes_for_submission
    os.chdir(ROOT);EXP.mkdir(parents=True,exist_ok=True);(EXP/'slurm').mkdir(exist_ok=True)
    with (EXP/'dispatcher.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
        state=json.loads(STATE.read_text()) if STATE.exists() else dict(recipe=plan['recipe'],stages=plan['stages'],controller_pid=os.getpid())
        while True:
            if STATE.exists():state=json.loads(STATE.read_text())
            state['controller_pid']=os.getpid();done=tick(state,args.max_live)
            if done or args.once:break
            time.sleep(15)


if __name__=='__main__':main()
