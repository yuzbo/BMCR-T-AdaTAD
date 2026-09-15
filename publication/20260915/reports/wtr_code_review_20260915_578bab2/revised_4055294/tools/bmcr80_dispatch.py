"""Own corrected BMCR80 stages; no warm retraining and no DS3 jobs."""
import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.fidelity_dispatch import command,nodes_for_submission
from tools.bmcr80_report import EPOCHS,RECIPE,save_selection,build_report

EXP=ROOT/'bmcr80_20260913';RUNS=EXP/'runs';STATE=EXP/'deployment.json'


def stages():
    table={}
    for b in ('s','b'):
        table[f'{b}_audit']=dict(kind='audit',backbone=b,dependencies=[],requires=[f'{b}_warm/terminal.pth'],
              done=f'{b}_audit/completed.json',args=['tools/full_audit.py','--backbone',b,'--total-epochs','80'])
        table[f'{b}_preflight']=dict(kind='preflight',backbone=b,dependencies=[f'{b}_audit'],requires=[],
              done=f'preflight_{b}_bmcr/completed.json',args=['tools/full_train.py','--backbone',b,'--phase','joint','--variant','bmcr','--total-epochs','80','--preflight'])
        table[f'{b}_train']=dict(kind='train',backbone=b,dependencies=['s_preflight','b_preflight'],requires=[],
              done=f'{b}_bmcr/completed.json',args=['tools/full_train.py','--backbone',b,'--phase','joint','--variant','bmcr','--total-epochs','80'])
        for e in EPOCHS:
            table[f'{b}_test_{e:02}']=dict(kind='test',backbone=b,epoch=e,dependencies=['s_preflight','b_preflight'],
                  requires=[f'{b}_bmcr/epoch_{e-20:02}.pth'],done=f'{b}_bmcr_test_epoch_{e:02}/completed.json',
                  args=['tools/full_eval.py','--backbone',b,'--variant','bmcr','--total-epochs','80','--milestone',str(e),'--metrics-only'])
        for target in (60,80,'selected'):
            deps=[f'{b}_train']+[f'{b}_test_{e:02}' for e in EPOCHS] if target=='selected' else [f'{b}_test_{target:02}']
            table[f'{b}_profile_{target}']=dict(kind='profile',backbone=b,target=target,dependencies=deps,requires=[],done=None,args=None)
    return table


def save(state):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)


def artifact_complete(name,stage,table):
    if stage['kind']=='profile':
        if not all(table[d].get('status')=='COMPLETED' for d in stage['dependencies']):return False
        b=stage['backbone'];e=stage['target']
        if e=='selected':
            selection=save_selection(RUNS,b)
            if not selection['all_candidates_evaluated']:return False
            e=selection['best']['total_epochs']
        stage['selected_total_epoch']=e
        stage['args']=['tools/full_eval.py','--backbone',b,'--variant','bmcr','--total-epochs','80','--milestone',str(e),'--profile-only']
        stage['done']=f'{b}_bmcr_test_epoch_{e:02}/profile.json'
    path=RUNS/stage['done']
    if not path.exists():return False
    data=json.loads(path.read_text());kind=stage['kind']
    if kind=='profile':
        if data.get('primary_case')!='full' or data['unresolved_matrix_ops']:
            raise ValueError('Incomplete primary full-window profile')
    elif kind=='test':
        if data['test_videos']!=211 or data['test_windows']!=792 or data['initialization'].get('recipe')!=RECIPE:
            raise ValueError('Wrong or incomplete BMCR80 test')
        if data['initialization']['total_epochs']!=stage['epoch']:raise ValueError('Wrong milestone')
    else:
        if data.get('recipe')!=RECIPE or data.get('backbone')!=stage['backbone']:raise ValueError('Wrong BMCR80 receipt')
        if kind=='audit':
            if data['windows']!=128 or data['fit_swaps']<2 or data['holdout_swaps']<2:raise ValueError('Incomplete training-only audit')
            if not (path.parent/'scales.json').exists():return False
        elif kind=='preflight':
            if data['successful_updates']!=2 or not data['real_joint_initialization'] or not data['strict_reload']:
                raise ValueError('Preflight did not exercise the actual joint initialization')
        elif data['successful_updates']!=6000 or data['completed_epochs']!=60:
            raise ValueError('BMCR80 joint budget incomplete')
    return True


def tick(state):
    if state.get('cancelled_by_user'):return True
    queue=command('squeue','-u',os.environ['USER'],'-h','-o','%i|%T')
    if queue.returncode:state['controller_error']=queue.stderr;save(state);return False
    rows=dict(x.split('|',1) for x in queue.stdout.splitlines() if '|' in x);table=state['stages']
    for name,stage in table.items():
        try:
            if artifact_complete(name,stage,table):stage['status']='COMPLETED';continue
        except (KeyError,ValueError,FileNotFoundError,json.JSONDecodeError) as error:
            stage['status']='FAILED';stage['failure_note']=str(error);continue
        job=stage.get('job_id')
        if str(job) in rows:stage['status']=rows[str(job)]
        elif job:
            result=command('sacct','-j',str(job),'-X','-P','-n','--format=JobIDRaw,State,ExitCode')
            found=next((x.split('|') for x in result.stdout.splitlines() if x.split('|')[0]==str(job)),None)
            if found:
                stage['status']='FAILED' if found[1].startswith('COMPLETED') else found[1].split()[0]
                stage['exit_code']=found[2];stage['failure_note']='Required artifact missing; diagnose before any retry.'
        elif stage.get('status')!='FAILED':stage['status']='WAITING'
    live=[v for v in table.values() if str(v.get('job_id')) in rows]
    test_kind=lambda v:v['kind'] in ('test','profile')
    tests=sum(test_kind(v) for v in live);training=len(live)-tests
    priority=[f'{b}_test_{e:02}' for e in EPOCHS for b in ('s','b')]
    priority += [f'{b}_profile_{e}' for e in (60,80,'selected') for b in ('s','b')]
    priority += ['s_audit','b_audit','s_preflight','b_preflight','s_train','b_train']
    if len(rows)<16:
        for name in priority:
            stage=table[name]
            if stage.get('job_id') or stage.get('status') in ('COMPLETED','FAILED'):continue
            if not all(table[d].get('status')=='COMPLETED' for d in stage['dependencies']):continue
            if not all((RUNS/x).exists() for x in stage['requires']):continue
            if (test_kind(stage) and tests>=1) or (not test_kind(stage) and training>=2):continue
            if stage['args'] is None:continue
            placement=nodes_for_submission()
            if placement is None:state['resource_note']='Waiting for verified4090 pool inventory';break
            eligible,excluded=placement;node_args=['--exclude='+','.join(excluded)] if excluded else []
            limit='24:00:00' if stage['kind']=='train' else '01:00:00' if stage['kind']=='preflight' else '02:00:00'
            submitted=command('sbatch','--parsable','--partition=gpu','--qos=gpugpu','--nodes=1','--ntasks=1',
                 '--gres=gpu:1','--cpus-per-task=6','--time='+limit,*node_args,'--job-name=bmcr80-'+name,
                 '--output='+str(EXP/'slurm/%j.log'),str(EXP/'site/run_job.sh'),*stage['args'])
            if submitted.returncode==0:
                stage['job_id']=int(submitted.stdout.strip().split(';')[0]);stage['status']='PENDING'
                stage.setdefault('attempts',[]).append(stage['job_id']);stage['eligible_nodes']=eligible
                print(f'submitted {name}: {stage["job_id"]}',flush=True)
            else:stage['submission_error']=submitted.stderr.strip()
            break
    state.update(updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),account_jobs_observed=len(rows))
    save(state)
    signature=[(k,v.get('status')) for k,v in table.items()]
    for b in ('s','b'):
        p=RUNS/f'{b}_bmcr/progress.json';signature.append((b,p.read_text() if p.exists() else None))
    if signature!=getattr(tick,'last_report',None):
        build_report(EXP,RUNS,state);tick.last_report=signature
    return all(v.get('status')=='COMPLETED' for v in table.values())


def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true',help='Run one scheduling tick, which may submit a job');args=p.parse_args()
    os.chdir(ROOT);EXP.mkdir(exist_ok=True);RUNS.mkdir(exist_ok=True);(EXP/'slurm').mkdir(exist_ok=True)
    with (EXP/'dispatcher.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
        state=json.loads(STATE.read_text()) if STATE.exists() else dict(recipe=RECIPE,stages=stages(),
                    protocol='BMCR S/B corrected warm20 reuse + joint60;200/211;seed3407;EMA full-test25..80;peak/60/80')
        if state.get('recipe')!=RECIPE:raise RuntimeError('Wrong deployment receipt')
        while True:
            if STATE.exists():state=json.loads(STATE.read_text())
            state['controller_pid']=os.getpid();done=tick(state)
            if done or args.once:break
            time.sleep(30)


if __name__=='__main__':main()
