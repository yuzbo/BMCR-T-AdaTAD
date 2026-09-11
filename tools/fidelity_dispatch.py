"""Fresh H65 correction experiment; only controls jobs recorded in this receipt."""
import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.fidelity_select import EPOCHS, save_selection
EXP=ROOT/'fidelity_20260911'
RUNS=Path(os.environ.get('H65_RUNS_DIR',EXP/'runs')).resolve()
STATE=EXP/'deployment.json'
ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING'}
NODES=['g0063','g0056','g0014','g0043','g0044','g0059','g0066','g0087']


def command(*args):
    return subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)


def save(state):
    temp=STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state,indent=2)+'\n')
    temp.replace(STATE)


def node_for_submission():
    available=[]
    for node in NODES:
        info=command('scontrol','show','node',node)
        if info.returncode or re.search(r'State=\S*(DOWN|DRAIN|FAIL)',info.stdout):
            continue
        allocation=re.search(r'AllocTRES=.*?gres/gpu=(\d+)',info.stdout)
        allocated=int(allocation[1]) if allocation else 0
        available.append((allocated,node))
    return min(available)[1] if available else None


def stages():
    result={}
    for b in ('s','b'):
        result[f'preflight_{b}']=dict(kind='preflight',dependencies=[],requires=[],
            done=f'preflight_{b}_h65/completed.json',
            args=['tools/full_train.py','--backbone',b,'--phase','joint','--variant','h65','--preflight'])
        result[f'{b}_warm']=dict(kind='train',dependencies=['preflight_s','preflight_b'],requires=[],
            done=f'{b}_warm/completed.json',
            args=['tools/full_train.py','--backbone',b,'--phase','warm','--variant','h65'])
        result[f'{b}_h65']=dict(kind='train',dependencies=[f'{b}_warm'],requires=[],
            done=f'{b}_h65/completed.json',
            args=['tools/full_train.py','--backbone',b,'--phase','joint','--variant','h65'])
        for epoch in EPOCHS:
            name=f'{b}_test_{epoch:02}'
            args=['tools/full_eval.py','--backbone',b,'--variant','h65','--milestone',str(epoch)]
            if epoch<60:
                args+=['--metrics-only']
            result[name]=dict(kind='test',dependencies=['preflight_s','preflight_b'],
                requires=[f'{b}_h65/epoch_{epoch-20:02}.pth'],
                done=f'{b}_h65_test_epoch_{epoch:02}/completed.json',args=args)
        result[f'{b}_selected_profile']=dict(kind='profile',dependencies=[f'{b}_h65']+[f'{b}_test_{e:02}' for e in EPOCHS],
            requires=[],done=None,args=None,backbone=b)
    return result


def artifact_complete(name, stage):
    if stage['kind']=='profile':
        selected=save_selection(RUNS,stage['backbone'])
        if not selected['all_candidates_evaluated']:
            return False
        best=selected['best']
        stage['selected_total_epoch']=best['total_epochs']
        stage['args']=['tools/full_eval.py','--backbone',stage['backbone'],'--variant','h65',
                       '--milestone',str(best['total_epochs']),'--profile-only']
        path=Path(best['folder'])/'profile.json'
        if not path.exists():
            return False
        p=json.loads(path.read_text())
        return p.get('primary_case')=='full' and not p['unresolved_matrix_ops']
    path=RUNS/stage['done']
    if not path.exists():
        return False
    d=json.loads(path.read_text())
    if stage['kind']=='test':
        if d['test_videos']!=211 or d['test_windows']!=792:
            raise ValueError(f'{name}: incomplete full test')
        if d['initialization'].get('fidelity_revision')!='lr_identity_crop_validity_v1':
            raise ValueError(f'{name}: wrong checkpoint recipe')
        if name.endswith('_60'):
            profile=json.loads(path.with_name('profile.json').read_text())
            if profile.get('primary_case')!='full' or profile['unresolved_matrix_ops']:
                raise ValueError(f'{name}: invalid terminal profile')
    else:
        if d.get('fidelity_revision')!='lr_identity_crop_validity_v1':
            raise ValueError(f'{name}: wrong training recipe')
        updates=2 if stage['kind']=='preflight' else 2000 if name.endswith('_warm') else 4000
        if d['successful_updates']!=updates:
            raise ValueError(f'{name}: incomplete update budget')
    return True


def tick(state):
    queue=command('squeue','-u',os.environ['USER'],'-h','-o','%i|%T')
    if queue.returncode:
        state['controller_error']=queue.stderr.strip(); save(state); return False
    rows=dict(line.split('|',1) for line in queue.stdout.splitlines() if '|' in line)
    table=state['stages']
    for name,stage in table.items():
        dependencies_done=all(table[d].get('status')=='COMPLETED' for d in stage['dependencies'])
        try:
            complete=(stage['kind']!='profile' or dependencies_done) and artifact_complete(name,stage)
        except (ValueError,KeyError,FileNotFoundError,json.JSONDecodeError) as error:
            stage['status']='FAILED'; stage['failure_note']=str(error); continue
        if complete:
            stage['status']='COMPLETED'; continue
        job=stage.get('job_id')
        if str(job) in rows:
            stage['status']=rows[str(job)]
        elif job:
            accounting=command('sacct','-j',str(job),'-X','-P','-n','--format=JobIDRaw,State,ExitCode')
            found=next((x.split('|') for x in accounting.stdout.splitlines() if x.split('|')[0]==str(job)),None)
            if found:
                stage['status']='FAILED' if found[1].startswith('COMPLETED') else found[1].split()[0]
                stage['exit_code']=found[2]
                stage['failure_note']='Required artifact missing; diagnose own log before retry.'
        elif stage.get('status')!='FAILED':
            stage['status']='WAITING'
    for b in ('s','b'):
        save_selection(RUNS,b)
    # Count live queue membership, including a job that has just written its
    # completion artifact but has not yet released its GPU allocation.
    live=[s for s in table.values() if str(s.get('job_id')) in rows]
    tests=sum(s['kind'] in ('test','profile') for s in live)
    training=len(live)-tests
    priority=[f'{b}_test_{e:02}' for e in EPOCHS for b in ('s','b')]
    priority += [f'{b}_selected_profile' for b in ('s','b')]
    priority += ['preflight_s','preflight_b','s_warm','b_warm','s_h65','b_h65']
    if len(rows)<16:
        for name in priority:
            stage=table[name]
            if stage.get('job_id') or stage.get('status') in ('COMPLETED','FAILED'):
                continue
            if not all(table[d].get('status')=='COMPLETED' for d in stage['dependencies']):
                continue
            if not all((RUNS/path).exists() for path in stage['requires']):
                continue
            is_test=stage['kind'] in ('test','profile')
            if (is_test and tests>=1) or (not is_test and training>=2):
                continue
            duration='02:00:00' if is_test else '01:00:00' if stage['kind']=='preflight' else '12:00:00'
            node=node_for_submission()
            if node is None:
                state['resource_note']='No healthy known4090 node was readable; waiting.'
                break
            submitted=command('sbatch','--parsable','--partition=gpu','--qos=gpugpu','--nodes=1',
                '--ntasks=1','--gres=gpu:1','--cpus-per-task=6','--time='+duration,'--nodelist='+node,
                '--job-name=h65-fix-'+name,'--output='+str(EXP/'slurm/%j.log'),
                str(EXP/'site/run_job.sh'),*stage['args'])
            if submitted.returncode==0:
                stage['job_id']=int(submitted.stdout.strip().split(';')[0])
                stage.setdefault('attempts',[]).append(stage['job_id'])
                stage['status']='PENDING'
                stage['node_requested']=node
                print(f'submitted {name}: {stage["job_id"]}',flush=True)
            else:
                stage['submission_error']=submitted.stderr.strip()
            break
    state['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    state['account_jobs_observed']=len(rows)
    save(state)
    return all(s.get('status')=='COMPLETED' for s in table.values())


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--once',action='store_true'); args=parser.parse_args()
    os.chdir(ROOT); EXP.mkdir(exist_ok=True); RUNS.mkdir(parents=True,exist_ok=True); (EXP/'slurm').mkdir(exist_ok=True)
    with (EXP/'dispatcher.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); lock.write(str(os.getpid())); lock.flush()
        state=json.loads(STATE.read_text()) if STATE.exists() else dict(stages=stages(),
            protocol='200 train /211 test,20+40 epochs,H65 corrections only; test-peak joint EMA25..60 + terminal reporting')
        while True:
            if STATE.exists(): state=json.loads(STATE.read_text())
            state['controller_pid']=os.getpid()
            done=tick(state)
            if done:
                result=command(sys.executable,'tools/fidelity_compare.py')
                state['comparison']='COMPLETED' if result.returncode==0 else result.stderr.strip()
                save(state)
            if done or args.once: break
            time.sleep(30)


if __name__=='__main__': main()
