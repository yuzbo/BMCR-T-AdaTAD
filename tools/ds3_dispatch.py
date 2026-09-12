"""DS3-only Slurm stages; records failures, never retries a failed job blindly."""
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
from tools.ds3_select import EPOCHS,save_selection
EXP=ROOT/'ds3_20260912';RUNS=Path(os.environ.get('DS3_RUNS_DIR',EXP/'runs'))
STATE=EXP/'deployment.json'
FINAL_POLICIES=('PONLY','T24U','D8','DAD','S75','S50','F000','F001','F011','F110','F111')


def stages():
    result={'preflight':dict(kind='preflight',dependencies=[],requires=[],done='preflight_b/completed.json',
             args=['tools/ds3_preflight.py','--backbone','both'])}
    for b in ('s','b'):
        for policy in ('D768G','D768L','Z16','Z24','Z36','ZR24'):
            result[f'{b}_{policy}']=dict(kind='test',dependencies=['preflight'],requires=[],done=f'{b}_{policy}/completed.json',
                args=['tools/ds3_eval.py','--backbone',b,'--policy',policy])
        result[f'{b}_pilot']=dict(kind='pilot',dependencies=['preflight',f'{b}_D768G',f'{b}_D768L',f'{b}_Z24'],requires=[],
            done=f'{b}_d1/pilot_completed.json',args=['tools/ds3_train.py','--backbone',b,'--pilot'])
        result[f'{b}_train']=dict(kind='train',dependencies=[f'{b}_pilot'],requires=[],done=f'{b}_d1/completed.json',
            args=['tools/ds3_train.py','--backbone',b])
        for epoch in EPOCHS:
            arguments=['tools/ds3_eval.py','--backbone',b,'--policy','T24A','--epoch',str(epoch)]
            if epoch not in (60,80):arguments.append('--metrics-only')
            result[f'{b}_T24A_{epoch:02}']=dict(kind='test',dependencies=['preflight'],requires=[f'{b}_d1/epoch_{epoch:02}.pth'],
                done=f'{b}_T24A_epoch_{epoch:02}/completed.json',args=arguments)
        dependencies=[f'{b}_train']+[f'{b}_T24A_{epoch:02}' for epoch in EPOCHS]
        result[f'{b}_selected_profile']=dict(kind='selected_profile',backbone=b,dependencies=dependencies,requires=[],args=None,done=None)
        for policy in FINAL_POLICIES:
            result[f'{b}_selected_{policy}']=dict(kind='selected_test',backbone=b,policy=policy,dependencies=dependencies,
                requires=[],args=None,done=None)
    return result


def save(state):
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(STATE)


def complete(stage):
    if stage['kind'] in ('selected_profile','selected_test'):
        selection=save_selection(stage['backbone'],RUNS)
        if not selection['all_candidates_evaluated']:return False
        epoch=selection['best']['epoch'];policy=stage.get('policy','T24A');stage['selected_epoch']=epoch
        stage['args']=['tools/ds3_eval.py','--backbone',stage['backbone'],'--policy',policy,'--epoch',str(epoch)]
        if stage['kind']=='selected_profile':stage['args'].append('--profile-only')
        stage['done']=f'{stage["backbone"]}_{policy}_epoch_{epoch:02}/'+('profile.json' if stage['kind']=='selected_profile' else 'completed.json')
    if not stage['done'] or not (RUNS/stage['done']).exists():return False
    value=json.loads((RUNS/stage['done']).read_text())
    if stage['kind']=='selected_profile':
        return value['primary_case']=='full' and all(not case['unresolved_matrix_ops'] for case in value['cases'].values())
    if stage['kind']=='preflight':
        values=[json.loads((RUNS/f'preflight_{b}/completed.json').read_text()) for b in ('s','b')]
        return all(x['successful_updates']==2 and x['teacher_parameters_and_buffers_unchanged'] and x['strict_ema_roundtrip'] for x in values)
    if value.get('recipe')!='ds3_ltia8_d1_80_v1':raise ValueError('wrong DS3 recipe')
    if stage['kind'] in ('test','selected_test'):
        if value['test_videos']!=211 or value['test_windows']!=792:raise ValueError('incomplete test set')
        return True
    updates=100 if stage['kind']=='pilot' else 8000
    if value['successful_updates']!=updates:raise ValueError('incomplete DS3 training budget')
    return True


def tick(state):
    queue=command('squeue','-u',os.environ['USER'],'-h','-o','%i|%T')
    if queue.returncode:state['controller_error']=queue.stderr;save(state);return False
    rows=dict(line.split('|',1) for line in queue.stdout.splitlines() if '|' in line);table=state['stages']
    for name,stage in table.items():
        try:
            if complete(stage):stage['status']='COMPLETED';continue
        except (ValueError,KeyError,FileNotFoundError,json.JSONDecodeError) as error:
            stage['status']='FAILED';stage['failure_note']=str(error);continue
        job=stage.get('job_id')
        if str(job) in rows:stage['status']=rows[str(job)]
        elif job:
            accounting=command('sacct','-j',str(job),'-X','-P','-n','--format=JobIDRaw,State,ExitCode')
            found=next((line.split('|') for line in accounting.stdout.splitlines() if line.split('|')[0]==str(job)),None)
            if found:
                stage['status']='FAILED' if found[1].startswith('COMPLETED') else found[1].split()[0]
                stage['exit_code']=found[2];stage['failure_note']='Required artifact missing; inspect this job log before retry.'
        elif stage.get('status')!='FAILED':stage['status']='WAITING'
    for b in ('s','b'):save_selection(b,RUNS)
    live=[stage for stage in table.values() if str(stage.get('job_id')) in rows]
    is_test=lambda stage:stage['kind'] in ('test','selected_test','selected_profile')
    tests=sum(is_test(stage) for stage in live);training=len(live)-tests
    priority=[f'{b}_T24A_{epoch:02}' for epoch in EPOCHS for b in ('s','b')]
    priority+=['preflight','s_train','b_train','s_pilot','b_pilot']
    priority+=[f'{b}_{policy}' for b in ('s','b') for policy in ('D768G','D768L','Z24','Z16','Z36','ZR24')]
    priority+=[f'{b}_selected_profile' for b in ('s','b')]
    priority+=[f'{b}_selected_{policy}' for b in ('s','b') for policy in FINAL_POLICIES]
    if len(rows)<16:
        for name in priority:
            stage=table[name]
            if stage.get('job_id') or stage.get('status') in ('COMPLETED','FAILED'):continue
            if not all(table[d].get('status')=='COMPLETED' for d in stage['dependencies']):continue
            if not all((RUNS/path).exists() for path in stage['requires']):continue
            if (is_test(stage) and tests>=1) or (not is_test(stage) and training>=2):continue
            placement=nodes_for_submission()
            if placement is None:state['resource_note']='Waiting for verified4090 pool inventory';break
            eligible,excluded=placement;node_args=['--exclude='+','.join(excluded)] if excluded else []
            duration='24:00:00' if stage['kind']=='train' else '02:00:00'
            submit=command('sbatch','--parsable','--partition=gpu','--qos=gpugpu','--nodes=1','--ntasks=1',
                '--gres=gpu:1','--cpus-per-task=6','--time='+duration,*node_args,'--job-name=ds3-'+name,
                '--output='+str(EXP/'slurm/%j.log'),str(EXP/'site/run_job.sh'),*stage['args'])
            if submit.returncode==0:
                stage['job_id']=int(submit.stdout.strip().split(';')[0]);stage['status']='PENDING'
                stage.setdefault('attempts',[]).append(stage['job_id']);stage['eligible_nodes']=eligible
                print(f'submitted {name}: {stage["job_id"]}',flush=True)
            else:stage['submission_error']=submit.stderr.strip()
            break
    state.update(updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),account_jobs_observed=len(rows))
    save(state);return all(stage.get('status')=='COMPLETED' for stage in table.values())


def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    os.chdir(ROOT);EXP.mkdir(exist_ok=True);RUNS.mkdir(parents=True,exist_ok=True);(EXP/'slurm').mkdir(exist_ok=True)
    with (EXP/'dispatcher.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
        state=json.loads(STATE.read_text()) if STATE.exists() else dict(stages=stages(),
            protocol='DS3-L D1 dense teacher frozen,all200 train/211 test,single3407,80epochs8000updates,T24A test-peak5..80')
        while True:
            if STATE.exists():state=json.loads(STATE.read_text())
            state['controller_pid']=os.getpid();done=tick(state)
            if done or a.once:break
            time.sleep(30)


if __name__=='__main__':main()
