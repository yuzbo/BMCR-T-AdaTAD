"""Receipt-based parallel submission with planned checkpoint continuation."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));EXP=ROOT/'research/paper'
from h65.paper.runtime import json_write
LEGACY=None
CPU_CHILDREN={}


def completed(stage):
    path=EXP/stage['done']
    if not path.exists():return False
    data=json.loads(path.read_text())
    if stage['kind'] in ('preflight','inline_preflight'):
        if data.get('real_task_updates')!=2 or not data.get('no_gt_inference') or not data.get('strict_state_reload'):raise ValueError('Incomplete real-update preflight')
    elif stage['kind']=='train':
        if data['successful_updates']!=data['expected_updates'] or data['completed_epochs']!=data['config']['epochs']:raise ValueError('Incomplete full course')
    elif stage['kind']=='eval':
        wanted=211 if data['config']['dataset']=='thumos' else 4728
        if data['test_videos']!=wanted or not data.get('gflops'):raise ValueError('Incomplete full test or real compute ledger')
    elif stage['kind']=='calibrate':
        if not data.get('video_disjoint_router_oof') or not data.get('actual_interventions'):raise ValueError('Missing real OOF intervention calibration')
    elif stage['kind']=='diagnostics':
        if not data.get('actual_traces') or not data.get('cases'):raise ValueError('Missing executed diagnostic cases')
    elif stage['kind']=='analysis':
        if data.get('paired_videos')!=211 or not data.get('official_AP_reproduced'):raise ValueError('Paired AP reproduction failed')
    return True


def blocked_assets(stage,resources):
    blocked=[]
    for key in stage.get('assets',[]):
        if key.startswith('asset:') and key[6:] not in resources.get('verified_downloads',{}):blocked.append(key)
        elif key.startswith('mae:') and key[4:] not in resources.get('decoder_pretrain',{}):blocked.append(key)
        elif key=='data:anet':
            path=Path(resources['datasets']['anet']['ready_file'])
            if not path.exists() or json.loads(path.read_text()).get('status')!='READY':blocked.append(key)
    return blocked


def tick(state,max_live,legacy_path=None):
    from tools.fidelity_dispatch import command,nodes_for_submission
    query=command('squeue','-u',os.environ['USER'],'-h','-o','%i|%T')
    if query.returncode:state['queue_error']=query.stderr;return False
    queue=dict(line.split('|',1) for line in query.stdout.splitlines() if '|' in line)
    resources=json.loads((EXP/'resources.local.json').read_text());stages=state['stages']
    gone=[]
    for name,stage in stages.items():
        try:
            if completed(stage):stage['status']='COMPLETED';continue
        except (ValueError,KeyError,json.JSONDecodeError) as error:stage.update(status='FAILED',failure=str(error));continue
        if stage['kind']=='inline_preflight':
            stage['status']='RUNNING_INLINE' if stages[stage['runs_with']].get('status')=='RUNNING' else 'WAITING_INLINE';continue
        if stage.get('cpu_pid'):
            pid=stage['cpu_pid'];child=CPU_CHILDREN.get(pid)
            alive=child.poll() is None if child else Path(f'/proc/{pid}').exists()
            stage['status']='RUNNING_CPU' if alive else 'FAILED';continue
        jid=stage.get('job_id')
        if str(jid) in queue:stage['status']=queue[str(jid)]
        elif jid:gone.append((name,str(jid)))
        elif stage.get('status')!='FAILED':stage['status']='WAITING'
    if gone:
        status=command('sacct','-j',','.join(j for _,j in gone),'-X','-n','-P','--format=JobIDRaw,State,ExitCode')
        accounts={r[0]:r[1:] for line in status.stdout.splitlines() if len(r:=line.split('|'))>=3}
        for name,jid in gone:
            if jid not in accounts:continue
            stage=stages[name];job_state,exit_code=accounts[jid][:2]
            if job_state.split()[0] in ('PENDING','RUNNING','COMPLETING','CONFIGURING'):continue
            if exit_code=='75:0' and stage['kind']=='train' and (ROOT/'research/paper/runs'/stage['config_id']/'latest.pth').exists():
                stage.setdefault('continuations',[]).append(dict(job_id=int(jid),reason='planned_checkpoint_time_slice'))
                stage.pop('job_id');stage['status']='WAITING'
            else:stage.update(status='FAILED',scheduler_state=job_state,exit_code=exit_code)
    legacy=None;legacy_stages={}
    if legacy_path:
        legacy_path=Path(legacy_path);legacy=json.loads(legacy_path.read_text());legacy_stages=legacy['stages']
        if LEGACY is not None:LEGACY.tick(legacy,0)
        for stage in legacy_stages.values():
            if str(stage.get('job_id')) in queue:stage['status']=queue[str(stage['job_id'])]
    # Count only this program and its explicitly inherited legacy allocations.
    live=sum(str(s.get('job_id')) in queue for s in [*stages.values(),*legacy_stages.values()])
    train_live=sum(str(s.get('job_id')) in queue and s['kind']=='train' for s in [*stages.values(),*legacy_stages.values()])
    ready=[]
    for name,stage in stages.items():
        if stage['kind']=='inline_preflight':continue
        if stage.get('job_id') or stage.get('cpu_pid') or stage.get('status') in ('COMPLETED','FAILED'):continue
        stage['waiting_assets']=blocked_assets(stage,resources)
        stage['waiting_dependencies']=[x for x in stage.get('dependencies',[]) if stages[x].get('status')!='COMPLETED']
        stage['waiting_files']=[x for x in stage.get('requires',[]) if not (ROOT/x).exists()]
        if not any(stage[x] for x in ('waiting_assets','waiting_dependencies','waiting_files')):
            if stage['kind']!='train' or train_live<max(1,max_live-2):ready.append((stage['priority'],name,stage))
    if not any(s.get('status')=='RUNNING_CPU' for s in stages.values()):
        analysis=next(((name,stage) for _,name,stage in ready if stage['kind']=='analysis'),None)
        if analysis:
            name,stage=analysis;log=(EXP/'slurm'/f'{name}.cpu.log').open('a')
            child=subprocess.Popen(['nice','-n','10',sys.executable,'-u',*stage['args']],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close();CPU_CHILDREN[child.pid]=child;stage.update(cpu_pid=child.pid,status='RUNNING_CPU')
    ready=[row for row in ready if row[2]['kind']!='analysis']
    slots=min(max_live-live,16-len(queue))
    for _,name,stage in sorted(ready,key=lambda x:(x[0],x[1]))[:max(0,slots)]:
        if stage['kind']=='train' and train_live>=max(1,max_live-2):continue
        placement=nodes_for_submission()
        if placement is None:state['resource_note']='No verified 4090 partition placement';break
        eligible,excluded=placement
        result=command('sbatch','--parsable','--partition=gpu','--qos=gpugpu','--nodes=1','--ntasks=1','--gres=gpu:1',
            '--cpus-per-task=6','--time='+('12:00:00' if stage['kind']=='train' else '06:00:00'),
            '--signal=B:USR1@600',*(['--exclude='+','.join(excluded)] if excluded else []),
            '--job-name=paper-'+name[:95],'--output='+str(EXP/'slurm/%j.log'),str(EXP/'site/run_job.sh'),*stage['args'])
        if result.returncode:stage['submission_error']=result.stderr;break
        jid=int(result.stdout.strip().split(';')[0]);stage.update(job_id=jid,status='PENDING',eligible_nodes=eligible)
        stage.setdefault('attempts',[]).append(dict(job_id=jid,submitted_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),source_revision=state['source_revision']))
        train_live+=stage['kind']=='train';print(f'{name}: submitted {jid}',flush=True)
    if not ready and live<max_live and train_live<max(1,max_live-2) and LEGACY is not None:
        # A single legacy allocation can use an otherwise idle program slot.
        LEGACY.tick(legacy,1)
    state.update(updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),account_jobs=len(queue),max_live=max_live,legacy_live=sum(str(s.get('job_id')) in queue for s in legacy_stages.values()))
    return all(s.get('status')=='COMPLETED' for s in stages.values())


def main(args):
    global LEGACY
    plan=json.loads((EXP/'plan.json').read_text())
    if not args.submit:print(json.dumps(dict(stages=len(plan['stages']),submit=False,performance_gates=False)));return
    import fcntl
    if args.inherit_legacy:
        if not args.legacy_receipt:raise ValueError('Explicit legacy receipt required')
        old_root=Path(args.legacy_receipt).resolve().parents[2]
        spec=importlib.util.spec_from_file_location('paper_legacy_dispatch',old_root/'tools/frame_dispatch.py')
        LEGACY=importlib.util.module_from_spec(spec);spec.loader.exec_module(LEGACY)
        from tools.fidelity_dispatch import command,nodes_for_submission
        LEGACY.command=command;LEGACY.nodes_for_submission=nodes_for_submission
    os.chdir(ROOT);(EXP/'slurm').mkdir(exist_ok=True)
    with (EXP/'dispatcher.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
        while True:
            resources=json.loads((EXP/'resources.local.json').read_text())
            finished_transfer=any(v['status']=='transfer_pending' and Path(v['checkpoint']).exists() and Path(v['checkpoint']).stat().st_size==v['expected_bytes']
                                  for v in resources.get('pending_downloads',{}).values())
            if finished_transfer:
                subprocess.run([sys.executable,str(ROOT/'tools/paper_assets.py'),'--site-root',str(ROOT.parent)],check=True)
            path=EXP/'deployment.json'
            state=json.loads(path.read_text()) if path.exists() else dict(recipe=plan['recipe'],stages=plan['stages'])
            plan=json.loads((EXP/'plan.json').read_text())
            for name,stage in plan['stages'].items():state['stages'].setdefault(name,stage)
            state.update(controller_pid=os.getpid(),source_revision=(ROOT/'source_revision.txt').read_text().strip())
            done=tick(state,args.max_live,args.legacy_receipt);json_write(path,state)
            if done or args.once:break
            time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');p.add_argument('--once',action='store_true')
    p.add_argument('--max-live',type=int,default=10);p.add_argument('--legacy-receipt');p.add_argument('--inherit-legacy',action='store_true');main(p.parse_args())
