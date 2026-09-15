"""Execute the user's scoped retirement of superseded H65 courses on the 4090 host."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path('/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/paper_20260913')
EXP=ROOT/'research/paper'
KEEP={'train_review5485_full_v2_b_seed42','train_thumos_s_point_uniform_seed42','train_thumos_b_point_uniform_seed42'}
reason='User 2026-09-15: retire low-value old H65 courses after Fast-Track redesign; preserve artifacts'

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def run(*args):
    return subprocess.run(args,text=True,capture_output=True,check=True)

def main():
    stamp=time.strftime('%Y%m%d_%H%M%S')
    archive=EXP/'retired_fasttrack'/stamp;archive.mkdir(parents=True)
    deploy=EXP/'deployment.json';planfile=EXP/'plan.json'
    state=json.loads(deploy.read_text());pid=state['controller_pid']
    cmdline=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if str(ROOT/'tools/paper_dispatch.py') not in cmdline:
        raise RuntimeError('Controller process is not the expected H65 owner')
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        if not (Path('/proc')/str(pid)).exists():break
        time.sleep(.1)
    else:raise RuntimeError('Owner did not stop; no Slurm cancellation performed')
    state=json.loads(deploy.read_text());plan=json.loads(planfile.read_text())
    shutil.copy2(deploy,archive/'deployment_before.json');shutil.copy2(planfile,archive/'plan_before.json')
    script=ROOT/'tools/paper_dispatch.py'
    shutil.copy2(script,archive/'paper_dispatch_before.py')
    source=script.read_text()
    needle="    for name,stage in stages.items():\n        try:\n"
    replacement="    for name,stage in stages.items():\n        if stage.get('cancelled_by_user'):\n            stage['status']='CANCELLED';continue\n        try:\n"
    if needle not in source:raise RuntimeError('Unexpected owner implementation')
    source=source.replace(needle,replacement,1)
    source=source.replace("if stage.get('job_id') or stage.get('cpu_pid') or stage.get('status') in ('COMPLETED','FAILED'):continue",
        "if stage.get('cancelled_by_user') or stage.get('job_id') or stage.get('cpu_pid') or stage.get('status') in ('COMPLETED','FAILED'):continue",1)
    source=source.replace("return all(s.get('status')=='COMPLETED' for s in stages.values())",
        "return all(s.get('status') in ('COMPLETED','CANCELLED') for s in stages.values())",1)
    compile(source,str(script),'exec');script.write_text(source)
    queue={line.split('|')[0]:line for line in run('squeue','-u','sczc063','-h','-o','%i|%j|%T').stdout.splitlines()}
    cancelled=[];active=[];kept=[]
    for name,stage in state['stages'].items():
        done=EXP/stage['done']
        if stage.get('status')=='COMPLETED' or done.exists():continue
        if name in KEEP:
            kept.append(dict(stage=name,job_id=stage.get('job_id'),status=stage.get('status')));continue
        jid=str(stage.get('job_id',''))
        if jid in queue:
            detail=run('scontrol','show','job',jid,'-o').stdout
            if str(ROOT.parent) not in detail or 'yuzibo' not in detail:
                raise RuntimeError('Live job does not belong to H65: '+jid)
            active.append(dict(job_id=jid,stage=name,kind=stage['kind'],scheduler_before=detail))
        stage.update(cancelled_by_user=True,status='CANCELLED',cancellation_reason=reason)
        if jid:stage['cancelled_job_id']=stage.pop('job_id')
        stage.pop('cpu_pid',None)
        cancelled.append(name)
        if name in plan['stages']:
            plan['stages'][name].update(cancelled_by_user=True,status='CANCELLED',cancellation_reason=reason)
    state['legacy_disabled_by_fasttrack']=True
    write(deploy,state);write(planfile,plan)
    # Other members' jobs are absent from this owner manifest and never targeted.
    for item in active:
        if item['kind']=='train':
            result=subprocess.run(['scancel','--signal=USR1','--batch',item['job_id']],capture_output=True,text=True)
            item['checkpoint_signal_returncode']=result.returncode
        else:
            result=subprocess.run(['scancel',item['job_id']],capture_output=True,text=True)
            item['cancel_returncode']=result.returncode
    receipt=dict(user_authorization=reason,owner_stopped=pid,owner_command=cmdline,
        cancelled_stages=cancelled,active_jobs=active,retained_courses=kept,archive=str(archive),
        artifacts_deleted=False,legacy_inheritance_disabled=True,status='saving owned training checkpoints')
    write(archive/'receipt.json',receipt)
    print(json.dumps(dict(receipt=str(archive/'receipt.json'),cancelled_stages=len(cancelled),jobs=[x['job_id'] for x in active],retained=kept)),flush=True)
    time.sleep(45)
    remaining=set(run('squeue','-u','sczc063','-h','-o','%i').stdout.split())
    for item in active:
        if item['job_id'] in remaining:
            result=subprocess.run(['scancel',item['job_id']],capture_output=True,text=True)
            item['cancel_returncode']=result.returncode
    log=(EXP/'dispatcher_fasttrack_retirement.log').open('a')
    owner=subprocess.Popen([sys.executable,'-u',str(script),'--submit','--max-live','5'],cwd=ROOT,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    receipt.update(status='retirement applied',new_owner_pid=owner.pid,active_jobs=active)
    write(archive/'receipt.json',receipt);write(EXP/'FASTTRACK_RETIREMENT.json',receipt)
    print(json.dumps(dict(new_owner_pid=owner.pid,receipt=str(EXP/'FASTTRACK_RETIREMENT.json'))),flush=True)

if __name__=='__main__':main()
