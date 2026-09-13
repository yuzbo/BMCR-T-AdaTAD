"""One-time conversion of our pending checks to same-allocation full courses."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));EXP=ROOT/'research/paper'
from h65.paper.runtime import json_write
from tools.fidelity_dispatch import command


def main():
    state=json.loads((EXP/'deployment.json').read_text());plan=json.loads((EXP/'plan.json').read_text())
    pid=state['controller_pid'];proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists():
        if b'paper_dispatch.py' not in proc.read_bytes():raise RuntimeError('Unexpected controller PID')
        os.kill(pid,signal.SIGTERM)
        for _ in range(25):
            if not proc.exists():break
            time.sleep(.2)
        else:raise RuntimeError('Controller remains active')
    def queue():
        result=command('squeue','-u',os.environ['USER'],'-h','-o','%i|%T')
        if result.returncode:raise RuntimeError(result.stderr)
        return dict(x.split('|',1) for x in result.stdout.splitlines() if '|' in x)
    rows=queue();changes=[];pending=[]
    for name,new in plan['stages'].items():
        old=state['stages'].get(name,{})
        if new['kind']=='inline_preflight' and old.get('kind')=='preflight' and rows.get(str(old.get('job_id')))=='PENDING':pending.append(str(old['job_id']))
    if pending:
        result=command('scancel','--state=PENDING',*pending)
        if result.returncode:raise RuntimeError(result.stderr)
    rows=queue();running_checks=set()
    for name,new in plan['stages'].items():
        old=state['stages'].get(name,{})
        if new['kind']=='inline_preflight' and old.get('kind')=='preflight' and rows.get(str(old.get('job_id'))) in ('RUNNING','CONFIGURING'):
            running_checks.add(name);continue
        merged=dict(old);merged.update(new)
        if str(old.get('job_id')) in pending:
            merged.pop('job_id',None);merged['status']='WAITING_INLINE'
            changes.append(dict(stage=name,former_job=old['job_id'],new_course=new['runs_with']))
        state['stages'][name]=merged
    for check in running_checks:
        ident=state['stages'][check]['config_id'];train=state['stages']['train_'+ident]
        train['dependencies']=[check]
    state['source_revision']=(ROOT/'source_revision.txt').read_text().strip();json_write(EXP/'deployment.json',state)
    record=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),converted_pending_checks=changes,running_checks_preserved=sorted(running_checks))
    json_write(EXP/'integrated_course_receipt.json',record)
    log=(EXP/'dispatcher.log').open('a');legacy=ROOT.parent/'fpw_3d_20260913/research/frame/deployment.json'
    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/paper_dispatch.py'),'--submit','--max-live','10','--legacy-receipt',str(legacy),'--inherit-legacy'],
        cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record['controller_pid']=child.pid;json_write(EXP/'integrated_course_receipt.json',record);print(json.dumps(record))


if __name__=='__main__':main()
