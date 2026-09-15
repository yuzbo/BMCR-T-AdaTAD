"""Move only the old FPW controller and its pending jobs into the paper queue."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));EXP=ROOT/'research/paper'
from h65.paper.runtime import json_write
from tools.fidelity_dispatch import command


def main():
    old=ROOT.parent/'fpw_3d_20260913';receipt=old/'research/frame/deployment.json'
    state=json.loads(receipt.read_text());pid=state['controller_pid'];proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists():
        if b'frame_dispatch.py' not in proc.read_bytes():raise RuntimeError('Recorded PID is not the old FPW controller')
        os.kill(pid,signal.SIGTERM)
        for _ in range(25):
            if not proc.exists():break
            time.sleep(.2)
        else:raise RuntimeError('Old controller did not stop')
    queue=command('squeue','-u',os.environ['USER'],'-h','-o','%i|%T')
    if queue.returncode:raise RuntimeError(queue.stderr)
    rows=dict(x.split('|',1) for x in queue.stdout.splitlines() if '|' in x)
    pending=[str(s['job_id']) for s in state['stages'].values() if rows.get(str(s.get('job_id')))=='PENDING']
    if pending:
        cancel=command('scancel','--state=PENDING',*pending)
        if cancel.returncode:raise RuntimeError(cancel.stderr)
    now=command('squeue','-u',os.environ['USER'],'-h','-o','%i|%T')
    if now.returncode:raise RuntimeError(now.stderr)
    remaining=dict(x.split('|',1) for x in now.stdout.splitlines() if '|' in x)
    moved=[]
    for name,stage in state['stages'].items():
        jid=str(stage.get('job_id'))
        if jid in pending and jid not in remaining:
            stage.setdefault('paper_handover',[]).append(dict(job_id=int(jid),reason='latest user prioritizes complete paper models'))
            stage.pop('job_id');stage['status']='WAITING';moved.append(dict(stage=name,job_id=int(jid)))
    state['paper_queue_owner']=str(EXP/'deployment.json');state['former_controller_pid']=pid;state['controller_pid']=None
    json_write(receipt,state)
    record=dict(stopped_controller=pid,pending_jobs_returned_to_queue=moved,running_jobs_preserved=True,
                completed_artifacts_preserved=True,legacy_receipt=str(receipt),time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    json_write(EXP/'legacy_handover.json',record)
    log=(EXP/'dispatcher.log').open('a')
    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/paper_dispatch.py'),'--submit','--max-live','8',
                            '--legacy-receipt',str(receipt),'--inherit-legacy'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record['paper_controller_pid']=child.pid;json_write(EXP/'legacy_handover.json',record);print(json.dumps(record))


if __name__=='__main__':main()
