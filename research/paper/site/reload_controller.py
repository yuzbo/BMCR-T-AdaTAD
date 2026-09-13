"""Reload the paper controller after code maintenance, preserving every job receipt."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));EXP=ROOT/'research/paper'
from h65.paper.runtime import json_write


def main():
    state=json.loads((EXP/'deployment.json').read_text());old=state['controller_pid'];proc=Path(f'/proc/{old}/cmdline')
    if proc.exists():
        if b'paper_dispatch.py' not in proc.read_bytes():raise RuntimeError('Unexpected PID; no process stopped')
        os.kill(old,signal.SIGTERM)
        for _ in range(25):
            if not proc.exists():break
            time.sleep(.2)
        else:raise RuntimeError('Controller has not stopped')
    legacy=ROOT.parent/'fpw_3d_20260913/research/frame/deployment.json';log=(EXP/'dispatcher.log').open('a')
    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'tools/paper_dispatch.py'),'--submit','--max-live','10',
        '--legacy-receipt',str(legacy),'--inherit-legacy'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(old_pid=old,new_pid=child.pid,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),jobs_preserved={k:v['job_id'] for k,v in state['stages'].items() if v.get('job_id')},
                source_revision=(ROOT/'source_revision.txt').read_text().strip())
    path=EXP/'controller_reloads.json';history=json.loads(path.read_text()) if path.exists() else []
    history.append(record);json_write(path,history);print(json.dumps(record))


if __name__=='__main__':main()
