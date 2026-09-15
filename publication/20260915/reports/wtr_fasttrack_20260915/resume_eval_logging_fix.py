"""Resume the two saved courses after the reviewed evaluation-only logging repair."""
import json,os,signal,subprocess,sys,time
from pathlib import Path

owner=Path('/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/paper_20260913')
core=Path('/data/run01/sczc063/yuzibo/wtr_fasttrack_20260915')
exp=owner/'research/paper';statefile=exp/'deployment.json'
expected='6e2fc7f78424e2485e8d33b079fac926eb0cf4a6'
if (core/'EVALUATION_REVISION').read_text().strip()!=expected:raise RuntimeError('Evaluator repair is not deployed')
state=json.loads(statefile.read_text());pid=state['controller_pid'];proc=Path('/proc')/str(pid)
if proc.exists():
    if str(owner/'tools/paper_dispatch.py') not in (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode():
        raise RuntimeError('Unexpected queue owner')
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        if not proc.exists():break
        time.sleep(.1)
    else:raise RuntimeError('Owner did not exit')
state=json.loads(statefile.read_text());before={}
for name in ['wtr_d_v_s42','wtr_d_u_s42']:
    key='wtr_train_'+name;s=state['stages'][key]
    if s['status']!='FAILED':raise RuntimeError('Expected the diagnosed failed course, not a live job')
    run=core/'research/paper/runs'/name
    if not (run/'latest.pth').exists() or not (run/'epoch_010.pth').exists():raise RuntimeError('Saved training checkpoint is missing')
    before[key]=dict(s)
    s.setdefault('execution_repairs',[]).append(dict(job_id=s.pop('job_id'),exit_code=s.get('exit_code'),
        reason='Internal geometry tensor in evaluation JSON; no training/model change',evaluation_source_revision=expected))
    for field in ['exit_code','scheduler_state','failure']:s.pop(field,None)
    s['status']='WAITING';s['resume_from']='latest.pth at completed epoch10'
receipt_dir=core/'research/wtr_fasttrack/evaluation_repair';receipt_dir.mkdir(parents=True,exist_ok=True)
(receipt_dir/'failed_stages.json').write_text(json.dumps(before,indent=2)+'\n')
tmp=statefile.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(statefile)
log=(exp/'dispatcher_fasttrack_retirement.log').open('a')
child=subprocess.Popen([sys.executable,'-u',str(owner/'tools/paper_dispatch.py'),'--submit','--max-live','7'],cwd=owner,
    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
log.close();receipt=dict(owner_pid=child.pid,model_science_sha=(core/'WTR_FASTTRACK_SCIENCE_SHA').read_text().strip(),
    evaluation_source_revision=expected,preserved_updates=1000,action='resume; finish pending full eval then continue training')
(receipt_dir/'resume.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
