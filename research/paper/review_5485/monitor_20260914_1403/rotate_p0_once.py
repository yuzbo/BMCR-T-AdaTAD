"""One resource rotation after the first six P0 courses all reached epoch20 testing."""
import json,subprocess,time
from pathlib import Path
BASE=Path('/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910')
OLD=BASE/'paper_20260913';NEW=BASE/'support_review_20260914';EXP=OLD/'research/paper'
receipt=NEW/'research/paper/review_5485/monitor_20260914_1403/resource_rotation.json'
receipt.parent.mkdir(parents=True,exist_ok=True)
if receipt.exists():
    print(receipt.read_text());raise SystemExit(0)
state=json.loads((EXP/'deployment.json').read_text())
queue={r[0]:r[1:] for line in subprocess.check_output(['squeue','-u','sczc063','-h','-o','%i|%T|%R'],text=True).splitlines() if len(r:=line.split('|'))==3}
pbd=[state['stages']['train_review5485_p01_'+bb+'_seed42']['job_id'] for bb in ('s','b')]
waiting=[j for j in pbd if queue.get(str(j))==['PENDING','(AssocGrpGRES)']]
if not waiting:
    print(json.dumps(dict(action='none',reason='PBD no longer waits on account GPU capacity')));raise SystemExit(0)
for bb in ('s','b'):
    for root,ident in [(OLD,f'thumos_{bb}_point_full_seed42'),(OLD,f'thumos_{bb}_point_uniform_seed42'),(NEW,f'review5485_full_v2_{bb}_seed42')]:
        result=json.loads((root/'research/paper/runs'/ident/'eval_020_ema/completed.json').read_text())
        assert result['test_videos']==211 and result['test_windows']==792 and result['config']['seed']==42 and result['config']['epochs']==80
data=json.loads((EXP/'data_preparation_job.json').read_text());protected=data.get('allocation_job_id')
targets=[]
for bb in ('s','b')[:len(waiting)]:
    ident=f'thumos_{bb}_point_full_seed42';stage=state['stages']['train_'+ident];job=stage['job_id']
    assert queue.get(str(job),[''])[0]=='RUNNING' and job!=protected
    run=EXP/'runs'/ident;assert (run/'latest.pth').exists()
    targets.append(dict(config_id=ident,job_id=job,latest_checkpoint=str(run/'latest.pth'),expected_epochs=80,seed=42))
command=['scancel','--signal=USR1','--batch',*[str(t['job_id']) for t in targets]]
subprocess.run(command,check=True)
record=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),reason='Resource breadth within P0: oldest two allocations yield; no score-based stopping',
            targets=targets,pbd_waiting_jobs=waiting,protected_data_allocation=protected,command=command,
            continuation='Existing trainer saves optimizer/EMA/RNG/cursor, exits75; existing dispatcher resumes the same seed42 80-epoch courses')
receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
