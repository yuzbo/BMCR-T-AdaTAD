import json,subprocess,time
from pathlib import Path
from collections import Counter
base=Path('/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910')
old=base/'paper_20260913';new=base/'support_review_20260914';exp=old/'research/paper'
def record(path):
    return json.loads(path.read_text()) if path.exists() else None
def tail(path,size=3000):
    if not path.exists():return None
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-size));return stream.read().decode('utf-8',errors='replace')
state=record(exp/'deployment.json')
out=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),deployment=state,registration=record(new/'research/paper/review_5485/registration.json'),
         controller_alive=Path(f"/proc/{state['controller_pid']}/cmdline").exists(),
         queue=subprocess.check_output(['squeue','-u','sczc063','-h','-o','%i|%j|%T|%R'],text=True),courses={},logs={},evidence={})
out['native_adatad_registration']=record(new/'research/paper/native_adatad/registration.json')
for name,row in state['stages'].items():
    if row.get('job_id') and row.get('status') in ('RUNNING','PENDING','FAILED'):
        out['logs'][name]=tail(exp/'slurm'/f"{row['job_id']}.log")
    if row['kind']=='train' and row.get('priority',99)<10:
        root=new if row['config_id'].startswith(('review5485_','native_adatad_')) else old
        run=root/'research/paper/runs'/row['config_id'];check=root/'research/paper/runs'/('preflight_'+row['config_id'])
        out['courses'][name]=dict(job_id=row.get('job_id'),status=row.get('status'),metadata=record(run/'metadata.json'),
            preflight=record(check/'completed.json'),inference_contract=record(check/'inference_contract.json'),train_tail=tail(run/'train.jsonl'))
        out['evidence'][row['config_id']]=[dict(source=str(p),record=record(p)) for p in sorted(run.glob('eval_*/completed.json'))]
    if name.startswith(('eval_public_adatad','eval_native_adatad')):
        path=exp/row['done']
        if path.exists():out['evidence'][name]=[dict(source=str(path),record=record(path))]
for rel in ['runs/preflight_strict_inference_s/inference_contract.json','runs/preflight_strict_inference_s/completed.json',
            'recovery_20260914_0240/shard_repair.json','data_preparation_job.json','assets/anet_ready.json']:
    out[rel]=record(exp/rel)
data=out['data_preparation_job.json'];data_job=data['job_id'];is_step=data.get('phase')=='cpu_step_in_owned_training_allocation'
logs=[Path(data['log_path'])] if is_step else sorted((exp/'slurm').glob(f'{data_job}_anet*.log'))
out['anet_logs']={str(p):tail(p) for p in logs}
out['anet_accounting']=subprocess.check_output(['sacct','-j',str(data_job),*([] if is_step else ['-X']),'-n','-P','--format=JobIDRaw,State,Elapsed,ExitCode'],text=True)
reports=Path('/data/run01/sczc063/yuzibo/bcr_tad_v3_implementation/OpenTAD/reports/data/anet_preparation')
out['anet_preparation']=record(reports/'preparation.json')
out['anet_journal_tail']=tail(reports/'prepared_videos.jsonl')
out['controller_log']=tail(exp/'slurm/controller_review_priority.log')
print(json.dumps(out))
