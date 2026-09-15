"""Submit validation/resume after the desktop relay finishes, without a GPU download."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'research/paper';OUT=EXP/'recovery_20260914_0240'
incoming=Path('/data/run01/sczc063/yuzibo/bcr_tad_v3_implementation/activitynet/downloads/repair_downloads_1288466/v1-2_val.tar.gz.09')
assert incoming.exists() and incoming.stat().st_size==4194304000
queue=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j|%T'],text=True)
assert 'paper-anet-relay-resume' not in queue
cmd=['sbatch','--parsable','--partition=gpu','--qos=gpugpu','--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-task=8','--time=12:00:00','--job-name=paper-anet-relay-resume',
     '--output='+str(EXP/'slurm/%j_anet_relay_resume.log'),'--wrap',sys.executable+' -u '+str(ROOT/'tools/paper_data/repair_anet_shards.py')]
result=subprocess.run(cmd,text=True,capture_output=True,check=True);jid=int(result.stdout.strip().split(';')[0])
record=dict(job_id=jid,phase='verify_relayed_shard_then_resume',previous_failed_job=1288754,time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),command=cmd)
(EXP/'data_preparation_job.json').write_text(json.dumps(record,indent=2)+'\n');(OUT/'relay_resume_submission.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
