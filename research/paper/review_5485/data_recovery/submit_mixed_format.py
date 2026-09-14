import json,subprocess,sys,time
from pathlib import Path
base=Path('/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910')
old=base/'paper_20260913';new=base/'support_review_20260914';exp=old/'research/paper'
out=new/'research/paper/review_5485/data_recovery';out.mkdir(exist_ok=True)
receipt=out/'mixed_format_submission.json'
if receipt.exists():
    print(receipt.read_text());raise SystemExit(0)
previous=json.loads((exp/'data_preparation_job.json').read_text())
queue=subprocess.check_output(['squeue','-u','sczc063','-h','-o','%i|%j'],text=True)
assert not any(line.split('|')[0]==str(previous['job_id']) for line in queue.splitlines())
cmd=['sbatch','--parsable','--partition=gpu','--qos=gpugpu','--nodes=1','--ntasks=1','--gres=gpu:1',
     '--cpus-per-task=8','--time=12:00:00','--job-name=paper-anet-all-formats',
     '--output='+str(exp/'slurm/%j_anet_all_formats.log'),'--wrap',
     sys.executable+' -u '+str(old/'tools/paper_data/prepare_anet_archives.py')+' --ready-output '+str(exp/'assets/anet_ready.json')]
jid=int(subprocess.check_output(cmd,text=True).strip().split(';')[0])
record=dict(job_id=jid,phase='resume_all_observed_video_formats',previous_job=previous['job_id'],
            observed_formats=['.mp4','.mkv','.webm'],data_fix_revision='deee9f2',
            preserved_archives=True,prepared_before=dict(training=9062,validation=4312),
            time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),command=cmd)
(out/'previous_preparation_job.json').write_text(json.dumps(previous,indent=2)+'\n')
receipt.write_text(json.dumps(record,indent=2)+'\n')
(exp/'data_preparation_job.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
