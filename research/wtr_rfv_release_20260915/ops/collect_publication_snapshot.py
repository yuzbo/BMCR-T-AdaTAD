"""Read-only snapshot of retained courses for the user-requested GitHub release."""
import json
from pathlib import Path
from deploy_probe import remote,SITES,OUT

body="""import json,subprocess,time
from pathlib import Path
root=Path('/data/run01/sczc063/yuzibo/wtr_fasttrack_20260915/research/paper/runs')
files={};courses={}
for name in ('wtr_d_v_s42','wtr_d_u_s42','wtr_s_v_s42','wtr_s_u_s42'):
 run=root/name
 text=(run/'train.jsonl').read_text()
 lines=text.splitlines();valid=[]
 for line in lines:
  try:json.loads(line)
  except json.JSONDecodeError:break
  valid.append(line)
 files[name+'/train.jsonl']='\\n'.join(valid)+'\\n'
 meta=json.loads((run/'metadata.json').read_text())
 files[name+'/metadata.json']=json.dumps(meta,indent=2)
 latest=json.loads(valid[-1])
 epochs={}
 for epoch in (10,20,40,60,80):
  directory=run/f'eval_{epoch:03}_ema'
  metric=directory/'metrics.json'
  if metric.exists():
   epochs[str(epoch)]=json.loads(metric.read_text())
   files[name+f'/eval_{epoch:03}_ema/metrics.json']=metric.read_text()
   for note in ('metadata.json','evaluation.json','completed.json'):
    path=directory/note
    if path.exists():files[name+f'/eval_{epoch:03}_ema/'+note]=path.read_text()
 for note in ('completed.json','yielded.json','cost_table.json'):
  path=run/note
  if path.exists():files[name+'/'+note]=path.read_text()
 courses[name]=dict(last_training_record=latest,metrics=epochs,original_run=str(run),log_rows=len(valid),
  source_revision=meta.get('source_revision'),science_sha=meta.get('wtr_fasttrack_science_sha'))
queue=subprocess.run(['squeue','-j','1290651,1290652,1290654,1290655','-h','-o','%i|%T|%R'],capture_output=True,text=True)
print(json.dumps(dict(recorded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),queue=queue.stdout,courses=courses,files=files)))
"""
snapshot=json.loads(remote('4090',SITES['4090']['python']+' -',body.encode()))
folder=OUT/'publication_snapshot';folder.mkdir(exist_ok=True)
for name,value in snapshot.pop('files').items():
    path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value,encoding='utf8')
(folder/'SNAPSHOT.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf8')
summary={name:dict(updates=row['last_training_record'].get('successful_updates'),
    epoch=row['last_training_record'].get('epoch'),available_eval_epochs=list(row['metrics']),
    metrics={e:m.get('average_mAP',m) for e,m in row['metrics'].items()}) for name,row in snapshot['courses'].items()}
print(json.dumps(dict(recorded_at=snapshot['recorded_at'],queue=snapshot['queue'],courses=summary),ensure_ascii=False))
