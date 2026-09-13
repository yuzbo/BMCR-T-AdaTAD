"""Read only small receipts and training logs from this project's own remote root."""
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
remote = r'''
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
root=Path('/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/ds3_20260912')
exp=root/'ds3_20260912'
out={'captured_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'remote_root':str(root),'progress':{},'training_epoch_means':{},'bmcr_preparation_files':[]}
for b in ('s','b'):
 p=exp/'runs'/f'{b}_d1'/'progress.json'
 out['progress'][b]=json.loads(p.read_text()) if p.exists() else None
 log=p.parent/'train.jsonl'
 groups=defaultdict(list)
 if log.exists():
  for line in log.read_text().splitlines():
   try: item=json.loads(line)
   except json.JSONDecodeError: continue
   groups[item['epoch']+1].append(item)
 out['training_epoch_means'][b]=[{ 'epoch':e,'records':len(v),'last_update':v[-1]['successful_updates'],'losses':{k:sum(x['losses'][k] for x in v)/len(v) for k in v[0]['losses']},'lr_last':v[-1]['lr']} for e,v in sorted(groups.items())]
prep=root/'bmcr_fidelity_20260912'
if prep.exists():
 out['bmcr_preparation_files']=[str(p.relative_to(prep)) for p in prep.rglob('*') if p.is_file() and p.suffix in ('.json','.md')]
for name in ('deployment.json','progress.json'):
 p=prep/name
 if p.exists():out['bmcr_'+name]=json.loads(p.read_text())
print(json.dumps(out,ensure_ascii=False))
'''
cmd = ['ssh','-F','C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/motivation/ssh_config',
       '-o','BatchMode=yes','-o','ConnectTimeout=10','bcr-4090',
       '/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python','-']
result = subprocess.run(cmd,input=remote,text=True,encoding='utf-8',capture_output=True,timeout=45)
if result.returncode:
    raise RuntimeError(result.stderr)
data = json.loads(result.stdout)
(root/'evidence/progress_snapshot.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'captured_at':data['captured_at'],'progress':data['progress'],
 'training_epochs':{k:len(v) for k,v in data['training_epoch_means'].items()},
 'bmcr_preparation_files':data['bmcr_preparation_files']},ensure_ascii=False,indent=2))
