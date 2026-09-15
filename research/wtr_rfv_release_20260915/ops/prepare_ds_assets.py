"""Extract frozen raw trainable states, leaving the four original courses intact."""
import json
from pathlib import Path
import subprocess
import time
from deploy_probe import remote,SITES,SSH_CONFIG,OUT

options=['-F',SSH_CONFIG,'-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3']
folder=OUT/'runtime_local/ds_diagnostic';folder.mkdir(parents=True,exist_ok=True)
program="""import json,torch
from pathlib import Path
root=Path('/data/run01/sczc063/yuzibo/wtr_fasttrack_20260915/research/paper/runs')
out=Path('/data/run01/sczc063/yuzibo/rfv_20260915/assets/ds_epoch40');out.mkdir(parents=True,exist_ok=True)
records=[]
for name in ('wtr_d_v_s42','wtr_s_v_s42'):
 original=root/name/'epoch_040.pth';path=out/(name+'_raw40.pth')
 if not path.exists():
  payload=torch.load(original,map_location='cpu',weights_only=False)
  meta=payload['metadata']
  if meta['wtr_fasttrack_science_sha']!='40552945ad5d56b7404f833cd86f998e8558e8b1' or payload['successful_updates']!=4000 or payload['epoch_index']!=40:
   raise ValueError('Requested original405 epoch40 is not this checkpoint')
  value=dict(learned=payload['learned'],metadata=meta,original_checkpoint=str(original),parameter_state='raw learned',
   successful_updates=payload['successful_updates'],epoch_index=payload['epoch_index'],sample_cursor=payload['sample_cursor'])
  temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)
  del payload,value
 records.append(dict(name=name,path=str(path),bytes=path.stat().st_size,original_checkpoint=str(original)))
print(json.dumps(records))
"""
records=json.loads(remote('4090',SITES['4090']['python']+' -',program.encode()))
target=SITES['a100']['root']+'/assets/ds_epoch40';remote('a100','mkdir -p '+target)
def copy(args):
    for attempt in range(3):
        result=subprocess.run(['scp',*options,*args],capture_output=True)
        if result.returncode==0:return
        time.sleep(2)
    raise RuntimeError(result.stderr.decode(errors='replace'))
for row in records:
    path=folder/Path(row['path']).name
    if not path.exists() or path.stat().st_size!=row['bytes']:
        temporary=path.with_suffix('.download');copy([SITES['4090']['host']+':'+row['path'],str(temporary)])
        if temporary.stat().st_size!=row['bytes']:raise RuntimeError('Incomplete asset download')
        temporary.replace(path)
    destination=target+'/'+path.name
    copy([str(path),SITES['a100']['host']+':'+destination])
    row['a100_path']=destination
    print(json.dumps(dict(name=row['name'],transferred_bytes=row['bytes'],a100_path=destination)),flush=True)
(OUT/'DS_ASSET_RECEIPT.json').write_text(json.dumps(records,indent=2),encoding='utf8')
