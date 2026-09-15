"""Export true V2-S20/60 EMA payloads and transfer them without changing training."""
import json
from pathlib import Path
import subprocess
import time
from deploy_probe import remote,SSH_CONFIG,OUT
from deploy_autodl import ssh,upload,ROOT as AUTODL_ROOT

REMOTE='/data/run01/sczc063/yuzibo/rfv_20260915/assets'
SOURCE='/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/support_review_20260914/research/paper/runs/review5485_full_v2_s_seed42'
PYTHON='/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python'
body='''import json,torch
from pathlib import Path
source=Path(SOURCE);out=Path(OUTPUT);out.mkdir(parents=True,exist_ok=True)
records=[]
for epoch in (20,60):
 path=source/f'epoch_{epoch:03}.pth';value=torch.load(path,map_location='cpu');meta=value['metadata']
 if meta['source_revision']!='db9c749c7bbbdcc1b5f1f14e10b1e1d0f087cacc':raise ValueError('Historical source changed')
 if meta['config']['id']!='review5485_full_v2_s_seed42' or value['successful_updates']!=epoch*100:raise ValueError('Wrong checkpoint course/update')
 target=out/f'v2_s_epoch{epoch:03}_ema_replay.pth'
 if not target.exists():
  torch.save(dict(ema=value['ema'],metadata=meta,epoch_index=value['epoch_index'],successful_updates=value['successful_updates'],
                  original_checkpoint=str(path)),target)
 records.append(dict(epoch=epoch,path=str(target),bytes=target.stat().st_size,original_checkpoint=str(path),
                     updates=value['successful_updates'],source_revision=meta['source_revision'],state='ema'))
 del value
print(json.dumps(records))
'''
script='SOURCE='+repr(SOURCE)+'\nOUTPUT='+repr(REMOTE)+'\n'+body
records=json.loads(remote('4090',PYTHON+' -',script.encode()))
local=OUT/'assets';local.mkdir(exist_ok=True)
ssh('mkdir -p '+AUTODL_ROOT+'/assets')
for row in records:
    path=local/Path(row['path']).name
    if not path.exists() or path.stat().st_size!=row['bytes']:
        command=['scp','-F',SSH_CONFIG,'-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3',
            'bcr-4090:'+row['path'],str(path)]
        for attempt in range(3):
            result=subprocess.run(command,capture_output=True)
            if result.returncode==0:break
            time.sleep(2)
        else:raise RuntimeError('EMA download failed')
    if path.stat().st_size!=row['bytes']:raise RuntimeError('EMA transfer size differs')
    target=AUTODL_ROOT+'/assets/'+path.name;upload(path,target);row['autodl_path']=target
    print(json.dumps(row),flush=True)
(OUT/'rise_assets.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
