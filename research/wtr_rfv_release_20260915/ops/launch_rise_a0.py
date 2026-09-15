"""Run the two historical checkpoint replays against the completed shared T bank."""
from deploy_autodl import ssh,PYTHON,ROOT,OUT
import json

CAPTURE='3a7e93f3f77350e9598011db1dd821cedaee8f0b'
REFERENCE='2ca4d4b5f2ad657410bfc5c2c4cebd3bc6a11a32'
ANALYSIS='b679ae0ec46510bd85dc303dfacd4b6a6fd5288d'
body='''import json,os,subprocess,time
from pathlib import Path
root=Path(ROOT);code=root/'versions'/CAPTURE;analysis=root/'versions'/ANALYSIS;data=root/'results'/CAPTURE[:7];reference=root/'results'/REFERENCE[:7]
receipt=root/'receipts'/('rise_a0_launch_'+CAPTURE[:7]+'.json')
if receipt.exists():print(receipt.read_text());raise SystemExit(0)
apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
if 'GPU-140dfb95-91d3-eb14-bcd7-68de19f0ba9d' in apps:raise RuntimeError('GPU1 occupied; historical replay not started')
commands=[]
for epoch in (20,60):
 checkpoint=root/f'assets/v2_s_epoch{epoch:03}_ema_replay.pth'
 if not checkpoint.exists():raise RuntimeError('Historical EMA export missing')
 for shard in (0,1):
  commands.append(PYTHON+' -u tools/rfv_run.py replay --config configs/rfv/T-U.json --resources runtime/resources.json --protocol runtime/protocol.json --cohort mini --shard '+str(shard)+' --shards 2 --checkpoint '+str(checkpoint)+' --action-manifest '+str(reference/f'mini_c40_shard{shard}')+' --output '+str(data/f'mini_c{epoch}_shard{shard}'))
for earlier,later in ((20,40),(40,60),(20,60)):
 command=PYTHON+' '+str(analysis/'tools/rfv_analyze.py')+' drift'
 for shard in (0,1):command+=' --earlier '+str((reference if earlier==40 else data)/f'mini_c{earlier}_shard{shard}')+' --later '+str((reference if later==40 else data)/f'mini_c{later}_shard{shard}')
 commands.append(command+' --output '+str(data/f'RISE_A0_{earlier}_{later}'))
job=root/('rise_a0_'+CAPTURE[:7]+'.sh');job.write_text('#!/bin/bash\\nset -euo pipefail\\ncd '+str(code)+'\\n'+'\\n'.join(commands)+'\\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',H65_SOURCE_REVISION=CAPTURE)
log=root/'logs'/('rise_a0_'+CAPTURE[:7]+'.log')
with log.open('w') as stream:process=subprocess.Popen(['bash',str(job)],cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,physical_gpu=1,capture_revision=CAPTURE,analysis_revision=ANALYSIS,
 commands=commands,log=str(log),scope='historical V2-S20/40/60 fixed-action existence probe; not new T trajectory',started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
header='\n'.join(key+'='+repr(value) for key,value in dict(ROOT=ROOT,PYTHON=PYTHON,CAPTURE=CAPTURE,REFERENCE=REFERENCE,ANALYSIS=ANALYSIS).items())+'\n'
result=ssh(PYTHON+' -',(header+body).encode(),attempts=1)
(OUT/'rise_a0_launch.json').write_text(result,encoding='utf-8');print(result)
