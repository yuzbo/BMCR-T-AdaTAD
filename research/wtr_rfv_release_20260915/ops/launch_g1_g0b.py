"""Launch the preregistered next two RFV experiments after real G0a completion."""
import json
from deploy_autodl import ssh,PYTHON,ROOT,OUT

CAPTURE='2ca4d4b5f2ad657410bfc5c2c4cebd3bc6a11a32'
ANALYSIS='b679ae0ec46510bd85dc303dfacd4b6a6fd5288d'
body='''import json,os,subprocess,time
from pathlib import Path
root=Path(ROOT);capture=root/'versions'/CAPTURE;analysis=root/'versions'/ANALYSIS
data=root/'results'/CAPTURE[:7];out=root/'results'/ANALYSIS[:7]
out.mkdir(parents=True,exist_ok=True)
g0a=json.loads((data/'analysis_g0a/T_LOCAL_CF.json').read_text())
if not g0a['g0a_loss_pass']:raise RuntimeError('G0a has no registered loss signal; do not expand to full AP')
for shard in (0,1):
 if not (data/f'mini_c40_shard{shard}/completed_{shard}.json').exists():raise RuntimeError('Mini bank incomplete')
commands={
 'g1_mini_cpu': [PYTHON,str(analysis/'tools/rfv_fit.py'),'--bank',str(data/'mini_c40_shard0'),'--bank',str(data/'mini_c40_shard1'),
                 '--output',str(out/'G1_MINI'),'--device','cpu'],
 'g0b_all_windows': ['bash',str(root/'g0b_all_windows.sh')]
}
script='#!/bin/bash\\nset -euo pipefail\\ncd '+str(capture)+'\\n'+PYTHON+' -u tools/rfv_run.py local_cf --config configs/rfv/T-U.json --resources runtime/resources.json --protocol runtime/protocol.json --all-windows --output '+str(data/'T_LOCAL_CF_G0B')+'\\n'+PYTHON+' '+str(analysis/'tools/rfv_analyze.py')+' headroom --capture '+str(data/'T_LOCAL_CF_G0B')+' --output '+str(data/'analysis_g0b')+'\\n'
(root/'g0b_all_windows.sh').write_text(script)
results=[]
for role,command in commands.items():
 receipt=root/'receipts'/(role+'.json')
 if receipt.exists():results.append(json.loads(receipt.read_text()));continue
 if role=='g0b_all_windows':
  gpu_uuid='GPU-140dfb95-91d3-eb14-bcd7-68de19f0ba9d'
  for _ in range(15):
   apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
   if gpu_uuid not in apps:break
   time.sleep(2)
  else:raise RuntimeError('GPU1 still occupied; G0b was not started')
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',
          CUDA_VISIBLE_DEVICES='' if role.endswith('cpu') else '1',H65_SOURCE_REVISION=ANALYSIS if role.endswith('cpu') else CAPTURE)
 log=root/'logs'/(role+'.log')
 with log.open('w') as stream:
  process=subprocess.Popen(command,cwd=analysis,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 saved=dict(role=role,pid=process.pid,command=command,log=str(log),started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            device='cpu' if role.endswith('cpu') else 'physical GPU1',capture_revision=CAPTURE,analysis_revision=ANALYSIS)
 receipt.write_text(json.dumps(saved,indent=2));results.append(saved)
print(json.dumps(results))
'''
header='\n'.join(key+'='+repr(value) for key,value in dict(ROOT=ROOT,CAPTURE=CAPTURE,ANALYSIS=ANALYSIS,PYTHON=PYTHON).items())+'\n'
result=ssh(PYTHON+' -',(header+body).encode(),attempts=1)
(OUT/'next_experiments.json').write_text(result,encoding='utf-8')
print(result)
