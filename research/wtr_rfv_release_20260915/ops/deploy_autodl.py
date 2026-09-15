"""Prepare and start the reviewed RFV capture in the handed-over AutoDL GPU1."""
import json
from pathlib import Path
import subprocess
import sys
import time
from deploy_probe import package,OUT,LOCAL

HOST='root@connect.nmb1.seetacloud.com'
PYTHON='/root/autodl-tmp/envs/opentad/bin/python'
ROOT='/root/autodl-tmp/rfv_20260915'
ATLAS='/root/autodl-tmp/wtr_characterization_20260915'
SHA='2ca4d4b5f2ad657410bfc5c2c4cebd3bc6a11a32'
CODE=ROOT+'/versions/'+SHA
OPTIONS=['-p','44909','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3']


def ssh(command,data=None,attempts=3):
    for attempt in range(attempts):
        process=subprocess.run(['ssh',*OPTIONS,HOST,command],input=data,capture_output=True)
        if process.returncode==0:return process.stdout.decode()
        if process.returncode!=255:break
        time.sleep(2)
    raise RuntimeError((process.stdout+process.stderr).decode(errors='replace')[-9000:])


def upload(path,target):
    args=['scp',*['-P' if value=='-p' else value for value in OPTIONS],str(path),HOST+':'+target]
    for attempt in range(4):
        result=subprocess.run(args,capture_output=True)
        if result.returncode==0:return
        time.sleep(2)
    raise RuntimeError(result.stderr.decode(errors='replace'))


def main():
    archive=package(SHA);ssh('mkdir -p '+CODE)
    upload(archive,CODE+'/source.tar.gz')
    upload(OUT/'autodl_hardware_check.py',CODE+'/hardware_smoke.py')
    upload(LOCAL/'research/rfv_sprint_20260915/runtime_local/resources.json',CODE+'/base_resources.json')
    ssh('tar -xzf '+CODE+'/source.tar.gz -C '+CODE)
    body='''import json,os,subprocess,sys,time
from pathlib import Path
code=Path(CODE);root=Path(ROOT);atlas=Path(ATLAS)
sys.path[:0]=[str(code),str(code/'upstream')]
from h65.paper.runtime import json_write
handoff=atlas/'queue/rfv_gpu1_handoff.json'
if not handoff.exists():raise RuntimeError('Missing actual Atlas GPU1 handoff receipt')
res=json.loads((code/'base_resources.json').read_text());source=json.loads((atlas/'resources.json').read_text())
res['datasets']=source['datasets'];assets=atlas/'assets'
res['encoders']['thumos:s']['checkpoint']=str(assets/'v2_s_initial_source.pth')
res['encoders']['thumos:s']['scout_checkpoint']=str(assets/'v2_s_initial_source.pth')
res['teachers']={'thumos:s':str(assets/'adatad_s_ema.pth')}
res['atlas_light']={'s':str(assets/'v2_s_epoch040_light.pth')}
res['wtr_initialization']=str(assets/'v2_s_epoch040_light.pth')
res.update(gpu_type='4080',execution_backend='autodl_owner',assigned_physical_gpu=1,handoff_receipt=str(handoff))
json_write(code/'base_resources.json',res)
subprocess.run([PYTHON,str(code/'tools/rfv_run.py'),'prepare','--resources',str(code/'base_resources.json'),'--output',str(code/'runtime')],check=True)
env=dict(os.environ,PYTHONPATH=str(code)+':'+str(code/'upstream'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
check=subprocess.run([PYTHON,str(code/'hardware_smoke.py')],cwd=code,env=env,capture_output=True,text=True)
json_write(root/'receipts/autodl_hardware_test.json',dict(source_revision=SHA,exit_code=check.returncode,stdout=check.stdout,stderr=check.stderr))
if check.returncode:raise RuntimeError(check.stdout+check.stderr)
print(json.dumps(dict(prepared=True,source_revision=SHA,hardware_test=check.stdout)))
'''
    constants=dict(CODE=CODE,ROOT=ROOT,ATLAS=ATLAS,PYTHON=PYTHON,SHA=SHA,RETRY='--retry' in sys.argv)
    source='\n'.join(key+'='+repr(value) for key,value in constants.items())+'\n'+body
    result=ssh(PYTHON+' -',source.encode())
    (OUT/'autodl_prepared.txt').write_text(result,encoding='utf-8');print(result,flush=True)
    start='''import json,os,subprocess,time
from pathlib import Path
root=Path(ROOT);code=Path(CODE);result=root/'results'/SHA[:7]
receipt=root/'receipts'/(SHA[:7]+'_capture_launch.json')
attempt=1
if receipt.exists():
 if not RETRY:
  print(receipt.read_text());raise SystemExit(0)
 previous=json.loads(receipt.read_text());pid=previous['pid'];status=Path('/proc')/str(pid)/'stat'
 if status.exists() and status.read_text().split()[2]!='Z':raise RuntimeError('Previous RFV owner is still alive')
 attempt=previous.get('attempt',1)+1
 archived=receipt.with_name(receipt.stem+'_attempt'+str(attempt-1)+'.json')
 archived.write_text(receipt.read_text())
# The owner has reserved physical GPU1. Confirm it has no compute process.
apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
if 'GPU-140dfb95-91d3-eb14-bcd7-68de19f0ba9d' in apps:raise RuntimeError('Handed-over GPU1 has a compute process')
result.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
common=PYTHON+' -u tools/rfv_run.py '
tail=' --resources runtime/resources.json --protocol runtime/protocol.json --output '
commands=[common+'smoke --config configs/rfv/T-V.json'+tail+str(result/'smoke_autodl'),
 common+'local_cf --config configs/rfv/T-U.json'+tail+str(result/'T_LOCAL_CF_G0A'),
 common+'bank --config configs/rfv/T-U.json --cohort mini --shard 0 --shards 2'+tail+str(result/'mini_c40_shard0'),
 common+'bank --config configs/rfv/T-U.json --cohort mini --shard 1 --shards 2'+tail+str(result/'mini_c40_shard1')]
job=root/('capture_'+SHA[:7]+'.sh')
job.write_text('#!/bin/bash\\nset -euo pipefail\\ncd '+str(code)+'\\n'+'\\n'.join(commands)+'\\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',H65_SOURCE_REVISION=SHA,
 PYTHONPATH=str(code)+':'+str(code/'upstream'))
log=root/'logs'/('capture_'+SHA[:7]+('_attempt'+str(attempt) if attempt>1 else '')+'.log')
with log.open('w') as stream:
 process=subprocess.Popen(['bash',str(job)],cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
saved=dict(pid=process.pid,attempt=attempt,physical_gpu=1,backend='owner_assigned_direct_process',source_revision=SHA,
 code=str(code),results=str(result),log=str(log),commands=commands,started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
 handoff_receipt=ATLAS+'/queue/rfv_gpu1_handoff.json')
receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(saved,indent=2))
print(json.dumps(saved))
'''
    payload='\n'.join(key+'='+repr(value) for key,value in constants.items())+'\n'+start
    launched=ssh(PYTHON+' -',payload.encode(),attempts=1)
    (OUT/'autodl_launch.json').write_text(launched,encoding='utf-8');print(launched,flush=True)


if __name__=='__main__':main()
