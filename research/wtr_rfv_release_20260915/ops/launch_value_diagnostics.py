"""Run one within-video diagnostic and the distinct historical forecast probe."""
from deploy_probe import package,OUT
from deploy_autodl import ssh,upload,PYTHON,ROOT

REV='d62ea556c56ca6157289a866aaafc79c4fa427c6'
CODE=ROOT+'/versions/'+REV
ssh('mkdir -p '+CODE);upload(package(REV),CODE+'/source.tar.gz')
ssh('tar -xzf '+CODE+'/source.tar.gz -C '+CODE)
body='''import json,os,subprocess,time
from pathlib import Path
root=Path(ROOT);code=Path(CODE);out=root/'results'/REV[:7];out.mkdir(parents=True,exist_ok=True)
current=root/'results/2ca4d4b';history=root/'results/3a7e93f'
for base,epoch in [(history,20),(current,40),(history,60)]:
 for shard in (0,1):
  if not (base/f'mini_c{epoch}_shard{shard}/completed_{shard}.json').exists():raise RuntimeError('Historical/shared bank incomplete')
commands={
 'action_holdout_cpu':[PYTHON,str(code/'tools/rfv_action_holdout.py'),'--bank',str(current/'mini_c40_shard0'),
    '--bank',str(current/'mini_c40_shard1'),'--output',str(out/'ACTION_HOLDOUT')],
 'rise_b0_gpu0':[PYTHON,str(code/'tools/rfv_forecast.py'),'--variant','plain_m','--device','cuda',
    '--capture-equivalence',str(code/'research/rfv_sprint_20260915/CAPTURE_EQUIVALENCE.json'),'--output',str(out/'RISE_B0')]
}
for kind,base,epoch in [('anchor',history,20),('current',current,40),('future',history,60)]:
 for shard in (0,1):commands['rise_b0_gpu0'] += ['--'+kind+'-bank',str(base/f'mini_c{epoch}_shard{shard}')]
results=[]
for role,command in commands.items():
 receipt=root/'receipts'/(role+'.json')
 if receipt.exists():results.append(json.loads(receipt.read_text()));continue
 if role=='rise_b0_gpu0':
  if not Path('/root/autodl-tmp/wtr_characterization_20260915/queue/atlas_s_gpu_release.json').exists():raise RuntimeError('Atlas GPU0 release missing')
  apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
  if 'GPU-da4cac68-a006-0219-7e01-306b04c2450a' in apps:raise RuntimeError('Released GPU0 is occupied')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if role=='rise_b0_gpu0' else '',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
          PYTHONUNBUFFERED='1',H65_SOURCE_REVISION=REV)
 log=root/'logs'/(role+'.log')
 with log.open('w') as stream:process=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 result=dict(pid=process.pid,role=role,source_revision=REV,device='physical GPU0' if role=='rise_b0_gpu0' else 'CPU',
             command=command,log=str(log),started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),scope='diagnostic only; no task/FVD unlock')
 receipt.write_text(json.dumps(result,indent=2));results.append(result)
print(json.dumps(results))
'''
header='\n'.join(key+'='+repr(value) for key,value in dict(ROOT=ROOT,CODE=CODE,REV=REV,PYTHON=PYTHON).items())+'\n'
result=ssh(PYTHON+' -',(header+body).encode(),attempts=1)
(OUT/'value_diagnostics_launch.json').write_text(result,encoding='utf-8');print(result)
