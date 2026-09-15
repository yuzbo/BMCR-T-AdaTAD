"""One RFV-owned GPU0 process: saved RISE export, real reverse contract, six heads."""
import json
from deploy_probe import package,OUT
from deploy_autodl import ssh,upload,PYTHON,ROOT

REV='800bcd10a66c69d0d57142ac02389adbc45d2975'
CODE=ROOT+'/versions/'+REV
RECEIPT=ROOT+'/receipts/reverse_'+REV[:7]+'.json'
existing=json.loads(ssh(PYTHON+' -',('from pathlib import Path\np=Path('+repr(RECEIPT)+')\nprint(p.read_text() if p.exists() else "null")\n').encode()))
if existing:print(json.dumps(existing));raise SystemExit(0)
tech=json.loads((OUT/'REVERSE_TECH_PASS.json').read_text())
if tech['source_revision']!=REV or tech['status']!='TECH_PASS':raise RuntimeError('New source needs its technical receipt')
ssh('mkdir -p '+CODE);upload(package(REV),CODE+'/source.tar.gz')
ssh('tar -xzf '+CODE+'/source.tar.gz -C '+CODE)
upload(OUT/'REVERSE_TECH_PASS.json',CODE+'/REVERSE_TECH_PASS.json')
body=r'''import json,os,subprocess,time
from pathlib import Path
root=Path(ROOT);code=Path(CODE);receipt=Path(RECEIPT)
if receipt.exists():print(receipt.read_text());raise SystemExit(0)
release=Path('/root/autodl-tmp/wtr_characterization_20260915/queue/atlas_s_gpu_release.json')
if not release.exists():raise RuntimeError('GPU0 release receipt missing')
apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
uuid='GPU-da4cac68-a006-0219-7e01-306b04c2450a'
if uuid in apps:raise RuntimeError('GPU0 is occupied; preserve its owner')
runtime=code/'runtime';runtime.mkdir(exist_ok=True)
resources=json.loads((root/'versions/2ca4d4b5f2ad657410bfc5c2c4cebd3bc6a11a32/runtime/resources.json').read_text())
resources.update(assigned_physical_gpu=0,handoff_receipt=str(release))
(runtime/'resources.json').write_text(json.dumps(resources,indent=2))
result=root/'results'/REV[:7];result.mkdir(parents=True,exist_ok=True)
bank=root/'results/2ca4d4b';future=root/'results/3a7e93f';reference=root/'results/b644d87/VALUE_R1'
common=['--bank',str(bank/'mini_c40_shard0'),'--bank',str(bank/'mini_c40_shard1')]
commands={
 'rise_choices':[PYTHON,str(code/'tools/rfv_rise_choices.py'),'--forecast-run',str(root/'results/d62ea55/RISE_B0'),
  '--current-bank',str(bank/'mini_c40_shard0'),'--current-bank',str(bank/'mini_c40_shard1'),
  '--future-bank',str(future/'mini_c60_shard0'),'--future-bank',str(future/'mini_c60_shard1'),
  '--output',str(result/'RISE_CHOICES'),'--device','cuda'],
 'reverse_contract':[PYTHON,str(code/'tools/rfv_reverse_contract.py'),*common,'--resources',str(runtime/'resources.json'),
  '--normalization-checkpoint',str(reference/'within_plain_r0_s42.pth'),'--output',str(result/'REVERSE_CONTRACT')],
 'reverse_mini':[PYTHON,str(code/'tools/rfv_reverse_mini.py'),*common,'--reference-run',str(reference),
  '--contract',str(result/'REVERSE_CONTRACT/REVERSE_CONTRACT.json'),'--output',str(result/'REVERSE_MINI'),'--device','cuda']}
runner=root/('reverse_'+REV[:7]+'.py')
program='import json,subprocess,time\nfrom pathlib import Path\ncommands='+repr(commands)+'\nresult=Path('+repr(str(result))+')\n'
program+="""records=[]
for name,command in commands.items():
 if name=='reverse_mini':
  path=result/'REVERSE_CONTRACT/REVERSE_CONTRACT.json'
  if not path.exists() or not json.loads(path.read_text())['passed']:
   records.append(dict(stage=name,status='NOT_STARTED_CONTRACT_FAIL'));continue
 start=time.time();completed=subprocess.run(command)
 records.append(dict(stage=name,exit_code=completed.returncode,wall_seconds=time.time()-start))
 (result/'sprint_progress.json').write_text(json.dumps(records,indent=2))
(result/'sprint_finished.json').write_text(json.dumps(dict(stages=records,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')),indent=2))
"""
runner.write_text(program)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',H65_SOURCE_REVISION=REV)
log=root/'logs'/('reverse_'+REV[:7]+'.log')
with log.open('w') as stream:
 process=subprocess.Popen([PYTHON,str(runner)],cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
saved=dict(pid=process.pid,source_revision=REV,physical_gpu=0,gpu_uuid=uuid,allocation_id='autodl-reverse-'+str(process.pid),
 started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),commands=commands,log=str(log),results=str(result),
 technical_receipt=str(code/'REVERSE_TECH_PASS.json'),scope='fixed B0 export + at most24 replay forwards +6 heads; no detector long course')
receipt.write_text(json.dumps(saved,indent=2));(result/'run_receipt.json').write_text(json.dumps(saved,indent=2))
print(json.dumps(saved))
'''
header='\n'.join(k+'='+repr(v) for k,v in dict(REV=REV,CODE=CODE,ROOT=ROOT,RECEIPT=RECEIPT,PYTHON=PYTHON).items())+'\n'
compile(header+body,'remote_reverse_dispatch','exec')
result=ssh(PYTHON+' -',(header+body).encode(),attempts=1)
(OUT/'REVERSE_LAUNCH.json').write_text(result,encoding='utf8');print(result)
