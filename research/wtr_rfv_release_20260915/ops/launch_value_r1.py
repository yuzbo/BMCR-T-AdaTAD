"""Dispatch the reviewed R1 comparison once to the released AutoDL GPU0."""
import json
from deploy_probe import package,OUT
from deploy_autodl import ssh,upload,PYTHON,ROOT

REV='b644d870d1845abbc1e4fd5ab7780f29ff96a53a'
CODE=ROOT+'/versions/'+REV
RECEIPT=ROOT+'/receipts/value_r1_'+REV[:7]+'.json'
existing=json.loads(ssh(PYTHON+' -',('from pathlib import Path\nimport json\np=Path('+repr(RECEIPT)+')\nprint(p.read_text() if p.exists() else "null")\n').encode()))
if existing:
    print(json.dumps(existing));raise SystemExit(0)
ssh('mkdir -p '+CODE);upload(package(REV),CODE+'/source.tar.gz')
ssh('tar -xzf '+CODE+'/source.tar.gz -C '+CODE)
upload(OUT/'VALUE_R1_TECH_PASS.json',CODE+'/VALUE_R1_TECH_PASS.json')
body='''import json,os,subprocess,time
from pathlib import Path
root=Path(ROOT);code=Path(CODE);receipt=Path(RECEIPT)
if receipt.exists():print(receipt.read_text());raise SystemExit(0)
tech=json.loads((code/'VALUE_R1_TECH_PASS.json').read_text())
if tech['source_revision']!=REV or tech['status']!='TECH_PASS':raise RuntimeError('R1 technical receipt differs')
release=Path('/root/autodl-tmp/wtr_characterization_20260915/queue/atlas_s_gpu_release.json')
if not release.exists():raise RuntimeError('Atlas GPU0 release missing')
apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
uuid='GPU-da4cac68-a006-0219-7e01-306b04c2450a'
if uuid in apps:raise RuntimeError('RFV GPU0 is occupied; do not duplicate or cancel its owner')
out=root/'results'/REV[:7]/'VALUE_R1';out.mkdir(parents=True,exist_ok=True)
bank=root/'results/2ca4d4b'
for shard in (0,1):
 if not (bank/f'mini_c40_shard{shard}/completed_{shard}.json').exists():raise RuntimeError('Shared bank incomplete')
command=[PYTHON,str(code/'tools/rfv_value_r1.py'),'--bank',str(bank/'mini_c40_shard0'),
 '--bank',str(bank/'mini_c40_shard1'),'--output',str(out),'--device','cuda']
log=root/'logs'/('value_r1_'+REV[:7]+'.log')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',H65_SOURCE_REVISION=REV)
with log.open('w') as stream:
 process=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(pid=process.pid,started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),source_revision=REV,
 allocation_id='autodl-rfv-r1-'+str(process.pid),physical_gpu=0,gpu_uuid=uuid,command=command,
 log=str(log),output=str(out),scope='18 offline heads; no new CF; no detector course',technical_admission=tech)
receipt.write_text(json.dumps(result,indent=2));(out/'run_receipt.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''
header='\n'.join(k+'='+repr(v) for k,v in dict(ROOT=ROOT,CODE=CODE,RECEIPT=RECEIPT,REV=REV,PYTHON=PYTHON).items())+'\n'
result=ssh(PYTHON+' -',(header+body).encode(),attempts=1)
(OUT/'VALUE_R1_LAUNCH.json').write_text(result,encoding='utf8');print(result)
