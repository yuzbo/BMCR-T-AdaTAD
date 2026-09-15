"""Move only our still-pending diagnostic to the released AutoDL GPU1."""
import json
import subprocess
from deploy_probe import SITES,SSH_CONFIG,OUT,package
from deploy_autodl import ssh,upload,PYTHON,ROOT
REV='72459727f61454194f0d84865f31a16ed4e10448';CODE=ROOT+'/versions/'+REV
RECEIPT=ROOT+'/receipts/ds_diagnostic_'+REV[:7]+'_autodl.json'
existing=json.loads(ssh(PYTHON+' -',('from pathlib import Path\np=Path('+repr(RECEIPT)+')\nprint(p.read_text() if p.exists() else "null")\n').encode()))
if existing:print(json.dumps(existing));raise SystemExit(0)
site=SITES['a100']
cancel=r'''import json,subprocess
from pathlib import Path
receipt=Path(ROOT)/'receipts/ds_move_to_autodl.json'
if receipt.exists():print(receipt.read_text());raise SystemExit(0)
state=subprocess.run(['squeue','-j','248477','-h','-o','%T'],capture_output=True,text=True,check=True).stdout.strip()
if state!='PENDING':
 print(json.dumps(dict(moved=False,state=state,reason='Only an own PENDING diagnostic can move')));raise SystemExit(0)
subprocess.run(['scancel','248477'],check=True)
saved=dict(moved=True,job_id=248477,previous_state=state,reason='Use already released AutoDL GPU1; no duplicate diagnostic')
receipt.write_text(json.dumps(saved,indent=2));print(json.dumps(saved))
'''
payload='ROOT='+repr(site['root'])+'\n'+cancel
options=['-F',SSH_CONFIG,'-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3']
process=subprocess.run(['ssh',*options,site['host'],site['python']+' -'],input=payload.encode(),capture_output=True)
if process.returncode:raise RuntimeError('Inspect our cancellation receipt before retrying: '+process.stderr.decode(errors='replace'))
moved=json.loads(process.stdout);(OUT/'DS_A100_MOVE.json').write_text(json.dumps(moved,indent=2),encoding='utf8')
if not moved['moved']:print(json.dumps(moved));raise SystemExit(0)
ssh('mkdir -p '+CODE+' '+ROOT+'/assets/ds_epoch40')
upload(package(REV),CODE+'/source.tar.gz');ssh('tar -xzf '+CODE+'/source.tar.gz -C '+CODE)
for row in json.loads((OUT/'DS_ASSET_RECEIPT.json').read_text()):
    name=row['path'].split('/')[-1]
    upload(OUT/'runtime_local/ds_diagnostic'/name,ROOT+'/assets/ds_epoch40/'+name)
body='''import json,subprocess,os,time
from pathlib import Path
root=Path(ROOT);code=Path(CODE);receipt=Path(RECEIPT)
if receipt.exists():print(receipt.read_text());raise SystemExit(0)
uuid='GPU-140dfb95-91d3-eb14-bcd7-68de19f0ba9d'
apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
if uuid in apps:raise RuntimeError('Released GPU1 has a compute owner; do not duplicate or cancel it')
resources=json.loads((root/'versions/2ca4d4b5f2ad657410bfc5c2c4cebd3bc6a11a32/runtime/resources.json').read_text())
resources['assigned_physical_gpu']=1
runtime=code/'runtime';runtime.mkdir(exist_ok=True);(runtime/'resources.json').write_text(json.dumps(resources,indent=2))
out=root/'results'/REV[:7]/'DS_DIAGNOSTIC';out.mkdir(parents=True,exist_ok=True)
command=[PYTHON,str(code/'tools/rfv_ds_diagnostic.py'),'--checkpoint',str(root/'assets/ds_epoch40/wtr_d_v_s42_raw40.pth'),
 '--checkpoint',str(root/'assets/ds_epoch40/wtr_s_v_s42_raw40.pth'),'--resources',str(runtime/'resources.json'),'--output',str(out)]
log=root/'logs'/('ds_diagnostic_'+REV[:7]+'.log')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',H65_SOURCE_REVISION=REV)
with log.open('w') as stream:
 process=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
saved=dict(pid=process.pid,source_revision=REV,physical_gpu=1,allocation_id='autodl-ds-'+str(process.pid),
 started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),original_science='40552945ad5d56b7404f833cd86f998e8558e8b1',
 command=command,log=str(log),output=str(out),cancelled_pending_a100_job=248477,scope='Fixed raw40 diagnostic only; no old-course or optimizer changes')
receipt.write_text(json.dumps(saved,indent=2));(out/'run_receipt.json').write_text(json.dumps(saved,indent=2));print(json.dumps(saved))
'''
header='\n'.join(k+'='+repr(v) for k,v in dict(REV=REV,CODE=CODE,ROOT=ROOT,RECEIPT=RECEIPT,PYTHON=PYTHON).items())+'\n'
compile(header+body,'autodl_ds_dispatch','exec')
result=ssh(PYTHON+' -',(header+body).encode(),attempts=1)
(OUT/'DS_AUTODL_LAUNCH.json').write_text(result,encoding='utf8');print(result)
