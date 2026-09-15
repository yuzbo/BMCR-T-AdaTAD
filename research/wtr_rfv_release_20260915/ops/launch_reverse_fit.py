"""Start the untrained six heads, reusing the independently clean800 replay."""
import json
from deploy_probe import package,OUT
from deploy_autodl import ssh,upload,PYTHON,ROOT
REV='4aa1ca242c37b3f7e2aa727b05419628a74e4e90'
CODE=ROOT+'/versions/'+REV;RECEIPT=ROOT+'/receipts/reverse_fit_'+REV[:7]+'.json'
check='from pathlib import Path\np=Path('+repr(RECEIPT)+')\nprint(p.read_text() if p.exists() else "null")\n'
previous=json.loads(ssh(PYTHON+' -',check.encode()))
if previous:print(json.dumps(previous));raise SystemExit(0)
ssh('mkdir -p '+CODE);upload(package(REV),CODE+'/source.tar.gz');ssh('tar -xzf '+CODE+'/source.tar.gz -C '+CODE)
body='''import json,os,subprocess,time
from pathlib import Path
root=Path(ROOT);code=Path(CODE);receipt=Path(RECEIPT)
if receipt.exists():print(receipt.read_text());raise SystemExit(0)
finished=json.loads((root/'results/800bcd1/sprint_finished.json').read_text())
if finished['stages'][-1]['stage']!='reverse_mini' or finished['stages'][-1]['exit_code']!=1:
 raise RuntimeError('Prior failed launch is not the expected untrained normalization check')
contract=root/'results/800bcd1/REVERSE_CONTRACT/REVERSE_CONTRACT.json'
review=json.loads((code/'research/rfv_sprint_20260915/REVERSE_RISE_800_RECHECK.json').read_text())
if not review['passed'] or any(review['reverse_contract_maxima'][k]!=0 for k in ('replay_max_error','cached_gain_error','gain_sum_max')):
 raise RuntimeError('The completed replay lacks its independent zero-error review')
uuid='GPU-da4cac68-a006-0219-7e01-306b04c2450a'
apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
if uuid in apps:raise RuntimeError('GPU0 is occupied; preserve current owner')
bank=root/'results/2ca4d4b';out=root/'results'/REV[:7]/'REVERSE_MINI';out.mkdir(parents=True,exist_ok=True)
command=[PYTHON,str(code/'tools/rfv_reverse_mini.py'),'--bank',str(bank/'mini_c40_shard0'),'--bank',str(bank/'mini_c40_shard1'),
 '--reference-run',str(root/'results/b644d87/VALUE_R1'),'--contract',str(contract),'--output',str(out),'--device','cuda']
log=root/'logs'/('reverse_fit_'+REV[:7]+'.log')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',H65_SOURCE_REVISION=REV)
with log.open('w') as stream:
 process=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
saved=dict(pid=process.pid,started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),source_revision=REV,physical_gpu=0,
 allocation_id='autodl-reverse-fit-'+str(process.pid),contract_capture_source='800bcd10a66c69d0d57142ac02389adbc45d2975',
 independent_review=str(code/'research/rfv_sprint_20260915/REVERSE_RISE_800_RECHECK.json'),command=command,
 log=str(log),output=str(out),scope='six previously untrained heads only; no repeated RISE or GPU replay')
receipt.write_text(json.dumps(saved,indent=2));(out/'run_receipt.json').write_text(json.dumps(saved,indent=2));print(json.dumps(saved))
'''
header='\n'.join(k+'='+repr(v) for k,v in dict(REV=REV,CODE=CODE,ROOT=ROOT,RECEIPT=RECEIPT,PYTHON=PYTHON).items())+'\n'
compile(header+body,'reverse_fit_dispatch','exec')
result=ssh(PYTHON+' -',(header+body).encode(),attempts=1)
(OUT/'REVERSE_FIT_LAUNCH.json').write_text(result,encoding='utf8');print(result)
