"""Evaluate already fitted R1 experiment heads on their own fit rows, once."""
from deploy_autodl import ssh,PYTHON,ROOT,OUT
import json

REV='b644d870d1845abbc1e4fd5ab7780f29ff96a53a'
body='''import sys,os,json,subprocess
from pathlib import Path
root=Path(ROOT);code=root/'versions'/REV;out=root/'results'/REV[:7]/'VALUE_R1'
target=out/'FIT_DIAGNOSTIC.json'
if target.exists():print(target.read_text());raise SystemExit(0)
apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
if 'GPU-da4cac68-a006-0219-7e01-306b04c2450a' in apps:raise RuntimeError('GPU0 occupied')
sys.path[:0]=[str(code),str(code/'upstream')]
import torch
torch.set_num_threads(2)
from h65.rfv.dataset import load_bank
from h65.rfv.value import from_snapshot
from tools.rfv_fit import evaluate,average_seeds
from tools.rfv_value_r1 import unseen_split,ARMS
from h65.paper.runtime import json_write
bank=root/'results/2ca4d4b'
rows,binding=load_bank([bank/'mini_c40_shard0',bank/'mini_c40_shard1'])
fit8,held8,split=unseen_split(rows)
fit16=[row for row in rows if row['partition']=='fit' and row['action_pairs']]
seeds=binding['protocol']['head_seeds'];reports={};summary={}
for suite,fit in [('full',fit16),('within',fit8)]:
 reports[suite]={};summary[suite]={}
 for arm in ARMS:
  values=[]
  for seed in seeds:
   payload=torch.load(out/f'{suite}_{arm}_s{seed}.pth',map_location='cpu',weights_only=False)
   model=from_snapshot(payload['snapshot'],'cuda')
   result=evaluate(model,fit,'cuda');values.append(result)
   reports[suite][f'{arm}_s{seed}']=dict(fit=result,objective=payload['objective'],last_training=payload['history'][-1])
   del model,payload
  summary[suite][arm]=average_seeds(values)
result=dict(scope='fit-only readback of saved heads; no retraining, CF query, selection or holdout reevaluation',
 source_revision=REV,summary=summary,head_reports=reports,optimizer_updates=0,new_cf_queries=0,
 allocation_id='autodl-rfv-r1-fit-readback-'+str(os.getpid()),physical_gpu=0)
json_write(target,result);print(json.dumps(result))
'''
head='ROOT='+repr(ROOT)+'\nREV='+repr(REV)+'\n'
result=json.loads(ssh('CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 '+PYTHON+' -',(head+body).encode()))
(OUT/'R1_FIT_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps({suite:{arm:record['mean'] for arm,record in arms.items()} for suite,arms in result['summary'].items()},indent=2))
