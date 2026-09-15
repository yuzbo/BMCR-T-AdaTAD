"""Check the exact locked within-video8/8 input support, without fitting."""
from deploy_autodl import ssh,PYTHON,ROOT,OUT
import json
REV='b644d870d1845abbc1e4fd5ab7780f29ff96a53a'
body='''import sys,json,collections
from pathlib import Path
root=Path(ROOT);code=root/'versions'/REV;out=root/'results'/REV[:7]/'VALUE_R1'
sys.path[:0]=[str(code),str(code/'upstream')]
import numpy as np,torch
from h65.rfv.dataset import load_bank
from tools.rfv_value_r1 import unseen_split
from h65.raw.value import SCALARS
torch.set_num_threads(2)
bank=root/'results/2ca4d4b';rows,binding=load_bank([bank/'mini_c40_shard0',bank/'mini_c40_shard1'])
fit,held,identity=unseen_split(rows)
registered=json.loads((out/'config.json').read_text())['within_split']
assert identity==registered and len(fit)==len(held)==25
arrays={name:np.concatenate([r['arrays']['descriptor'] for r in values]) for name,values in [('fit',fit),('held',held)]}
assert len(arrays['fit'])==len(arrays['held'])==200
snapshot=torch.load(out/'within_plain_r0_s42.pth',map_location='cpu',weights_only=False)['snapshot']['state']
scale=snapshot['input_scale'].numpy();mean=snapshot['input_mean'].numpy()
counts={}
for name,values in [('fit',fit),('held',held)]:
 delta=[];parity=[];partner=[]
 for row in values:
  slots={f:i for i,(f,v) in enumerate(zip(row['selected_frame_ids'],row['selected_valid'])) if v}
  for remove,insert in row['action_pairs']:
   delta.append(insert-remove);slot=slots[remove];parity.append(slot%2)
   partner.append(row['selected_frame_ids'][slot^1]-remove)
 counts[name]=dict(videos=len(values),candidates=len(delta),frame_delta=dict(collections.Counter(delta)),
  selected_slot_parity=dict(collections.Counter(parity)),paired_slot_frame_offset=dict(collections.Counter(partner)))
lo=arrays['fit'].min(0);hi=arrays['fit'].max(0)
outside=(arrays['held']<lo)|(arrays['held']>hi)
scalar=[]
for j,name in enumerate(SCALARS):
 i=384+j
 scalar.append(dict(name=name,index=i,fit_min=float(lo[i]),fit_max=float(hi[i]),fit_scale=float(scale[i]),
  held_min=float(arrays['held'][:,i].min()),held_max=float(arrays['held'][:,i].max()),
  held_outside_fit_range_fraction=float(outside[:,i].mean())))
clipped=np.flatnonzero(scale<=1.00001e-6)
changed=[int(i) for i in clipped if np.any(arrays['held'][:,i]!=arrays['fit'][0,i])]
result=dict(scope='exact registered within-video8/8; both original shards; CPU input read only',source_revision=REV,
 counts=counts,scalars=scalar,clipped_columns=clipped.tolist(),clipped_columns_with_held_change=changed,
 held_max_abs_standardized=float(np.abs((arrays['held']-mean)/scale).max()),
 new_cf_queries=0,optimizer_updates=0,model_forward_evaluations=0,
 interpretation='Range excursions alone are not protocol errors; no resplitting or gate changes')
(out/'WITHIN_SUPPORT_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
result=json.loads(ssh('CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 '+PYTHON+' -',
    ('ROOT='+repr(ROOT)+'\nREV='+repr(REV)+'\n'+body).encode()))
(OUT/'R1_WITHIN_SUPPORT_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
