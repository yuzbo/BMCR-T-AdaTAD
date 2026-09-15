"""Read exact registered8/8 packing coordinates; no model/GT computation."""
from deploy_autodl import ssh,PYTHON,ROOT,OUT
import json
body='''from pathlib import Path
import json,collections,numpy as np
root=Path(ROOT);out=root/'results/b644d87/VALUE_R1'
split=json.loads((out/'config.json').read_text())['within_split']
counts={name:{key:collections.Counter() for key in ['remove_clip_slot','insert_clip_slot','tubelet_position','frame_delta']} for name in ['fit','held']}
states=[];similarity=[]
for shard in [0,1]:
 folder=root/'results/2ca4d4b'/f'mini_c40_shard{shard}'/'groups'
 for path in sorted(folder.glob('*.json')):
  row=json.loads(path.read_text())
  if row['state_key'] not in split:continue
  selected=[frame for frame,valid in zip(row['selected_frame_ids'],row['selected_valid']) if valid]
  assert len(selected)==384
  slots={frame:i for i,frame in enumerate(selected)}
  local={name:[] for name in ['fit','held']}
  for pair,action in zip(row['action_pairs'],row['actions']):
   remove,insert=pair
   arm='fit' if action['id'] in split[row['state_key']]['fit'] else 'held'
   assert action['id'] in split[row['state_key']][arm]
   before=slots[remove];after=sorted((set(selected)-{remove})|{insert}).index(insert)
   values={'remove_clip_slot':before%16,'insert_clip_slot':after%16,'tubelet_position':(before//2)%8,'frame_delta':insert-remove}
   for key,value in values.items():counts[arm][key][value]+=1
   local[arm].append(dict(id=action['id'],selected_slot=before,**values))
  with np.load(folder/row['arrays_file']) as arrays:
   vector=arrays['descriptor'][0];support=vector[192:288];global_=vector[288:384]
   cosine=float(np.dot(support,global_)/(np.linalg.norm(support)*np.linalg.norm(global_)))
  similarity.append(cosine);states.append(dict(state_key=row['state_key'],video_id=row['video_id'],rows=local,support_global_cosine=cosine))
assert len(states)==25
assert all(sum(counts[name]['frame_delta'].values())==200 for name in counts)
result=dict(scope='original locked25-state8/8; packing readback only',source_revision='b644d870d1845abbc1e4fd5ab7780f29ff96a53a',
 packing='encoder.py:105–111 packs16 selectedframes,8 tubelets perclip',counts=counts,states=states,
 support_global_cosine=dict(min=min(similarity),mean=float(np.mean(similarity)),max=max(similarity)),
 new_cf_queries=0,optimizer_updates=0,model_forward_evaluations=0,
 interpretation='Observed operator-position support only; no split/gate change or causal attribution')
(out/'PACKING_SUPPORT_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
result=json.loads(ssh(PYTHON+' -',('ROOT='+repr(ROOT)+'\n'+body).encode()))
(OUT/'R1_PACKING_SUPPORT_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps({key:value for key,value in result.items() if key!='states'},indent=2))
