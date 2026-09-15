"""Register native baselines in the existing persistent paper queue; no new controller."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from h65.paper.runtime import json_write


def build(root):
    root=Path(root); stages={}; configs=[]
    for bb in ('s','b'):
        for k in (384,768):
            ident=f'native_adatad_{bb}_uniform_k{k}_seed42'
            cfg=dict(id=ident,recipe='native_adatad_uniform_v1',dataset='thumos',backbone=bb,head='point',frames=k,seed=42,
                epochs=80,batch_size=2,head_lr=1e-4,adapter_lr=2e-4,weight_decay=.05,warmup_epochs=5,ema_decay=.999,
                eval_epochs=[10,20,40,60,80],comparison='Native AdaTAD '+('uniform half-rate' if k==384 else 'dense reference'),
                initialization='provided official dense EMA',objective='GT only',
                training_protocol='Official adapter/head optimizer groups; 80 new epochs with 5 warmup epochs and cosine decay; BF16',
                provenance='Upstream AdaTAD model with input/time-axis sampling adaptation; not an official published sparse result')
            configs.append(cfg); config=str(root/'configs/native_adatad'/f'{ident}.json')
            out=root/'research/paper/runs'/ident; tool=str(root/'tools/native_adatad_run.py')
            resources=str(root/'research/paper/resources.local.json')
            common=[tool,'--config',config,'--resources',resources]
            key='eval_'+ident+'_official'
            evaluation=[*common,'--mode','eval']
            if k==384:
                evaluation=[*common,'--mode','eval-course','--preflight-output',str(root/'research/paper/runs'/('preflight_'+ident))]
            stages[key]=dict(kind='eval',config_id=ident,priority=-3 if k==384 else -2,dependencies=[],assets=[],
                done=str(out/'official_ema/completed.json'),args=[*evaluation,'--output',str(out/'official_ema')])
            if k!=384: continue
            check='preflight_'+ident; train='train_'+ident
            audit=root/'research/paper/runs'/check
            stages[check]=dict(kind='inline_preflight',config_id=ident,runs_with=key,dependencies=[],assets=[],done=str(audit/'completed.json'))
            stages[train]=dict(kind='train',config_id=ident,priority=-1,dependencies=[],assets=[],contains_preflight=check,
                requires=[str(audit/'completed.json')],
                done=str(out/'completed.json'),resume_checkpoint=str(out/'latest.pth'),
                args=[*common,'--mode','course','--preflight-output',str(audit),'--output',str(out),'--slice-hours','10'])
            stages['eval_'+ident+'_terminal_learned']=dict(kind='eval',config_id=ident,priority=15,dependencies=[],assets=[],
                requires=[str(out/'terminal.pth')],done=str(out/'terminal_learned/completed.json'),
                args=[*common,'--mode','eval','--checkpoint',str(out/'terminal.pth'),'--state','learned','--output',str(out/'terminal_learned')])
    return dict(recipe='native_adatad_uniform_v1',configs=configs,stages=stages,new_training_courses=2,
                frozen_official_evaluations=4,new_stages=len(stages),seed=42,performance_gates=False,
                preserves_existing_courses=75,total_training_courses=77)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime-root',default=str(ROOT));p.add_argument('--owner-plan');a=p.parse_args()
    plan=build(a.runtime_root)
    for cfg in plan['configs']: json_write(ROOT/'configs/native_adatad'/f'{cfg["id"]}.json',cfg)
    json_write(ROOT/'research/paper/native_adatad/plan.json',plan)
    if a.owner_plan:
        path=Path(a.owner_plan); owner=json.loads(path.read_text())
        for name,stage in plan['stages'].items():
            if name in owner['stages'] and owner['stages'][name]!=stage: raise ValueError('Existing native stage differs: '+name)
            owner['stages'].setdefault(name,stage)
        owner['native_adatad_extension']=dict(configs=4,training_courses=2,stages=10,seed=42)
        json_write(path,owner)
    print(json.dumps(dict(configs=len(plan['configs']),stages=len(plan['stages']),new_training_courses=2)))
