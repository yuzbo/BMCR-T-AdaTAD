"""Generate registered-schema controls, not new-method flags that paper_train would ignore.

Only writes new JSON files in --output. Never edits running configs/plan/dispatcher.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from .core import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--schedule-epochs',type=int,default=80)
    args=p.parse_args()
    base=json.loads(Path(args.base).read_text())
    if base.get('recipe')!='support_consistent_paper_v3' or base.get('mod_start')!=4:
        raise ValueError('Use the audited review5485_full_v2_S/B config as base.')
    if args.epochs>args.schedule_epochs or args.epochs<1:raise ValueError('Invalid horizon.')
    # Existing implementations only: plan-aware feature routing is a genuine registered control.
    variants={
        'fixed_k384_full_ds':dict(train_plan_mode='fixed',fixed_plan=1,dynamic_budget=False),
        'fixed_k384_amod_ds':dict(train_plan_mode='fixed',fixed_plan=4,dynamic_budget=False),
        'fixed_k384_uniform_ds':dict(train_plan_mode='fixed',fixed_plan=4,dynamic_budget=False,
                                     attention_uniform=True,depth_gate='uniform'),
        'mixed_planaware':dict(plan_aware_frame=True),
    }
    manifest=[]
    for name,changes in variants.items():
        cfg=copy.deepcopy(base);cfg.update(changes)
        cfg['id']=f'wtr_{name}_{cfg["backbone"]}_seed{cfg["seed"]}'
        cfg.update(epochs=args.epochs,schedule_epochs=args.schedule_epochs,
                   eval_epochs=sorted({e for e in [10,20,40,60,args.epochs] if e<=args.epochs}),
                   comparison='WTR_CONTROL_'+name,hypothesis='Control, not a proven WTR improvement.')
        target=Path(args.output)/(cfg['id']+'.json')
        if target.exists():raise FileExistsError(target)
        atomic_json(target,cfg);manifest.append(dict(config_id=cfg['id'],config=str(target),
             status='GENERATED_NOT_SUBMITTED',epochs=args.epochs,source='verified existing paper_train flags'))
    atomic_json(Path(args.output)/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
