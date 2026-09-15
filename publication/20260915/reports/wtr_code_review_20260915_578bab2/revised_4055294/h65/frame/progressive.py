"""Training-only progressive least-damage block deletion, with latent alignment.

PBD-style TAD adaptation; preserves first/last layers. Not an official PBD
reproduction: criterion is our frozen TAD loss + aligned native feature loss.
"""
from dataclasses import replace
import torch
from .objectives import training_objectives

def drop_one(model,data,target):
    keep=list(model.policy.static_keep or range(12));candidates=[i for i in keep if i not in (0,11)]
    was_training=model.training;model.eval();rows=[]
    try:
        with torch.no_grad():
            selection,_,_=model.anchor.select(data['inputs'],data['masks'],data['metas'])
            baseline,_=model.forward_native(data,selection=selection,policy=replace(model.policy,mode='compact'),apply_router=False)
            base=float(training_objectives(baseline,target,data,model.teacher)['cost'])
            for layer in candidates:
                remaining=tuple(i for i in keep if i!=layer)
                value,_=model.forward_native(data,selection=selection,policy=replace(model.policy,static_keep=remaining,static_depth=len(remaining),mode='compact'),apply_router=False)
                cost=float(training_objectives(value,target,data,model.teacher)['cost'])
                rows.append(dict(drop=layer,objective=cost,delta=cost-base))
        chosen=min(rows,key=lambda r:(r['objective'],r['drop']))['drop'];keep=tuple(i for i in keep if i!=chosen)
        model.policy=replace(model.policy,static_keep=keep,static_depth=len(keep))
        return dict(selected_drop=chosen,remaining=keep,candidates=rows,baseline=base,selection_source='current training microbatch; no test labels',extra_student_forwards=len(rows)+1)
    finally:model.train(was_training)
