"""Joint task learning; feature, output and self targets have distinct sources."""
import torch
import torch.nn.functional as F
from h65.frame.objectives import native_weights,feature_distance,bernoulli_kl
from h65.full.objectives import auxiliary_losses
from .geometry import feature_target_data,candidate_mask


def objectives(model,data,plan_index):
    cfg=model.config;weights=cfg['loss']
    native,detail=model.forward_native(data,force_plan=plan_index)
    losses=model.readout.loss(native,data)
    result={'task':losses['cost']};cost=weights.get('task',1.)*losses['cost']
    canonical=feature_target_data(data);weight,valid=native_weights(canonical)
    need_full=bool(weights.get('self_feature',0) or weights.get('full_gt',0))
    full=None;counts=dict(external_teacher=0,shared_full=0)
    if need_full and not cfg.get('dense_baseline',False):
        if detail['plan']['frames']==768 and detail['plan']['depth']==1 and detail['plan']['space']==1:
            full=native
        else:
            full,_=model.forward_native(data,force_plan=0,preview=detail['preview'],apply_refiner=False)
            counts['shared_full']=1
        if weights.get('full_gt',0):
            value=model.readout.loss(full,data)['cost'];result['full_task']=value;cost=cost+weights['full_gt']*value
        if weights.get('self_feature',0):
            value=feature_distance(native,full.detach(),weight);result['self_feature']=value;cost=cost+weights['self_feature']*value
    target=None
    if model.teacher is not None and (weights.get('feature',0) or weights.get('output_kd',0)):
        target=model.teacher.dense_native(data['inputs']);counts['external_teacher']=1
    if weights.get('feature',0) and target is not None:
        value=feature_distance(native,target,weight);result['feature']=value;cost=cost+weights['feature']*value
    if weights.get('difference',0) and target is not None:
        pair=valid[:,1:]&valid[:,:-1]
        value=F.smooth_l1_loss(native.diff(dim=-1),target.detach().float().diff(dim=-1),reduction='none').mean(1)
        value=(value*pair).sum()/pair.sum().clamp_min(1);result['difference']=value;cost=cost+weights['difference']*value
    if weights.get('output_kd',0):
        if model.teacher is None or model.readout.family!='point':raise ValueError('Output KD requires compatible point heads')
        student=model.readout.raw_head(native,data['masks'])
        with torch.no_grad():teacher=model.teacher.raw_head(target,data['masks'])
        kd=bernoulli_kl(student['logits'],teacher['logits'],student['valid'])
        confidence=teacher['logits'].sigmoid().amax(-1)*student['valid']
        delta=F.smooth_l1_loss(student['offsets'],teacher['offsets'],reduction='none').mean(-1)
        kd=kd+.1*(delta*confidence).sum()/confidence.sum().clamp_min(1)
        result['output_kd']=kd;cost=cost+weights['output_kd']*kd
    if cfg.get('train_scout',True) and not cfg.get('dense_baseline',False):
        aux=auxiliary_losses(detail['preview'],detail['selection'],canonical['masks'],canonical['gt_segments'],
                             dict(action=.25,transition=.1,boundary=.25),canonical['gt_boundary_validity'])
        for key,value in aux.items():result[key]=value;cost=cost+value
    result['cost']=cost
    return result,detail,counts
