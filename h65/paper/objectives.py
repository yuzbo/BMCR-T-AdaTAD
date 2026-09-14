"""Joint task learning; feature, output and self targets have distinct sources."""
import torch
import torch.nn.functional as F
from h65.frame.objectives import native_weights,feature_distance,bernoulli_kl
from h65.full.objectives import auxiliary_losses
from .geometry import feature_target_data,candidate_mask


def objectives(model,data,plan_index):
    cfg=model.config;weights=cfg['loss']
    native,detail=model.forward_native(data,force_plan=plan_index,capture_support=cfg.get('support_reference',False))
    losses=model.readout.loss(native,data)
    result={'task':losses['cost']};cost=weights.get('task',1.)*losses['cost']
    if cfg.get('graph_recovery') and detail['plan_index']!=0 and cfg.get('anchor_consistency_weight',0):
        from .graph_recovery import anchor_consistency
        value=anchor_consistency(native,detail['anchors'],detail['queries'],model.encoder.depth)
        result['anchor_consistency']=value;cost=cost+cfg['anchor_consistency_weight']*value
    canonical=feature_target_data(data);weight,valid=native_weights(canonical)
    need_full=bool(weights.get('self_feature',0) or weights.get('full_gt',0))
    full=None;counts=dict(external_teacher=0,shared_full=0)
    from .profile import execution_flops
    counts['student_forward_gflops']=execution_flops(model,detail)/1e9
    if cfg.get('support_reference',False):
        from .support_targets import same_support_targets
        terms=same_support_targets(model,data,detail);counts['support_teacher']=1
        counts['support_forward_gflops']=detail['support_forward_gflops']
        for key,value in terms.items():result['support_'+key]=value
        if cfg.get('support_loss',False):cost=cost+cfg.get('support_weight',.1)*(terms['pre_tia']+terms['post_tia'])/2
    if need_full and not cfg.get('dense_baseline',False):
        if detail['plan']['frames']==768 and detail['plan']['depth']==1 and detail['plan']['space']==1:
            full=native
        else:
            full,full_detail=model.forward_native(data,force_plan=0,preview=detail['preview'],apply_refiner=False)
            counts['shared_full']=1
            counts['shared_full_forward_gflops']=execution_flops(model,full_detail)/1e9
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
