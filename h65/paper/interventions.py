"""Real frame and T/D/S capacity actions; no teacher substitution is called an action."""
from contextlib import contextmanager
import torch
from .routing import candidate_pairs,swap_selection
from h65.frame.measure import matrix_counter
from .geometry import candidate_mask


@contextmanager
def evaluation_state(model):
    modes=[(m,m.training) for m in model.modules()]
    model.eval()
    try:yield
    finally:
        for module,mode in modes:module.training=mode


def policy_pair(model,kind,number):
    field={'temporal':'frames','depth':'depth','spatial':'space'}[kind]
    pairs=[]
    for i,a in enumerate(model.menu):
        for j,b in enumerate(model.menu):
            if i>=j or a[field]==b[field]:continue
            if all(a[k]==b[k] for k in ('frames','depth','space') if k!=field):pairs.append((i,j))
    return pairs[number%len(pairs)]


@torch.no_grad()
def collect_action(model,data,kind,number=0,measure=False):
    if len(data['inputs'])!=1:raise ValueError('Paired actual interventions use one window')
    masks=candidate_mask(data)
    with evaluation_state(model),torch.autocast('cuda',dtype=torch.bfloat16):
        base,action=(1,None) if kind=='frame' else policy_pair(model,kind,number)
        def execute(index,selection=None,preview=None):
            if measure:
                with matrix_counter() as counter:
                    counter.phase=['student_and_task_head']
                    features,detail=model.forward_native(data,force_plan=index,selection=selection,preview=preview,apply_refiner=False)
                    losses=model.readout.loss(features,data)
                if counter.unresolved_matrix_ops():raise RuntimeError(counter.unresolved_matrix_ops())
                flops=2*sum(counter.macs.values())
            else:
                features,detail=model.forward_native(data,force_plan=index,selection=selection,preview=preview,apply_refiner=False)
                losses=model.readout.loss(features,data);flops=None
            return features,detail,model.readout.components(losses),flops
        f0,s0,l0,c0=execute(base)
        pair=None;frame_features=None
        if kind=='frame':
            pairs=candidate_pairs(s0['preview'],s0['selection'],masks,model.config.get('partner_scope','local'))
            if not pairs:return None
            pair=pairs[number%len(pairs)];row,remove,insert=pair
            frame_features=model.frame_router.features(s0['preview'],s0['selection'],masks,[pair])
            changed=swap_selection(s0['selection'],row,remove,insert)
            f1,s1,l1,c1=execute(base,changed,s0['preview'])
            mu,lv=model.frame_router.distribution(s0['preview'],s0['selection'],masks,[pair])
            predicted=mu[0]*model.frame_router.scales
            sigma=lv[0].mul(.5).exp()*model.frame_router.scales
            positions=torch.tensor(sorted({remove//2,insert//2}),device=f0.device)
        else:
            f1,s1,l1,c1=execute(action,preview=s0['preview'])
            mu,lv=model.budget_router.distribution(s0['context'])
            predicted=(mu[0,action]-mu[0,base])*model.budget_router.scales
            sigma=(lv[0,action].exp()+lv[0,base].exp()).sqrt()*model.budget_router.scales
            if kind=='temporal':
                a=torch.zeros_like(masks).scatter(1,s0['selection'].indices,s0['selection'].valid)
                b=torch.zeros_like(masks).scatter(1,s1['selection'].indices,s1['selection'].valid)
                positions=(a^b).reshape(1,-1,2).any(-1)[0].nonzero().flatten()
            else:
                key='depth_masks' if kind=='depth' else 'spatial_masks'
                changed=(torch.stack(s0['trace'][key])!=torch.stack(s1['trace'][key])).any(0)
                changed=changed.reshape(1,s0['anchors'].features.shape[1],-1).any(-1)[0]
                nearest=(s0['queries'].centers[0,:,None]-s0['anchors'].centers[0,None]).abs().argmin(-1)
                positions=changed[nearest].nonzero().flatten()
        if model.teacher is not None:
            target=model.teacher.dense_native(data['inputs']);repair_source='external_official'
        else:
            target,_=model.forward_native(data,force_plan=0,preview=s0['preview'],apply_refiner=False);repair_source='shared_full_student'
        repaired=f0.clone();positions=positions[s0['queries'].valid[0,positions]]
        repaired[0,:,positions]=target[0,:,positions]
        repair_loss=model.readout.components(model.readout.loss(repaired,data))
        result=dict(video_name=data['metas'][0]['video_name'],action_type=kind,base_plan=base,action_plan=action,
                    frame_pair=pair,actual_delta=(l0-l1).cpu().tolist(),repair_delta=(l0-repair_loss).cpu().tolist(),
                    predicted_delta=predicted.cpu().tolist(),predicted_sigma=sigma.cpu().tolist(),
                    context=s0['context'][0].cpu().tolist(),frame_features=None if frame_features is None else frame_features[0].cpu().tolist(),
                    actual_reencoded=True,actual_affected_graph='complete current student including global TIA and decoder',
                    repair_scope='query positions nearest changed native states; representation proxy only',
                    repair_source=repair_source,repair_positions=positions.cpu().tolist(),
                    base_forward_flops=c0,action_forward_flops=c1,
                    delta_forward_flops=None if c0 is None else c1-c0,
                    cost_scope='actual student forward and task-head forward; paired scoring bypasses frame-refiner decisions')
        return result


def action_loss(model,record,device):
    target=torch.tensor([record['actual_delta']],device=device,dtype=torch.float32)
    if record['action_type']=='frame':
        router=model.frame_router;router.update_scales(target)
        return router.regression_loss(torch.tensor([record['frame_features']],device=device),target)
    router=model.budget_router
    with torch.no_grad():router.scales.mul_(.9).add_(target.abs().mean(0).clamp_min(1e-4),alpha=.1)
    return router.pair_loss(torch.tensor([record['context']],device=device),record['base_plan'],record['action_plan'],target)
