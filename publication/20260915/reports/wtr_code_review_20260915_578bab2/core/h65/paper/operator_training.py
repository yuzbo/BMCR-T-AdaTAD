"""Actual D/S exchanges; both branches re-execute the same current continuation policy."""
import torch
import torch.nn.functional as F
from .interventions import evaluation_state
from h65.atlas.reference import deterministic_fp32


def collect_operator_action(model,data,sequence):
    axes=(['T'] if model.config.get('temporal_value') else [])+[a for a in ('D','S') if model.config['operator_policy'][a]=='value']
    if not axes:return None
    axis=axes[sequence%len(axes)]
    if axis=='T':
        from .temporal_value import collect_temporal_action
        return collect_temporal_action(model,data,sequence)
    layer=[4,6,8,10][sequence//len(axes)%4]
    plan=model.plan(model.config['fixed_plan']);plan['wtr_capture']=True
    with evaluation_state(model),deterministic_fp32():
        baseline,detail=model.forward_native(data,force_plan=plan,apply_refiner=False)
        decision=detail['trace']['operator_decisions'][(axis,layer)]
        mask,allowed=decision['mask'],decision['allowed']
        generator=torch.Generator().manual_seed(model.config['seed']+sequence)
        candidates=[]
        for pack in range(len(mask)):
            # S exchanges preserve the allocated native-time quota as well as packed total.
            groups=range(8) if axis=='S' else [None]
            for native in groups:
                lo,hi=(0,mask.shape[1]) if native is None else (native*(mask.shape[1]//8),(native+1)*(mask.shape[1]//8))
                keep=mask[pack,lo:hi].nonzero().flatten()+lo
                drop=(allowed[pack,lo:hi]&~mask[pack,lo:hi]).nonzero().flatten()+lo
                if len(keep) and len(drop):candidates.append((pack,keep,drop))
        if not candidates:return None
        pack,kept,dropped=candidates[int(torch.randint(len(candidates),(1,),generator=generator))]
        remove=int(kept[int(torch.randint(len(kept),(1,),generator=generator))])
        insert=int(dropped[int(torch.randint(len(dropped),(1,),generator=generator))])
        override=dict(axis=axis,layer=layer,pack=pack,remove=remove,insert=insert)
        before=model.readout.detector.rpn_head.loss_normalizer.detach().clone()
        try:
            base_loss=model.readout.components(model.readout.loss(baseline.float(),data)).detach()
            changed_plan=dict(plan,wtr_override=override,wtr_capture=False)
            changed,changed_detail=model.forward_native(data,force_plan=changed_plan,
                selection=detail['selection'],preview=detail['preview'],apply_refiner=False)
            changed_loss=model.readout.components(model.readout.loss(changed.float(),data)).detach()
        finally:model.readout.detector.rpn_head.loss_normalizer=before
        gain=base_loss-changed_loss
        if not torch.isfinite(gain).all():raise RuntimeError('Nonfinite operator counterfactual')
        # Every mask is recomputed downstream. Only the requested local action is forced.
        base_d=detail['trace']['depth_masks'][layer]
        new_d=changed_detail['trace']['depth_masks'][layer]
        base_s=detail['trace']['spatial_masks'][layer]
        new_s=changed_detail['trace']['spatial_masks'][layer]
        if axis=='S' and not torch.equal(base_d,new_d):raise RuntimeError('S CF changed its common attention support')
        if not torch.equal(base_d.sum(-1),new_d.sum(-1)) or not torch.equal(base_s.sum(-1),new_s.sum(-1)):
            raise RuntimeError('Operator CF changed packed capacity')
        from .profile import execution_flops
        detail['plan_index']=changed_detail['plan_index']=model.config['fixed_plan']
        total_cost=(execution_flops(model,detail)+execution_flops(model,changed_detail))/1e9
        payload={key:value[pack:pack+1].detach().cpu() for key,value in decision.items()}
        payload.update(axis=axis,layer=layer,remove=remove,insert=insert,gain_cls_loc=gain.cpu(),
            selected_frame_slots=detail['selection'].indices.cpu(),selected_valid=detail['selection'].valid.cpu(),
            frame_ids=[list(map(int,meta['frame_inds'])) for meta in data['metas']],
            video_id=data['metas'][0]['video_name'],sequence=sequence,
            depth_masks=[x.cpu() for x in detail['trace']['depth_masks']],
            spatial_masks=[x.cpu() for x in detail['trace']['spatial_masks']])
        record=dict(action=override,video_id=data['metas'][0]['video_name'],gain_cls_loc=gain.cpu().tolist(),
            base_loss=base_loss.cpu().tolist(),changed_loss=changed_loss.cpu().tolist(),forward_gflops=total_cost,
            query_forwards=2,changed_depth_tokens=int((base_d!=new_d).sum()),
            changed_ffn_tokens=int((base_s!=new_s).sum()),
            continuation='same current policy re-executed; S recomputed after D; common D before S action',
            split='training',candidate_set='same pre-action legal packed tokens',detector_update=0)
    # Train the adapter and head from the detached actual decision state; hard mask has no surrogate gradient.
    router=model.encoder.engine.value_router
    state=payload['state'].to(gain.device);valid=payload['valid'].to(gain.device)
    geo=payload['geometry'].to(gain.device);allowed=payload['allowed'].to(gain.device)
    with torch.autocast('cuda',enabled=False):
        prediction,_=router(axis,state,valid,geo,layer,model.encoder.depth,allowed)
        difference=prediction[0,insert]-prediction[0,remove]
        scale=model.config.get('operator_gain_scale',.01)
        loss=F.smooth_l1_loss(difference/scale,gain.float()/scale)
    record['predicted_gain_cls_loc']=difference.detach().cpu().tolist()
    return loss,record,payload
