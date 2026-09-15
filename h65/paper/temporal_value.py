"""The same Q_T(s, C_T, m) interface for Standard and bounded Raw acquisition."""
from pathlib import Path
import torch
from torch import nn
from h65.raw.contracts import EpisodePublic,PreviewTimeline,RawSelection,proposals,geometric_swaps,swap,repartition
from h65.raw.value import TemporalValueHead,descriptors
from h65.transport import Selection


def public_state(output,selection,masks,metas):
    if len(masks)!=1:raise ValueError('Fast-Track temporal policy uses one episode per batch')
    meta=metas[0];valid=tuple(masks[0].tolist());n=sum(valid)
    frame_ids=tuple(int(i) for i in meta['frame_inds'])
    fps=float(meta['fps'])
    ep=EpisodePublic(meta['video_name'],int(frame_ids[0]),'policy',
        str(Path(meta.get('data_path',''))/(meta['video_name']+'.mp4')),fps,int(meta['total_frames']),
        float(meta['duration']),frame_ids[0],int(meta['snippet_stride']),frame_ids,valid)
    timeline=PreviewTimeline(frame_ids[:n],tuple(i/fps for i in frame_ids[:n]),(True,)*n,frame_ids[:n])
    preview={k:v[:,:n] if isinstance(v,torch.Tensor) and v.ndim>=2 and v.shape[1]==768 else v for k,v in output.items()}
    proposal=proposals(ep,timeline,'O')
    support=RawSelection(tuple(frame_ids[i] for i in selection.indices[0].tolist()),tuple(selection.valid[0].tolist()))
    return ep,timeline,preview,proposal,support


def to_standard(raw,old,frame_ids,candidate_valid):
    # Only valid official occurrences are eligible. Repeated suffix frame IDs
    # must never redirect a real observation to a padded candidate position.
    if candidate_valid.shape!=(1,len(frame_ids)):
        raise ValueError('Standard candidate validity must match the episode')
    lookup={}
    for i,(frame,valid) in enumerate(zip(frame_ids,candidate_valid[0].tolist())):
        if valid:lookup.setdefault(int(frame),i)
    if any(int(frame) not in lookup for frame in raw.support):
        raise ValueError('A real temporal observation is outside valid candidates')
    ids=torch.tensor([[lookup[int(frame)] for frame in raw.frame_ids]],device=old.indices.device)
    valid=torch.tensor([raw.valid],dtype=torch.bool,device=old.indices.device)
    if not torch.equal(valid,old.valid) or not bool(candidate_valid.gather(1,ids)[valid].all()):
        raise ValueError('Temporal selection changed valid capacity or selected padding')
    return Selection(ids,ids.float(),valid,old.density,old.rates)


class TemporalCoreRouter(nn.Module):
    def __init__(self):
        super().__init__();self.network=TemporalValueHead()
        self.network.target_scale.fill_(.01)

    def refine(self,output,selection,masks,metas,plan):
        ep,timeline,preview,proposal,current=public_state(output,selection,masks,metas)
        changes=[];count=0
        with torch.no_grad():
            for round_index in range(4):
                pairs=geometric_swaps(ep,proposal,current,16,round_index)
                if not pairs:break
                x=descriptors(ep,timeline,preview,proposal,current,pairs,plan)
                gain=self.network.utility(x.float());count+=len(pairs)
                index=int(gain.argmax())
                if float(gain[index])<=0:break
                before=current;current=swap(current,*pairs[index],proposal)
                changes.append(dict(remove=pairs[index][0],insert=pairs[index][1],predicted_gain=float(gain[index]),
                                    **repartition(before,current,ep.fps)))
        return to_standard(current,selection,ep.official_frame_ids,masks),dict(changes=changes,pair_count=count)


def collect_temporal_action(model,data,sequence):
    from .interventions import evaluation_state
    from h65.atlas.reference import deterministic_fp32
    from .profile import execution_flops
    with evaluation_state(model),deterministic_fp32():
        base,detail=model.forward_native(data,force_plan=model.config['fixed_plan'])
        ep,timeline,preview,proposal,current=public_state(detail['preview'],detail['selection'],data['masks'],data['metas'])
        pairs=geometric_swaps(ep,proposal,current,16,sequence%4)
        if not pairs:return None
        pair=pairs[sequence%len(pairs)]
        x=descriptors(ep,timeline,preview,proposal,current,[pair],detail['plan']).detach()
        changed=swap(current,*pair,proposal)
        changed_selection=to_standard(changed,detail['selection'],ep.official_frame_ids,data['masks'])
        before=model.readout.detector.rpn_head.loss_normalizer.detach().clone()
        try:
            base_loss=model.readout.components(model.readout.loss(base,data)).detach()
            after,new_detail=model.forward_native(data,force_plan=model.config['fixed_plan'],
                selection=changed_selection,preview=detail['preview'],apply_refiner=False)
            after_loss=model.readout.components(model.readout.loss(after,data)).detach()
        finally:model.readout.detector.rpn_head.loss_normalizer=before
        gain=base_loss-after_loss
        cost=(execution_flops(model,detail)+execution_flops(model,new_detail))/1e9
        record=dict(action=dict(axis='T',remove=pair[0],insert=pair[1]),video_id=ep.video_id,
            gain_cls_loc=gain.cpu().tolist(),base_loss=base_loss.cpu().tolist(),changed_loss=after_loss.cpu().tolist(),
            forward_gflops=cost,query_forwards=2,continuation='same current D/S/recovery policy re-executed',
            split='training',candidate_set='O',**repartition(current,changed,ep.fps))
        payload=dict(selected_frame_slots=detail['selection'].indices.cpu(),selected_valid=detail['selection'].valid.cpu(),
            frame_ids=[list(ep.official_frame_ids)],descriptor=x.cpu(),gain_cls_loc=gain.cpu())
    with torch.autocast('cuda',enabled=False):
        prediction=model.frame_router.network(x.float())[0]
        scale=model.config.get('operator_gain_scale',.01)
        loss=torch.nn.functional.smooth_l1_loss(prediction/scale,gain.float()/scale)
    record['predicted_gain_cls_loc']=prediction.detach().cpu().tolist()
    return loss,record,payload
