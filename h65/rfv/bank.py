"""Shared Standard-T state/action capture and exact fixed-support re-execution."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from h65.raw.contracts import geometric_swaps,swap,repartition,RawSelection
from h65.raw.value import descriptors,FEATURE_DIM
from h65.raw.model import interpolate_preview
from h65.paper.temporal_value import public_state,to_standard
from h65.paper.geometry import candidate_mask
from h65.paper.runtime import json_write,build_config
from h65.atlas.reference import deterministic_fp32
from h65.frame.measure import matrix_counter


def revision(root):
    return (Path(root)/'WTR_RFV_SCIENCE_SHA').read_text().strip()


def checkpoint_identity(payload,path,state):
    meta=payload['metadata']
    return dict(path=str(path),original_checkpoint=payload.get('original_checkpoint',str(path)),
        parameter_state=state,epoch=payload.get('epoch_index'),updates=payload.get('successful_updates'),
        source_revision=meta['source_revision'],course_id=meta.get('config',{}).get('id'),
        train_scout=meta.get('config',{}).get('train_scout'))


class FixedTModel:
    def __init__(self,cfg,resources,checkpoint=None,state='ema'):
        from h65.paper.model import PaperModel
        from h65.paper.fasttrack import initialize
        from h65.full.runtime import seed_all
        seed_all(cfg['seed'])
        self.cfg=copy.deepcopy(cfg);self.resources=resources
        self.model_cfg=build_config(cfg,resources)
        self.model=PaperModel(self.model_cfg,cfg,resources,with_teacher=False).cuda().float().eval()
        self.initialization=initialize(self.model,resources)
        path=checkpoint or resources['wtr_initialization']
        payload=torch.load(path,map_location='cpu')
        self.identity=checkpoint_identity(payload,path,state)
        own=self.model.state_dict();loaded=[];not_executed=[]
        for key,value in payload[state].items():
            if key.startswith(('frame_router.','budget_router.','encoder.engine.value_router.')):
                not_executed.append(key);continue
            if key not in own:
                raise ValueError('Checkpoint detector tensor is absent from RFV: '+key)
            if own[key].shape!=value.shape:raise ValueError('Checkpoint shape differs: '+key)
            own[key].copy_(value.to(own[key]));loaded.append(key)
        if not any(k.startswith('decoder.') for k in loaded) or not any(k.startswith('readout.') for k in loaded):
            raise ValueError('RFV needs complete corresponding checkpoint detector/recovery values')
        self.identity.update(restored_detector_tensors=len(loaded),unexecuted_router_keys=not_executed,
            replay_policy='Standard K384; fixed support; D100/S100; full current Cross/readout; refiner bypassed')
        self.model.requires_grad_(False)

    @torch.no_grad()
    def preview(self,data):
        with matrix_counter() as counter:
            counter.phase=['rfv_cheap_preview']
            value=self.model.encoder.preview(data['inputs'],candidate_mask(data))
        if counter.unresolved_matrix_ops():raise RuntimeError(counter.unresolved_matrix_ops())
        return value,2*sum(counter.macs.values())/1e9

    def seed(self,data,preview):
        return self.model.encoder.select(preview,candidate_mask(data),384,'uniform',data['metas'])

    @torch.no_grad()
    def execute(self,data,selection,preview,measure=False):
        def forward():
            native,detail=self.model.forward_native(data,force_plan=self.cfg['fixed_plan'],
                selection=selection,preview=preview,apply_refiner=False)
            loss=self.model.readout.components(self.model.readout.loss(native,data)).detach()
            if not bool(torch.isfinite(loss).all()):raise RuntimeError('Nonfinite actual T loss')
            return native,detail,loss
        if measure:
            with matrix_counter() as counter:
                counter.phase=['rfv_fixed_support_and_task_loss']
                native,detail,loss=forward()
            if counter.unresolved_matrix_ops():raise RuntimeError(counter.unresolved_matrix_ops())
            flops=2*sum(counter.macs.values())/1e9
        else:
            native,detail,loss=forward();flops=None
        return native,detail,loss,flops


def cheap_graph_state(ep,timeline,preview,support,pairs):
    """192 physical nodes derived from the same Standard768 cheap observation."""
    device=preview['hidden'].device
    lo,hi=ep.bounds;span=max(hi-lo,1)
    times=torch.linspace(0,1,192,device=device)
    frame_nodes=(lo+times*(hi-lo)).tolist()
    hidden=interpolate_preview(preview['hidden'][0].detach(),timeline.frame_ids,frame_nodes)
    actionness=interpolate_preview(preview['action_logits'][0,:,None].detach(),timeline.frame_ids,frame_nodes).sigmoid().flatten()
    transition=interpolate_preview(preview['transition_logits'][0,:,None].detach(),timeline.frame_ids,frame_nodes).flatten()
    selected=torch.tensor([(x-lo)/span for x in support.support],device=device,dtype=torch.float32)
    right=torch.searchsorted(selected,times).clamp(0,len(selected)-1)
    left=(right-1).clamp(0)
    distance=torch.minimum((times-selected[left]).abs(),(times-selected[right]).abs())
    buckets=torch.bucketize(selected,(times[:-1]+times[1:])/2)
    occupancy=torch.bincount(buckets,minlength=192).float()/max(len(selected)/192,1.)
    return dict(cheap=hidden,times=times,node_valid=torch.ones(192,device=device,dtype=torch.bool),
        actionness=actionness,transition=transition,support_distance=distance,support_occupancy=occupancy,
        remove_times=torch.tensor([(a-lo)/span for a,b in pairs],device=device),
        insert_times=torch.tensor([(b-lo)/span for a,b in pairs],device=device),
        support_times=torch.tensor([(x-lo)/span for x in support.frame_ids],device=device),
        support_valid=torch.tensor(support.valid,device=device,dtype=torch.bool))


def action_strata(data,pairs):
    # Supervision-only bookkeeping, never part of cheap_graph_state/descriptor.
    from h65.atlas.data import window_metadata,position_metadata
    meta=window_metadata(data)
    slots={int(frame):i for i,frame in reversed(list(enumerate(meta['frame_indices'])))}
    output=[]
    for remove,insert in pairs:
        a=position_metadata(meta,slots[remove]);b=position_metadata(meta,slots[insert])
        regions={a['region'],b['region']}
        region='boundary' if regions&{'start','end'} else 'interior' if regions&{'interior','overlap'} else 'background'
        durations=[x['action_duration'] for x in (a,b) if x.get('action_duration') is not None]
        output.append(dict(region=region,action_duration=min(durations) if durations else None))
    return output


@torch.no_grad()
def collect_state(runtime,data,preview,selection,round_index=0,pairs=None):
    start=time.perf_counter();masks=candidate_mask(data)
    ep,timeline,public,proposal,current=public_state(preview,selection,masks,data['metas'])
    offered=geometric_swaps(ep,proposal,current,16,round_index) if pairs is None else [tuple(p) for p in pairs]
    native,detail,base_loss,one_cost=runtime.execute(data,selection,preview,measure=True)
    x=descriptors(ep,timeline,public,proposal,current,offered,detail['plan']).detach()
    if x.shape!=(len(offered),FEATURE_DIM):raise ValueError('Temporal descriptor shape differs')
    values=[];actions=[];best_gain=0.;best_selection=selection;best_native=native;best_loss=base_loss
    replay_error=0.;no_op_error=0.
    if offered:
        _,_,again,_=runtime.execute(data,selection,preview)
        no_op_error=float((again-base_loss).abs().max())
    for index,pair in enumerate(offered):
        changed=swap(current,*pair,proposal)
        mapped=to_standard(changed,selection,ep.official_frame_ids,masks)
        feature,_,loss,_=runtime.execute(data,mapped,preview)
        gain=base_loss-loss;values.append(gain)
        repeat_error=None
        if index==0:
            _,_,repeat,_=runtime.execute(data,mapped,preview)
            repeat_error=float((loss-repeat).abs().max());replay_error=max(replay_error,repeat_error)
        actions.append(dict(id=f'{pair[0]}->{pair[1]}',remove=pair[0],insert=pair[1],gain_cls_loc=gain.cpu().tolist(),
            changed_loss=loss.cpu().tolist(),replay_max_error=repeat_error,**repartition(current,changed,ep.fps)))
        if float(gain.sum())>best_gain:
            best_gain=float(gain.sum());best_selection=mapped;best_native=feature;best_loss=loss
    target=torch.stack(values) if values else x.new_empty((0,2))
    arrays=dict(descriptor=x,target=target,**cheap_graph_state(ep,timeline,public,current,offered))
    arrays={k:v.detach().cpu().numpy() for k,v in arrays.items()}
    meta=data['metas'][0]
    state_key=f"{ep.video_id}__f{ep.window_start_frame:07d}__r{round_index}"
    row=dict(schema='T_VALUE_BANK_V1',state_key=state_key,video_id=ep.video_id,
        window_index=int(meta['characterization_window_index']),window_start_frame=ep.window_start_frame,
        window_id=f'{ep.video_id}:start{ep.window_start_frame}:stride{ep.snippet_stride}:slots768',
        episode=ep.record(),round_index=round_index,checkpoint=runtime.identity,
        selected_frame_ids=list(current.frame_ids),selected_valid=list(current.valid),
        selected_slots=selection.indices[0].cpu().tolist(),candidate_frame_ids=list(proposal.frame_ids),
        action_pairs=[list(p) for p in offered],actions=actions,base_loss=base_loss.cpu().tolist(),
        best_gain=best_gain,best_loss=best_loss.cpu().tolist(),best_changed_slots=best_selection.indices[0].cpu().tolist(),
        no_op_error=no_op_error,replay_max_error=replay_error,
        query_forwards=1+len(offered)+(2 if offered else 0),one_label_forward_gflops=one_cost,
        label_query_gflops=one_cost*(1+len(offered)+(2 if offered else 0)),
        cost_scope='measured fixed-capacity forward plus task-loss MACs, reused for same-shape candidates; preview recorded per window',
        continuation='one forced T exchange then full current D100/S100/Cross/readout; no further T refinement in the label',
        preview_scope='Standard768 cheap observation; graph input is physical interpolation to192, not a different decoder preview',
        prediction_inputs='descriptor and npz cheap/physical/support arrays only; no GT/heavy/future features',
        strata=action_strata(data,offered),wall_seconds=time.perf_counter()-start)
    return row,arrays,best_selection,best_native,native


def save_state(folder,row,arrays,partition,science_sha):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    row=dict(row,partition=partition,capture_source_revision=science_sha)
    name=row['state_key'];array_file=folder/(name+'.npz')
    with array_file.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream,**arrays)
    array_file.with_suffix('.tmp').replace(array_file)
    row['arrays_file']=array_file.name
    json_write(folder/(name+'.json'),row)
    return row


def assert_same_actions(first,second):
    keys=('video_id','window_id','window_start_frame','round_index','selected_frame_ids','selected_valid',
          'candidate_frame_ids','action_pairs','continuation')
    for key in keys:
        if first[key]!=second[key]:raise ValueError('Cross-checkpoint action identity differs: '+key)
    for key in ('official_frame_ids','official_valid','transform','fps','snippet_stride'):
        left=first['episode'][key];right=second['episode'][key]
        if key in ('official_frame_ids','official_valid'):left,right=tuple(left),tuple(right)
        if left!=right:raise ValueError('Cross-checkpoint episode differs: '+key)
