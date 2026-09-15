"""Frozen complete computation on exactly the student's selected RGB support."""
import torch
import torch.nn.functional as F
from h65.frame.geometry import source_times
from .geometry import candidate_mask

class _StateStatistics(torch.autograd.Function):
    """Same masked NMSE, with per-pack FP32 temporaries and recomputed gradients."""
    @staticmethod
    def forward(ctx,value,target,valid):
        count=valid.sum().clamp_min(1)
        error_sum=value.new_zeros((),dtype=torch.float32);scale_sum=error_sum.clone();cosine_sum=error_sum.clone()
        for i in range(value.shape[0]):
            v=value[i:i+1].float();t=target[i:i+1].float();mask=valid[i:i+1]
            scale_sum=scale_sum+(t.square().mean(-1)*mask).sum()
            error_sum=error_sum+((v-t).square().mean(-1)*mask).sum()
            cosine_sum=cosine_sum+(F.cosine_similarity(v,t,dim=-1)*mask).sum()
        scale=scale_sum/count;denominator=count*value.shape[-1]*scale.clamp_min(1e-6)
        error=error_sum/count/scale.clamp_min(1e-6);cosine=cosine_sum/count;rms=scale.sqrt()
        if ctx.needs_input_grad[0]:ctx.save_for_backward(value,target,valid,denominator)
        ctx.mark_non_differentiable(cosine,rms)
        return error,cosine,rms

    @staticmethod
    def backward(ctx,gradient,unused_cosine,unused_rms):
        value,target,valid,denominator=ctx.saved_tensors
        result=torch.empty(value.shape,dtype=value.dtype,device=value.device)
        factor=2*gradient/denominator
        for i in range(value.shape[0]):
            delta=value[i:i+1].float()-target[i:i+1].float()
            result[i:i+1]=(delta*valid[i:i+1,:,None]*factor).to(value.dtype)
        return result,None,None

def support_key(data,selection,resolution):
    times=source_times(candidate_mask(data),data['metas'])
    return dict(candidate_ids=selection.indices.detach(),contributor_valid=selection.valid.detach(),
                source_frame_times=times.gather(1,selection.indices),spatial_shape=(resolution//16,resolution//16),
                tia_temporal_size=selection.indices.shape[1]//2,pack_order='selected time order / 16 observations / paired tubelets',
                augmentation='the same input tensor in this objective call')

def compare_states(student,reference,valid):
    rows=[];terms={key:[] for key in ('attention','pre_tia','post_tia')}
    if not student:raise ValueError('Same-support supervision requires captured student states')
    for layer,states in student.items():
        if layer not in reference:raise ValueError('Missing original-layer target')
        for key,value in states.items():
            target=reference[layer][key].detach()
            if target.shape!=value.shape or valid.shape!=value.shape[:2]:raise ValueError('Same-support state shape mismatch')
            error,cosine,rms=_StateStatistics.apply(value,target,valid)
            terms[key].append(error)
            rows.append(dict(original_block_id=layer-1,point=key,normalized_mse=error.detach(),
                             cosine=cosine,reference_rms=rms))
    return {k:torch.stack(v).mean() for k,v in terms.items()},rows

def same_support_targets(model,data,detail):
    reference=model.support_reference
    if reference is None:raise RuntimeError('Frozen support reference was not instantiated for training/diagnostics')
    reference.eval();selection=detail['selection'];keys=support_key(data,selection,model.encoder.resolution)
    if reference.resolution!=model.encoder.resolution:raise ValueError('Reference and student spatial support differ')
    layers=tuple(detail['trace']['state_taps'])
    plan=dict(frames=selection.indices.shape[1],depth=1.,space=1.,id='same_support_full')
    with torch.no_grad():
        _,_,trace=reference.encode(data['inputs'],selection,plan,capture=True,state_capture=True,support_layers=layers)
    native_valid=selection.valid.reshape(len(selection.indices),-1,2).any(-1)
    b=detail['trace']['patch_input_shape'][0];p=keys['spatial_shape'][0]*keys['spatial_shape'][1]
    valid=native_valid.reshape(b,8).repeat_interleave(p,1)
    student=detail['trace']['state_taps']
    if not model.config.get('support_loss',False):student={i:{k:v.detach() for k,v in row.items()} for i,row in student.items()}
    else:student={i:{k:v.detach() if k=='attention' else v for k,v in row.items()} for i,row in student.items()}
    terms,diagnostics=compare_states(student,trace['state_taps'],valid)
    detail['support_key']=keys;detail['support_diagnostics']=diagnostics
    detail['support_reference_source']=reference.provenance
    from .profile import encoder_macs
    detail['support_forward_gflops']=2*encoder_macs(model,trace)/1e9
    return terms
