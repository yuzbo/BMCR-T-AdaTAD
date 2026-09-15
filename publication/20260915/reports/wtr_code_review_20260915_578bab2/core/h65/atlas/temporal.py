"""Individual RGB support selection, original-time interpolation, unchanged head."""
import time
import numpy as np
import torch
from .reference import deterministic_fp32


def interpolate(features, source, valid, length=768):
    x=source[valid].float()
    z=features[:,valid].float()
    if len(x) == 0:
        raise ValueError('Temporal support contains no observed feature')
    query=torch.arange(length,device=z.device,dtype=torch.float32)
    right=torch.searchsorted(x.contiguous(),query).clamp(0,len(x)-1)
    left=(right-1).clamp(0,len(x)-1)
    weight=((query-x[left])/(x[right]-x[left]).clamp_min(1e-8)).clamp(0,1)
    return (z[:,left]*(1-weight)+z[:,right]*weight)[None]


def selected_features(reference,data,indices,variant='physical'):
    wrapper=reference.detector.backbone
    inputs=data['inputs'].index_select(3,indices)
    frames,_=wrapper.model.data_preprocessor.preprocess(wrapper.tensor_to_list(inputs),None,False)
    b,n,c,k,h,w=frames.shape
    if (b,n)!=(1,1) or k%16:
        raise ValueError('Selected RGB must have one window and 16-observation execution packing')
    clips=frames.reshape(b,n,c,k//16,16,h,w).permute(0,3,1,2,4,5,6).reshape(k//16,c,16,h,w).contiguous()
    previous=[block.adapter.temporal_size for block in reference.vit.blocks]
    try:
        for block in reference.vit.blocks:
            block.adapter.temporal_size=k//2
        raw=reference.vit(clips)
    finally:
        for block,size in zip(reference.vit.blocks,previous):
            block.adapter.temporal_size=size
    native=raw.mean((-1,-2)).permute(1,0,2).reshape(reference.vit.embed_dims,-1)
    selected_valid=data['masks'][0,indices]
    pair_valid=selected_valid.reshape(-1,2)
    valid=pair_valid.any(-1)
    centers=(indices.reshape(-1,2).float()*pair_valid).sum(-1)/pair_valid.sum(-1).clamp_min(1)
    if variant=='physical':
        features=interpolate(native,centers,valid)
    else:
        mode='nearest' if variant=='packed_nearest' else 'linear'
        kwargs={} if mode=='nearest' else dict(align_corners=False)
        observed=native[:,valid][None]
        # Naive packed controls still provide exactly the original head axis.
        features=torch.nn.functional.interpolate(observed,size=768,mode=mode,**kwargs)
    return features,native,centers,valid


def execute_temporal(reference,data,indices,variant='physical',force_profile=False):
    from h65.frame.measure import matrix_counter
    key=('temporal',len(indices),variant)
    profile=force_profile or key not in reference.costs
    torch.cuda.synchronize();start=time.perf_counter()
    with deterministic_fp32():
        if profile:
            with matrix_counter() as counter:
                features,native,centers,valid=selected_features(reference,data,indices,variant)
                prediction=reference.prediction_head(features,data)
            if counter.unresolved_matrix_ops():
                raise RuntimeError(str(counter.unresolved_matrix_ops()))
            cost=2*sum(counter.macs.values())/1e9
            reference.costs[key]=cost
            reference.profile_records.append(dict(kind='selected_rgb',frames=len(indices),gflops=cost,
                macs=dict(counter.macs),source='Actual operator dispatch including physical recovery and head'))
        else:
            features,native,centers,valid=selected_features(reference,data,indices,variant)
            prediction=reference.prediction_head(features,data)
            cost=reference.costs[key]
        torch.cuda.synchronize();model_ms=(time.perf_counter()-start)*1000
        losses=reference.loss_head(features,data)
    reference.executions+=1;reference.scoring_gflops+=cost
    return dict(losses=losses,loss=float(losses.sum()),features=features,prediction=prediction,
        proposals=prediction[0][0],scores=prediction[1][0],gflops=cost,model_ms=model_ms,
        selected_indices=indices.cpu().tolist(),native=native,centers=centers,valid=valid)


def uniform_indices(data,budget=384):
    valid=int(data['masks'].sum())
    device=data['inputs'].device
    if valid>=budget:
        return torch.floor(torch.arange(budget,device=device)*valid/budget).long()
    return torch.arange(budget,device=device)


def support_groups(data):
    """16 groups of individual frames; four time strata x four interleaved phases.

    Padding and at most 15 remainder observations are mandatory for every policy.
    No group is a contiguous 16-frame clip selector.
    """
    valid=int(data['masks'].sum())
    count=valid//16
    groups=[];mandatory=list(range(valid,768))
    for part in np.array_split(np.arange(valid),4):
        used=part[:4*count]
        mandatory.extend(part[4*count:].tolist())
        for phase in range(4):
            groups.append(used[phase::4].tolist())
    return groups,mandatory


def group_indices(data,groups,mandatory,chosen):
    ids=sorted(mandatory+[i for g in chosen for i in groups[g]])
    if not ids:
        raise ValueError('A legal temporal plan must contain observed support')
    padded=((len(ids)+15)//16)*16
    ids.extend([767]*(padded-len(ids)))
    return torch.tensor(ids,device=data['inputs'].device,dtype=torch.long)
