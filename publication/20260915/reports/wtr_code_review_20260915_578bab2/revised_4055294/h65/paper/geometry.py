"""Candidate/feature/detector axes remain explicit for both TAD datasets."""
import torch
from h65.frame.geometry import make_anchors, make_queries, decoder_metadata


def candidate_mask(data):
    t = data['inputs'].shape[3]; mask = data['masks'].bool()
    if t % mask.shape[-1]: raise ValueError('Detector and RGB axes must have an integral ratio')
    return mask.repeat_interleave(t//mask.shape[-1], dim=-1)


def feature_target_data(data):
    result = dict(data); result['masks'] = candidate_mask(data)
    scale = result['masks'].shape[-1] / data['masks'].shape[-1]
    if 'gt_segments' in data: result['gt_segments'] = [x*scale for x in data['gt_segments']]
    return result


def interpolate_anchors(anchors, queries):
    rows=[]
    for features, times, valid, target in zip(anchors.features,anchors.centers,anchors.valid,queries.centers):
        f=features[valid]; t=times[valid].detach()
        if not len(t): raise ValueError('No real anchor')
        if bool((t.diff()<0).any()): raise ValueError('Physical source times must be ordered')
        # ActivityNet resize can repeat actual video frames. Merge equal physical
        # centers for interpolation, while the decoder keeps all contextual memory.
        unique, inverse, counts = torch.unique_consecutive(t,return_inverse=True,return_counts=True)
        if len(unique)!=len(t):
            f=f.new_zeros((len(unique),f.shape[-1])).index_add(0,inverse,f)/counts[:,None]
            t=unique
        if len(t)==1: rows.append(f.expand(len(target),-1)); continue
        right=torch.searchsorted(t.contiguous(),target.detach().contiguous()).clamp(1,len(t)-1)
        left=right-1; fraction=((target.detach()-t[left])/(t[right]-t[left])).clamp(0,1)
        rows.append(f[left]*(1-fraction[:,None])+f[right]*fraction[:,None])
    return torch.stack(rows)*queries.valid[...,None]


def scout_context(output, masks, trainable=True):
    hidden=output['hidden'] if trainable else output['hidden'].detach()
    b,t,c=hidden.shape; valid=masks.reshape(b,t//2,2)
    return (hidden.reshape(b,t//2,2,c)*valid[...,None]).sum(2)/valid.sum(-1).clamp_min(1)[...,None]
