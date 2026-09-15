"""Original clip identities, nested budgets, and native time interpolation."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str = 'aux'
    clips8: int = 24
    clips12: int = 24
    spatial_ratio: float = 1.0
    selection: str = 'adaptive'


POLICIES = {
    'D768G': Policy('D768G', 'global', 48, 48, selection='uniform'),
    'D768L': Policy('D768L', 'local', 48, 48, selection='uniform'),
    **{f'Z{k}': Policy(f'Z{k}', 'interpolate', k, k, selection='uniform') for k in (16,24,36)},
    'ZR24': Policy('ZR24', 'interpolate', 24, 24, selection='random'),
    'PONLY': Policy('PONLY', clips8=0, clips12=0),
    'T24U': Policy('T24U', selection='uniform'),
    'T24A': Policy('T24A'),
    'D8': Policy('D8', clips8=48, clips12=0),
    'DAD': Policy('DAD', clips8=24, clips12=12),
    'S75': Policy('S75', clips8=48, clips12=48, spatial_ratio=.75),
    'S50': Policy('S50', clips8=48, clips12=48, spatial_ratio=.5),
}
for temporal in (0,1):
    for spatial in (0,1):
        for depth in (0,1):
            name=f'F{temporal}{spatial}{depth}'
            count=24 if temporal else 48
            POLICIES[name]=Policy(name, clips8=count, clips12=count//2 if depth else count,
                                  spatial_ratio=.75 if spatial else 1.)


def uniform_ids(count, budget, device=None):
    budget=min(count,budget)
    if budget == 0: return torch.empty(0,dtype=torch.long,device=device)
    if budget == 1: return torch.tensor([count//2],device=device)
    return torch.linspace(0,count-1,budget,device=device).round().long()


def select_with_anchors(scores, budget, anchors, mode, generator=None):
    count=scores.numel(); budget=min(budget,count)
    if budget == 0: return anchors[:0]
    if budget == count: return torch.arange(count,device=scores.device)
    if mode == 'uniform': return uniform_ids(count,budget,scores.device)
    available=torch.ones(count,dtype=torch.bool,device=scores.device);available[anchors]=False
    candidates=available.nonzero().flatten()
    order=(torch.randperm(len(candidates),generator=generator,device=scores.device) if mode=='random'
           else torch.argsort(scores[candidates],descending=True,stable=True))
    return torch.cat((anchors,candidates[order[:budget-len(anchors)]] )).sort().values


def route_depths(logits, masks, policy, seed=3407, sample_seeds=None):
    """Depths are indexed by original clip ID, never a reordered detector axis."""
    batch,time=masks.shape
    if time % 16: raise ValueError('candidate grid must contain whole16-observation clips')
    clips=time//16
    depths=torch.zeros((batch,clips),dtype=torch.long,device=masks.device)
    if policy.kind in ('global','local'):
        return depths.fill_(12)
    if policy.clips8 == 48 and policy.clips12 in (0,48):
        return depths.fill_(8 if policy.clips12==0 else 12)
    for row in range(batch):
        valid=masks[row].reshape(clips,16).any(-1).nonzero().flatten()
        n=len(valid); k8=min(n,policy.clips8);k12=min(n,policy.clips12)
        if not k8: continue
        anchors=uniform_ids(n,min(k12 or k8,max(2,(k12 or k8)//4)),masks.device)
        generator=torch.Generator(device=masks.device).manual_seed(sample_seeds[row] if sample_seeds is not None else seed+row)
        first=select_with_anchors(logits[row,valid,0],k8,anchors,policy.selection,generator)
        depths[row,valid[first]]=8
        if k12:
            if k12==k8: second=first
            else:
                # Mandatory full-depth anchors are also admitted to the prefix.
                if policy.selection=='uniform':
                    first=torch.unique(torch.cat((anchors,first)),sorted=True)
                    extra=first[~torch.isin(first,anchors)][:k8-len(anchors)]
                    first=torch.cat((anchors,extra)).sort().values
                    depths[row].zero_();depths[row,valid[first]]=8
                local_anchors=torch.where(torch.isin(first,anchors))[0]
                chosen=select_with_anchors(logits[row,valid[first],1],k12,local_anchors,
                                           'adaptive' if policy.selection=='adaptive' else 'random',generator)
                second=first[chosen]
            depths[row,valid[second]]=12
    return depths


def interpolate_native(values, positions, query):
    """Linear interpolation on physical centers, with constant endpoint extension."""
    if positions.numel()==0: raise ValueError('interpolation needs an observed tubelet')
    if positions.numel()==1: return values[:1].expand(len(query),-1)
    right=torch.searchsorted(positions.contiguous(),query.contiguous()).clamp(1,len(positions)-1)
    left=right-1
    alpha=((query-positions[left])/(positions[right]-positions[left]).clamp_min(1e-8)).clamp(0,1)
    return values[left]*(1-alpha[:,None])+values[right]*alpha[:,None]


def native_geometry(masks, metas):
    batch,time=masks.shape
    valid=masks.reshape(batch,time//2,2).any(-1)
    frame_indices=[]
    for row,meta in enumerate(metas):
        if 'frame_inds' in meta:
            indices=torch.as_tensor(meta['frame_inds'],device=masks.device,dtype=torch.float32).flatten()
        else:
            indices=torch.arange(time,device=masks.device).float()*meta.get('snippet_stride',1)+meta.get('window_start_frame',0)
        if len(indices)!=time: raise ValueError('source frame indices must match the candidate grid')
        frame_indices.append(indices)
    frame_indices=torch.stack(frame_indices)
    return valid,frame_indices,frame_indices.reshape(batch,time//2,2).mean(-1)
