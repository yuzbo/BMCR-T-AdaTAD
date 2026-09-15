"""Selected contextual pairs and original-time queries remain different grids."""
import torch
from .contracts import AnchorBatch,QueryBatch

def source_times(masks,metas):
    rows=[]
    for row,meta in enumerate(metas):
        if 'frame_inds' in meta:
            times=torch.as_tensor(meta['frame_inds'],device=masks.device,dtype=torch.float32).flatten()
        else:
            times=torch.arange(masks.shape[1],device=masks.device).float()*meta.get('snippet_stride',1)+meta.get('window_start_frame',0)
        if len(times)!=masks.shape[1]:raise ValueError('Frame provenance length mismatch')
        rows.append(times)
    return torch.stack(rows)

def make_anchors(features,selection,masks,metas,trace=None):
    b,a,c=features.shape;k=selection.indices.shape[1]
    if k%2 or a!=k//2:raise ValueError('Native features must precede rank interpolation: A=K/2')
    times=source_times(masks,metas);ids=selection.indices.reshape(b,a,2)
    contributor=times.gather(1,selection.indices).reshape(b,a,2)
    flags=selection.valid.reshape(b,a,2);counts=flags.sum(-1)
    centers=(contributor*flags).sum(-1)/counts.clamp_min(1)
    spans=torch.where(counts==2,contributor[...,1]-contributor[...,0],0.)
    pos=torch.arange(a,device=features.device)[None].expand(b,-1);trace=trace or {}
    depth=trace.get('last_heavy_depth',features.new_full((b,a),12))
    quality=trace.get('spatial_quality',features.new_ones((b,a)))
    return AnchorBatch(features,ids,contributor,flags,centers,spans,counts>0,pos//8,pos%8,depth,quality)

def make_queries(masks,metas,selection):
    b,t=masks.shape;times=source_times(masks,metas);pairs=masks.reshape(b,t//2,2)
    centers=(times.reshape(b,t//2,2)*pairs).sum(-1)/pairs.sum(-1).clamp_min(1)
    membership=torch.zeros_like(masks,dtype=torch.long).scatter_add(1,selection.indices,selection.valid.long())>0
    return QueryBatch(times,centers,pairs.any(-1),masks.bool(),membership)

def interpolate_anchors(anchors,queries):
    rows=[]
    for f,t,v,q in zip(anchors.features,anchors.centers,anchors.valid,queries.centers):
        f,t=f[v],t[v].detach()
        if not len(t):raise ValueError('At least one real selected contributor is required')
        if len(t)==1:rows.append(f[:1].expand(len(q),-1));continue
        if not bool((t.diff()>0).all()):raise ValueError('Valid anchor centers must be increasing')
        right=torch.searchsorted(t.contiguous(),q.detach().contiguous()).clamp(1,len(t)-1);left=right-1
        fraction=((q.detach()-t[left])/(t[right]-t[left])).clamp(0,1)
        rows.append(f[left]*(1-fraction[:,None])+f[right]*fraction[:,None])
    return torch.stack(rows)*queries.valid[...,None]

def decoder_metadata(anchors,queries):
    count=queries.candidate_mask.sum(-1);start=queries.frame_times[:,:1]
    end=queries.frame_times.gather(1,(count-1)[:,None]);span=(end-start).clamp_min(1)
    times=(anchors.contributor_times-start[:,:,None])/span[:,:,None]
    am=torch.stack((times[...,0],times[...,1],(anchors.centers-start)/span,anchors.spans/span,
        anchors.contributor_valid[...,0].float(),anchors.contributor_valid[...,1].float(),
        anchors.clip_position.float()/max(anchors.features.shape[1]//8-1,1),
        anchors.tubelet_position.float()/7,anchors.last_heavy_depth.float()/12,anchors.spatial_quality.float()),-1)
    gaps=[]
    for t,v,q in zip(anchors.centers,anchors.valid,queries.centers):
        t=t[v];right=torch.searchsorted(t.contiguous(),q.contiguous()).clamp_max(len(t)-1)
        left=(torch.searchsorted(t.contiguous(),q.contiguous(),right=True)-1).clamp_min(0)
        gaps.append(torch.stack(((q-t[left]).abs(),(t[right]-q).abs()),-1))
    gaps=torch.stack(gaps)/span[:,:,None]
    q=queries.centers.shape[1];member=queries.membership.reshape(len(count),q,2)
    valid_fraction=queries.candidate_mask.reshape(len(count),q,2).float().mean(-1)
    rank=torch.linspace(0,1,q,device=span.device)[None].expand(len(count),-1)
    qm=torch.stack(((queries.centers-start)/span,rank,gaps[...,0],gaps[...,1],valid_fraction,
        member[...,0].float(),member[...,1].float(),queries.valid.float()),-1)
    return am*anchors.valid[...,None],qm*queries.valid[...,None]

def scout_context(output,masks):
    hidden=output['hidden'].detach();b,t,c=hidden.shape
    if c!=96 or t!=masks.shape[1]:raise ValueError('Expected existing per-candidate96-channel scout state')
    valid=masks.reshape(b,t//2,2)
    return (hidden.reshape(b,t//2,2,c)*valid[...,None]).sum(2)/valid.sum(-1).clamp_min(1)[...,None]
