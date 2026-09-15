"""Original-axis graph state with contributor-aware evidence and residual repair."""
from dataclasses import replace
import torch
from torch import nn
from .edge_ops import GraphMessage


def unobserved_queries(masks,metas):
    from h65.frame.contracts import QueryBatch
    from h65.frame.geometry import source_times
    b,t=masks.shape;times=source_times(masks,metas);valid=masks.reshape(b,t//2,2)
    centers=(times.reshape(b,t//2,2)*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    return QueryBatch(times,centers,valid.any(-1),masks.bool(),torch.zeros_like(masks,dtype=torch.bool))


def query_geometry(queries):
    count=queries.candidate_mask.sum(-1)
    start=queries.frame_times[:,:1]
    end=queries.frame_times.gather(1,(count-1).clamp_min(0)[:,None])
    position=(queries.centers-start)/(end-start).clamp_min(1)
    z=torch.zeros_like(position)
    return torch.stack((position,z,z,z,z),-1)*queries.valid[...,None]


def anchor_reliability(anchors,queries,depth=12):
    count=queries.candidate_mask.sum(-1)
    start=queries.frame_times[:,:1]
    end=queries.frame_times.gather(1,(count-1).clamp_min(0)[:,None])
    step=((end-start)/(count-1).clamp_min(1)[:,None]).clamp_min(1)
    excess=(anchors.spans/step-1).clamp_min(0)
    quality=(anchors.contributor_valid.float().mean(-1)*anchors.spatial_quality.detach().clamp(0,1)*
             (anchors.last_heavy_depth.detach()/depth).clamp(0,1)*torch.exp(-.5*excess))
    return quality*anchors.valid


def deposit(anchors,queries,depth=12):
    """Each actual contributor votes for its ORIGINAL candidate-pair query.

    A mixed tubelet may vote at two different query positions. It is never
    assigned by its packed index. Reliability discounts wide mixed supports.
    """
    b,a,c=anchors.features.shape;q=queries.centers.shape[1]
    index=anchors.contributor_indices//2;valid=anchors.contributor_valid
    if bool((index[valid]>=q).any()):raise ValueError('Contributor outside original query axis')
    reliability=anchor_reliability(anchors,queries,depth)
    vote=valid.float()*reliability[...,None]/valid.sum(-1,keepdim=True).clamp_min(1)
    feature=anchors.features.float()[:,:,None].expand(-1,-1,2,-1)
    numerator=feature.new_zeros((b,q,c)).scatter_add(1,index.flatten(1)[...,None].expand(-1,-1,c),(feature*vote[...,None]).flatten(1,2))
    mass=vote.new_zeros((b,q)).scatter_add(1,index.flatten(1),vote.flatten(1))
    slots=vote.new_zeros((b,q)).scatter_add(1,index.flatten(1),valid.float().flatten(1))
    quality=vote.new_zeros((b,q)).scatter_add(1,index.flatten(1),(valid*reliability[...,None]).flatten(1))/slots.clamp_min(1)
    return numerator/mass.clamp_min(1e-12)[...,None],(slots/2).clamp(0,1),quality*queries.valid


def gather_contributors(features,anchors):
    b,a,_=anchors.contributor_indices.shape;c=features.shape[-1]
    index=anchors.contributor_indices//2
    values=features.gather(1,index.flatten(1)[...,None].expand(-1,-1,c)).reshape(b,a,2,c)
    valid=anchors.contributor_valid
    return (values*valid[...,None]).sum(2)/valid.sum(-1).clamp_min(1)[...,None]


class GraphRecovery(nn.Module):
    def __init__(self,channels,depth=12,couplings=(),frame_graph=False,mode='referral',referrals=2,width=128):
        super().__init__();self.channels=channels;self.width=width;self.depth=depth
        self.couplings=tuple(couplings);self.frame_graph=frame_graph
        self.cheap=nn.Sequential(nn.LayerNorm(96),nn.Linear(96,width))
        self.meta=nn.Linear(5,width)
        levels=self.couplings or (depth,)
        self.source=nn.ModuleDict({str(level):nn.Sequential(nn.LayerNorm(channels),nn.Linear(channels,width)) for level in levels})
        self.fusion_gate=nn.ModuleDict({str(level):nn.Linear(5,1) for level in levels})
        for gate in self.fusion_gate.values():nn.init.zeros_(gate.weight);nn.init.zeros_(gate.bias)
        keys=[str(level) for level in levels] if self.couplings else ['post0','post1','post2']
        if frame_graph:keys.append('preview')
        self.messages=nn.ModuleDict({key:GraphMessage(width,mode,referrals) for key in keys})
        self.feedback=nn.ModuleDict({str(level):nn.Linear(width,channels) for level in self.couplings})
        self.output=nn.Linear(width,channels);self.anchor_gate=nn.Linear(7,1)
        for module in [self.output,self.anchor_gate,*self.feedback.values()]:
            nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)
        if frame_graph:
            self.rate=nn.Linear(width,1);nn.init.zeros_(self.rate.weight);nn.init.zeros_(self.rate.bias)

    def forward(self,stage,queries,*,cheap=None,state=None,anchors=None,level=None,cross=None):
        b,q=queries.valid.shape;c=self.width
        if stage=='start':
            geometry=query_geometry(queries)
            x=(self.cheap(cheap)+self.meta(geometry))*queries.valid[...,None]
            macs=b*q*(96*c+5*c);indices=weights=None;info=[]
            if self.frame_graph:
                x,indices,weights,record=self.messages['preview'](x,queries.valid,geometry)
                macs+=record['macs'];info.append(record)
                rate=self.rate(x).squeeze(-1).repeat_interleave(2,-1)*queries.candidate_mask
                macs+=b*q*c
            else:rate=None
            return (x,indices,weights,geometry),rate,dict(macs=macs,edges=info)
        x,indices,weights,geometry=state
        if stage=='step':
            key=str(level);a=anchors.features.shape[1]
            projected=self.source[key](anchors.features)
            observed,coverage,quality=deposit(replace(anchors,features=projected),queries,self.depth)
            meta=torch.stack((geometry[...,0],coverage,quality,queries.valid.float(),coverage*float(level)/self.depth),-1)
            gate=self.fusion_gate[key](meta).sigmoid()*coverage[...,None]*quality[...,None]
            x=x+gate*(observed-x)
            geometry=torch.cat((geometry[...,:3],(coverage*quality)[...,None],(coverage*float(level)/self.depth)[...,None]),-1)
            macs=b*a*self.channels*c+b*q*5;records=[]
            message_keys=[key] if self.couplings else ['post0','post1','post2']
            for message_key in message_keys:
                x,indices,weights,record=self.messages[message_key](x,queries.valid,geometry,indices,weights)
                macs+=record['macs'];records.append(record)
            if self.couplings:
                gathered=gather_contributors(x,anchors)
                feedback=self.feedback[key](gathered)*anchor_reliability(anchors,queries,self.depth)[...,None]
                macs+=b*a*c*self.channels
            else:feedback=None
            return (x,indices,weights,geometry),feedback,dict(macs=macs,edges=records)
        if stage=='readout':
            real,coverage,quality=deposit(anchors,queries,self.depth)
            inputs=torch.cat((geometry,coverage[...,None],quality[...,None]),-1)
            gate=self.anchor_gate(inputs).tanh()*coverage[...,None]*quality[...,None]
            # Both new output maps are zero at initialization, with nonzero
            # inputs: output/gate learn first and then drive the graph upstream.
            result=cross.transpose(1,2)+self.output(x)+gate*(real-cross.transpose(1,2))
            result=result*queries.valid[...,None]
            record=dict(macs=b*q*(c*self.channels+7),anchor_gate_abs=gate.detach().abs().mean(),
                        observed_query_fraction=(coverage.detach()>0).float().mean())
            return result.transpose(1,2),record
        raise ValueError(stage)


def anchor_consistency(recovered,anchors,queries,depth=12):
    predicted=gather_contributors(recovered.transpose(1,2),anchors)
    target=anchors.features.detach().float();weight=anchor_reliability(anchors,queries,depth).detach()
    scale=target.square().mean(-1).clamp_min(1e-6)
    error=(predicted.float()-target).square().mean(-1)/scale
    return (error*weight).sum()/weight.sum().clamp_min(1)
