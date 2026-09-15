"""Bounded temporal relation context; reuse the existing indexed message kernel."""
import math
import torch
from h65.paper.edge_ops import EdgeRouter,GraphMessage,gather_nodes,masked_softmax

OFFSETS=(0,-1,1,-2,2,-4,4,-8,8)
DYNAMIC_NEIGHBORS=4


class TemporalEdges(EdgeRouter):
    def __init__(self,channels,dynamic=False):
        # Keep the same trainable parameter shapes as the registered G1 control.
        super().__init__(channels,mode='fixed_local',referrals=1)
        self.dynamic=dynamic
        self.degree=1+DYNAMIC_NEIGHBORS if dynamic else len(OFFSETS)

    def forward(self,x,valid,geometry,state=None,grid=None):
        b,n,c=x.shape
        raw=torch.arange(n,device=x.device)[:,None]+torch.tensor(OFFSETS,device=x.device)[None]
        inside=(raw>=0)&(raw<n)
        indices=raw.clamp(0,n-1)[None].expand(b,-1,-1)
        available=inside[None]&valid[...,None]&gather_nodes(valid,indices)
        normalized=self.norm(x);query=self.query(normalized);key=self.key(normalized)
        neighbors=gather_nodes(key,indices)
        content=torch.matmul(query.unsqueeze(-2),neighbors.transpose(-1,-2)).squeeze(-2).float()/math.sqrt(self.width)
        meta=gather_nodes(geometry,indices)
        dt=meta[...,0]-geometry[...,None,0]
        # All five inputs have a temporal/coverage meaning; no dummy spatial axes.
        relative=torch.stack((dt,dt.abs(),dt.square(),
            geometry[...,None,3].expand_as(dt),meta[...,3]),-1)
        logits=content*self.log_temperature.clamp(-2,2).exp()+self.geometry(relative).squeeze(-1).float()
        if self.dynamic:
            # The address decision is discrete; only retained continuous scores
            # receive edge-weight gradients. Self refresh always remains legal.
            chosen=logits[...,1:].masked_fill(~available[...,1:],-torch.inf).topk(DYNAMIC_NEIGHBORS,-1).indices+1
            chosen=torch.cat((torch.zeros_like(chosen[...,:1]),chosen),-1)
            indices=indices.gather(-1,chosen);available=available.gather(-1,chosen);logits=logits.gather(-1,chosen)
        weights=masked_softmax(logits,available)
        return (indices,weights),dict(macs=2*b*n*c*self.width+b*n*len(OFFSETS)*(self.width+5*16+16),
            candidate_slots=b*n*len(OFFSETS),referral_paths=0,
            retained_edges=(weights.detach()>0).sum(),max_degree=self.degree,
            candidate_degree=len(OFFSETS))


class TemporalGraphMessage(GraphMessage):
    def __init__(self,width=64,dynamic=False):
        super().__init__(width,mode='fixed_local',referrals=1)
        self.router=TemporalEdges(width,dynamic)


def graph_protocol():
    return dict(nodes=192,width=64,candidate_offsets=list(OFFSETS),candidate_degree=len(OFFSETS),
        static_retained_degree=len(OFFSETS),dynamic_retained_degree=1+DYNAMIC_NEIGHBORS,
        dynamic_rule='self plus top4 non-self among the exact same bounded candidates',
        edge_score='content compatibility plus learned signed/absolute/squared dt and endpoint coverage',
        parameter_shapes='same for Static and Dynamic; Plain-L matched within1%',
        overhead='actual retained degrees differ and must be counted',referral_expansion=False,
        kernel='existing h65.paper.edge_ops.GraphMessage/indexed_attention, not GraphKV')
