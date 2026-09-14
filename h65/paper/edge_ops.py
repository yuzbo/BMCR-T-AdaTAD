"""Bounded sparse edge lists, weighted referral and genuine indexed attention.

This is a GM-inspired video adaptation, not the original multi-edge Qwen model.
No N-by-N affinity or adjacency matrix is formed by these modules.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def gather_nodes(values, indices):
    """[B,N,...] gathered by [B,Q,K] indices, without expanding B*Q*N."""
    b,n=values.shape[:2]
    flat=(indices+torch.arange(b,device=indices.device).view(b,*([1]*(indices.ndim-1)))*n).reshape(-1)
    return values.reshape(b*n,*values.shape[2:]).index_select(0,flat).reshape(*indices.shape,*values.shape[2:])


def normalize_weights(weights):
    return weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)


def masked_softmax(logits, valid):
    logits=logits.float().masked_fill(~valid,-torch.inf)
    maximum=logits.amax(-1,keepdim=True)
    maximum=torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
    value=torch.exp(logits-maximum)*valid
    return normalize_weights(value)


def coalesce_topk(indices, weights, degree):
    """Coalesce duplicate destinations with a deterministic sorted prefix sum.

    Memory is bounded by candidate slots, not by the number of possible nodes.
    The hard addresses are discrete; gradients flow through retained masses.
    """
    ordered,order=torch.sort(indices,dim=-1,stable=True)
    mass=weights.gather(-1,order).float()
    slots=indices.shape[-1]
    position=torch.arange(slots,device=indices.device).expand_as(indices)
    starts=torch.ones_like(indices,dtype=torch.bool)
    starts[...,1:]=ordered[...,1:]!=ordered[...,:-1]
    begin=torch.where(starts,position,torch.zeros_like(position)).cummax(-1).values
    prefix=mass.cumsum(-1)
    before=prefix.gather(-1,(begin-1).clamp_min(0))*(begin>0)
    cumulative=prefix-before
    ends=torch.ones_like(starts);ends[...,:-1]=starts[...,1:]
    total=torch.where(ends,cumulative.clamp_min(0),torch.zeros_like(cumulative))
    value,keep=total.topk(min(degree,slots),dim=-1,sorted=True)
    targets=ordered.gather(-1,keep)
    if degree>slots:
        targets=F.pad(targets,(0,degree-slots));value=F.pad(value,(0,degree-slots))
    return targets,normalize_weights(value)


def initial_edges(valid, grid=None, local_only=False):
    """16 geometry-defined candidates: local continuity and multiscale refresh."""
    b,n=valid.shape;device=valid.device
    if grid is None:
        offsets=[0,-1,1,-2,2,-3,3,-4,-5,4,5,-6,6,-7,7,8] if local_only else [0,-1,1,-2,2,-4,4,-8,8,-16,16,-32,32,-64,64,128]
        raw=torch.arange(n,device=device)[:,None]+torch.tensor(offsets,device=device)[None]
        inside=(raw>=0)&(raw<n);indices=raw.clamp(0,n-1)
    else:
        t,h,w=grid
        if n!=t*h*w:raise ValueError('Graph grid does not match the packed token domain')
        z,y,x=torch.meshgrid(torch.arange(t,device=device),torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
        centers=torch.stack((z,y,x),-1).reshape(n,3)
        local=[(0,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1),(-1,0,0),(1,0,0),(0,1,1)]
        extra=[(0,-1,-1),(0,-1,1),(0,1,-1),(-1,-1,0),(-1,1,0),(1,-1,0),(1,1,0),(-1,0,1)] if local_only else [(-2,0,0),(2,0,0),(-4,0,0),(4,0,0),(0,-3,0),(0,3,0),(0,0,-3),(0,0,3)]
        raw=centers[:,None]+torch.tensor(local+extra,device=device)[None]
        maximum=torch.tensor([t-1,h-1,w-1],device=device)
        inside=((raw>=0)&(raw<=maximum)).all(-1)
        raw=torch.minimum(raw.clamp_min(0),maximum)
        indices=(raw[...,0]*h+raw[...,1])*w+raw[...,2]
    indices=indices[None].expand(b,-1,-1)
    available=gather_nodes(valid,indices)&valid[...,None]&inside[None]
    return coalesce_topk(indices,available.float(),16)


class EdgeRouter(nn.Module):
    def __init__(self,channels,degree=16,width=32,mode='referral',referrals=2):
        super().__init__()
        if degree!=16 or mode not in ('referral','no_referral','fixed_local'):
            raise ValueError('The registered graph matrix uses degree16 and explicit referral controls')
        self.degree=degree;self.width=width;self.mode=mode;self.referrals=referrals
        self.norm=nn.LayerNorm(channels)
        self.query=nn.Linear(channels,width,bias=False);self.key=nn.Linear(channels,width,bias=False)
        self.geometry=nn.Sequential(nn.Linear(5,16),nn.GELU(),nn.Linear(16,1))
        self.log_temperature=nn.Parameter(torch.zeros(()))

    def forward(self,x,valid,geometry,state=None,grid=None):
        b,n,c=x.shape;seed_i,seed_w=initial_edges(valid,grid,self.mode=='fixed_local')
        if state is None or self.mode=='fixed_local':indices,weights=seed_i,seed_w
        else:indices,weights=state
        normalized=self.norm(x);q=self.query(normalized);k=self.key(normalized)
        macs=2*b*n*c*self.width;candidate_slots=0;referral_paths=0
        rounds=self.referrals if self.mode=='referral' else 1
        for _ in range(rounds):
            if self.mode=='fixed_local':candidate,prior=seed_i,seed_w
            else:
                parts_i=[indices,seed_i];parts_w=[weights,seed_w]
                if self.mode=='referral':
                    first,slots=weights.topk(4,-1);via=indices.gather(-1,slots)
                    next_i=gather_nodes(indices,via);next_w=gather_nodes(weights,via)
                    second,slots2=next_w.topk(4,-1)
                    referred=next_i.gather(-1,slots2).flatten(-2)
                    # Batched 1x1 by 1x4 products expose sparse path composition
                    # to the same actual matrix counter used by the backbone.
                    paths=torch.matmul(first[...,None,None],second.unsqueeze(-2)).squeeze(-2).flatten(-2)
                    macs+=paths.numel()
                    parts_i.append(referred);parts_w.append(paths);referral_paths+=paths.numel()
                candidate=torch.cat(parts_i,-1);prior=torch.cat(parts_w,-1)
            candidate,prior=coalesce_topk(candidate,prior,candidate.shape[-1])
            count=candidate.shape[-1];candidate_slots+=b*n*count
            neighbor=gather_nodes(k,candidate)
            compatibility=torch.matmul(q.unsqueeze(-2),neighbor.transpose(-1,-2)).squeeze(-2).float()/math.sqrt(self.width)
            meta=gather_nodes(geometry,candidate)
            relative=torch.cat((meta[...,:3]-geometry[...,None,:3],meta[...,3:5]),-1)
            bias=self.geometry(relative).squeeze(-1).float()
            available=(prior>0)&gather_nodes(valid,candidate)&valid[...,None]
            logits=compatibility*self.log_temperature.clamp(-2,2).exp()+bias+prior.clamp_min(1e-12).log()
            mass=masked_softmax(logits,available)
            indices,weights=coalesce_topk(candidate,mass,self.degree)
            macs+=b*n*count*(self.width+5*16+16)
        return (indices,weights),dict(macs=macs,candidate_slots=candidate_slots,referral_paths=referral_paths,
            retained_edges=int((weights.detach()>0).sum()),max_degree=self.degree)


def indexed_attention(q,k,v,indices,weights,query_mask,chunk=512):
    """Actual gathered K/V, with QK and AV visible to the matrix counter.

    q/k/v are [B,N,H,D]. Processing bounded query chunks controls gathered KV
    activation memory. K/V projections are reused, not repeated per query.
    """
    b,n,h,d=q.shape;flat_query=query_mask.flatten().nonzero().flatten()
    output=q.new_zeros((b*n,h,d));degree=indices.shape[-1]
    flat_i=indices.reshape(b*n,degree);flat_w=weights.reshape(b*n,degree)
    flat_q=q.reshape(b*n,h,d);flat_k=k.reshape(b*n,h,d);flat_v=v.reshape(b*n,h,d)
    for ids in flat_query.split(chunk):
        if not len(ids):continue
        targets=flat_i.index_select(0,ids)+(ids//n)[:,None]*n
        keys=flat_k.index_select(0,targets.flatten()).reshape(len(ids),degree,h,d).transpose(1,2)
        values=flat_v.index_select(0,targets.flatten()).reshape(len(ids),degree,h,d).transpose(1,2)
        mass=flat_w.index_select(0,ids)
        # Query rows are valid and always have a valid self refresh edge.
        bias=mass.clamp_min(1e-12).log().masked_fill(mass<=0,-torch.inf)[:,None,None]
        query=flat_q.index_select(0,ids).unsqueeze(-2)
        attended=F.scaled_dot_product_attention(query,keys,values,attn_mask=bias.to(query.dtype),dropout_p=0.).squeeze(-2)
        output=output.index_copy(0,ids,attended.to(output.dtype))
    return output.reshape(b,n,h*d),len(flat_query),2*len(flat_query)*degree*h*d


class GraphKVAttention(nn.Module):
    def __init__(self,channels,mode='referral',referrals=2):
        super().__init__();self.router=EdgeRouter(channels,mode=mode,referrals=referrals)

    def forward(self,attn,x,selected,valid,geometry,state,grid):
        from .engine import qkv_bias
        b,n,c=x.shape;h=attn.num_heads;d=c//h
        (indices,weights),routing=self.router(x,valid,geometry,state,grid)
        active=valid if selected is None else valid&selected
        bias=qkv_bias(attn)
        # Full K/V projection is explicit and counted even though access is sparse.
        k=F.linear(x,attn.qkv.weight[c:2*c],None if bias is None else bias[c:2*c]).reshape(b,n,h,d)
        v=F.linear(x,attn.qkv.weight[2*c:],None if bias is None else bias[2*c:]).reshape(b,n,h,d)
        ids=active.flatten().nonzero().flatten();flat=x.reshape(-1,c)
        packed=F.linear(flat.index_select(0,ids),attn.qkv.weight[:c],None if bias is None else bias[:c])
        q=packed.new_zeros((b*n,c)).index_copy(0,ids,packed).reshape(b,n,h,d)
        value,qrows,pairs=indexed_attention(q,k,v,indices,weights,active)
        projected=attn.proj_drop(attn.proj(value.reshape(-1,c).index_select(0,ids)))
        output=x.new_zeros((b*n,c)).index_copy(0,ids,projected.to(x.dtype)).reshape_as(x)
        return output,indices,weights,dict(q=qrows,kv=b*n,qk_av_macs=pairs,score_qk=0,
            graph_router_macs=routing['macs'],graph_candidate_slots=routing['candidate_slots'],
            graph_referral_paths=routing['referral_paths'],graph_retained_edges=routing['retained_edges'])


class GraphMessage(nn.Module):
    def __init__(self,width=128,mode='referral',referrals=2):
        super().__init__();self.width=width;self.heads=4
        self.router=EdgeRouter(width,mode=mode,referrals=referrals)
        self.norm=nn.LayerNorm(width);self.qkv=nn.Linear(width,width*3);self.proj=nn.Linear(width,width)
        self.ffn=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))

    def forward(self,x,valid,geometry,indices=None,weights=None):
        b,n,c=x.shape
        (indices,weights),routing=self.router(x,valid,geometry,None if indices is None else (indices,weights))
        q,k,v=self.qkv(self.norm(x)).reshape(b,n,3,self.heads,c//self.heads).unbind(2)
        message,qrows,pairs=indexed_attention(q,k,v,indices,weights,valid)
        x=(x+self.proj(message))*valid[...,None]
        x=(x+self.ffn(x))*valid[...,None]
        macs=routing['macs']+b*n*(3*c*c+c*c+4*c*c)+pairs
        return x,indices,weights,dict(macs=macs,**{k:v for k,v in routing.items() if k!='macs'})
