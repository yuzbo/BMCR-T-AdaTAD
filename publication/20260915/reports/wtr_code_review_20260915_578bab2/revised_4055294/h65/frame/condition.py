"""Exact batched equivalent of FormalScout.condition, opt-in system control."""
import torch


def vectorized_condition(scout,output,selection,masks):
    hidden=output['hidden'];b,t,_=hidden.shape
    if t%16:raise ValueError('Production candidate grid must be16-aligned')
    member=torch.zeros_like(masks,dtype=torch.long).scatter_add(1,selection.indices,selection.valid.long())>0
    pos=torch.arange(t,device=masks.device).expand(b,-1);local=torch.arange(16,device=masks.device)
    distance=(local[:,None]-local[None]).abs().expand(b,t//16,16,16)
    valid=masks.reshape(b,-1,16);membership=member.reshape(b,-1,16)
    allowed=valid[..., :,None]&valid[...,None,:]&(membership[..., :,None]!=membership[...,None,:])
    index=distance.masked_fill(~allowed,17).argmin(-1);feasible=allowed.any(-1).reshape(b,t)
    absolute=index+torch.arange(t//16,device=masks.device)[None,:,None]*16
    partner=torch.where(feasible,absolute.reshape(b,t),pos)
    kept=selection.indices.float();counts=selection.valid.sum(-1)
    search=torch.searchsorted(kept.contiguous(),pos.float().contiguous());upper=(counts-1)[:,None]
    left=kept.gather(1,torch.minimum((search-1).clamp_min(0),upper));right=kept.gather(1,torch.minimum(search,upper))
    # Keep the legacy reduction length/order even in a partially valid window.
    # Only the tiny batch axis loops; all candidate/partner work is vectorized.
    mean=torch.stack([hidden[row,member[row]].mean(0) for row in range(b)])
    other=hidden.gather(1,partner[...,None].expand_as(hidden))
    scalars=torch.stack((member.float(),(partner-pos)/t,(pos-left).abs()/t,(right-pos).abs()/t,
                         output['action_logits'].detach().sigmoid(),
                         (selection.valid.sum(-1)/masks.sum(-1))[:,None].expand(-1,t)),-1)
    context=torch.cat((hidden,other,mean[:,None].expand(-1,t,-1),output['representation'],scalars),-1)
    utility=scout.conditional(context.float()).masked_fill(~feasible[...,None],0)
    return dict(utility=utility,member=member,partner=partner,feasible=feasible)
