"""Parameter-free A-MoD scores and exact capacity masks."""
import torch

def incoming_attention(q,k,query_valid=None,chunk=64):
    # Detached routing score uses the SAME projected previous-layer Q/K.
    # This additional QK is deliberately visible to the matrix counter.
    with torch.no_grad():
        result=q.new_zeros((q.shape[0],q.shape[1],k.shape[-2]),dtype=torch.float32)
        scale=q.shape[-1]**-.5
        keys=k.float().transpose(-2,-1)
        for begin in range(0,q.shape[-2],chunk):
            scores=(q[:,:,begin:begin+chunk].float()*scale)@keys
            probability=scores.softmax(-1)
            if query_valid is not None:probability=probability*query_valid[:,None,begin:begin+chunk,None]
            result+=probability.sum(-2)
        denom=q.shape[-2] if query_valid is None else query_valid.sum(-1).clamp_min(1)[:,None,None]
        return (result/denom).mean(1)

def capacity_mask(scores,ratio,valid=None,uniform=False):
    if not 0<=ratio<=1:raise ValueError('Capacity must lie in[0,1]')
    valid=torch.ones_like(scores,dtype=torch.bool) if valid is None else valid.bool()
    n=scores.shape[-1];counts=valid.sum(-1,keepdim=True)
    keep=(counts.float()*ratio).round().long()
    if ratio>0:keep=torch.where(counts>0,keep.clamp_min(1),0)
    ranks=torch.arange(n,device=scores.device).expand_as(scores)
    if uniform:
        ids=ranks.masked_fill(~valid,n).sort(-1).values
        slot=(ranks.float()*(counts-1).clamp_min(0)/(keep-1).clamp_min(1)).round().long()
        slot=torch.where(keep==1,(counts-1).clamp_min(0)//2,slot).clamp(0,n-1)
        chosen=ids.gather(-1,slot).clamp_max(n-1)
        result=torch.zeros_like(ranks).scatter_add(-1,chosen,(ranks<keep).long())>0
        return result&valid
    order=scores.float().masked_fill(~valid,-float('inf')).argsort(dim=-1,descending=True,stable=True)
    return torch.zeros_like(valid).scatter(-1,order,ranks<keep)&valid
