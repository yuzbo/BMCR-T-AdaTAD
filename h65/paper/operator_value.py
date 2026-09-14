"""Task-grounded D/S exchange routing with exact packed and native-time quotas."""
import torch
from torch import nn


def exact_mask(scores,allowed,counts,uniform=False):
    """Select explicit integer counts along the final axis, with stable ties."""
    shape=scores.shape;n=shape[-1]
    scores=scores.reshape(-1,n);allowed=allowed.reshape(-1,n)
    counts=counts.reshape(-1).long()
    if bool(((counts<0)|(counts>allowed.sum(-1))).any()):
        raise ValueError('Requested operator quota exceeds its legal set')
    result=torch.zeros_like(allowed)
    for row in range(len(scores)):
        legal=allowed[row].nonzero().flatten();k=int(counts[row])
        if not k:continue
        if uniform:
            indices=((torch.arange(k,device=scores.device)+.5)*len(legal)/k).floor().long()
            chosen=legal[indices]
        else:
            chosen=legal[scores[row,legal].argsort(descending=True,stable=True)[:k]]
        result[row,chosen]=True
    return result.reshape(shape)


def depth_mask(scores,valid,ratio,uniform=False):
    counts=(valid.sum(-1).float()*ratio).round().long()
    return exact_mask(scores,valid,counts,uniform)


def ff_natives_quota(admitted,valid,ratio,relative_to_all=True):
    """Largest-remainder distribution of a packed total across eight native times."""
    allowed=admitted.reshape(len(admitted),8,-1).sum(-1)
    base=valid.sum(-1) if relative_to_all else admitted.sum(-1)
    total=(base.float()*ratio).round().long().minimum(allowed.sum(-1))
    ideal=allowed.float()*total[:,None]/allowed.sum(-1,keepdim=True).clamp_min(1)
    quota=ideal.floor().long()
    for row in range(len(allowed)):
        remaining=int(total[row]-quota[row].sum())
        order=(ideal[row]-quota[row]).argsort(descending=True,stable=True)
        eligible=order[quota[row,order]<allowed[row,order]]
        quota[row,eligible[:remaining]]+=1
    if not torch.equal(quota.sum(-1),total):
        raise RuntimeError('Native-time FFN quotas do not conserve packed capacity')
    return quota


def spatial_mask(scores,admitted,valid,ratio,uniform=False,relative_to_all=True):
    quota=ff_natives_quota(admitted,valid,ratio,relative_to_all)
    return exact_mask(scores.reshape(len(scores),8,-1),admitted.reshape(len(scores),8,-1),quota,uniform).reshape_as(admitted)


def apply_exchange(mask,override,axis,layer,allowed):
    if override is None or override['axis']!=axis or override['layer']!=layer:
        return mask
    row,remove,insert=(override[k] for k in ('pack','remove','insert'))
    if not bool(mask[row,remove]) or bool(mask[row,insert]) or not bool(allowed[row,insert]):
        raise ValueError('Counterfactual exchange is outside the current legal action set')
    result=mask.clone();result[row,remove]=False;result[row,insert]=True
    return result


class OperatorValueRouter(nn.Module):
    """Two signed task components. State inputs are detached from the task encoder."""
    def __init__(self,channels,axes,width=32):
        super().__init__();self.channels=channels;self.width=width
        self.adapters=nn.ModuleDict({axis:nn.Sequential(nn.LayerNorm(channels),nn.Linear(channels,width),nn.GELU()) for axis in axes})
        self.heads=nn.ModuleDict({axis:nn.Sequential(nn.Linear(width*2+8,64),nn.GELU(),nn.Linear(64,2)) for axis in axes})
        for head in self.heads.values():
            nn.init.normal_(head[-1].weight,std=1e-3);nn.init.zeros_(head[-1].bias)

    def features(self,axis,state,valid,geometry,layer,depth,admitted=None):
        hidden=self.adapters[axis](state.detach().float())
        mean=(hidden*valid[...,None]).sum(1,keepdim=True)/valid.sum(1)[:,None,None].clamp_min(1)
        n=state.shape[1];p=n//8
        geo=geometry.detach().float()
        if geo.shape!=(*valid.shape,3):raise ValueError('Expected physical time/y/x token geometry')
        flags=valid if admitted is None else admitted
        scalars=torch.cat((geo,torch.full_like(geo[...,:1],layer/depth),flags[...,None].float(),
            valid.float().mean(-1)[:,None,None].expand(-1,n,-1),
            flags.float().mean(-1)[:,None,None].expand(-1,n,-1),
            torch.arange(n,device=state.device).remainder(p).float()[None,:,None].expand(len(state),-1,-1)/max(p-1,1)),-1)
        return torch.cat((hidden,mean.expand_as(hidden),scalars),-1)

    def forward(self,axis,state,valid,geometry,layer,depth,admitted=None):
        features=self.features(axis,state,valid,geometry,layer,depth,admitted)
        return self.heads[axis](features),features

    def macs(self,axis,rows):
        return rows*sum(m.in_features*m.out_features for module in (self.adapters[axis],self.heads[axis])
                        for m in module.modules() if isinstance(m,nn.Linear))
