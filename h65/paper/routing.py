"""Joint finite-menu budgets and signed, uncertainty-aware frame exchange."""
import torch
from torch import nn
from h65.frame.router import ActionRouter, candidate_pairs, swap_selection


def plans():
    values=[(768,1.,1.),(384,1.,1.),(384,1.,.48),(384,.5,1.),(384,.5,.48),
            (320,1.,1.),(256,1.,1.),(320,.5,.48),(256,.5,.48),
            (384,.75,1.),(384,1.,.75),(320,.75,.75),
            (768,1.,.48),(768,.5,1.),(768,.5,.48)]
    return [dict(id=f'K{k}_D{int(d*100)}_S{int(s*100)}',frames=k,depth=d,space=s) for k,d,s in values]


def video_context(output,masks,budget_fraction):
    hidden=output['hidden'].detach().float();valid=masks[...,None]
    count=valid.sum(1).clamp_min(1)
    mean=(hidden*valid).sum(1)/count
    var=((hidden-mean[:,None]).square()*valid).sum(1)/count
    action=output['action_logits'].detach().float().sigmoid()
    scalars=torch.stack(((action*masks).sum(1)/masks.sum(1).clamp_min(1),
                         action.masked_fill(~masks,0).amax(1),masks.float().mean(1),
                         action.new_full((len(action),),budget_fraction)),-1)
    return torch.cat((mean,var.sqrt(),scalars),-1)


class BudgetRouter(nn.Module):
    def __init__(self,menu=None):
        super().__init__();self.menu=menu or plans();n=len(self.menu)
        self.network=nn.Sequential(nn.LayerNorm(196),nn.Linear(196,128),nn.GELU(),nn.Linear(128,n*4))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        self.register_buffer('cost_gflops',torch.full((n,),float('nan')))
        self.register_buffer('reference_gflops',torch.tensor(float('nan')))
        self.register_buffer('scales',torch.full((2,),.001))
        self.register_buffer('sigma_calibration',torch.ones(2))

    def distribution(self,context):
        value=self.network(context.float()).reshape(-1,len(self.menu),4)
        mean=value[...,:2];mean=mean-mean[:,:1]
        return mean,value[...,2:].clamp(-8,8)

    def choose(self,context,fraction,risk_weight=.25):
        if not bool(torch.isfinite(self.cost_gflops).all()) or not bool(torch.isfinite(self.reference_gflops)):
            raise RuntimeError('Actual full-window cost table is required before routed deployment')
        mean,logvar=self.distribution(context)
        score=(mean-risk_weight*logvar.mul(.5).exp()*self.sigma_calibration).mean(-1)
        feasible=self.cost_gflops<=self.reference_gflops*fraction
        if not bool(feasible.any()):raise ValueError('Requested budget is below this finite menu')
        score=score.masked_fill(~feasible[None],-torch.inf)
        # Use the largest affordable plan for exact ties before the value head learns.
        maximum=score.amax(-1,keepdim=True);tied=score==maximum
        indices=self.cost_gflops[None].expand_as(score).masked_fill(~tied,-torch.inf).argmax(-1)
        return indices,dict(mean=mean,logvar=logvar,feasible=feasible)

    def pair_loss(self,context,base,action,target):
        mean,logvar=self.distribution(context)
        predicted=mean[:,action]-mean[:,base]
        variance=logvar[:,action].exp()+logvar[:,base].exp()
        target=target.detach()/self.scales.clamp_min(1e-4)
        return .5*((predicted-target).square()/variance+variance.log()).mean()


class FrameRouter(ActionRouter):
    def __init__(self):
        super().__init__()
        self.network[-1]=nn.Linear(64,4)
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        self.register_buffer('sigma_calibration',torch.ones(2))

    def distribution(self,output,selection,masks,pairs):
        raw=self.network(self.features(output,selection,masks,pairs).float())
        return raw[:,:2],raw[:,2:].clamp(-8,8)

    def refine(self,output,selection,masks,scope='local',risk_weight=.25):
        pairs=candidate_pairs(output,selection,masks,scope)
        if not pairs:return selection,dict(changes=[],pair_count=0)
        mean,logvar=self.distribution(output,selection,masks,pairs)
        score=(mean-risk_weight*logvar.mul(.5).exp()*self.sigma_calibration).mean(-1)
        result=selection;changes=[]
        for row in range(len(masks)):
            ids=[i for i,p in enumerate(pairs) if p[0]==row]
            if not ids:continue
            index=score.new_tensor(ids,dtype=torch.long);best=index[score[index].argmax()]
            if float(score[best])>0:
                _,remove,insert=pairs[int(best)];result=swap_selection(result,row,remove,insert)
                changes.append(dict(row=row,remove=remove,insert=insert,predicted=float(mean[best].mean()),
                                    conservative_gain=float(score[best])))
        return result,dict(changes=changes,pair_count=len(pairs))

    def regression_loss(self,features,target):
        raw=self.network(features.float());mean=raw[:,:2];logvar=raw[:,2:].clamp(-8,8)
        normalized=target.detach()/self.scales.clamp_min(1e-4)
        return .5*((mean-normalized).square()*(-logvar).exp()+logvar).mean()
