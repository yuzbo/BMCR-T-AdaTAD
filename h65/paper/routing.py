"""Joint finite-menu budgets and signed, uncertainty-aware frame exchange."""
import torch
from torch import nn
from h65.frame.router import ActionRouter,swap_selection


def candidate_pairs(output,selection,masks,scope='local'):
    member=torch.zeros_like(masks,dtype=torch.long).scatter_add(1,selection.indices,selection.valid.long())>0
    result=[]
    for row in range(len(masks)):
        selected=member[row].nonzero().flatten();insert=(masks[row]&~member[row]).nonzero().flatten()
        if not len(insert) or not len(selected):continue
        distance=(insert[:,None]-selected[None]).abs()
        if scope=='local':distance=distance.masked_fill(insert[:,None]//16!=selected[None]//16,masks.shape[1]+1)
        nearest,position=distance.min(1);keep=nearest<=masks.shape[1]
        pairs=torch.stack((torch.full_like(insert,row),selected[position],insert),1)[keep]
        if output.get('coverage_bins'):
            length=int(masks[row].sum());bins=min(output['coverage_bins'],length,len(selected))
            bucket=(torch.arange(length,device=masks.device)*bins//length).clamp_max(bins-1)
            counts=torch.bincount(bucket[selected],minlength=bins)
            pairs=pairs[(bucket[pairs[:,1]]==bucket[pairs[:,2]])|(counts[bucket[pairs[:,1]]]>1)]
        result.extend(map(tuple,pairs.tolist()))
    return result


def plans():
    values=[(768,1.,1.),(384,1.,1.),(384,1.,.48),(384,.5,1.),(384,.5,.48),
            (320,1.,1.),(256,1.,1.),(320,.5,.48),(256,.5,.48),
            (384,.75,1.),(384,1.,.75),(320,.75,.75),
            (768,1.,.48),(768,.5,1.),(768,.5,.48)]
    return [dict(id=f'K{k}_D{int(d*100)}_S{int(s*100)}',frames=k,depth=d,space=s) for k,d,s in values]


def video_context(output,masks):
    hidden=output['hidden'].detach().float();valid=masks[...,None]
    count=valid.sum(1).clamp_min(1)
    mean=(hidden*valid).sum(1)/count
    var=((hidden-mean[:,None]).square()*valid).sum(1)/count
    action=output['action_logits'].detach().float().sigmoid()
    scalars=torch.stack(((action*masks).sum(1)/masks.sum(1).clamp_min(1),
                         action.masked_fill(~masks,0).amax(1),masks.float().mean(1),
                         ((action[:,1:]-action[:,:-1]).abs()*(masks[:,1:]&masks[:,:-1])).sum(1)/(masks[:,1:]&masks[:,:-1]).sum(1).clamp_min(1)),-1)
    return torch.cat((mean,var.sqrt(),scalars),-1)


class BudgetRouter(nn.Module):
    def __init__(self,menu=None):
        super().__init__();self.menu=menu or plans();n=len(self.menu)
        self.network=nn.Sequential(nn.LayerNorm(196),nn.Linear(196,128),nn.GELU(),nn.Linear(128,n*4))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        self.register_buffer('cost_gflops',torch.full((n,),float('nan')))
        self.register_buffer('reference_gflops',torch.tensor(float('nan')))
        self.register_buffer('fixed_nonencoder_macs',torch.full((n,),float('nan'),dtype=torch.float64))
        self.register_buffer('scales',torch.full((2,),.001))
        self.register_buffer('sigma_calibration',torch.ones(2))

    def distribution(self,context):
        value=self.network(context.float()).reshape(-1,len(self.menu),4)
        mean=value[...,:2];mean=mean-mean[:,:1]
        logvar=value[...,2:].clamp(-8,8)
        # The full-reference gain is exactly zero relative to itself.
        logvar=torch.cat((torch.full_like(logvar[:,:1],-torch.inf),logvar[:,1:]),1)
        return mean,logvar

    def choose(self,context,fraction,risk_weight=.25,distribution=None):
        if not bool(torch.isfinite(self.cost_gflops).all()) or not bool(torch.isfinite(self.reference_gflops)):
            raise RuntimeError('Actual full-window cost table is required before routed deployment')
        mean,logvar=self.distribution(context) if distribution is None else distribution
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
        rows=torch.arange(len(mean),device=mean.device)
        base=torch.as_tensor(base,device=mean.device).expand(len(mean))
        action=torch.as_tensor(action,device=mean.device).expand(len(mean))
        predicted=mean[rows,action]-mean[rows,base]
        variance=logvar[rows,action].exp()+logvar[rows,base].exp()
        target=target.detach()/self.scales.clamp_min(1e-4)
        return .5*((predicted-target).square()/variance+variance.log()).mean()


class PlanFeatureNorm(nn.Module):
    def __init__(self,norm):super().__init__();self.norm=norm
    def forward(self,value):return torch.cat((self.norm(value[...,:294]),value[...,294:]),-1)


class FrameRouter(ActionRouter):
    def __init__(self,plan_aware=False):
        super().__init__()
        self.network[-1]=nn.Linear(64,4)
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        self.register_buffer('sigma_calibration',torch.ones(2))
        self.plan_aware=plan_aware;self.use_plan_context=True
        if plan_aware:
            old=self.network[1]
            with torch.random.fork_rng(devices=[]):new=nn.Linear(300,64)
            with torch.no_grad():new.weight.zero_();new.weight[:,:294].copy_(old.weight);new.bias.copy_(old.bias)
            self.network[0]=PlanFeatureNorm(self.network[0]);self.network[1]=new

    def features(self,output,selection,masks,pairs,plan=None):
        hidden=output['hidden'].detach();b,t,c=hidden.shape
        if not pairs:return hidden.new_empty((0,300 if self.plan_aware else 294))
        member=torch.zeros_like(masks,dtype=torch.long).scatter_add(1,selection.indices,selection.valid.long())>0
        mean=(hidden*member[...,None]).sum(1)/member.sum(-1,keepdim=True).clamp_min(1)
        row,remove,insert=torch.tensor(pairs,device=hidden.device,dtype=torch.long).unbind(1)
        action=output['action_logits'].detach().sigmoid()
        ratio=selection.valid.sum(-1)/masks.sum(-1)
        scalars=torch.stack((remove.float()/t,insert.float()/t,(insert-remove).float()/t,
                             action[row,remove].float(),action[row,insert].float(),ratio[row].float()),-1).to(hidden.dtype)
        value=torch.cat((hidden[row,remove],hidden[row,insert],mean[row],scalars),-1)
        if self.plan_aware:
            if plan is None or not self.use_plan_context:condition=value.new_zeros(6)
            else:condition=value.new_tensor([plan['frames']/768,plan['depth'],plan['space'],float(plan.get('full_kv',False)),float(plan.get('depth_bypass')=='light'),len(plan.get('mod_layers',(1,3,5,7,9)))/12])
            value=torch.cat((value,condition.expand(len(value),-1)),-1)
        return value

    def distribution(self,output,selection,masks,pairs,plan=None):
        raw=self.network(self.features(output,selection,masks,pairs,plan).float())
        return raw[:,:2],raw[:,2:].clamp(-8,8)

    def refine(self,output,selection,masks,scope='local',risk_weight=.25,plan=None):
        pairs=candidate_pairs(output,selection,masks,scope)
        if not pairs:return selection,dict(changes=[],pair_count=0)
        mean,logvar=self.distribution(output,selection,masks,pairs,plan)
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
