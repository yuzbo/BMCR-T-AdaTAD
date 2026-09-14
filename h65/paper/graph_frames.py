"""Coverage-preserving frame exchange with cheap, pre-encoder graph context."""
import torch
from torch import nn
from .routing import FrameRouter
from h65.frame.router import swap_selection


def protect_coverage(selection,preview,masks,bins=32):
    changes=[];result=selection
    for row,mask in enumerate(masks):
        length=int(mask.sum());count=int(selection.valid[row].sum());groups=min(bins,length,count)
        if not groups:continue
        bucket=(torch.arange(length,device=mask.device)*groups//length).clamp_max(groups-1)
        selected=result.indices[row,result.valid[row]]
        totals=torch.bincount(bucket[selected],minlength=groups)
        for group in (totals==0).nonzero().flatten().tolist():
            candidates=(bucket==group).nonzero().flatten()
            insert=candidates[preview['rate_logits'][row,candidates].argmax()]
            eligible=selected[totals[bucket[selected]]>1]
            remove=eligible[preview['rate_logits'][row,eligible].argmin()]
            result=swap_selection(result,row,int(remove),int(insert))
            totals[bucket[remove]]-=1;totals[group]+=1
            selected=result.indices[row,result.valid[row]]
            changes.append(dict(row=row,remove=int(remove),insert=int(insert)))
    return result,changes


class GraphFrameNetwork(nn.Module):
    def __init__(self,base,base_width):
        super().__init__();self.base=base;self.base_width=base_width
        self.graph=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,64),nn.GELU(),nn.Linear(64,4))
        nn.init.zeros_(self.graph[-1].weight);nn.init.zeros_(self.graph[-1].bias)
    def forward(self,value):return self.base(value[...,:self.base_width])+self.graph(value[...,self.base_width:])


class GraphFrameRouter(FrameRouter):
    def __init__(self,base):
        nn.Module.__init__(self)
        self.plan_aware=base.plan_aware;self.use_plan_context=base.use_plan_context
        self.register_buffer('scales',base.scales.detach().clone())
        self.register_buffer('sigma_calibration',base.sigma_calibration.detach().clone())
        self.base_width=300 if self.plan_aware else 294
        self.network=GraphFrameNetwork(base.network,self.base_width)
    def features(self,output,selection,masks,pairs,plan=None):
        base=super().features(output,selection,masks,pairs,plan)
        if not pairs:return base.new_empty((0,self.base_width+384))
        context=output['graph_context'].detach()
        row,remove,insert=torch.tensor(pairs,device=context.device,dtype=torch.long).unbind(1)
        valid=masks.reshape(len(masks),-1,2).any(-1)
        mean=(context*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        return torch.cat((base,context[row,remove//2],context[row,insert//2],mean[row]),-1)
