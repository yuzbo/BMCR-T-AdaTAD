"""407D Plain controls and bounded cheap Graph-conditioned temporal Value."""
import copy
import torch
from torch import nn
from h65.raw.value import FEATURE_DIM
from h65.paper.edge_ops import gather_nodes
from h65.rfv.graph import TemporalGraphMessage


def interpolate_nodes(context,times,queries):
    """Physical-time interpolation, independent of selected/packed ranks."""
    right=torch.searchsorted(times.contiguous(),queries.contiguous()).clamp(1,times.shape[1]-1)
    left=right-1
    lo=times.gather(1,left);hi=times.gather(1,right)
    fraction=((queries-lo)/(hi-lo).clamp_min(1e-8)).clamp(0,1)[...,None]
    return gather_nodes(context,left)*(1-fraction)+gather_nodes(context,right)*fraction


class TemporalProbe(nn.Module):
    def __init__(self,variant='plain_m',hidden=128,graph_width=64):
        super().__init__()
        if variant not in ('plain_m','plain_l','static_graph','dynamic_graph'):
            raise ValueError('Unregistered G1-T predictor')
        self.spec=dict(variant=variant,hidden=hidden,graph_width=graph_width)
        self.variant=variant
        self.register_buffer('input_mean',torch.zeros(FEATURE_DIM))
        self.register_buffer('input_scale',torch.ones(FEATURE_DIM))
        self.register_buffer('target_scale',torch.ones(2))
        width=FEATURE_DIM
        if variant!='plain_m':
            self.register_buffer('cheap_mean',torch.zeros(96))
            self.register_buffer('cheap_scale',torch.ones(96))
            self.node_adapter=nn.Linear(101,graph_width)
            if variant=='plain_l':
                # Same node evidence/pooling as Graph, with no edges/messages.
                self.node_mlp=nn.Sequential(nn.LayerNorm(graph_width),nn.Linear(graph_width,2*graph_width),
                    nn.GELU(),nn.Linear(2*graph_width,graph_width))
            else:
                self.graph=TemporalGraphMessage(graph_width,dynamic=variant=='dynamic_graph')
            width+=3*graph_width
        self.network=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,64),nn.GELU(),nn.Linear(64,2))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,state):
        local=(state['descriptor']-self.input_mean)/self.input_scale
        if self.variant!='plain_m':
            # Only observed cheap nodes and public time/support quantities enter.
            hidden=(state['cheap']-self.cheap_mean)/self.cheap_scale
            valid=state['node_valid']
            scalar=torch.stack((state['times'],state['actionness'],state['transition'],
                state['support_distance'],state['support_occupancy']),-1)
            nodes=self.node_adapter(torch.cat((hidden,scalar),-1))*valid[...,None]
            geometry=torch.stack((state['times'],torch.zeros_like(state['times']),torch.zeros_like(state['times']),
                state['support_occupancy'],valid.float()),-1)
            if self.variant=='plain_l':context=(nodes+self.node_mlp(nodes))*valid[...,None]
            else:context,_,_,_=self.graph(nodes,valid,geometry)
            remove=interpolate_nodes(context,state['times'],state['remove_times'])
            insert=interpolate_nodes(context,state['times'],state['insert_times'])
            support=interpolate_nodes(context,state['times'],state['support_times'])
            mask=state['support_valid']
            pooled=(support*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]
            local=torch.cat((local,remove,insert,pooled[:,None].expand(-1,local.shape[1],-1)),-1)
        return self.network(local)*self.target_scale

    @torch.no_grad()
    def set_normalization(self,descriptor,target,cheap):
        self.input_mean.copy_(descriptor.mean(0));self.input_scale.copy_(descriptor.std(0).clamp_min(1e-6))
        self.target_scale.copy_(target.square().mean(0).sqrt().clamp_min(1e-8))
        if self.variant!='plain_m':
            self.cheap_mean.copy_(cheap.mean(0));self.cheap_scale.copy_(cheap.std(0).clamp_min(1e-6))

    def snapshot(self):
        return dict(spec=copy.deepcopy(self.spec),state={k:v.detach().cpu().clone() for k,v in self.state_dict().items()})


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_plain_width(graph_width=64):
    # The static/dynamic Graph variants have the same actual parameter set.
    with torch.random.fork_rng(devices=[]):
        graph=TemporalProbe('dynamic_graph',graph_width=graph_width)
        target=parameter_count(graph)
        tiny=TemporalProbe('plain_l',hidden=1,graph_width=graph_width)
        coefficient=FEATURE_DIM+3*graph_width+65
        constant=parameter_count(tiny)-coefficient
        width=round((target-constant)/coefficient)
        plain=TemporalProbe('plain_l',hidden=width,graph_width=graph_width)
    if abs(parameter_count(plain)-target)/target>.01:
        raise ValueError('Plain-L parameter count differs from Graph by more than 1%')
    return width


def from_snapshot(snapshot,device='cpu'):
    model=TemporalProbe(**snapshot['spec']).to(device)
    model.load_state_dict(snapshot['state'],strict=True)
    return model


class ProbeEMA:
    """Genuine cumulative trainable-parameter EMA, with frozen fit normalization."""
    def __init__(self,model,decay=.99):
        self.decay=decay;self.updates=0
        self.state={k:v.detach().clone() for k,v in model.state_dict().items()}
        self.parameters=set(dict(model.named_parameters()))

    @torch.no_grad()
    def update(self,model):
        for key,value in model.state_dict().items():
            if key in self.parameters:self.state[key].mul_(self.decay).add_(value.detach(),alpha=1-self.decay)
            else:self.state[key].copy_(value)
        self.updates+=1

    def snapshot(self,model):
        return dict(spec=copy.deepcopy(model.spec),state={k:v.detach().cpu().clone() for k,v in self.state.items()},
            ema=dict(decay=self.decay,optimizer_updates=self.updates,kind='cumulative every-step parameter EMA'))
