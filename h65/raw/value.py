"""Shared domain-indicator-free, set-conditioned temporal exchange value."""
import torch
from torch import nn
from .model import interpolate_preview


SCALARS = ('remove_time','insert_time','signed_distance','remove_left_gap','remove_right_gap',
    'insert_left_gap','insert_right_gap','remove_preview_gap','insert_preview_gap',
    'remove_official_member','insert_official_member','remove_preview_observed','insert_preview_observed',
    'remove_membership','insert_membership','valid_fraction','support_density','depth_capacity','space_capacity',
    'remove_actionness','insert_actionness','remove_transition','insert_transition')
FEATURE_DIM = 96*4+len(SCALARS)


def descriptors(episode, timeline, preview, proposal, selection, pairs, plan=None):
    """Only common cheap observations and public physical/support metadata enter here."""
    hidden = preview['hidden'][0].detach()
    device = hidden.device
    lo, hi = episode.bounds
    span = max(hi-lo,1)
    support = torch.tensor(selection.support,device=device,dtype=torch.float32)
    candidate_hidden = interpolate_preview(hidden,timeline.frame_ids,selection.support)
    support_mean = candidate_hidden.mean(0)
    global_mean = hidden.mean(0)
    official = set(episode.official_frame_ids[:sum(episode.official_valid)])
    seen = set(timeline.frame_ids)
    preview_ids = torch.tensor(timeline.frame_ids,device=device,dtype=torch.float32)
    plan = plan or dict(depth=1.,space=1.)
    def gaps(frame):
        left = support[support < frame]
        right = support[support > frame]
        return ((frame-float(left[-1]))/span if len(left) else (frame-lo)/span,
                (float(right[0])-frame)/span if len(right) else (hi-frame)/span)
    rows = []
    for remove,insert in pairs:
        h = interpolate_preview(hidden,timeline.frame_ids,[remove,insert])
        action = interpolate_preview(preview['action_logits'][0,:,None].detach(),timeline.frame_ids,[remove,insert]).sigmoid().flatten()
        transition = interpolate_preview(preview['transition_logits'][0,:,None].detach(),timeline.frame_ids,[remove,insert]).flatten()
        scalars = [(remove-lo)/span,(insert-lo)/span,(insert-remove)/span,*gaps(remove),*gaps(insert),
            float((preview_ids-remove).abs().min())/span,float((preview_ids-insert).abs().min())/span,
            float(remove in official),float(insert in official),float(remove in seen),float(insert in seen),
            1.,0.,sum(selection.valid)/len(selection.valid),len(selection.support)/max(hi-lo+1,1),plan['depth'],plan['space'],
            float(action[0]),float(action[1]),float(transition[0]),float(transition[1])]
        rows.append(torch.cat((h[0],h[1],support_mean,global_mean,hidden.new_tensor(scalars))))
    return torch.stack(rows) if rows else hidden.new_empty((0,FEATURE_DIM))


class TemporalValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('input_mean',torch.zeros(FEATURE_DIM))
        self.register_buffer('input_scale',torch.ones(FEATURE_DIM))
        self.register_buffer('target_scale',torch.ones(2))
        self.network = nn.Sequential(nn.Linear(FEATURE_DIM,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,2))
        # A fresh Value router must preserve the Uniform support and choose STOP.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x):
        return self.network((x-self.input_mean)/self.input_scale)*self.target_scale

    def utility(self,x):
        return self(x).sum(-1)


def simple_proxy(x):
    # Pre-registered cheap transition gain; no fitted weights and no GT.
    return x[:,-1]-x[:,-2]
