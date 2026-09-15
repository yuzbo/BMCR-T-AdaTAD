"""Read-only decomposition of the trainer's single global clipping operation."""
import torch


def derivatives(loss,parameters,retain_graph=False):
    values=torch.autograd.grad(loss,parameters,retain_graph=retain_graph,allow_unused=True)
    return [torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,values)]


def add_vectors(left,right):
    return [a+b for a,b in zip(left,right)]


def group_name(name):
    if name.startswith('encoder.engine.value_router.'):return 'value_router'
    if name.startswith('readout.'):return 'detector_head'
    if name.startswith('decoder.'):return 'recovery'
    if name.startswith('encoder.scout.'):return 'scout'
    if name.startswith('encoder.backbone.'):return 'encoder_adapter' if 'adapter' in name else 'encoder_other'
    if name.startswith('encoder.engine.'):return 'light_operators'
    return 'other'


def gradient_summary(names,values):
    totals={}
    for name,value in zip(names,values):
        key=group_name(name);square=value.double().square().sum()
        totals[key]=totals.get(key,0.)+square
    norms={key:float(square.sqrt()) for key,square in totals.items()}
    norm=sum(value**2 for value in norms.values())**.5
    return dict(global_norm=norm,groups=norms,clip_coefficient=min(1.,1./(norm+1e-6)),
        clip_max_norm=1.,coefficient_scope='same global norm1 rule; no clipping or optimizer mutation performed')
