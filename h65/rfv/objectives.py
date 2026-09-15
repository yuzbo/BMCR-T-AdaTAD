"""State-local ranking of actual loss reductions, with the R0 Huber control."""
import math
import torch
from torch.nn import functional as F


def frozen_rank_scale(fit_target):
    """One positive scale preserves the ordering of raw cls + loc gains."""
    return fit_target.sum(-1).square().mean().sqrt().clamp_min(1e-8)


def centered_log_distribution(scores, valid, temperature=1.):
    count=valid.sum(-1,keepdim=True)
    if not bool((count>0).all()):
        raise ValueError('A ranking competition needs at least one valid candidate')
    mean=scores.masked_fill(~valid,0.).sum(-1,keepdim=True)/count
    logits=((scores-mean)/temperature).masked_fill(~valid,-torch.inf)
    return F.log_softmax(logits,dim=-1)


def value_loss(prediction,target,valid,component_scale,objective='r0',rank_scale=None,temperature=1.):
    """R1 = candidate-set JS + 0.1 * the unchanged R0 normalized Huber.

    Ranking uses (gain_cls + gain_loc) / one frozen fit scale. Equivalently,
    in a component-scaled notation w_c=s_c/r and w_l=s_l/r: no task-objective
    reweighting is introduced. STOP keeps its raw zero-gain meaning downstream.
    """
    component=F.smooth_l1_loss(prediction/component_scale,target/component_scale,reduction='none').mean(-1)
    huber=component[valid].mean()
    if objective=='r0':return dict(total=huber,rank=huber.detach()*0.,huber=huber)
    if objective!='r1' or rank_scale is None or temperature<=0:
        raise ValueError('R1 requires its frozen fit rank scale and positive temperature')
    target_log=centered_log_distribution(target.detach().sum(-1)/rank_scale,valid,temperature)
    student_log=centered_log_distribution(prediction.sum(-1)/rank_scale,valid,temperature)
    teacher=target_log.exp();student=student_log.exp()
    # Mask the -inf padding before subtraction; padded candidates carry no mass.
    target_log=target_log.masked_fill(~valid,0.)
    student_log=student_log.masked_fill(~valid,0.)
    mixture_log=((teacher+student)*.5).clamp_min(torch.finfo(student.dtype).tiny).log()
    js=.5*((teacher*(target_log-mixture_log)).sum(-1)+(student*(student_log-mixture_log)).sum(-1))
    rank=js.mean()
    return dict(total=rank+.1*huber,rank=rank,huber=huber)


def objective_record(name,fit_target,component_scale,temperature=1.):
    scale=float(frozen_rank_scale(fit_target))
    return dict(name=name,rank_scale=scale,rank_scale_source='fit raw cls+loc RMS',
        component_scale=component_scale.detach().cpu().tolist(),
        component_weights=(component_scale/scale).detach().cpu().tolist(),
        temperature=temperature,huber_weight=1. if name=='r0' else .1,
        ranking_utility='(raw gain_cls + raw gain_loc) / common fit scale',
        stop_utility=0.,js_max=math.log(2.))
