"""H65 targets and schedules audited against fixed-commit source."""
import math
import torch
import torch.nn.functional as F
from .geometry import occupancy


def curriculum(phase, updates):
    if phase == 'warm':
        return dict(alpha=0., adapt=0., bridge=0., contribution=0., action=1., transition=.5, boundary=2.)
    blend = .5 - .5 * math.cos(math.pi * min(updates / 2000, 1))
    feedback = .5 - .5 * math.cos(math.pi * min(max((updates - 667) / 1333, 0), 1))
    return dict(alpha=blend, adapt=blend, bridge=.25 * feedback, contribution=feedback,
                action=1 - .75 * blend, transition=.5 - .4 * blend, boundary=2 - 1.75 * blend)


def targets(masks, segments):
    times = torch.arange(masks.shape[1], device=masks.device).float()
    actions, transitions = [], []
    for mask, boxes in zip(masks, segments):
        if boxes.numel():
            action = ((times[:, None] >= boxes[:, 0]) & (times[:, None] < boxes[:, 1])).any(-1)
            distance = (times[:, None] - boxes.flatten()[None]).abs()
            kernels = torch.exp(-.5 * (distance / 2).square()) * (distance <= 4) * mask[:, None]
            kernel_mass = kernels.sum(0)
            kernels = kernels[:, kernel_mass > 0]
            target = (kernels / kernels.sum(0).clamp_min(1e-8)).mean(1) if kernels.shape[1] else times * 0
        else:
            action, target = times.bool() & False, times * 0
        actions.append(action.float())
        transitions.append(target)
    return torch.stack(actions), torch.stack(transitions)


def distribution_loss(logits, target, masks, active=None):
    mass = target.sum(-1)
    use = mass > 1e-8
    if active is not None:
        use &= active
    log_prob = (logits.float() / .7).masked_fill(~masks, -1e4).log_softmax(-1)
    value = -(target / mass[:, None].clamp_min(1e-8) * log_prob).sum(-1)
    return value[use].mean() if bool(use.any()) else logits.sum() * 0


def auxiliary_losses(output, selection, masks, segments, weights):
    action, target = targets(masks, segments)
    action_loss = F.binary_cross_entropy_with_logits(output['action_logits'][masks].float(), action[masks])
    transition_loss = distribution_loss(output['aux_transition'], target, masks)
    local_mass = F.conv1d(occupancy(selection, masks)[:, None], target.new_ones(1, 1, 9), padding=4)[:, 0]
    active = target.sum(-1) > 0
    boundary = (target * torch.exp(-local_mass)).sum(-1)
    boundary_loss = boundary[active].mean() if bool(active.any()) else local_mass.sum() * 0
    return dict(loss_actionness=weights['action'] * action_loss,
                loss_transition=weights['transition'] * transition_loss,
                loss_boundary=weights['boundary'] * boundary_loss)
