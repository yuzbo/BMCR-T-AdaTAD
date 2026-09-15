"""Signed, set-conditioned one-swap measurements on a frozen EMA teacher."""
import torch
from scipy.optimize import linear_sum_assignment
from h65.transport import Selection
from .geometry import exchange


def localization_cost(proposals, scores, gt, labels, fixed=None):
    """One slot per GT, class-aware Hungarian match; dummy misses cost 2.

    Real match: 1-IoU + .25 boundary-L1 / GT duration, eligibility score >= .05.
    Unmatched/low-confidence GT is never silently excluded from the denominator.
    """
    n, p = len(gt), len(proposals)
    if not n:
        return 0., [], 0
    duration = (gt[:, 1] - gt[:, 0]).clamp_min(1e-6)
    intersection = (torch.minimum(gt[:, None, 1], proposals[None, :, 1]) -
                    torch.maximum(gt[:, None, 0], proposals[None, :, 0])).clamp_min(0)
    union = duration[:, None] + (proposals[:, 1] - proposals[:, 0])[None].clamp_min(0) - intersection
    iou = intersection / union.clamp_min(1e-6)
    distance = (gt[:, None] - proposals[None]).abs().sum(-1) / duration[:, None]
    cost = 1 - iou + .25 * distance
    eligible = scores[:, labels.long()].T >= .05
    cost = cost.masked_fill(~eligible, 2.).clamp_max(2.)
    augmented = torch.cat((cost, cost.new_full((n, n), 2.)), 1)
    if fixed is None:
        _, matched = linear_sum_assignment(augmented.cpu().numpy())
        matched = matched.tolist()
    else:
        matched = fixed
    value = augmented[torch.arange(n, device=gt.device), torch.tensor(matched, device=gt.device)].mean()
    observed = sum(index < p and bool(eligible[row, index]) and float(cost[row, index]) < 2
                   for row, index in enumerate(matched))
    return float(value), matched, observed


def classification_cost(head, raw, boxes, labels, fixed=None):
    if fixed is None:
        mapped = [mapping.to_rank(gt) for mapping, gt in zip(raw['maps'], boxes)]
        gt_cls, _ = head.prepare_targets(raw['points'], mapped, labels)
        target = torch.stack(gt_cls)
        normalizer = max(int(((target.sum(-1) > 0) & raw['valid']).sum()), 1)
        target = target * (1 - head.label_smoothing) + head.label_smoothing / (head.num_classes + 1)
        fixed = (target, normalizer)
    target, normalizer = fixed
    value = head.cls_loss(raw['logits'][raw['valid']], target[raw['valid']], reduction='sum') / normalizer
    return float(value), fixed


def row_selection(selection, row):
    return Selection(**{name: getattr(selection, name)[row:row + 1].detach() for name in Selection.__dataclass_fields__})


@torch.no_grad()
def counterfactual_targets(teacher, inputs, masks, metas, boxes, labels, provisional, conditional,
                           allowed_rows=None, candidates=None):
    teacher.eval()
    if allowed_rows is None:
        allowed_rows = masks.new_ones(len(inputs))
    observations = []
    for row in torch.where(allowed_rows)[0].tolist():
        member = conditional['member'][row]
        partner = conditional['partner'][row]
        if candidates is not None:
            feasible = partner.new_tensor(candidates)
        else:
            # Both membership states are scored at inference. With an alternating
            # route, nearest partners are usually not reciprocal, so sampling
            # only members would almost never supervise insertion utilities.
            chosen = []
            for membership in (True, False):
                eligible = torch.where(conditional['feasible'][row] & (member == membership))[0]
                if chosen:
                    previous = chosen[0]
                    # A reciprocal pair already supplies both labels below.
                    eligible = eligible[(eligible != partner[previous]) | (partner[eligible] != previous)]
                if len(eligible):
                    chosen.append(int(eligible[torch.randint(len(eligible), (), device=eligible.device)]))
            feasible = partner.new_tensor(chosen)
        if not len(feasible):
            continue
        route = row_selection(provisional, row)
        rgb, mask = inputs[row:row + 1], masks[row:row + 1]
        gt, lab = [boxes[row]], [labels[row]]
        baseline = teacher.raw_route(rgb, mask, route)
        cls_base, assignment = classification_cost(teacher.detector.rpn_head, baseline, gt, lab)
        loc_base, matching, matched = localization_cost(baseline['proposals'][0], baseline['scores'][0], gt[0], lab[0])
        for candidate in feasible.tolist():
            is_member = bool(member[candidate])
            counterpart = int(partner[candidate])
            remove, insert = (candidate, counterpart) if is_member else (counterpart, candidate)
            swapped = teacher.raw_route(rgb, mask, exchange(route, 0, remove, insert))
            cls_fixed, _ = classification_cost(teacher.detector.rpn_head, swapped, gt, lab, assignment)
            cls_rematch, _ = classification_cost(teacher.detector.rpn_head, swapped, gt, lab)
            loc, _, new_matched = localization_cost(swapped['proposals'][0], swapped['scores'][0], gt[0], lab[0])
            loc_fixed, _, _ = localization_cost(swapped['proposals'][0], swapped['scores'][0], gt[0], lab[0], matching)
            target = [cls_fixed - cls_base, loc - loc_base]
            common = dict(row=row, remove=remove, insert=insert, cls_base=cls_base, loc_base=loc_base,
                          cls_rematched_delta=cls_rematch-cls_base, loc_fixed_delta=loc_fixed-loc_base,
                          matched_before=matched, matched_after=new_matched, gt_count=len(gt[0]))
            signed = target if is_member else [-x for x in target]
            observations.append(dict(**common, candidate=candidate, candidate_is_member=is_member, target=signed))
            # The reciprocal label is valid only if the explicit partner is reciprocal.
            # Otherwise that candidate describes a different exchange and must not be mislabeled.
            if int(partner[counterpart]) == candidate:
                observations.append(dict(**common, candidate=counterpart, candidate_is_member=not is_member,
                                         target=[-x for x in signed]))
    return observations
