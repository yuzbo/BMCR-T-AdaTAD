"""Fixed-budget sampling and physical-time alignment.

Mechanism reference: H65 commit 04c35a3, structured_selection.py and
duca_online_frame_selector.py. This module has no dependency on that fork.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Selection:
    indices: torch.Tensor
    continuous: torch.Tensor
    valid: torch.Tensor
    density: torch.Tensor
    rates: torch.Tensor


class _FixedBudgetGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, rates, temperature):
        ctx.save_for_backward(rates)
        ctx.temperature = temperature
        return rates

    @staticmethod
    def backward(ctx, grad):
        rates, = ctx.saved_tensors
        slope = rates * (1 - rates) / ctx.temperature
        baseline = (grad * slope).sum() / slope.sum().clamp_min(torch.finfo(slope.dtype).eps)
        return slope * (grad - baseline), None, None


def calibrated_rates(logits, budget, temperature=0.7):
    """Solve sum(sigmoid((logits-threshold)/temperature))=budget."""
    if budget == logits.numel():
        return logits * 0 + 1
    with torch.no_grad():
        lower = logits.min() - 32 * temperature
        upper = logits.max() + 32 * temperature
        for _ in range(48):
            midpoint = (lower + upper) / 2
            mass = ((logits - midpoint) / temperature).sigmoid().sum()
            lower = torch.where(mass > budget, midpoint, lower)
            upper = torch.where(mass > budget, upper, midpoint)
        rates = ((logits - (lower + upper) / 2) / temperature).sigmoid()
    return _FixedBudgetGradient.apply(logits, rates, temperature)


def sample_rates(logits, masks, budget=384, alpha=1.0, temperature=0.7,
                   coverage_floor=0.05, smoothing_kernel=5):
    """Hard exact-K systematic sampling from capacity-bounded retention rates.

Only prefix masks produced by the OpenTAD video window pipeline are supported.
Short final windows use min(K, valid_length) unique observations; remaining
physical execution slots repeat the final frame and are marked invalid.
"""
    if logits.ndim != 2 or masks.shape != logits.shape:
        raise ValueError("logits and masks must be [B,T]")
    if not 1 <= budget <= logits.shape[1]:
        raise ValueError("budget must be between 1 and T")
    if temperature <= 0 or not 0 <= alpha <= 1 or not 0 <= coverage_floor < 1:
        raise ValueError("invalid density parameters")
    if smoothing_kernel < 1 or smoothing_kernel % 2 != 1:
        raise ValueError("smoothing kernel must be positive and odd")
    rows, coordinates, valids, densities, rate_rows = [], [], [], [], []
    for scores, mask in zip(logits.float(), masks.bool()):
        length = int(mask.sum())
        if length == 0 or not torch.equal(mask, torch.arange(mask.numel(), device=mask.device) < length):
            raise ValueError("a nonempty contiguous valid prefix is required")
        k = min(budget, length)
        scores = scores[:length]
        radius = smoothing_kernel // 2
        if radius:
            scores = F.avg_pool1d(F.pad(scores[None, None], (radius, radius), mode="replicate"),
                                 smoothing_kernel, stride=1)[0, 0]
        rates = calibrated_rates(scores, k, temperature)
        rates = (1 - coverage_floor) * rates + coverage_floor * k / length
        rates = (1 - alpha) * k / length + alpha * rates
        rates = rates.clamp(0, 1)
        # Stage-1/companion uses endpoint anchors rather than systematic midpoints.
        if alpha == 0:
            hard = (torch.linspace(0, length - 1, k, device=scores.device).round().long()
                    if k > 1 else scores.new_zeros(1, dtype=torch.long))
            positions = hard.float() + rates.sum() * 0
        else:
            cumulative = rates.cumsum(0)
            thresholds = torch.arange(k, device=scores.device) + 0.5
            hard = torch.searchsorted(cumulative.detach().contiguous(), thresholds).clamp_max(length - 1)
            before = F.pad(cumulative[:-1], (1, 0))
            fraction = ((thresholds - before[hard]) / rates[hard].clamp_min(1e-7)).clamp(0, 1)
            direction = torch.where(hard < length - 1, 1, -1)
            positions = hard.float() + direction * (fraction - fraction.detach())
        rows.append(F.pad(hard, (0, budget - k), value=length - 1))
        coordinates.append(F.pad(positions, (0, budget - k), value=length - 1))
        valids.append(torch.arange(budget, device=scores.device) < k)
        densities.append(F.pad(rates / k, (0, logits.shape[1] - length)))
        rate_rows.append(F.pad(rates, (0, logits.shape[1] - length)))
    return Selection(torch.stack(rows), torch.stack(coordinates), torch.stack(valids),
                     torch.stack(densities), torch.stack(rate_rows))


def gather_with_transport(frames, selection, bridge_weight=0.25):
    """Exact hard forward; local, detached RGB slope carries policy gradients.

frames: [B,1,3,T,H,W]. The bridge has identically zero forward value.
"""
    if frames.ndim != 6 or frames.shape[1] != 1:
        raise ValueError("expected OpenTAD frame input [B,1,3,T,H,W]")
    frames = frames.float()
    index = selection.indices[:, None, None, :, None, None]
    shape = (frames.shape[0], 1, frames.shape[2], index.shape[3], *frames.shape[-2:])
    hard = frames.gather(3, index.expand(shape))
    if bridge_weight == 0 or not torch.is_grad_enabled():
        return hard
    left = (index - 1).clamp_min(0)
    lengths = (selection.density > 0).sum(-1)[:, None, None, None, None, None]
    right = torch.minimum(index + 1, lengths - 1)
    slope = ((frames.gather(3, right.expand(shape)) - frames.gather(3, left.expand(shape))) /
             (right - left).clamp_min(1)).detach()
    displacement = selection.continuous - selection.continuous.detach()
    displacement = displacement * selection.valid
    return hard + bridge_weight * slope * displacement[:, None, None, :, None, None]


def interpolate_irregular(values, source_times, query_times):
    """Linear interpolation on a strictly increasing timeline, edge replicated.

values [C,S], source_times [S], query_times [Q]. Geometry is metadata;
gradients remain on values, never on detached hard selection coordinates.
"""
    source_times, query_times = source_times.detach(), query_times.detach()
    if source_times.numel() == 1:
        return values[:, :1].expand(-1, query_times.numel())
    query_times = query_times.clamp(source_times[0], source_times[-1])
    right = torch.searchsorted(source_times.contiguous(), query_times.contiguous()).clamp(1, source_times.numel() - 1)
    left = right - 1
    fraction = (query_times - source_times[left]) / (source_times[right] - source_times[left])
    return values[:, left] * (1 - fraction) + values[:, right] * fraction


def restore_tubelets(features, selection, original_masks):
    """Restore [B,C,K/2] onto original T-frame detector lattice.

Tubelet i represents the mean physical position of hard frames 2i and 2i+1.
Full selection equals official linear interpolate(..., align_corners=False).
"""
    if selection.indices.shape[1] % 2 or features.shape[-1] * 2 != selection.indices.shape[1]:
        raise ValueError("features must have one output per two selected frames")
    centers = selection.indices.float().unflatten(1, (-1, 2)).mean(-1)
    counts = (selection.valid.sum(-1) + 1) // 2
    query = torch.arange(original_masks.shape[1], device=features.device, dtype=torch.float32)
    rows = [interpolate_irregular(feat[:, :int(n)], time[:int(n)], query)
            for feat, time, n in zip(features, centers, counts)]
    return torch.stack(rows) * original_masks[:, None].to(features.dtype)
