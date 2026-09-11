"""Separate action, policy-encoder and auxiliary-encoder gradient paths."""
import torch
from torch import nn
import torch.nn.functional as F
from h65.scout import H65Scout, transition_descriptors


def scaled_gradient(value, scale):
    return value.detach() + scale * (value - value.detach())


class FormalScout(H65Scout):
    def __init__(self):
        super().__init__()
        # Present in both variants so warmup checkpoints share an exact schema.
        self.conditional = nn.Sequential(nn.LayerNorm(358), nn.Linear(358, 64), nn.GELU(), nn.Linear(64, 2))
        nn.init.zeros_(self.conditional[-1].weight)
        nn.init.zeros_(self.conditional[-1].bias)

    def forward(self, frames, masks, adapt_scale=1., variant='h65'):
        b, _, c, t, h, w = frames.shape
        flat = frames[:, 0].permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float() / 255
        small = F.interpolate(flat, (self.spatial_size, self.spatial_size), mode='bilinear', align_corners=False)
        stem = self.spatial_stem(small).flatten(1).reshape(b, t, -1).transpose(1, 2) * masks[:, None]
        self.reference.device = frames.device
        for module in self.temporal.modules():
            if hasattr(module, 'window_mask'):
                module.window_mask = module.window_mask.to(frames.device)
        actions, hidden_rows = [], []
        for row in range(b):
            mask = masks[row:row + 1, None].float()
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(frames.device) if frames.is_cuda else None
            out, hidden = self.temporal.encoder(stem[row:row + 1], mask)
            action, decoder_hidden = out, hidden
            for decoder in self.temporal.decoders:
                action, decoder_hidden = decoder(action.softmax(1) * mask, decoder_hidden * mask, mask)
            if self.training:
                # Same dropout realization, detached RGB stem; auxiliary scale .25
                # remains active even while the policy scale is zero in warmup.
                devices = [frames.device.index] if frames.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state(cuda_rng, frames.device)
                    _, replay = self.temporal.encoder(stem[row:row + 1].detach(), mask)
                hidden = hidden.detach() + (replay - replay.detach())
            actions.append(action[:, 1] - action[:, 0])
            hidden_rows.append(hidden.transpose(1, 2))
        action = torch.cat(actions)
        hidden = torch.cat(hidden_rows)
        descriptors = transition_descriptors(action.detach(), hidden, masks)
        representation = self.transition(scaled_gradient(descriptors, adapt_scale))
        transition = self.transition_head(representation).squeeze(-1).masked_fill(~masks, 0)
        auxiliary = (self.transition_head(self.transition(scaled_gradient(descriptors, .25))).squeeze(-1)
                     if self.training else transition)
        contribution = self.contribution_head(representation) if variant == 'h65' else None
        rates = transition if contribution is None else transition + self.utility_fusion(contribution).squeeze(-1)
        return dict(action_logits=action, aux_transition=auxiliary, transition_logits=transition,
                    representation=representation, hidden=scaled_gradient(hidden, adapt_scale),
                    contribution_logits=contribution, rate_logits=rates.masked_fill(~masks, 0))

    def condition(self, output, selection, masks):
        """Explicit nearest opposite-membership exchange within each 16-slot cell."""
        hidden = output['hidden']
        b, t, _ = hidden.shape
        member = torch.zeros_like(masks, dtype=torch.float32).scatter_add(1, selection.indices, selection.valid.float()) > 0
        partner = torch.arange(t, device=masks.device).expand(b, -1).clone()
        feasible = torch.zeros_like(masks)
        for row in range(b):
            for start in range(0, t, 16):
                candidates = torch.arange(start, min(start + 16, t), device=masks.device)
                valid = masks[row, candidates]
                for membership in (False, True):
                    src = candidates[valid & (member[row, candidates] == membership)]
                    dst = candidates[valid & (member[row, candidates] != membership)]
                    if len(src) and len(dst):
                        partner[row, src] = dst[(src[:, None] - dst[None]).abs().argmin(-1)]
                        feasible[row, src] = True
        positions = torch.arange(t, device=masks.device).float().expand(b, -1)
        left_gaps, right_gaps = [], []
        for indices, valid in zip(selection.indices, selection.valid):
            kept = indices[valid].float()
            query = positions[0]
            index = torch.searchsorted(kept, query)
            left = kept[(index - 1).clamp(0, len(kept) - 1)]
            right = kept[index.clamp(0, len(kept) - 1)]
            left_gaps.append((query - left).abs() / t)
            right_gaps.append((right - query).abs() / t)
        selected_mean = (hidden * member[..., None]).sum(1) / member.sum(1, keepdim=True).clamp_min(1)
        partner_hidden = hidden.gather(1, partner[..., None].expand(-1, -1, hidden.shape[-1]))
        scalars = torch.stack((member.float(), (partner - positions) / t, torch.stack(left_gaps),
                               torch.stack(right_gaps), output['action_logits'].detach().sigmoid(),
                               (selection.valid.sum(-1) / masks.sum(-1)).unsqueeze(1).expand(-1, t)), -1)
        context = torch.cat((hidden, partner_hidden, selected_mean[:, None].expand(-1, t, -1),
                             output['representation'], scalars), -1)
        utility = self.conditional(context.float()).masked_fill(~feasible[..., None], 0)
        return dict(utility=utility, member=member, partner=partner, feasible=feasible)
