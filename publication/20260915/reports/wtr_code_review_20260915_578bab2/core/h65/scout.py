"""Low-resolution ASFormer scout and H65 relational transition/utility heads."""
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F


def load_asformer():
    root = Path(__file__).resolve().parents[1] / "references/ASFormer"
    sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location("h65_asformer_reference", root / "model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(root))
    return module


def transition_descriptors(logits, hidden, masks):
    probability = logits.float().sigmoid().clamp(1e-7, 1 - 1e-7)
    entropy = -(probability * probability.log() + (1 - probability) * (1 - probability).log())
    dl = torch.diff(logits, prepend=logits[:, :1], dim=1)
    de = torch.diff(entropy, prepend=entropy[:, :1], dim=1)
    dh = torch.diff(hidden, prepend=hidden[:, :1], dim=1)
    cosine = F.pad(1 - F.cosine_similarity(hidden[:, 1:], hidden[:, :-1], dim=-1), (1, 0))
    features = torch.cat((dl[..., None], dl.abs()[..., None], de[..., None], de.abs()[..., None],
                          dh, dh.abs(), cosine[..., None]), -1)
    valid = masks.clone()
    valid[:, 0] = False
    valid[:, 1:] &= masks[:, :-1]
    return features.masked_fill(~valid[..., None], 0)


class H65Scout(nn.Module):
    def __init__(self, hidden_dim=96, scorer_dim=64, spatial_size=64):
        super().__init__()
        self.spatial_size = spatial_size
        self.reference = load_asformer()
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.temporal = self.reference.MyTransformer(1, 2, 2, 2, hidden_dim, hidden_dim, 2, 0.0)
        self.transition = nn.Sequential(nn.LayerNorm(2 * hidden_dim + 5),
                                        nn.Linear(2 * hidden_dim + 5, scorer_dim), nn.GELU())
        self.transition_head = nn.Linear(scorer_dim, 1)
        self.contribution_head = nn.Linear(scorer_dim, 2)
        nn.init.normal_(self.contribution_head.weight, std=1e-3)
        nn.init.zeros_(self.contribution_head.bias)
        self.utility_fusion = nn.Linear(2, 1)
        nn.init.zeros_(self.utility_fusion.weight)
        nn.init.zeros_(self.utility_fusion.bias)

    def forward(self, frames, masks, adapt_scale=1.0, use_contribution=True):
        b, _, c, t, h, w = frames.shape
        # The supported OpenTAD raw frame pipeline supplies RGB in [0,255].
        flat = frames[:, 0].permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float() / 255
        small = F.interpolate(flat, size=(self.spatial_size, self.spatial_size), mode="bilinear", align_corners=False)
        stem = self.spatial_stem(small).flatten(1).reshape(b, t, -1).transpose(1, 2)
        stem = stem * masks[:, None]
        self.reference.device = frames.device
        for module in self.temporal.modules():
            if hasattr(module, "window_mask"):
                module.window_mask = module.window_mask.to(frames.device)
        logits_rows, hidden_rows = [], []
        for i in range(b):
            mask = masks[i:i+1, None].float()
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(frames.device) if frames.is_cuda else None
            out, hidden = self.temporal.encoder(stem[i:i+1], mask)
            action, decoder_hidden = out, hidden
            for decoder in self.temporal.decoders:
                action, decoder_hidden = decoder(action.softmax(1) * mask, decoder_hidden * mask, mask)
            if self.training and adapt_scale:
                # H65 detector feedback reaches the ASFormer encoder, not its RGB stem.
                devices = [frames.device.index] if frames.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state(cuda_rng, frames.device)
                    _, replay = self.temporal.encoder(stem[i:i+1].detach(), mask)
                policy_hidden = hidden.detach() + adapt_scale * (replay - replay.detach())
            else:
                policy_hidden = hidden.detach()
            logits_rows.append(action[:, 1] - action[:, 0])
            hidden_rows.append(policy_hidden.transpose(1, 2))
        action_logits = torch.cat(logits_rows)
        hidden = torch.cat(hidden_rows)
        descriptors = transition_descriptors(action_logits.detach(), hidden, masks)
        representation = self.transition(descriptors)
        transition = self.transition_head(representation).squeeze(-1).masked_fill(~masks, 0)
        utility = self.contribution_head(representation).masked_fill(~masks[..., None], 0) if use_contribution else None
        rate_logits = transition if utility is None else transition + self.utility_fusion(utility).squeeze(-1)
        return dict(action_logits=action_logits, transition_logits=transition,
                    contribution_logits=utility, rate_logits=rate_logits.masked_fill(~masks, 0))


def scout_losses(output, masks, segments):
    """Train-only action and Gaussian boundary supervision; no GT in forward."""
    times = torch.arange(masks.shape[1], device=masks.device).float()
    action, boundary = [], []
    for boxes in segments:
        boxes = boxes.to(times)
        if boxes.numel():
            inside = ((times[:, None] + 0.5 >= boxes[:, 0]) & (times[:, None] + 0.5 <= boxes[:, 1])).any(-1)
            distance = (times[:, None] - boxes.flatten()[None]).abs()
            peaks = (torch.exp(-0.5 * (distance / 2).square()) * (distance <= 4)).amax(-1)
        else:
            inside, peaks = torch.zeros_like(times).bool(), torch.zeros_like(times)
        action.append(inside.float())
        boundary.append(peaks)
    action, boundary = torch.stack(action), torch.stack(boundary) * masks
    action_loss = F.binary_cross_entropy_with_logits(output["action_logits"][masks], action[masks])
    boundary_loss = F.binary_cross_entropy_with_logits(output["transition_logits"][masks], boundary[masks])
    target = boundary / boundary.sum(-1, keepdim=True).clamp_min(1e-8)
    log_probability = (output["transition_logits"] / 0.7).masked_fill(~masks, -1e4).log_softmax(-1)
    distribution_loss = -(target * log_probability).sum(-1).mean()
    return dict(loss_actionness=action_loss, loss_transition=distribution_loss, loss_boundary=boundary_loss)
