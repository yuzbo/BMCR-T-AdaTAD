"""External H65 temporal frontend around original AdaTAD components.

All upstream Python files remain unchanged. Sparse VideoMAE/Adapter execution
uses the official configurable total_frames; original detector lattice,
assignment, losses and post-processing are retained through time restoration.
"""
import copy
from pathlib import Path

import torch
from torch import nn

import opentad.datasets  # register upstream frame transforms
from opentad.models.builder import build_detector
from .scout import H65Scout, scout_losses
from .transport import sample_rates, gather_with_transport, restore_tubelets, interpolate_irregular


class H65AdaTAD(nn.Module):
    def __init__(self, official_model_cfg, checkpoint, budget=384):
        super().__init__()
        if budget % 16:
            raise ValueError("VideoMAE execution requires a multiple of 16 frames")
        self.budget = budget
        config = copy.deepcopy(official_model_cfg)
        config.backbone.backbone.total_frames = budget
        # Input-gradient contribution targets require autograd.grad, which the
        # official reentrant activation-checkpoint path does not support.
        config.backbone.backbone.with_cp = False
        config.backbone.custom.pretrain = None
        config.backbone.custom.pre_processing_pipeline[0].t1 = budget // 16
        post = config.backbone.custom.post_processing_pipeline
        post[1].t1 = budget // 16
        config.backbone.custom.post_processing_pipeline = post[:2]
        core = build_detector(config)
        payload = torch.load(checkpoint, map_location="cpu")
        key = "state_dict_ema" if "state_dict_ema" in payload else "state_dict"
        state = {name.removeprefix("module."): value for name, value in payload[key].items()}
        core.load_state_dict(state, strict=True)
        self.backbone = core.backbone
        del core.backbone
        self.detector = core
        self.scout = H65Scout()
        self.initialization = dict(checkpoint=str(checkpoint), state_key=key,
                                   checkpoint_epoch=payload.get("epoch"), strict_load=True)

    def features(self, frames, masks, selection, bridge_weight=0.25):
        selected = gather_with_transport(frames, selection, bridge_weight)
        tubelets = self.backbone(selected)
        return restore_tubelets(tubelets, selection, masks), selected

    def detection_losses(self, features, masks, metas, gt_segments, gt_labels):
        # PyTorch2.0.1 CUDA nearest1d mask resize has no BF16 kernel.
        # Keep original detector operators in FP32 without patching upstream.
        with torch.autocast(features.device.type, enabled=False):
            return self.detector.forward_train(features.float(), masks, metas, gt_segments, gt_labels)

    def forward_test(self, frames, masks, metas=None, alpha=1.0, use_contribution=True):
        output = self.scout(frames, masks, use_contribution=use_contribution)
        selection = sample_rates(output["rate_logits"], masks, self.budget, alpha=alpha)
        features, _ = self.features(frames, masks, selection, bridge_weight=0)
        with torch.autocast(features.device.type, enabled=False):
            predictions = self.detector.forward_test(features.float(), masks, metas)
        return predictions, selection

    def contribution_targets(self, frames, masks, metas, gt_segments, gt_labels):
        """Uniform companion, detached |input*gradient| targets for cls and reg.

This is a first-order sensitivity target, not a measured deletion benefit.
It adds one uniform forward and two input-gradient calculations in training.
"""
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            uniform = sample_rates(frames.new_zeros(masks.shape, dtype=torch.float32), masks,
                                   self.budget, alpha=0)
            selected = gather_with_transport(frames, uniform, 0).detach().requires_grad_(True)
            tubelets = self.backbone(selected)
            features = restore_tubelets(tubelets, uniform, masks)
            losses = self.detection_losses(features, masks, metas, gt_segments, gt_labels)
            targets = []
            for kind in ("cls", "reg"):
                objective = sum(value for name, value in losses.items() if "loss" in name and kind in name)
                gradient = torch.autograd.grad(objective, selected, retain_graph=True, create_graph=False)[0]
                contribution = (selected.detach() * gradient.detach()).abs().mean((1, 2, 4, 5))
                dense = []
                query = torch.arange(masks.shape[1], device=frames.device, dtype=torch.float32)
                for row, indices, valid in zip(contribution, uniform.indices, uniform.valid):
                    dense.append(interpolate_irregular(row[valid][None], indices[valid].float(), query)[0])
                field = torch.stack(dense) * masks
                targets.append(field / field.sum(-1, keepdim=True).clamp_min(1e-8))
            return torch.stack(targets, -1).detach(), {k: float(v.detach()) for k, v in losses.items()}
        finally:
            for module, training in modes:
                module.training = training

    def forward_train(self, frames, masks, metas, gt_segments, gt_labels, *,
                      alpha=1.0, bridge_weight=0.25, contribution_weight=1.0,
                      adapt_scale=1.0):
        use_contribution = contribution_weight > 0
        output = self.scout(frames, masks, adapt_scale=adapt_scale, use_contribution=use_contribution)
        selection = sample_rates(output["rate_logits"], masks, self.budget, alpha=alpha)
        features, _ = self.features(frames, masks, selection, bridge_weight)
        losses = self.detection_losses(features, masks, metas, gt_segments, gt_labels)
        auxiliary = scout_losses(output, masks, gt_segments)
        losses.update(loss_actionness=0.25 * auxiliary["loss_actionness"],
                      loss_transition=0.10 * auxiliary["loss_transition"],
                      loss_boundary=0.25 * auxiliary["loss_boundary"])
        teacher_info = None
        if use_contribution:
            targets, teacher_info = self.contribution_targets(frames, masks, metas, gt_segments, gt_labels)
            log_probability = output["contribution_logits"].float().masked_fill(~masks[..., None], -1e4).log_softmax(1)
            losses["loss_contribution"] = -contribution_weight * (targets * log_probability).sum(1).mean()
        losses["cost"] = sum(value for name, value in losses.items() if name != "cost")
        return losses, selection, teacher_info
