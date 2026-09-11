"""New recognition-pretrained H65 around unchanged upstream AdaTAD modules."""
import copy
import torch
from torch import nn
import torch.nn.functional as F
import opentad.datasets
from opentad.models.builder import build_detector
from h65.transport import sample_rates, gather_with_transport, interpolate_irregular, Selection
from .scout import FormalScout
from .geometry import maps_for, mix_rows
from .objectives import auxiliary_losses, distribution_loss


class FormalH65(nn.Module):
    def __init__(self, official_cfg, pretrain=None, variant='h65', budget=384):
        super().__init__()
        self.variant, self.budget = variant, budget
        cfg = copy.deepcopy(official_cfg)
        cfg.backbone.backbone.total_frames = budget
        cfg.backbone.backbone.with_cp = False
        cfg.backbone.custom.pretrain = pretrain
        cfg.backbone.custom.temporal_checkpointing = True
        cfg.backbone.custom.temporal_checkpointing_chunk_num = 1
        cfg.backbone.custom.temporal_checkpointing_chunk_dim = 0
        cfg.backbone.custom.pre_processing_pipeline[0].t1 = budget // 16
        cfg.backbone.custom.post_processing_pipeline[1].t1 = budget // 16
        cfg.backbone.custom.post_processing_pipeline[2].size = budget
        cfg.projection.max_seq_len = budget
        core = build_detector(cfg)
        # Upstream initializes this buffer as an integer, then rebinds it to a
        # float on the first train call. Start it as float so EMA/resume retain
        # its fractional running normalization rather than integer-casting it.
        core.rpn_head.loss_normalizer = core.rpn_head.loss_normalizer.float()
        self.backbone = core.backbone
        # Upstream freezes block weights lazily inside its first forward and
        # gives remaining non-Adapter parameters lr0. Set the same trainable
        # parameter set explicitly before creating our optimizer.
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_('adapter' in name)
        del core.backbone
        self.detector = core
        self.scout = FormalScout()
        self.register_buffer('utility_scales', torch.ones(2))

    def route(self, inputs, masks, weights):
        output = self.scout(inputs, masks, weights['adapt'], self.variant)
        provisional = sample_rates(output['rate_logits'], masks, self.budget, alpha=weights['alpha'])
        conditional = None
        selection = provisional
        if self.variant == 'bmcr':
            conditional = self.scout.condition(output, provisional, masks)
            # Predicted values are standardized by train-only audit component scales.
            # Preregistered unit gain and equal component weights; one refinement.
            correction = conditional['utility'].mean(-1)
            scores = output['rate_logits'] + correction
            selection = sample_rates(scores, masks, self.budget, alpha=weights['alpha'])
        return output, provisional, selection, conditional

    def encode(self, inputs, selection, bridge=0., row_scale=None, need_input_grad=False):
        if row_scale is not None:
            values = {key: getattr(selection, key) for key in Selection.__dataclass_fields__}
            coordinate = selection.continuous
            values['continuous'] = coordinate.detach() + row_scale[:, None] * (coordinate - coordinate.detach())
            selection = Selection(**values)
        selected = gather_with_transport(inputs, selection, bridge)
        selected = selected * selection.valid[:, None, None, :, None, None]
        if need_input_grad and not selected.requires_grad:
            selected.requires_grad_(True)
        features = self.backbone(selected)
        return features * selection.valid[:, None], selected

    def train_batch(self, inputs, masks, metas, gt_segments, gt_labels, weights, teacher=None):
        output, provisional, learned, conditional = self.route(inputs, masks, weights)
        batch = len(inputs)
        teacher_rows = torch.zeros(batch, device=inputs.device, dtype=torch.bool)
        if batch < 2:
            raise ValueError('formal joint protocol requires batch2 for a uniform companion')
        teacher_rows[torch.randperm(batch, device=inputs.device)[:batch // 2]] = True
        uniform = sample_rates(output['rate_logits'], masks, self.budget, alpha=0.)
        selection = mix_rows(learned, uniform, teacher_rows)
        row_scale = (~teacher_rows).float() * batch / int((~teacher_rows).sum())
        contribution = self.variant == 'h65' and weights['contribution'] > 0
        features, selected = self.encode(inputs, selection, weights['bridge'], row_scale, contribution)
        geometry = maps_for(selection, masks)
        mapped_gt = [mapping.to_rank(boxes) for mapping, boxes in zip(geometry, gt_segments)]
        with torch.autocast(inputs.device.type, enabled=False):
            losses = self.detector.forward_train(features.float(), selection.valid, metas, mapped_gt, gt_labels)
        losses.pop('cost')
        losses.update(auxiliary_losses(output, selection, masks, gt_segments, weights))
        diagnostics = {}
        if contribution:
            for channel, kind in enumerate(('cls', 'reg')):
                gradient = torch.autograd.grad(losses[kind + '_loss'], selected, retain_graph=True, create_graph=False)[0]
                sensitivity = (selected.detach() * gradient.detach()).abs().mean((1, 2, 4, 5))
                query = torch.arange(masks.shape[1], device=inputs.device).float()
                dense = torch.stack([interpolate_irregular(value[valid][None], position[valid].float(), query)[0]
                                     for value, position, valid in zip(sensitivity, selection.indices, selection.valid)]) * masks
                losses['loss_contribution_' + kind] = weights['contribution'] * distribution_loss(
                    output['contribution_logits'][..., channel], dense, masks, teacher_rows)
                diagnostics['teacher_' + kind + '_mass'] = dense[teacher_rows].sum(-1).tolist()
        if self.variant == 'bmcr' and teacher is not None and weights['contribution'] > 0:
            from .utility import counterfactual_targets
            observations = counterfactual_targets(teacher, inputs, masks, metas, gt_segments, gt_labels,
                                                  provisional, conditional, allowed_rows=~teacher_rows)
            if observations:
                prediction = torch.stack([conditional['utility'][o['row'], o['candidate']] for o in observations])
                target = prediction.new_tensor([o['target'] for o in observations]) / self.utility_scales
                losses['loss_counterfactual'] = .1 * weights['contribution'] * F.smooth_l1_loss(prediction.float(), target)
            diagnostics['counterfactuals'] = observations
        losses['cost'] = sum(losses.values())
        diagnostics['teacher_rows'] = teacher_rows.tolist()
        diagnostics['unique_selected'] = selection.valid.sum(-1).tolist()
        return losses, diagnostics

    def predictions(self, inputs, masks, metas=None):
        weights = dict(alpha=1., adapt=1.)
        _, _, selection, _ = self.route(inputs, masks, weights)
        features, _ = self.encode(inputs, selection)
        with torch.autocast(inputs.device.type, enabled=False):
            proposals, scores = self.detector.forward_test(features.float(), selection.valid, metas)
        geometry = maps_for(selection, masks)
        return ([mapping.to_true(p) for mapping, p in zip(geometry, proposals)], scores), selection

    def forward(self, inputs, masks, metas, return_loss=False, post_cfg=None, ext_cls=None, **kwargs):
        if return_loss:
            return self.train_batch(inputs, masks, metas, **kwargs)
        predictions, _ = self.predictions(inputs, masks, metas)
        return self.detector.post_processing(predictions, metas, post_cfg, ext_cls)

    def raw_route(self, inputs, masks, selection):
        """Expose original head tensors through a read-only hook, without duplicating head code."""
        features, _ = self.encode(inputs, selection)
        captured = {}
        head = self.detector.rpn_head
        cls_values = []
        feature_module = self.detector.neck if self.detector.with_neck else self.detector.projection
        hooks = [head.cls_head.register_forward_hook(lambda module, args, result: cls_values.append(result)),
                 feature_module.register_forward_hook(lambda module, args, result: captured.update(pyramid=result))]
        try:
            with torch.autocast(inputs.device.type, enabled=False):
                proposals, scores = self.detector.forward_test(features.float(), selection.valid)
        finally:
            for hook in hooks:
                hook.remove()
        logits = torch.cat(cls_values, -1).transpose(1, 2)
        pyramid, mask_list = captured['pyramid']
        points = head.prior_generator(pyramid)
        valid = torch.cat(mask_list, -1)
        if valid.shape != logits.shape[:2]:
            raise RuntimeError('captured prior/mask geometry does not match original head')
        maps = maps_for(selection, masks)
        return dict(logits=logits, points=points, valid=valid,
                    proposals=[m.to_true(p) for m, p in zip(maps, proposals)], scores=scores, maps=maps)
