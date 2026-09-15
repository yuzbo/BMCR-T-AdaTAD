"""Frozen dense reference teacher for frame training."""
from __future__ import annotations
import torch
from torch import nn, Tensor


class OriginalTeacher(nn.Module):
    def __init__(self, model_cfg, checkpoint=None):
        super().__init__()
        # Import lazily so objective utilities remain usable without the
        # optional OpenTAD runtime (for unit tests and feature-only training).
        from h65.ds3.model import DenseTeacher
        self.model = DenseTeacher(model_cfg, checkpoint, scope="global")
        self.model.requires_grad_(False)
        self.model.eval()

    @property
    def vit(self):
        return self.model.vit

    def train(self, mode=True):
        super().train(False); self.model.eval(); return self

    @torch.no_grad()
    def dense_native(self, inputs):
        return self.model.dense_native(inputs)[0]

    def loss(self, native, data):
        args = _data_args(data)
        return self.model.loss(native, *args)

    def predictions(self, native, masks, metas=None):
        return self.model.predictions(native, masks, metas)

    def raw_head(self,native,masks):
        captured={};classes=[];offsets=[];detector=self.model.detector;head=detector.rpn_head
        module=detector.neck if detector.with_neck else detector.projection
        hooks=[head.cls_head.register_forward_hook(lambda m,a,o:classes.append(o)),
               module.register_forward_hook(lambda m,a,o:captured.update(pyramid=o))]
        hooks += [scale.register_forward_hook(lambda m,a,o:offsets.append(torch.relu(o))) for scale in head.scale]
        try:self.model.predictions(native,masks)
        finally:
            for hook in hooks:hook.remove()
        _,mask_list=captured['pyramid']
        return dict(logits=torch.cat(classes,-1).transpose(1,2),offsets=torch.cat(offsets,-1).transpose(1,2),
                    valid=torch.cat(mask_list,-1))

    def post_processing(self, *args, **kwargs):
        return self.model.detector.post_processing(*args, **kwargs)


def _data_args(data):
    if isinstance(data, dict):
        return (data.get("masks", data.get("mask")),
                data.get("metas", data.get("metadata", [{}])),
                data.get("gt_segments", data.get("segments")),
                data.get("gt_labels", data.get("labels")))
    return tuple(data)
