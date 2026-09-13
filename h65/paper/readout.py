"""Independent trainable task readout; external teacher is never modified."""
import copy
from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F


class TaskReadout(nn.Module):
    def __init__(self, model_cfg, family='point', checkpoint=None, trainable=True):
        super().__init__()
        from opentad.models.builder import build_detector
        cfg=copy.deepcopy(model_cfg); cfg.pop('backbone',None)
        self.detector=build_detector(cfg); self.family=family
        if hasattr(self.detector,'rpn_head'):
            self.detector.rpn_head.loss_normalizer=self.detector.rpn_head.loss_normalizer.float()
        self.initialization=dict(source=str(checkpoint) if checkpoint else 'new task head')
        if checkpoint:
            payload=torch.load(checkpoint,map_location='cpu')
            state=payload.get('state_dict_ema',payload.get('state_dict',payload.get('model',payload)))
            state={k.removeprefix('module.'):v for k,v in state.items()}
            expected=self.detector.state_dict()
            selected={k:state[k] for k in expected if k in state}
            missing=set(expected)-set(selected)
            if missing: raise ValueError('Missing task-head weights: '+str(sorted(missing)))
            self.detector.load_state_dict(selected,strict=True)
        self.requires_grad_(trainable); self.trainable=trainable
        if not trainable: self.eval()

    def train(self, mode=True):
        return super().train(mode and getattr(self,'trainable',True))

    def features(self,native,masks):
        return F.interpolate(native.float(),size=masks.shape[-1],mode='linear',align_corners=False)

    def loss(self,native,data,update_normalizer=False):
        if self.family=='tadtr':
            import os,tempfile
            from pathlib import Path
            import torch.distributed as dist
            if not dist.is_initialized():
                descriptor,path=tempfile.mkstemp(prefix='tadtr_single_rank_');os.close(descriptor)
                dist.init_process_group('nccl' if native.is_cuda else 'gloo',init_method=Path(path).resolve().as_uri(),rank=0,world_size=1)
        head=getattr(self.detector,'rpn_head',None)
        before=getattr(head,'loss_normalizer',None)
        with torch.autocast(native.device.type,enabled=False):
            losses=self.detector.forward_train(self.features(native,data['masks']),data['masks'],data.get('metas'),
                                               data['gt_segments'],data['gt_labels'])
        if before is not None:
            if self.training: self.next_normalizer=head.loss_normalizer
            if not update_normalizer: head.loss_normalizer=before
        return losses

    def commit_normalizer(self):
        head=getattr(self.detector,'rpn_head',None)
        if head is not None and hasattr(self,'next_normalizer'):
            head.loss_normalizer=self.next_normalizer.detach()

    def components(self,losses):
        if self.family=='point': return torch.stack((losses['cls_loss'],losses['reg_loss']))
        cls=losses['loss_class']
        loc=losses['loss_bbox']+losses['loss_iou']
        return torch.stack((cls,loc))

    def predictions(self,native,masks,metas=None):
        with torch.autocast(native.device.type,enabled=False):
            return self.detector.forward_test(self.features(native,masks),masks,metas)

    def post_processing(self,predictions,metas,post_cfg,ext_cls):
        return self.detector.post_processing(predictions,metas,post_cfg,ext_cls)

    def raw_head(self,native,masks):
        if self.family!='point': raise ValueError('Pointwise KD is not valid for a query head')
        captured={};classes=[];offsets=[];head=self.detector.rpn_head
        module=self.detector.neck if self.detector.with_neck else self.detector.projection
        hooks=[head.cls_head.register_forward_hook(lambda m,a,o:classes.append(o)),
               module.register_forward_hook(lambda m,a,o:captured.update(pyramid=o))]
        hooks += [scale.register_forward_hook(lambda m,a,o:offsets.append(torch.relu(o))) for scale in head.scale]
        try: self.predictions(native,masks)
        finally:
            for hook in hooks: hook.remove()
        _,valid=captured['pyramid']
        return dict(logits=torch.cat(classes,-1).transpose(1,2),offsets=torch.cat(offsets,-1).transpose(1,2),
                    valid=torch.cat(valid,-1))
