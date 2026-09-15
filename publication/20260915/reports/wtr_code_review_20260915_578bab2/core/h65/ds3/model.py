"""DS3-L with frozen dense TAD weights and compact local-TIA clip execution."""
import copy
import torch
from torch import nn
import torch.nn.functional as F
import opentad.datasets
from opentad.models.builder import build_detector
from .auxiliary import Auxiliaries, preview_images
from .routes import POLICIES, route_depths, native_geometry, interpolate_native


class DenseTeacher(nn.Module):
    def __init__(self, model_cfg, checkpoint=None, scope='local'):
        super().__init__()
        if scope not in ('local','global'):raise ValueError(scope)
        cfg=copy.deepcopy(model_cfg)
        cfg.backbone.custom.pretrain=None
        cfg.backbone.custom.temporal_checkpointing=False
        cfg.backbone.backbone.with_cp=False
        cfg.backbone.backbone.total_frames=16 if scope=='local' else 768
        core=build_detector(cfg)
        core.rpn_head.loss_normalizer=core.rpn_head.loss_normalizer.float()
        self.provenance=dict(scope=scope,original_tia_size=384,runtime_tia_size=8 if scope=='local' else 384)
        if checkpoint is not None:
            payload=torch.load(checkpoint,map_location='cpu')
            key='state_dict_ema' if 'state_dict_ema' in payload else 'state_dict'
            state={name.removeprefix('module.'):value for name,value in payload[key].items()}
            core.load_state_dict(state,strict=True)
            self.provenance.update(checkpoint=str(checkpoint),state_key=key,source_epoch=payload.get('epoch'),strict=True)
        self.backbone=core.backbone
        del core.backbone
        self.detector=core
        self.scope=scope
        self.requires_grad_(False);self.eval()

    @property
    def vit(self):return self.backbone.model.backbone

    def train(self, mode=True):
        return super().train(False)

    def prepare_clips(self, inputs):
        frames,_=self.backbone.model.data_preprocessor.preprocess(self.backbone.tensor_to_list(inputs),None,False)
        frames=self.backbone.pre_processing_pipeline(dict(frames=frames))['frames']
        return frames.flatten(0,1).contiguous()

    def pool_maps(self, maps):
        return self.backbone.post_processing_pipeline.transforms[0](dict(feats=maps[:,None]))['feats']

    def grid(self, clip_features):
        return self.backbone.post_processing_pipeline.transforms[1](dict(feats=clip_features))['feats']

    def to_detector(self, native):
        if native.shape[-1]!=384:raise ValueError('detector interface requires native384')
        return self.backbone.post_processing_pipeline.transforms[2](dict(feats=native))['feats'].float()

    @torch.no_grad()
    def dense_native(self, inputs, capture=False):
        saved=dict(mlp_inputs={},mlp_outputs={},attention_clips=[0]*12,mlp_tokens=[0]*12);hooks=[]
        def count_attention(index):
            def hook(module,args):saved['attention_clips'][index]+=args[0].shape[0]
            return hook
        def count_mlp(index):
            def hook(module,args):saved['mlp_tokens'][index]+=args[0].numel()//args[0].shape[-1]
            return hook
        def store_input(index):
            def hook(module,args):saved['mlp_inputs'][index]=args[0].detach()
            return hook
        def store_output(index):
            def hook(module,args,result):saved['mlp_outputs'][index]=result.detach()
            return hook
        def store_exit(module,args,result):saved['exit8_tokens']=result.detach()
        if capture:
            hooks.append(self.vit.blocks[7].register_forward_hook(store_exit))
            for index in range(12):
                hooks.append(self.vit.blocks[index].attn.register_forward_pre_hook(count_attention(index)))
                hooks.append(self.vit.blocks[index].mlp.register_forward_pre_hook(count_mlp(index)))
            for index in range(8,12):
                hooks.append(self.vit.blocks[index].mlp.register_forward_pre_hook(store_input(index)))
                hooks.append(self.vit.blocks[index].mlp.register_forward_hook(store_output(index)))
        try:
            clips=self.prepare_clips(inputs)
            maps=self.vit(clips)
            native=self.grid(self.pool_maps(maps))
            if capture:
                h,w=maps.shape[-2:]
                saved['exit8']=self.pool_tokens(saved.pop('exit8_tokens'),h,w)
                saved.update(spatial_shape=(h,w),dense_clips=clips.shape[0],dense_layers=12)
            return native,saved
        finally:
            for hook in hooks:hook.remove()

    def pool_tokens(self, tokens, h, w):
        x=self.vit.norm(tokens)
        maps=x.reshape(len(x),8,h,w,self.vit.embed_dims).permute(0,4,1,2,3)
        return self.pool_maps(maps)

    def predictions(self, native, masks, metas=None):
        features=self.to_detector(native)
        with torch.autocast(features.device.type,enabled=False):
            return self.detector.forward_test(features,masks,metas)

    def loss(self, native, masks, metas, gt_segments, gt_labels):
        features=self.to_detector(native)
        with torch.autocast(features.device.type,enabled=False):
            return self.detector.forward_train(features,masks,metas,gt_segments,gt_labels)


class ClipEngine:
    """Heavy layers execute only their current packed clip/token batch."""
    def __init__(self, teacher):
        if teacher.scope!='local':raise ValueError('compact clips require localTIA8; global teacher is a separate reference')
        self.teacher=teacher

    def embed(self, clips):
        vit=self.teacher.vit;h,w=clips.shape[-2]//vit.patch_size,clips.shape[-1]//vit.patch_size
        x=vit.patch_embed(clips)[0]
        if (h,w)!=vit.grid_size:
            pos=vit.pos_embed.reshape(-1,*vit.grid_size,vit.embed_dims).permute(0,3,1,2)
            pos=F.interpolate(pos,size=(h,w),mode='bicubic',align_corners=False)
            pos=pos.permute(0,2,3,1).flatten(1,2).reshape(1,-1,vit.embed_dims)
        else:pos=vit.pos_embed
        return vit.pos_drop(x+pos),h,w

    def block(self, x, index, h, w, ratio, aux, trace):
        block=self.teacher.vit.blocks[index]
        trace['attention_clips'][index]=len(x)
        trace['attention_tokens'][index]=x.shape[0]*x.shape[1]
        if ratio==1 or index<8:
            trace['heavy_mlp_tokens'][index]=x.shape[0]*x.shape[1]
            return block(x,h,w)
        x=x+block.drop_path(block.attn(block.norm1(x)))
        normalized=block.norm2(x)
        replacement=aux.surrogates[str(index)](normalized)
        score=aux.spatial_scores[str(index)](normalized).squeeze(-1).reshape(len(x),8,h*w)
        count=max(1,round(h*w*ratio))
        selected=score.argsort(dim=-1,descending=True,stable=True)[...,:count].sort(-1).values
        selected=selected+torch.arange(8,device=x.device)[None,:,None]*(h*w)
        selected=selected.flatten(1)
        packed=normalized.gather(1,selected[...,None].expand(-1,-1,x.shape[-1]))
        heavy=block.mlp(packed)
        trace['heavy_mlp_tokens'][index]=packed.shape[0]*packed.shape[1]
        residual=replacement.scatter(1,selected[...,None].expand_as(heavy),heavy)
        x=x+block.drop_path(residual)
        return block.adapter(x,h,w) if block.use_adapter else x

    def execute(self, clips, depths, ratio=1., aux=None):
        flat=depths.flatten();prefix=(flat>=8).nonzero().flatten()
        full=(flat[prefix]==12).nonzero().flatten()
        trace=dict(attention_clips=[0]*12,attention_tokens=[0]*12,heavy_mlp_tokens=[0]*12,
                   heavy_input_clips=len(prefix),spatial_ratio=ratio)
        if not len(prefix):return dict(prefix_ids=prefix,full_ids=prefix,at8=None,at12=None,trace=trace)
        x,h,w=self.embed(clips.index_select(0,prefix))
        for i in range(8):x=self.block(x,i,h,w,1.,aux,trace)
        at8=self.teacher.pool_tokens(x,h,w)
        at12=None
        if len(full):
            x=x.index_select(0,full)
            for i in range(8,12):x=self.block(x,i,h,w,ratio,aux,trace)
            at12=self.teacher.pool_tokens(x,h,w)
        return dict(prefix_ids=prefix,full_ids=prefix[full],at8=at8,at12=at12,trace=trace)


class DS3(nn.Module):
    def __init__(self, model_cfg, checkpoint, with_aux=True, mobile_pretrained=None, scope='local'):
        super().__init__()
        self.teacher=DenseTeacher(model_cfg,checkpoint,scope)
        self.aux=Auxiliaries(self.teacher.vit.embed_dims,mobile_pretrained) if with_aux else None
        self.engine=ClipEngine(self.teacher) if scope=='local' else None
        self.policy='D768G' if scope=='global' else 'T24A' if with_aux else 'D768L'

    def train(self, mode=True):
        super().train(mode);self.teacher.eval();return self

    def native(self, inputs, masks, metas, policy=None, preview=None):
        policy=policy or self.policy
        spec=POLICIES[policy] if isinstance(policy,str) else policy
        b=len(inputs);channels=self.teacher.vit.embed_dims
        if inputs.shape[1:4]!=(1,3,768):raise ValueError('DS3 production input is[B,1,3,768,H,W]')
        valid,frames,centers=native_geometry(masks,metas)
        if spec.kind in ('global','local'):
            required='global' if spec.kind=='global' else 'local'
            if self.teacher.scope!=required:raise ValueError('dense reference scope mismatch')
            features,_=self.teacher.dense_native(inputs)
            depth=torch.full((b,48),12,device=masks.device,dtype=torch.long)
            tokens=8*(inputs.shape[-2]//16)*(inputs.shape[-1]//16)
            trace=dict(attention_clips=[b*48]*12,attention_tokens=[b*48*tokens]*12,
                       heavy_mlp_tokens=[b*48*tokens]*12,heavy_input_clips=b*48,spatial_ratio=1.)
        else:
            if self.engine is None:raise ValueError('sparse deployment cannot use globalTIA')
            if spec.kind=='aux':
                if self.aux is None:raise ValueError('D1 deployment needs trained auxiliary modules')
                if spec.clips8==48 and spec.clips12 in (0,48):
                    # Every native position is replaced by a heavy or exit8
                    # observation; no preview or clip decision is needed.
                    features=inputs.new_zeros((b,channels,384),dtype=torch.float32)
                    logits=inputs.new_zeros((b,48,2))
                else:
                    features,logits=self.aux.preview(preview if preview is not None else preview_images(inputs))
                    features=features.float()
            else:
                features=inputs.new_zeros((b,channels,384),dtype=torch.float32)
                logits=inputs.new_zeros((b,48,2),dtype=torch.float32)
            seeds=[]
            for row,meta in enumerate(metas):
                suffix=str(meta.get('video_name','')).rsplit('_',1)[-1]
                seeds.append(3407+(int(suffix) if suffix.isdigit() else row)*1000003+int(meta.get('window_start_frame',0)))
            depth=route_depths(logits,masks,spec,sample_seeds=seeds)
            if bool((depth>0).any()):
                execution=self.engine.execute(self.teacher.prepare_clips(inputs),depth,spec.spatial_ratio,self.aux)
                clips=features.reshape(b,channels,48,8).permute(0,2,1,3).reshape(b*48,channels,8)
                if spec.kind=='aux' and len(execution['prefix_ids'])>len(execution['full_ids']):
                    shallow=depth.flatten()[execution['prefix_ids']]==8
                    clips=clips.index_copy(0,execution['prefix_ids'][shallow],self.aux.exit8(execution['at8'][shallow]).float())
                if execution['at12'] is not None:clips=clips.index_copy(0,execution['full_ids'],execution['at12'].float())
                features=self.teacher.grid(clips)
                trace=execution['trace']
            else:
                trace=dict(attention_clips=[0]*12,attention_tokens=[0]*12,heavy_mlp_tokens=[0]*12,heavy_input_clips=0,spatial_ratio=spec.spatial_ratio)
            if spec.kind=='interpolate' and not bool((depth==12).all()):
                observed=(depth==12).repeat_interleave(8,1)&valid
                features=torch.stack([interpolate_native(value[:,seen].T,time[seen],time).T
                                      for value,seen,time in zip(features,observed,centers)])
        source=depth.repeat_interleave(8,1)
        full=(source==12)&valid&(spec.spatial_ratio==1.)
        kinds=torch.zeros_like(source)
        kinds[source==8]=1
        kinds[source==12]=2 if spec.spatial_ratio==1. else 3
        if spec.kind=='interpolate':kinds[source==0]=4
        kinds[~valid]=-1
        trace.update(depths=depth,valid_mask=valid,source_frame_indices=frames,tubelet_centers=centers,
                     full_heavy_observed_mask=full,estimated_mask=valid&~full,
                     source_kind=kinds,exit_depth=source)
        return features,trace

    def predictions(self, inputs, masks, metas=None, policy=None, preview=None):
        native,trace=self.native(inputs,masks,metas or [{} for _ in inputs],policy,preview)
        return self.teacher.predictions(native,masks,metas),trace

    def forward(self, inputs, masks, metas, post_cfg, ext_cls, policy=None, **kwargs):
        predictions,_=self.predictions(inputs,masks,metas,policy)
        return self.teacher.detector.post_processing(predictions,metas,post_cfg,ext_cls)
