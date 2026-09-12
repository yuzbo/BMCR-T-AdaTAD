"""One frame-selected model: original-axis recovery and optional A-MoD/space updates."""
import copy
from dataclasses import replace
import torch
from torch import nn
import torch.nn.functional as F
from .contracts import EnginePolicy
from .reader import FrozenAnchor
from .geometry import make_anchors,make_queries,scout_context
from .decoder import build_decoder
from .engine import PackedStateEngine
from .teachers import OriginalTeacher
from .router import ActionRouter


class FrameModel(nn.Module):
    def __init__(self,model_cfg,config,resources):
        super().__init__();self.config=copy.deepcopy(config);b=config['backbone'];source=resources['anchors'][b]
        self.anchor=FrozenAnchor(model_cfg,source['checkpoint'],source['variant'],config['budget'])
        self.teacher=OriginalTeacher(model_cfg,resources['official'][b])
        self.decoder=build_decoder(channels=self.anchor.vit.embed_dims,**config['decoder'])
        self.engine=PackedStateEngine(self.anchor.vit.embed_dims)
        self.policy=EnginePolicy(**config.get('engine',{}));self.router=ActionRouter()
        self.engine.requires_grad_(False);self.router.requires_grad_(bool(config.get('router',False)))
        if config.get('train_adapters',False):
            for name,p in self.anchor.model.backbone.named_parameters():p.requires_grad_('adapter' in name)
        if self.policy.spatial_ratio<1 and self.policy.use_light:
            for i in self.policy.mod_layers:self.engine.light[i].requires_grad_(True)
        self.teacher.eval();self.anchor.eval()

    def train(self,mode=True):
        super().train(mode);self.anchor.eval();self.teacher.eval();return self

    def forward_native(self,data,selection=None,policy=None,selector=None,apply_router=True):
        inputs,masks,metas=data['inputs'],data['masks'],data['metas']
        self.anchor.configure_budget(self.config['budget'])
        if selection is None:selection,output,provisional=self.anchor.select(inputs,masks,metas,selector or self.config['selector'])
        else:
            with torch.no_grad():output=self.anchor.model.scout(inputs,masks,1.,self.anchor.model.variant)
            provisional=selection
        anchor_selection=selection;routing=dict(changes=[])
        if apply_router and self.config.get('router',False):
            selection,routing=self.router.refine(output,selection,masks,self.config.get('partner_scope','local'))
        policy=policy or replace(self.policy,mode=self.config.get('train_execution','dense_mask') if self.training else 'compact')
        use_engine=(policy.depth_schedule!='none' or policy.spatial_ratio<1 or policy.static_depth<12 or self.config.get('train_adapters',False) or self.config.get('resolution',160)!=160)
        if use_engine:
            rgb=self.anchor.selected_rgb(inputs,selection)
            resolution=self.config.get('resolution',160)
            if rgb.shape[-1]!=resolution:
                b,_,c,t,h,w=rgb.shape
                images=rgb[:,0].permute(0,2,1,3,4).reshape(b*t,c,h,w)
                images=F.interpolate(images,size=(resolution,resolution),mode='bilinear',align_corners=False)
                rgb=images.reshape(b,t,c,resolution,resolution).permute(0,2,1,3,4)[:,None]
            clips=self.anchor.prepare_clips(rgb);valid=selection.valid.reshape(len(masks),-1,2).any(-1)
            tokens,h,w,trace=self.engine(self.anchor.vit,clips,policy,valid)
            features=self.anchor.pool_tokens(tokens,h,w)
        else:
            with torch.no_grad():features,trace=self.anchor.encode_native(inputs,selection)
        anchors=make_anchors(features,selection,masks,metas,trace);queries=make_queries(masks,metas,selection)
        context=scout_context(output,masks)
        native=self.decoder(anchors,queries,context).float()
        trace.update(unique_selected=selection.valid.sum(-1),selected_indices=selection.indices,selected_valid=selection.valid,
            anchor_contributor_times=anchors.contributor_times,anchor_centers=anchors.centers,query_centers=queries.centers,
            query_valid=queries.valid,decoder_queries=queries.centers.shape[1],decoder_memory=features.shape[1],
            anchor_feature_space=anchors.feature_space,routing=routing)
        return native,dict(trace=trace,selection=selection,anchor_selection=anchor_selection,scout_output=output,provisional=provisional)

    def predictions(self,inputs,masks,metas):
        native,detail=self.forward_native(dict(inputs=inputs,masks=masks,metas=metas))
        return self.teacher.predictions(native,masks,metas),detail

    def forward(self,inputs,masks,metas,post_cfg=None,ext_cls=None,**kwargs):
        prediction,_=self.predictions(inputs,masks,metas)
        return self.teacher.post_processing(prediction,metas,post_cfg,ext_cls)

    def learned_state(self):
        names={n for n,p in self.named_parameters() if p.requires_grad}
        if self.config.get('router',False):names.add('router.scales')
        state=self.state_dict();return {n:state[n] for n in sorted(names)}

    def load_learned(self,state):
        expected=self.learned_state()
        if set(state)!=set(expected):raise ValueError('Checkpoint trainable modules do not match this recipe')
        current=self.state_dict()
        with torch.no_grad():
            for name,value in state.items():current[name].copy_(value)

    def full_policy(self):return replace(self.policy,depth_ratio=1.,spatial_ratio=1.,query_ratio=1.,static_depth=12)
