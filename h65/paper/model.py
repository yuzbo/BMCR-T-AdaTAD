"""Complete paper model: jointly trained readout, recovery and conditional budgets."""
import copy
import torch
from torch import nn
from h65.frame.geometry import make_anchors,make_queries
from h65.frame.teachers import OriginalTeacher
from .encoder import NativeEncoder
from .decoder import PaperDecoder
from .readout import TaskReadout
from .geometry import candidate_mask,scout_context
from .routing import BudgetRouter,FrameRouter,video_context,plans


class PaperModel(nn.Module):
    def __init__(self,model_cfg,cfg,resources):
        super().__init__();self.config=copy.deepcopy(cfg);key=cfg['dataset']+':'+cfg['backbone']
        source=resources['encoders'][key]
        self.encoder=NativeEncoder(model_cfg.model,source,source['scout_checkpoint'],source.get('variant','h65'),
            train_backbone=cfg.get('train_backbone',False),train_adapters=cfg.get('train_adapters',True),
            train_scout=cfg.get('train_scout',True) and not cfg.get('dense_baseline',False))
        teacher_path=resources.get('teachers',{}).get(key)
        self.teacher=OriginalTeacher(model_cfg.paper_point_model,teacher_path) if teacher_path else None
        head_source=teacher_path if cfg['head']=='point' else None
        self.readout=TaskReadout(model_cfg.model,cfg['head'],head_source,trainable=cfg.get('train_head',True))
        self.decoder=PaperDecoder(self.encoder.channels,self.encoder.depth,cfg.get('multidepth',True),
                                 cfg.get('decoder','cross'),cfg.get('provenance',True),cfg.get('scout_context',True))
        recovery=resources.get('recovery_initialization',{}).get(key)
        if recovery and cfg.get('initialize_recovery',True) and cfg.get('decoder','cross')=='cross':
            self.decoder.load_recovery(recovery)
        self.menu=plans();self.budget_router=BudgetRouter(self.menu);self.frame_router=FrameRouter()
        self.frame_router.requires_grad_(cfg.get('frame_utility',True) and cfg.get('selector','anchor')=='anchor')
        self.budget_router.network.requires_grad_(cfg.get('dynamic_budget',True))
        if not cfg.get('spatial',True):self.encoder.engine.requires_grad_(False)
        if cfg.get('dense_baseline',False):
            self.decoder.requires_grad_(False);self.encoder.engine.requires_grad_(False)
            self.budget_router.requires_grad_(False);self.frame_router.requires_grad_(False)
        if cfg.get('shallow_full',False):
            self.shallow_project=nn.Linear(self.encoder.channels,96)
            nn.init.zeros_(self.shallow_project.weight);nn.init.zeros_(self.shallow_project.bias)
        self.model_cfg=model_cfg

    def train(self,mode=True):
        super().train(mode)
        self.encoder.train(mode);self.readout.train(mode)
        if self.teacher is not None:self.teacher.eval()
        return self

    def plan(self,value):
        if isinstance(value,dict):result=dict(value)
        else:
            index=value if isinstance(value,int) else next(i for i,p in enumerate(self.menu) if p['id']==value)
            result=dict(self.menu[index])
        if not self.config.get('temporal',True):result['frames']=768
        if not self.config.get('depth',True):result['depth']=1.
        if not self.config.get('spatial',True):result['space']=1.
        if self.config.get('full_kv',False):result['full_kv']=True
        if self.config.get('structured',False):result['structured']=True
        if self.config.get('attention_uniform',False):result['gate']='uniform'
        if self.config.get('static_depth'):result['static_depth']=self.config['static_depth']
        return result

    def shallow(self,inputs,masks):
        import torch.nn.functional as F
        with torch.no_grad():
            frames,_=self.encoder.backbone.model.data_preprocessor.preprocess(self.encoder.backbone.tensor_to_list(inputs),None,False)
            b,_,c,t,h,w=frames.shape
            images=F.interpolate(frames[:,0].permute(0,2,1,3,4).reshape(b*t,c,h,w),size=(80,80),mode='bilinear',align_corners=False)
            clips=images.reshape(b,t,c,80,80).permute(0,2,1,3,4).reshape(b,c,t//16,16,80,80).permute(0,2,1,3,4,5).reshape(b*t//16,c,16,80,80)
            tokens,_=self.encoder.vit.patch_embed(clips)
            features=tokens.reshape(b,t//16,8,-1,self.encoder.channels).mean(3).reshape(b,t//2,-1)
        return self.shallow_project(features)*masks.reshape(b,-1,2).any(-1)[...,None]

    def forward_native(self,data,force_plan=None,selection=None,preview=None,apply_refiner=True,execution='compact'):
        inputs=data['inputs'];masks=candidate_mask(data);metas=data['metas']
        if self.config.get('dense_baseline',False):force_plan=0
        if preview is None:
            if self.config.get('dense_baseline',False):
                preview=dict(hidden=inputs.new_zeros((len(inputs),masks.shape[1],96),dtype=torch.float32),
                             action_logits=inputs.new_zeros(masks.shape,dtype=torch.float32),
                             rate_logits=inputs.new_zeros(masks.shape,dtype=torch.float32))
            else:preview=self.encoder.preview(inputs,masks)
        context=video_context(preview,masks,self.config['budget_fraction'])
        distribution=None if self.config.get('dense_baseline',False) else self.budget_router.distribution(context)
        if force_plan is None:
            if self.config.get('dynamic_budget',True):
                index,_=self.budget_router.choose(context,self.config['budget_fraction'],self.config.get('risk_weight',.25))
                if len(index)!=1:raise ValueError('Routed inference uses one video window per GPU batch')
                force_plan=int(index[0])
            else:force_plan=self.config.get('fixed_plan',5)
        plan=self.plan(force_plan)
        if selection is None:selection=self.encoder.select(preview,masks,plan['frames'],'uniform' if self.config.get('dense_baseline',False) else self.config.get('selector','anchor'),metas)
        before=selection;routing=dict(changes=[],pair_count=0)
        if apply_refiner and self.config.get('frame_utility',True) and self.config.get('selector','anchor')=='anchor' and plan['frames']<768:
            selection,routing=self.frame_router.refine(preview,selection,masks,self.config.get('partner_scope','local'),self.config.get('risk_weight',.25))
        native,levels,trace=self.encoder.encode(inputs,selection,plan,self.config.get('multidepth',True) and not self.config.get('dense_baseline',False),execution)
        anchors=make_anchors(native,selection,masks,metas,trace);queries=make_queries(masks,metas,selection)
        if self.config.get('dense_baseline',False):
            recovered=native.transpose(1,2).float()
        else:
            cheap=scout_context(preview,masks,self.config.get('train_scout',True))
            if hasattr(self,'shallow_project'):cheap=cheap+self.shallow(inputs,masks)
            recovered=self.decoder(anchors,queries,cheap,levels).float()
        trace.update(selected_indices=selection.indices,selected_valid=selection.valid,
                     anchor_centers=anchors.centers,contributor_times=anchors.contributor_times,
                     query_centers=queries.centers,query_valid=queries.valid,
                     unique_selected_candidates=selection.valid.sum(-1),plan={k:v for k,v in plan.items() if k!='route_masks'})
        return recovered,dict(trace=trace,selection=selection,anchor_selection=before,preview=preview,
                               context=context,distribution=distribution,plan=plan,routing=routing,
                               layer_features=levels,anchors=anchors,queries=queries)

    def predictions(self,data,force_plan=None):
        native,detail=self.forward_native(data,force_plan)
        return self.readout.predictions(native,data['masks'],data['metas']),detail

    def learned_state(self):
        names={n for n,p in self.named_parameters() if p.requires_grad}
        # Actual mutable buffers: scout BN, student loss normalizer, calibrated routers.
        prefixes=['readout.','budget_router.','frame_router.']
        if self.config.get('train_scout',True):prefixes.append('encoder.scout.')
        names|={n for n,b in self.named_buffers() if n.startswith(tuple(prefixes))}
        state=self.state_dict();return {n:state[n] for n in sorted(names)}

    def load_learned(self,state):
        expected=self.learned_state()
        if set(expected)!=set(state):raise ValueError('Paper checkpoint trainable-state contract mismatch')
        own=self.state_dict()
        with torch.no_grad():
            for name,value in state.items():own[name].copy_(value)
