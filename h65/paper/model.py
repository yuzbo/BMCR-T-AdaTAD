"""Complete paper model: jointly trained readout, recovery and conditional budgets."""
import copy
import torch
from torch import nn
from h65.frame.geometry import make_anchors,make_queries
from h65.frame.teachers import OriginalTeacher
from .encoder import NativeEncoder
from .decoder import PaperDecoder,PaperMAEDecoder
from .readout import TaskReadout
from .geometry import candidate_mask,scout_context
from .routing import BudgetRouter,FrameRouter,video_context,plans


class PaperModel(nn.Module):
    def __init__(self,model_cfg,cfg,resources,with_teacher=True):
        super().__init__();self.config=copy.deepcopy(cfg);key=cfg['dataset']+':'+cfg['backbone']
        source=resources['encoders'][key]
        if cfg.get('recognition_only'):source=resources['recognition_pretrain'][cfg['backbone']]
        self.encoder=NativeEncoder(model_cfg.model,source,None if cfg.get('random_scout') else source['scout_checkpoint'],source.get('variant','h65'),
            train_backbone=cfg.get('train_backbone',False),train_adapters=cfg.get('train_adapters',True),
            train_scout=cfg.get('train_scout',True) and not cfg.get('dense_baseline',False),resolution=cfg.get('resolution',160),seed=cfg['seed'],
            depth_bypass=cfg.get('depth_bypass')=='light',train_norm=cfg.get('train_norm',False))
        teacher_path=resources.get('teachers',{}).get(key)
        self.teacher=OriginalTeacher(model_cfg.paper_point_model,teacher_path) if teacher_path and with_teacher and cfg.get('instantiate_external_teacher',True) and not cfg.get('dense_baseline',False) else None
        head_source=teacher_path if cfg['head']=='point' and not cfg.get('random_head') else None
        self.readout=TaskReadout(model_cfg.model,cfg['head'],head_source,trainable=cfg.get('train_head',True))
        if cfg.get('decoder')=='mae':
            self.decoder=PaperMAEDecoder(self.encoder.channels,self.encoder.depth,resources['decoder_pretrain'][cfg['backbone']],cfg.get('mae_pretrained',True),cfg.get('decoder_input_alignment'))
        else:
            self.decoder=PaperDecoder(self.encoder.channels,self.encoder.depth,cfg.get('multidepth',True),
                                     cfg.get('decoder','cross'),cfg.get('provenance',True),cfg.get('scout_context',True))
        recovery=resources.get('recovery_initialization',{}).get(key)
        if recovery and cfg.get('initialize_recovery',True) and cfg.get('decoder','cross')=='cross':
            self.decoder.load_recovery(recovery)
        self.menu=plans();self.budget_router=BudgetRouter(self.menu);self.frame_router=FrameRouter(cfg.get('plan_aware_frame',False))
        self.frame_router.requires_grad_(cfg.get('frame_utility',True) and cfg.get('selector','anchor')=='anchor')
        self.budget_router.network.requires_grad_(cfg.get('dynamic_budget',True))
        if not cfg.get('spatial',True) or not cfg.get('use_light',True):self.encoder.engine.requires_grad_(False)
        if cfg.get('dense_baseline',False):
            self.decoder.requires_grad_(False);self.encoder.engine.requires_grad_(False)
            self.budget_router.requires_grad_(False);self.frame_router.requires_grad_(False)
        if cfg.get('shallow_full',False):
            self.shallow_project=nn.Linear(self.encoder.channels,96)
            nn.init.zeros_(self.shallow_project.weight);nn.init.zeros_(self.shallow_project.bias)
        self.model_cfg=model_cfg
        self.training_epoch=0
        if cfg.get('mod_start') is not None:
            active=set(range(cfg['mod_start'],self.encoder.depth-1,2))
            for i,m in enumerate(self.encoder.engine.light):m.requires_grad_(i in active and cfg.get('spatial',True))
            if hasattr(self.encoder.engine,'depth_attention'):
                for i in range(self.encoder.depth):
                    self.encoder.engine.depth_attention[i].requires_grad_(i in active)
                    self.encoder.engine.depth_ffn[i].requires_grad_(i in active)
        self.support_reference=None
        if cfg.get('support_reference') and with_teacher:
            self.ensure_support_reference()
        if cfg.get('static_compression'):
            self.register_buffer('retained_blocks',torch.ones(self.encoder.depth,dtype=torch.bool))
            self.register_buffer('compression_stage',torch.tensor(-1,dtype=torch.long))

    def ensure_support_reference(self):
        if self.support_reference is None:
            cfg=self.config;source=self.encoder.provenance['source']
            # Reconstruct from fixed assets; the framework wrapper is not pickleable.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg['seed'])
                self.support_reference=NativeEncoder(self.model_cfg.model,source,None if cfg.get('random_scout') else source['scout_checkpoint'],source.get('variant','h65'),
                    train_backbone=False,train_adapters=False,train_scout=False,resolution=cfg.get('resolution',160),seed=cfg['seed']).requires_grad_(False).eval().to(next(self.encoder.parameters()).device)
        return self.support_reference

    def train(self,mode=True):
        super().train(mode)
        self.encoder.train(mode);self.readout.train(mode)
        if self.teacher is not None:self.teacher.eval()
        if self.support_reference is not None:self.support_reference.eval()
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
        if not self.config.get('use_light',True):result['use_light']=False
        if self.config.get('static_depth'):result['static_depth']=self.config['static_depth']
        if value!=0:
            if 'frames' in self.config:result['frames']=self.config['frames']
            if 'depth_capacity' in self.config:result['depth']=self.config['depth_capacity']
            if 'space_capacity' in self.config:result['space']=self.config['space_capacity']
        if self.config.get('capacity_course')=='predeclared_progressive' and self.training:
            scaled=(self.training_epoch+1)*40/self.config['epochs']
            floor=(1.,1.) if scaled<=5 else (.75,.75) if scaled<=10 else (.5,.75) if scaled<=20 else (.5,.48)
            result['depth']=max(result['depth'],floor[0]);result['space']=max(result['space'],floor[1])
        if self.config.get('kv_mode'):result['full_kv']=self.config['kv_mode']=='full'
        if self.config.get('depth_bypass'):result['depth_bypass']=self.config['depth_bypass']
        if self.config.get('mod_start') is not None:result['mod_layers']=list(range(self.config['mod_start'],self.encoder.depth-1,2))
        if self.config.get('depth_gate'):result['depth_gate']=self.config['depth_gate']
        if hasattr(self,'retained_blocks'):result['static_keep']=self.retained_blocks.nonzero().flatten().tolist()
        if self.config.get('dense_baseline',False):
            result.update(frames=768,depth=1.,space=1.,depth_bypass='hold')
            result.pop('static_keep',None)
        result['nominal_id']=result['id']
        result['id']=f"K{result['frames']}_D{int(result['depth']*100)}_S{int(result['space']*100)}"
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

    def forward_native(self,data,force_plan=None,selection=None,preview=None,apply_refiner=True,execution='compact',diagnostics=False,capture_support=False,operator_diagnostics=False):
        inputs=data['inputs'];masks=candidate_mask(data);metas=data['metas']
        if self.config.get('dense_baseline',False):force_plan=0
        if preview is None:
            if self.config.get('dense_baseline',False):
                preview=dict(hidden=inputs.new_zeros((len(inputs),masks.shape[1],96),dtype=torch.float32),
                             action_logits=inputs.new_zeros(masks.shape,dtype=torch.float32),
                             rate_logits=inputs.new_zeros(masks.shape,dtype=torch.float32))
            else:preview=self.encoder.preview(inputs,masks)
        context=video_context(preview,masks)
        distribution=None if self.config.get('dense_baseline',False) else self.budget_router.distribution(context)
        if force_plan is None:
            if self.config.get('dynamic_budget',True):
                index,_=self.budget_router.choose(context,self.config['budget_fraction'],self.config.get('risk_weight',.25),distribution)
                if len(index)!=1:raise ValueError('Routed inference uses one video window per GPU batch')
                force_plan=int(index[0])
            else:force_plan=self.config.get('fixed_plan',5)
        plan=self.plan(force_plan)
        plan_index=force_plan if isinstance(force_plan,int) else next((i for i,p in enumerate(self.menu) if p['id']==force_plan),None) if isinstance(force_plan,str) else None
        if selection is None:selection=self.encoder.select(preview,masks,plan['frames'],'uniform' if self.config.get('dense_baseline',False) else self.config.get('selector','anchor'),metas)
        before=selection;routing=dict(changes=[],pair_count=0)
        if apply_refiner and self.config.get('frame_utility',True) and self.config.get('selector','anchor')=='anchor' and plan['frames']<768:
            selection,routing=self.frame_router.refine(preview,selection,masks,self.config.get('partner_scope','local'),self.config.get('risk_weight',.25),plan=plan)
        capture='diagnostic' if diagnostics else self.config.get('multidepth',True) and not self.config.get('dense_baseline',False)
        support_layers=(tuple(i+1 for i in plan.get('static_keep',range(self.encoder.depth))) if self.config.get('support_layers')=='retained' else tuple(sorted({self.encoder.depth//2,3*self.encoder.depth//4,self.encoder.depth}))) if capture_support else None
        native,levels,trace=self.encoder.encode(inputs,selection,plan,capture,execution,capture_support,support_layers,operator_diagnostics)
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
                               layer_features=levels,anchors=anchors,queries=queries,plan_index=plan_index)

    def predictions(self,data,force_plan=None):
        native,detail=self.forward_native(data,force_plan)
        return self.readout.predictions(native,data['masks'],data['metas']),detail

    def learned_state(self):
        names={n for n,p in self.named_parameters() if p.requires_grad}
        # Actual mutable buffers: scout BN, student loss normalizer, calibrated routers.
        prefixes=['readout.','budget_router.','frame_router.']
        if self.config.get('train_scout',True):prefixes.append('encoder.scout.')
        names|={n for n,b in self.named_buffers() if n.startswith(tuple(prefixes))}
        if hasattr(self,'retained_blocks'):names|={'retained_blocks','compression_stage'}
        state=self.state_dict()
        # Fixed sinusoidal positions are explicitly non-persistent upstream.
        return {n:state[n] for n in sorted(names) if n in state}

    def load_learned(self,state):
        expected=self.learned_state()
        if set(expected)!=set(state):raise ValueError('Paper checkpoint trainable-state contract mismatch')
        own=self.state_dict()
        with torch.no_grad():
            for name,value in state.items():own[name].copy_(value)
