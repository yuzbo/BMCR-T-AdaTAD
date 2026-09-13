"""Native VideoMAE / OpenTAD InternVideo1-MQ encoding on selected RGB."""
import copy
import torch
from torch import nn
from h65.full.scout import FormalScout
from h65.transport import Selection, sample_rates, gather_with_transport
from h65.frame.contracts import EnginePolicy
from .engine import PackedStateEngine


def checkpoint_state(path):
    payload=torch.load(path,map_location='cpu')
    state=payload.get('state_dict_ema',payload.get('state_dict',payload.get('model',payload)))
    return {k.removeprefix('module.'):v for k,v in state.items()}


class NativeEncoder(nn.Module):
    def __init__(self,model_cfg,source,scout_source,variant='h65',train_backbone=False,train_adapters=True,train_scout=True,resolution=160,seed=42,depth_bypass=False,train_norm=False):
        super().__init__()
        self.seed=seed
        from opentad.models.builder import build_backbone
        cfg=copy.deepcopy(model_cfg.backbone);cfg.custom.pretrain=None;cfg.custom.temporal_checkpointing=False
        cfg.backbone.with_cp=False
        self.backbone=build_backbone(cfg); self.scout=FormalScout();self.variant=variant;self.resolution=resolution
        self.channels=self.vit.embed_dims;self.depth=len(self.vit.blocks)
        self.engine=PackedStateEngine(self.channels,self.depth,depth_bypass=depth_bypass)
        self.engine.requires_grad_(False)
        for i in range(1,self.depth-1,2):self.engine.light[i].requires_grad_(True)
        if depth_bypass:
            for i in range(1,self.depth-1):
                self.engine.depth_attention[i].requires_grad_(True);self.engine.depth_ffn[i].requires_grad_(True)
        state=checkpoint_state(source['checkpoint'])
        if source['kind']=='task':
            prefix='backbone.'
            selected={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
            self.backbone.load_state_dict(selected,strict=True)
        elif source['kind']=='recognition':
            # OpenTAD's converted recognizer checkpoint; adapters are newly added.
            expected=self.backbone.model.state_dict()
            # The RGB feature engine never executes the recognition classifier.
            selected={k:v for k,v in state.items() if k in expected and k.startswith('backbone.')}
            missing=set(expected)-set(selected)
            required={k for k in missing if k.startswith('backbone.') and 'adapter' not in k}
            if required:raise ValueError('Missing pretrained backbone keys: '+str(sorted(required)[:20]))
            bad=[k for k,v in selected.items() if v.shape!=expected[k].shape]
            if bad:raise ValueError('Pretrained shape mismatch: '+str(bad))
            self.backbone.model.load_state_dict(selected,strict=False)
            self.pretraining_missing=sorted(missing)
        else:raise ValueError(source['kind'])
        if scout_source is not None:
            scout=checkpoint_state(scout_source)
            self.scout.load_state_dict({k.removeprefix('scout.'):v for k,v in scout.items() if k.startswith('scout.')},strict=True)
        self.backbone.requires_grad_(False)
        for name,p in self.vit.named_parameters():p.requires_grad_(train_backbone or (train_adapters and 'adapter' in name) or (train_norm and 'norm' in name))
        self.scout.requires_grad_(train_scout);self.train_scout=train_scout
        self.provenance=dict(source=source,scout_source=str(scout_source),variant=variant,depth=self.depth,channels=self.channels)

    @property
    def vit(self):return self.backbone.model.backbone

    def train(self,mode=True):
        super().train(mode)
        # Frozen pretraining stochastic layers stay in the same execution mode;
        # parameter gradients remain enabled for requested adaptation groups.
        self.backbone.eval()
        self.scout.train(mode and self.train_scout)
        return self

    def preview(self,inputs,masks):
        if self.train_scout and self.training:return self.scout(inputs,masks,1.,self.variant)
        with torch.no_grad():return self.scout(inputs,masks,1.,self.variant)

    def select(self,output,masks,budget,selector='anchor',metas=None):
        if selector=='random':
            import torch.nn.functional as F
            rows=[];flags=[]
            for row,mask in enumerate(masks):
                length=int(mask.sum());count=min(budget,length);meta=(metas or [{}]*len(masks))[row]
                name=str(meta.get('video_name',''));times=torch.as_tensor(meta.get('frame_inds',[0])).flatten()
                seed=self.seed+sum((i+1)*ord(c) for i,c in enumerate(name))+int(times[0])
                generator=torch.Generator(device=masks.device).manual_seed(seed)
                ids=torch.randperm(length,generator=generator,device=masks.device)[:count].sort().values
                rows.append(F.pad(ids,(0,budget-count),value=length-1));flags.append(torch.arange(budget,device=masks.device)<count)
            ids=torch.stack(rows);valid=torch.stack(flags)
            rates=masks.float()*valid.sum(-1,keepdim=True)/masks.sum(-1,keepdim=True)
            return Selection(ids,ids.float(),valid,rates/valid.sum(-1,keepdim=True),rates)
        if selector not in ('uniform','anchor'):raise ValueError(selector)
        alpha=0. if selector=='uniform' else 1.
        provisional=sample_rates(output['rate_logits'],masks,budget,alpha=alpha)
        if selector=='anchor' and self.variant=='bmcr':
            condition=self.scout.condition(output,provisional,masks)
            return sample_rates(output['rate_logits']+condition['utility'].mean(-1),masks,budget,alpha=1.)
        return provisional

    def prepare(self,inputs,selection):
        rgb=gather_with_transport(inputs,selection,.25 if self.training and self.train_scout else 0.)
        rgb=rgb*selection.valid[:,None,None,:,None,None]
        if rgb.shape[-2:]!=(self.resolution,self.resolution):
            import torch.nn.functional as F
            b,_,c,t,h,w=rgb.shape
            images=F.interpolate(rgb[:,0].permute(0,2,1,3,4).reshape(b*t,c,h,w),size=(self.resolution,self.resolution),mode='bilinear',align_corners=False)
            rgb=images.reshape(b,t,c,self.resolution,self.resolution).permute(0,2,1,3,4)[:,None]
        frames,_=self.backbone.model.data_preprocessor.preprocess(self.backbone.tensor_to_list(rgb),None,False)
        b,n,c,k,h,w=frames.shape
        if n!=1 or k%16:raise ValueError('Native reader uses actual candidate frames in 16-observation packing')
        clips=frames.reshape(b,n,c,k//16,16,h,w).permute(0,3,1,2,4,5,6).reshape(b*k//16,c,16,h,w).contiguous()
        return clips

    def pool(self,x,h,w,batch):
        x=self.vit.norm(x)
        return x.reshape(batch,-1,8,h*w,self.channels).mean(3).reshape(batch,-1,self.channels)

    def encode(self,inputs,selection,plan,capture=True,execution='compact',state_capture=False,support_layers=None,operator_diagnostics=False):
        budget=selection.indices.shape[1]
        for block in self.vit.blocks:
            if block.use_adapter:block.adapter.temporal_size=budget//2
        policy=EnginePolicy(mode=execution,depth_schedule='amod' if plan['depth']<1 else 'none',
                            depth_ratio=plan['depth'],spatial_ratio=plan['space'],
                            mod_layers=tuple(plan.get('mod_layers',range(1,self.depth-1,2))),static_depth=self.depth,
                            use_light=plan.get('use_light',True),amod_full_kv=plan.get('full_kv',False),
                            gate=plan.get('gate','attention'),structured=plan.get('structured',False))
        policy.depth_bypass=plan.get('depth_bypass','hold')
        policy.depth_gate=plan.get('depth_gate',policy.gate)
        if plan.get('route_masks') is not None:policy.route_masks=plan['route_masks']
        if plan.get('static_depth') is not None:policy.static_depth=plan['static_depth']
        if plan.get('static_keep') is not None:policy.static_keep=plan['static_keep']
        if plan.get('query_ratio') is not None:policy.query_ratio=plan['query_ratio']
        clips=self.prepare(inputs,selection);valid=selection.valid.reshape(len(inputs),-1,2).any(-1)
        levels=tuple(range(1,self.depth+1)) if capture=='diagnostic' else tuple(sorted({self.depth//2,3*self.depth//4,self.depth})) if capture else ()
        if support_layers is not None:levels=tuple(sorted(set(levels)|set(support_layers)))
        x,h,w,trace,taps=self.engine(self.vit,clips,policy,valid,levels,state_capture,operator_diagnostics)
        native=self.pool(x,h,w,len(inputs))
        features={level:self.pool(value,h,w,len(inputs)) for level,value in taps.items()}
        if capture=='diagnostic':
            import torch.nn.functional as F
            spatial=[];temporal=[];drift=[];previous=None
            for level,value in taps.items():
                token=value.detach().float().reshape(len(inputs),-1,h,w,self.channels)
                weight=valid[...,None,None]
                horizontal=(F.cosine_similarity(token[:,:,:,:-1],token[:,:,:,1:],dim=-1)*weight).sum()/(valid.sum()*h*(w-1)).clamp_min(1)
                vertical=(F.cosine_similarity(token[:,:,:-1],token[:,:,1:],dim=-1)*weight).sum()/(valid.sum()*(h-1)*w).clamp_min(1)
                spatial.append(float((horizontal+vertical)/2))
                pooled=features[level].detach().float()
                pairs=valid[:,:-1]&valid[:,1:]
                temporal.append(float((F.cosine_similarity(pooled[:,:-1],pooled[:,1:],dim=-1)*pairs).sum()/pairs.sum().clamp_min(1)))
                drift.append(0. if previous is None else float(((1-F.cosine_similarity(pooled,previous,dim=-1))*valid).sum()/valid.sum().clamp_min(1)))
                previous=pooled
            trace['diagnostic']=dict(layer_ids=list(taps),spatial_neighbor_cosine=spatial,selected_temporal_neighbor_cosine=temporal,layer_drift=drift)
        trace.update(native_length=native.shape[1],encoder_family='internvideo1_mq' if self.depth==24 else 'videomae')
        return native,features,trace
