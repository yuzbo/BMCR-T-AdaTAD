"""Existing frame selector and exact pre-interpolation native readout."""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from h65.transport import Selection,sample_rates,gather_with_transport


class FrozenAnchor(nn.Module):
    def __init__(self,model_cfg,checkpoint,variant,budget=384):
        super().__init__()
        from h65.full.model import FormalH65
        self.model=FormalH65(model_cfg,pretrain=None,variant=variant,budget=384)
        payload=torch.load(checkpoint,map_location='cpu')
        key='state_dict_ema' if 'state_dict_ema' in payload else 'state_dict'
        self.model.load_state_dict(payload[key],strict=True)
        self.provenance=dict(checkpoint=str(Path(checkpoint).resolve()),state_key=key,variant=variant,
                            completed_epochs=payload.get('completed_epochs'),metadata=payload.get('metadata',{}))
        del payload
        self.model.requires_grad_(False);self.model.eval();self.budget=budget
        self.configure_budget(budget)

    @property
    def vit(self):return self.model.backbone.model.backbone

    def train(self,mode=True):
        super().train(False);return self

    def configure_budget(self,budget):
        if budget%16 or not 16<=budget<=768:raise ValueError('Frame budget must be16-multiple in[16,768]')
        self.budget=budget;self.model.budget=budget
        for block in self.vit.blocks:
            if block.use_adapter:block.adapter.temporal_size=budget//2

    @torch.no_grad()
    def select(self,inputs,masks,metas,mode='anchor'):
        output=self.model.scout(inputs,masks,1.,self.model.variant)
        provisional=sample_rates(output['rate_logits'],masks,self.budget,alpha=1.)
        if mode=='anchor':
            selection=provisional
            if self.model.variant=='bmcr':
                condition=self.model.scout.condition(output,provisional,masks)
                selection=sample_rates(output['rate_logits']+condition['utility'].mean(-1),masks,self.budget,alpha=1.)
        elif mode=='uniform':selection=sample_rates(output['rate_logits'],masks,self.budget,alpha=0.)
        elif mode=='random':
            rows=[];flags=[];rates=[]
            for row,mask in enumerate(masks):
                length=int(mask.sum());count=min(length,self.budget)
                name=str(metas[row].get('video_name',''));seed=3407+sum((i+1)*ord(c) for i,c in enumerate(name))+int(metas[row].get('window_start_frame',0))
                generator=torch.Generator(device=masks.device).manual_seed(seed)
                ids=torch.randperm(length,generator=generator,device=masks.device)[:count].sort().values
                rows.append(F.pad(ids,(0,self.budget-count),value=length-1));flags.append(torch.arange(self.budget,device=masks.device)<count)
                rates.append(mask.float()*count/length)
            indices=torch.stack(rows);valid=torch.stack(flags);rate=torch.stack(rates)
            selection=Selection(indices,indices.float(),valid,rate/valid.sum(-1,keepdim=True),rate)
        else:raise ValueError(mode)
        return selection,output,provisional

    def selected_rgb(self,inputs,selection):
        return gather_with_transport(inputs,selection,0.)*selection.valid[:,None,None,:,None,None]

    def prepare_clips(self,selected_rgb):
        wrapper=self.model.backbone
        frames,_=wrapper.model.data_preprocessor.preprocess(wrapper.tensor_to_list(selected_rgb),None,False)
        b,n,c,k,h,w=frames.shape
        if n!=1 or k!=self.budget:raise ValueError('Wrong selected RGB packing')
        return frames.reshape(b,n,c,k//16,16,h,w).permute(0,3,1,2,4,5,6).reshape(b*(k//16),c,16,h,w).contiguous()

    def pool_maps(self,maps):
        clips,channels,time,h,w=maps.shape;per_clip=maps.mean((-1,-2));batch=clips//(self.budget//16)
        return per_clip.reshape(batch,self.budget//16,channels,time).permute(0,1,3,2).reshape(batch,self.budget//2,channels)

    def pool_tokens(self,tokens,h,w):
        tokens=self.vit.norm(tokens)
        maps=tokens.reshape(len(tokens),8,h,w,self.vit.embed_dims).permute(0,4,1,2,3)
        return self.pool_maps(maps)

    def encode_native(self,inputs,selection):
        clips=self.prepare_clips(self.selected_rgb(inputs,selection));maps=self.vit(clips);features=self.pool_maps(maps)
        p=maps.shape[-1]*maps.shape[-2];rows=len(clips)*8*p;layers=len(self.vit.blocks)
        trace=dict(q=[rows]*layers,kv=[rows]*layers,heavy_mlp=[rows]*layers,light=[0]*layers,tia=[rows]*layers,
                   score_qk=[0]*layers,physical_candidates=self.budget,unique_selected=selection.valid.sum(-1),
                   patch_input_shape=list(clips.shape),last_heavy_depth=features.new_full(features.shape[:2],layers),
                   spatial_quality=features.new_ones(features.shape[:2]),native_capture='before final rank interpolate')
        return features,trace
