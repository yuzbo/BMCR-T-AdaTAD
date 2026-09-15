"""Real A-MoD/FFN packing on the selected-frame grid, with complete global TIA."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from h65.frame.gates import incoming_attention,capacity_mask
from h65.frame.cost import empty_trace

def qkv_bias(attn):
    if hasattr(attn,'q_bias'):
        return torch.cat((attn.q_bias,torch.zeros_like(attn.v_bias),attn.v_bias))
    return attn.qkv.bias

def heads(x,h):return x.reshape(x.shape[0],x.shape[1],h,-1).transpose(1,2)

def attention(attn,x,selected=None,full_kv=False,dense_mask=False,need_scores=False,query_valid=None):
    b,n,c=x.shape;h=attn.num_heads;bias=qkv_bias(attn)
    if selected is None:
        q,k,v=F.linear(x,attn.qkv.weight,bias).chunk(3,-1)
        q,k,v=heads(q,h),heads(k,h),heads(v,h)
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=attn.attn_drop.p)
        scores=incoming_attention(q.detach(),k.detach(),query_valid) if need_scores else None
        y=attn.proj_drop(attn.proj(y.transpose(1,2).reshape(b,n,c)))
        return y,scores,dict(q=b*n,kv=b*n,qk_av_macs=2*b*n*n*c,score_qk=b*n*n*c if need_scores else 0)
    output=torch.zeros_like(x);counts=selected.sum(-1);qrows=kvrows=pairs=0
    if dense_mask:
        rows=(counts>0).nonzero().flatten()
        if len(rows):
            part=x.index_select(0,rows);mask=selected.index_select(0,rows)
            q,k,v=F.linear(part,attn.qkv.weight,bias).chunk(3,-1)
            q,k,v=heads(q,h),heads(k,h),heads(v,h)
            kv_mask=None if full_kv else mask[:,None,None,:]
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=kv_mask,dropout_p=attn.attn_drop.p)
            y=attn.proj_drop(attn.proj(y.transpose(1,2).reshape(len(rows),n,c)))*mask[...,None]
            output=output.index_copy(0,rows,y.to(output.dtype))
            qrows=kvrows=len(rows)*n;pairs=len(rows)*n*n
    else:
        for count in counts.unique().tolist():
            if count==0:continue
            rows=(counts==count).nonzero().flatten();part=x.index_select(0,rows);mask=selected.index_select(0,rows)
            ids=mask.nonzero()[:,1].reshape(len(rows),count)
            packed=part.gather(1,ids[...,None].expand(-1,-1,c))
            if full_kv:
                q=F.linear(packed,attn.qkv.weight[:c],None if bias is None else bias[:c])
                k=F.linear(part,attn.qkv.weight[c:2*c],None if bias is None else bias[c:2*c])
                v=F.linear(part,attn.qkv.weight[2*c:],None if bias is None else bias[2*c:]);kn=n
            else:q,k,v=F.linear(packed,attn.qkv.weight,bias).chunk(3,-1);kn=count
            y=F.scaled_dot_product_attention(heads(q,h),heads(k,h),heads(v,h),dropout_p=attn.attn_drop.p)
            y=attn.proj_drop(attn.proj(y.transpose(1,2).reshape(len(rows),count,c)))
            scattered=torch.zeros_like(part).scatter(1,ids[...,None].expand_as(y),y.to(part.dtype))
            output=output.index_copy(0,rows,scattered)
            qrows+=len(rows)*count;kvrows+=len(rows)*kn;pairs+=len(rows)*count*kn
    return output,None,dict(q=qrows,kv=kvrows,qk_av_macs=2*pairs*c,score_qk=0)

class PackedStateEngine(nn.Module):
    def __init__(self,channels,layers=12,light_width=32,depth_bypass=False):
        super().__init__();self.channels=channels
        self.light=nn.ModuleList([nn.Sequential(nn.Linear(channels,light_width),nn.GELU(),nn.Linear(light_width,channels)) for _ in range(layers)])
        for module in self.light:nn.init.zeros_(module[-1].weight);nn.init.zeros_(module[-1].bias)
        if depth_bypass:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed()+73)
                self.depth_attention=nn.ModuleList([nn.Sequential(nn.Linear(channels,light_width),nn.GELU(),nn.Linear(light_width,channels)) for _ in range(layers)])
                self.depth_ffn=nn.ModuleList([nn.Sequential(nn.Linear(channels,light_width),nn.GELU(),nn.Linear(light_width,channels)) for _ in range(layers)])
            for module in [*self.depth_attention,*self.depth_ffn]:
                nn.init.zeros_(module[-1].weight);nn.init.zeros_(module[-1].bias)

    def _spatial(self,score,allowed,ratio,structured,h,w,uniform=False):
        # One quota per native time; in joint runs it applies to admitted depth tokens.
        shape=score.shape;score=score.reshape(-1,8,h*w);allowed=allowed.reshape_as(score)
        if not structured:return capacity_mask(score,ratio,allowed,uniform).reshape(shape)
        if h%2 or w%2:raise ValueError('2x2 tiles require an even spatial grid')
        tile_score=score.reshape(-1,8,h//2,2,w//2,2).mean((3,5)).flatten(2)
        tile_valid=allowed.reshape(-1,8,h//2,2,w//2,2).any(3).any(-1).flatten(2)
        tiles=capacity_mask(tile_score,ratio,tile_valid,uniform).reshape(-1,8,h//2,w//2)
        expanded=tiles.repeat_interleave(2,-2).repeat_interleave(2,-1).reshape_as(allowed)
        return (expanded&allowed).reshape(shape)

    def forward(self,vit,clips,policy,native_valid=None,capture_layers=(),state_capture=False,operator_diagnostics=False,graph_context=None):
        def mlp(module,value):
            return checkpoint(module,value,use_reentrant=False) if self.training and torch.is_grad_enabled() and value.requires_grad else module(value)
        def light_update(module,value,mask):
            flat=value.reshape(-1,value.shape[-1]);ids=mask.flatten().nonzero().flatten()
            if not len(ids):return torch.zeros_like(value),0
            update=mlp(module,flat.index_select(0,ids))
            return torch.zeros_like(flat).index_copy(0,ids,update.to(flat.dtype)).reshape_as(value),len(ids)
        h,w=clips.shape[-2]//vit.patch_size,clips.shape[-1]//vit.patch_size
        x=vit.patch_embed(clips)[0]
        if (h,w)!=vit.grid_size:
            pos=vit.pos_embed.reshape(-1,*vit.grid_size,vit.embed_dims).permute(0,3,1,2)
            pos=F.interpolate(pos,size=(h,w),mode='bicubic',align_corners=False)
            pos=pos.permute(0,2,3,1).flatten(1,2).reshape(1,-1,vit.embed_dims)
        else:pos=vit.pos_embed
        x=vit.pos_drop(x+pos);b,n,c=x.shape;depth=len(vit.blocks);p=h*w
        if native_valid is None:
            native_len=next(block.adapter.temporal_size for block in vit.blocks if block.use_adapter)
            native_valid=torch.ones((b//(native_len//8),native_len),dtype=torch.bool,device=x.device)
        valid=native_valid.reshape(b,8).repeat_interleave(p,1)
        if len(self.light)!=depth or 0 in policy.mod_layers or depth-1 in policy.mod_layers:raise ValueError('A-MoD contract: dense first/last and matching block count')
        if policy.mode not in ('compact','dense_mask'):raise ValueError(policy.mode)
        mods=set(policy.mod_layers)
        keep_layers=set(policy.static_keep) if policy.static_keep is not None else set(torch.linspace(0,depth-1,policy.static_depth).round().long().tolist())
        if 0 not in keep_layers or depth-1 not in keep_layers:raise ValueError('First and last layers remain dense')
        tr=empty_trace(depth);tr['qk_av_macs']=[0]*depth;previous_scores=None;captures={}
        tr['query_masks']=[]
        tr.update(graph_router_macs=[0]*depth,graph_candidate_slots=[0]*depth,graph_referral_paths=[0]*depth,graph_retained_edges=[0]*depth,graph_edges={})
        tr.update(kv_masks=[],ffn_light_masks=[],depth_bypass_masks=[],age_before_reentry=[],
                  depth_attention_light=[0]*depth,depth_ffn_light=[0]*depth,state_taps={},operator_errors={})
        age=torch.zeros_like(valid,dtype=torch.long)
        last=torch.zeros((b,n),device=x.device);quality=torch.zeros_like(last)
        graph_state=None
        def coupling(value,layer):
            if graph_context is None or layer not in graph_context['coupling_layers']:return value
            state,feedback,record=graph_context['step'](value,h,w,last,quality,layer,graph_context['state'])
            graph_context['state']=state;graph_context['macs']+=record['macs']
            graph_context['records'].append(dict(stage='layer_'+str(layer),**record))
            if graph_context.get('capture'):
                tr.setdefault('time_graph_layers',{})[layer]=dict(indices=state[1].detach(),weights=state[2].detach(),geometry=state[3].detach())
            if feedback is None:return value
            return value+feedback.reshape(b,8,1,c).expand(-1,-1,p,-1).reshape_as(value)
        for i,block in enumerate(vit.blocks):
            if i not in keep_layers:
                tr['depth_masks'].append(torch.zeros_like(valid));tr['spatial_masks'].append(torch.zeros_like(valid));tr['query_masks'].append(torch.zeros_like(valid))
                tr['kv_masks'].append(torch.zeros_like(valid));tr['ffn_light_masks'].append(torch.zeros_like(valid))
                tr['depth_bypass_masks'].append(valid);tr['age_before_reentry'].append(age.clone());age=age+valid
                if i+1 in capture_layers:captures[i+1]=x
                continue
            is_mod=policy.depth_schedule=='amod' and i in mods and policy.depth_ratio<1
            query_sparse=i in mods and policy.query_ratio<1
            need_scores=(i+1 in mods and i+1 in keep_layers and
                         ((policy.depth_schedule=='amod' and policy.depth_ratio<1) or policy.spatial_ratio<1 or policy.query_ratio<1))
            if policy.static_keep is not None:
                next_kept=next((j for j in range(i+1,depth) if j in keep_layers),None)
                need_scores=next_kept in mods and policy.spatial_ratio<1
            admitted=torch.ones_like(valid)
            if is_mod:
                if previous_scores is None:raise RuntimeError('A-MoD requires the preceding dense attention scores')
                admitted=capacity_mask(previous_scores,policy.depth_ratio,valid,uniform=getattr(policy,'depth_gate',policy.gate)=='uniform')
                if policy.depth_mask is not None:admitted=policy.depth_mask.reshape(b,8).repeat_interleave(p,1)&valid
                if policy.route_masks is not None:admitted=policy.route_masks['depth'][i].to(x.device)
            attention_selected=admitted if is_mod else None
            if query_sparse:
                if previous_scores is None:raise RuntimeError('Sparse Q needs previous dense scores')
                attention_selected=capacity_mask(previous_scores,policy.query_ratio,admitted&valid)
                if policy.route_masks is not None and 'query' in policy.route_masks:attention_selected=policy.route_masks['query'][i].to(x.device)
            tr['query_masks'].append(torch.ones_like(valid) if attention_selected is None else attention_selected)
            bypass=~admitted&valid
            kv_mask=(torch.ones_like(valid) if attention_selected is None else attention_selected.any(-1,keepdim=True).expand_as(valid) if policy.amod_full_kv or query_sparse else attention_selected)
            tr['kv_masks'].append(kv_mask)
            tr['depth_bypass_masks'].append(bypass);tr['age_before_reentry'].append(age.clone())
            age=torch.where(admitted&valid,torch.zeros_like(age),age+valid)
            spatial_active=i in mods and policy.spatial_ratio<1
            graph_active=getattr(policy,'graph_kv',False) and i in mods
            all_heavy=not is_mod and not spatial_active and not query_sparse and not graph_active
            if all_heavy and not need_scores:
                before_tia={};handles=[]
                if state_capture and i+1 in capture_layers:
                    handles.append(block.norm2.register_forward_pre_hook(lambda module,args:before_tia.update(attention=args[0])))
                    if block.use_adapter:handles.append(block.adapter.register_forward_pre_hook(lambda module,args:before_tia.update(value=args[0])))
                if self.training and torch.is_grad_enabled():
                    # Mixed-budget/shared-full passes precede one backward.
                    # Checkpoint recomputation must retain this pass's TIA axis.
                    temporal=native_valid.shape[1]
                    def dense_block(value,block=block,temporal=temporal):
                        previous=block.adapter.temporal_size if block.use_adapter else None
                        if block.use_adapter:block.adapter.temporal_size=temporal
                        try:return block(value,h,w)
                        finally:
                            if block.use_adapter:block.adapter.temporal_size=previous
                    x=checkpoint(dense_block,x,use_reentrant=False)
                else:x=block(x,h,w)
                for handle in handles:handle.remove()
                if state_capture and i+1 in capture_layers:tr['state_taps'][i+1]=dict(attention=before_tia['attention'],pre_tia=before_tia.get('value',x),post_tia=x)
                tr['ffn_light_masks'].append(torch.zeros_like(valid))
                tr['q'][i]=tr['kv'][i]=tr['heavy_mlp'][i]=tr['tia'][i]=b*n
                tr['qk_av_macs'][i]=2*b*n*n*c;tr['depth_masks'].append(admitted);tr['spatial_masks'].append(admitted)
                last.fill_(i+1);quality+=1
                x=coupling(x,i+1)
                if i+1 in capture_layers:captures[i+1]=x
                continue
            normalized_attention=block.norm1(x)
            if graph_active:
                if graph_context is None:raise ValueError('Graph attention requires physical-time context')
                t=graph_context['times'].reshape(b,8,1).expand(-1,-1,p).reshape(b,n)
                yy,xx=torch.meshgrid(torch.linspace(0,1,h,device=x.device),torch.linspace(0,1,w,device=x.device),indexing='ij')
                yy=yy.flatten().repeat(8)[None].expand(b,-1);xx=xx.flatten().repeat(8)[None].expand(b,-1)
                geometry=torch.stack((t,yy,xx,quality/max(i,1),last/depth),-1).detach()
                module=self.graph_attention[str(i)]
                def run_graph(value,indices,weights,geo,module=module,attn=block.attn,selected=attention_selected,valid=valid,grid=(8,h,w)):
                    return module(attn,value,selected,valid,geo,None if indices is None else (indices,weights),grid)
                previous_i,previous_w=graph_state if graph_state is not None else (None,None)
                if self.training and torch.is_grad_enabled() and normalized_attention.requires_grad:
                    delta,indices,weights,count=checkpoint(run_graph,normalized_attention,previous_i,previous_w,geometry,use_reentrant=False)
                else:delta,indices,weights,count=run_graph(normalized_attention,previous_i,previous_w,geometry)
                graph_state=(indices,weights);scores=None
                for key in ('graph_router_macs','graph_candidate_slots','graph_referral_paths','graph_retained_edges'):tr[key][i]=count[key]
                if graph_context.get('capture'):
                    tr['graph_edges'][i]=dict(indices=indices.detach(),weights=weights.detach(),geometry=geometry)
                fraction=getattr(policy,'graph_fraction',1.)
                if fraction<1:
                    dense,_,dense_count=attention(block.attn,normalized_attention,attention_selected,full_kv=True,need_scores=False,query_valid=valid)
                    delta=fraction*delta+(1-fraction)*dense
                    for key in ('q','kv','qk_av_macs','score_qk'):count[key]+=dense_count[key]
            else:
                delta,scores,count=attention(block.attn,normalized_attention,attention_selected,
                    full_kv=policy.amod_full_kv or query_sparse,dense_mask=policy.mode=='dense_mask',need_scores=need_scores,query_valid=valid)
            if getattr(policy,'depth_bypass','hold')=='light':
                extra,rows=light_update(self.depth_attention[i],normalized_attention,bypass)
                delta=delta+extra;tr['depth_attention_light'][i]=rows
            x=x+block.drop_path(delta);attention_state=x;normalized=block.norm2(x)
            heavy=admitted
            if spatial_active:
                score=previous_scores
                if score is None:score=normalized.detach().float().square().mean(-1)
                heavy=self._spatial(score,admitted,policy.spatial_ratio,policy.structured,h,w,uniform=policy.gate=='uniform')
                if policy.spatial_mask is not None:heavy=policy.spatial_mask.reshape_as(heavy)&admitted
                if policy.route_masks is not None:heavy=policy.route_masks['spatial'][i].to(x.device)&admitted
            light_mask=admitted&~heavy
            tr['ffn_light_masks'].append(light_mask)
            if policy.mode=='dense_mask':
                residual=mlp(block.mlp,normalized)*heavy[...,None];heavy_rows=b*n
                light_rows=0
                if policy.use_light and bool(light_mask.any()):
                    residual=residual+mlp(self.light[i],normalized)*light_mask[...,None];light_rows=b*n
            else:
                flat=normalized.reshape(-1,c);indices=heavy.flatten().nonzero().flatten()
                residual=torch.zeros_like(flat)
                if len(indices):residual=residual.index_copy(0,indices,mlp(block.mlp,flat.index_select(0,indices)).to(flat.dtype))
                heavy_rows=len(indices);indices=light_mask.flatten().nonzero().flatten();light_rows=0
                if policy.use_light and len(indices):
                    residual=residual.index_copy(0,indices,mlp(self.light[i],flat.index_select(0,indices)).to(flat.dtype));light_rows=len(indices)
                residual=residual.reshape_as(x)
            if getattr(policy,'depth_bypass','hold')=='light':
                extra,rows=light_update(self.depth_ffn[i],normalized,bypass)
                residual=residual+extra;tr['depth_ffn_light'][i]=rows
            if operator_diagnostics:
                target_mask=(light_mask|bypass)&valid
                if bool(target_mask.any()):
                    with torch.no_grad():
                        target=block.mlp(normalized[target_mask].detach()).float()
                        actual=residual[target_mask].detach().float()
                        tr['operator_errors'][i+1]=dict(normalized_mse=(actual-target).square().mean()/target.square().mean().clamp_min(1e-6),tokens=int(target_mask.sum()))
            x=x+block.drop_path(residual)
            before_tia=x
            if block.use_adapter:x=block.adapter(x,h,w)
            if state_capture and i+1 in capture_layers:tr['state_taps'][i+1]=dict(attention=attention_state,pre_tia=before_tia,post_tia=x)
            for key in ('q','kv','score_qk','qk_av_macs'):tr[key][i]=count[key]
            tr['heavy_mlp'][i]=heavy_rows;tr['light'][i]=light_rows;tr['tia'][i]=b*n if block.use_adapter else 0
            tr['routing_qk'][i]=count['score_qk'];tr['depth_masks'].append(admitted);tr['spatial_masks'].append(heavy)
            last=torch.where(admitted,last.new_tensor(i+1),last);quality+=heavy
            if scores is not None:previous_scores=scores
            x=coupling(x,i+1)
            if i+1 in capture_layers:captures[i+1]=x
        batch=native_valid.shape[0];a=native_valid.shape[1]
        tr.update(patch_input_shape=list(clips.shape),physical_candidates=a*2,
                  last_heavy_depth=last.reshape(batch,a,p).amax(-1),
                  spatial_quality=quality.reshape(batch,a,p).mean(-1)/depth,
                  score_definition='previous dense attention mean over heads and valid queries; no output score multiplier',
                  score_extra_qk_is_counted=True,mode=policy.mode,depth_schedule=policy.depth_schedule,
                  amod_full_kv=policy.amod_full_kv,layers=depth)
        return x,h,w,tr,captures
