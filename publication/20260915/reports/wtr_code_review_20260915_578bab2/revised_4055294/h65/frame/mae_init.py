"""Reuse VideoMAE decoder blocks, projection, norm and mask token for TAD latents.

This changes token geometry and replaces RGB prediction. It is an initialization
experiment, not an assertion that an RGB decoder is already a TAD latent model.
Source: MCG-NJU/VideoMAE, modeling_pretrain.py and modeling_finetune.py.
"""
import re
import torch
from torch import nn
import torch.nn.functional as F
from .geometry import interpolate_anchors,decoder_metadata

def read_pretraining(path):
    payload=torch.load(path,map_location='cpu')
    state=payload.get('model',payload.get('state_dict',payload))
    state={k.removeprefix('module.'):v for k,v in state.items()}
    ids=sorted({int(m.group(1)) for k in state if (m:=re.match(r'decoder\.blocks\.(\d+)\.',k))})
    if not ids or ids!=list(range(len(ids))):raise ValueError('A complete VideoMAE pretraining decoder is required; encoder-only assets are not accepted')
    weight=state['encoder_to_decoder.weight'];width,channels=weight.shape
    return state,dict(width=width,channels=channels,layers=len(ids),heads=width//64,source=str(path))

class MAEAttention(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.heads=heads;self.qkv=nn.Linear(width,3*width,bias=False)
        self.q_bias=nn.Parameter(torch.zeros(width));self.v_bias=nn.Parameter(torch.zeros(width));self.proj=nn.Linear(width,width)
    def forward(self,x,valid):
        b,n,c=x.shape;bias=torch.cat((self.q_bias,torch.zeros_like(self.v_bias),self.v_bias))
        q,k,v=F.linear(x,self.qkv.weight,bias).reshape(b,n,3,self.heads,c//self.heads).permute(2,0,3,1,4).unbind(0)
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=valid[:,None,None,:],dropout_p=0.)
        return self.proj(y.transpose(1,2).reshape(b,n,c))

class MAEMLP(nn.Module):
    def __init__(self,width):super().__init__();self.fc1=nn.Linear(width,4*width);self.fc2=nn.Linear(4*width,width)
    def forward(self,x):return self.fc2(F.gelu(self.fc1(x)))

class MAEBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.norm1=nn.LayerNorm(width,eps=1e-6);self.attn=MAEAttention(width,heads)
        self.norm2=nn.LayerNorm(width,eps=1e-6);self.mlp=MAEMLP(width)
    def forward(self,x,valid):
        x=x+self.attn(self.norm1(x),valid);return (x+self.mlp(self.norm2(x)))*valid[...,None]

class MAELatentDecoder(nn.Module):
    def __init__(self,channels,width,layers,heads,context_dim=96):
        super().__init__();self.encoder_to_decoder=nn.Linear(channels,width,bias=False);self.mask_token=nn.Parameter(torch.zeros(1,1,width))
        self.decoder=nn.Module();self.decoder.blocks=nn.ModuleList([MAEBlock(width,heads) for _ in range(layers)]);self.decoder.norm=nn.LayerNorm(width,eps=1e-6)
        for module in self.decoder.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.mask_token,std=.02,a=-.02,b=.02)
        self.anchor_meta=nn.Linear(10,width);self.query_meta=nn.Linear(8,width);self.context_proj=nn.Linear(context_dim,width)
        self.head=nn.Linear(width,channels);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.initialization=dict(kind='random_same_architecture',width=width,layers=layers,heads=heads)
    def load_pretraining(self,path):
        source,spec=read_pretraining(path)
        required={k:v for k,v in self.state_dict().items() if k.startswith(('decoder.','encoder_to_decoder.')) or k=='mask_token'}
        missing=[k for k in required if k not in source];bad=[k for k,v in required.items() if k in source and source[k].shape!=v.shape]
        extra_blocks=[k for k in source if k.startswith('decoder.blocks.') and k not in required]
        if missing or bad or extra_blocks:raise ValueError(dict(missing=missing,shape_mismatch=bad,unsupported_decoder_keys=extra_blocks))
        with torch.no_grad():
            current=self.state_dict()
            for k in required:current[k].copy_(source[k])
        self.initialization=dict(kind='verified_videomae_block_initialization',**spec,copied_keys=sorted(required),
            replaced=['RGB head','fixed patch-grid positions'],new=['physical source/query metadata','scout context','zero latent residual head'])
        return self.initialization
    def forward(self,anchors,queries,context):
        base=interpolate_anchors(anchors,queries);am,qm=decoder_metadata(anchors,queries)
        memory=self.encoder_to_decoder(anchors.features)+self.anchor_meta(am)
        q=self.encoder_to_decoder(base)+self.mask_token+self.query_meta(qm)+self.context_proj(context)
        x=torch.cat((memory,q),1);valid=torch.cat((anchors.valid,queries.valid),1)
        for block in self.decoder.blocks:x=block(x,valid)
        q=self.decoder.norm(x[:,memory.shape[1]:])
        return ((base+self.head(q))*queries.valid[...,None]).transpose(1,2)
