"""Original-time latent readout; cross blocks have no query self-attention."""
import torch
from torch import nn
from .geometry import interpolate_anchors,decoder_metadata

class QueryBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.norm_q=nn.LayerNorm(width);self.norm_m=nn.LayerNorm(width)
        self.cross=nn.MultiheadAttention(width,heads,dropout=0.,batch_first=True)
        self.norm_ffn=nn.LayerNorm(width);self.ffn=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
    def forward(self,q,m,valid):
        memory=self.norm_m(m)
        update,_=self.cross(self.norm_q(q),memory,memory,key_padding_mask=~valid,need_weights=False)
        q=q+update;return q+self.ffn(self.norm_ffn(q))

class TemporalBlock(nn.Module):
    def __init__(self,width):
        super().__init__();self.norm=nn.LayerNorm(width);self.temporal=nn.Conv1d(width,width,3,padding=1,groups=width)
        self.ffn=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
    def forward(self,q):
        q=q+self.temporal(self.norm(q).transpose(1,2)).transpose(1,2)
        return q+self.ffn(self.norm(q))

class FullAxisDecoder(nn.Module):
    def __init__(self,kind,channels,context_dim=96,width=192,layers=2,heads=3,use_provenance=True,use_scout=True):
        super().__init__();self.kind=kind;self.use_provenance=use_provenance;self.use_scout=use_scout
        if kind=='interpolate':return
        if kind not in ('cross','tcn'):raise ValueError(kind)
        self.base_proj=nn.Linear(channels,width);self.context_proj=nn.Linear(context_dim,width);self.query_meta=nn.Linear(8,width)
        if kind=='cross':
            self.memory_proj=nn.Linear(channels,width);self.anchor_meta=nn.Linear(10,width)
            self.blocks=nn.ModuleList([QueryBlock(width,heads) for _ in range(layers)])
        else:self.blocks=nn.ModuleList([TemporalBlock(width) for _ in range(layers)])
        self.head=nn.Linear(width,channels);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,anchors,queries,context):
        base=interpolate_anchors(anchors,queries)
        if self.kind=='interpolate':return base.transpose(1,2)
        am,qm=decoder_metadata(anchors,queries)
        if not self.use_provenance:am=torch.zeros_like(am);qm=torch.zeros_like(qm)
        if not self.use_scout:context=torch.zeros_like(context)
        q=(self.base_proj(base)+self.context_proj(context)+self.query_meta(qm))*queries.valid[...,None]
        if self.kind=='cross':
            memory=self.memory_proj(anchors.features)+self.anchor_meta(am)
            for block in self.blocks:q=block(q,memory,anchors.valid)
        else:
            for block in self.blocks:q=block(q)*queries.valid[...,None]
        return ((base+self.head(q))*queries.valid[...,None]).transpose(1,2)

def build_decoder(kind,channels,context_dim=96,width=192,layers=2,heads=3,use_provenance=True,use_scout=True):
    if kind=='mae':
        from .mae_init import MAELatentDecoder
        return MAELatentDecoder(channels,width,layers,heads,context_dim)
    return FullAxisDecoder(kind,channels,context_dim,width,layers,heads,use_provenance,use_scout)
