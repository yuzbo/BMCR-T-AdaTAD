"""Residual cross readout with optional zero-initialized multi-depth memory."""
import torch
from torch import nn
from h65.frame.decoder import FullAxisDecoder
from h65.frame.geometry import decoder_metadata
from .geometry import interpolate_anchors
from h65.frame.mae_init import MAELatentDecoder


class PaperDecoder(FullAxisDecoder):
    def __init__(self,channels,encoder_depth=12,multidepth=True,kind='cross',use_provenance=True,use_scout=True):
        width=240 if kind=='tcn' else 192
        super().__init__(kind,channels,width=width,layers=2,heads=3,
                         use_provenance=use_provenance,use_scout=use_scout)
        self.encoder_depth=encoder_depth;self.multidepth=multidepth and kind=='cross'
        self.levels=tuple(sorted({encoder_depth//2,3*encoder_depth//4,encoder_depth}))
        if self.multidepth:
            self.level_proj=nn.ModuleDict({str(level):nn.Linear(channels,width) for level in self.levels[:-1]})
            self.level_gate=nn.Parameter(torch.zeros(len(self.blocks),len(self.levels)-1))

    def load_recovery(self,path):
        payload=torch.load(path,map_location='cpu');state=payload['ema']
        subset={k.removeprefix('decoder.'):v for k,v in state.items() if k.startswith('decoder.')}
        expected={k for k in self.state_dict() if not k.startswith(('level_proj.','level_gate'))}
        if set(subset)!=expected:raise ValueError('R03 initialization must match the exact baseline decoder')
        self.load_state_dict(subset,strict=False)

    def forward(self,anchors,queries,context,layer_features=None):
        base=interpolate_anchors(anchors,queries)
        if self.kind=='interpolate':return base.transpose(1,2)
        am,qm=decoder_metadata(anchors,queries)
        am=am.clone();am[...,8]=anchors.last_heavy_depth/self.encoder_depth
        if not self.use_provenance:am=torch.zeros_like(am);qm=torch.zeros_like(qm)
        if not self.use_scout:context=torch.zeros_like(context)
        q=(self.base_proj(base)+self.context_proj(context)+self.query_meta(qm))*queries.valid[...,None]
        if self.kind=='cross':
            memory=self.memory_proj(anchors.features)+self.anchor_meta(am)
            for i,block in enumerate(self.blocks):
                current=memory
                if self.multidepth:
                    for j,level in enumerate(self.levels[:-1]):
                        current=current+self.level_gate[i,j]*self.level_proj[str(level)](layer_features[level])
                q=block(q,current,anchors.valid)
        else:
            for block in self.blocks:q=block(q)*queries.valid[...,None]
        return ((base+self.head(q))*queries.valid[...,None]).transpose(1,2)


class PaperMAEDecoder(MAELatentDecoder):
    def __init__(self,channels,depth,asset,pretrained):
        super().__init__(channels,asset['width'],asset['layers'],asset['heads'])
        self.encoder_depth=depth
        if pretrained:self.load_pretraining(asset['checkpoint'] if 'checkpoint' in asset else asset['source'])

    def forward(self,anchors,queries,context,layer_features=None):
        base=interpolate_anchors(anchors,queries);am,qm=decoder_metadata(anchors,queries)
        am=am.clone();am[...,8]=anchors.last_heavy_depth/self.encoder_depth
        memory=self.encoder_to_decoder(anchors.features)+self.anchor_meta(am)
        q=self.encoder_to_decoder(base)+self.mask_token+self.query_meta(qm)+self.context_proj(context)
        x=torch.cat((memory,q),1);valid=torch.cat((anchors.valid,queries.valid),1)
        for block in self.decoder.blocks:x=block(x,valid)
        q=self.decoder.norm(x[:,memory.shape[1]:])
        return ((base+self.head(q))*queries.valid[...,None]).transpose(1,2)
