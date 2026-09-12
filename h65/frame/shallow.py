"""Separate P01 variant: extra full-timeline, original-pair shallow observation."""
import torch
from torch import nn
import torch.nn.functional as F

class FullTimelineStem(nn.Module):
    def __init__(self,channels,resolution=80):
        super().__init__();self.resolution=resolution;self.project=nn.Linear(channels,96)
        nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)
    def forward(self,reader,inputs,masks):
        with torch.no_grad():
            wrapper=reader.model.backbone
            frames,_=wrapper.model.data_preprocessor.preprocess(wrapper.tensor_to_list(inputs),None,False)
            b,n,c,t,h,w=frames.shape
            images=frames[:,0].permute(0,2,1,3,4).reshape(b*t,c,h,w)
            images=F.interpolate(images,size=(self.resolution,self.resolution),mode='bilinear',align_corners=False)
            frames=images.reshape(b,t,c,self.resolution,self.resolution).permute(0,2,1,3,4)
            clips=frames.reshape(b,c,t//16,16,self.resolution,self.resolution).permute(0,2,1,3,4,5).reshape(b*t//16,c,16,self.resolution,self.resolution)
            tokens,_=reader.vit.patch_embed(clips);pooled=tokens.reshape(b,t//16,8,-1,reader.vit.embed_dims).mean(3).reshape(b,t//2,-1)
        valid=masks.reshape(b,t//2,2).any(-1)
        return self.project(pooled)*valid[...,None]
