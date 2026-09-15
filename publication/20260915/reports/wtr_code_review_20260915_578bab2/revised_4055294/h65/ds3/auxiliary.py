"""Trainable modules are outside the frozen OpenTAD backbone tree."""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


def preview_images(inputs, size=64):
    b, _, c, t, h, w=inputs.shape
    images=inputs[:,0].permute(0,2,1,3,4).reshape(b*t,c,h,w)
    return F.interpolate(images.float(),size=(size,size),mode='bilinear',align_corners=False).reshape(b,t,c,size,size)


class Preview(nn.Module):
    def __init__(self, channels, pretrained=None, hidden=128):
        super().__init__()
        network=mobilenet_v3_small(weights=None)
        if pretrained is not None:
            network.load_state_dict(torch.load(pretrained,map_location='cpu'),strict=True)
        self.visual=network.features
        self.reduce=nn.Conv1d(576,hidden,1)
        self.position=nn.Parameter(torch.randn(1,hidden,8)*.01)
        self.temporal=nn.Sequential(nn.Conv1d(hidden,hidden,3,padding=1,groups=hidden),nn.GELU(),
            nn.Conv1d(hidden,hidden,1),nn.GELU(),nn.Conv1d(hidden,hidden,3,padding=2,dilation=2,groups=hidden),
            nn.GELU(),nn.Conv1d(hidden,hidden,1))
        self.output=nn.Conv1d(hidden,channels,1)
        self.utility=nn.Conv1d(hidden,2,1,bias=False)
        self.register_buffer('mean',torch.tensor([123.675,116.28,103.53])[None,:,None,None])
        self.register_buffer('std',torch.tensor([58.395,57.12,57.375])[None,:,None,None])

    def forward(self, images):
        b,t,c,h,w=images.shape
        visual=self.visual((images.reshape(b*t,c,h,w)-self.mean)/self.std).mean((-1,-2))
        visual=visual.reshape(b,t//2,2,-1).mean(2).transpose(1,2)
        hidden=self.reduce(visual)
        hidden=hidden+self.position.repeat(1,1,hidden.shape[-1]//8)
        hidden=hidden+self.temporal(hidden)
        logits=self.utility(hidden.reshape(b,hidden.shape[1],-1,8).mean(-1)).transpose(1,2)
        return self.output(hidden),logits


class ExitHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden=min(192,channels//2)
        self.network=nn.Sequential(nn.Conv1d(channels,hidden,1),nn.GELU(),
            nn.Conv1d(hidden,hidden,3,padding=1,groups=hidden),nn.GELU(),nn.Conv1d(hidden,channels,1))

    def forward(self, features):
        return features+self.network(features)


class Auxiliaries(nn.Module):
    def __init__(self, channels, mobile_pretrained=None, surrogate_width=32):
        super().__init__()
        self.preview=Preview(channels,mobile_pretrained)
        self.exit8=ExitHead(channels)
        self.surrogates=nn.ModuleDict({str(i):nn.Sequential(nn.Linear(channels,surrogate_width),nn.GELU(),
                                                        nn.Linear(surrogate_width,channels)) for i in range(8,12)})
        self.spatial_scores=nn.ModuleDict({str(i):nn.Linear(channels,1,bias=False) for i in range(8,12)})
