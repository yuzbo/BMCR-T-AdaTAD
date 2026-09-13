"""Exact epoch/cursor resume and trainable-state EMA for long dataset courses."""
from contextlib import contextmanager
import random
import numpy as np
import torch
from torch.utils.data import Dataset,Sampler


class EpochDataset(Dataset):
    def __init__(self,dataset,seed):self.dataset=dataset;self.seed=seed;self.epoch=0
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        seed=(self.seed+self.epoch*100003+index*97)%(2**32)
        py=random.getstate();np_state=np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
                return self.dataset[index]
        finally:random.setstate(py);np.random.set_state(np_state)


class EpochSampler(Sampler):
    def __init__(self,dataset,seed,epoch=0,start=0):self.size=len(dataset);self.seed=seed;self.epoch=epoch;self.start=start
    def __len__(self):return self.size-self.start
    def __iter__(self):
        return iter(torch.randperm(self.size,generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()[self.start:])


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


class EMAState:
    def __init__(self,model,decay):self.decay=decay;self.values={k:v.detach().clone() for k,v in model.learned_state().items()}
    @torch.no_grad()
    def update(self,model):
        exact=('cost_gflops','reference_gflops','fixed_nonencoder_macs','sigma_calibration')
        for key,value in model.learned_state().items():
            if not value.is_floating_point() or key.endswith(exact):self.values[key].copy_(value)
            else:self.values[key].mul_(self.decay).add_(value.detach(),alpha=1-self.decay)
    @contextmanager
    def apply(self,model,values=None):
        before={k:v.detach().clone() for k,v in model.learned_state().items()}
        model.load_learned(values or self.values)
        try:yield
        finally:model.load_learned(before)


def optimizer_groups(model,cfg):
    action_ids=set()
    scout=model.encoder.scout
    for module in [scout.temporal.encoder.conv_out,*[x.conv_out for x in scout.temporal.decoders]]:
        action_ids.update(id(p) for p in module.parameters())
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if name.startswith('encoder.backbone.'):
            kind='adapter' if 'adapter' in name else 'backbone'
        elif name.startswith('encoder.scout.'):
            kind='scout_action' if id(p) in action_ids else 'scout'
        elif name.startswith('readout.'):
            kind='head_offset' if any(x in name for x in ('reference_points','sampling_offsets')) else 'head'
        else:kind='new'
        rates=dict(new=cfg['lr'],head=cfg['head_lr'],head_offset=cfg['head_lr']*.1,
                   adapter=cfg['adapter_lr'],backbone=cfg['backbone_lr'],scout=cfg['scout_lr'],scout_action=2*cfg['scout_lr'])
        decay=0. if p.ndim<=1 or name.endswith('bias') else cfg['weight_decay']
        key=(kind,decay);groups.setdefault(key,dict(params=[],lr=rates[kind],weight_decay=decay,name=kind))['params'].append(p)
    return list(groups.values())


def cpu_state(values):return {k:v.detach().cpu() for k,v in values.items()}
