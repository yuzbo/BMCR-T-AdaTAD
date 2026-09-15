"""DS3 resource/data configuration and the predeclared80-epoch aux trajectory."""
import copy
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from opentad.datasets.builder import build_dataset, collate
from h65.full.runtime import config as base_config, EpochSampler, seed_all, json_write, to_gpu, rng_state, restore_rng

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'ds3_20260912'
RUNS=Path(os.environ.get('DS3_RUNS_DIR',EXP/'runs')).resolve()
RESOURCES=Path(os.environ.get('H65_RESOURCE_ROOT',ROOT/'resources')).resolve()
OFFICIAL={b:RESOURCES/f'checkpoints/adatad_{b}_ema.pth' for b in ('s','b')}
MOBILE=RESOURCES/'checkpoints/mobilenet_v3_small.pth'
RECIPE='ds3_ltia8_d1_80_v1'
EPOCHS=80


def config(backbone):
    cfg=base_config(backbone)
    keys=['video_name','data_path','fps','duration','snippet_stride','window_start_frame','resize_length',
          'window_size','offset_frames','frame_inds']
    for split in ('train','val','test'):
        for transform in cfg.dataset[split].pipeline:
            if transform.type=='Collect':transform['meta_keys']=keys
    return cfg


def initialize():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU execution needs our Slurm allocation')
    gpu=torch.cuda.get_device_name(0)
    if '4090' not in gpu:raise RuntimeError(f'expected4090, got{gpu}')
    seed_all(3407);torch.set_num_threads(4)
    return dict(gpu=gpu,slurm_job_id=os.environ['SLURM_JOB_ID'],torch=torch.__version__,cuda=torch.version.cuda,seed=3407)


def train_loader(cfg, workers=2):
    dataset=build_dataset(cfg.dataset.train)
    if len(dataset)!=200:raise RuntimeError('DS3 training requires all200 videos')
    sampler=EpochSampler(dataset,3407)
    generator=torch.Generator()
    loader=DataLoader(dataset,batch_size=2,sampler=sampler,num_workers=workers,collate_fn=collate,
                      pin_memory=True,generator=generator,persistent_workers=False)
    return loader,sampler,generator


def optimizer(aux):
    decay=[];no_decay=[]
    for parameter in aux.parameters():
        if not parameter.requires_grad:raise RuntimeError('an auxiliary parameter was unintentionally frozen')
        (decay if parameter.ndim>=2 else no_decay).append(parameter)
    return torch.optim.AdamW([dict(params=decay,weight_decay=.05),dict(params=no_decay,weight_decay=0.)],lr=1e-4)


def scheduler(optim):
    import math
    def factor(step):
        if step<500:return .1+.9*step/500
        return .5*(1+math.cos(math.pi*min((step-500)/7500,1)))
    return torch.optim.lr_scheduler.LambdaLR(optim,factor)


class AuxEMA:
    def __init__(self, aux):self.module=copy.deepcopy(aux).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, aux):
        source=aux.state_dict()
        for name,value in self.module.state_dict().items():
            if value.is_floating_point():value.mul_(.999).add_(source[name],alpha=.001)
            else:value.copy_(source[name])


def save_checkpoint(path, model, ema, optim, schedule, epochs, updates, metadata):
    value=dict(aux=model.aux.state_dict(),aux_ema=ema.module.state_dict(),optimizer=optim.state_dict(),
               scheduler=schedule.state_dict(),completed_epochs=epochs,successful_updates=updates,
               rng=rng_state(),metadata=metadata)
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def load_aux(model, path, ema=True):
    value=torch.load(path,map_location='cpu')
    if value['metadata']['recipe']!=RECIPE or value['metadata']['teacher']['runtime_tia_size']!=8:
        raise RuntimeError('aux checkpoint belongs to a different DS3 recipe')
    if value['metadata']['channels']!=model.teacher.vit.embed_dims:raise RuntimeError('wrong backbone checkpoint')
    model.aux.load_state_dict(value['aux_ema' if ema else 'aux'],strict=True)
    return value
