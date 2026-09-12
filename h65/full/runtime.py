"""Shared runtime, resource config, complete epoch sampling, and resumable EMA."""
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Sampler, DataLoader
from mmengine.config import Config
import opentad.datasets
from opentad.datasets.builder import build_dataset, collate
from .model import FormalH65
from .data import LoadFramesWithBoundaryValidity

ROOT = Path(__file__).resolve().parents[2]
RESOURCE = Path(os.environ.get('H65_RESOURCE_ROOT', ROOT / 'resources')).expanduser().resolve()
RUNS = Path(os.environ.get('H65_RUNS_DIR', ROOT / 'runs')).expanduser().resolve()
PRETRAIN = {backbone: Path(os.environ.get(f'H65_PRETRAIN_{backbone.upper()}',
            RESOURCE / f'checkpoints/videomae_{backbone}_k400.pth')).expanduser().resolve() for backbone in ('s', 'b')}
OFFICIAL = {backbone: Path(os.environ.get(f'H65_OFFICIAL_{backbone.upper()}',
            RESOURCE / f'checkpoints/adatad_{backbone}_ema.pth')).expanduser().resolve() for backbone in ('s', 'b')}


def config(backbone):
    cfg = Config.fromfile(str(ROOT / f'upstream/configs/adatad/thumos/e2e_thumos_videomae_{backbone}_768x1_160_adapter.py'))
    ann = str(RESOURCE / 'thumos14/annotations/thumos_14_anno.json')
    for split in ('train', 'val', 'test'):
        data = cfg.dataset[split]
        data.ann_file = ann
        data.class_map = str(RESOURCE / 'thumos14/annotations/category_idx.txt')
        data.data_path = str(RESOURCE / 'thumos14/videos' / ('validation' if split == 'train' else 'test'))
    cfg.evaluation.ground_truth_filename = ann
    for transform in cfg.dataset.train.pipeline:
        if transform.type == 'LoadFrames':
            transform.type = 'LoadFramesWithBoundaryValidity'
        elif transform.type in ('ConvertToTensor', 'Collect'):
            transform['keys'].append('gt_boundary_validity')
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_gpu():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU execution requires our independent Slurm allocation')
    name = torch.cuda.get_device_name(0)
    if '4090' not in name:
        raise RuntimeError(f'expected 4090, got {name}')
    torch.set_num_threads(4)
    seed_all(3407)
    return dict(gpu=name, slurm_job_id=os.environ['SLURM_JOB_ID'], torch=torch.__version__, cuda=torch.version.cuda,
                source_revision=os.environ.get('H65_SOURCE_REVISION','unrecorded'))


def to_gpu(sample):
    return {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else
            [x.cuda(non_blocking=True) for x in value] if key in ('gt_segments', 'gt_labels', 'gt_boundary_validity') else value
            for key, value in sample.items()}


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')
    temp.replace(path)


class EpochSampler(Sampler):
    def __init__(self, dataset, seed=3407):
        self.size, self.seed, self.epoch = len(dataset), seed, 0

    def __len__(self):
        return self.size

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        return iter(order)


def train_loader(cfg, workers=2):
    dataset = build_dataset(cfg.dataset.train)
    if len(dataset) != 200:
        raise RuntimeError(f'expected all200 THUMOS training videos; got {len(dataset)}')
    sampler = EpochSampler(dataset)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=workers, collate_fn=collate,
                        pin_memory=True, persistent_workers=False)
    return loader, sampler


def optimizer_for(model, backbone, phase):
    groups = model.detector.get_optim_groups(dict(weight_decay=.05, lr=1e-4))
    adapter = [p for name, p in model.backbone.named_parameters() if p.requires_grad]
    if any('adapter' not in name and p.requires_grad for name, p in model.backbone.named_parameters()):
        raise RuntimeError('frozen VideoMAE has unexpected trainable parameters')
    groups.append(dict(params=adapter, lr=2e-4 if backbone == 's' else 1e-4, weight_decay=.05))
    scout_groups = {}
    trunk_lr, action_lr, scorer_lr = (5e-5, 1e-4, 5e-5) if phase == 'warm' else (1e-5, 2e-5, 5e-5)
    action_ids = {id(p) for p in model.scout.temporal.encoder.conv_out.parameters()}
    for decoder in model.scout.temporal.decoders:
        action_ids.update(id(p) for p in decoder.conv_out.parameters())
    for name, parameter in model.scout.named_parameters():
        if name.startswith('spatial_stem') or name.startswith('temporal.'):
            # In fixed H65 source only encoder/decoder conv_out classifiers use
            # action_head_lr; decoder attention/MLP layers belong to the trunk.
            lr = action_lr if id(parameter) in action_ids else trunk_lr
        else:
            lr = scorer_lr
        decay = .05 if name.endswith('.weight') and parameter.ndim >= 2 else 0.
        scout_groups.setdefault((lr, decay), []).append(parameter)
    for (lr, decay), parameters in sorted(scout_groups.items()):
        groups.append(dict(params=parameters, lr=lr, weight_decay=decay))
    return torch.optim.AdamW(groups)


class EMA:
    def __init__(self, model, cfg):
        self.module = FormalH65(cfg.model, variant=model.variant).cuda().eval()
        self.module.load_state_dict(model.state_dict(), strict=True)
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, decay=.999):
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            if value.is_floating_point():
                value.mul_(decay).add_(source[name], alpha=1 - decay)
            else:
                value.copy_(source[name])


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def save_checkpoint(path, model, ema, optimizer, scheduler, epoch, updates, metadata):
    payload = dict(state_dict=model.state_dict(), state_dict_ema=ema.module.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), scaler=None, amp='bf16', completed_epochs=epoch,
                   successful_updates=updates, rng=rng_state(), metadata=metadata)
    temporary = Path(path).with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
