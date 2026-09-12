"""Independent FPW recipes/resources. Dry-run configuration requires no CUDA."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'research/frame'
RECIPE='fpw_frame_writeback_amod_v1'


def read_config(path):
    path=Path(path)
    if path.suffix=='.json':cfg=json.loads(path.read_text())
    else:
        import runpy
        cfg=runpy.run_path(str(path))['config']
    if cfg.get('temporal_unit','candidate_frame')!='candidate_frame':raise ValueError('Original-clip temporal routing is retired')
    if cfg.get('budget',384)%16:raise ValueError('VideoMAE computational packing requires16-aligned frame budget')
    cfg.setdefault('recipe',RECIPE);cfg.setdefault('seed',3407);cfg.setdefault('budget',384)
    cfg.setdefault('epochs',20);cfg.setdefault('batch_size',1);cfg.setdefault('accumulate',2)
    cfg.setdefault('ema_decay',.99);cfg.setdefault('selector','anchor');cfg.setdefault('engine',{})
    cfg.setdefault('loss',dict(gt_weight=1.,feature_weight=1.));cfg.setdefault('train_adapters',False)
    cfg.setdefault('decoder',dict(kind='cross',width=192,layers=2,heads=3,use_provenance=True,use_scout=True))
    return cfg


def data_config(backbone):
    from h65.full.runtime import config
    cfg=config(backbone)
    keys=['video_name','data_path','fps','duration','snippet_stride','window_start_frame','resize_length','window_size','offset_frames','frame_inds']
    for split in ('train','val','test'):
        for transform in cfg.dataset[split].pipeline:
            if transform.type=='Collect':transform['meta_keys']=keys
    return cfg


def read_resources(path):
    data=json.loads(Path(path).read_text())
    for b in ('s','b'):
        for value in (data['anchors'][b]['checkpoint'],data['official'][b]):
            if not Path(value).is_file():raise FileNotFoundError(value)
    return data


def dry_description(config,resources):
    return dict(config=config,resources_file=str(resources),gpu_execution=False,
                protocol='200train/211test792windows;candidate-frame selection;matrix-conv FLOPs + best meanmAP primary',
                intended_optimizer_updates=config['epochs']*100,latency_is_decision_gate=False)


def json_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def convert(x):
        if hasattr(x,'detach'):return x.detach().cpu().tolist()
        if hasattr(x,'tolist'):return x.tolist()
        return str(x)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=convert)+'\n');tmp.replace(path)


def tensor_json(value):
    import torch
    if torch.is_tensor(value):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:tensor_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [tensor_json(v) for v in value]
    if hasattr(value,'tolist'):return value.tolist()
    return value
