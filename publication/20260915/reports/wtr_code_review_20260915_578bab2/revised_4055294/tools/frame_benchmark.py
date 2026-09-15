"""Same-allocation interleaved measurements; latency is never an acceptance gate."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['s','b'],required=True);p.add_argument('--epoch',type=int,default=20);p.add_argument('--output',required=True);a=p.parse_args()
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from opentad.datasets.builder import build_dataset,collate
    from h65.frame.runtime import EXP,read_config,read_resources,data_config,json_write
    from h65.full.runtime import initialize_gpu,to_gpu
    from h65.frame.model import FrameModel
    from h65.frame.contracts import EnginePolicy
    from tools.full_eval import profile_case
    from tools.frame_profile import profile_case_model
    hardware=initialize_gpu();resources=read_resources(EXP/'resources.local.json');mc=data_config(a.backbone);models={}
    for name in ('R01_interpolate','R03_cross','D02_amod50','J01_joint'):
        ident=f'{name}_{a.backbone}';cfg=read_config(ROOT/f'configs/frame/{ident}.json');model=FrameModel(mc.model,cfg,resources).cuda().eval()
        if name!='R01_interpolate':
            payload=torch.load(EXP/f'runs/{ident}/epoch_{a.epoch:02}.pth',map_location='cpu')
            if payload['metadata']['config']!=cfg:raise ValueError('Benchmark checkpoint/config mismatch')
            model.load_learned(payload['ema'])
            if 'runtime_policy' in payload:model.policy=EnginePolicy(**payload['runtime_policy'])
        models[ident]=model
    dataset=build_dataset(mc.dataset.test);samples={}
    for i,data in enumerate(DataLoader(dataset,batch_size=1,num_workers=2,collate_fn=collate)):
        samples.setdefault(profile_case(data),(i,data))
        if len(samples)==3:break
    result=dict(**hardware,node=os.uname().nodename,allocated_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),epoch=a.epoch,cases={},
        latency_is_decision_gate=False,scope='same allocation; GPU-resident RGB to pre-NMS; all comparison models resident',
        cache_scope='first measured model forward and warmed interleaving; filesystem/video cache is not flushed or called cold')
    names=list(models)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for case,(index,cpu) in samples.items():
            data=to_gpu(cpu);first={};timings={name:[] for name in names}
            for name in names:
                torch.cuda.synchronize();start=time.perf_counter();models[name].predictions(data['inputs'],data['masks'],data['metas']);torch.cuda.synchronize();first[name]=1000*(time.perf_counter()-start)
            for _ in range(5):
                for name in names:models[name].predictions(data['inputs'],data['masks'],data['metas'])
            for repeat in range(20):
                order=names[repeat%len(names):]+names[:repeat%len(names)]
                for name in order:
                    torch.cuda.synchronize();start=time.perf_counter();models[name].predictions(data['inputs'],data['masks'],data['metas']);torch.cuda.synchronize();timings[name].append(1000*(time.perf_counter()-start))
            result['cases'][case]=dict(window_index=index,valid_candidates=int(data['masks'].sum()),models={})
            for name in names:
                profile,_=profile_case_model(models[name],data,repeats=1)
                result['cases'][case]['models'][name]=dict(first_forward_ms=first[name],raw_warm_interleaved_ms=timings[name],
                    median_ms=float(np.median(timings[name])),mean_ms=float(np.mean(timings[name])),p95_ms=float(np.percentile(timings[name],95)),
                    gflops=profile['matrix_conv_flops']/1e9,macs_by_component=profile['macs_by_component'])
    if set(result['cases'])!={'full','partial','short'}:raise ValueError('All three real cases required')
    json_write(Path(a.output)/'completed.json',result)

if __name__=='__main__':main()
