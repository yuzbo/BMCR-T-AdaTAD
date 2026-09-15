"""Actual complete-model FLOPs; latency is reported, never a route gate."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]


def profile_case_model(model,data,repeats=20):
    import torch
    import numpy as np
    from h65.frame.measure import matrix_counter
    from h65.frame.runtime import tensor_json
    counter=matrix_counter();hooks=[]
    detector=model.teacher.model.detector
    modules=[(model.anchor.vit,'backbone_adapter'),(model.engine,'backbone_adapter'),
             (model.anchor.model.scout,'scout'),(model.anchor.model.scout.conditional,'frame_route'),
             (model.decoder,'decoder'),(model.router,'frame_route'),(detector.projection,'projection'),(detector.rpn_head,'head')]
    if detector.with_neck:modules.append((detector.neck,'neck'))
    modules += [(m,'light') for m in model.engine.light]
    for module,name in modules:
        hooks.append(module.register_forward_pre_hook(lambda m,a,name=name:counter.phase.append(name)))
        hooks.append(module.register_forward_hook(counter.leave_scope))
    try:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16),counter:
            _,detail=model.predictions(data['inputs'],data['masks'],data['metas'])
    finally:
        for hook in hooks:hook.remove()
    unknown=counter.unresolved_matrix_ops()
    if unknown:raise RuntimeError('Uncounted matrix operators: '+str(unknown))
    trace=detail['trace'];small={k:v for k,v in trace.items() if k not in ('depth_masks','spatial_masks','query_masks')}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(5):model.predictions(data['inputs'],data['masks'],data['metas'])
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(repeats):
            begin=time.perf_counter();model.predictions(data['inputs'],data['masks'],data['metas']);torch.cuda.synchronize();times.append(time.perf_counter()-begin)
    median=float(np.median(times));p95=float(np.percentile(times,95))
    result=dict(total_macs=sum(counter.macs.values()),matrix_conv_flops=2*sum(counter.macs.values()),
        macs_by_component=dict(counter.macs),uncounted_operator_calls={k:v for k,v in counter.operations.items() if k not in counter.counted},
        unresolved_matrix_ops=unknown,fused_attention=counter.fused_attention,execution=tensor_json(small),
        latency_seconds=times,latency_mean_ms=sum(times)/len(times)*1000,latency_median_ms=median*1000,
        latency_p95_ms=p95*1000,latency_samples=len(times),peak_gib=torch.cuda.max_memory_allocated()/2**30,
        scope='batch1 GPU-resident RGB to pre-NMS output; includes scout/route/encoder/decoder/head; excludes decode and NMS',
        precision='BF16 matmul/conv with FP32 original detector',latency_is_decision_gate=False,
        input_metadata=tensor_json(data['metas']),valid_candidates=int(data['masks'].sum()))
    return result,trace


def profile_windows(model,samples,out,repeats=20):
    import numpy as np
    from h65.full.runtime import to_gpu
    from h65.frame.runtime import json_write
    cases={}
    for name in ('full','partial','short'):
        if name not in samples:continue
        index,sample=samples[name];result,trace=profile_case_model(model,to_gpu(sample),repeats)
        cases[name]=dict(window_index=index,**result)
        arrays={}
        for key in ('depth_masks','spatial_masks','query_masks'):
            if key in trace:arrays[key]=np.stack([x.detach().cpu().numpy() for x in trace[key]])
        if arrays:np.savez_compressed(out/f'execution_{name}.npz',**arrays)
    if 'full' not in cases:raise RuntimeError('A full768 window is required for primary compute comparison')
    result=dict(primary_case='full',cases=cases,**cases['full']);json_write(out/'profile.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--checkpoint')
    parser.add_argument('--resources',default=str(ROOT/'research/frame/resources.local.json'));parser.add_argument('--cases',default='full,partial,short')
    parser.add_argument('--output');parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    if args.dry_run:print(json.dumps(dict(config=args.config,checkpoint=args.checkpoint,cases=args.cases,gpu_execution=False)));return
    import torch
    from torch.utils.data import DataLoader
    from opentad.datasets.builder import build_dataset,collate
    from h65.full.runtime import initialize_gpu,seed_all
    from h65.frame.runtime import read_config,read_resources,data_config,json_write,EXP
    from h65.frame.model import FrameModel
    from h65.frame.contracts import EnginePolicy
    from tools.full_eval import profile_case
    hardware=initialize_gpu();cfg=read_config(args.config);seed_all(cfg['seed']);mc=data_config(cfg['backbone'])
    model=FrameModel(mc.model,cfg,read_resources(args.resources)).cuda().eval();epoch=0
    if args.checkpoint:
        payload=torch.load(args.checkpoint,map_location='cpu')
        if payload['metadata']['config']!=cfg:raise ValueError('Profile checkpoint/config mismatch')
        model.load_learned(payload['ema']);epoch=payload['completed_epochs']
        if 'runtime_policy' in payload:model.policy=EnginePolicy(**payload['runtime_policy'])
    out=Path(args.output) if args.output else EXP/'runs'/f'{cfg["id"]}_profile_{epoch:02}';out.mkdir(parents=True,exist_ok=True)
    dataset=build_dataset(mc.dataset.test);samples={}
    for i,data in enumerate(DataLoader(dataset,batch_size=1,num_workers=2,collate_fn=collate)):
        samples.setdefault(profile_case(data),(i,data))
        if len(samples)==3:break
    result=profile_windows(model,samples,out)
    json_write(out/'completed.json',dict(**hardware,config_id=cfg['id'],checkpoint=args.checkpoint,epoch=epoch,
        gflops=result['matrix_conv_flops']/1e9,latency_ms=result['latency_mean_ms'],cases=list(result['cases']),
        measured_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),not_a_full_test_result=True))


if __name__=='__main__':main()
