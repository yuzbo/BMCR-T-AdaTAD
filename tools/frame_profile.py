"""Actual complete-model FLOPs; latency is reported, never a route gate."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]


def profile_case_model(model,data,repeats=20):
    import torch
    from tools.full_eval import ArithmeticCounter
    from h65.frame.runtime import tensor_json
    class Counter(ArithmeticCounter):
        extra={'aten._scaled_dot_product_efficient_attention.default','aten._scaled_dot_product_attention_math.default'}
        counted=ArithmeticCounter.counted|extra
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            if str(func) not in self.extra:return super().__torch_dispatch__(func,types,args,kwargs)
            result=func(*args,**(kwargs or {}));self.operations[str(func)]+=1
            self.macs[self.phase[-1] if self.phase else 'routing_and_other']+=self.attention_macs(*args[:3])
            self.fused_attention.append(dict(q=list(args[0].shape),k=list(args[1].shape),v=list(args[2].shape),macs=self.attention_macs(*args[:3])))
            return result
    counter=Counter();hooks=[]
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
    ordered=sorted(times);p95=ordered[min(len(ordered)-1,int(.95*(len(ordered)-1)))]
    result=dict(total_macs=sum(counter.macs.values()),matrix_conv_flops=2*sum(counter.macs.values()),
        macs_by_component=dict(counter.macs),uncounted_operator_calls={k:v for k,v in counter.operations.items() if k not in counter.counted},
        unresolved_matrix_ops=unknown,fused_attention=counter.fused_attention,execution=tensor_json(small),
        latency_seconds=times,latency_mean_ms=sum(times)/len(times)*1000,latency_median_ms=ordered[len(ordered)//2]*1000,
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
    from tools.frame_eval import run
    run(args.config,args.resources,args.checkpoint,args.output,profile_only=True)


if __name__=='__main__':main()
