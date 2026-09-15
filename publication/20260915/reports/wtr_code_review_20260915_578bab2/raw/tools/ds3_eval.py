"""Complete native-time evaluation and measured executed arithmetic/latency."""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from opentad.datasets.builder import build_dataset,collate
from opentad.cores.test_engine import gather_ddp_results
from opentad.evaluations import build_evaluator
from h65.ds3.model import DS3
from h65.ds3.routes import POLICIES
from h65.ds3.runtime import RUNS,OFFICIAL,RECIPE,config,initialize,load_aux,json_write,to_gpu
from tools.full_eval import ArithmeticCounter,profile_case


def output_name(backbone,policy,epoch=0):
    return f'{backbone}_{policy}'+(f'_epoch_{epoch:02}' if POLICIES[policy].kind=='aux' else '')


def latency(function):
    for _ in range(5):function()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();values=[]
    for _ in range(20):
        began=time.perf_counter();function();torch.cuda.synchronize();values.append(time.perf_counter()-began)
    return dict(samples_seconds=values,mean_seconds=statistics.mean(values),p50_seconds=statistics.median(values),
                p95_seconds=sorted(values)[18],peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                p95_definition='nearest-rank on20 retained samples')


def profile(model,sample):
    counter=ArithmeticCounter();hooks=[];vit=model.teacher.vit;head=model.teacher.detector
    measured=dict(attention_clips=[0]*12,heavy_mlp_tokens=[0]*12,patch_inputs=[])
    modules=[(vit.patch_embed,'patch_embed'),(head.projection,'projection'),(head.rpn_head,'detection_head')]
    if head.with_neck:modules.append((head.neck,'neck'))
    for index,block in enumerate(vit.blocks):
        modules.extend([(block.attn,'attention'),(block.mlp,'heavy_mlp'),(block.adapter,'TIA')])
        def attention(module,args,index=index):measured['attention_clips'][index]+=len(args[0])
        def mlp(module,args,index=index):measured['heavy_mlp_tokens'][index]+=args[0].numel()//args[0].shape[-1]
        hooks.extend([block.attn.register_forward_pre_hook(attention),block.mlp.register_forward_pre_hook(mlp)])
    def patch(module,args):measured['patch_inputs'].append(list(args[0].shape))
    hooks.append(vit.patch_embed.register_forward_pre_hook(patch))
    if model.aux is not None:
        modules.extend([(model.aux.preview,'preview'),(model.aux.exit8,'exit8')])
        modules.extend((module,'surrogate_mlp') for module in model.aux.surrogates.values())
        modules.extend((module,'spatial_decision') for module in model.aux.spatial_scores.values())
    for module,name in modules:
        def enter(module,args,name=name):counter.phase.append(name)
        hooks.extend([module.register_forward_pre_hook(enter),module.register_forward_hook(counter.leave_scope)])
    def forward():return model.predictions(sample['inputs'],sample['masks'],sample['metas'])
    try:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16),counter:_,trace=forward()
    finally:
        for hook in hooks:hook.remove()
    if measured['attention_clips']!=trace['attention_clips'] or measured['heavy_mlp_tokens']!=trace['heavy_mlp_tokens']:
        raise RuntimeError('declared route differs from actual module input counts')
    if counter.unresolved_matrix_ops():raise RuntimeError(f'unresolved matrix operators: {counter.unresolved_matrix_ops()}')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        total=latency(forward)
        clips=model.teacher.prepare_clips(sample['inputs'])
        # Prepared-route backbone time includes gather/scatter and surrogate MLP;
        # preview, routing, preprocessing, native reconstruction and head excluded.
        if model.teacher.scope=='global':backbone_call=lambda:model.teacher.vit(clips)
        else:backbone_call=lambda:model.engine.execute(clips,trace['depths'],POLICIES[model.policy].spatial_ratio,model.aux)
        backbone=latency(backbone_call)
    return dict(total_macs=sum(counter.macs.values()),matrix_conv_flops=2*sum(counter.macs.values()),
                macs_by_component=dict(counter.macs),flops_convention='2MAC matrix/conv incl actual QK/AV; excludes elementwise ops',
                other_operator_calls={k:v for k,v in counter.operations.items() if k not in counter.counted},
                unresolved_matrix_ops=[],fused_attention=counter.fused_attention,measured_execution=measured,
                route_depths=trace['depths'].cpu().tolist(),native_source_kind=trace['source_kind'].cpu().tolist(),
                tubelet_centers=trace['tubelet_centers'].cpu().tolist(),valid_input_candidates=int(sample['masks'].sum()),
                input_shape=list(sample['inputs'].shape),video_name=sample['metas'][0]['video_name'],
                window_start_frame=int(sample['metas'][0].get('window_start_frame',0)),model_timing=total,backbone_timing=backbone,
                precision='BF16 autocast feature modules, FP32 detector',
                model_timing_scope='batch1 GPU-resident raw input; preview, routing, backbone, native assembly and detector; excludes decode/NMS',
                backbone_timing_scope='normalized clips and fixed route prepared; packed backbone plus sparse MLP replacements only')


def profile_windows(model,samples):
    cases={}
    for case in ('full','partial','short'):
        index,sample=samples[case]
        cases[case]=dict(**profile(model,to_gpu(sample)),window_index=index)
    return dict(primary_case='full',cases=cases,sample_selection='first full768,partial385..767,short<=384 in fixed test order')


def duration_recall(result,database):
    bins={name:dict(total=0,matched=0) for name in ('<=2s','2..8s','>8s')}
    for name,proposals in result.items():
        top=sorted(proposals,key=lambda x:x['score'],reverse=True)[:100]
        for annotation in database[name]['annotations']:
            if annotation['label']=='Ambiguous':continue
            a,b=annotation['segment'];length=b-a
            group=bins['<=2s' if length<=2 else '2..8s' if length<=8 else '>8s'];group['total']+=1
            for proposal in top:
                c,d=proposal['segment'];intersection=max(0.,min(b,d)-max(a,c))
                if proposal['label']==annotation['label'] and intersection/max(length+d-c-intersection,1e-8)>=.5:
                    group['matched']+=1;break
    return dict(definition='class-matched recall at tIoU0.5 using top100 post-NMS detections/video; fixed duration bins',
                bins={k:dict(**v,recall=v['matched']/max(v['total'],1)) for k,v in bins.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['s','b'],required=True)
    p.add_argument('--policy',choices=list(POLICIES),required=True);p.add_argument('--epoch',type=int,default=80)
    p.add_argument('--profile-only',action='store_true');p.add_argument('--metrics-only',action='store_true');a=p.parse_args()
    hardware=initialize();cfg=config(a.backbone);spec=POLICIES[a.policy]
    out=RUNS/output_name(a.backbone,a.policy,a.epoch);out.mkdir(parents=True,exist_ok=True)
    dataset=build_dataset(cfg.dataset.test);db=json.loads(Path(cfg.dataset.test.ann_file).read_text())['database']
    expected={name for name,info in db.items() if info['subset']=='validation'}
    if len(expected)!=211 or {row[0] for row in dataset.data_list}!=expected or len(dataset)!=792:
        raise RuntimeError('evaluation requires exactly all211 videos /792 windows')
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2,pin_memory=True,collate_fn=collate)
    model=DS3(cfg.model,str(OFFICIAL[a.backbone]),with_aux=spec.kind=='aux',scope='global' if spec.kind=='global' else 'local')
    initialization=dict(teacher=model.teacher.provenance,recipe=RECIPE)
    if spec.kind=='aux':
        path=RUNS/f'{a.backbone}_d1/epoch_{a.epoch:02}.pth';payload=load_aux(model,path)
        if payload['completed_epochs']!=a.epoch or payload['successful_updates']!=a.epoch*100:
            raise RuntimeError('EMA must match the requested full epoch')
        initialization.update(aux_checkpoint=str(path),aux_state='aux_ema',completed_epochs=a.epoch,
                              successful_updates=payload['successful_updates'])
    model.cuda().eval();model.policy=a.policy;samples={}
    if a.profile_only:
        metrics=json.loads((out/'metrics.json').read_text())
        if metrics['initialization']!=initialization:raise RuntimeError('profile checkpoint differs from saved accuracy')
        for index,sample in enumerate(loader):
            samples.setdefault(profile_case(sample),(index,sample))
            if len(samples)==3:break
        json_write(out/'profile.json',dict(**hardware,**profile_windows(model,samples)))
        return
    result={};cfg.post_processing.sliding_window=True
    torch.cuda.synchronize();began=time.perf_counter()
    for index,cpu_sample in enumerate(loader):
        samples.setdefault(profile_case(cpu_sample),(index,cpu_sample));sample=to_gpu(cpu_sample)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            window=model(**sample,post_cfg=cfg.post_processing,ext_cls=dataset.class_map)
        for name,proposals in window.items():result.setdefault(name,[]).extend(proposals)
        if index%100==0:
            json_write(out/'progress.json',dict(windows=index+1,total_windows=792,videos=len(result)))
            print(f'{a.backbone} {a.policy}: {index+1}/792 windows',flush=True)
    if set(result)!=expected:raise RuntimeError('missing test videos')
    rendezvous=out/f'dist_{os.environ["SLURM_JOB_ID"]}'
    dist.init_process_group('gloo',init_method=rendezvous.as_uri(),rank=0,world_size=1)
    try:result=gather_ddp_results(1,result,cfg.post_processing)
    finally:dist.destroy_process_group()
    torch.cuda.synchronize();e2e=time.perf_counter()-began
    prediction=dict(results=result);json_write(out/'result_detection.json',prediction)
    scores=build_evaluator(dict(prediction_filename=prediction,**cfg.evaluation)).evaluate()
    metrics=dict(**hardware,backbone=a.backbone,policy=a.policy,recipe=RECIPE,initialization=initialization,
                 metrics=scores,test_videos=len(result),test_windows=792,e2e_seconds=e2e,
                 e2e_scope='full dataloader iteration including decode, CPU transforms, transfer, model, window/video NMS; excludes AP and prediction file writing; filesystem-cache state uncontrolled',
                 duration_recall=duration_recall(result,db),
                 selection=('T24A peak over full-test EMA epochs5..80 every5; other policies use the same selected auxiliary checkpoint'
                            if spec.kind=='aux' else 'fixed provided teacher; no new training or checkpoint selection'))
    json_write(out/'metrics.json',metrics);print(scores,flush=True)
    if not a.metrics_only:json_write(out/'profile.json',dict(**hardware,**profile_windows(model,samples)))
    json_write(out/'completed.json',dict(**metrics,profile_complete=not a.metrics_only))


if __name__=='__main__':main()
