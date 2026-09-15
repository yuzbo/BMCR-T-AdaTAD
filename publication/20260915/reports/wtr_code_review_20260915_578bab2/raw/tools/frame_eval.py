"""Complete211/792 original-axis evaluation for frame write-back routes."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
from h65.frame.runtime import read_config,EXP,json_write


def run(config_path,resources_path,checkpoint_path=None,output=None,profile_only=False,state_key='ema',selector=None,policy_override=None,disable_router=False,budget=None):
    import torch
    from dataclasses import asdict
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from opentad.datasets.builder import build_dataset,collate
    from opentad.evaluations import build_evaluator
    from opentad.cores.test_engine import gather_ddp_results
    from h65.full.runtime import initialize_gpu,to_gpu
    from h65.frame.runtime import data_config,read_resources,tensor_json
    from h65.frame.model import FrameModel
    from h65.frame.contracts import EnginePolicy
    from tools.full_eval import profile_case
    from tools.frame_profile import profile_windows
    hardware=initialize_gpu();cfg=read_config(config_path);resources=read_resources(resources_path)
    model_cfg=data_config(cfg['backbone']);model=FrameModel(model_cfg.model,cfg,resources).cuda().eval()
    epoch=0;updates=0
    if checkpoint_path:
        payload=torch.load(checkpoint_path,map_location='cpu')
        if payload['metadata']['config']!=cfg:raise RuntimeError('Evaluation recipe/checkpoint mismatch')
        model.load_learned(payload[state_key]);epoch=payload['completed_epochs'];updates=payload['successful_updates']
        if 'runtime_policy' in payload:model.policy=EnginePolicy(**payload['runtime_policy'])
        del payload
    if selector:model.config['selector']=selector
    if disable_router:model.config['router']=False
    if budget is not None:model.config['budget']=budget
    if policy_override:
        from dataclasses import replace
        model.policy=replace(model.policy,**policy_override)
    out=(Path(output) if output else EXP/'runs'/f'{cfg["id"]}_eval_{epoch:02}_{state_key}').resolve()
    out.mkdir(parents=True,exist_ok=True)
    dataset=build_dataset(model_cfg.dataset.test);db=json.loads(Path(model_cfg.dataset.test.ann_file).read_text())['database']
    expected={name for name,value in db.items() if value['subset']=='validation'}
    if {x[0] for x in dataset.data_list}!=expected or len(expected)!=211 or len(dataset)!=792:raise RuntimeError('Full211-video/792-window test required')
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2,collate_fn=collate,pin_memory=True)
    variant=cfg['id']+(f'_{selector}' if selector else '')+('_router_off' if disable_router else '')
    if budget is not None or policy_override:
        variant+=f'_K{model.config["budget"]}_D{int(model.policy.depth_ratio*100)}_S{int(model.policy.spatial_ratio*100)}_Q{int(model.policy.query_ratio*100)}'
    metadata=dict(**hardware,id=variant,parent_config_id=cfg['id'],backbone=cfg['backbone'],config=cfg,anchor=model.anchor.provenance,
                  checkpoint=str(checkpoint_path) if checkpoint_path else None,checkpoint_state=state_key,epoch=epoch,successful_updates=updates,
                  selector=model.config['selector'],runtime_router=bool(model.config.get('router')),runtime_budget=model.config['budget'],policy_override=policy_override,runtime_policy=asdict(model.policy),recipe=cfg['recipe'],
                  selection=('fixed supplied anchor; no new checkpoint selection or recovery training' if cfg.get('evaluate_only') else 'full-test EMA milestone peak; terminal online separately evaluated'),latency_is_decision_gate=False)
    if profile_only:
        saved=json.loads((out/'metrics.json').read_text())
        if saved['checkpoint']!=metadata['checkpoint'] or saved['checkpoint_state']!=state_key:raise RuntimeError('Profile must preserve completed test checkpoint')
        samples={}
        for index,data in enumerate(loader):
            samples.setdefault(profile_case(data),(index,data))
            if len(samples)==3:break
        profile=profile_windows(model,samples,out)
        saved.update(profile='profile.json',profile_repaired=True,gflops=profile['matrix_conv_flops']/1e9,
                     latency_ms=profile['latency_mean_ms'],status='complete')
        json_write(out/'completed.json',saved);return
    result={};samples={};begin=time.perf_counter();model_cfg.post_processing.sliding_window=True
    for index,cpu in enumerate(loader):
        samples.setdefault(profile_case(cpu),(index,cpu));data=to_gpu(cpu)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            if cfg.get('vector_condition'):
                model.anchor.vector_condition=False;original,_,_=model.anchor.select(data['inputs'],data['masks'],data['metas'])
                model.anchor.vector_condition=True;vectorized,_,_=model.anchor.select(data['inputs'],data['masks'],data['metas'])
                if not torch.equal(original.indices,vectorized.indices) or not torch.equal(original.valid,vectorized.valid):raise RuntimeError(f'C01 selected-frame identity failed at window {index}')
            predictions,detail=model.predictions(data['inputs'],data['masks'],data['metas'])
            window=model.teacher.post_processing(predictions,data['metas'],model_cfg.post_processing,dataset.class_map)
        for name,values in window.items():result.setdefault(name,[]).extend(values)
        if index in (0,1,2,100):
            trace={k:v for k,v in detail['trace'].items() if k not in ('depth_masks','spatial_masks','query_masks')}
            json_write(out/f'case_{index:03}.json',dict(index=index,metas=tensor_json(data['metas']),trace=tensor_json(trace),
                detections=window,ground_truth=db[data['metas'][0]['video_name']].get('annotations',[])))
        if index%100==0:
            json_write(out/'progress.json',dict(windows=index+1,total_windows=792));print(f'{cfg["id"]}: {index+1}/792',flush=True)
    if set(result)!=expected:raise RuntimeError('Missing test videos')
    rendezvous=out/f'dist_{os.environ["SLURM_JOB_ID"]}'
    dist.init_process_group('gloo',init_method=rendezvous.as_uri(),rank=0,world_size=1)
    try:result=gather_ddp_results(1,result,model_cfg.post_processing)
    finally:dist.destroy_process_group()
    e2e=time.perf_counter()-begin;prediction=dict(results=result)
    json_write(out/'result_detection.json',prediction)
    metrics=build_evaluator(dict(prediction_filename=prediction,**model_cfg.evaluation)).evaluate()
    record=dict(**metadata,metrics=metrics,test_videos=211,test_windows=792,e2e_seconds=e2e,
                e2e_scope='dataset decode/preprocess/transfer/model/window+videoNMS; excludes AP and file writing; cache uncontrolled',
                e2e_includes_C01_identity_checks=bool(cfg.get('vector_condition')),C01_identical_selections=792 if cfg.get('vector_condition') else None,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    json_write(out/'metrics.json',record)
    profile=profile_windows(model,samples,out)
    record.update(profile='profile.json',gflops=profile['matrix_conv_flops']/1e9,latency_ms=profile['latency_mean_ms'],status='complete')
    json_write(out/'completed.json',record);json_write(out/'progress.json',dict(windows=792,total_windows=792,status='complete'));print(json.dumps(record['metrics']),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resources',default=str(EXP/'resources.local.json'))
    p.add_argument('--checkpoint');p.add_argument('--output');p.add_argument('--state',choices=['ema','learned'],default='ema')
    p.add_argument('--selector',choices=['anchor','uniform','random']);p.add_argument('--disable-router',action='store_true');p.add_argument('--budget',type=int);p.add_argument('--policy-json');p.add_argument('--profile-only',action='store_true');p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    if args.dry_run:print(json.dumps(dict(config=read_config(args.config),checkpoint=args.checkpoint,gpu_execution=False)));return
    run(args.config,args.resources,args.checkpoint,args.output,args.profile_only,args.state,args.selector,json.loads(args.policy_json) if args.policy_json else None,args.disable_router,args.budget)


if __name__=='__main__':main()
