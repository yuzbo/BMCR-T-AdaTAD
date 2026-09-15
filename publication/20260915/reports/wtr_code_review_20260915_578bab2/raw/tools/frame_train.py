"""Full-dataset independent FPW route training; --dry-run creates no CUDA context."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
from h65.frame.runtime import read_config,dry_description,EXP,json_write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--resources',default=str(EXP/'resources.local.json'))
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    cfg=read_config(args.config)
    if args.dry_run:print(json.dumps(dry_description(cfg,args.resources),indent=2));return
    import torch
    from dataclasses import asdict
    from torch.utils.data import DataLoader
    from opentad.datasets.builder import build_dataset,collate
    from h65.full.runtime import initialize_gpu,EpochSampler,to_gpu,rng_state,restore_rng,seed_all
    from h65.frame.runtime import read_resources,data_config,tensor_json
    from h65.frame.model import FrameModel
    from h65.frame.objectives import training_objectives,shared_full_student_stopgrad
    from h65.frame.utility import utility_training_loss
    from h65.frame.contracts import EnginePolicy
    hardware=initialize_gpu();seed_all(cfg['seed']);resources=read_resources(args.resources);model_cfg=data_config(cfg['backbone'])
    dataset=build_dataset(model_cfg.dataset.train)
    if len(dataset)!=200:raise RuntimeError('Full200 training videos are required')
    sampler=EpochSampler(dataset,cfg['seed']);generator=torch.Generator()
    loader=DataLoader(dataset,batch_size=cfg['batch_size'],sampler=sampler,num_workers=2,collate_fn=collate,
                      pin_memory=True,generator=generator,persistent_workers=False)
    accumulate=cfg['accumulate'];steps_per_epoch=len(loader)//accumulate
    if steps_per_epoch!=100 or len(loader)%accumulate:raise RuntimeError('Expected100 optimizer updates per full epoch')
    model=FrameModel(model_cfg.model,cfg,resources).cuda().train()
    groups={}
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        lr=cfg.get('adapter_lr',1e-5) if name.startswith('anchor.') else cfg.get('learning_rate',1e-4)
        decay=.05 if param.ndim>1 else 0.;groups.setdefault((lr,decay),[]).append(param)
    if not groups:raise RuntimeError('This recipe has no trainable parameters')
    optimizer=torch.optim.AdamW([dict(params=values,lr=lr,weight_decay=decay) for (lr,decay),values in groups.items()])
    total=cfg['epochs']*100
    def factor(step):
        if step<100:return .1+.9*step/100
        return .01+.99*.5*(1+math.cos(math.pi*min((step-100)/max(total-100,1),1)))
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    ema={k:v.detach().clone() for k,v in model.learned_state().items()}
    out=EXP/'runs'/('preflight_'+cfg['id'] if args.preflight else cfg['id']);out.mkdir(parents=True,exist_ok=True)
    metadata=dict(**hardware,recipe=cfg['recipe'],config=cfg,anchor=model.anchor.provenance,
                  official_teacher=resources['official'][cfg['backbone']],train_videos=200,test_videos=211,
                  batch_size=cfg['batch_size'],accumulate=accumulate,updates_per_epoch=100,
                  temporal_unit='candidate_frame',native_output=384,original_detector_axis=768,
                  latency_is_decision_gate=False,primary_comparison=['matrix_conv_flops','best_average_mAP'],
                  preflight=args.preflight,EMA_decay=cfg['ema_decay'],trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  activation_recomputation='engine full blocks or heavy/light MLP during backward; excluded from inference FLOPs',
                  decoder_initialization=getattr(model.decoder,'initialization',dict(kind='zero_residual_random')))
    start=updates=0;queries=0;latest=out/'latest.pth'
    if latest.exists() and not args.preflight:
        checkpoint=torch.load(latest,map_location='cpu')
        if checkpoint['metadata']['config']!=cfg:raise RuntimeError('Resume recipe differs')
        model.load_learned(checkpoint['learned']);ema={k:v.to('cuda') for k,v in checkpoint['ema'].items()}
        if 'runtime_policy' in checkpoint:model.policy=EnginePolicy(**checkpoint['runtime_policy'])
        optimizer.load_state_dict(checkpoint['optimizer']);schedule.load_state_dict(checkpoint['scheduler'])
        start=checkpoint['completed_epochs'];updates=checkpoint['successful_updates'];queries=checkpoint['teacher_queries'];restore_rng(checkpoint['rng'])
    json_write(out/'config.json',metadata);begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def save(path,epoch):
        payload=dict(learned=model.learned_state(),ema=ema,optimizer=optimizer.state_dict(),scheduler=schedule.state_dict(),
                     completed_epochs=epoch,successful_updates=updates,teacher_queries=queries,metadata=metadata,rng=rng_state(),runtime_policy=asdict(model.policy))
        temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)
    optimizer.zero_grad(set_to_none=True);bucket={};micro=0;gt_gradient_verified=False
    for epoch in range(start,cfg['epochs']):
        sampler.epoch=epoch;generator.manual_seed(cfg['seed']+epoch)
        for data in loader:
            data=to_gpu(data);step_begin=time.perf_counter()
            if 'gt_boundary_validity' not in data or any('frame_inds' not in m for m in data['metas']):raise RuntimeError('Missing real coordinate/boundary metadata')
            budgets=cfg.get('train_budgets',[cfg['budget']]);model.config['budget']=budgets[updates%len(budgets)]
            if cfg.get('mixed_gates',False):
                from dataclasses import replace
                combo=(updates//len(budgets))%4
                policy=replace(model.policy,depth_ratio=.5 if combo&1 else 1.,spatial_ratio=.48 if combo&2 else 1.,mode=cfg.get('train_execution','dense_mask'))
            else:policy=None
            with torch.autocast('cuda',dtype=torch.bfloat16):
                target=model.teacher.dense_native(data['inputs']);queries+=1
                if updates in cfg.get('progressive_drop_updates',[]) and micro%accumulate==0:
                    from h65.frame.progressive import drop_one
                    deletion=drop_one(model,data,target)
                    with (out/'progressive_drop.jsonl').open('a') as stream:stream.write(json.dumps(dict(update=updates,**deletion))+'\n')
                native,detail=model.forward_native(data,policy=policy)
                loss_cfg={k:v for k,v in cfg['loss'].items() if k in ('gt_weight','feature_weight','difference_weight','output_kd_weight')}
                losses=training_objectives(native,target,data,model.teacher,**loss_cfg)
                if args.preflight and not gt_gradient_verified:
                    gt=model.teacher.loss(native,data)['cost'];gradient=torch.autograd.grad(gt,native,retain_graph=True)[0]
                    if not torch.isfinite(gradient).all() or gradient.abs().sum()==0:raise RuntimeError('Student GT input gradient missing')
                    gt_gradient_verified=True
                if cfg['loss'].get('self_weight',0):
                    full,_=model.forward_native(data,selection=detail['selection'],policy=model.full_policy(),apply_router=False)
                    full_gt=model.teacher.loss(full,data)['cost']
                    self_loss=full_gt+shared_full_student_stopgrad(native,full,data,label_source='self')
                    losses['self_loss']=self_loss;losses['cost']=losses['cost']+cfg['loss']['self_weight']*self_loss
                if cfg.get('router',False) and updates%8==0:
                    utility,records=utility_training_loss(model,data,detail,target,max_pairs=2)
                    losses['utility_loss']=utility;losses['cost']=losses['cost']+utility
                    with (out/'interventions.jsonl').open('a') as stream:
                        for rec in records:stream.write(json.dumps(dict(update=updates,**rec))+'\n')
            if not torch.isfinite(losses['cost']):raise RuntimeError('Nonfinite FPW training cost')
            (losses['cost']/accumulate).backward();micro+=1
            for key,value in losses.items():bucket[key]=bucket.get(key,0.)+float(value.detach())/accumulate
            if micro%accumulate:continue
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step();schedule.step();optimizer.zero_grad(set_to_none=True);updates+=1
            with torch.no_grad():
                for key,value in model.learned_state().items():ema[key].mul_(cfg['ema_decay']).add_(value,alpha=1-cfg['ema_decay'])
            record=dict(epoch=epoch+1,successful_updates=updates,losses=bucket,grad_norm=float(norm),
                        learning_rates=[g['lr'] for g in optimizer.param_groups],teacher_queries=queries,
                        last_microbatch_seconds=time.perf_counter()-step_begin,peak_gib=torch.cuda.max_memory_allocated()/2**30,
                        budget=model.config['budget'],depth_ratio=(policy or model.policy).depth_ratio,spatial_ratio=(policy or model.policy).spatial_ratio,
                        main_forward_heavy_mlp_tokens=sum(detail['trace']['heavy_mlp']),main_forward_attention_q_tokens=sum(detail['trace']['q']),
                        main_forward_attention_kv_tokens=sum(detail['trace']['kv']),routing_qk_macs=sum(detail['trace']['score_qk']),static_keep=model.policy.static_keep)
            with (out/'train.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            if updates%10==0 or args.preflight:print(json.dumps(record),flush=True)
            bucket={}
            if args.preflight and updates==2:
                save(latest,0);loaded=torch.load(latest,map_location='cpu');model.load_learned(loaded['ema'])
                model.eval()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):preds,_=model.predictions(data['inputs'],data['masks'],data['metas'])
                if not all(torch.isfinite(x).all() for group in preds for x in group):raise RuntimeError('Nonfinite preflight predictions')
                json_write(out/'completed.json',dict(**metadata,successful_updates=2,gt_input_gradient=gt_gradient_verified,
                           strict_aux_reload=True,peak_gib=record['peak_gib'],teacher_queries=queries));return
        model.config['budget']=cfg['budget'];save(latest,epoch+1)
        if (epoch+1)%5==0:save(out/f'epoch_{epoch+1:02}.pth',epoch+1)
        json_write(out/'progress.json',dict(completed_epochs=epoch+1,successful_updates=updates,expected_updates=total,teacher_queries=queries,elapsed_seconds=time.perf_counter()-begin))
    save(out/'terminal.pth',cfg['epochs'])
    json_write(out/'completed.json',dict(**metadata,completed_epochs=cfg['epochs'],successful_updates=updates,teacher_queries=queries,elapsed_seconds=time.perf_counter()-begin))


if __name__=='__main__':main()
