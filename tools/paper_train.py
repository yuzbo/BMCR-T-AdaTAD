"""Complete paper courses; checkpoints, real-action training and inline THUMOS tests."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]


def main(args):
    from h65.paper.runtime import read_config,read_resources,build_config,json_write,dry_description
    cfg=read_config(args.config)
    if args.dry_run:print(json.dumps(dry_description(cfg,args.resources),indent=2));return 0
    resources=read_resources(args.resources);dataset_name=cfg['dataset'];ds=resources['datasets'][dataset_name]
    required_asset='internvideo_mq' if cfg['backbone']=='internvideo_mq' else 'anet:'+cfg['backbone'] if dataset_name=='anet' else None
    if required_asset and required_asset not in resources.get('verified_downloads',{}):raise RuntimeError('Required downloaded tensor asset is not verified: '+required_asset)
    if dataset_name=='anet' and not args.preflight:
        ready=Path(ds['ready_file'])
        if not ready.exists() or json.loads(ready.read_text())['status']!='READY':raise RuntimeError('Full ActivityNet preparation is not READY')
    import torch
    from torch.utils.data import DataLoader
    from opentad.datasets.builder import build_dataset,collate
    from h65.full.runtime import initialize_gpu,seed_all,to_gpu
    from h65.paper.model import PaperModel
    from h65.paper.training import EpochDataset,EpochSampler,EMAState,optimizer_groups,rng_state,restore_rng,cpu_state
    from h65.paper.profile import calibrate_costs
    from h65.paper.objectives import objectives
    from h65.paper.interventions import collect_action,action_loss,evaluation_state,scheduled_actions
    from h65.paper.evaluation import evaluate
    begin=time.perf_counter();hardware=initialize_gpu();seed_all(cfg['seed']);model_cfg=build_config(cfg,resources)
    out=Path(args.output or ROOT/'research/paper/runs'/cfg['id']).resolve();out.mkdir(parents=True,exist_ok=True)
    if args.preflight and dataset_name=='anet':
        missing=[name for name in ds['train_ids'] if not (Path(ds['train_videos'])/('v_'+name+'.mp4')).is_file()]
        block=out/'preflight_missing.txt';block.write_text('\n'.join(missing)+'\n')
        model_cfg.dataset.train.block_list=str(block)
    dataset=build_dataset(model_cfg.dataset.train);built={r[0] for r in dataset.data_list};expected=set(ds['train_ids'])
    if not built or not built<=expected:raise RuntimeError('Unexpected training video IDs')
    if dataset_name=='thumos' and built!=expected:raise RuntimeError('THUMOS requires all 200 training videos')
    wrapped=EpochDataset(dataset,cfg['seed']);n=len(wrapped);accumulate=cfg['accumulate'];per_epoch=math.ceil(n/accumulate)
    model=PaperModel(model_cfg,cfg,resources).cuda().train();optimizer=torch.optim.AdamW(optimizer_groups(model,cfg))
    total=cfg['epochs']*per_epoch;warmup=max(1,int(cfg.get('warmup_epochs',1)*per_epoch))
    schedule_total=cfg.get('schedule_epochs',cfg['epochs'])*per_epoch
    def factor(step):
        if step<warmup:return max(.01,(step+1)/warmup)
        return .01+.99*.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,schedule_total-warmup))))
    schedule=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    metadata=dict(hardware,recipe=cfg['recipe'],config=cfg,encoder=model.encoder.provenance,
                  teacher=resources.get('teachers',{}).get(dataset_name+':'+cfg['backbone']),
                  teacher_kind='external_official' if model.teacher is not None else 'none' if cfg.get('dense_baseline') else 'shared_full_student',
                  train_videos=len(built),annotation_train_videos=len(expected),official_gt_filtered_ids=sorted(expected-built),
                  expected_updates=total,updates_per_epoch=per_epoch,preflight=args.preflight,
                  scheduler_horizon_epochs=cfg.get('schedule_epochs',cfg['epochs']),
                  candidate_frames=768,detector_length=768 if dataset_name=='thumos' else 192,
                  augmentation='per-video-index and epoch deterministic; exact cursor resume',
                  source_revision=(ROOT/'source_revision.txt').read_text().strip() if (ROOT/'source_revision.txt').exists() else hardware['source_revision'])
    latest=out/'latest.pth';updates=epoch=cursor=0
    counts=dict(external_teacher=0,shared_full=0,actual_pairs=0,action_student_forwards=0,action_head_forwards=0,action_external_teacher=0,action_shared_full=0)
    saved=None;previous_seconds=0.
    if args.resume and latest.exists():
        saved=torch.load(latest,map_location='cpu')
        if saved['metadata']['config']!=cfg:raise RuntimeError('Resume configuration mismatch')
        model.load_learned(saved['learned']);optimizer.load_state_dict(saved['optimizer']);schedule.load_state_dict(saved['scheduler'])
        updates=saved['successful_updates'];epoch=saved['epoch_index'];cursor=saved['sample_cursor'];counts=saved['query_counts']
        previous_seconds=saved.get('elapsed_seconds',0.)
        if cursor==n:epoch+=1;cursor=0
    probe_loader=DataLoader(wrapped,batch_size=1,shuffle=False,num_workers=2,collate_fn=collate,pin_memory=True)
    if saved is None:
        from h65.paper.geometry import candidate_mask
        for cpu in probe_loader:
            data=to_gpu(cpu)
            if int(candidate_mask(data).sum())==768:break
        else:raise RuntimeError('No full training window for primary cost calibration')
        costs=calibrate_costs(model,data);json_write(out/'cost_table.json',costs)
    ema=EMAState(model,cfg['ema_decay'])
    if saved:
        ema.values={k:v.to(next(model.parameters()).device) for k,v in saved['ema'].items()};restore_rng(saved['rng'])
    json_write(out/'metadata.json',metadata)
    if (out/'yielded.json').exists():(out/'yielded.json').unlink()
    requested_stop=[False]
    signal.signal(signal.SIGUSR1,lambda *unused:requested_stop.__setitem__(0,True))
    slice_seconds=args.slice_hours*3600
    def save_checkpoint(path,epoch_index,sample_cursor):
        value=dict(learned=cpu_state(model.learned_state()),ema=cpu_state(ema.values),optimizer=optimizer.state_dict(),scheduler=schedule.state_dict(),
                   successful_updates=updates,epoch_index=epoch_index,sample_cursor=sample_cursor,query_counts=counts,metadata=metadata,rng=rng_state(),
                   elapsed_seconds=previous_seconds+time.perf_counter()-begin)
        tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
    def yield_job(epoch_index,sample_cursor):
        if sample_cursor==n:
            epoch_index+=1;sample_cursor=0
            if epoch_index in cfg['eval_epochs'] or epoch_index==cfg['epochs']:
                save_checkpoint(out/f'epoch_{epoch_index:03}.pth',epoch_index,0)
        save_checkpoint(latest,epoch_index,sample_cursor)
        json_write(out/'yielded.json',dict(reason='planned_time_slice',successful_updates=updates,epoch_index=epoch_index,sample_cursor=sample_cursor))
        return 75
    # Finish an own saved checkpoint's pending full test before continuing that run.
    if not args.preflight and dataset_name=='thumos':
        for e in cfg['eval_epochs']:
            path=out/f'epoch_{e:03}.pth';target=out/f'eval_{e:03}_ema'
            if e<=epoch and path.exists() and not (target/'completed.json').exists():
                if time.perf_counter()-begin>slice_seconds-3600:return yield_job(epoch,cursor)
                checkpoint=torch.load(path,map_location='cpu')
                with ema.apply(model,checkpoint['ema']):evaluate(model,model_cfg,resources,target,dict(**metadata,epoch=e,checkpoint_state='ema'))
                del checkpoint
    optimizer.zero_grad(set_to_none=True)
    initial_head={k:v.detach().clone() for k,v in model.readout.named_parameters()} if args.preflight else None
    teacher_before={k:v.detach().clone() for k,v in model.teacher.state_dict().items()} if args.preflight and model.teacher is not None else None
    for current_epoch in range(epoch,cfg['epochs']):
        wrapped.epoch=current_epoch;start=cursor if current_epoch==epoch else 0
        sampler=EpochSampler(wrapped,cfg['seed'],current_epoch,start)
        loader=DataLoader(wrapped,batch_size=1,sampler=sampler,num_workers=2,collate_fn=collate,pin_memory=True)
        bucket={};used=0;position=start;denominator=min(accumulate,n-position)
        for cpu in loader:
            data=to_gpu(cpu);step_start=time.perf_counter()
            plan_index=(0 if updates%2==0 else 4) if args.preflight else cfg.get('fixed_plan',5) if cfg.get('train_plan_mode')=='fixed' else updates%len(model.menu)
            with torch.autocast('cuda',dtype=torch.bfloat16):losses,detail,queries=objectives(model,data,plan_index)
            for k,v in queries.items():counts[k]+=v
            if used==0 and cfg['loss'].get('action',0)>0 and (args.preflight or updates%cfg['action_interval']==0) and not cfg.get('dense_baseline',False):
                actions=scheduled_actions(cfg,updates if args.preflight else updates//cfg['action_interval'],args.preflight)
                for kind,number in actions:
                    record=collect_action(model,data,kind,number,measure=args.preflight)
                    if record is None:continue
                    record['successful_updates']=updates
                    with (out/'interventions.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                    auxiliary=action_loss(model,record,data['inputs'].device)
                    losses['cost']=losses['cost']+cfg['loss'].get('action',.1)*auxiliary
                    losses['action_'+kind]=auxiliary;counts['actual_pairs']+=1
                    counts['action_student_forwards']+=2;counts['action_head_forwards']+=3
                    counts['action_external_teacher']+=int(record['repair_source']=='external_official')
                    counts['action_shared_full']+=int(record['repair_source']=='shared_full_student')
            if not torch.isfinite(losses['cost']):raise RuntimeError('Nonfinite complete-model loss')
            (losses['cost']/denominator).backward();model.readout.commit_normalizer();used+=1;position+=1
            for key,value in losses.items():bucket[key]=bucket.get(key,0.)+float(value.detach())/denominator
            if used<denominator:continue
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step();schedule.step();optimizer.zero_grad(set_to_none=True);updates+=1;ema.update(model)
            log=dict(epoch=current_epoch+1,successful_updates=updates,sample_cursor=position,losses=bucket,grad_norm=float(norm),
                     learning_rates=[g['lr'] for g in optimizer.param_groups],query_counts=dict(counts),plan=detail['trace']['plan'],
                     last_microbatch_seconds=time.perf_counter()-step_start,peak_gib=torch.cuda.max_memory_allocated()/2**30)
            with (out/'train.jsonl').open('a') as stream:stream.write(json.dumps(log)+'\n')
            bucket={};used=0;denominator=min(accumulate,n-position)
            if updates%20==0:print(json.dumps({k:log[k] for k in ('epoch','successful_updates','losses','peak_gib')}),flush=True)
            if args.preflight and updates>=2:
                if cfg.get('train_head',True) and all(torch.equal(initial_head[k],v) for k,v in model.readout.named_parameters()):raise RuntimeError('Student task head did not update')
                if teacher_before is not None and any(not torch.equal(teacher_before[k],v) for k,v in model.teacher.state_dict().items()):raise RuntimeError('External teacher changed')
                state=cpu_state(model.learned_state());model.load_learned(state)
                with evaluation_state(model),torch.no_grad(),ema.apply(model):
                    native,detail_a=model.forward_native(data)
                    changed=dict(data);changed['gt_segments']=[x*0 for x in data['gt_segments']];changed['gt_labels']=[x*0+999 for x in data['gt_labels']]
                    other,detail_b=model.forward_native(changed)
                    if not torch.isfinite(native).all() or not torch.allclose(native,other,atol=1e-6,rtol=1e-5):raise RuntimeError('Inference depends on GT or produces nonfinite features')
                    if not torch.equal(detail_a['selection'].indices,detail_b['selection'].indices):raise RuntimeError('Routing used GT')
                json_write(out/'completed.json',dict(**metadata,successful_updates=updates,teacher_frozen=True,student_head_updated=cfg.get('train_head',True),
                                                    strict_state_reload=True,no_gt_inference=True,real_task_updates=2,query_counts=counts,peak_gib=log['peak_gib']))
                return 0
            if updates%100==0:save_checkpoint(latest,current_epoch,position)
            if requested_stop[0] or time.perf_counter()-begin>slice_seconds:
                return yield_job(current_epoch,position)
        cursor=0;save_checkpoint(latest,current_epoch+1,0)
        if current_epoch+1 in cfg['eval_epochs'] or current_epoch+1==cfg['epochs']:
            save_checkpoint(out/f'epoch_{current_epoch+1:03}.pth',current_epoch+1,0)
        json_write(out/'progress.json',dict(completed_epochs=current_epoch+1,successful_updates=updates,expected_updates=total,query_counts=counts))
        if dataset_name=='thumos' and current_epoch+1 in cfg['eval_epochs']:
            if time.perf_counter()-begin>slice_seconds-3600:return yield_job(current_epoch+1,0)
            with ema.apply(model):evaluate(model,model_cfg,resources,out/f'eval_{current_epoch+1:03}_ema',dict(**metadata,epoch=current_epoch+1,checkpoint_state='ema'))
        if time.perf_counter()-begin>slice_seconds:return yield_job(current_epoch+1,0)
    save_checkpoint(out/'terminal.pth',cfg['epochs'],0)
    json_write(out/'completed.json',dict(**metadata,completed_epochs=cfg['epochs'],successful_updates=updates,query_counts=counts,elapsed_seconds=previous_seconds+time.perf_counter()-begin))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--resources',default=str(ROOT/'research/paper/resources.local.json'))
    parser.add_argument('--output');parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--slice-hours',type=float,default=10);parser.add_argument('--dry-run',action='store_true')
    raise SystemExit(main(parser.parse_args()))
