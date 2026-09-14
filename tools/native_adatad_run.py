"""Official-checkpoint uniform evaluation and GT-only 80-epoch adaptation."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'upstream')]


def main(args):
    from h65.paper.native_adatad import read_native_config, build_native_config, NativeAdaTAD, evaluate_native, inference_mode
    from h65.paper.runtime import read_resources, json_write
    cfg = read_native_config(args.config)
    if args.mode in ('course', 'eval-course'):
        check = Path(args.preflight_output)
        if not (check/'completed.json').exists():
            result = subprocess.run([sys.executable, '-u', __file__, '--config', args.config,
                '--resources', args.resources, '--mode', 'preflight', '--output', str(check)])
            if result.returncode: return result.returncode
        receipt = json.loads((check/'completed.json').read_text())
        if receipt.get('real_task_updates') != 2 or not receipt.get('no_gt_inference') or receipt['config'] != cfg:
            raise RuntimeError('Native GPU preflight incomplete')
        command = [sys.executable, '-u', __file__, '--config', args.config, '--resources', args.resources,
            '--mode', 'eval' if args.mode == 'eval-course' else 'train', '--resume', '--slice-hours', str(args.slice_hours)]
        if args.output: command.extend(['--output', args.output])
        os.execv(sys.executable, command)
    import torch
    from torch.utils.data import DataLoader
    from opentad.datasets.builder import build_dataset, collate
    from h65.full.runtime import initialize_gpu, seed_all, to_gpu
    from h65.paper.training import EpochDataset, EpochSampler, EMAState, cpu_state, rng_state, restore_rng
    hardware = initialize_gpu(); seed_all(cfg['seed']); resources = read_resources(args.resources)
    native_cfg = build_native_config(cfg, resources)
    model = NativeAdaTAD(native_cfg, cfg, resources).cuda()
    out = Path(args.output or ROOT/'research/paper/runs'/cfg['id']).resolve(); out.mkdir(parents=True, exist_ok=True)
    metadata = dict(hardware, config=cfg, recipe=cfg['recipe'], role='external_retested' if args.mode=='eval' and not args.checkpoint else 'external_adapted',
        encoder=model.provenance, source_revision=(ROOT/'source_revision.txt').read_text().strip(),
        candidate_frames=768, rgb_frames=cfg['frames'], detector_length=cfg['frames'],
        physical_stride=4*(768//cfg['frames']), original_window_stride=4, resolution=160,
        frame_sampling='original crop/window followed by frame_inds[::ratio]; phase zero',
        model_scope='Unmodified upstream ActionFormer + VideoMAE AdaTAD backbone/TIA; no scout, Cross, custom D/S, budget router or KD',
        training_seed=cfg['seed'] if args.mode!='eval' or args.checkpoint else None, evaluation_seed=42,
        precision='BF16 autocast, matching current paper evaluation',
        checkpoint_selection='All 10/20/40/60/80 full-test EMA milestones retained; peak and terminal reported separately')
    if args.mode == 'eval':
        if args.checkpoint:
            payload = torch.load(args.checkpoint, map_location='cpu')
            if payload['metadata']['config'] != cfg: raise ValueError('Native checkpoint/config mismatch')
            model.load_learned(payload[args.state])
            metadata.update(epoch=payload['epoch_index'], checkpoint_state=args.state,
                            successful_updates=payload['successful_updates'], checkpoint=str(args.checkpoint))
        else:
            metadata.update(epoch=None, checkpoint_state='official_ema',
                evaluation_id=cfg['id']+'_official_direct',
                checkpoint_selection='Provided official dense checkpoint; zero updates under sparse input; no test selection')
        evaluate_native(model, native_cfg, resources, out, metadata)
        return 0
    model.train(); optimizer = model.optimizer(cfg)
    dataset = build_dataset(native_cfg.dataset.train)
    ids = {row[0] for row in dataset.data_list}
    if ids != set(resources['datasets']['thumos']['train_ids']) or len(dataset) != 200:
        raise RuntimeError('Native adaptation requires all 200 training videos')
    wrapped = EpochDataset(dataset, 42); n = len(wrapped); batch = cfg['batch_size']
    per_epoch = math.ceil(n/batch); total = per_epoch*cfg['epochs']; warmup = per_epoch*cfg['warmup_epochs']
    def factor(step):
        if step < warmup: return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1., (step-warmup)/max(1,total-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    ema = EMAState(model, cfg['ema_decay']); updates = epoch = cursor = 0
    begin = time.perf_counter(); previous_seconds = 0.; latest = out/'latest.pth'
    metadata.update(train_videos=200, expected_updates=total, updates_per_epoch=per_epoch,
        preflight=args.mode=='preflight', objective='Official detection GT losses only',
        optimizer='Official parameter groups: detector 1e-4; all adapter weights 2e-4; AdamW; head no-decay rules retained',
        course='80 new adaptation epochs from the supplied dense EMA; not training from recognition initialization',
        training_cost_scope='Actual update count and GPU wall time; backward FLOPs not claimed measured',
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.resume and latest.exists():
        saved = torch.load(latest, map_location='cpu')
        if saved['metadata']['config'] != cfg: raise RuntimeError('Resume configuration mismatch')
        model.load_learned(saved['learned']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        ema.values = {k:v.cuda() for k,v in saved['ema'].items()}
        updates = saved['successful_updates']; epoch = saved['epoch_index']; cursor = saved['sample_cursor']
        previous_seconds = saved['elapsed_seconds']; restore_rng(saved['rng']); del saved
        if cursor == n: epoch += 1; cursor = 0
    json_write(out/'metadata.json', metadata)
    requested = [False]; signal.signal(signal.SIGUSR1, lambda *unused: requested.__setitem__(0, True))
    def save(path, e, pos):
        value = dict(learned=cpu_state(model.learned_state()), ema=cpu_state(ema.values),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), successful_updates=updates,
            epoch_index=e, sample_cursor=pos, metadata=metadata, rng=rng_state(),
            elapsed_seconds=previous_seconds+time.perf_counter()-begin)
        temp = path.with_suffix('.tmp'); torch.save(value, temp); temp.replace(path)
    def yield_job(e, pos):
        save(latest, e, pos)
        json_write(out/'yielded.json', dict(reason='planned_checkpoint_time_slice', successful_updates=updates,
                                            epoch_index=e, sample_cursor=pos))
        return 75
    if (out/'yielded.json').exists(): (out/'yielded.json').unlink()
    if args.mode != 'preflight':
        for e in cfg['eval_epochs']:
            path = out/f'epoch_{e:03}.pth'; dest = out/f'eval_{e:03}_ema'
            if e <= epoch and path.exists() and not (dest/'completed.json').exists():
                if time.perf_counter()-begin > args.slice_hours*3600-3600: return yield_job(epoch,cursor)
                saved = torch.load(path,map_location='cpu')
                with ema.apply(model,saved['ema']): evaluate_native(model,native_cfg,resources,dest,dict(metadata,epoch=e,checkpoint_state='ema'))
                del saved
    before = cpu_state(model.learned_state()) if args.mode=='preflight' else None
    if before is not None: before = {k:v.clone() for k,v in before.items()}
    for e in range(epoch, cfg['epochs']):
        wrapped.epoch=e; start=cursor if e==epoch else 0; position=start
        loader=DataLoader(wrapped,batch_size=batch,sampler=EpochSampler(wrapped,42,e,start),
                          num_workers=2,collate_fn=collate,pin_memory=True)
        for cpu in loader:
            data=to_gpu(cpu); optimizer.zero_grad(set_to_none=True); step_start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16): losses=model.task_loss(data)
            if not torch.isfinite(losses['cost']): raise RuntimeError('Nonfinite native GT loss')
            losses['cost'].backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); updates+=1; position+=data['inputs'].shape[0]; ema.update(model)
            row=dict(epoch=e+1,successful_updates=updates,sample_cursor=position,
                losses={k:float(v.detach()) for k,v in losses.items()},grad_norm=float(norm),
                learning_rates=[g['lr'] for g in optimizer.param_groups],step_seconds=time.perf_counter()-step_start,
                peak_gib=torch.cuda.max_memory_allocated()/2**30)
            with (out/'train.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            if updates%20==0: print(json.dumps(row),flush=True)
            if args.mode=='preflight' and updates==2:
                current=cpu_state(model.learned_state())
                changed=[k for k in current if not torch.equal(before[k],current[k])]
                adapter_changed=any('.adapter.' in k for k in changed)
                head_changed=any(k.startswith(('detector.projection.','detector.rpn_head.')) and 'loss_normalizer' not in k for k in changed)
                if not adapter_changed or not head_changed: raise RuntimeError('Native adapter/head failed to update')
                model.load_learned(current)
                with inference_mode(model),ema.apply(model):
                    prediction=model.predictions(data)
                    changed_gt=dict(data,gt_segments=[torch.zeros_like(x)+100000 for x in data['gt_segments']])
                    again=model.predictions(changed_gt)
                    flat=lambda pair: [t for group in pair for t in group]
                    if any(not torch.isfinite(x).all() for x in flat(prediction)): raise RuntimeError('Nonfinite native prediction')
                    if any(not torch.equal(x,y) for x,y in zip(flat(prediction),flat(again))): raise RuntimeError('GT-dependent native inference')
                json_write(out/'completed.json',dict(metadata,real_task_updates=2,no_gt_inference=True,
                    strict_state_reload=True,student_head_updated=True,adapter_updated=True,
                    successful_updates=updates,peak_gib=row['peak_gib'],preflight_updates_discarded=True))
                return 0
            if updates%100==0: save(latest,e,position)
            if (requested[0] or time.perf_counter()-begin>args.slice_hours*3600) and position<n: return yield_job(e,position)
        cursor=0; save(latest,e+1,0)
        if e+1 in cfg['eval_epochs']: save(out/f'epoch_{e+1:03}.pth',e+1,0)
        json_write(out/'progress.json',dict(completed_epochs=e+1,successful_updates=updates,expected_updates=total))
        if requested[0] or time.perf_counter()-begin>args.slice_hours*3600-3600: return yield_job(e+1,0)
        if e+1 in cfg['eval_epochs']:
            with ema.apply(model): evaluate_native(model,native_cfg,resources,out/f'eval_{e+1:03}_ema',dict(metadata,epoch=e+1,checkpoint_state='ema'))
    save(out/'terminal.pth',cfg['epochs'],0)
    json_write(out/'completed.json',dict(metadata,completed_epochs=cfg['epochs'],successful_updates=updates,
        elapsed_seconds=previous_seconds+time.perf_counter()-begin))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    parser.add_argument('--resources',default=str(ROOT/'research/paper/resources.local.json'))
    parser.add_argument('--mode',choices=['eval','preflight','train','course','eval-course'],required=True)
    parser.add_argument('--output'); parser.add_argument('--preflight-output'); parser.add_argument('--checkpoint')
    parser.add_argument('--state',choices=['learned','ema'],default='ema'); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--slice-hours',type=float,default=10)
    raise SystemExit(main(parser.parse_args()))
