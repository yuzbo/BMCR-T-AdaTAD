"""Corrected H65/BMCR courses; BMCR80 reuses warm20 and trains joint60."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'upstream')]
import torch
from opentad.cores.scheduler import build_scheduler
from h65.full.runtime import (config, PRETRAIN, RUNS, initialize_gpu, to_gpu, json_write, train_loader,
                              optimizer_for, EMA, save_checkpoint, restore_rng)
from h65.full.model import FormalH65
from h65.full.objectives import curriculum
from h65.full.course import BMCR80_RECIPE, warm_origin, validate_scales, validate_resume


def initialize_joint(model, backbone, total_epochs):
    """The BMCR80 preflight and formal run use this exact warm/scales path."""
    checkpoint = RUNS / f'{backbone}_warm/terminal.pth'
    warm = torch.load(checkpoint, map_location='cpu')
    if warm['metadata'].get('fidelity_revision') != 'lr_identity_crop_validity_v1':
        raise RuntimeError('joint training requires a newly corrected warm checkpoint')
    if warm['completed_epochs'] != 20 or warm['successful_updates'] != 2000:
        raise RuntimeError('joint training requires the full20-epoch/2000-update warm EMA')
    origin = warm_origin(warm, checkpoint, backbone) if total_epochs == 80 else None
    model.load_state_dict(warm['state_dict_ema'], strict=True)
    del warm
    audit = None
    if model.variant == 'bmcr':
        audit = json.loads((RUNS / f'{backbone}_audit/scales.json').read_text())
        scales = validate_scales(audit, origin) if total_epochs == 80 else audit['scales']
        model.utility_scales.copy_(model.utility_scales.new_tensor(scales))
    return origin, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['s', 'b'], required=True)
    parser.add_argument('--phase', choices=['warm', 'joint'], required=True)
    parser.add_argument('--variant', choices=['h65', 'bmcr'], default='h65')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--total-epochs', type=int, choices=[60,80], default=60)
    args = parser.parse_args()
    if args.total_epochs == 80 and (args.phase != 'joint' or args.variant != 'bmcr'):
        parser.error('the80-epoch course is BMCR joint60 from existing corrected warm20')
    hardware = initialize_gpu()
    name = f'{args.backbone}_{args.phase}' if args.phase == 'warm' else f'{args.backbone}_{args.variant}'
    out = RUNS / ('preflight_' + name if args.preflight else name)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config(args.backbone)
    loader, sampler = train_loader(cfg, args.workers)
    load_joint = args.phase == 'joint' and (not args.preflight or args.total_epochs == 80)
    model = FormalH65(cfg.model, None if load_joint else str(PRETRAIN[args.backbone]), args.variant).cuda()
    origin, audit = initialize_joint(model, args.backbone, args.total_epochs) if load_joint else (None,None)
    optimizer = optimizer_for(model, args.backbone, args.phase)
    epochs = 20 if args.phase == 'warm' else args.total_epochs-20
    scheduler, _ = build_scheduler(dict(type='LinearWarmupCosineAnnealingLR', warmup_epoch=2 if args.phase == 'warm' else 3,
                                         max_epoch=epochs), optimizer, len(loader))
    ema = EMA(model, cfg)
    start_epoch, updates = 0, 0
    latest = out / 'latest.pth'
    if latest.exists() and not args.preflight:
        resume = torch.load(latest, map_location='cpu')
        validate_resume(resume['metadata'],args.total_epochs,args.phase,args.variant,args.backbone,origin)
        if args.total_epochs == 80 and resume['metadata'].get('utility_audit') != audit:
            raise RuntimeError('Resume must retain the original training-only utility scales')
        model.load_state_dict(resume['state_dict'], strict=True)
        ema.module.load_state_dict(resume['state_dict_ema'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        start_epoch, updates = resume['completed_epochs'], resume['successful_updates']
        restore_rng(resume['rng'])
    metadata = dict(**hardware, backbone=args.backbone, variant=args.variant, phase=args.phase,
                    initialization='corrected_warm20_ema' if load_joint else str(PRETRAIN[args.backbone]), train_videos=len(loader.dataset),
                    samples_per_epoch=len(sampler), updates_per_epoch=len(loader), phase_epochs=epochs,
                    full_training=not args.preflight, upstream_modified=False, protocol=f'pure-global-TIA-{args.variant.upper()}-{args.total_epochs}',
                    parent_checkpoint=str((RUNS/f'{args.backbone}_warm/terminal.pth').resolve()) if load_joint else None,
                    terminal_state='state_dict_ema', candidate_frames=768, heavy_frames=384, global_tia_temporal_size=192,
                    precision='BF16 backbone/scout, FP32 detector', ema_decay=.999,
                    fidelity_revision='lr_identity_crop_validity_v1',
                    boundary_supervision='only real endpoints; crop-created endpoints excluded',
                    optimizer_groups=[dict(base_lr=base_lr, weight_decay=group['weight_decay'],
                                           parameters=sum(p.numel() for p in group['params']))
                                      for base_lr, group in zip(scheduler.base_lrs, optimizer.param_groups)])
    if args.total_epochs == 80:
        metadata.update(recipe=BMCR80_RECIPE,course_total_epochs=80,seed=3407,
                        warm_origin=origin,utility_audit=audit,expected_joint_updates=6000,
                        shared_warm_retrained=False,joint_optimizer_reset=True,
                        lr_schedule='3epoch linear warmup + cosine to joint60; curriculum unchanged',
                        checkpoint_selection='full211-test EMA peak at total25..80 every5; total60 and80 retained')
    json_write(out / 'config.json', metadata)
    before = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    history = out / 'train.jsonl'
    model.train()
    for epoch in range(start_epoch, epochs):
        sampler.epoch = epoch
        for batch_index, data in enumerate(loader):
            if 'gt_boundary_validity' not in data:
                raise RuntimeError('training pipeline dropped GT boundary-validity labels')
            data = to_gpu(data)
            weights = curriculum(args.phase, updates)
            if args.preflight:
                # Exercise alpha0 first and all joint gradients second, in the real batch2 geometry.
                weights = curriculum('warm' if updates == 0 else 'joint', 2000)
            optimizer.zero_grad(set_to_none=True)
            measured = time.perf_counter()
            use_teacher = args.variant == 'bmcr' and weights['contribution'] > 0 and (updates % 8 == 0 or args.preflight)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses, diagnostics = model.train_batch(**data, weights=weights, teacher=ema.module if use_teacher else None)
            if not torch.isfinite(losses['cost']):
                raise RuntimeError(f'nonfinite cost at epoch{epoch}, batch{batch_index}')
            losses['cost'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            updates += 1
            record = dict(epoch=epoch, total_epoch=epoch+1+(20 if args.phase=='joint' else 0), batch=batch_index, successful_updates=updates,
                          losses={key: float(value.detach()) for key, value in losses.items()}, weights=weights,
                          grad_norm=float(norm), seconds=time.perf_counter()-measured, diagnostics=diagnostics,
                          peak_gib=torch.cuda.max_memory_allocated()/2**30,
                          learning_rates=[group['lr'] for group in optimizer.param_groups])
            with history.open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            if updates % 10 == 0 or args.preflight:
                print(json.dumps(record), flush=True)
            if args.preflight and updates == 2:
                save_checkpoint(latest, model, ema, optimizer, scheduler, 0, updates, metadata)
                payload = torch.load(latest, map_location='cpu')
                ema.module.load_state_dict(payload['state_dict_ema'], strict=True)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    predictions, selection = ema.module.predictions(data['inputs'][:1], data['masks'][:1])
                if not all(torch.isfinite(x).all() for group in predictions for x in group):
                    raise RuntimeError('preflight predictions are nonfinite')
                json_write(out/'completed.json', dict(**metadata, successful_updates=updates,
                           peak_gib=record['peak_gib'], seconds=time.perf_counter()-before,
                           selected_counts=selection.valid.sum(-1).tolist(), strict_reload=True,
                           real_joint_initialization=load_joint, utility_scales=model.utility_scales.tolist(),
                           counterfactual_labels_last_step=len(diagnostics.get('counterfactuals',[]))))
                return
        save_checkpoint(latest, model, ema, optimizer, scheduler, epoch+1, updates, metadata)
        if (epoch+1) % 5 == 0:
            milestone = out/f'epoch_{epoch+1:02}.pth'
            temporary = milestone.with_suffix('.copying')
            shutil.copy2(latest, temporary)
            temporary.replace(milestone)
        json_write(out/'progress.json', dict(completed_epochs=epoch+1,total_epochs=epoch+1+(20 if args.phase=='joint' else 0), successful_updates=updates,
                   expected_updates=epochs*len(loader), elapsed_seconds=time.perf_counter()-before))
    shutil.copy2(latest, out/'terminal.pth')
    json_write(out/'completed.json', dict(**metadata, completed_epochs=epochs, successful_updates=updates,
               elapsed_seconds=time.perf_counter()-before, peak_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__ == '__main__':
    main()
