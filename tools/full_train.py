"""Full 20+40 course for NEW H65/BMCR S/B; never train official baselines."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['s', 'b'], required=True)
    parser.add_argument('--phase', choices=['warm', 'joint'], required=True)
    parser.add_argument('--variant', choices=['h65', 'bmcr'], default='h65')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    hardware = initialize_gpu()
    name = f'{args.backbone}_{args.phase}' if args.phase == 'warm' else f'{args.backbone}_{args.variant}'
    out = RUNS / ('preflight_' + name if args.preflight else name)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config(args.backbone)
    loader, sampler = train_loader(cfg, args.workers)
    model = FormalH65(cfg.model, str(PRETRAIN[args.backbone]), args.variant).cuda()
    if args.phase == 'joint' and not args.preflight:
        warm = torch.load(RUNS / f'{args.backbone}_warm/terminal.pth', map_location='cpu')
        if warm['metadata'].get('fidelity_revision') != 'lr_identity_crop_validity_v1':
            raise RuntimeError('joint training requires a newly corrected warm checkpoint')
        if warm['completed_epochs'] != 20 or warm['successful_updates'] != 2000:
            raise RuntimeError('joint training requires the full20-epoch/2000-update warm EMA')
        model.load_state_dict(warm['state_dict_ema'], strict=True)
        if args.variant == 'bmcr':
            audit = json.loads((RUNS / f'{args.backbone}_audit/scales.json').read_text())
            model.utility_scales.copy_(model.utility_scales.new_tensor(audit['scales']))
    optimizer = optimizer_for(model, args.backbone, args.phase)
    epochs = 20 if args.phase == 'warm' else 40
    scheduler, _ = build_scheduler(dict(type='LinearWarmupCosineAnnealingLR', warmup_epoch=2 if args.phase == 'warm' else 3,
                                         max_epoch=epochs), optimizer, len(loader))
    ema = EMA(model, cfg)
    start_epoch, updates = 0, 0
    latest = out / 'latest.pth'
    if latest.exists() and not args.preflight:
        resume = torch.load(latest, map_location='cpu')
        if (resume['metadata'].get('fidelity_revision') != 'lr_identity_crop_validity_v1' or
            any(resume['metadata'][key] != value for key, value in
                [('backbone', args.backbone), ('phase', args.phase), ('variant', args.variant)])):
            raise RuntimeError('resume checkpoint belongs to a different experiment recipe')
        model.load_state_dict(resume['state_dict'], strict=True)
        ema.module.load_state_dict(resume['state_dict_ema'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        start_epoch, updates = resume['completed_epochs'], resume['successful_updates']
        restore_rng(resume['rng'])
    metadata = dict(**hardware, backbone=args.backbone, variant=args.variant, phase=args.phase,
                    initialization=str(PRETRAIN[args.backbone]), train_videos=len(loader.dataset),
                    samples_per_epoch=len(sampler), updates_per_epoch=len(loader), phase_epochs=epochs,
                    full_training=not args.preflight, upstream_modified=False, protocol='pure-global-TIA-H65-60',
                    parent_checkpoint=str(RUNS/f'{args.backbone}_warm/terminal.pth') if args.phase=='joint' and not args.preflight else None,
                    terminal_state='state_dict_ema', candidate_frames=768, heavy_frames=384, global_tia_temporal_size=192,
                    precision='BF16 backbone/scout, FP32 detector', ema_decay=.999,
                    fidelity_revision='lr_identity_crop_validity_v1',
                    boundary_supervision='only real endpoints; crop-created endpoints excluded',
                    optimizer_groups=[dict(base_lr=base_lr, weight_decay=group['weight_decay'],
                                           parameters=sum(p.numel() for p in group['params']))
                                      for base_lr, group in zip(scheduler.base_lrs, optimizer.param_groups)])
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
            record = dict(epoch=epoch, batch=batch_index, successful_updates=updates,
                          losses={key: float(value.detach()) for key, value in losses.items()}, weights=weights,
                          grad_norm=float(norm), seconds=time.perf_counter()-measured, diagnostics=diagnostics,
                          peak_gib=torch.cuda.max_memory_allocated()/2**30)
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
                           selected_counts=selection.valid.sum(-1).tolist(), strict_reload=True))
                return
        save_checkpoint(latest, model, ema, optimizer, scheduler, epoch+1, updates, metadata)
        if (epoch+1) % 5 == 0:
            milestone = out/f'epoch_{epoch+1:02}.pth'
            temporary = milestone.with_suffix('.copying')
            shutil.copy2(latest, temporary)
            temporary.replace(milestone)
        json_write(out/'progress.json', dict(completed_epochs=epoch+1, successful_updates=updates,
                   expected_updates=epochs*len(loader), elapsed_seconds=time.perf_counter()-before))
    shutil.copy2(latest, out/'terminal.pth')
    json_write(out/'completed.json', dict(**metadata, completed_epochs=epochs, successful_updates=updates,
               elapsed_seconds=time.perf_counter()-before, peak_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__ == '__main__':
    main()
