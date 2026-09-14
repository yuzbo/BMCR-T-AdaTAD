#!/usr/bin/env python3
"""Run resumable frozen-model measurements; never optimize model parameters."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'upstream')]

import numpy as np
import torch
from torch.utils.data import DataLoader
from h65.atlas.data import CharacterizationData, window_metadata
from h65.atlas.reference import FrozenReference
from h65.full.runtime import to_gpu
from h65.paper.runtime import json_write


def identity(value):
    return value


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--resources', required=True)
    p.add_argument('--backbone', choices=['s', 'b'], required=True)
    p.add_argument('--mode', choices=['baseline', 'preflight', 'calibration', 'population', 'allocation', 'recovery'], required=True)
    p.add_argument('--split', choices=['development', 'publication'], default='publication')
    p.add_argument('--output', required=True)
    p.add_argument('--limit-videos', type=int)
    p.add_argument('--limit-windows', type=int)
    p.add_argument('--one-window',action='store_true')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--axis', choices=['T', 'D', 'S'])
    p.add_argument('--protocol', default=str(ROOT/'research/atlas_20260915/protocol.json'))
    return p.parse_args()


def evaluate_records(reference, dataset, folder, expected_windows):
    from opentad.evaluations import build_evaluator
    from h65.paper.evaluation import merge_windows
    files = sorted((folder/'windows').glob('*.json'))
    if len(files) != expected_windows:
        raise RuntimeError(f'Full AP requires all {expected_windows} windows; found {len(files)}')
    predictions = {v: [] for v in dataset.by_video}
    costs = []
    for path in files:
        row = json.loads(path.read_text())
        for name, values in row['predictions'].items():
            predictions[name].extend(values)
        costs.append(row['gflops'])
    post = copy.deepcopy(reference.cfg.post_processing)
    post.sliding_window = True
    predictions = merge_windows(predictions, post)
    ground_truth = folder/'ground_truth.json'
    dataset.ground_truth(ground_truth)
    result = dict(results=predictions)
    json_write(folder/'result_detection.json', result)
    ecfg = copy.deepcopy(reference.cfg.evaluation)
    ecfg.ground_truth_filename = str(ground_truth)
    ecfg.subset = 'training' if dataset.split == 'development' else 'validation'
    ecfg.prediction_filename = result
    ecfg.thread = 4
    metrics = build_evaluator(ecfg).evaluate()
    record = dict(metrics=metrics, videos=len(predictions), windows=len(files),
                  mean_gflops=float(np.mean(costs)), total_gflops=float(np.sum(costs)),
                  precision='FP32; TF32 disabled', state='official state_dict_ema',
                  training_updates=0, checkpoint=reference.model.provenance,
                  source_revision=os.environ.get('ATLAS_SOURCE_REVISION', 'working-tree'),
                  completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    json_write(folder/'completed.json', record)
    print(json.dumps(record), flush=True)
    return record


def main():
    args = arguments()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    resources = json.loads(Path(args.resources).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reference = FrozenReference(args.backbone, resources)
    ids = sorted(resources['datasets']['thumos']['train_ids' if args.split == 'development' else 'test_ids'])
    if args.limit_videos:
        ids = sorted(np.random.default_rng(42).choice(ids,min(len(ids),args.limit_videos),replace=False).tolist())
    dataset = CharacterizationData(reference.cfg, resources, args.split, ids,one_window=args.one_window)
    if args.limit_windows:
        dataset.indices = dataset.indices[:args.limit_windows]
    full_windows = len(dataset)
    if args.split == 'publication' and not args.limit_videos and not args.limit_windows:
        if len(ids) != 211 or full_windows != 792:
            raise RuntimeError(f'Publication population must be 211/792, got {len(ids)}/{full_windows}')
    dataset.indices = dataset.indices[args.shard::args.shards]
    manifest = dict(arguments=vars(args), videos=len(ids), total_windows=full_windows,
                    shard_windows=len(dataset), checkpoint=reference.model.provenance,
                    source_revision=os.environ.get('ATLAS_SOURCE_REVISION', 'working-tree'))
    json_write(output/f'manifest_{args.shard}.json', manifest)
    (output/'windows').mkdir(exist_ok=True)
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=args.workers,
                        collate_fn=identity, pin_memory=True)
    start = time.perf_counter()
    completed = 0
    for cpu in loader:
        meta = window_metadata(cpu)
        target = output/'windows'/f'{meta["window_index"]:05d}.json'
        if target.exists():
            completed += 1
            continue
        data = to_gpu(cpu)
        if args.mode == 'baseline':
            measured = reference.execute(data)
            record = dict(meta=meta, loss_cls_reg=measured['losses'].tolist(),
                          gflops=measured['gflops'], model_ms=measured['model_ms'],
                          predictions=reference.postprocess(measured, data, dataset.class_map))
        else:
            from h65.atlas.experiments import run_window
            record = run_window(reference, data, dataset.class_map, args, resources)
        json_write(target, record)
        completed += 1
        progress = dict(completed=completed, shard_windows=len(dataset),
                        total_windows=full_windows, seconds=time.perf_counter()-start,
                        window=meta['window_index'], backbone=args.backbone, mode=args.mode)
        json_write(output/f'progress_{args.shard}.json', progress)
        if completed == 1 or completed % 20 == 0:
            print(json.dumps(progress), flush=True)
    json_write(output/f'accounting_{args.shard}.json', reference.accounting())
    json_write(output/f'shard_{args.shard}_done.json', dict(windows=len(dataset), completed=completed))
    if args.mode == 'baseline' and args.shards == 1:
        evaluate_records(reference, dataset, output, full_windows)


if __name__ == '__main__':
    main()
