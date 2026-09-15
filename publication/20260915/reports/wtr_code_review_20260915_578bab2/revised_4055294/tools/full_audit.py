"""Training-only128-window swap audit, with video-isolated diagnostic calibration."""
import argparse
import collections
import json
import random
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'upstream')]
import numpy as np
import torch
from scipy.stats import spearmanr
from opentad.datasets.builder import build_dataset, collate
from h65.transport import sample_rates, interpolate_irregular
from h65.full.geometry import maps_for
from h65.full.model import FormalH65
from h65.full.runtime import config, RUNS, initialize_gpu, to_gpu, json_write
from h65.full.utility import counterfactual_targets
from h65.full.course import BMCR80_RECIPE, COMPONENT_ORDER, warm_origin


def audit_dataset(cfg):
    data = cfg.dataset.val.copy()
    data.subset_name = 'training'
    data.data_path = cfg.dataset.train.data_path
    data.ioa_thresh = 0.
    dataset = build_dataset(data)
    # Include background windows, and clip every overlapping GT to the observed
    # window before deriving utility; never map off-window GT onto an endpoint.
    strata = collections.defaultdict(list)
    for index, (name, info, annotation, centers) in enumerate(dataset.data_list):
        boxes = annotation['gt_segments'].copy()
        valid = (boxes[:, 0] < centers[-1]) & (boxes[:, 1] > centers[0])
        boxes = boxes[valid].clip(centers[0], centers[-1])
        labels = annotation['gt_labels'][valid]
        dataset.data_list[index][2] = dict(gt_segments=boxes, gt_labels=labels)
        duration = (boxes[:, 1]-boxes[:, 0])/4
        category = ('background' if not len(boxes) else 'short' if (duration <= 16).any() else
                    'repeated' if len(set(labels.tolist())) < len(labels) else 'general')
        strata[category].append(index)
    rng = random.Random(3407)
    chosen = []
    for category in ('short', 'repeated', 'background', 'general'):
        rng.shuffle(strata[category])
        chosen.extend(strata[category][:32])
    remainder = [i for i in range(len(dataset)) if i not in chosen]
    rng.shuffle(remainder)
    chosen = (chosen + remainder)[:128]
    if len(chosen) != 128:
        raise RuntimeError('training-only audit has fewer than128 windows')
    return dataset, chosen


def sensitivity(model, sample):
    masks = sample['masks']
    selection = sample_rates(masks.float()*0, masks, 384, alpha=0)
    with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        features, selected = model.encode(sample['inputs'], selection, need_input_grad=True)
        mapped = [m.to_rank(gt) for m, gt in zip(maps_for(selection, masks), sample['gt_segments'])]
        with torch.autocast('cuda', enabled=False):
            losses = model.detector.forward_train(features.float(), selection.valid, sample['metas'], mapped, sample['gt_labels'])
        fields = []
        for kind in ('cls', 'reg'):
            gradient = torch.autograd.grad(losses[kind+'_loss'], selected, retain_graph=kind=='cls')[0]
            values = (selected.detach()*gradient.detach()).abs().mean((1, 2, 4, 5))[0]
            valid = selection.valid[0]
            query = torch.arange(masks.shape[1], device=masks.device).float()
            fields.append(interpolate_irregular(values[valid][None], selection.indices[0, valid].float(), query)[0])
    return torch.stack(fields, -1).cpu().numpy()


def candidates_for(selection, condition, boxes, labels):
    feasible = torch.where(condition['member'][0] & condition['feasible'][0])[0]
    if not len(feasible):
        return [], {}
    distance = (feasible[:, None]-boxes.flatten()[None]).abs().amin(-1) if boxes.numel() else feasible.float()*0+1e6
    inside = ((feasible[:, None]>=boxes[:, 0]) & (feasible[:, None]<boxes[:, 1])).any(-1) if boxes.numel() else feasible.bool() & False
    categories = dict(boundary=distance<=4, core=inside & (distance>4), background=~inside & (distance>4))
    short = boxes[(boxes[:, 1]-boxes[:, 0])<=16]
    categories['short'] = ((feasible[:, None]>=short[:, 0]) & (feasible[:, None]<short[:, 1])).any(-1) if short.numel() else feasible.bool() & False
    repeated_labels = [label for label in labels.unique() if int((labels==label).sum())>1]
    repeated = boxes[torch.stack([labels==label for label in repeated_labels]).any(0)] if repeated_labels else boxes[:0]
    categories['repeated'] = ((feasible[:, None]>=repeated[:, 0]) & (feasible[:, None]<repeated[:, 1])).any(-1) if repeated.numel() else feasible.bool() & False
    chosen, names = [], {}
    for name, valid in categories.items():
        available = feasible[valid]
        if len(available):
            index = int(available[torch.randint(len(available), ())])
            chosen.append(index)
            names.setdefault(index, []).append(name)
    return sorted(set(chosen)), names


def correlation(a, b):
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:
        return None
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['s', 'b'], required=True)
    parser.add_argument('--total-epochs', type=int, choices=[60,80], default=60)
    args = parser.parse_args()
    hardware = initialize_gpu()
    cfg = config(args.backbone)
    out = RUNS/f'{args.backbone}_audit'
    out.mkdir(parents=True, exist_ok=True)
    model = FormalH65(cfg.model, variant='h65').cuda().eval().requires_grad_(False)
    checkpoint = RUNS/f'{args.backbone}_warm/terminal.pth'
    payload = torch.load(checkpoint, map_location='cpu')
    if payload['completed_epochs'] != 20:
        raise RuntimeError('audit requires full20-epoch warm EMA')
    origin = warm_origin(payload,checkpoint,args.backbone) if args.total_epochs == 80 else None
    model.load_state_dict(payload['state_dict_ema'], strict=True)
    del payload
    dataset, indices = audit_dataset(cfg)
    video_ids = sorted({dataset.data_list[i][0] for i in indices})
    random.Random(3407).shuffle(video_ids)
    fit_ids = set(video_ids[:int(.8*len(video_ids))])
    records = []
    for number, index in enumerate(indices):
        sample = to_gpu(collate([dataset[index]]))
        fields = sensitivity(model, sample)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            output, provisional, _, _ = model.route(sample['inputs'], sample['masks'], dict(alpha=1., adapt=1.))
            condition = model.scout.condition(output, provisional, sample['masks'])
            candidates, categories = candidates_for(provisional, condition, sample['gt_segments'][0], sample['gt_labels'][0])
            measurements = counterfactual_targets(model, sample['inputs'], sample['masks'], sample['metas'],
                             sample['gt_segments'], sample['gt_labels'], provisional, condition, candidates=candidates)
        video = sample['metas'][0]['video_name']
        for measurement in measurements:
            if measurement['candidate'] != measurement['remove']:
                continue
            remove, insert = measurement['remove'], measurement['insert']
            action = output['action_logits'][0].float().sigmoid()
            transition = output['transition_logits'][0].float()
            feature = [float(action[remove]-action[insert]), float(transition[remove]-transition[insert]),
                       float((remove-insert)/768), *fields[remove].tolist(), *fields[insert].tolist()]
            records.append(dict(**measurement, video=video, window_index=index, strata=categories[remove],
                                partition='fit' if video in fit_ids else 'holdout',
                                attribution_delta=(fields[remove]-fields[insert]).tolist(), diagnostic_features=feature))
        json_write(out/'observations.json', records)
        print(f'audit {number+1}/128: {video}, {len(measurements)} labels', flush=True)
    fitting = [record for record in records if record['partition']=='fit']
    held = [record for record in records if record['partition']=='holdout']
    target = np.array([record['target'] for record in fitting])
    if len(fitting)<2 or len(held)<2:
        raise RuntimeError('audit did not produce a video-isolated fit and holdout sample')
    scales = []
    for channel in range(2):
        magnitude = np.abs(target[:, channel])
        nonzero = magnitude[magnitude>1e-8]
        scales.append(max(float(np.median(nonzero)), 1e-4) if len(nonzero) else 1.)
    # Diagnostic ridge is trained ONLY on fit videos; it never initializes BMCR.
    x = np.array([r['diagnostic_features'] for r in fitting])
    x_test = np.array([r['diagnostic_features'] for r in held])
    mean, std = x.mean(0), x.std(0).clip(1e-6)
    x, x_test = (x-mean)/std, (x_test-mean)/std
    x, x_test = np.c_[np.ones(len(x)), x], np.c_[np.ones(len(x_test)), x_test]
    coefficients = np.linalg.solve(x.T@x + np.eye(x.shape[1]), x.T@(target/np.array(scales)))
    prediction = x_test@coefficients
    actual = np.array([r['target'] for r in held])
    attr = np.array([r['attribution_delta'] for r in held])
    diagnostics = dict(fit_videos=sorted(fit_ids), holdout_videos=sorted(set(video_ids)-fit_ids),
                        fit_swaps=len(fitting), holdout_swaps=len(held), windows=128,
                        strata_counts=dict(collections.Counter(s for r in records for s in r['strata'])),
                        attribution_spearman=[correlation(attr[:, i], actual[:, i]) for i in range(2)],
                        diagnostic_ridge_spearman=[correlation(prediction[:, i], actual[:, i]) for i in range(2)],
                        gt_instance_count=sum(r['gt_count'] for r in records),
                        matched_gt_before=sum(r['matched_before'] for r in records),
                        interpretation='training-only calibration and falsification; no test tuning or performance claim')
    provenance = dict(recipe=BMCR80_RECIPE,backbone=args.backbone,warm_origin=origin) if args.total_epochs == 80 else {}
    json_write(out/'scales.json', dict(scales=scales, source='median nonzero absolute signed utility on fit videos only',
                                     component_order=COMPONENT_ORDER, **provenance))
    json_write(out/'completed.json', dict(**hardware, **diagnostics, **provenance))


if __name__ == '__main__':
    main()
