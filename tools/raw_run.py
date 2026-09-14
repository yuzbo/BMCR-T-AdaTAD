#!/usr/bin/env python3
"""Resumable Raw-v1 technical gate and training-side counterfactual mini-bank."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'upstream')]
import numpy as np
import torch
from h65.raw.contracts import (RawSelection,preview_timeline,proposals,physical_uniform,
    geometric_swaps,swap,repartition,tubelets)
from h65.raw.data import EpisodeData,RGBReader
from h65.raw.model import FrozenRawModel
from h65.raw.value import descriptors
from h65.paper.runtime import json_write
from h65.atlas.reference import deterministic_fp32


def cpu_data(data,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else
        [x.to(device) if isinstance(x,torch.Tensor) else x for x in v] if isinstance(v,list) else v
        for k,v in data.items()}


def close_result(a,b):
    if not torch.allclose(a['recovered'],b['recovered'],atol=1e-5,rtol=1e-5):
        raise RuntimeError('Same-frame recovered features differ')
    for xs,ys in zip(a['prediction'],b['prediction']):
        for x,y in zip(xs,ys):
            if not torch.allclose(x,y,atol=1e-5,rtol=1e-5):
                raise RuntimeError('Same-frame detector predictions differ')


def legacy_result(raw, data, selection, preview):
    from h65.frame.geometry import make_anchors,make_queries
    from h65.paper.geometry import scout_context
    with deterministic_fp32():
        native,levels,trace = raw.model.encoder.encode(data['inputs'],selection,raw.plan,capture=True)
        anchors = make_anchors(native,selection,data['masks'],data['metas'],trace)
        queries = make_queries(data['masks'],data['metas'],selection)
        context = scout_context(preview,data['masks'],False)
        recovered = raw.model.decoder(anchors,queries,context,levels)
        prediction = raw.model.readout.predictions(recovered,data['masks'],data['metas'])
    return dict(recovered=recovered,prediction=prediction),context


def gate_window(raw,dataset,ordinal):
    ep = dataset.episode(ordinal)
    reader = RGBReader(ep)
    timeline = preview_timeline(ep)
    original = cpu_data(dataset.source[ordinal],raw.device)
    if tuple(int(i) for i in original['metas'][0]['frame_inds']) != ep.official_frame_ids:
        raise RuntimeError('Header-only episode changed official frame IDs')
    target = dataset.target_data(ep,raw.device)
    if not torch.equal(target['masks'],original['masks']) or not torch.equal(target['gt_segments'][0],original['gt_segments'][0]):
        raise RuntimeError('Episode GT/mask projection differs from the Atlas contract')
    with deterministic_fp32():
        original_preview = raw.model.encoder.preview(original['inputs'],original['masks'])
        selected = raw.model.encoder.select(original_preview,original['masks'],384,'uniform',original['metas'])
    ids = tuple(ep.official_frame_ids[i] for i in selected.indices[0].tolist())
    selection = RawSelection(ids,tuple(selected.valid[0].tolist()))
    rgb = reader.read(ids)
    official_rgb = original['inputs'].index_select(3,selected.indices[0]).cpu()
    pixel_error = float((rgb-official_rgb).abs().max())
    if pixel_error != 0:
        raise RuntimeError(f'Raw reader differs from official RGB: max error {pixel_error}')
    baseline,context = legacy_result(raw,original,selected,original_preview)
    bridge = raw.execute(ep,selection,reader,timeline,original_preview,target,context_override=context,rgb_override=rgb)
    close_result(baseline,bridge)
    original_predictions = raw.postprocess(baseline,ep,dataset.source.class_map)
    bridge_predictions = raw.postprocess(bridge,ep,dataset.source.class_map)
    max_feature_error = float((baseline['recovered']-bridge['recovered']).abs().max())
    del original,baseline,bridge,original_preview,rgb,official_rgb,context
    preview,preview_cost = raw.preview(reader,timeline)
    variants = []
    for domain in ('O','R+'):
        proposal = proposals(ep,timeline,domain)
        uniform = physical_uniform(ep,proposal)
        result = raw.execute(ep,uniform,reader,timeline,preview,target)
        replay = raw.execute(ep,uniform,reader,timeline,preview,target)
        close_result(result,replay)
        if not torch.equal(result['loss'],replay['loss']):
            raise RuntimeError('No-op CF changed the task loss')
        pairs = geometric_swaps(ep,proposal,uniform)
        feature = descriptors(ep,timeline,preview,proposal,uniform,pairs)
        if not torch.isfinite(feature).all():
            raise RuntimeError('Nonfinite public descriptor')
        changes=[]
        if pairs:
            changed = swap(uniform,*pairs[0],proposal)
            action = raw.execute(ep,changed,reader,timeline,preview,target)
            changes.append(dict(remove=pairs[0][0],insert=pairs[0][1],gain=(result['loss']-action['loss']).tolist(),
                                **repartition(uniform,changed,ep.fps)))
        variants.append(dict(domain=domain,proposal_size=len(proposal.frame_ids),selection=asdict(uniform),
            tubelets=asdict(tubelets(uniform,ep.fps)),loss=result['loss'].tolist(),
            inference_gflops=result['gflops']+preview_cost,preview_gflops=preview_cost,
            predictions=raw.postprocess(result,ep,dataset.source.class_map),changes=changes,
            descriptor_shape=list(feature.shape),noop_loss_error=0.))
    return dict(passed=True,episode=ep.record(),legacy_predictions=original_predictions,bridge_predictions=bridge_predictions,
                max_rgb_error=pixel_error,max_recovered_error=max_feature_error,variants=variants,io=reader.accounting())


def gate_ap(raw,dataset,rows,output):
    from h65.paper.evaluation import merge_windows
    from h65.atlas.statistics import ap_cache
    ids = sorted(dataset.source.by_video)
    gt=output/'ground_truth.json'
    dataset.source.ground_truth(gt)
    post=copy.deepcopy(raw.cfg.post_processing);post.sliding_window=True
    metrics={}
    for label in ('legacy_predictions','bridge_predictions'):
        pred={v:[] for v in ids}
        for row in rows:
            for v,values in row[label].items():
                pred[v].extend(values)
        pred=merge_windows(pred,post)
        _,metrics[label]=ap_cache(pred,gt,'training',ids)
        json_write(output/f'{label}.json',dict(results=pred))
    if metrics['legacy_predictions'] != metrics['bridge_predictions']:
        raise RuntimeError('Full-video official AP differs across reader paths')
    return dict(metrics=metrics,official_ap_cache_matches=True)


def bank_window(raw,dataset,ordinal,protocol,output,revision,pair_count=4):
    ep=dataset.episode(ordinal)
    split=next(k for k,v in protocol['splits'].items() if ep.video_id in v)
    reader=RGBReader(ep);timeline=preview_timeline(ep)
    preview,preview_cost=raw.preview(reader,timeline)
    target=dataset.target_data(ep,raw.device)
    count=0;group_files=[]
    official=proposals(ep,timeline,'O')
    common_uniform=physical_uniform(ep,official)
    common_perturbations=geometric_swaps(ep,official,common_uniform)
    common_states=[common_uniform]
    if common_perturbations:
        common_states.append(swap(common_uniform,*common_perturbations[len(common_perturbations)//2],official))
    for domain in ('O','R+'):
        proposal=proposals(ep,timeline,domain)
        for state_id,selection in enumerate(common_states):
            path=output/'groups'/f'{ep.key}__{domain.replace("+","plus")}__{state_id}.json'
            group_files.append(path.name)
            if path.exists():
                saved=json.loads(path.read_text())
                if saved['source_revision'] != revision:
                    raise RuntimeError('Cannot resume a CF bank under a different source revision')
                count+=len(saved['actions']);continue
            # Take four evenly spread actions from the same 16-pair proposal budget.
            candidates=geometric_swaps(ep,proposal,selection,limit=16,round_index=state_id)
            stride=16//pair_count
            pairs=[candidates[i] for i in range(0,len(candidates),stride)]
            x=descriptors(ep,timeline,preview,proposal,selection,pairs).cpu().tolist()
            base=raw.execute(ep,selection,reader,timeline,preview,target)
            baseline_loss=base['loss'].clone();base_cost=base['gflops'];del base
            actions=[]
            for index,pair in enumerate(pairs):
                changed=swap(selection,*pair,proposal)
                result=raw.execute(ep,changed,reader,timeline,preview,target)
                replay_error=None
                if index == 0:
                    replay=raw.execute(ep,changed,reader,timeline,preview,target)
                    close_result(result,replay)
                    replay_error=float((result['loss']-replay['loss']).abs().max())
                    if replay_error != 0:
                        raise RuntimeError('CF replay noise is nonzero in the deterministic FP32 path')
                    del replay
                gain=baseline_loss-result['loss']
                actions.append(dict(remove=pair[0],insert=pair[1],descriptor=x[index],
                    gain_cls_loc=gain.tolist(),changed_loss=result['loss'].tolist(),gflops=result['gflops'],
                    replay_max_error=replay_error,**repartition(selection,changed,ep.fps)))
                del result
            row=dict(episode=ep.record(),split=split,domain=domain,state_id=state_id,selection=asdict(selection),
                preview=asdict(timeline),proposal=asdict(proposal),baseline_loss=baseline_loss.tolist(),
                actions=actions,source_revision=revision,checkpoint=raw.provenance,
                preview_gflops=preview_cost,base_gflops=base_cost,training_updates=0,
                semantics='signed full downstream cls/loc gain; actual ordered pair/pack repartition included')
            json_write(path,row);count+=len(actions)
            print(json.dumps(dict(group=path.name,swaps=len(actions),split=split)),flush=True)
    json_write(output/'io'/f'{ep.key}.json',reader.accounting())
    return count,group_files


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--resources',required=True);p.add_argument('--protocol',required=True)
    p.add_argument('--stage',choices=['gate','mini-bank','full-bank'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--gate-receipt')
    p.add_argument('--mini-review')
    p.add_argument('--limit-videos',type=int);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    protocol=json.loads(Path(args.protocol).read_text());resources=json.loads(Path(args.resources).read_text())
    if args.stage != 'gate':
        gate=json.loads(Path(args.gate_receipt).read_text()) if args.gate_receipt else {}
        if not gate.get('passed') or gate.get('videos',0) < 6:
            raise ValueError('Mini-bank requires a passed complete six-video development GPU gate')
    if args.stage == 'full-bank':
        review=json.loads(Path(args.mini_review).read_text()) if args.mini_review else {}
        if not review.get('proceed_to_full_bank') or not review.get('rationale'):
            raise ValueError('Full bank requires an explicit review of the measured mini-bank signal')
    ids=(protocol['gate_videos'] if args.stage=='gate' else
         sorted(sum(protocol['mini' if args.stage=='mini-bank' else 'splits'].values(),[])))
    if args.limit_videos:
        ids=ids[:args.limit_videos]
    if any(v not in resources['datasets']['thumos']['train_ids'] for v in ids):
        raise ValueError('Technical/CF development jobs must use training videos only')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    revision=(ROOT/'CODE_REVISION').read_text().strip()
    raw=FrozenRawModel(resources)
    dataset=EpisodeData(raw.cfg,resources,ids)
    manifest=dict(stage=args.stage,ids=ids,windows=len(dataset),source_revision=revision,protocol=protocol,
                  checkpoint=raw.provenance,gpu=torch.cuda.get_device_name(0),args=vars(args),training_updates=0)
    manifest_path=output/f'manifest_{args.shard}.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Output manifest differs; use a new output directory')
    json_write(manifest_path,manifest)
    if args.stage == 'gate':
        rows=[]
        for ordinal in range(len(dataset)):
            ep=dataset.episode(ordinal)
            path=output/'windows'/f'{ep.key}.json'
            if path.exists():
                row=json.loads(path.read_text())
            else:
                row=gate_window(raw,dataset,ordinal);json_write(path,row)
            rows.append(row)
            print(json.dumps(dict(passed=ep.key,windows_completed=len(rows),total=len(dataset))),flush=True)
        ap=gate_ap(raw,dataset,rows,output)
        receipt=dict(passed=True,videos=len(ids),windows=len(rows),source_revision=revision,
            gpu=torch.cuda.get_device_name(0),axes=[384,192,384,768],training_updates=0,**ap)
        json_write(output/'passed.json',receipt)
    else:
        ordinals=[dataset.source.indices.index(values[len(values)//2]) for name,values in sorted(dataset.source.by_video.items())]
        selected=ordinals[args.shard::args.shards]
        total=0;group_files=[]
        for ordinal in selected:
            count,files=bank_window(raw,dataset,ordinal,protocol,output,revision,4 if args.stage=='mini-bank' else 8)
            total+=count;group_files.extend(files)
            json_write(output/f'progress_{args.shard}.json',dict(swaps=total,last_video=dataset.episode(ordinal).video_id))
        json_write(output/f'completed_{args.shard}.json',dict(swaps=total,videos=len(selected),source_revision=revision,group_files=group_files,
            status='mini-bank collected; scientific signal review required before full bank',training_updates=0))


if __name__ == '__main__':
    main()
