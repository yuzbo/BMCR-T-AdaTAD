#!/usr/bin/env python3
"""RFV shared bank, bounded T-local-CF, full-video smoke and fixed-action replay."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]


def prepare(args):
    from h65.paper.runtime import json_write
    from h65.paper.fasttrack import router_splits
    resources=json.loads(Path(args.resources).read_text());splits=router_splits(resources)
    ordered=list(splits['fit']);random.Random(4215).shuffle(ordered)
    mini=dict(fit=sorted(ordered[:32]),calibration=sorted(splits['calibration'])[:10],holdout=sorted(ordered[32:42]))
    base=json.loads((ROOT/'configs/wtr_fast/T-V.json').read_text())
    for label,value in [('T-U',False),('T-V',True)]:
        cfg=copy.deepcopy(base)
        cfg.update(id='rfv_'+label.lower().replace('-','_')+'_s42',comparison=label,rfv=True,
            temporal_value=value,selector='anchor' if value else 'uniform',frame_utility=value,
            primary=['actual_complete_model_flops','epoch80_full_test_mAP'],primary_endpoint_epoch=80,
            requires_evidence='rfv_plain_and_headroom',rfv_prefit_required=value)
        json_write(ROOT/'configs/rfv'/f'{label}.json',cfg)
    resources.update(wtr_review_directory=str(ROOT/'research/rfv_sprint_20260915/reviews'),
        wtr_gate_directory=str(ROOT/'research/rfv_sprint_20260915/gates'))
    protocol=dict(schema='T_VALUE_BANK_V1',seed=42,splits=splits,mini=mini,
        mini_holdout_scope='inner development holdout drawn from outer fit pool; outer20 never used by mini',
        window_sampling='one deterministic middle official window per video; full local-CF uses all windows',
        bank_rounds=[0],actions_per_state=16,local_cf_rounds=4,local_cf_pairs_per_round=16,
        graph_nodes=192,graph_width=64,graph_degree=16,head_seeds=[42,43,44],head_steps=2000,
        beta=[1.,1.05,1.1,1.2],beta_formula='post+(beta-1)*(post-anchor)',
        headroom_g0a='calibration20 middle windows; complete bounded greedy rollout; loss/regret only',
        headroom_g0b='same20 all windows; complete official-style detection AP',
        primary_endpoint=80,official_test_tuning=False,
        label='signed cls+loc improvement from one forced T exchange followed by current D100/S100/Cross/readout',
        detector_frontend='Standard768 preview; Graph192 nodes are derived evidence, not a new Raw decoder frontend')
    out=Path(args.output)
    json_write(out/'resources.json',resources);json_write(out/'protocol.json',protocol)
    print(json.dumps(dict(resources=str(out/'resources.json'),protocol=str(out/'protocol.json'),
        mini_counts={k:len(v) for k,v in mini.items()},outer_counts={k:len(v) for k,v in splits.items()})))


def run(args):
    import numpy as np
    import torch
    from h65.paper.runtime import json_write,read_config
    from h65.full.runtime import to_gpu
    from h65.rfv.hardware import initialize as initialize_hardware
    from h65.atlas.data import CharacterizationData
    from h65.rfv.bank import FixedTModel,collect_state,save_state,revision,assert_same_actions
    from h65.atlas.reference import deterministic_fp32
    from h65.paper.geometry import candidate_mask
    from h65.paper.temporal_value import to_standard
    from h65.raw.contracts import RawSelection
    protocol=json.loads(Path(args.protocol).read_text())
    resources=json.loads(Path(args.resources).read_text())
    cfg=read_config(args.config);science=revision(ROOT)
    hardware=initialize_hardware(resources)
    runtime=FixedTModel(cfg,resources,args.checkpoint,args.state)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.mode=='smoke':
        return smoke(runtime,protocol,resources,out,hardware,science)
    groups=protocol['mini' if args.cohort=='mini' else 'splits']
    roles={video:role for role,ids in groups.items() for video in ids}
    if args.mode in ('local_cf','smoke'):
        ids=list(protocol['splits']['calibration']) if args.mode=='local_cf' else [protocol['splits']['fit'][0]]
        roles={video:'calibration' if args.mode=='local_cf' else 'fit' for video in ids}
    else:ids=sorted(roles)
    ids=sorted(ids)[args.shard::args.shards]
    if not ids:raise ValueError('Empty RFV shard')
    source=CharacterizationData(runtime.model_cfg,resources,'development',ids,one_window=not args.all_windows and args.mode!='smoke')
    capture_manifest=dict(schema='T_VALUE_BANK_V1',mode=args.mode,cohort=args.cohort,source_revision=science,
        checkpoint=runtime.identity,initialization=runtime.initialization,protocol=protocol,
        config=cfg,video_ids=ids,shard=args.shard,shards=args.shards,all_windows=args.all_windows or args.mode=='smoke',
        hardware=hardware,split='training/development',outer_holdout_used=args.cohort=='full' and args.mode in ('bank','replay'))
    manifest_path=out/f'manifest_{args.shard}.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        for key in ('schema','mode','cohort','source_revision','checkpoint','protocol','config','video_ids','shard','shards','all_windows'):
            if old[key]!=capture_manifest[key]:raise ValueError('Refuse to resume a different RFV capture: '+key)
    else:json_write(manifest_path,capture_manifest)
    if args.mode=='replay':
        ref=Path(args.action_manifest)
        reference={}
        for path in sorted((ref/'groups').glob('*.json')):
            row=json.loads(path.read_text())
            reference.setdefault((row['video_id'],row['window_start_frame']),[]).append(row)
        if not reference:raise ValueError('Fixed action manifest has no states')
    else:reference={}
    total_forwards=0;files=[];processed=0;started=time.perf_counter()
    with deterministic_fp32():
        for ordinal in range(len(source)):
            cpu=source[ordinal];meta=cpu['metas'][0];video=meta['video_name'];start=int(meta['window_start_frame'])
            window_key=f'{video}__f{start:07d}'
            completed=out/'windows'/(window_key+'.json')
            if completed.exists():
                receipt=json.loads(completed.read_text());files.extend(receipt['group_files']);continue
            if args.limit_videos and processed>=args.limit_videos:break
            data=to_gpu(cpu);preview,preview_cost=runtime.preview(data);selection=runtime.seed(data,preview)
            window_files=[];rows=[];best_native=None;uniform_native=None
            if args.mode=='replay':
                items=reference.get((video,start))
                if items is None:raise ValueError('Replay window missing from fixed action manifest')
            else:items=[None]* (4 if args.mode=='local_cf' else 1)
            for round_index,ref in enumerate(items):
                if ref is not None:
                    raw=RawSelection(tuple(ref['selected_frame_ids']),tuple(ref['selected_valid']))
                    selection=to_standard(raw,selection,tuple(meta['frame_inds'].tolist()),candidate_mask(data))
                    round_index=ref['round_index']
                row,arrays,changed,native,base_native=collect_state(runtime,data,preview,selection,round_index,
                    None if ref is None else ref['action_pairs'])
                if ref is not None:assert_same_actions(ref,row)
                if uniform_native is None:
                    uniform_native=base_native
                row['preview_once_gflops']=preview_cost if not rows else 0.
                saved=save_state(out/'groups',row,arrays,roles[video],science)
                window_files.append(saved['state_key']+'.json');rows.append(saved)
                total_forwards+=row['query_forwards'];best_native=native
                if args.mode=='local_cf':
                    if row['best_gain']<=0:break
                    selection=changed
                elif args.mode!='replay':break
            receipt=dict(video_id=video,window_start_frame=start,group_files=window_files,
                initial_loss=rows[0]['base_loss'],final_loss=rows[-1]['best_loss'],
                accepted_swaps=sum(row['best_gain']>0 for row in rows) if args.mode=='local_cf' else None,
                loss_gain=sum(rows[0]['base_loss'])-sum(rows[-1]['best_loss']),
                query_forwards=sum(r['query_forwards'] for r in rows),
                label_query_gflops=sum(r['label_query_gflops'] for r in rows)+preview_cost,
                replay_max_error=max(r['replay_max_error'] for r in rows),no_op_error=max(r['no_op_error'] for r in rows))
            if args.all_windows or args.mode=='smoke':
                post=copy.deepcopy(runtime.model_cfg.post_processing);post.sliding_window=True
                prediction={}
                for name,features in [('uniform',uniform_native),('local_cf',best_native if args.mode=='local_cf' else uniform_native)]:
                    head=runtime.model.readout.predictions(features,data['masks'],data['metas'])
                    prediction[name]=runtime.model.readout.post_processing(head,data['metas'],post,source.class_map)
                receipt['predictions']=prediction
                receipt['prediction_head_calls']=2
            json_write(completed,receipt);files.extend(window_files);processed+=1
            json_write(out/f'progress_{args.shard}.json',dict(completed_windows=len(list((out/'windows').glob('*.json'))),
                total_shard_windows=len(source),last_video=video,query_forwards_this_process=total_forwards,
                wall_seconds=time.perf_counter()-started))
            print(json.dumps(dict(video=video,window=ordinal+1,total=len(source),groups=len(rows),loss_gain=receipt['loss_gain'])),flush=True)
    complete=len([p for p in (out/'windows').glob('*.json') if json.loads(p.read_text())['video_id'] in ids])==len(source)
    if complete:
        if args.all_windows:finish_ap(runtime,source,out,args)
        json_write(out/f'completed_{args.shard}.json',dict(source_revision=science,video_ids=ids,
            group_files=sorted(set(files)),windows=len(source),checkpoint=runtime.identity,mode=args.mode))
    print(json.dumps(dict(complete=complete,groups=len(set(files)),windows_this_process=processed,output=str(out))),flush=True)


def smoke(runtime,protocol,resources,out,hardware,science):
    """Use the real evaluator/serialization path on one complete fit video."""
    import torch
    from h65.atlas.data import CharacterizationData
    from h65.full.runtime import to_gpu
    from h65.paper.profile import calibrate_costs
    from h65.paper.evaluation import evaluate
    from h65.paper.runtime import json_write
    from h65.paper.geometry import candidate_mask
    chosen=protocol['splits']['fit'][0]
    source=CharacterizationData(runtime.model_cfg,resources,'development',[chosen],one_window=False)
    source.ground_truth(out/'smoke_ground_truth.json')
    full=None
    for index in range(len(source)):
        cpu=source[index]
        if int(candidate_mask(cpu).sum())==768:full=to_gpu(cpu);break
    if full is None:raise ValueError('The registered smoke video needs a full window for cost calibration')
    calibrate_costs(runtime.model,full)
    no_gt={key:value for key,value in full.items() if not key.startswith('gt_')}
    no_gt['metas']=[{key:value for key,value in meta.items()
        if key!='characterization_gt' and not key.startswith('gt_')} for meta in full['metas']]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        _,detail=runtime.model.predictions(no_gt,runtime.cfg['fixed_plan'])
    if not bool(candidate_mask(no_gt).gather(1,detail['selection'].indices)[detail['selection'].valid].all()):
        raise ValueError('No-GT inference selected invalid padding')
    cfg=copy.deepcopy(runtime.model_cfg);res=copy.deepcopy(resources)
    cfg.dataset.test.ann_file=str(out/'smoke_ground_truth.json')
    cfg.dataset.test.subset_name='training';cfg.dataset.test.data_path=res['datasets']['thumos']['train_videos']
    cfg.evaluation.ground_truth_filename=str(out/'smoke_ground_truth.json');cfg.evaluation.subset='training'
    res['datasets']['thumos']['test_ids']=[chosen]
    metadata=dict(hardware,source_revision=science,epoch=0,checkpoint_state='ema',
        rfv_scope='complete fit-video smoke, not official test or scientific effectiveness',checkpoint=runtime.identity)
    record,_=evaluate(runtime.model,cfg,res,out/'evaluation',metadata,force_plan=runtime.cfg['fixed_plan'],profile=False)
    if record['test_videos']!=1 or record['test_windows']!=len(source):raise ValueError('Smoke omitted video windows')
    json_write(out/'passed.json',dict(scope='complete_video_e2e_smoke',passed=True,source_revision=science,
        config=runtime.cfg,video=chosen,windows=len(source),postprocess_json_metrics=True,no_gt_inference=True,
        evidence=str(out/'evaluation/completed.json'),scientific_gate=False))
    print(json.dumps(dict(smoke='PASS',video=chosen,windows=len(source))),flush=True)


def finish_ap(runtime,source,out,args):
    from h65.paper.runtime import json_write
    from h65.paper.evaluation import merge_windows
    from opentad.evaluations import build_evaluator
    if args.shards!=1:raise ValueError('AP aggregation must run after a single complete video cohort, not per shard')
    source.ground_truth(out/'development_ground_truth.json')
    summaries={}
    for name in ('uniform','local_cf'):
        result={}
        for path in sorted((out/'windows').glob('*.json')):
            row=json.loads(path.read_text())
            for video,segments in row['predictions'][name].items():result.setdefault(video,[]).extend(segments)
        if set(result)!=set(source.by_video):raise ValueError('Full-video smoke/AP omitted videos')
        post=copy.deepcopy(runtime.model_cfg.post_processing);post.sliding_window=True
        predictions=dict(results=merge_windows(result,post))
        json_write(out/(name+'_predictions.json'),predictions)
        evaluation=copy.deepcopy(dict(runtime.model_cfg.evaluation))
        evaluation.update(ground_truth_filename=str(out/'development_ground_truth.json'),subset='training')
        # Read back the exact serialized payload used for official-style metrics.
        serialized=json.loads((out/(name+'_predictions.json')).read_text())
        metrics=build_evaluator(dict(prediction_filename=serialized,**evaluation)).evaluate()
        summaries[name]=metrics
    report=dict(mode=args.mode,scope='training-side complete videos/all official windows',videos=len(source.by_video),
        windows=len(source),metrics=summaries,checkpoint=runtime.identity,postprocess_json_metrics=True,
        scientific_headroom_claim=args.mode=='local_cf')
    json_write(out/'full_video_metrics.json',report)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','bank','local_cf','replay','smoke'])
    parser.add_argument('--resources',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--protocol');parser.add_argument('--config',default=str(ROOT/'configs/rfv/T-U.json'))
    parser.add_argument('--checkpoint');parser.add_argument('--state',choices=['learned','ema'],default='ema')
    parser.add_argument('--cohort',choices=['mini','full'],default='mini');parser.add_argument('--action-manifest')
    parser.add_argument('--all-windows',action='store_true');parser.add_argument('--limit-videos',type=int)
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    if args.mode=='prepare':return prepare(args)
    if not args.protocol:parser.error('--protocol is required')
    if args.mode=='replay' and not args.action_manifest:parser.error('--action-manifest is required for fixed replay')
    if not 0<=args.shard<args.shards:parser.error('Invalid shard')
    if (args.all_windows or args.mode=='smoke') and args.shards!=1:parser.error('Use a whole-cohort allocation for AP/smoke')
    run(args)


if __name__=='__main__':main()
