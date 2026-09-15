"""Full target-split evaluation, usable inline in the training allocation."""
import copy
import time
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from opentad.datasets.builder import build_dataset,collate
from opentad.evaluations import build_evaluator
from opentad.models.utils.post_processing import batched_nms
from h65.full.runtime import to_gpu
from .runtime import json_write
from .interventions import evaluation_state
from .profile import measure,execution_flops
from .geometry import candidate_mask


def merge_windows(result,post_cfg):
    if not post_cfg.sliding_window or post_cfg.nms is None:return result
    merged={}
    for name,rows in result.items():
        if not rows:merged[name]=[];continue
        segments=torch.Tensor([r['segment'] for r in rows]);scores=torch.Tensor([r['score'] for r in rows])
        classes=[];labels=[]
        for row in rows:
            if row['label'] not in classes:classes.append(row['label'])
            labels.append(classes.index(row['label']))
        segments,scores,labels=batched_nms(segments,scores,torch.Tensor(labels),**post_cfg.nms)
        merged[name]=[dict(segment=[round(float(x),2) for x in seg],label=classes[int(label)],score=round(float(score),4))
                      for seg,score,label in zip(segments,scores,labels)]
    return merged


@torch.no_grad()
def evaluate(model,model_cfg,resources,out,metadata,force_plan=None,profile=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dataset=build_dataset(model_cfg.dataset.test)
    expected=set(resources['datasets'][model.config['dataset']]['test_ids'])
    actual={x[0] for x in dataset.data_list}
    if actual!=expected:raise RuntimeError(f'Incomplete test dataset: expected {len(expected)}, built {len(actual)}')
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2,collate_fn=collate,pin_memory=True)
    post=copy.deepcopy(model_cfg.post_processing);post.sliding_window=model.config['dataset']=='thumos'
    if model.config['dataset']=='anet':
        from opentad.models.utils.post_processing.classifier import CUHKANETClassifier
        ext=CUHKANETClassifier(path=resources['datasets']['anet']['classifier'],topk=2)
    else:ext=dataset.class_map
    result={};samples={};plan_counts={};window_costs=[];full_costs=[];model_times=[];begin=time.perf_counter()
    from h65.frame.geometry import source_times
    window_file=out/f'window_execution_{metadata["slurm_job_id"]}.jsonl'
    window_file.write_text('');data_wait=h2d_time=post_time=0.;last_end=time.perf_counter()
    with evaluation_state(model),torch.autocast('cuda',dtype=torch.bfloat16):
        for index,cpu in enumerate(loader):
            fetched=time.perf_counter();data_wait+=fetched-last_end
            data=to_gpu(cpu);torch.cuda.synchronize();model_start=time.perf_counter();h2d_time+=model_start-fetched
            prediction,detail=model.predictions(data,force_plan);torch.cuda.synchronize()
            model_times.append((time.perf_counter()-model_start)*1000)
            post_start=time.perf_counter();window=model.readout.post_processing(prediction,data['metas'],post,ext);post_time+=time.perf_counter()-post_start
            for name,rows in window.items():result.setdefault(name,[]).extend(rows)
            key=detail['plan']['id'];plan_counts[key]=plan_counts.get(key,0)+1
            count=int(candidate_mask(data).sum());kind='full' if count==768 else 'partial' if count>384 else 'short'
            cost=execution_flops(model,detail)/1e9;window_costs.append(cost)
            if kind=='full':full_costs.append(cost)
            if kind not in samples:samples[kind]=(index,cpu)
            real_times=source_times(candidate_mask(data),data['metas']).gather(1,detail['selection'].indices)[detail['selection'].valid]
            window_record=dict(index=index,video_name=data['metas'][0]['video_name'],kind=kind,plan=detail['trace']['plan'],valid_candidates=count,
                physical_slots=detail['plan']['frames'],valid_selected_candidates=int(detail['selection'].valid.sum()),unique_selected_physical_frames=int(real_times.unique().numel()),
                source_gap_max=float(real_times.diff().max()) if len(real_times)>1 else 0.,gflops=cost,model_ms=model_times[-1],
                execution={k:detail['trace'][k] for k in ('q','kv','heavy_mlp','light','depth_attention_light','depth_ffn_light','tia','score_qk')},frame_swaps=detail['routing']['changes'])
            if model.config.get('recipe')=='graph_tad_v1':
                window_record['graph']=dict(recovery_macs=detail['trace'].get('graph_recovery_macs',0),
                    router_macs=detail['trace'].get('graph_router_macs'),referral_paths=detail['trace'].get('graph_referral_paths'),
                    candidate_slots=detail['trace'].get('graph_candidate_slots'),retained_edges=detail['trace'].get('graph_retained_edges'),
                    coverage_changes=detail['trace'].get('graph_coverage_changes'))
            import json
            with window_file.open('a') as stream:stream.write(json.dumps(window_record)+'\n')
            if index%100==0:
                json_write(out/'progress.json',dict(windows=index+1,total_windows=len(dataset)))
                print(f'{model.config["id"]} {metadata.get("epoch")}: {index+1}/{len(dataset)}',flush=True)
            last_end=time.perf_counter()
    if set(result)!=expected:raise RuntimeError('Evaluation omitted video IDs')
    nms_start=time.perf_counter();result=merge_windows(result,post);global_nms_seconds=time.perf_counter()-nms_start;elapsed=time.perf_counter()-begin
    predictions=dict(results=result);json_write(out/'result_detection.json',predictions)
    metrics=build_evaluator(dict(prediction_filename=predictions,**model_cfg.evaluation)).evaluate()
    record=dict(**metadata,metrics=metrics,test_videos=len(expected),test_windows=len(dataset),
                e2e_seconds=elapsed,e2e_scope='decode/preprocess/H2D/model/window and video NMS; AP and diagnostics excluded',
                plan_distribution=plan_counts,force_plan=force_plan,
                checkpoint_selection='fixed epoch80 EMA primary; intermediate full-test milestones are descriptive only',
                primary_endpoint_epoch=80,is_primary_endpoint=metadata.get('epoch')==80,
                dataset_total_gflops=sum(window_costs),dataset_mean_gflops=float(np.mean(window_costs)),
                full_window_mean_gflops=float(np.mean(full_costs)) if full_costs else None,
                gflops=float(np.mean(full_costs or window_costs)),
                compute_scope='actual execution-count ledger calibrated and checked against matrix/conv operators; mean full-window cohort primary',
                window_gflops_quantiles=np.percentile(window_costs,[10,50,90]).tolist(),
                dataset_model_ms_quantiles=np.percentile(model_times,[10,50,90,95]).tolist(),dataset_model_mean_ms=float(np.mean(model_times)))
    record.update(window_execution_file=window_file.name,data_wait_seconds=data_wait,h2d_seconds=h2d_time,
                  window_postprocessing_seconds=post_time,video_nms_seconds=global_nms_seconds,
                  timing_note='data-wait is exposed DataLoader wait, not isolated decode throughput; model timing includes existing trace generation')
    revision=Path(__file__).resolve().parents[2]/'EVALUATION_REVISION'
    record['evaluation_source_revision']=revision.read_text().strip() if revision.exists() else metadata.get('source_revision')
    np.savez_compressed(out/'window_distribution.npz',gflops=np.asarray(window_costs),model_ms=np.asarray(model_times))
    json_write(out/'metrics.json',record)
    if profile:
        profiles={}
        for kind,(index,cpu) in samples.items():
            profiles[kind]=dict(window_index=index,**measure(model,to_gpu(cpu),force_plan))
        json_write(out/'profile.json',profiles)
        primary=profiles.get('full',next(iter(profiles.values())))
        record.update(representative_gflops=primary['matrix_conv_flops']/1e9,latency_ms=primary['latency_mean_ms'],profile='profile.json')
    record['status']='complete';json_write(out/'completed.json',record)
    print(metrics,flush=True);return record,samples
