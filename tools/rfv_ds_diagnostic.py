#!/usr/bin/env python3
"""Fixed raw40 D/S route crossover and detached-Value clipping decomposition."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
import torch
from h65.paper.runtime import build_config,json_write
from h65.paper.model import PaperModel
from h65.paper.fasttrack import initialize as initialize_model,router_splits
from h65.paper.training import EpochDataset
from h65.paper.objectives import objectives
from h65.paper.operator_training import collect_operator_action
from h65.paper.profile import execution_flops
from h65.atlas.data import CharacterizationData
from h65.atlas.reference import deterministic_fp32
from h65.full.runtime import to_gpu,seed_all
from h65.rfv.hardware import initialize as initialize_hardware
from h65.rfv.gradients import derivatives,add_vectors,gradient_summary

SCIENCE='40552945ad5d56b7404f833cd86f998e8558e8b1'


@torch.no_grad()
def route_crossover(model,data,axis):
    model.eval();original=copy.deepcopy(model.config['operator_policy'])
    try:
        with deterministic_fp32():
            features,value=model.forward_native(data,force_plan=model.config['fixed_plan'],apply_refiner=False)
            loss_value=model.readout.components(model.readout.loss(features,data)).cpu()
            cost_value=execution_flops(model,value)/1e9
            model.config['operator_policy'][axis]='uniform'
            features,uniform=model.forward_native(data,force_plan=model.config['fixed_plan'],
                preview=value['preview'],selection=value['selection'],apply_refiner=False)
            loss_uniform=model.readout.components(model.readout.loss(features,data)).cpu()
            cost_uniform=execution_flops(model,uniform)/1e9
            changed={};quota={}
            for field in ('depth_masks','spatial_masks'):
                changed[field]=[];quota[field]=[]
                for left,right in zip(value['trace'][field],uniform['trace'][field]):
                    if not torch.equal(left.sum(-1),right.sum(-1)):raise RuntimeError('Crossover changed packed quota')
                    changed[field].append(int((left!=right).sum()))
                    quota[field].append(left.sum(-1).cpu().tolist())
            return dict(video_id=data['metas'][0]['video_name'],window_start_frame=int(data['metas'][0]['window_start_frame']),
                axis=axis,value_loss_cls_loc=loss_value.tolist(),uniform_loss_cls_loc=loss_uniform.tolist(),
                value_minus_uniform_loss=(loss_value-loss_uniform).tolist(),changed_tokens_by_layer=changed,
                quotas=quota,value_forward_gflops=cost_value,uniform_forward_gflops=cost_uniform,
                same_parameters=True,same_temporal_support=True,same_capacity=True,
                scope='raw40 same-model route crossover on development windows; not dataset mAP')
    finally:model.config['operator_policy']=original


def gradient_group(model,saved,cpu_samples,sequence):
    model.load_learned(saved['learned']);model.train();model.training_epoch=39
    named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    names=[name for name,p in named];parameters=[p for name,p in named]
    buffers={term:[torch.zeros_like(p) for p in parameters] for term in ('task','self_feature','value')}
    losses_log=[];cf_record=None;denominator=len(cpu_samples)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(model.config['seed']+sequence)
        for micro,cpu in enumerate(cpu_samples):
            data=to_gpu(cpu)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                losses,detail,_=objectives(model,data,model.config['fixed_plan'])
            task=losses['task']*model.config['loss']['task']/denominator
            feature=losses['self_feature']*model.config['loss']['self_feature']/denominator
            buffers['task']=add_vectors(buffers['task'],derivatives(task,parameters,retain_graph=True))
            buffers['self_feature']=add_vectors(buffers['self_feature'],derivatives(feature,parameters))
            if micro==0:
                found=collect_operator_action(model,data,sequence)
                if found is None:raise RuntimeError('Selected fit group has no legal D/S exchange')
                auxiliary,cf_record,_=found
                value=auxiliary*model.config['loss']['operator_value']/denominator
                buffers['value']=add_vectors(buffers['value'],derivatives(value,parameters))
            model.readout.commit_normalizer()
            losses_log.append(dict(video_id=data['metas'][0]['video_name'],
                task=float(task.detach()),self_feature=float(feature.detach())))
            del losses,detail,data
    summary={key:gradient_summary(names,value) for key,value in buffers.items()}
    base=add_vectors(buffers['task'],buffers['self_feature'])
    ordinary=gradient_summary(names,base);combined=gradient_summary(names,add_vectors(base,buffers['value']))
    outside=sum(value**2 for key,value in summary['value']['groups'].items() if key!='value_router')**.5
    if not all(torch.isfinite(torch.tensor(s['global_norm'])) for s in [*summary.values(),ordinary,combined]):
        raise RuntimeError('Nonfinite diagnostic gradients')
    return dict(sequence=sequence,accumulation=denominator,losses=losses_log,counterfactual=cf_record,
        per_loss=summary,ordinary_without_value=ordinary,cf_update_with_value=combined,
        value_gradient_outside_router=outside,
        same_detector_gradient_clip_ratio=combined['clip_coefficient']/ordinary['clip_coefficient'],
        bf16=True,grad_scaler=False,optimizer_steps=0,
        interpretation='Same task/feature gradient, with versus without this detached Value auxiliary; no optimizer trajectory claim')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',action='append',required=True)
    parser.add_argument('--resources',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();resources=json.loads(Path(args.resources).read_text())
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    revision=(ROOT/'source_revision.txt').read_text().strip();hardware=initialize_hardware(resources)
    partitions=router_splits(resources)
    cal=sorted(partitions['calibration'])[:4];fit=set(partitions['fit'])
    config=dict(source_revision=revision,original_science=SCIENCE,checkpoints=args.checkpoint,
        route_videos=cal,gradient_accumulation_groups=2,gradient_samples_per_group=2,
        gradient_epoch=39,gradient_sequence=[0,1],parameters='raw learned epoch40',
        official_test_used=False,outer_holdout_used=False,optimizer_steps=0,
        scope='bounded fixed-model route and gradient diagnostic; no new course')
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:
        raise ValueError('Diagnostic inputs changed')
    json_write(out/'config.json',config)
    results={}
    for path in args.checkpoint:
        saved=torch.load(path,map_location='cpu',weights_only=False)
        cfg=saved['metadata']['config'];name=cfg['id']
        if saved['metadata']['wtr_fasttrack_science_sha']!=SCIENCE or saved['epoch_index']!=40 or saved['successful_updates']!=4000:
            raise ValueError('Use the original405 epoch40 checkpoint')
        if saved['metadata']['router_label_splits']!=partitions:
            raise ValueError('Diagnostic label partition differs from the original course')
        axis='D' if cfg['operator_policy']['D']=='value' else 'S'
        target=out/(name+'.json')
        if target.exists():results[name]=json.loads(target.read_text());continue
        seed_all(cfg['seed']);native_cfg=build_config(cfg,resources)
        model=PaperModel(native_cfg,cfg,resources,with_teacher=False).cuda()
        initialize_model(model,resources);model.load_learned(saved['learned'])
        source=CharacterizationData(native_cfg,resources,'development',cal,one_window=True)
        route=[route_crossover(model,to_gpu(source[index]),axis) for index in range(len(source))]
        from opentad.datasets.builder import build_dataset,collate
        dataset=build_dataset(native_cfg.dataset.train);wrapped=EpochDataset(dataset,cfg['seed']);wrapped.epoch=39
        indices=[i for i,item in enumerate(dataset.data_list) if item[0] in fit][:4]
        if len(indices)!=4:raise ValueError('Four fit training samples are required')
        samples=[collate([wrapped[index]]) for index in indices]
        gradients=[gradient_group(model,saved,samples[2*i:2*i+2],i) for i in range(2)]
        record=dict(config=config,original_checkpoint=saved['original_checkpoint'],checkpoint_metadata=saved['metadata'],
            raw_state='original learned (not EMA)',axis=axis,route=route,gradients=gradients,
            hardware=hardware,training_indices=indices,optimizer_steps=0,status='MEASURED',task_gate_changed=False)
        json_write(target,record);results[name]=record
        print(json.dumps(dict(completed=name,clip_ratios=[v['same_detector_gradient_clip_ratio'] for v in gradients],
            route_loss_delta=[sum(v['value_minus_uniform_loss']) for v in route])),flush=True)
        del model,saved;torch.cuda.empty_cache()
    json_write(out/'DS_DIAGNOSTIC.json',dict(config=config,results=results,status='MEASURED',optimizer_steps=0))


if __name__=='__main__':main()
