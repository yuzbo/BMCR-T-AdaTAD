#!/usr/bin/env python3
"""One fixed407D reverse-consistency mini, retaining the original physical8/8."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
import numpy as np
import torch
from h65.paper.runtime import json_write
from h65.rfv.dataset import load_bank,bank_identities
from h65.rfv.value import TemporalProbe,ProbeEMA,from_snapshot,parameter_count
from h65.rfv.reverse import reverse_descriptors,paired_predictions,reverse_loss
from h65.rfv.metrics import ranking_metrics,video_aggregate,paired_video_difference
from tools.rfv_value_r1 import unseen_split

BASELINE_REVISION='b644d870d1845abbc1e4fd5ab7780f29ff96a53a'
EXTRA=('execution_rate','negative_gain_execution_rate','harmful_state_rate')


def summarize(records):
    result=video_aggregate(records)
    for key in EXTRA:
        for video,values in result['video_means'].items():
            source=[r[key] for r in records if r['video_id']==video and r.get(key) is not None]
            values[key]=float(np.mean(source)) if source else None
        values=[v[key] for v in result['video_means'].values() if v[key] is not None]
        result['mean'][key]=float(np.mean(values)) if values else None
    return result


def seed_average(evaluations):
    records=[dict(video_id=video,**values) for evaluation in evaluations
        for video,values in evaluation['video_means'].items()]
    result=summarize(records);result['seed_count']=len(evaluations)
    result['state_count']=evaluations[0]['state_count']
    for video,values in result['video_means'].items():
        values['states']=evaluations[0]['video_means'][video]['states']
    return result


def batch(rows,device):
    # Every member of the original8/8 and calibration16 has one common length
    # within a batch. Padding is still masked, including the derived reverse view.
    maximum=max(len(r['action_pairs']) for r in rows)
    result={}
    for key in ('descriptor','reverse','target'):
        values=[]
        for row in rows:
            value=torch.from_numpy(row['arrays'][key])
            padding=maximum-len(value)
            if padding:value=torch.cat((value,torch.zeros((padding,value.shape[-1]),dtype=value.dtype)))
            values.append(value)
        result[key]=torch.stack(values).to(device)
    result['valid']=torch.stack([torch.arange(maximum)<len(r['action_pairs']) for r in rows]).to(device)
    return result


@torch.no_grad()
def evaluate(model,rows,mode,device):
    model.eval();records=[]
    for row in rows:
        if not row['action_pairs']:continue
        data=batch([row],device)
        if mode=='forward':prediction=model({'descriptor':data['descriptor']})
        elif mode=='difference':prediction=paired_predictions(model,data['descriptor'],data['reverse'])[2]
        else:raise ValueError('Unregistered reverse readout')
        scores=prediction[0].sum(-1).cpu().numpy();target=row['arrays']['target'].sum(-1)
        metric=ranking_metrics(scores,target,noise=max(1e-8,3*max(row['no_op_error'],row['replay_max_error'])))
        selected=None if metric['chose_stop'] else int(np.argmax(scores))
        harmful=selected is not None and float(target[selected])<0
        metric.update(execution_rate=float(selected is not None),harmful_state_rate=float(harmful),
            negative_gain_execution_rate=float(harmful) if selected is not None else None)
        records.append(dict(video_id=row['video_id'],state_key=row['state_key'],**metric))
    return summarize(records),records


def simple_controls(rows):
    result={}
    for name in ('stop','uniform_swap'):
        records=[]
        for row in rows:
            target=row['arrays']['target'].sum(-1)
            if not len(target):continue
            gain=0. if name=='stop' else float(target.mean())
            negative=float(np.mean(target<0))
            records.append(dict(video_id=row['video_id'],regret=max(0.,float(target.max()))-gain,
                chosen_gain=gain,stop_regret=max(0.,float(target.max())),ndcg=None,spearman=None,topk_overlap=None,
                execution_rate=0. if name=='stop' else 1.,harmful_state_rate=0. if name=='stop' else negative,
                negative_gain_execution_rate=None if name=='stop' else negative))
        result[name]=summarize(records)
    return result


def fit(condition,seed,rows,reference,args):
    torch.manual_seed(seed);model=TemporalProbe('plain_m').to(args.device)
    for key in ('input_mean','input_scale','target_scale'):
        getattr(model,key).copy_(reference['snapshot']['state'][key].to(args.device))
    ema=ProbeEMA(model,.99);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    generator=torch.Generator().manual_seed(seed+1000);history=[];start=time.perf_counter()
    model.train()
    for step in range(args.steps):
        indices=torch.randint(len(rows),(min(8,len(rows)),),generator=generator).tolist()
        data=batch([rows[i] for i in indices],args.device)
        positive,negative,_=paired_predictions(model,data['descriptor'],data['reverse'])
        loss=reverse_loss(positive,negative,data['target'],data['valid'],model.target_scale,condition)
        if not bool(torch.isfinite(loss)):raise RuntimeError('TECH_FAIL: nonfinite reverse objective')
        optimizer.zero_grad();loss.backward()
        grad_norm=torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in model.parameters() if p.grad is not None]))
        if not bool(torch.isfinite(grad_norm)):raise RuntimeError('TECH_FAIL: nonfinite reverse gradient')
        optimizer.step();ema.update(model)
        if step==0 or (step+1)%200==0:
            history.append(dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(grad_norm),
                wall_seconds=time.perf_counter()-start))
    return model,dict(snapshot=model.snapshot(),ema=ema.snapshot(model),optimizer=optimizer.state_dict(),
        seed=seed,steps=args.steps,condition=condition,history=history,parameter_count=parameter_count(model),
        normalization='exact original forward-fit8 P0 buffers; reverse does not refit statistics',
        utility='raw cls+loc; STOP=0',model_used_for_evaluation='raw snapshot')


def input_conditioning(rows,state):
    result={}
    for key in ('descriptor','reverse'):
        values=np.concatenate([r['arrays'][key] for r in rows])
        normalized=(values-state['input_mean'].numpy())/state['input_scale'].numpy()
        result[key]=dict(max_abs=float(np.abs(normalized).max()),
            signed_distance_range=[float(normalized[:,386].min()),float(normalized[:,386].max())],
            finite=bool(np.isfinite(normalized).all()))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bank',action='append',required=True)
    parser.add_argument('--reference-run',required=True);parser.add_argument('--contract',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();torch.set_num_threads(4)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    revision=(ROOT/'source_revision.txt').read_text().strip();rows,binding=load_bank(args.bank)
    if binding['cohort']!='mini' or len(binding['capture_revisions'])!=1:raise ValueError('Use the unchanged single-capture mini')
    contract=json.loads(Path(args.contract).read_text())
    contract_source=contract['config']['source_revision']
    if not (contract['passed'] and contract['status']=='TECH_PASS' and contract['config']['bank']==binding):
        raise ValueError('This bank needs a successful real reverse contract')
    if contract_source!=revision:
        # The completed800 replay was independently inspected: all three loss
        # errors are exactly zero, and reverse.py is unchanged in this fit repair.
        # Reassess its stored measurements against an independent fixed bound.
        if contract_source!='800bcd10a66c69d0d57142ac02389adbc45d2975' or not all(
                r[key]<=1e-8 for r in contract['records'] for key in ('replay_max_error','gain_sum_max','cached_gain_error')):
            raise ValueError('Different contract source lacks the reviewed fixed-error admission')
    fit8,held8,split=unseen_split(rows);cal=[r for r in rows if r['partition']=='calibration' and r['action_pairs']]
    evaluations={'fit8':fit8,'held8':held8,'calibration':cal}
    for group in evaluations.values():
        for row in group:row['arrays']['reverse'],_=reverse_descriptors(row,row['arrays']['descriptor'])
    seeds=binding['protocol']['head_seeds'];args.steps=binding['protocol']['head_steps']
    references={}
    for seed in seeds:
        saved=torch.load(Path(args.reference_run)/f'within_plain_r0_s{seed}.pth',map_location='cpu',weights_only=False)
        if saved['source_revision']!=BASELINE_REVISION or saved['experiment']!=dict(suite='within',arm='plain_r0',bank=binding):
            raise ValueError('P0 reference is not the completed matched R1-era regression baseline')
        references[seed]=saved
    expected=TemporalProbe('plain_m')
    # The original fit computed these reductions on CUDA. Recomputing on CPU
    # needlessly changes rounding; the experiment uses the saved buffers exactly.
    original_stats=references[seeds[0]]['snapshot']['state']
    for saved in references.values():
        for key in ('input_mean','input_scale','target_scale'):
            if not torch.equal(original_stats[key],saved['snapshot']['state'][key]):raise ValueError('Original seeds do not share one normalization')
    config=dict(schema='RFV_REVERSE_MINI_V1',source_revision=revision,bank=binding,split=split,
        reference_run=args.reference_run,reference_source=BASELINE_REVISION,contract=args.contract,contract_source_revision=contract_source,
        seeds=seeds,steps=args.steps,conditions=['p1','p2'],readouts=['p0','p1','p1_bi','p2'],
        descriptor='unchanged407; inverse on true S-prime; no added observations',stop_threshold=0.,
        objective='original component-normalized Huber; no JS',normalization='original forward fit8 only',
        primary='held8 P2 minus P1-bi raw regret, paired video CI and seed directions; also STOP/random',
        scope='development structural mini; original packing-position-family OOD retained',
        calibration_role='descriptive only; no tuning',outer_holdout_used=False,inner10_used=False,
        training_pair_count=sum(len(r['action_pairs']) for r in fit8),reverse_labels_independent=False,
        new_training_cf_queries=0,task_unlocked=False)
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:
        raise ValueError('Reverse experiment configuration changed')
    json_write(out/'config.json',config)
    if (out/'REVERSE_MINI.json').exists():print((out/'metrics.json').read_text());return
    json_write(out/'manifest.json',dict(config=config,input_views={k:bank_identities(v) for k,v in evaluations.items()}))
    (out/'source_revision.txt').write_text(revision+'\n')
    conditioning=input_conditioning(fit8,references[seeds[0]]['snapshot']['state'])
    json_write(out/'input_conditioning.json',conditioning)
    models={};histories={}
    for condition in ('p1','p2'):
        for seed in seeds:
            key=f'{condition}_s{seed}';path=out/(key+'.pth')
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=False)
                if saved['config']!=config or saved['condition']!=condition or saved['seed']!=seed:
                    raise ValueError('Saved reverse head belongs to a different comparison')
                model=from_snapshot(saved['snapshot'],args.device)
            else:
                model,saved=fit(condition,seed,fit8,references[seed],args)
                saved['config']=config;torch.save(saved,path)
            models[key]=model.cpu();histories[key]=saved['history']
            json_write(out/'progress.json',dict(completed=key,heads=len(models),total_heads=6,source_revision=revision))
            print(json.dumps(dict(completed=key,final_loss=saved['history'][-1]['loss'])),flush=True)
    json_write(out/'selection_locked.json',dict(source_revision=revision,selection='none: all conditions, raw snapshots and STOP preregistered'))
    results={};records={}
    for name,group in evaluations.items():
        results[name]={arm:[] for arm in ('p0','p1','p1_bi','p2')};records[name]={}
        for seed in seeds:
            p0=from_snapshot(references[seed]['snapshot'],args.device)
            for arm,model,mode in [('p0',p0,'forward'),('p1',models[f'p1_s{seed}'],'forward'),
                    ('p1_bi',models[f'p1_s{seed}'],'difference'),('p2',models[f'p2_s{seed}'],'difference')]:
                result,detail=evaluate(model.to(args.device),group,mode,args.device)
                results[name][arm].append(result);records[name][f'{arm}_s{seed}']=detail
    combined={name:{arm:seed_average(value) for arm,value in by_arm.items()} for name,by_arm in results.items()}
    controls={name:simple_controls(group) for name,group in evaluations.items()}
    comparisons={}
    for left,right in (('p1_bi','p1'),('p2','p1_bi'),('p2','p0')):
        comparisons[left+'_minus_'+right]={metric:paired_video_difference(combined['held8'][left],combined['held8'][right],metric)
            for metric in ('regret','chosen_gain','ndcg','spearman','negative_gain_execution_rate')}
    versus={name:paired_video_difference(combined['held8']['p2'],value,'regret') for name,value in controls['held8'].items()}
    interval=comparisons['p2_minus_p1_bi']['regret']['ci95']
    stable=all(a['mean']['regret']<b['mean']['regret'] for a,b in zip(results['held8']['p2'],results['held8']['p1_bi']))
    passed=bool(interval is not None and interval[1]<0 and stable and all(v['mean']<0 for v in versus.values()))
    macs=sum(layer.in_features*layer.out_features for layer in expected.network if isinstance(layer,torch.nn.Linear))
    cost=dict(head_parameters=parameter_count(expected),linear_macs_per_forward_candidate=macs,
        inference_head_calls_per_candidate=dict(p0=1,p1=1,p1_bi=2,p2=2),
        training_head_calls_per_candidate=dict(p0=1,p1=2,p2=2),
        caveat='P1-bi/P2 additionally construct the reverse view. Same support capacity does not imply equal total inference cost.')
    report=dict(config=config,status='STRUCTURE_SIGNAL_PASS' if passed else 'STRUCTURE_SIGNAL_FAIL',passed=passed,
        aggregate=combined,seed_evaluations=results,controls=controls,comparisons=comparisons,p2_minus_controls=versus,
        seed_direction_stable=stable,input_conditioning=conditioning,cost=cost,task_unlocked=False,
        limitation='Original held8 has a different packing-position family. No general cheap-Value impossibility or task-level conclusion follows.')
    json_write(out/'REVERSE_MINI.json',report);json_write(out/'value_metrics.json',combined)
    json_write(out/'state_metrics.json',records);json_write(out/'cost.json',cost)
    json_write(out/'metrics.json',dict(status=report['status'],passed=passed,task_unlocked=False))
    print(json.dumps(dict(status=report['status'],held8={k:v['mean'] for k,v in combined['held8'].items()})),flush=True)


if __name__=='__main__':main()
