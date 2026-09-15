#!/usr/bin/env python3
"""Matched Plain/Graph fits with three seeds and a sealed final holdout decision."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
import numpy as np
import torch
from h65.rfv.dataset import load_bank,collate_states,normalization,INPUTS
from h65.rfv.value import TemporalProbe,ProbeEMA,parameter_count,matched_plain_width
from h65.rfv.metrics import ranking_metrics,video_aggregate,paired_video_difference
from h65.rfv.objectives import value_loss,objective_record
from h65.paper.runtime import json_write


@torch.no_grad()
def evaluate(model,rows,device):
    model.eval();scores=[]
    for row in rows:
        if not row['action_pairs']:continue
        batch=collate_states([row],device)
        prediction=model({key:batch[key] for key in INPUTS})[0].sum(-1).cpu().numpy()
        target=row['arrays']['target'].sum(-1)
        scores.append(dict(video_id=row['video_id'],state_key=row['state_key'],
            **ranking_metrics(prediction,target,noise=max(1e-8,3*max(row['no_op_error'],row['replay_max_error'])))))
    return video_aggregate(scores)


def controls(rows):
    results={}
    for name in ('stop','uniform_swap','transition_proxy'):
        values=[]
        for row in rows:
            if not row['action_pairs']:continue
            target=row['arrays']['target'].sum(-1);x=row['arrays']['descriptor']
            prediction=np.full(len(target),-1.) if name=='stop' else x[:,-1]-x[:,-2]
            metric=ranking_metrics(prediction,target)
            if name=='stop':metric.update(ndcg=None,spearman=None,topk_overlap=None)
            if name=='uniform_swap':
                metric.update(regret=max(0.,float(target.max()))-float(target.mean()),chosen_gain=float(target.mean()),
                    ndcg=None,spearman=None,topk_overlap=None)
            values.append(dict(video_id=row['video_id'],**metric))
        results[name]=video_aggregate(values)
    return results


def train_one(variant,seed,rows,stats,args):
    torch.manual_seed(seed)
    hidden=matched_plain_width() if variant=='plain_l' else 128
    model=TemporalProbe(variant,hidden=hidden).to(args.device)
    model.set_normalization(*[value.to(args.device) for value in stats])
    objective=objective_record(getattr(args,'objective','r0'),stats[1],model.target_scale,
        getattr(args,'temperature',1.))
    ema=ProbeEMA(model,.99)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    fit=[x for x in rows if x['partition']=='fit' and x['action_pairs']]
    generator=torch.Generator().manual_seed(seed+1000);history=[];start=time.perf_counter()
    model.train()
    for step in range(args.steps):
        indices=torch.randint(len(fit),(min(8,len(fit)),),generator=generator).tolist()
        batch=collate_states([fit[i] for i in indices],args.device)
        prediction=model({key:batch[key] for key in INPUTS})
        losses=value_loss(prediction,batch['target'],batch['action_valid'],model.target_scale,
            objective['name'],objective['rank_scale'],objective['temperature'])
        loss=losses['total']
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite Value fit')
        optimizer.zero_grad();loss.backward();optimizer.step();ema.update(model)
        if (step+1)%200==0 or step==0:
            history.append(dict(step=step+1,loss=float(loss.detach()),rank=float(losses['rank'].detach()),
                huber=float(losses['huber'].detach()),wall_seconds=time.perf_counter()-start))
    calibration=evaluate(model,[x for x in rows if x['partition']=='calibration'],args.device)
    payload=dict(snapshot=model.snapshot(),ema=ema.snapshot(model),optimizer=optimizer.state_dict(),
        seed=seed,steps=args.steps,parameter_count=parameter_count(model),calibration=calibration,history=history,
        normalization='fit only',utility='raw gain_cls + gain_loc',objective=objective)
    return model,payload


def average_seeds(evaluations):
    rows=[]
    for evaluation in evaluations:
        rows.extend(dict(video_id=video,**values) for video,values in evaluation['video_means'].items())
    result=video_aggregate(rows)
    result['seed_evaluation_rows']=result['state_count']
    result['seed_count']=len(evaluations)
    result['state_count']=evaluations[0]['state_count']
    for video,values in result['video_means'].items():
        values['states']=evaluations[0]['video_means'][video]['states']
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bank',action='append',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--steps',type=int,default=2000);parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    args=parser.parse_args();torch.set_num_threads(4)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'GRAPH_G1_T.json').exists():
        print((out/'GRAPH_G1_T.json').read_text());return
    rows,binding=load_bank(args.bank);stats=normalization(rows)
    if args.steps!=binding['protocol']['head_steps'] or args.seeds!=binding['protocol']['head_seeds']:
        raise ValueError('Head steps/seeds differ from the preregistered shared comparison')
    if len(binding['capture_revisions'])!=1:
        raise ValueError('Mixed capture revisions require explicit forward-equivalence review before fitting')
    variants=('plain_m','plain_l','static_graph','dynamic_graph');models={};payloads={}
    for variant in variants:
        for seed in args.seeds:
            model,payload=train_one(variant,seed,rows,stats,args)
            key=f'{variant}_s{seed}';payload['bank']=binding
            torch.save(payload,out/(key+'.pth'));payloads[key]=payload
            models[key]=model.cpu()
            print(json.dumps(dict(fitted=key,parameters=payload['parameter_count'],calibration=payload['calibration']['mean'])),flush=True)
    calibration={variant:average_seeds([payloads[f'{variant}_s{s}']['calibration'] for s in args.seeds]) for variant in variants}
    comparison=paired_video_difference(calibration['dynamic_graph'],calibration['static_graph'],'regret')
    selected='dynamic_graph' if comparison['ci95'] is not None and comparison['ci95'][1]<0 else 'static_graph'
    lock=dict(selected_graph=selected,seeds=args.seeds,steps=args.steps,selection_data='calibration only',
        calibration=calibration,dynamic_minus_static=comparison,bank=binding,
        holdout_scope='inner probe holdout' if binding['cohort']=='mini' else 'sealed outer router-label holdout')
    if (out/'selection_locked.json').exists():
        if json.loads((out/'selection_locked.json').read_text())!=lock:
            raise RuntimeError('Selection changed after holdout lock')
    else:json_write(out/'selection_locked.json',lock)
    holdout=[x for x in rows if x['partition']=='holdout']
    evaluations={}
    for variant in variants:
        evaluations[variant]=[evaluate(models[f'{variant}_s{s}'].to(args.device),holdout,args.device) for s in args.seeds]
    combined={key:average_seeds(value) for key,value in evaluations.items()}
    simple=controls(holdout)
    plain=combined['plain_m']['mean'];plain_pass=plain['regret'] is not None and all(
        plain['regret']<value['mean']['regret'] for value in simple.values())
    regret=paired_video_difference(combined[selected],combined['plain_l'],'regret')
    ndcg=paired_video_difference(combined[selected],combined['plain_l'],'ndcg')
    stable=all(g['mean']['regret']<p['mean']['regret'] for g,p in zip(evaluations[selected],evaluations['plain_l']))
    graph_pass=(regret['ci95'] is not None and regret['ci95'][1]<0 and ndcg['mean'] is not None and ndcg['mean']>=0 and stable)
    report=dict(bank=binding,selected_graph=selected,seeds=args.seeds,steps=args.steps,
        learner_source_revision=(ROOT/'source_revision.txt').read_text().strip(),
        holdout=combined,seed_holdout=evaluations,controls=simple,graph_minus_plain_l=dict(regret=regret,ndcg=ndcg),
        seed_directions_stable=stable,plain_learnability=plain_pass,graph_value_gate=graph_pass,
        task_gate='WAITING',final=False,holdout_evaluated_after_selection_lock=True,
        scope='mini preliminary' if binding['cohort']=='mini' else 'full outer-holdout G1a',
        status='PASS' if graph_pass else 'FAIL',scientific_claim='same-state predictor comparison, not closed-loop task performance')
    json_write(out/'T_VALUE_MINI.json',dict(passed=plain_pass,scope=report['scope'],value_metrics=combined['plain_m'],controls=simple,bank=binding))
    json_write(out/'GRAPH_G1_T.json',report)
    print(json.dumps(dict(plain_learnability=plain_pass,graph_value_gate=graph_pass,selected_graph=selected,task_gate='WAITING')),flush=True)


if __name__=='__main__':main()
