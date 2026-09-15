#!/usr/bin/env python3
"""Explain the completed historicalB0 decisions without fitting or selecting beta."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
import numpy as np
import torch
from h65.rfv.dataset import load_bank
from h65.rfv.metrics import forecast,ranking_metrics
from tools.rfv_forecast import common_state_scores,aggregate
from h65.paper.runtime import json_write


def decision(scores,ids,target):
    # STOP wins ties at zero, exactly matching ranking_metrics' >0 rule.
    extended=np.concatenate(([0.],np.asarray(scores,dtype=float)))
    order=np.argsort(-extended,kind='stable');first,second=map(int,order[:2])
    names=['STOP',*ids]
    return dict(id=names[first],index=first-1 if first else None,score=float(extended[first]),
        actual_gain=0. if first==0 else float(target[first-1]),runner_up=names[second],
        margin=float(extended[first]-extended[second]),chose_stop=first==0)


def verify_legacy(actual,expected):
    if set(actual['video_means'])!=set(expected['video_means']):raise ValueError('Legacy video cohort differs')
    maximum=0.
    for video,values in actual['video_means'].items():
        for key in ('regret','chosen_gain','stop_regret','ndcg','spearman','topk_overlap'):
            a=values[key];b=expected['video_means'][video][key]
            if (a is None)!=(b is None):raise ValueError('Legacy metric validity differs')
            if a is not None:maximum=max(maximum,abs(a-b))
    if maximum>1e-10:raise ValueError('Export no longer reproduces recordedB0 metrics: '+str(maximum))
    return maximum


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--forecast-run',required=True)
    parser.add_argument('--current-bank',action='append',required=True);parser.add_argument('--future-bank',action='append',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();torch.set_num_threads(2)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    run=Path(args.forecast_run);legacy=json.loads((run/'RISE_B_T.json').read_text())
    lock=json.loads((run/'selection_locked.json').read_text())
    if legacy['variant']!='plain_m' or lock['beta']!=legacy['beta'] or lock['seeds']!=legacy['seeds']:
        raise ValueError('This export is bound to the completed Plain-M B0')
    current,pb=load_bank(args.current_bank);future,fb=load_bank(args.future_bank)
    if pb!=legacy['bank']['current'] or fb!=legacy['bank']['future']:
        raise ValueError('Export inputs differ from the completedB0')
    revision=(ROOT/'source_revision.txt').read_text().strip()
    binding=dict(source_revision=revision,original_forecast_source=legacy['consumer_source_revision'],
        run=str(run),beta=legacy['beta'],seeds=legacy['seeds'],current=pb,future=fb,device=args.device)
    if (out/'RISE_CHOICES.json').exists():
        prior=json.loads((out/'RISE_CHOICES.json').read_text())
        if prior['binding']!=binding:raise ValueError('Existing export belongs to a different input/function binding')
        print(json.dumps(prior['summary']));return
    json_write(out/'manifest.json',binding)
    row_map={row['state_key']:row for row in current};records=[];checks=[];trajectories=[]
    for seed_index,seed in enumerate(legacy['seeds']):
        path=run/f'function_trajectory_s{seed}.pth';payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload['bank']['current']!=pb or payload['bank']['future']!=fb:raise ValueError('Snapshot bank binding differs')
        snapshots=dict(anchor=payload['anchor']['snapshot'],current_post=payload['current']['snapshot'],true_ema=payload['current']['ema'])
        trajectories.append(dict(seed=seed,path=str(path),ema_updates=snapshots['true_ema']['ema']['optimizer_updates']))
        for partition in ('fit','calibration','holdout'):
            scores=common_state_scores(snapshots,current,future,partition,args.device)
            if partition=='holdout':
                for name in ('anchor','current_post','true_ema','future'):
                    maximum=verify_legacy(aggregate(scores,name,legacy['beta']),legacy['seed_controls'][name][seed_index])
                    checks.append(dict(seed=seed,name=name,max_metric_error=maximum))
            for item in scores:
                source=row_map[item['state_key']];ids=[a['id'] for a in source['actions']]
                prediction=dict(item['prediction'])
                prediction['future']=forecast(prediction['anchor'],prediction['current_post'],legacy['beta'])
                choices={name:decision(values,ids,item['target']) for name,values in prediction.items()}
                post=prediction['current_post'];future_scores=prediction['future']
                delta=future_scores-post
                changed=choices['future']['id']!=choices['current_post']['id']
                gain_delta=choices['future']['actual_gain']-choices['current_post']['actual_gain']
                max_delta=float(np.abs(delta).max())
                records.append(dict(seed=seed,partition=partition,video_id=item['video_id'],state_key=item['state_key'],
                    candidate_ids=ids,actual_future_gain=item['target'].tolist(),
                    raw_scores={name:values.tolist() for name,values in prediction.items()},choices=choices,
                    extrapolation_displacement=delta.tolist(),max_abs_displacement=max_delta,
                    displacement_l2=float(np.linalg.norm(delta)),future_choice_changed=changed,
                    selected_actual_gain_delta=gain_delta,changed_choice_equal_actual_gain=changed and gain_delta==0.,
                    predicted_sign_flip_fraction=float(np.mean(np.sign(post)!=np.sign(future_scores))),
                    rank_order_changed=not np.array_equal(np.argsort(-post,kind='stable'),np.argsort(-future_scores,kind='stable')),
                    norm_bound_below_post_margin=2*max_delta<choices['current_post']['margin']))
        del payload
    summaries={}
    for partition in ('fit','calibration','holdout'):
        values=[r for r in records if r['partition']==partition]
        summaries[partition]=dict(seed_state_rows=len(values),unique_videos=len({r['video_id'] for r in values}),
            choice_changes=sum(r['future_choice_changed'] for r in values),
            choice_changes_equal_gain=sum(r['changed_choice_equal_actual_gain'] for r in values),
            rank_order_changes=sum(r['rank_order_changed'] for r in values),
            stop_changes=sum(r['choices']['future']['chose_stop']!=r['choices']['current_post']['chose_stop'] for r in values),
            norm_bound_below_margin=sum(r['norm_bound_below_post_margin'] for r in values),
            max_abs_displacement=max(r['max_abs_displacement'] for r in values),
            mean_selected_actual_gain_delta=float(np.mean([r['selected_actual_gain_delta'] for r in values])))
    with (out/'candidate_scores.jsonl').open('w') as stream:
        for row in records:stream.write(json.dumps(row)+'\n')
    report=dict(binding=binding,status='MEASURED',summary=summaries,trajectory_files=trajectories,
        legacy_metric_checks=checks,records_file='candidate_scores.jsonl',optimizer_updates=0,new_cf_queries=0,
        beta_reselected=False,forecast_gate_changed=False,fvd_unlocked=False,
        limitation='Historical retrospectiveB0. Seed-state counts are not independent video sample counts.')
    json_write(out/'RISE_CHOICES.json',report);print(json.dumps(summaries))


if __name__=='__main__':main()
