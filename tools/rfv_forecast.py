#!/usr/bin/env python3
"""Historical same-state RFV function forecast with cumulative EMA controls."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
import numpy as np
import torch
from h65.rfv.dataset import load_bank,collate_states,normalization,INPUTS
from h65.rfv.bank import assert_same_actions
from h65.rfv.value import ProbeEMA,from_snapshot
from h65.rfv.metrics import forecast,ranking_metrics,video_aggregate,paired_video_difference
from h65.paper.runtime import json_write
from tools.rfv_fit import train_one,average_seeds


def continue_fit(model,payload,rows,args,seed):
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    optimizer.load_state_dict(copy.deepcopy(payload['optimizer']))
    ema=ProbeEMA(model,payload['ema']['ema']['decay'])
    ema.state={key:value.to(args.device).clone() for key,value in payload['ema']['state'].items()}
    ema.updates=payload['ema']['ema']['optimizer_updates']
    fit=[row for row in rows if row['partition']=='fit' and row['action_pairs']]
    generator=torch.Generator().manual_seed(seed+1000)
    model.train()
    for step in range(args.steps):
        take=torch.randint(len(fit),(min(8,len(fit)),),generator=generator).tolist()
        batch=collate_states([fit[i] for i in take],args.device)
        output=model({key:batch[key] for key in INPUTS})/model.target_scale
        error=torch.nn.functional.smooth_l1_loss(output,batch['target']/model.target_scale,reduction='none').mean(-1)
        loss=error[batch['action_valid']].mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite post fit')
        optimizer.zero_grad();loss.backward();optimizer.step();ema.update(model)
    return dict(snapshot=model.snapshot(),ema=ema.snapshot(model),optimizer=optimizer.state_dict(),
        seed=seed,steps=args.steps*2,normalization='frozen from anchor fit only',
        sampling='same preregistered state sampling order repeated at target phase change')


@torch.no_grad()
def common_state_scores(snapshots,current,future,partition,device):
    models={key:from_snapshot(value,device).eval() for key,value in snapshots.items()}
    later={row['state_key']:row for row in future};output=[]
    for row in current:
        if row['partition']!=partition or not row['action_pairs']:continue
        target=later[row['state_key']];assert_same_actions(row,target)
        batch=collate_states([row],device)
        # Each complete function independently recomputes its own node adapter,
        # Graph, head and normalization from identical raw sP arrays.
        inputs={key:batch[key] for key in INPUTS}
        prediction={name:model(inputs)[0].sum(-1).cpu().numpy() for name,model in models.items()}
        output.append(dict(video_id=row['video_id'],state_key=row['state_key'],prediction=prediction,
            target=target['arrays']['target'].sum(-1)))
    return output


def aggregate(scores,name,beta=None):
    rows=[]
    for row in scores:
        prediction=(forecast(row['prediction']['anchor'],row['prediction']['current_post'],beta)
                    if name=='future' else row['prediction'][name])
        rows.append(dict(video_id=row['video_id'],state_key=row['state_key'],**ranking_metrics(prediction,row['target'])))
    return video_aggregate(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--anchor-bank',action='append',required=True)
    parser.add_argument('--current-bank',action='append',required=True);parser.add_argument('--future-bank',action='append',required=True)
    parser.add_argument('--variant',choices=['plain_m','plain_l','static_graph','dynamic_graph'],default='plain_m')
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--steps',type=int,default=2000)
    args=parser.parse_args();torch.set_num_threads(4)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'RISE_B_T.json').exists():print((out/'RISE_B_T.json').read_text());return
    anchor,ab=load_bank(args.anchor_bank);current,pb=load_bank(args.current_bank);future,fb=load_bank(args.future_bank)
    if ab['protocol']!=pb['protocol'] or ab['protocol']!=fb['protocol'] or len({ab['cohort'],pb['cohort'],fb['cohort']})!=1:
        raise ValueError('Forecast cohorts or registrations differ')
    indices=[{row['state_key']:row for row in bank} for bank in (anchor,current,future)]
    if set(indices[0])!=set(indices[1]) or set(indices[0])!=set(indices[2]):raise ValueError('Forecast action states differ')
    for key in indices[0]:
        assert_same_actions(indices[0][key],indices[1][key]);assert_same_actions(indices[1][key],indices[2][key])
    if not ab['parameter']['epoch']<pb['parameter']['epoch']<fb['parameter']['epoch']:
        raise ValueError('Forecast requires increasing checkpoint time')
    if args.steps!=ab['protocol']['head_steps']:raise ValueError('Unregistered optimizer budget')
    seeds=ab['protocol']['head_seeds'];betas=ab['protocol']['beta'];stats=normalization(anchor)
    snapshots={};calibration=[]
    for seed in seeds:
        model,before=train_one(args.variant,seed,anchor,stats,args)
        after=continue_fit(model,before,current,args,seed)
        states=dict(anchor=before['snapshot'],current_post=after['snapshot'],true_ema=after['ema'])
        snapshots[seed]=states
        torch.save(dict(anchor=before,current=after,bank=dict(anchor=ab,current=pb,future=fb)),out/f'function_trajectory_s{seed}.pth')
        scores=common_state_scores(states,current,future,'calibration',args.device)
        calibration.append({beta:aggregate(scores,'future',beta) for beta in betas})
    cal={beta:average_seeds([row[beta] for row in calibration]) for beta in betas}
    chosen=min(betas,key=lambda beta:(cal[beta]['mean']['regret'],beta))
    lock=dict(beta=chosen,variant=args.variant,seeds=seeds,steps_per_phase=args.steps,
        selection='future-checkpoint calibration videos only; retrospective cross-video forecast',
        normalization='anchor fit only; fixed thereafter',future_fit_labels_used=False,
        calibration={str(beta):value for beta,value in cal.items()})
    path=out/'selection_locked.json'
    if path.exists() and json.loads(path.read_text())!=lock:raise RuntimeError('Forecast selection changed after holdout lock')
    json_write(path,lock)
    evaluations={name:[] for name in ('anchor','current_post','true_ema','future')}
    for seed in seeds:
        scores=common_state_scores(snapshots[seed],current,future,'holdout',args.device)
        for name in evaluations:evaluations[name].append(aggregate(scores,name,chosen))
    means={name:average_seeds(values) for name,values in evaluations.items()}
    differences={control:{metric:paired_video_difference(means['future'],means[control],metric)
        for metric in ('regret','ndcg','topk_overlap')} for control in ('current_post','true_ema')}
    passed=chosen>1
    for control,metrics in differences.items():
        passed=passed and metrics['regret']['ci95'] is not None and metrics['regret']['ci95'][1]<0
        passed=passed and all(metrics[key]['mean'] is not None and metrics[key]['mean']>=0 for key in ('ndcg','topk_overlap'))
    report=dict(scope='RISE-B0 historical offline cross-video function forecast, not new T-V trajectory',
        variant=args.variant,beta=chosen,seeds=seeds,bank=dict(anchor=ab,current=pb,future=fb),
        controls=means,seed_controls=evaluations,future_minus_control=differences,
        same_raw_state=True,complete_functions_recomputed=True,true_ema_updates=args.steps*2,
        future_fit_labels_used=False,forecast_value_gate=bool(passed),fvd_unlocked=False,
        status='PASS' if passed else 'FAIL',consumer_source_revision=(ROOT/'source_revision.txt').read_text().strip(),
        timing_limitation='Beta saw later calibration labels; no strict chronological forecasting claim')
    json_write(out/'RISE_B_T.json',report)
    print(json.dumps(dict(beta=chosen,forecast_value_gate=bool(passed),fvd_unlocked=False)))


if __name__=='__main__':main()
