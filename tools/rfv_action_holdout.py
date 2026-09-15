#!/usr/bin/env python3
"""One fixed Plain-M diagnostic: unseen actions within existing fit videos."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
import numpy as np
import torch
from h65.rfv.dataset import load_bank,normalization,collate_states,INPUTS
from tools.rfv_fit import train_one,evaluate,average_seeds
from h65.paper.runtime import json_write


def subset(row,indices,partition):
    result=copy.deepcopy(row);result['partition']=partition
    for key in ('action_pairs','actions','strata'):result[key]=[row[key][i] for i in indices]
    for key in ('descriptor','remove_times','insert_times','target'):
        result['arrays'][key]=row['arrays'][key][indices].copy()
    return result


@torch.no_grad()
def centered_error(model,rows):
    values={}
    for row in rows:
        batch=collate_states([row],'cpu')
        prediction=model({key:batch[key] for key in INPUTS})[0].sum(-1).numpy()
        target=row['arrays']['target'].sum(-1)
        variance=float(np.mean((target-target.mean())**2))
        if variance<=1e-16:continue
        error=float(np.mean(((prediction-prediction.mean())-(target-target.mean()))**2)/variance)
        values.setdefault(row['video_id'],[]).append(error)
    return dict(mean=float(np.mean([np.mean(v) for v in values.values()])) if values else None,
        video_means={key:float(np.mean(value)) for key,value in values.items()},constant_predictor_reference=1.)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bank',action='append',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'ACTION_HOLDOUT_DIAGNOSTIC.json').exists():print((out/'ACTION_HOLDOUT_DIAGNOSTIC.json').read_text());return
    rows,binding=load_bank(args.bank)
    if binding['cohort']!='mini':raise ValueError('This fixed diagnostic uses the existing mini fit cohort')
    fit=[];held=[];ids={}
    for row in rows:
        if row['partition']!='fit' or not row['action_pairs']:continue
        if len(row['action_pairs'])!=16:raise ValueError('The registered 8/8 diagnostic expects16 actions per fit state')
        order=sorted(range(16),key=lambda i:(row['action_pairs'][i][1],row['action_pairs'][i][0]))
        train,unseen=order[::2],order[1::2]
        fit.append(subset(row,train,'fit'));held.append(subset(row,unseen,'held_actions'))
        ids[row['state_key']]=dict(fit_action_ids=[row['actions'][i]['id'] for i in train],
                                  held_action_ids=[row['actions'][i]['id'] for i in unseen])
    calibration=[row for row in rows if row['partition']=='calibration' and row['action_pairs']]
    training_rows=fit+calibration;stats=normalization(training_rows)
    settings=argparse.Namespace(device='cpu',steps=binding['protocol']['head_steps'])
    reports=[]
    for seed in binding['protocol']['head_seeds']:
        model,payload=train_one('plain_m',seed,training_rows,stats,settings)
        torch.save(payload,out/f'plain_m_action_holdout_s{seed}.pth')
        report=dict(seed=seed,fit=evaluate(model,fit,'cpu'),held_actions=evaluate(model,held,'cpu'),
            calibration=evaluate(model,calibration,'cpu'),centered_error={name:centered_error(model,values)
                for name,values in [('fit',fit),('held_actions',held),('calibration',calibration)]})
        reports.append(report)
        print(json.dumps(dict(seed=seed,held_actions=report['held_actions']['mean'],calibration=report['calibration']['mean'])),flush=True)
    report=dict(scope='within-fit-video unseen-action diagnosis only',status='COMPLETE',scientific_gate=False,
        bank=binding,fit_videos=len(fit),training_actions=8*len(fit),held_actions=8*len(held),
        action_split='sort physical(insert,remove), alternate even/odd without reading gain',action_ids=ids,
        normalization='fit8 actions only',head='unchanged Plain-M407->128->64->2',
        steps=settings.steps,seeds=binding['protocol']['head_seeds'],new_cf_queries=0,outer_holdout_used=False,
        seed_reports=reports,aggregate={name:average_seeds([row[name] for row in reports]) for name in ('fit','held_actions','calibration')},
        source_revision=(ROOT/'source_revision.txt').read_text().strip(),
        interpretation='Within-video success plus cross-video failure suggests a coverage increment; failure on held actions requires descriptor/memorization diagnosis first')
    json_write(out/'ACTION_HOLDOUT_DIAGNOSTIC.json',report)


if __name__=='__main__':main()
