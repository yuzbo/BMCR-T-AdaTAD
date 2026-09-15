"""Evaluate predeclared value extrapolations against later ACTUAL interventions.

Forecasts are retrospective mechanism tests, not evidence of better TAD mAP.
The --future file is NEVER passed to fit_router.py or used to construct its teacher.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import numpy as np
import torch
from scipy.stats import spearmanr
from .core import read_jsonl, align_records, extrapolate, atomic_json


def rank_quality(pred, truth):
    p=np.asarray(pred,float);t=np.asarray(truth,float)
    if len(p)<2 or np.ptp(p)==0 or np.ptp(t)==0:
        return None
    return float(spearmanr(p,t).statistic)


def cluster_ci(rows, n=2000, seed=42):
    """Equal-weight video mean and resample whole videos, not correlated actions."""
    by=defaultdict(list)
    for video,value in rows:
        if value is not None and np.isfinite(value):by[video].append(value)
    a=np.array([np.mean(v) for v in by.values()])
    if len(a)<2:return dict(mean=None if len(a)==0 else float(a.mean()),ci95=None,videos=len(a))
    rng=np.random.default_rng(seed)
    boot=np.mean(a[rng.integers(0,len(a),(n,len(a)))],axis=1)
    return dict(mean=float(a.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),videos=len(a))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anchor',required=True);p.add_argument('--current',required=True)
    p.add_argument('--future',required=True);p.add_argument('--ema')
    p.add_argument('--betas',default='1,1.025,1.05,1.1,1.2')
    p.add_argument('--split',choices=['fit','calibration','holdout'],default='holdout')
    p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--output',required=True)
    args=p.parse_args()
    collections=[read_jsonl(x) for x in (args.anchor,args.current,args.future)]
    if args.ema:collections.append(read_jsonl(args.ema))
    matched=align_records(*collections)
    if not matched:raise RuntimeError('Empty bank.')
    stages=[{r[i]['successful_updates'] for r in matched} for i in range(3)]
    if any(len(s)!=1 for s in stages):raise ValueError('Mixed checkpoints in one file.')
    a,c,f=(next(iter(s)) for s in stages)
    if not a<c<f:raise ValueError('Need anchor < current < future successful_updates.')
    fit=np.asarray([r[1]['actual_delta'] for r in matched if r[1]['split']=='fit'])
    if fit.size==0:raise RuntimeError('No video-disjoint fit split for common scale.')
    scale=np.maximum(np.mean(np.abs(fit),axis=0),1e-4)
    variants={f'beta_{float(b):g}':float(b) for b in args.betas.split(',')}
    if args.ema:variants['ema']=None
    grouped=defaultdict(list)
    for rows in matched:
        r=rows[0]
        if r['split']==args.split:
            grouped[(r['video_name'],r['support_id'],r['action_type'])].append(rows)
    results={}
    for name,beta in variants.items():
        rank=[];regret=[];mse=[];top1=[]
        for (video,support,kind),records in grouped.items():
            anchor=torch.tensor([r[0]['actual_delta'] for r in records],dtype=torch.float32)
            current=torch.tensor([r[1]['actual_delta'] for r in records],dtype=torch.float32)
            truth=np.asarray([r[2]['actual_delta'] for r in records],float)
            pred=(np.asarray([r[3]['actual_delta'] for r in records]) if beta is None
                  else extrapolate(anchor,current,beta).numpy())
            pv=(pred/scale).mean(-1);tv=(truth/scale).mean(-1)
            rank.append((video,rank_quality(pv,tv)))
            # Include a zero-gain no-op; do not force a harmful swap.
            pi=int(np.argmax(np.r_[pv,0.]));ti=int(np.argmax(np.r_[tv,0.]))
            tv=np.r_[tv,0.]
            regret.append((video,float(tv[ti]-tv[pi])));top1.append((video,float(pi==ti)))
            mse.append((video,float(np.mean(((pred-truth)/scale)**2))))
        results[name]={k:cluster_ci(rows,args.bootstrap) for k,rows in
                       [('spearman',rank),('normalized_regret',regret),('top1_agreement',top1),('normalized_mse',mse)]}
    atomic_json(args.output,dict(kind='retrospective_value_forecast_not_TAD_mAP',
        stage_updates=[a,c,f],split=args.split,common_scale=scale.tolist(),
        groups=len(grouped),variants=results,
        beta_selection='Predeclared sweep only. Do not choose beta using this holdout result.',
        evidence_limit='A better value forecast does not certify a better deployed allocation.'))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
