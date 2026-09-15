#!/usr/bin/env python3
"""Finite T-local headroom and fixed-action historical drift reports."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
from h65.paper.runtime import json_write
from h65.rfv.metrics import drift_metrics,video_aggregate


def headroom(args):
    folder=Path(args.capture);manifest=json.loads((folder/'manifest_0.json').read_text())
    done=json.loads((folder/'completed_0.json').read_text())
    if manifest['mode']!='local_cf' or manifest['shards']!=1 or done['source_revision']!=manifest['source_revision']:
        raise ValueError('Not a complete registered T-local-CF capture')
    expected=set(manifest['protocol']['splits']['calibration'])
    rows=[json.loads(path.read_text()) for path in sorted((folder/'windows').glob('*.json'))]
    if {row['video_id'] for row in rows}!=expected or len(rows)!=done['windows']:
        raise ValueError('Headroom capture lacks the complete declared calibration cohort')
    by_video={video:[row for row in rows if row['video_id']==video] for video in sorted(expected)}
    gain=np.array([np.mean([row['loss_gain'] for row in items]) for items in by_video.values()])
    noise=max(1e-8,3*max(max(row['replay_max_error'],row['no_op_error']) for row in rows))
    rng=np.random.default_rng(42)
    samples=gain[rng.integers(0,len(gain),size=(args.bootstrap,len(gain)))].mean(1)
    interval=np.quantile(samples,[.025,.975]).tolist()
    report=dict(scope='bounded greedy local-CF, not exhaustive four-step oracle',split='calibration',
        videos=len(expected),windows=len(rows),checkpoint=manifest['checkpoint'],capture_source_revision=manifest['source_revision'],
        loss_gain_mean=float(gain.mean()),loss_gain_ci95=interval,noise_epsilon=noise,
        g0a_loss_pass=interval[0]>noise,positive_video_fraction=float((gain>noise).mean()),
        query_forwards=sum(row['query_forwards'] for row in rows),label_query_gflops=sum(row['label_query_gflops'] for row in rows),
        accepted_swaps=sum(row['accepted_swaps'] for row in rows),g0b_ap_pass=None,task_course_eligible=False,
        video_loss_gains=dict(zip(by_video,gain.tolist())),bootstrap=args.bootstrap)
    if manifest['all_windows']:
        from h65.atlas.statistics import ap_cache,cluster_ap
        ground_truth=folder/'development_ground_truth.json';ids=sorted(expected);results={}
        for label in ('uniform','local_cf'):
            prediction=json.loads((folder/(label+'_predictions.json')).read_text())['results']
            cache,official=ap_cache(prediction,ground_truth,'training',ids)
            results[label]=cluster_ap(cache,ids,args.bootstrap)
        delta=np.asarray(results['local_cf']['bootstrap_average'])-np.asarray(results['uniform']['bootstrap_average'])
        ap_delta=results['local_cf']['average_mAP']-results['uniform']['average_mAP']
        ci=np.quantile(delta,[.025,.975]).tolist()
        report.update(ap=results,ap_delta_pp=ap_delta,ap_delta_ci95_pp=ci,
            ap07_delta_pp=results['local_cf']['AP07']-results['uniform']['AP07'],
            g0b_ap_pass=bool(ap_delta>0 and ci[0]>0),task_course_eligible=bool(ap_delta>0 and ci[0]>0))
    else:report['limitation']='Representative-window loss evidence only; full calibration-video AP still required'
    json_write(Path(args.output)/'T_LOCAL_CF.json',report)
    print(json.dumps({key:report[key] for key in ('videos','windows','loss_gain_mean','g0a_loss_pass','g0b_ap_pass','task_course_eligible')}))


def drift(args):
    from h65.rfv.dataset import load_bank,require_equivalent_captures
    from h65.rfv.bank import assert_same_actions
    early,early_binding=load_bank(args.earlier);late,late_binding=load_bank(args.later)
    equivalence=require_equivalent_captures((early_binding,late_binding),args.capture_equivalence)
    a={row['state_key']:row for row in early};b={row['state_key']:row for row in late}
    if set(a)!=set(b):raise ValueError('Drift banks do not share every action state')
    rows=[];strata={}
    durations=[item['action_duration'] for row in early if row['partition']=='fit' for item in row['strata']
        if item['action_duration'] is not None]
    duration_edges=np.quantile(durations,[1/3,2/3]).tolist() if durations else None
    for key in sorted(a):
        before=a[key];after=b[key];assert_same_actions(before,after)
        x=before['arrays']['target'].sum(-1);y=after['arrays']['target'].sum(-1)
        if not len(x):continue
        noise=max(1e-8,3*max(before['no_op_error'],before['replay_max_error'],after['no_op_error'],after['replay_max_error']))
        record=dict(video_id=before['video_id'],state_key=key,**drift_metrics(x,y,noise))
        rows.append(record)
        for group in ('boundary','interior','background'):
            take=np.array([item['region']==group for item in before['strata']])
            if take.any():strata.setdefault(group,[]).append(dict(video_id=before['video_id'],**drift_metrics(x[take],y[take],noise)))
        if duration_edges:
            for group,index in [('short_action',0),('long_action',1)]:
                take=np.array([item['action_duration'] is not None and
                    (item['action_duration']<=duration_edges[0] if index==0 else item['action_duration']>=duration_edges[1])
                    for item in before['strata']])
                if take.any():strata.setdefault(group,[]).append(dict(video_id=before['video_id'],**drift_metrics(x[take],y[take],noise)))
    summary=video_aggregate(rows)
    flips={}
    for row in rows:
        if row['sign_flip_rate'] is not None:flips.setdefault(row['video_id'],[]).append(row['sign_flip_rate'])
    summary['sign_flip_rate']=float(np.mean([np.mean(items) for items in flips.values()])) if flips else None
    changes={}
    for row in rows:changes.setdefault(row['video_id'],[]).append(row['absolute_change_mean'])
    summary['absolute_change_mean']=float(np.mean([np.mean(items) for items in changes.values()])) if changes else None
    report=dict(scope='RISE-A0 historical weights under a fixed RFV replay policy',earlier=early_binding,later=late_binding,
        capture_equivalence=equivalence,
        action_identity_verified=True,aggregate=summary,strata={key:video_aggregate(value) for key,value in strata.items()},
        state_metrics=rows,status='MEASURED',fvd_unlocked=False,
        duration_edges_seconds=duration_edges,duration_strata_source='tertiles of observed fit-action durations; descriptive only',
        interpretation='Actual drift measurement; not a same-state function forecast or new T-V trajectory')
    json_write(Path(args.output)/'RISE_A_T.json',report)
    print(json.dumps(dict(drift=summary,fvd_unlocked=False)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['headroom','drift'])
    parser.add_argument('--capture');parser.add_argument('--earlier',action='append');parser.add_argument('--later',action='append')
    parser.add_argument('--output',required=True);parser.add_argument('--bootstrap',type=int,default=10000)
    parser.add_argument('--capture-equivalence')
    args=parser.parse_args()
    if args.mode=='headroom':
        if not args.capture:parser.error('--capture required')
        headroom(args)
    else:
        if not args.earlier or not args.later:parser.error('--earlier/--later required')
        drift(args)


if __name__=='__main__':main()
