#!/usr/bin/env python3
"""Preregistered R0/R1 comparison on the shared bank and fixed unseen swaps."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
import torch
from h65.paper.runtime import json_write
from h65.rfv.dataset import load_bank,normalization,bank_identities
from h65.rfv.metrics import paired_video_difference
from h65.rfv.value import from_snapshot
from tools.rfv_fit import train_one,evaluate,average_seeds,controls
from tools.rfv_action_holdout import subset

ARMS={'plain_r0':('plain_m','r0'),'plain_r1':('plain_m','r1'),'plain_l_r1':('plain_l','r1')}


def unseen_split(rows):
    fit=[];held=[];identity={}
    for row in rows:
        if row['partition']!='fit' or not row['action_pairs']:continue
        if len(row['action_pairs'])!=16:raise ValueError('Registered physical8/8 split requires16 candidates')
        order=sorted(range(16),key=lambda i:(row['action_pairs'][i][1],row['action_pairs'][i][0]))
        fit.append(subset(row,order[::2],'fit'));held.append(subset(row,order[1::2],'held'))
        identity[row['state_key']]=dict(fit=[row['actions'][i]['id'] for i in order[::2]],
            held=[row['actions'][i]['id'] for i in order[1::2]])
    return fit,held,identity


def compare(left,right):
    return {metric:paired_video_difference(left,right,metric) for metric in ('regret','ndcg','topk_overlap','spearman')}


def improved(interval):
    return interval['ci95'] is not None and interval['ci95'][1]<0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bank',action='append',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();torch.set_num_threads(4)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    revision=(ROOT/'source_revision.txt').read_text().strip()
    if (out/'VALUE_R1_GATE.json').exists():
        prior=json.loads((out/'VALUE_R1_GATE.json').read_text())
        if prior['source_revision']!=revision:raise ValueError('Finished experiment belongs to a different source')
        print(json.dumps(dict(status=prior['status'],already_complete=True)));return
    rows,binding=load_bank(args.bank)
    if binding['cohort']!='mini':raise ValueError('This first R1 gate uses the registered mini; final outer20 stays sealed')
    if len(binding['capture_revisions'])!=1:raise ValueError('R1 must compare one shared captured bank')
    seeds=binding['protocol']['head_seeds'];args.steps=binding['protocol']['head_steps'];args.temperature=1.
    fit8,held8,split=unseen_split(rows)
    cal=[r for r in rows if r['partition']=='calibration']
    holdout=[r for r in rows if r['partition']=='holdout']
    # Reuse the complete set for the cross-video test; the within-video test
    # withholds eight labels before any input or target normalization is fitted.
    suites={'full':(rows,holdout),'within':(fit8+cal,held8)}
    config=dict(schema='RFV_FAST_SPRINT_V3_VALUE_R1',source_revision=revision,bank=binding,
        arms=ARMS,seeds=seeds,steps=args.steps,temperature=1.,huber_weight=.1,stop_threshold=0.,
        primary='actual raw cls+loc finite-budget regret',input_revision='unchanged407D',
        candidate_revision='unchanged geometric4round/16pairs',within_split=split,
        gate='R1-R0 cross-video regretCI<0; beats all simple controls; within regretCI<0 and Spearman improvementCI>0; seed direction stable',
        official_test_used=False,outer_holdout_used=False,scope='mini development gate; formal160/20/20 admission still required')
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=json.loads(json.dumps(config)):
        raise ValueError('Experiment configuration changed after registration')
    json_write(out/'config.json',config);json_write(out/'manifest.json',dict(**config,identities=bank_identities(rows)))
    (out/'source_revision.txt').write_text(revision+'\n')
    models={};payloads={}
    for suite,(training,evaluation) in suites.items():
        stats=normalization(training)
        for arm,(variant,objective) in ARMS.items():
            args.objective=objective
            for seed in seeds:
                key=f'{suite}_{arm}_s{seed}';path=out/(key+'.pth')
                if path.exists():
                    payload=torch.load(path,map_location='cpu',weights_only=False)
                    if payload['source_revision']!=revision or payload['experiment']!=dict(suite=suite,arm=arm,bank=binding):
                        raise ValueError('Saved head differs from this frozen comparison')
                    model=from_snapshot(payload['snapshot'],args.device)
                else:
                    model,payload=train_one(variant,seed,training,stats,args)
                    payload.update(source_revision=revision,experiment=dict(suite=suite,arm=arm,bank=binding))
                    torch.save(payload,path)
                models[key]=model.cpu();payloads[key]=payload
                json_write(out/'progress.json',dict(completed=key,completed_heads=len(models),total_heads=18,source_revision=revision))
                print(json.dumps(dict(completed=key,calibration=payload['calibration']['mean'])),flush=True)
    lock=dict(source_revision=revision,baseline='plain_r1',temperature=1.,stop_threshold=0.,
        selection='architecture/objective/temperature/STOP preregistered; no inner or outer selection',
        calibration={key:value['calibration'] for key,value in payloads.items()})
    json_write(out/'selection_locked.json',lock)
    evaluations={};combined={};baselines={}
    for suite,(training,evaluation) in suites.items():
        evaluations[suite]={arm:[evaluate(models[f'{suite}_{arm}_s{s}'].to(args.device),evaluation,args.device) for s in seeds] for arm in ARMS}
        combined[suite]={arm:average_seeds(value) for arm,value in evaluations[suite].items()}
        baselines[suite]=controls(evaluation)
    difference={suite:compare(values['plain_r1'],values['plain_r0']) for suite,values in combined.items()}
    vs_simple={suite:{name:paired_video_difference(values['plain_r1'],control,'regret')
        for name,control in baselines[suite].items()} for suite,values in combined.items()}
    stable=all(new['mean']['regret']<old['mean']['regret'] for suite in suites
        for new,old in zip(evaluations[suite]['plain_r1'],evaluations[suite]['plain_r0']))
    within_rho=difference['within']['spearman']
    simple_win=all(result['mean'] is not None and result['mean']<0 for controls_ in vs_simple.values() for result in controls_.values())
    passed=bool(improved(difference['full']['regret']) and improved(difference['within']['regret']) and
        within_rho['ci95'] is not None and within_rho['ci95'][0]>0 and
        combined['within']['plain_r1']['mean']['spearman']>0 and stable and simple_win)
    report=dict(config=config,source_revision=revision,aggregate=combined,seed_evaluations=evaluations,
        controls=baselines,r1_minus_r0=difference,r1_minus_simple=vs_simple,seed_directions_stable=stable,
        status='LEARNABILITY_PASS' if passed else 'LEARNABILITY_FAIL',passed=passed,
        preliminary=True,task_unlocked=False,new_cf_queries=0,outer_holdout_used=False,
        next='Complete formal160/20/20 prefit/admission and recipe integration' if passed else 'Keep task-levelGraph/FVD locked; inspect representation under a new preregistration')
    json_write(out/'VALUE_R1_GATE.json',report);json_write(out/'value_metrics.json',combined)
    json_write(out/'metrics.json',dict(status=report['status'],passed=passed,preliminary=True,task_unlocked=False))
    print(json.dumps(dict(status=report['status'],passed=passed,task_unlocked=False)),flush=True)


if __name__=='__main__':main()
