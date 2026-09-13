"""Recalculate archived numeric evidence; no weights, server, or model inference."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

def audit(root: Path) -> dict:
    with (root/'data/factorial_records.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    result={}
    for bb in ('S','B'):
        group=[r for r in rows if r['backbone']==bb]
        if len(group)!=8: raise ValueError(f'{bb}: expected all eight cells')
        m={r['T']+r['D']+r['S']:float(r['average_mAP_pct']) for r in group}
        if len(m)!=8: raise ValueError('Duplicate cells')
        result[bb]={
            'TD_at_S_dense_pp':m['110']-m['100']-m['010']+m['000'],
            'DS_at_T_dense_pp':m['011']-m['010']-m['001']+m['000'],
            'DS_at_T_sparse_pp':m['111']-m['110']-m['101']+m['100'],
            'TDS_third_order_pp':m['111']-m['110']-m['101']-m['011']+m['100']+m['010']+m['001']-m['000'],
            'D_loss_at_T384_S1_pp':m['110']-m['100'],
            'DS_loss_at_T384_pp':m['111']-m['100'],
        }
    with (root/'data/historical_frontier.csv').open(encoding='utf-8-sig') as f: front=list(csv.DictReader(f))
    dom=[]
    for a in front:
        for b in front:
            if a is b: continue
            ax,ay=float(a['gflops']),float(a['mAP_pct']);bx,by=float(b['gflops']),float(b['mAP_pct'])
            if ax<=bx and ay>=by and (ax<bx or ay>by):dom.append({'dominates':a['method'],'dominated':b['method']})
    plans=json.loads((root/'EXPERIMENTS.json').read_text())
    for p in plans['new_runs']:
        assert p['seed']==42 and p['runs']==1
        assert p['status']=='PROPOSED_NOT_REGISTERED'
        assert not p['performance_dependencies']
        assert p['mAP'] is None and p['gflops'] is None
    out={'scope':'Arithmetic on archived repository records only; not weights/GPU/full-test rerun',
         'archived_seed':3407,'proposed_seed':42,'factorials':result,
         'historical_same_scope_dominance':dom,
         'ema_half_life_updates':{'decay_0.99':math.log(.5)/math.log(.99),'decay_0.999':math.log(.5)/math.log(.999)},
         'proposed_configs':len(plans['new_runs']),
         'proposed_optimizer_updates_estimate':sum(p['training_updates'] for p in plans['new_runs']),
         'tests':{'eight_cells_per_backbone':True,'all_new_seed42_single_run':True,'no_performance_dependencies':True,'no_fake_results':True}}
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    out=audit(a.root)
    (a.root/'data/numeric_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
