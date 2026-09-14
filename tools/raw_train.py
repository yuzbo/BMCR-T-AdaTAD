#!/usr/bin/env python3
"""Fit only the shared Temporal Value head; holdout labels never enter optimizer/scales."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
import numpy as np
import torch
from h65.raw.value import TemporalValueHead,simple_proxy,FEATURE_DIM,SCALARS
from h65.paper.runtime import json_write


def read_bank(paths,protocol):
    rows={}
    for folder in paths:
        for path in sorted((Path(folder)/'groups').glob('*.json')):
            row=json.loads(path.read_text())
            key=(row['episode']['video_id'],row['episode']['window_index'],row['domain'],row['state_id'])
            if key in rows:
                raise ValueError(f'Duplicate CF state across supplied banks: {key}')
            if row['episode']['split'] != 'development' or row['episode']['video_id'] not in protocol['splits'][row['split']]:
                raise ValueError('CF labels are not in their registered training-side split')
            rows[key]=row
    if not rows:
        raise ValueError('No CF records')
    revisions={r['source_revision'] for r in rows.values()}
    if len(revisions)!=1:
        raise ValueError('Mixed source revisions require a separate explicit experiment')
    return list(rows.values())


def distribution(rows):
    actions=[a for r in rows for a in r['actions']]
    y=np.array([a['gain_cls_loc'] for a in actions],dtype=float)
    changed=np.array([a['n_changed_pairs'] for a in actions])
    gain=y.sum(1)
    replay=[a['replay_max_error'] for a in actions if a['replay_max_error'] is not None]
    return dict(videos=len({r['episode']['video_id'] for r in rows}),states=len(rows),swaps=len(actions),
        gain_quantiles=np.quantile(y,[0,.1,.5,.9,1],axis=0).tolist(),positive_fraction=float((gain>0).mean()),
        nonzero_fraction=float((gain!=0).mean()),max_replay_error=max(replay,default=None),
        changed_pairs_quantiles=np.quantile(changed,[0,.5,.9,1]).tolist(),
        gain_changed_pairs_correlation=(float(np.corrcoef(gain,changed)[0,1]) if np.std(gain)>0 and np.std(changed)>0 else None),
        interpretation='diagnostics only; finite task-loss actions are not an AP upper bound')


def evaluate(head,rows):
    from scipy.stats import spearmanr
    by_video={}
    for row in rows:
        if not row['actions']:
            continue
        x=torch.tensor([a['descriptor'] for a in row['actions']],dtype=torch.float32)
        y=torch.tensor([a['gain_cls_loc'] for a in row['actions']],dtype=torch.float32)
        with torch.no_grad():
            prediction=head.utility(x)
            target=(y/head.target_scale).sum(-1)
            proxy=simple_proxy(x)
        best=max(0.,float(target.max()))
        chosen=float(target[prediction.argmax()]) if float(prediction.max())>0 else 0.
        proxy_chosen=float(target[proxy.argmax()]) if float(proxy.max())>0 else 0.
        uniform_swap=float(target.mean())  # Expected gain of a random legal exchange; no label selection.
        rho=float(spearmanr(prediction.numpy(),target.numpy()).statistic) if prediction.std()>0 and target.std()>0 else None
        by_video.setdefault(row['episode']['video_id'],[]).append(dict(value_regret=best-chosen,
            stop_regret=best,uniform_swap_regret=best-uniform_swap,proxy_regret=best-proxy_chosen,
            spearman=rho,value_gain=chosen,proxy_gain=proxy_chosen))
    means={v:{key:float(np.mean([r[key] for r in values if r[key] is not None]))
              for key in values[0] if any(r[key] is not None for r in values)} for v,values in by_video.items()}
    summary={key:float(np.mean([r[key] for r in means.values() if key in r]))
             for key in {k for r in means.values() for k in r}}
    return dict(videos=len(means),video_means=means,**summary)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',action='append',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    p.add_argument('--analyze-only',action='store_true');p.add_argument('--steps',type=int,default=2000)
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    protocol=json.loads(Path(args.protocol).read_text());rows=read_bank(args.bank,protocol)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    stats={split:distribution([r for r in rows if r['split']==split]) for split in protocol['splits']}
    json_write(out/'bank_diagnostics.json',stats)
    if args.analyze_only:
        print(json.dumps(stats));return
    observed={r['episode']['video_id'] for r in rows}
    mini=set(sum(protocol['mini'].values(),[]));full=set(sum(protocol['splits'].values(),[]))
    if observed not in (mini,full):
        raise ValueError('Refuse to train from an incomplete registered video cohort')
    if any(stats[s]['max_replay_error'] != 0. for s in stats):
        raise ValueError('CF replay contract failed')
    fit=[a for r in rows if r['split']=='fit' for a in r['actions']]
    x=torch.tensor([a['descriptor'] for a in fit],dtype=torch.float32)
    y=torch.tensor([a['gain_cls_loc'] for a in fit],dtype=torch.float32)
    if x.shape[1]!=FEATURE_DIM or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Descriptor/label contract failed')
    if y.abs().max()==0:
        raise ValueError('All fit gains are zero: no Value fit before signal review')
    head=TemporalValueHead()
    head.input_mean.copy_(x.mean(0));head.input_scale.copy_(x.std(0).clamp_min(1e-6))
    head.target_scale.copy_(y.square().mean(0).sqrt().clamp_min(1e-8))
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-3)
    generator=torch.Generator().manual_seed(42)
    head.train()
    for step in range(args.steps):
        index=torch.randint(len(x),(min(128,len(x)),),generator=generator)
        prediction=head(x[index])/head.target_scale
        loss=torch.nn.functional.smooth_l1_loss(prediction,y[index]/head.target_scale)
        optimizer.zero_grad();loss.backward();optimizer.step()
    head.eval()
    evaluation={split:evaluate(head,[r for r in rows if r['split']==split]) for split in protocol['splits']}
    holdout=evaluation['holdout']
    passed=(holdout['value_regret']<min(holdout['stop_regret'],holdout['uniform_swap_regret'],holdout['proxy_regret'])
            and holdout.get('spearman',0)>0)
    # The mini-bank is a pilot even if its four holdout videos happen to improve.
    eligible=passed and observed==full and args.steps==2000
    report=dict(evaluation=evaluation,holdout_gate_passed=passed,publication_eligible=eligible,
        bank_scope='full' if observed==full else 'mini',fit_videos=stats['fit']['videos'],steps=args.steps,
        fit_only_normalization=True,explicit_domain_indicator=False,detector_updates=0,
        source_revision=(ROOT/'CODE_REVISION').read_text().strip(),bank_revision=rows[0]['source_revision'],
        descriptor_scalars=SCALARS,protocol=protocol,
        gate='video-mean Value regret below stop/uniform-swap/proxy and positive within-state rank correlation')
    torch.save(dict(state_dict=head.state_dict(),report=report),out/'temporal_value.pth')
    json_write(out/'training_report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
