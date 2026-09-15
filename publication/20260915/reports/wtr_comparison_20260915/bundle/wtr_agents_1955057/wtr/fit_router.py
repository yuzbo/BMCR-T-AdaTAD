"""Offline frame-refiner pilot using real paired GT values and optional future-value JSD.

No backbone training; no Graph dependency; no future-checkpoint file accepted.
The checkpoint exported here is EVALUATION-ONLY and cannot resume the old optimizer.
This tests the EXISTING one-swap frame refiner, not a full T/S/D value-routing method.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import json
import torch
from .integration import setup_repository
from .core import read_jsonl,align_records,extrapolate,action_distribution,js_distill,atomic_json,file_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--current',required=True);p.add_argument('--anchor');p.add_argument('--ema')
    p.add_argument('--state',choices=['learned','ema'],default='learned')
    p.add_argument('--mode',choices=['none','current','ema','future'],default='future')
    p.add_argument('--beta',type=float,default=1.1);p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--lambda-js',type=float,default=.1)
    p.add_argument('--temperature',type=float,default=1.);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',required=True);p.add_argument('--allow-descendant',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.mode=='future' and not args.anchor:p.error('future requires --anchor')
    if args.mode=='ema' and not args.ema:p.error('ema requires --ema (actual EMA replay, not beta=1)')
    if args.dry_run:print(json.dumps(vars(args),indent=2));return
    if Path(args.output).exists():raise FileExistsError(args.output)
    setup_repository(args.repo,args.allow_descendant)
    from h65.paper.routing import FrameRouter
    current=read_jsonl(args.current)
    if any(r['action_type']!='frame' for r in current):raise ValueError('Frame-refiner pilot only.')
    source=file_digest(args.checkpoint)
    if any(r['checkpoint_sha256']!=source or r['checkpoint_state']!=args.state for r in current):
        raise ValueError('Post-state features must come from this exact checkpoint and state.')
    aux=None
    if args.mode in ('future','ema'):
        aux=read_jsonl(args.anchor if args.mode=='future' else args.ema)
        aligned=align_records(current,aux)
    else:aligned=[(r,r) for r in current]
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=payload['metadata']['config']
    if cfg.get('recipe')=='graph_tad_v1':raise ValueError('Non-Graph pilot only.')
    torch.manual_seed(args.seed)
    router=FrameRouter(cfg.get('plan_aware_frame',False))
    prefix='frame_router.'
    state={k[len(prefix):]:v for k,v in payload[args.state].items() if k.startswith(prefix)}
    router.load_state_dict(state,strict=True)
    fit=[r for r,_ in aligned if r['split']=='fit']
    if not fit:raise RuntimeError('No fit split.')
    fixed_scale=torch.tensor([r['actual_delta'] for r in fit]).abs().mean(0).clamp_min(1e-4)
    with torch.no_grad():router.scales.copy_(fixed_scale);router.sigma_calibration.fill_(1.)
    groups=defaultdict(list)
    for cur,other in aligned:groups[(cur['split'],cur['video_name'],cur['support_id'])].append((cur,other))
    cached=[];calibration=[]
    for (split,video,support),rows in groups.items():
        if split=='holdout':continue
        x=torch.tensor([r[0]['frame_features'] for r in rows],dtype=torch.float32)
        y=torch.tensor([r[0]['actual_delta'] for r in rows],dtype=torch.float32)
        old=torch.tensor([r[1]['actual_delta'] for r in rows],dtype=torch.float32)
        if args.mode=='future':
            if any(r[1]['successful_updates']>=r[0]['successful_updates'] for r in rows):
                raise ValueError('Anchor must precede current.')
            teacher=extrapolate(old,y,args.beta)
        elif args.mode=='ema':teacher=old.detach()
        else:teacher=y.detach()
        item=(x,y,action_distribution(teacher,fixed_scale,temperature=args.temperature).detach())
        (cached if split=='fit' else calibration).append(item)
    if not cached:raise RuntimeError('No trainable groups.')
    optimizer=torch.optim.AdamW(router.network.parameters(),lr=args.lr,weight_decay=.01)
    generator=torch.Generator().manual_seed(args.seed);history=[]
    router.train()
    for step in range(args.steps):
        x,y,teacher=cached[int(torch.randint(len(cached),(1,),generator=generator))]
        raw=router.network(x);mean=raw[:,:2];lv=raw[:,2:].clamp(-8,8)
        nll=.5*((mean-y/fixed_scale).square()*(-lv).exp()+lv).mean()
        student=action_distribution(mean*fixed_scale,fixed_scale,temperature=args.temperature)
        divergence=js_distill(student,teacher)
        loss=nll+(0. if args.mode=='none' else args.lambda_js)*divergence
        if not torch.isfinite(loss):raise RuntimeError('Non-finite router loss.')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(router.network.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step%100==0 or step==args.steps-1:
            history.append(dict(step=step+1,nll=float(nll.detach()),js=float(divergence.detach())))
    # Calibration split never updates score weights. Fit a conservative std multiplier only.
    router.eval()
    with torch.no_grad():
        if calibration:
            residuals=[];vars=[]
            for x,y,_ in calibration:
                r=router.network(x);residuals.append((r[:,:2]-y/fixed_scale).square())
                vars.append(r[:,2:].clamp(-8,8).exp())
            multiplier=(torch.cat(residuals).mean(0)/torch.cat(vars).mean(0).clamp_min(1e-6)).sqrt().clamp(.25,4.)
            router.sigma_calibration.copy_(multiplier)
    learned={k:v.detach().clone() for k,v in payload[args.state].items()}
    for name,value in router.state_dict().items():
        key=prefix+name
        if key not in learned or learned[key].shape!=value.shape:
            raise RuntimeError('Original checkpoint trainable-state contract changed: '+key)
        learned[key]=value.detach().clone()
    metadata=dict(payload['metadata'])
    metadata['wtr_offline_pilot']=dict(mode=args.mode,beta=args.beta,state_source=args.state,
        parent_sha256=source,steps=args.steps,fit_videos=sorted({r['video_name'] for r in fit}),
        bank_hash=current[0]['bank_hash'],fixed_component_scale=fixed_scale.tolist(),
        calibration='Separate training videos; sigma only; heldout windows untouched',
        scope='Only existing frame refiner changed; not full T/S/D or recursive end-to-end RISE',
        training_compute='Includes bank creation/reexecution; report separately from fit CPU time')
    exported=dict(learned=learned,ema={k:v.clone() for k,v in learned.items()},metadata=metadata,
        epoch_index=payload.get('epoch_index'),successful_updates=payload['successful_updates'],
        evaluation_only=True,optimizer_resume_forbidden=True)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix('.tmp');torch.save(exported,tmp);tmp.replace(out)
    atomic_json(out.with_suffix('.receipt.json'),dict(history=history,metadata=metadata['wtr_offline_pilot'],
        checkpoint_sha256=file_digest(out),official_test=False))
    print(f'Saved evaluation-only checkpoint to {out}; no detector training or full test performed.')


if __name__=='__main__':main()
