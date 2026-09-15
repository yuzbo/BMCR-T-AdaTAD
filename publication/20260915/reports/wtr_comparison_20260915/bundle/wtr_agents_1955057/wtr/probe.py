"""Build immutable training-video action banks; replay frame and operator counterfactuals.

Run --help without CUDA. No scheduler submission, repo edits, training or test-set search.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import random
import torch
from . import BASE_SHA
from .core import atomic_json, digest, jsonable
from .integration import load, training_data, sample_at, window_signature, pack_selection, unpack_selection, execute


def build(args):
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    model, cfg, resources, meta = load(args)
    from h65.full.runtime import to_gpu
    from h65.paper.geometry import candidate_mask
    from h65.paper.routing import candidate_pairs
    ds = training_data(cfg, model.config, args.epoch, model.config['seed'])
    plan_ids = [int(s) for s in args.plans.split(',')]
    if not plan_ids or any(not 0 <= p < len(model.menu) for p in plan_ids):
        raise ValueError('Invalid plan list.')
    if len({model.plan(p)['frames'] for p in plan_ids}) != 1:
        raise ValueError('First bank must share K, e.g. plans 1,3,4. Build another bank for another K.')
    entries = []; seen = set(); rng = random.Random(args.seed)
    for idx in range(len(ds)):
        cpu = sample_at(ds, idx); name = str(cpu['metas'][0]['video_name'])
        if name in seen:
            continue
        data = to_gpu(cpu); masks = candidate_mask(data)
        # Start on complete windows: padding diagnostics are a separate mandatory test.
        if int(masks.sum()) != 768:
            continue
        signature, info = window_signature(cpu)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            preview = model.encoder.preview(data['inputs'], masks)
            sel = model.encoder.select(preview, masks, model.plan(plan_ids[0])['frames'],
                                       args.selector, data['metas'])
        pairs = candidate_pairs(preview, sel, masks, args.scope)
        if not pairs:
            continue
        pairs = sorted(rng.sample(pairs, min(args.pairs, len(pairs))))
        entries.append(dict(index=idx, video_name=name, window_hash=signature, window_info=info,
                            selection=pack_selection(sel), pairs=pairs,
                            plans={str(p): model.plan(p) for p in plan_ids}))
        seen.add(name)
        if len(entries) >= args.max_videos:
            break
    if len(entries) < 6:
        raise RuntimeError('Need >=6 distinct complete training videos; no dataset substitution allowed.')
    order = sorted(seen, key=lambda n: digest([args.seed, n])); ncal = max(1, len(order)//5)
    split = {name: ('calibration' if i < ncal else 'holdout' if i < 2*ncal else 'fit')
             for i,name in enumerate(order)}
    for row in entries:
        row['split'] = split[row['video_name']]
    body = dict(schema='wtr.action_bank.v1', base_sha=BASE_SHA,
                config=model.config, checkpoint=meta, data_epoch=args.epoch,
                data_seed=model.config['seed'], proposal_seed=args.seed,
                measurement_loss_normalizer=model.readout.detector.rpn_head.loss_normalizer.detach().cpu().clone(),
                selector=args.selector, partner_scope=args.scope, entries=entries,
                notes='Training-video diagnostic; GT-conditioned rankings are not deployable oracles.')
    body['bank_hash'] = digest(body)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix('.tmp'); torch.save(body, temp); temp.replace(out)
    atomic_json(out.with_suffix('.manifest.json'), {k:v for k,v in body.items() if k != 'entries'} | {
        'videos': [dict(video_name=r['video_name'], split=r['split'], index=r['index'],
                        window_hash=r['window_hash']) for r in entries],
        'windows':len(entries), 'pairs_per_state':args.pairs, 'plan_ids':plan_ids})
    print(f"Saved {len(entries)} training-video windows; bank={body['bank_hash']}")


def record_base(bank, row, plan_id):
    support = digest(dict(bank=bank['bank_hash'], window=row['window_hash'],
                          selection=row['selection']['indices'], valid=row['selection']['valid'],
                          plan=row['plans'][str(plan_id)]))
    return dict(bank_hash=bank['bank_hash'], window_hash=row['window_hash'],
                video_name=row['video_name'], split=row['split'], index=row['index'],
                support_id=support, plan_index=int(plan_id))


def emit(stream, common, extra):
    stream.write(json.dumps(jsonable(common | extra), ensure_ascii=False, allow_nan=False)+'\n')
    stream.flush()


def frame_records(model, data, bank, row, plan_id, measured):
    from h65.paper.geometry import candidate_mask
    from h65.paper.routing import swap_selection
    sel = unpack_selection(row['selection'], data['inputs'].device)
    plan = row['plans'][str(plan_id)]
    base, detail, cb, scope = execute(model, data, plan, sel, measured=measured)
    preview = detail['preview']; masks = candidate_mask(data)
    common = record_base(bank, row, plan_id)
    with torch.no_grad():
        features = model.frame_router.features(preview, sel, masks, row['pairs'], detail['plan'])
        means, logvars = model.frame_router.distribution(preview, sel, masks, row['pairs'], detail['plan'])
        pred = means * model.frame_router.scales
        sigma = logvars.mul(.5).exp() * model.frame_router.scales
    for i,pair in enumerate(row['pairs']):
        changed = swap_selection(sel, *pair)
        loss, _, ca, _ = execute(model, data, plan, changed, preview=preview, measured=measured)
        identity = digest([common['support_id'], 'frame', pair])
        yield common | dict(action_id=identity, action_type='frame', frame_pair=pair,
            actual_delta=(base-loss), loss_base=base, loss_action=loss,
            frame_features=features[i], predicted_delta=pred[i], predicted_sigma=sigma[i],
            base_forward_flops=cb, action_forward_flops=ca, delta_forward_flops=ca-cb,
            target_origin='paired_GT_cls_reg_only', cost_scope=scope,
            downstream_mask_semantics='recomputed by the checkpoint; use plan1 for clean no-D/S frame forecast',
            note='Equal-K exchange. Do not divide utility by zero incremental cost.')


def operator_records(model, data, bank, row, plan_id, measured, groups, seed):
    """Fixed-support local operators: attention HOLD, FFN LIGHT, and their joint removal.

    Not original A-MoD depth bypass: these isolate access/update decisions while keeping
    global TIA and other masks fixed. Both branch parity and all-true validity are logged.
    """
    for key in ('frames','depth_capacity','space_capacity'):
        if key in model.config:
            raise ValueError(f'Fixed config override {key} would defeat operator isolation.')
    sel = unpack_selection(row['selection'], data['inputs'].device)
    original = dict(row['plans'][str(plan_id)])
    original.update(depth=1., space=1.)
    loss0, detail, _, _ = execute(model, data, original, sel, measured=measured)
    trace = detail['trace']; depth=model.encoder.depth
    packs = trace['patch_input_shape'][0]
    h=trace['patch_input_shape'][-2]//16; w=trace['patch_input_shape'][-1]//16
    valid = sel.valid.reshape(packs,8,2).any(-1).repeat_interleave(h*w,1)
    layers = list(original.get('mod_layers', range(1, depth-1, 2)))
    masks = {key:[valid.clone() for _ in range(depth)] for key in ('depth','query','spatial')}
    # <1 enables the corresponding code path; explicit all-true masks override capacity.
    # Attention full KV is forced. There is no cross-checkpoint adaptive mask recomputation.
    control = dict(original, query_ratio=.999, space=.999, full_kv=True,
                   use_light=True, route_masks=masks)
    base, frozen, cb, scope=execute(model,data,control,sel,preview=detail['preview'],measured=measured)
    parity=float((base-loss0).abs().max())
    common=record_base(bank,row,plan_id)
    candidates=[(pack,t,y,x) for pack in range(packs) for t in range(8)
                for y in range(0,h-1,2) for x in range(0,w-1,2)]
    rng=random.Random(seed+row['index'])
    selected=rng.sample(candidates,min(groups,len(candidates)))
    for layer in layers:
        for pack,t,y,x in selected:
            ids=[t*h*w+(y+dy)*w+x+dx for dy in (0,1) for dx in (0,1)]
            for kind in ('attention_hold','ffn_light','attention_hold+ffn_light'):
                route={k:[m.clone() for m in v] for k,v in masks.items()}
                if kind in ('attention_hold','attention_hold+ffn_light'):
                    route['query'][layer][pack,ids]=False
                if kind in ('ffn_light','attention_hold+ffn_light'):
                    route['spatial'][layer][pack,ids]=False
                plan=dict(control,route_masks=route)
                loss,_,ca,_=execute(model,data,plan,sel,preview=frozen['preview'],measured=measured)
                aid=digest([common['support_id'],kind,layer,pack,t,y,x])
                yield common|dict(action_id=aid,action_type=kind,original_block_id=layer,
                    token_group=[pack,t,y,x],actual_delta=base-loss,
                    removal_cost=loss-base,loss_base=base,loss_action=loss,
                    branch_parity_max_abs=parity,base_forward_flops=cb,action_forward_flops=ca,
                    delta_forward_flops=ca-cb,cost_scope=scope,
                    target_origin='paired_GT_cls_reg_only',
                    note='2x2 tile intervention; all other route masks fixed; original global TIA retained.')


def replay(args):
    out=Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    bank=torch.load(args.bank,map_location='cpu',weights_only=False)
    wanted=bank.pop('bank_hash'); actual=digest(bank); bank['bank_hash']=wanted
    if wanted!=actual:
        raise RuntimeError('Bank hash mismatch.')
    model,cfg,resources,meta=load(args)
    if bank['config']!=model.config:
        raise ValueError('Replay must use exactly the bank model config.')
    if 'measurement_loss_normalizer' not in bank:
        raise ValueError('Build a new bank with a fixed measurement loss normalizer.')
    model._wtr_measurement_normalizer=bank['measurement_loss_normalizer']
    ds=training_data(cfg,model.config,bank['data_epoch'],bank['data_seed'])
    from h65.full.runtime import to_gpu
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+'.tmp')
    count=0
    with tmp.open('w',encoding='utf-8') as stream:
        for row in bank['entries']:
            cpu=sample_at(ds,row['index']);signature,_=window_signature(cpu)
            if signature!=row['window_hash']:
                raise RuntimeError(f"Augmentation/window changed at {row['index']}; refuse unmatched values.")
            data=to_gpu(cpu)
            for pid in row['plans']:
                if args.kind=='operators' and int(pid)!=args.operator_plan:
                    continue
                iterator=(frame_records(model,data,bank,row,pid,args.measured) if args.kind=='frame' else
                          operator_records(model,data,bank,row,pid,args.measured,args.groups,args.seed))
                for record in iterator:
                    emit(stream,{},record|{'checkpoint_sha256':meta['checkpoint_sha256'],
                        'checkpoint_state':args.state,'successful_updates':meta['successful_updates'],
                        'checkout_sha':meta['probe_checkout']})
                    count+=1
            print(f"{row['video_name']}: {count} completed counterfactuals",flush=True)
    if count==0:
        raise RuntimeError('No records; check operator_plan is in the bank.')
    tmp.replace(out)
    atomic_json(out.with_suffix('.receipt.json'),dict(status='PROBE_COMPLETED',records=count,
        actual_traces=True,cases=count,bank_hash=wanted,metadata=meta,
        official_test=False,official_map_computed=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['build','replay'])
    p.add_argument('--repo',required=True);p.add_argument('--config',required=True)
    p.add_argument('--resources',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--state',choices=['ema','learned'],default='learned')
    p.add_argument('--output',required=True);p.add_argument('--allow-descendant',action='store_true')
    p.add_argument('--epoch',type=int,default=0);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-videos',type=int,default=32);p.add_argument('--pairs',type=int,default=8)
    p.add_argument('--plans',default='1');p.add_argument('--selector',choices=['anchor','uniform'],default='anchor')
    p.add_argument('--scope',choices=['local','global'],default='local')
    p.add_argument('--bank');p.add_argument('--kind',choices=['frame','operators'],default='frame')
    p.add_argument('--operator-plan',type=int,default=1);p.add_argument('--groups',type=int,default=1)
    p.add_argument('--measured',action='store_true');p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.pairs<1 or args.groups<1 or args.max_videos<6:
        p.error('pairs/groups >=1 and max-videos >=6 required.')
    if args.command=='replay' and not args.bank:
        p.error('replay requires --bank')
    if args.dry_run:
        print(json.dumps(vars(args),indent=2));return
    build(args) if args.command=='build' else replay(args)


if __name__=='__main__':main()
