"""No parameter updates: real interventions, finite references and controlled recovery."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from .actions import (LAYERS, UNIFORM_ORDER, GROUP_COUNTS, seed_for, positions,
    primitive_samples, full_masks, remove_points, grouped_masks, group_attention, finite_budget_subset)
from .compute import observation_intervention
from .data import window_metadata, position_metadata
from .reference import proxies
from .temporal import execute_temporal, uniform_indices, support_groups, group_indices


def scalar_record(base,action,definition,meta,center,sampling='population',kind='upgrade'):
    delta=np.asarray(base['losses'])-np.asarray(action['losses'])
    return dict(action=definition,kind=kind,sampling=sampling,
        base_loss_cls_reg=np.asarray(base['losses']).tolist(),
        action_loss_cls_reg=np.asarray(action['losses']).tolist(),
        value_cls_reg=delta.tolist(),value=float(delta.sum()),
        base_gflops=base['gflops'],action_gflops=action['gflops'],
        delta_gflops=action['gflops']-base['gflops'],
        position=position_metadata(meta,center))


def perturb(reference,data,start,kind):
    other=dict(data)
    if kind=='benign':
        image=data['inputs'][:,:,:,start:start+1]
        shape=image.shape
        flat=image.reshape(-1,3,160,160)
        small=torch.nn.functional.interpolate(flat,size=(158,158),mode='bilinear',align_corners=False)
        large=torch.nn.functional.interpolate(small,size=(160,160),mode='bilinear',align_corners=False)
        pixels=data['inputs'].clone();pixels[:,:,:,start:start+1]=large.reshape(shape)
        other['inputs']=pixels
    else:
        other['inputs']=observation_intervention(data['inputs'],start,start+1,kind)
    return reference.execute(other)


def frame_pairs(data,meta):
    base=uniform_indices(data,384)
    valid=int(data['masks'].sum())
    selected=base.cpu().numpy();missing=np.setdiff1d(np.arange(valid),selected)
    pairs=[];seen=set()
    for center in positions(valid):
        if len(missing)==0:break
        insert=int(missing[np.argmin(np.abs(missing-center))])
        remove=int(selected[np.argmin(np.abs(selected-insert))])
        if (remove,insert) not in seen:
            pairs.append(dict(remove=remove,insert=insert));seen.add((remove,insert))
    return base,pairs


def exchange_indices(base,pairs):
    result=base.clone()
    removes=[p['remove'] for p in pairs];inserts=[p['insert'] for p in pairs]
    if len(set(removes))!=len(removes) or len(set(inserts))!=len(inserts):
        raise ValueError('Coalition contains conflicting exchanges')
    for pair in pairs:
        result[result==pair['remove']]=pair['insert']
    return result.sort().values


def conditional_positions(data,meta):
    """GT-directed positions are never included in population estimates."""
    times=np.asarray(meta['frame_indices'])/meta['fps']
    valid=int(data['masks'].sum())
    intervals=[g for g in meta['gt'] if g['fully_contained']]
    selected=[]
    if intervals:
        for j in np.unique(np.linspace(0,len(intervals)-1,min(2,len(intervals))).round().astype(int)):
            start,end=intervals[j]['segment']
            for fraction in (.05,.5,.95):
                t=start+fraction*(end-start)
                selected.append(int(np.argmin(np.abs(times[:valid]-t))))
    background=[i for i in positions(valid,32) if position_metadata(meta,i)['region']=='background']
    if background:
        selected.extend([background[0],background[-1]])
    return sorted(set(selected))


def population(reference,data,class_map,meta,preflight=False):
    reference.load_light()
    dense=reference.execute(data,capture_proxies=True)
    points=primitive_samples(data,meta)
    candidates=positions(int(data['masks'].sum()))
    if preflight:
        points=points[:1];candidates=candidates[:1]
    records=[];coalitions=[]
    null=reference.execute(dict(data,inputs=data['inputs'].clone()))
    null_record=scalar_record(dense,null,dict(axis='null',operation='identical RGB replay'),meta,candidates[0],sampling='control')
    if not np.array_equal(null['losses'],dense['losses']):
        if np.max(np.abs(null['losses']-dense['losses']))>1e-7:
            raise RuntimeError('Null paired loss exceeds the locked FP32 numerical tolerance')
    records.append(null_record)
    benign=perturb(reference,data,candidates[meta['window_index']%len(candidates)],'benign')
    records.append(scalar_record(dense,benign,dict(axis='benign',operation='160-158-160 bilinear RGB'),meta,
        candidates[meta['window_index']%len(candidates)],sampling='control',kind='perturbation'))
    observation_values={}
    for kind in ('neighbor','interpolate','low_resolution'):
        values=[]
        for center in candidates:
            changed=perturb(reference,data,center,kind)
            # R=L(intervened)-L(dense), while record.value uses the upgrade orientation.
            row=scalar_record(changed,dense,dict(axis='O',operation=kind,candidate=int(center),span=1),meta,center,kind='removal_effect')
            row['proxies']=proxies(dense,center)
            records.append(row);values.append(row['value'])
        observation_values[kind]=values
    for axis in ('D','S'):
        values=[]
        for point in points:
            am,fm=remove_points(data,axis,[point])
            cheap=reference.execute(data,am,fm)
            row=scalar_record(cheap,dense,dict(axis=axis,operation='light_to_heavy',**point),meta,point['candidate_center'])
            row['proxies']=proxies(dense,point['candidate_center'])
            row['proxies']['attention_score']=float(dense['token_attention'][point['layer']-1][point['native_time'],point['y'],point['x']])
            records.append(row);values.append(row['value'])
            if preflight:
                ledger=reference.shape_cost(dense['gflops'],am,fm)
                if abs(ledger-cheap['gflops'])>1e-5:
                    raise RuntimeError(f'{axis} measured operator cost disagrees with the shape ledger: {ledger} vs {cheap["gflops"]}')
                if cheap['gflops']>=dense['gflops']:
                    raise RuntimeError('Light substitution did not reduce actual heavy execution')
        if not preflight:
            orders={'stratified':list(range(len(points))), 'low_single_effect':np.argsort(np.abs(values),kind='stable').tolist()}
            for policy,order in orders.items():
                for size in (1,2,4,8,16):
                    chosen=order[:size]
                    if len(chosen)!=size:continue
                    am,fm=remove_points(data,axis,[points[i] for i in chosen])
                    changed=reference.execute(data,am,fm)
                    joint=changed['loss']-dense['loss']
                    coalitions.append(dict(axis=axis,policy=policy,size=size,actions=chosen,R=joint,
                        sum_single_R=float(np.sum(np.asarray(values)[chosen])),
                        interaction=joint-float(np.sum(np.asarray(values)[chosen])),
                        delta_gflops=changed['gflops']-dense['gflops']))
    base_indices,pairs=frame_pairs(data,meta)
    if preflight:pairs=pairs[:1]
    tbase=execute_temporal(reference,data,base_indices)
    for pair in pairs:
        indices=exchange_indices(base_indices,[pair])
        changed=execute_temporal(reference,data,indices)
        row=scalar_record(tbase,changed,dict(axis='T',operation='individual_frame_exchange',**pair),meta,pair['insert'])
        row['proxies']=proxies(dense,pair['insert'])
        records.append(row)
        if abs(row['delta_gflops'])>1e-8:
            raise RuntimeError('Fixed-K temporal exchange changed the inference matrix budget')
    if not preflight:
        tvalues=[r['value'] for r in records if r['action']['axis']=='T']
        for policy,order in [('stratified',list(range(len(pairs)))),('high_single_value',np.argsort(-np.asarray(tvalues),kind='stable').tolist())]:
            for size in (1,2,4,8,16):
                chosen=order[:size]
                if len(chosen)!=size:continue
                actions=[pairs[i] for i in chosen]
                if len({a['remove'] for a in actions})!=size or len({a['insert'] for a in actions})!=size:continue
                changed=execute_temporal(reference,data,exchange_indices(base_indices,actions))
                value=tbase['loss']-changed['loss'];summed=float(np.asarray(tvalues)[chosen].sum())
                coalitions.append(dict(axis='T',policy=policy,size=size,actions=chosen,R=value,
                    sum_single_R=summed,interaction=value-summed,delta_gflops=changed['gflops']-tbase['gflops'],
                    orientation='exchange benefit, not removal harm'))
        for j,center in enumerate(conditional_positions(data,meta)):
            changed=perturb(reference,data,center,'interpolate')
            records.append(scalar_record(changed,dense,dict(axis='O',operation='interpolate',candidate=center,span=1),
                meta,center,sampling='tad_conditional',kind='removal_effect'))
            point=dict(layer=LAYERS[(j+meta['window_index'])%4],native_time=center//2,
                       y=(3*j+meta['video_ordinal'])%10,x=(7*j+meta['window_index'])%10,
                       candidate_center=center)
            for axis in ('D','S'):
                am,fm=remove_points(data,axis,[point]);changed=reference.execute(data,am,fm)
                records.append(scalar_record(changed,dense,dict(axis=axis,operation='light_to_heavy',**point),
                    meta,center,sampling='tad_conditional'))
        vals=np.asarray(observation_values['interpolate'])
        for policy,order in [('stratified',list(range(len(candidates)))),('low_single_effect',np.argsort(np.abs(vals),kind='stable').tolist())]:
            for size in (1,2,4,8,16):
                chosen=order[:size];other=dict(data);pixels=data['inputs'].clone()
                for i in chosen:
                    center=candidates[i]
                    single=observation_intervention(data['inputs'],center,center+1,'interpolate')
                    pixels[:,:,:,center:center+1]=single[:,:,:,center:center+1]
                other['inputs']=pixels;changed=reference.execute(other)
                joint=changed['loss']-dense['loss'];summed=float(vals[chosen].sum())
                coalitions.append(dict(axis='O',policy=policy,size=size,actions=chosen,R=joint,
                    sum_single_R=summed,interaction=joint-summed,delta_gflops=changed['gflops']-dense['gflops']))
    return dict(meta=meta,dense_loss_cls_reg=dense['losses'].tolist(),dense_gflops=dense['gflops'],
        predictions=reference.postprocess(dense,data,class_map),records=records,coalitions=coalitions,
        temporal_base_indices=base_indices.cpu().tolist(),light_reference=reference.light_provenance,
        preflight=preflight)


def allocation(reference,data,class_map,meta,axis,preflight=False,calibration=False):
    reference.load_light()
    dense=reference.execute(data,capture_proxies=True)
    mandatory=[]
    if axis=='T':
        groups,extra=support_groups(data)
        mandatory=[0,4,8,12]
        def execute(chosen,profile=False):
            indices=group_indices(data,groups,extra,chosen)
            if len(indices)==768 and torch.equal(indices,torch.arange(768,device=indices.device)):
                return dense
            return execute_temporal(reference,data,indices,force_profile=profile)
        attention=[float(dense['attention'][g].mean()) if len(g) else 0. for g in groups]
    else:
        def execute(chosen,profile=False):
            am,fm=grouped_masks(data,axis,chosen)
            result=reference.execute(data,am,fm,force_profile=profile)
            if profile:
                ledger=reference.shape_cost(dense['gflops'],am,fm)
                if abs(ledger-result['gflops'])>1e-5:
                    raise RuntimeError('Allocation operator count disagrees with its shape ledger')
            return result
        attention=group_attention(dense,data,axis)
    base=execute(mandatory,True)
    candidates=[i for i in range(16) if i not in mandatory]
    measured={}
    for i in candidates:
        result=execute([*mandatory,i])
        measured[i]=dict(value_cls_reg=(base['losses']-result['losses']).tolist(),
            value=base['loss']-result['loss'],delta_gflops=result['gflops']-base['gflops'])
    costs=np.zeros(16);values=np.zeros(16)
    for i,row in measured.items():
        costs[i]=row['delta_gflops'];values[i]=row['value']
    if calibration:
        return dict(meta=meta,axis=axis,marginal_actions=measured,mandatory_groups=mandatory,
                    base_gflops=base['gflops'],dense_gflops=dense['gflops'])
    rng=np.random.default_rng(seed_for(meta,900))
    order_cf=sorted(candidates,key=lambda i:(-(values[i]/costs[i] if costs[i]>1e-8 else -np.inf),i))
    static_file=Path(reference.resources.get('atlas_static',''))
    static_rules=json.loads(static_file.read_text()) if static_file.is_file() else {}
    static_order=static_rules.get(reference.backbone,{}).get(axis,list(range(16)))
    if not preflight and axis not in static_rules.get(reference.backbone,{}):
        raise RuntimeError('Publication allocation requires the frozen development-derived Static order')
    orders=dict(uniform=list(UNIFORM_ORDER),random=rng.permutation(16).tolist(),
                static=static_order,attention=np.argsort(-np.asarray(attention),kind='stable').tolist(),
                marginal_cf=order_cf)
    counts=(8,) if preflight else GROUP_COUNTS
    rows=[]
    for count in counts:
        if axis=='T':
            target_groups=[*mandatory]+[i for i in UNIFORM_ORDER if i not in mandatory][:count-len(mandatory)]
        else:target_groups=list(UNIFORM_ORDER[:count])
        uniform=execute(target_groups,True)
        target_cost=uniform['gflops'];extra_budget=target_cost-base['gflops']
        for policy,order in orders.items():
            if policy=='uniform':
                chosen=target_groups
            elif axis=='T':
                chosen=[*mandatory]+[int(i) for i in order if i not in mandatory][:count-len(mandatory)]
            else:
                chosen=finite_budget_subset(order,costs,extra_budget,count,.005*dense['gflops'],
                    values=values if policy=='marginal_cf' else None,rng=rng if policy=='random' else None)
            result=uniform if policy=='uniform' else execute(chosen,True)
            mismatch=result['gflops']-target_cost
            rows.append(dict(axis=axis,policy=policy,group_budget=count,chosen=chosen,
                actual_gflops=result['gflops'],target_gflops=target_cost,
                cost_mismatch=mismatch,matched=abs(mismatch)<=.005*dense['gflops'],
                loss_cls_reg=np.asarray(result['losses']).tolist(),
                selection_scope='Retrospective dense-attention diagnostic' if policy=='attention' else 'GT-assisted finite reference' if policy=='marginal_cf' else 'deployable fixed rule',
                predictions=reference.postprocess(result,data,class_map)))
    sequential=[]
    # Predeclared representative window subset; never chosen by measured effect.
    if preflight or (meta['video_ordinal']<20 and meta['window_index']%4==0):
        chosen=list(mandatory);current=base
        for step in range(min(8,len(candidates))):
            possibilities=[]
            for i in candidates:
                if i in chosen:continue
                result=execute([*chosen,i])
                delta=result['gflops']-current['gflops']
                value=current['loss']-result['loss']
                possibilities.append((value/delta if delta>1e-8 else -np.inf,i,value,delta,result))
            _,index,value,delta,result=max(possibilities,key=lambda r:(r[0],-r[1]))
            chosen.append(index);current=result
            sequential.append(dict(step=step+1,added=index,chosen=list(chosen),value=value,
                delta_gflops=delta,gflops=result['gflops'],loss_cls_reg=np.asarray(result['losses']).tolist()))
            if preflight:break
    return dict(meta=meta,axis=axis,dense_gflops=dense['gflops'],
        base_loss_cls_reg=np.asarray(base['losses']).tolist(),base_gflops=base['gflops'],
        mandatory_groups=mandatory,marginal_actions=measured,variants=rows,sequential=sequential,
        reference_scope='Finite predeclared groups of real primitive actions; not a mathematical oracle',
        light_reference=reference.light_provenance)


def run_window(reference,data,class_map,args,resources):
    meta=window_metadata(data)
    if args.mode=='population':
        return population(reference,data,class_map,meta)
    if args.mode=='allocation':
        if not args.axis:raise ValueError('Allocation needs one named axis')
        return allocation(reference,data,class_map,meta,args.axis)
    if args.mode=='calibration':
        if args.split!='development' or not args.axis:
            raise ValueError('Static calibration uses only development inputs and one named axis')
        return allocation(reference,data,class_map,meta,args.axis,calibration=True)
    if args.mode=='preflight':
        pop=population(reference,data,class_map,meta,True)
        checks={axis:allocation(reference,data,class_map,meta,axis,True) for axis in ('D','S','T')}
        return dict(meta=meta,population=pop,allocation=checks,passed=True)
    if args.mode=='recovery':
        from .recovery import recovery_window
        return recovery_window(reference,data,class_map,meta,resources)
    raise ValueError(args.mode)
