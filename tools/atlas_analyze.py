#!/usr/bin/env python3
"""Read immutable measurements, recompute full AP, and export plotting data."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
from h65.paper.runtime import json_write


def load_windows(folder,expected=None):
    files=sorted((Path(folder)/'windows').glob('*.json'))
    if expected is not None and len(files)!=expected:
        raise RuntimeError(f'{folder}: expected {expected} immutable windows, found {len(files)}')
    return [json.loads(path.read_text()) for path in files]


def calibrate(args):
    target=Path(args.output)
    current=json.loads(target.read_text()) if target.exists() else {}
    values={};details={}
    for axis in ('T','D','S'):
        rows=load_windows(Path(args.input)/f'calibration_{args.backbone}_{axis}',32)
        scores=defaultdict(list)
        for row in rows:
            for index,action in row['marginal_actions'].items():
                if action['delta_gflops']>1e-8:
                    scores[int(index)].append(action['value']/action['delta_gflops'])
        means={i:float(np.mean(scores[i])) if scores[i] else 0. for i in range(16)}
        values[axis]=sorted(range(16),key=lambda i:(-means[i],i))
        details[axis]=dict(mean_value_per_gflop=means,videos=32,split='development')
    if args.backbone in current and current[args.backbone]!=values:
        raise RuntimeError('Previously frozen Static ordering would change')
    current[args.backbone]=values
    current.setdefault('provenance',{})[args.backbone]=details
    json_write(target,current)
    print(json.dumps(dict(backbone=args.backbone,static_orders=values)),flush=True)


def concentration(value,weight):
    order=np.argsort(value,kind='stable');v=np.maximum(value[order],0);w=weight[order]
    total=float((v*w).sum())
    if total<=0:return None
    x=np.r_[0,np.cumsum(w)/w.sum()];y=np.r_[0,np.cumsum(v*w)/total]
    return dict(gini=float(1-2*np.trapz(y,x)),lorenz=np.interp(np.linspace(0,1,101),x,y).tolist(),
        top_p=[float(1-np.interp(1-p,x,y)) for p in (.05,.1,.2,.4,.6)])


def distribution(groups,replicates):
    ids=sorted(groups);parts=[np.asarray(groups[v]) for v in ids]
    value=np.concatenate(parts);video=np.concatenate([np.full(len(v),i) for i,v in enumerate(parts)])
    counts=np.asarray([len(v) for v in parts]);n=len(ids)
    grid=np.unique(np.r_[0,np.geomspace(1e-8,1,129),.005,.01,.02,.05])
    cdf_by_video=np.stack([(np.abs(v)[:,None]<=grid).mean(0) for v in parts])
    outputs={};rng=np.random.default_rng(42)
    # Same video draws for both weighting conventions and paired subsequent plots.
    draws=np.asarray([np.bincount(rng.integers(0,n,n),minlength=n) for _ in range(replicates)])
    for scheme in ('video_balanced','action_weighted'):
        weight=1/counts[video] if scheme=='video_balanced' else np.ones(len(value))
        central=concentration(value,weight)
        cdf=(cdf_by_video.mean(0) if scheme=='video_balanced' else (cdf_by_video*counts[:,None]).sum(0)/counts.sum())
        boot_cdf=[];boot_gini=[];boot_top=[];boot_lorenz=[]
        for draw in draws:
            vw=draw if scheme=='video_balanced' else draw*counts
            boot_cdf.append((cdf_by_video*vw[:,None]).sum(0)/vw.sum())
            item=concentration(value,weight*draw[video])
            if item is not None:
                boot_gini.append(item['gini']);boot_top.append(item['top_p']);boot_lorenz.append(item['lorenz'])
        outputs[scheme]=dict(cdf=cdf.tolist(),cdf_ci=np.quantile(boot_cdf,[.025,.975],axis=0).tolist(),
            concentration=central,
            gini_ci=np.quantile(boot_gini,[.025,.975]).tolist() if boot_gini else None,
            top_p_ci=np.quantile(boot_top,[.025,.975],axis=0).tolist() if boot_top else None,
            lorenz_ci=np.quantile(boot_lorenz,[.025,.975],axis=0).tolist() if boot_lorenz else None,
            negative_fraction=float(np.average(value<0,weights=weight)),
            zero_fraction=float(np.average(value==0,weights=weight)))
    return dict(videos=n,actions=len(value),cdf_x=grid.tolist(),lorenz_x=np.linspace(0,1,101).tolist(),
        raw_video_values={v:groups[v] for v in ids},**outputs)


def mean_ci(groups,replicates):
    values=np.asarray([np.mean(v,axis=0) for _,v in sorted(groups.items())])
    if not len(values):return dict(mean=None,ci=None,videos=0)
    rng=np.random.default_rng(42)
    draws=np.asarray([values[rng.integers(0,len(values),len(values))].mean(0) for _ in range(replicates)])
    result=dict(mean=np.mean(values,axis=0).tolist(),ci=np.quantile(draws,[.025,.975],axis=0).tolist(),videos=len(values))
    if values.ndim==2 and values.shape[1]==2:
        result.update(total_mean=float(values.mean(0).sum()),total_ci=np.quantile(draws.sum(1),[.025,.975]).tolist())
    return result


def population_summary(args,resources):
    from scipy.stats import spearmanr
    output={}
    train=json.loads(Path(resources['datasets']['thumos']['annotations']).read_text())['database']
    durations=[a['segment'][1]-a['segment'][0] for v in train.values() if v['subset']=='training' for a in v['annotations']]
    duration_edges=np.quantile(durations,[1/3,2/3]).tolist()
    for backbone in ((args.backbone,) if args.backbone else ('s','b')):
        rows=load_windows(Path(args.input)/f'population_{backbone}',792)
        if len({r['meta']['video_id'] for r in rows})!=211:raise RuntimeError('Population must contain all 211 videos')
        population_cache=Path(args.output)/'population_models'/f'{backbone}.json'
        contract=dict(backbone=backbone,bootstrap=args.bootstrap,windows=len(rows),
            videos=sorted({r['meta']['video_id'] for r in rows}),duration_edges_seconds=duration_edges,
            source_revisions=sorted({r.get('source_revision','legacy') for r in rows}),
            heavy_checkpoint=resources['teachers'][f'thumos:{backbone}'],light_reference=rows[0]['light_reference'])
        if population_cache.exists():
            cached=json.loads(population_cache.read_text())
            if cached.get('analysis_contract')!=contract:
                raise RuntimeError(f'Population cache belongs to a different frozen input: {population_cache}')
            output[backbone]=cached['summary']
            continue
        print(f'Computing full population statistics: {backbone}, bootstrap={args.bootstrap}',flush=True)
        groups=defaultdict(lambda:defaultdict(list));regions=defaultdict(lambda:defaultdict(list))
        boundary=defaultdict(lambda:defaultdict(list));duration=defaultdict(lambda:defaultdict(list))
        coalition=defaultdict(lambda:defaultdict(list));proxy=defaultdict(lambda:defaultdict(list))
        cases=defaultdict(list)
        for row in rows:
            video=row['meta']['video_id'];denom=max(sum(row['dense_loss_cls_reg']),1e-8)
            local=defaultdict(list)
            for action in row['records']:
                definition=action['action'];axis=definition['axis']
                key=f'O_{definition["operation"]}' if axis=='O' else axis
                value=action['value']/denom
                if action['sampling'] in ('population','control'):
                    groups[key][video].append(value)
                    if axis in ('T','D','S') or key=='O_interpolate':local[key].append(action)
                position=action['position'];region=position['region']
                if action['sampling']=='tad_conditional' and axis in ('D','S','O'):
                    if region in ('start','interior','end','background'):
                        regions[f'{axis}:{region}'][video].append(np.asarray(action['value_cls_reg'])/denom)
                    distance=position.get('boundary_distance')
                    if distance is not None and 0<=distance<=.5 and position.get('fully_contained'):
                        bin_id=min(9,int(distance/.05));boundary[f'{axis}:{bin_id}'][video].append(np.asarray(action['value_cls_reg'])/denom)
                    d=position.get('action_duration')
                    if d is not None and position.get('fully_contained'):
                        bin_id=int(np.searchsorted(duration_edges,d));duration[f'{axis}:{bin_id}'][video].append(np.asarray(action['value_cls_reg'])/denom)
            for key,items in local.items():
                truth=np.asarray([a['value'] for a in items]);positive=np.maximum(truth,0)
                for name in ('attention_score','actionness','entropy','feature_norm'):
                    if not all(name in a.get('proxies',{}) for a in items):continue
                    pred=np.asarray([a['proxies'][name] for a in items])
                    if len(items)>2 and np.ptp(truth)>0 and np.ptp(pred)>0:
                        proxy[f'{key}:{name}:spearman'][video].append(float(spearmanr(pred,truth).statistic))
                    k=max(1,int(np.ceil(.2*len(items))));discount=1/np.log2(np.arange(2,k+2))
                    ideal=float((np.sort(positive)[::-1][:k]*discount).sum())
                    if ideal>0:
                        ndcg=float((positive[np.argsort(-pred,kind='stable')[:k]]*discount).sum()/ideal)
                        proxy[f'{key}:{name}:ndcg'][video].append(ndcg)
            for item in row['coalitions']:
                coalition[f'{item["axis"]}:{item["policy"]}:{item["size"]}'][video].append(
                    np.asarray([item['R'],item['sum_single_R'],item['interaction']])/denom)
            cases[video].append(dict(window_index=row['meta']['window_index'],source=row['meta']))
        dist={key:distribution(value,args.bootstrap) for key,value in groups.items()}
        concentrations=[]
        for video,values in groups['D'].items():
            c=concentration(np.asarray(values),np.ones(len(values)))
            if c:concentrations.append((video,c['gini']))
        chosen=[]
        if concentrations:
            scores=np.asarray([r[1] for r in concentrations])
            for q in (.1,.5,.9):
                target=float(np.quantile(scores,q))
                candidates=sorted(concentrations,key=lambda r:(abs(r[1]-target),r[0]))
                picked=next((r for r in candidates if r[0] not in [v['video_id'] for v in chosen]),candidates[0])
                windows=sorted(cases[picked[0]],key=lambda r:r['window_index'])
                chosen.append(dict(quantile=q,video_id=picked[0],gini=picked[1],window_index=windows[len(windows)//2]['window_index']))
        output[backbone]=dict(distributions=dist,
            regions={k:mean_ci(v,args.bootstrap) for k,v in regions.items()},
            boundary={k:mean_ci(v,args.bootstrap) for k,v in boundary.items()},
            duration={k:mean_ci(v,args.bootstrap) for k,v in duration.items()},
            coalitions={k:mean_ci(v,args.bootstrap) for k,v in coalition.items()},
            proxies={k:mean_ci(v,args.bootstrap) for k,v in proxy.items()},cases=chosen,
            provenance=dict(source_revisions=sorted({r.get('source_revision','legacy') for r in rows}),
                heavy_checkpoint=resources['teachers'][f'thumos:{backbone}'],
                light_reference=rows[0]['light_reference'],split='publication',videos=211,windows=792))
        json_write(population_cache,dict(analysis_contract=contract,summary=output[backbone]))
        print(f'Full population statistics cached: {backbone}',flush=True)
    # A single-backbone precompute never creates the whole-atlas completion file.
    if args.backbone:return
    output['duration_edges_seconds']=duration_edges
    output['protocol']=dict(videos=211,windows=792,bootstrap=args.bootstrap,unit='video cluster',
        normalization='Signed total or cls/reg effect divided by the same window official dense total loss; raw data retained')
    json_write(Path(args.output)/'population.json',output)


def evaluate_variants(rows,resources,output,kind,replicates,backbone):
    from h65.paper.evaluation import merge_windows
    from h65.paper.native_adatad import build_native_config
    from h65.atlas.statistics import ap_cache,cluster_ap
    variants=defaultdict(lambda:defaultdict(list));costs=defaultdict(list);mismatch=defaultdict(list)
    ids=sorted(resources['datasets']['thumos']['test_ids'])
    for row in rows:
        for item in row['variants']:
            key=(f'{item["policy"]}:{item["group_budget"]}' if kind=='allocation' else item['method'])
            for video,pred in item['predictions'].items():variants[key][video].extend(pred)
            costs[key].append(item['actual_gflops'] if kind=='allocation' else item['gflops'])
            if kind=='allocation':mismatch[key].append(item['cost_mismatch'])
    out={}
    native=build_native_config(dict(backbone=backbone,frames=768),resources)
    post=copy_config(native.post_processing);post.sliding_window=True
    for key,pred in sorted(variants.items()):
        target=output/(key.replace(':','_')+'.json')
        contract=dict(backbone=backbone,kind=kind,variant=key,bootstrap=replicates,
            source_revisions=sorted({row.get('source_revision','legacy') for row in rows}),
            videos=ids,windows=len(rows))
        if target.exists():
            cached=json.loads(target.read_text())
            if cached.get('analysis_contract')!=contract:
                raise RuntimeError(f'AP cache belongs to a different frozen input or bootstrap setting: {target}')
            out[key]=cached;continue
        if len(costs[key])!=792:raise RuntimeError(f'Missing allocation predictions: {key}')
        for video in ids:pred.setdefault(video,[])
        pred=merge_windows(pred,post)
        caches,official=ap_cache(dict(pred),resources['datasets']['thumos']['annotations'],'validation',ids)
        statistics=cluster_ap(caches,ids,replicates)
        record=dict(**statistics,official=official,mean_gflops=float(np.mean(costs[key])),windows=792,
            analysis_contract=contract,
            max_cost_mismatch=max(map(abs,mismatch[key])) if mismatch[key] else 0.)
        json_write(target,record);out[key]=record
        print(f'AP and video bootstrap complete: {output.name}/{key}',flush=True)
    return out


def copy_config(config):
    import copy
    return copy.deepcopy(config)


def performance_summary(args,resources,kind):
    output={}
    for backbone in ('s','b'):
        output[backbone]={}
        axes=('T','D','S') if kind=='allocation' else ('recovery',)
        for axis in axes:
            name=f'allocation_{backbone}_{axis}' if kind=='allocation' else f'recovery_{backbone}'
            rows=load_windows(Path(args.input)/name,792)
            if len({r['meta']['video_id'] for r in rows})!=211:raise RuntimeError('AP requires all videos')
            folder=Path(args.output)/'ap'/name;folder.mkdir(parents=True,exist_ok=True)
            scores=evaluate_variants(rows,resources,folder,kind,args.bootstrap,backbone)
            if kind=='allocation':
                paired={}
                for count in (4,6,8,10,12,16):
                    a=scores[f'marginal_cf:{count}'];b=scores[f'uniform:{count}']
                    delta=np.asarray(a['bootstrap_average'])-np.asarray(b['bootstrap_average'])
                    paired[str(count)]=dict(delta_map=a['average_mAP']-b['average_mAP'],
                        ci=np.quantile(delta,[.025,.975]).tolist(),cost_difference=a['mean_gflops']-b['mean_gflops'])
                sequential=[dict(meta=row['meta'],marginal=row['marginal_actions'],steps=row['sequential']) for row in rows if row['sequential']]
                output[backbone][axis]=dict(scores=scores,paired_headroom=paired,sequential=sequential,
                    provenance=dict(source_revisions=sorted({r.get('source_revision','legacy') for r in rows}),
                        heavy_checkpoint=resources['teachers'][f'thumos:{backbone}'],light_reference=rows[0]['light_reference'],
                        scope='Frozen finite-group allocation; privileged selection costs are separate'))
            else:
                gap_edges=np.array([0,.05,.1,.2,.4,.8,1.6,np.inf])
                gap=defaultdict(lambda:defaultdict(list));end=defaultdict(lambda:defaultdict(list))
                for row in rows:
                    video=row['meta']['video_id'];g=np.asarray(row['query_gap_seconds'])
                    for item in row['variants']:
                        name=item['method'];e=item['endpoint']
                        end[name][video].append([*e['sum_abs_endpoint_error_seconds'],e['matched'],e['gt']])
                        for i in range(len(gap_edges)-1):
                            keep=(g>=gap_edges[i])&(g<gap_edges[i+1])
                            if keep.any():gap[f'{name}:{i}'][video].extend(np.asarray(item['nmse'])[keep].tolist())
                endpoint_values={}
                for method,video_records in end.items():
                    errors={};missing={}
                    for video,items in video_records.items():
                        start,finish,matched,total=np.asarray(items).sum(0)
                        if matched:errors[video]=[(start+finish)/(2*matched)]
                        if total:missing[video]=[100*(1-matched/total)]
                    endpoint_values[method]=dict(error=mean_ci(errors,args.bootstrap),missing_percent=mean_ci(missing,args.bootstrap))
                output[backbone]=dict(scores=scores,gap_edges=gap_edges[:-1].tolist(),
                    gap={k:mean_ci(v,args.bootstrap) for k,v in gap.items()},
                    endpoint=endpoint_values,provenance=dict(checkpoint=rows[0]['checkpoint'],
                        source_revisions=sorted({r.get('source_revision','legacy') for r in rows}),
                        scope='Same V2 checkpoint and shared heavy support; component ablation'))
    json_write(Path(args.output)/f'{kind}.json',output)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['calibrate','population','allocation','recovery'],required=True)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resources',required=True);p.add_argument('--backbone',choices=['s','b'])
    p.add_argument('--bootstrap',type=int,default=10000)
    args=p.parse_args()
    if args.mode=='calibrate':return calibrate(args)
    Path(args.output).mkdir(parents=True,exist_ok=True)
    resources=json.loads(Path(args.resources).read_text())
    if args.mode=='population':return population_summary(args,resources)
    return performance_summary(args,resources,args.mode)


if __name__=='__main__':main()
