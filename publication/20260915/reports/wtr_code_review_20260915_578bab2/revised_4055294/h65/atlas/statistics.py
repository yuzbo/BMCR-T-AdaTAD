"""Video-cluster statistics and official-order AP caches for empirical figures."""
from collections import defaultdict
import itertools
import numpy as np
from scipy.stats import spearmanr,kruskal,mannwhitneyu

THRESHOLDS=np.array([.3,.4,.5,.6,.7])


def finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def interval(values):
    array=np.asarray(values,dtype=float)
    array=array[np.isfinite(array)]
    return np.quantile(array,[.025,.975]).tolist() if len(array) else [None,None]


def weighted_concentration(values,weights,grid=None):
    grid=np.linspace(0,1,101) if grid is None else grid
    positive=np.maximum(np.asarray(values,dtype=float),0)
    order=np.argsort(positive,kind='stable');v=positive[order];w=np.asarray(weights)[order]
    total=float(np.sum(v*w))
    if total<=0:return dict(gini=None,lorenz=None,top_p=None)
    x=np.r_[0,np.cumsum(w)/np.sum(w)];y=np.r_[0,np.cumsum(v*w)/total]
    lorenz=np.interp(grid,x,y)
    return dict(gini=float(1-2*np.trapz(y,x)),lorenz=lorenz.tolist(),
                top_p=[float(1-np.interp(1-p,x,y)) for p in (.05,.1,.2,.4)])


def distribution(video_values,replicates=1000):
    data={k:np.asarray(v,dtype=float) for k,v in video_values.items() if len(v)}
    data={k:v[np.isfinite(v)] for k,v in data.items()}
    data={k:v for k,v in data.items() if len(v)}
    if not data:raise RuntimeError('No measured observations for the requested distribution')
    ids=sorted(data);parts=[data[k] for k in ids]
    all_values=np.concatenate(parts)
    x=np.unique(np.r_[0,np.geomspace(1e-7,max(.05,float(np.abs(all_values).max())),128),.005,.01,.02,.05])
    per_video=np.stack([(np.abs(v)[:,None]<=x).mean(0) for v in parts])
    weights=np.concatenate([np.full(len(v),1/(len(v)*len(parts))) for v in parts])
    base=weighted_concentration(all_values,weights)
    rng=np.random.default_rng(42);cdfs=[];ginis=[];tops=[];lorenz=[]
    for _ in range(replicates):
        draw=rng.integers(0,len(parts),len(parts));cdfs.append(per_video[draw].mean(0))
        values=np.concatenate([parts[i] for i in draw])
        w=np.concatenate([np.full(len(parts[i]),1/(len(parts[i])*len(parts))) for i in draw])
        summary=weighted_concentration(values,w)
        ginis.append(np.nan if summary['gini'] is None else summary['gini'])
        if summary['lorenz'] is not None:lorenz.append(summary['lorenz']);tops.append(summary['top_p'])
    return dict(videos=len(ids),actions=len(all_values),video_ids=ids,cdf_x=x.tolist(),cdf=per_video.mean(0).tolist(),
        cdf_ci=np.quantile(cdfs,[.025,.975],axis=0).tolist(),lorenz_x=np.linspace(0,1,101).tolist(),
        **base,gini_ci=interval(ginis),
        lorenz_ci=np.quantile(lorenz,[.025,.975],axis=0).tolist() if lorenz else None,
        top_p_ci=np.quantile(tops,[.025,.975],axis=0).tolist() if tops else None,
        undefined_concentration_bootstraps=int(np.sum(~np.isfinite(ginis))),
        raw_video_values={k:v.tolist() for k,v in data.items()},unit='video-balanced normalized paired loss change')


def holm(p_values):
    output=[None]*len(p_values)
    valid=[i for i,p in enumerate(p_values) if p is not None]
    order=sorted(valid,key=lambda i:p_values[i]);previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,p_values[index]*(len(order)-rank)));output[index]=previous
    return output


def region_statistics(rows,replicates=1000):
    names=['start','interior','end','background'];grouped=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row['region']!='background' and not row.get('fully_contained'):continue
        grouped[row['video_id']][row['region']].append(row['normalized_R'])
    ids=sorted(grouped)
    array=np.array([[np.mean(grouped[v][k]) if grouped[v][k] else np.nan for k in names] for v in ids])
    rng=np.random.default_rng(42)
    def kw(value):
        groups=[x[np.isfinite(x)] for x in value.T];groups=[x for x in groups if len(x)]
        if len(groups)<2:return np.nan
        if np.ptp(np.concatenate(groups))==0:return 0.
        return float(kruskal(*groups).statistic)
    statistic=kw(array);permuted=[]
    for _ in range(replicates):
        copy=array.copy()
        for i in range(len(copy)):
            valid=np.isfinite(copy[i]);copy[i,valid]=rng.permutation(copy[i,valid])
        permuted.append(kw(copy))
    kw_p=None if not np.isfinite(statistic) else float((1+np.sum(np.asarray(permuted)>=statistic))/(1+replicates))
    pairwise=[]
    for a,b in itertools.combinations(range(4),2):
        keep=np.isfinite(array[:,a])&np.isfinite(array[:,b]);x=array[keep,a];y=array[keep,b];n=len(x)
        if n<3:
            pairwise.append(dict(a=names[a],b=names[b],videos=n,p=None));continue
        observed=float(mannwhitneyu(x,y).statistic);center=n*n/2;null=[];deltas=[]
        for _ in range(replicates):
            swap=rng.integers(0,2,n).astype(bool)
            u=float(mannwhitneyu(np.where(swap,y,x),np.where(swap,x,y)).statistic)
            null.append(abs(u-center));sample=rng.integers(0,n,n);deltas.append(np.median(x[sample]-y[sample]))
        pairwise.append(dict(a=names[a],b=names[b],videos=n,U=observed,
            p=float((1+np.sum(np.asarray(null)>=abs(observed-center)))/(1+replicates)),
            cliffs_delta=float(2*observed/(n*n)-1),paired_median_difference=float(np.median(x-y)),
            paired_median_ci=interval(deltas)))
    adjusted=holm([p['p'] for p in pairwise])
    for row,p in zip(pairwise,adjusted):row['holm_p']=p
    return dict(regions=names,video_ids=ids,video_means=[[finite(x) for x in row] for row in array],
        KW=finite(statistic),KW_video_block_permutation_p=kw_p,pairwise=pairwise,
        inference='KW/MW statistics with within-video label permutations; no window-iid asymptotic p values')


def boundary_curve(rows,replicates=1000):
    edges=np.linspace(0,.5,11);grouped=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row.get('boundary_distance') is None or not row.get('fully_contained'):continue
        bin_id=min(9,int(np.searchsorted(edges,row['boundary_distance'],side='right')-1))
        if bin_id>=0:grouped[row['video_id']][bin_id].append(row['normalized_R'])
    ids=sorted(grouped)
    matrix=np.asarray([[np.mean(grouped[v][k]) if grouped[v][k] else np.nan for k in range(10)] for v in ids])
    if not len(matrix):return dict(x=((edges[:-1]+edges[1:])/2).tolist(),mean=[None]*10,ci=[[None]*10]*2,video_counts=[0]*10)
    rng=np.random.default_rng(42)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        mean=np.nanmean(matrix,axis=0)
        draws=np.asarray([np.nanmean(matrix[rng.integers(0,len(matrix),len(matrix))],axis=0) for _ in range(replicates)])
    cis=np.asarray([interval(draws[:,i]) for i in range(10)],dtype=object).T.tolist()
    return dict(x=((edges[:-1]+edges[1:])/2).tolist(),mean=[finite(x) for x in mean],ci=cis,
                video_counts=np.isfinite(matrix).sum(0).tolist())


def proxy_correlations(rows,replicates=1000):
    names=('actionness','confidence','entropy','feature_norm','attention_score');rng=np.random.default_rng(42);out={}
    for name in names:
        records=[r for r in rows if name in r.get('proxies',{}) and 'uniform' in r.get('sampling',[])]
        ids=sorted({r['video_id'] for r in records})
        groups=[[r for r in records if r['video_id']==v] for v in ids]
        def rho(items):
            x=np.array([r['proxies'][name] for r in items]);y=np.array([r['normalized_R'] for r in items])
            if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return np.nan
            return float(spearmanr(x,y).statistic)
        values=[]
        if groups:
            for _ in range(replicates):
                draw=rng.integers(0,len(groups),len(groups));values.append(rho([r for i in draw for r in groups[i]]))
        out[name]=dict(rho=finite(rho(records)),ci=interval(values),videos=len(ids),
                      x=[r['proxies'][name] for r in records],y=[r['normalized_R'] for r in records])
    return out


def ap_cache(predictions,ground_truth,subset,ids):
    from opentad.evaluations.mAP import mAP,segment_iou
    from tools.frame_errors import weighted_ap
    evaluator=mAP(ground_truth_filename=str(ground_truth),prediction_filename=dict(results=predictions),
                  subset=subset,tiou_thresholds=THRESHOLDS,thread=1)
    gt=evaluator.ground_truth;pred=evaluator.prediction;lookup={v:i for i,v in enumerate(ids)};caches=[]
    for label in range(len(evaluator.activity_index)):
        g=gt[gt.label==label].reset_index(drop=True);p=pred[pred.label==label].reset_index(drop=True)
        p=p.loc[p.score.values.argsort()[::-1]].reset_index(drop=True)
        groups={v:(x.index.values,x[['t-start','t-end']].values) for v,x in g.groupby('video-id')}
        locked=np.zeros((5,len(g)),dtype=bool);tp=np.zeros((5,len(p)),dtype=np.uint8)
        for i,(video,segment) in enumerate(zip(p['video-id'].values,p[['t-start','t-end']].values)):
            if video not in groups:continue
            indices,targets=groups[video];ious=segment_iou(segment,targets);order=ious.argsort()[::-1]
            for t,threshold in enumerate(THRESHOLDS):
                for j in order:
                    if ious[j]<threshold:break
                    if not locked[t,indices[j]]:locked[t,indices[j]]=True;tp[t,i]=1;break
        caches.append((tp,np.asarray([lookup[v] for v in p['video-id']],dtype=int),
                       np.bincount([lookup[v] for v in g['video-id']],minlength=len(ids))))
    score=np.nanmean([weighted_ap(*x,np.ones(len(ids),dtype=int)) for x in caches],axis=0)
    official=evaluator.evaluate()
    expected=np.array([official[f'mAP@{t:.1f}'] for t in THRESHOLDS])
    if not np.allclose(score,expected,atol=1e-8,rtol=0):
        raise RuntimeError('Video-bootstrap cache does not reproduce official AP and tie ordering')
    return caches,official


def cluster_ap(caches,ids,replicates=1000):
    from tools.frame_errors import score
    rng=np.random.default_rng(42);n=len(ids)
    actual=score(caches,np.ones(n,dtype=int));draws=[];per_video=[]
    for _ in range(replicates):
        weights=np.bincount(rng.integers(0,n,n),minlength=n);draws.append(score(caches,weights))
    for i in range(n):
        weights=np.zeros(n,dtype=int);weights[i]=1;per_video.append(score(caches,weights))
    draws=np.asarray(draws)
    return dict(average_mAP=float(actual.mean()*100),AP07=float(actual[-1]*100),
        threshold_mAP=(actual*100).tolist(),average_ci=interval(draws.mean(1)*100),AP07_ci=interval(draws[:,-1]*100),
        bootstrap_average=(draws.mean(1)*100).tolist(),bootstrap_AP07=(draws[:,-1]*100).tolist(),
        per_video_average=[finite(np.nanmean(v)*100) for v in per_video],video_ids=ids,
        official_AP_reproduced=True,video_bootstrap=True,
        per_video_scope='AP over classes with GT in that video; not additive decomposition of dataset mAP')


def paired_rank_summary(x,y,replicates=1000):
    values=[]
    for a,b in zip(x,y):
        if np.ptp(a)==0 or np.ptp(b)==0:values.append(np.nan)
        else:values.append(float(spearmanr(a,b).statistic))
    array=np.asarray(values);good=array[np.isfinite(array)]
    rng=np.random.default_rng(42)
    draws=[float(np.mean(good[rng.integers(0,len(good),len(good))])) for _ in range(replicates)] if len(good) else []
    return dict(value=float(good.mean()) if len(good) else None,ci=interval(draws),videos=len(good),undefined_videos=len(array)-len(good))


def ndcg_summary(predicted,truth,replicates=1000):
    values=[]
    for a,b in zip(predicted,truth):
        relevance=np.maximum(np.asarray(b),0);k=max(1,int(np.ceil(.2*len(b))))
        discount=1/np.log2(np.arange(2,k+2))
        ideal=float((np.sort(relevance)[::-1][:k]*discount).sum())
        if ideal<=0:values.append(np.nan);continue
        ranking=np.argsort(-np.asarray(a),kind='stable')[:k]
        values.append(float((relevance[ranking]*discount).sum()/ideal))
    good=np.asarray(values);good=good[np.isfinite(good)];rng=np.random.default_rng(42)
    draws=[float(np.mean(good[rng.integers(0,len(good),len(good))])) for _ in range(replicates)] if len(good) else []
    return dict(value=float(good.mean()) if len(good) else None,ci=interval(draws),videos=len(good),undefined_videos=len(values)-len(good),top_fraction=.2)
