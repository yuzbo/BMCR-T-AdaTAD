"""Action ranking and drift metrics with videos as the aggregation unit."""
import math
import numpy as np
from scipy.stats import spearmanr


def ranked(scores):
    return np.argsort(-np.asarray(scores, dtype=float), kind='stable')


def ranking_metrics(prediction, target, top_fraction=.2, noise=1e-8):
    prediction=np.asarray(prediction,dtype=float)
    target=np.asarray(target,dtype=float)
    if prediction.shape!=target.shape or prediction.ndim!=1 or not len(target):
        raise ValueError('Ranking requires identical nonempty action vectors')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite action values')
    k=max(1,math.ceil(len(target)*top_fraction))
    order=ranked(prediction);ideal=ranked(target)
    selected=int(order[0]) if prediction[order[0]]>0 else None
    gain=0. if selected is None else float(target[selected])
    positive=np.maximum(target,0.)
    discount=1/np.log2(np.arange(k)+2.)
    idcg=float((positive[ideal[:k]]*discount).sum())
    ndcg=float((positive[order[:k]]*discount).sum()/idcg) if idcg>noise else None
    rho=float(spearmanr(prediction,target).statistic) if np.ptp(prediction)>noise and np.ptp(target)>noise else None
    return dict(regret=max(0.,float(target.max()))-gain,chosen_gain=gain,
        stop_regret=max(0.,float(target.max())),ndcg=ndcg,spearman=rho,
        topk_overlap=len(set(order[:k])&set(ideal[:k]))/k,topk=k,actions=len(target),
        chose_stop=selected is None)


def video_aggregate(rows):
    groups={}
    for row in rows:groups.setdefault(row['video_id'],[]).append(row)
    metrics=('regret','chosen_gain','stop_regret','ndcg','spearman','topk_overlap')
    videos={}
    for video,items in groups.items():
        summary={}
        for key in metrics:
            values=[x[key] for x in items if x.get(key) is not None]
            summary[key]=float(np.mean(values)) if values else None
        summary['states']=len(items);videos[video]=summary
    mean={}
    for key in metrics:
        values=[x[key] for x in videos.values() if x[key] is not None]
        mean[key]=float(np.mean(values)) if values else None
    return dict(video_count=len(videos),state_count=len(rows),video_means=videos,mean=mean,
        informative_videos={key:sum(x[key] is not None for x in videos.values()) for key in metrics})


def paired_video_difference(left,right,metric,bootstrap=10000,seed=42):
    a=left['video_means'];b=right['video_means']
    if set(a)!=set(b):raise ValueError('Paired comparison requires identical video IDs')
    ids=sorted(v for v in a if a[v].get(metric) is not None and b[v].get(metric) is not None)
    if not ids:return dict(videos=0,mean=None,ci95=None)
    delta=np.array([a[v][metric]-b[v][metric] for v in ids])
    rng=np.random.default_rng(seed)
    draws=delta[rng.integers(0,len(delta),size=(bootstrap,len(delta)))].mean(1)
    return dict(videos=len(ids),mean=float(delta.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),
        video_differences=dict(zip(ids,delta.tolist())),bootstrap=bootstrap,
        scope='paired video-cluster interval; not a training-seed confidence interval')


def drift_metrics(earlier,later,noise=1e-8):
    earlier=np.asarray(earlier,dtype=float);later=np.asarray(later,dtype=float)
    result=ranking_metrics(earlier,later,noise=noise)
    clear=(np.abs(earlier)>noise)&(np.abs(later)>noise)
    result.update(sign_flip_rate=float(np.mean(np.sign(earlier[clear])!=np.sign(later[clear]))) if clear.any() else None,
        sign_clear_actions=int(clear.sum()),ambiguous_actions=int((~clear).sum()),noise_epsilon=float(noise),
        absolute_change_mean=float(np.mean(np.abs(later-earlier))))
    return result


def forecast(anchor,post,beta):
    if beta not in (1.,1.05,1.1,1.2):raise ValueError('Unregistered forecast beta')
    return post+(beta-1.)*(post-anchor)
