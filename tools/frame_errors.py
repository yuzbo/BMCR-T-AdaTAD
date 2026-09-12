"""DETAD FP taxonomy and paired video-cluster bootstrap on saved full tests.

Greedy matches and score-tie order follow OpenTAD. Bootstrap weights whole video
clusters with that original tie order fixed. It does not measure training-seed
or checkpoint-selection uncertainty, or implement DETAD's error-removal oracles.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
THRESHOLDS=np.array([.3,.4,.5,.6,.7])
TYPES=['true_positive','duplicate','wrong_label','localization','confusion','background']

def overlaps(pred,gt):
    inter=np.maximum(0,np.minimum(pred[1],gt[:,1])-np.maximum(pred[0],gt[:,0]))
    return inter/np.maximum(pred[1]-pred[0]+gt[:,1]-gt[:,0]-inter,1e-12)

def weighted_ap(tp,video_indices,gt_per_video,weights):
    npos=float(gt_per_video@weights)
    if not npos:return np.full(5,np.nan)
    w=weights[video_indices];cum_tp=np.cumsum(tp*w[None],1);cum_fp=np.cumsum((1-tp)*w[None],1)
    recall=cum_tp/npos;precision=cum_tp/np.maximum(cum_tp+cum_fp,1e-12)
    envelope=np.maximum.accumulate(precision[:,::-1],axis=1)[:,::-1]
    return (np.diff(np.pad(recall,((0,0),(1,0))),axis=1)*envelope).sum(1)

def prepare(predictions,ground_truth,ids):
    from opentad.evaluations.mAP import mAP
    evaluator=mAP(ground_truth_filename=str(ground_truth),prediction_filename=str(predictions),subset='validation',tiou_thresholds=THRESHOLDS,thread=1)
    gt=evaluator.ground_truth;pred=evaluator.prediction;vid_index={name:i for i,name in enumerate(ids)}
    all_gt={vid:(part[['t-start','t-end']].values,part['label'].values) for vid,part in gt.groupby('video-id')}
    caches=[];counts=np.zeros((5,6),dtype=np.int64);top_counts=np.zeros_like(counts);misses=np.zeros(5,dtype=np.int64)
    for label in range(len(evaluator.activity_index)):
        g=gt[gt.label==label].reset_index(drop=True);p=pred[pred.label==label].reset_index(drop=True)
        p=p.loc[p.score.values.argsort()[::-1]].reset_index(drop=True)
        groups={vid:(part.index.values,part[['t-start','t-end']].values) for vid,part in g.groupby('video-id')}
        locked=np.zeros((5,len(g)),dtype=bool);tp=np.zeros((5,len(p)),dtype=np.uint8);types=np.full((5,len(p)),5,dtype=np.uint8)
        videos=p['video-id'].values;segments=p[['t-start','t-end']].values
        for i,(video,segment) in enumerate(zip(videos,segments)):
            if video in groups:
                index,targets=groups[video];ious=overlaps(segment,targets);order=ious.argsort()[::-1]
                for threshold,limit in enumerate(THRESHOLDS):
                    for j in order:
                        if ious[j]<limit:break
                        if not locked[threshold,index[j]]:locked[threshold,index[j]]=True;tp[threshold,i]=1;break
            if video in all_gt:
                targets,labels=all_gt[video];ious=overlaps(segment,targets);j=int(ious.argmax());maximum=ious[j];same=labels[j]==label
                for threshold,limit in enumerate(THRESHOLDS):
                    if tp[threshold,i]:types[threshold,i]=0
                    elif maximum>=limit:types[threshold,i]=1 if same else 2
                    elif maximum>=.1:types[threshold,i]=3 if same else 4
        for threshold in range(5):
            counts[threshold]+=np.bincount(types[threshold],minlength=6)
            top_counts[threshold]+=np.bincount(types[threshold,:10*len(g)],minlength=6)
        misses+=(~locked).sum(1)
        gcount=np.bincount([vid_index[v] for v in g['video-id']],minlength=len(ids))
        caches.append((tp,np.array([vid_index[v] for v in videos],dtype=int),gcount))
    return caches,dict(thresholds=THRESHOLDS.tolist(),types=TYPES,all_saved_prediction_counts=counts.tolist(),
        top_10x_gt_per_class_counts=top_counts.tolist(),missed_gt=misses.tolist(),ground_truth_instances=len(gt),
        prediction_instances=len(pred),taxonomy_source='https://github.com/HumamAlwassel/DETAD/blob/master/src/action_detector_diagnosis.py',
        scope='FP taxonomy + misses; no additive error-oracle claims')

def score(caches,weights):return np.nanmean([weighted_ap(*c,weights) for c in caches],axis=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--reference-predictions',required=True)
    p.add_argument('--ground-truth');p.add_argument('--output',required=True);p.add_argument('--replicates',type=int,default=200);p.add_argument('--seed',type=int,default=3407);a=p.parse_args()
    from h65.frame.runtime import data_config,json_write
    ground=Path(a.ground_truth or data_config('s').evaluation.ground_truth_filename)
    database=json.loads(ground.read_text())['database'];ids=sorted(k for k,v in database.items() if v['subset']=='validation')
    if len(ids)!=211:raise ValueError('Expected complete211-video THUMOS14 protocol')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    current,taxonomy=prepare(a.predictions,ground,ids);reference,reference_taxonomy=prepare(a.reference_predictions,ground,ids)
    weights=np.ones(len(ids),dtype=int);actual=score(current,weights);baseline=score(reference,weights)
    for path,measured in [(a.predictions,actual),(a.reference_predictions,baseline)]:
        stored=json.loads((Path(path).parent/'metrics.json').read_text())['metrics'];expected=np.array([stored[f'mAP@{t:.1f}'] for t in THRESHOLDS])
        if not np.allclose(measured,expected,atol=1e-8,rtol=0):raise RuntimeError(dict(error='Cached matching differs from saved official full-test AP',path=path,measured=measured.tolist(),expected=expected.tolist()))
    rng=np.random.default_rng(a.seed);differences=[];missing=[]
    for _ in range(a.replicates):
        weights=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids))
        differences.append(score(current,weights)-score(reference,weights));missing.append(sum((c[2]@weights)==0 for c in current))
    deltas=np.array(differences);mean=deltas.mean(1);np.savez_compressed(out/'bootstrap.npz',threshold_deltas=deltas,mean_deltas=mean)
    result=dict(predictions=a.predictions,reference_predictions=a.reference_predictions,paired_videos=211,replicates=a.replicates,seed=a.seed,
        official_AP_reproduced=True,average_mAP=float(actual.mean()),reference_average_mAP=float(baseline.mean()),delta_pp=float((actual-baseline).mean()*100),
        percentile_95_ci_pp=(np.quantile(mean,[.025,.975])*100).tolist(),threshold_95_ci_pp=(np.quantile(deltas,[.025,.975],axis=0)*100).tolist(),
        resamples_with_absent_classes=int(np.count_nonzero(missing)),absent_class_rule='omit undefined classes jointly in both paired models',
        interpretation='conditional on selected checkpoints; video clusters iid; original score-tie order fixed; not training-seed uncertainty',taxonomy=taxonomy,reference_taxonomy=reference_taxonomy)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].hist(mean*100,bins=24,color='#297d8e');axes[0].axvline(0,color='black',lw=.7)
    axes[0].set(xlabel='Paired average mAP difference (pp)',ylabel='Bootstrap replicates')
    values=np.array(taxonomy['top_10x_gt_per_class_counts'])[2,1:];axes[1].bar(TYPES[1:],values,color='#8b7095');axes[1].tick_params(axis='x',rotation=30)
    axes[1].set(ylabel='False positives',title='tIoU .5; top10×GT per class')
    for ext in ('png','svg','pdf'):fig.savefig(out/f'error_bootstrap.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig);json_write(out/'completed.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('taxonomy','reference_taxonomy')}))

if __name__=='__main__':main()
