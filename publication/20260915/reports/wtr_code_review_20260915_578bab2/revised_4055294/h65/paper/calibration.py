"""Video-grouped out-of-fold calibration of real action values."""
import copy
import numpy as np
import torch
from .routing import BudgetRouter,FrameRouter


def tensors(records,frame):
    feature='frame_features' if frame else 'context'
    return (torch.tensor([r[feature] for r in records],dtype=torch.float32),
            torch.tensor([r['actual_delta'] for r in records],dtype=torch.float32),
            None if frame else torch.tensor([r['base_plan'] for r in records]),
            None if frame else torch.tensor([r['action_plan'] for r in records]))


def predict(router,values,frame):
    x,y,base,action=values
    with torch.no_grad():
        if frame:
            raw=router.network(x);mean=raw[:,:2];sigma=raw[:,2:].clamp(-8,8).mul(.5).exp()
        else:
            mean,lv=router.distribution(x);rows=torch.arange(len(x))
            mean=mean[rows,action]-mean[rows,base];sigma=(lv[rows,action].exp()+lv[rows,base].exp()).sqrt()
    return mean*router.scales,sigma*router.scales


def fit(records,frame,seed,steps=600):
    torch.manual_seed(seed);router=FrameRouter(len(records[0]['frame_features'])==300) if frame else BudgetRouter()
    x,y,base,action=tensors(records,frame)
    router.scales.copy_(y.abs().mean(0).clamp_min(1e-4))
    optimizer=torch.optim.AdamW(router.network.parameters(),lr=3e-4,weight_decay=.01)
    generator=torch.Generator().manual_seed(seed)
    for step in range(steps):
        ids=torch.randperm(len(x),generator=generator)[:min(128,len(x))]
        loss=router.regression_loss(x[ids],y[ids]) if frame else router.pair_loss(x[ids],base[ids],action[ids],y[ids])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite real-action calibration fit')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(router.network.parameters(),1.);optimizer.step()
    return router


def calibrate(records,seed=42,folds=4,steps=600):
    videos=sorted({r['video_name'] for r in records});rng=np.random.default_rng(seed);rng.shuffle(videos)
    if len(videos)<folds*2:raise ValueError('Video-grouped calibration requires at least eight videos')
    fold_of={v:i%folds for i,v in enumerate(videos)};oof=[];fitted={};statistics={}
    for frame in (False,True):
        kind='frame' if frame else 'budget';subset=[r for r in records if (r['action_type']=='frame')==frame]
        if not subset:continue
        rows=[]
        for fold in range(folds):
            train=[r for r in subset if fold_of[r['video_name']]!=fold];test=[r for r in subset if fold_of[r['video_name']]==fold]
            if not train or not test:raise ValueError('An OOF router fold has no actions')
            router=fit(train,frame,seed+fold,steps);mu,sigma=predict(router,tensors(test,frame),frame)
            for record,mean,std in zip(test,mu.tolist(),sigma.tolist()):
                rows.append(dict(video_name=record['video_name'],action_type=record['action_type'],fold=fold,
                                 actual_delta=record['actual_delta'],repair_delta=record['repair_delta'],mean=mean,sigma=std))
        actual=np.asarray([r['actual_delta'] for r in rows]);mean=np.asarray([r['mean'] for r in rows]);sigma=np.maximum(1e-8,[r['sigma'] for r in rows])
        ratio=np.abs(actual-mean)/sigma;scale=np.maximum(.1,np.quantile(ratio,.9,axis=0)/1.645)
        final=fit(subset,frame,seed+100,steps);final.sigma_calibration.copy_(torch.tensor(scale,dtype=torch.float32));fitted[kind]=final
        statistics[kind]=dict(actions=len(rows),oof_mae=np.abs(actual-mean).mean(0).tolist(),sigma_multiplier=scale.tolist(),
            coverage90_before=(np.abs(actual-mean)<=1.645*sigma).mean(0).tolist(),
            coverage90_after=(np.abs(actual-mean)<=1.645*sigma*scale).mean(0).tolist(),
            calibration_coverage_scope='OOF residual fit diagnostic; coverage after fitting is not an independent test guarantee')
        oof.extend(rows)
    return fitted,dict(videos=len(videos),folds=folds,video_folds=fold_of,statistics=statistics,oof=oof,
        representation_scope='Fixed final student was trained on the entire training split; only router fitting is video-disjoint OOF',
        test_labels_used=False,fit_steps=steps,seed=seed)
