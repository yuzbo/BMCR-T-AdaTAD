"""Verify compiled AP against existing 10000-draw full-data NumPy results."""
import copy
import json
import os
from pathlib import Path
import sys
import time
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
os.environ['CUDA_VISIBLE_DEVICES']=''
from h65.paper.runtime import json_write
from h65.atlas.fast_bootstrap import score_many,video_weights
from tools.frame_errors import score


def main():
    import torch
    from h65.paper.native_adatad import build_native_config
    from h65.paper.evaluation import merge_windows
    from h65.atlas.statistics import ap_cache
    torch.set_num_threads(4)
    folder=ROOT/'analysis/bootstrap_validation';folder.mkdir(parents=True,exist_ok=True)
    resources=json.loads((ROOT/'resources.json').read_text())
    ids=sorted(resources['datasets']['thumos']['test_ids'])
    artifact=folder/'real_cache.npz'
    if artifact.exists():
        packed=np.load(artifact)
        caches=[(packed[f'tp_{i}'],packed[f'video_{i}'],packed[f'gt_{i}']) for i in range(int(packed['classes']))]
    else:
        files=sorted((ROOT/'results/allocation_s_T/windows').glob('*.json'))
        if len(files)!=792:raise RuntimeError('Validation must use the completed 792-window collection')
        predictions={v:[] for v in ids}
        for index,path in enumerate(files):
            row=json.loads(path.read_text())
            chosen=next(v for v in row['variants'] if v['policy']=='attention' and v['group_budget']==10)
            for video,items in chosen['predictions'].items():predictions[video].extend(items)
            if index%100==0:print(f'Reading real AP validation input: {index+1}/792',flush=True)
        cfg=build_native_config(dict(backbone='s',frames=768),resources)
        post=copy.deepcopy(cfg.post_processing);post.sliding_window=True
        predictions=merge_windows(predictions,post)
        caches,official=ap_cache(predictions,resources['datasets']['thumos']['annotations'],'validation',ids)
        values={'classes':np.array(len(caches))}
        for i,(tp,video,gt) in enumerate(caches):values.update({f'tp_{i}':tp,f'video_{i}':video,f'gt_{i}':gt})
        temporary=folder/'real_cache.tmp.npz'
        np.savez_compressed(temporary,**values);temporary.replace(artifact)
        json_write(folder/'official.json',official)
    n=len(ids);weights=video_weights(n,10000)
    # Actual point estimate, all draws and every per-video diagnostic retain order.
    all_weights=np.concatenate((np.ones((1,n),dtype=np.int64),weights,np.eye(n,dtype=np.int64)))
    start=time.perf_counter();accelerated=score_many(caches,all_weights);elapsed=time.perf_counter()-start
    previous=json.loads((ROOT/'analysis/ap/allocation_s_T/attention_10.json').read_text())
    errors=dict(actual=float(np.max(np.abs(accelerated[0]*100-np.asarray(previous['threshold_mAP'])))),
        bootstrap_average=float(np.max(np.abs(accelerated[1:10001].mean(1)*100-np.asarray(previous['bootstrap_average'])))),
        bootstrap_AP07=float(np.max(np.abs(accelerated[1:10001,-1]*100-np.asarray(previous['bootstrap_AP07'])))))
    old_per=np.asarray([np.nan if x is None else x for x in previous['per_video_average']])
    new_per=np.nanmean(accelerated[10001:],axis=1)*100
    if not np.allclose(old_per,new_per,atol=1e-8,rtol=0,equal_nan=True):
        raise RuntimeError('Per-video diagnostic equivalence failed')
    # Check all five thresholds on real repeated/zero-weight draws against the
    # untouched NumPy reference, rather than only stored Avg-mAP and AP07.
    start=time.perf_counter();reference=np.asarray([score(caches,w) for w in all_weights[:65]])
    reference_seconds=time.perf_counter()-start
    errors['all_thresholds_65_weights']=float(np.nanmax(np.abs(reference-accelerated[:65])))
    if max(errors.values())>1e-8:
        raise RuntimeError(f'Weighted AP equivalence failed: {errors}')
    tiny=[(np.tile(np.array([0,1,0,1],dtype=np.uint8),(5,1)),np.array([0,0,1,1]),np.array([1,1,0])),
          (np.zeros((5,0),dtype=np.uint8),np.array([],dtype=int),np.array([0,0,1])),
          (np.zeros((5,1),dtype=np.uint8),np.array([0]),np.zeros(3,dtype=int))]
    cases=np.array([[1,1,1],[2,0,1],[0,0,0],[0,1,0]],dtype=np.int64)
    if not np.allclose(score_many(tiny,cases),np.asarray([score(tiny,w) for w in cases]),atol=1e-12,rtol=0,equal_nan=True):
        raise RuntimeError('Zero GT, zero prediction or repeated-video equivalence failed')
    record=dict(passed=True,full_videos=211,windows=792,replicates=10000,
        predictions=sum(len(c[1]) for c in caches),max_abs_errors=errors,
        accelerated_10212_weight_seconds_including_first_compile=elapsed,
        numpy_reference_65_weight_seconds=reference_seconds,
        scientific_change=False,fastmath=False,threads=4,rng='unchanged NumPy Generator(42)',
        source='Complete S/attention/group10 cache from the original implementation')
    json_write(folder/'passed.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    if '--launch' in sys.argv:
        folder=ROOT/'analysis/bootstrap_validation';folder.mkdir(parents=True,exist_ok=True)
        stream=(ROOT/'logs/bootstrap_validation.log').open('a')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve())],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        receipt=dict(pid=process.pid,started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),gpu=False)
        json_write(folder/'process.json',receipt);print(json.dumps(receipt))
    else:main()
