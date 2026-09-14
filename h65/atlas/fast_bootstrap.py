"""FP64 compiled equivalent of the fixed-order weighted AP calculation.

Only CPU arithmetic is compiled. Sorting, greedy matches and NumPy RNG are
unchanged. Project-local dependencies leave the OpenTAD environment untouched.
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'analysis_runtime'))
from numba import njit,prange,set_num_threads


@njit(parallel=True,cache=True,fastmath=False)
def _score_draws(tp,video_indices,gt_counts,offsets,weights):
    draws=weights.shape[0];thresholds=tp.shape[0];classes=len(offsets)-1
    out=np.zeros((draws,thresholds),dtype=np.float64)
    for draw in prange(draws):
        present=0
        for category in range(classes):
            npos=0.0
            for video in range(weights.shape[1]):
                npos+=gt_counts[category,video]*weights[draw,video]
            if npos==0:
                continue
            present+=1
            begin=offsets[category];end=offsets[category+1]
            precision=np.empty(end-begin,dtype=np.float64)
            for threshold in range(thresholds):
                true_positive=0;total=0
                for j in range(begin,end):
                    weight=weights[draw,video_indices[j]]
                    true_positive+=tp[threshold,j]*weight
                    total+=weight
                    precision[j-begin]=true_positive/max(total,1e-12)
                envelope=0.0;ap=0.0
                for j in range(end-1,begin-1,-1):
                    envelope=max(envelope,precision[j-begin])
                    # The weighted recall increment is tp[j]*weight/npos.
                    ap+=tp[threshold,j]*weights[draw,video_indices[j]]/npos*envelope
                out[draw,threshold]+=ap
        if present:
            for threshold in range(thresholds):out[draw,threshold]/=present
        else:
            for threshold in range(thresholds):out[draw,threshold]=np.nan
    return out


def video_weights(videos,replicates):
    """Retain exactly the previous Generator(42) call order."""
    rng=np.random.default_rng(42)
    draws=np.empty((replicates,videos),dtype=np.int64)
    for i in range(replicates):
        draws[i]=np.bincount(rng.integers(0,videos,videos),minlength=videos)
    return draws


def score_many(caches,weights):
    set_num_threads(4)
    offsets=np.r_[0,np.cumsum([len(c[1]) for c in caches])].astype(np.int64)
    tp=np.ascontiguousarray(np.concatenate([c[0] for c in caches],axis=1),dtype=np.uint8)
    indices=np.ascontiguousarray(np.concatenate([c[1] for c in caches]),dtype=np.int64)
    gt=np.ascontiguousarray(np.stack([c[2] for c in caches]),dtype=np.int64)
    return _score_draws(tp,indices,gt,offsets,np.ascontiguousarray(weights,dtype=np.int64))
