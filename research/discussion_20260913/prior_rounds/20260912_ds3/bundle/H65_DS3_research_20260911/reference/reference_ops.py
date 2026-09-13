"""Small CPU-only mathematical references, NOT an H65 implementation."""
from __future__ import annotations
from bisect import bisect_left
from math import exp, isfinite
from typing import Sequence

def interpolate_native(centers: Sequence[float], values: Sequence[Sequence[float]],
                       queries: Sequence[float]) -> list[list[float]]:
    """Linear interpolation in physical coordinates, clamp outside observations.
    All observed vector values are preserved at matching coordinates.
    Production torch code additionally needs per-sample valid masks and device support.
    """
    if not centers or len(centers)!=len(values):
        raise ValueError('Require nonempty aligned centers and vectors')
    if any(not isfinite(float(x)) for x in centers) or any(b<=a for a,b in zip(centers,centers[1:])):
        raise ValueError('Centers must be finite and strictly increasing')
    width=len(values[0])
    if width==0 or any(len(v)!=width for v in values): raise ValueError('Vector width mismatch')
    if any(not isfinite(float(v)) for row in values for v in row): raise ValueError('Nonfinite feature')
    output=[]
    for q in queries:
        if not isfinite(float(q)): raise ValueError('Nonfinite query')
        k=bisect_left(centers,q)
        if k==0: output.append(list(values[0])); continue
        if k==len(centers): output.append(list(values[-1])); continue
        if centers[k]==q: output.append(list(values[k])); continue
        a=(q-centers[k-1])/(centers[k]-centers[k-1])
        output.append([(1-a)*u+a*v for u,v in zip(values[k-1],values[k])])
    return output

def softmax(xs: Sequence[float]) -> list[float]:
    if not xs: raise ValueError('Empty logits')
    pivot=max(xs); ex=[exp(x-pivot) for x in xs]; den=sum(ex)
    return [x/den for x in ex]

def prefix_block_cost(active_counts: Sequence[int], segment_depths: Sequence[int]) -> int:
    if len(active_counts)!=len(segment_depths) or not active_counts: raise ValueError('Invalid depth plan')
    if any(x<0 for x in active_counts) or any(x<=0 for x in segment_depths): raise ValueError('Invalid counts')
    if any(b>a for a,b in zip(active_counts,active_counts[1:])): raise ValueError('Active sets must be nested')
    return sum(a*d for a,d in zip(active_counts,segment_depths))
