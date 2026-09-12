"""Shared tensor contracts for the parallel FPW implementation."""
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor

ANCHOR_META_DIM=10
QUERY_META_DIM=8
SCOUT_CONTEXT_DIM=96


@dataclass
class AnchorBatch:
    features: Tensor                 # B,A,C before rank-axis interpolation
    contributor_indices: Tensor      # B,A,2 candidate IDs
    contributor_times: Tensor        # B,A,2 original-video frame units
    contributor_valid: Tensor        # B,A,2 bool
    centers: Tensor                  # B,A valid-contributor mean
    spans: Tensor                    # B,A zero if fewer than two valid contributors
    valid: Tensor                    # B,A bool
    clip_position: Tensor            # B,A packed computational clip ID
    tubelet_position: Tensor         # B,A index0..7 inside computational clip
    last_heavy_depth: Tensor         # B,A
    spatial_quality: Tensor          # B,A
    feature_space: str='frame_selected_global_native'


@dataclass
class QueryBatch:
    frame_times: Tensor              # B,T original-video frame units
    centers: Tensor                  # B,T/2 valid-contributor mean
    valid: Tensor                    # B,T/2 bool
    candidate_mask: Tensor           # B,T bool
    membership: Tensor               # B,T bool; selected candidates, not teacher-exact features


@dataclass
class EnginePolicy:
    mode: str='compact'              # compact or dense_mask (same hard forward mask)
    depth_ratio: float=1.0           # A-MoD token capacity at alternating MoD blocks
    depth_schedule: str='none'       # none, amod; late only an explicitly separate control
    mod_layers: tuple=(1,3,5,7,9)     # zero-based: layers2/4/6/8/10; first/last stay dense
    amod_full_kv: bool=False          # main A-MoD packs selected QKV; fullKV is a named ablation
    spatial_ratio: float=1.0         # heavy FFN fraction at admitted native positions
    query_ratio: float=1.0           # Q fraction inside admitted native positions, KV stays full
    gate: str='attention'            # depth main: preceding attention mean; no learned depth router
    structured: bool=False           # 2x2 spatial FFN tiles, otherwise tokenwise
    static_depth: int=12             # retained static blocks; always retain first and last
    use_light: bool=True
    # Optional exact masks for paired intervention / dense-mask equivalence.
    depth_mask: Optional[Tensor]=None # B,A, bool
    spatial_mask: Optional[Tensor]=None # B,A,P, bool
    route_masks: Optional[dict]=None  # exact per-layer token masks for paired execution tests
    static_keep: Optional[tuple]=None # explicit progressive-drop retained layers, including endpoints


def checked_ratio(value):
    if not 0 <= value <= 1:raise ValueError('Execution ratios must lie in[0,1]')
    return value
