"""Reverse views of fixed-function Standard-T exchanges; no new observations."""
from bisect import bisect_left,bisect_right
import numpy as np
import torch
from h65.raw.contracts import EpisodePublic,RawProposal,RawSelection,swap
from h65.raw.value import FEATURE_DIM


def public_geometry(row):
    episode=EpisodePublic(**row['episode'])
    selected=RawSelection(tuple(row['selected_frame_ids']),tuple(row['selected_valid']))
    candidates=tuple(row['candidate_frame_ids'])
    official=set(episode.official_frame_ids[:sum(episode.official_valid)])
    proposal=RawProposal(candidates,tuple(f in official for f in candidates),
        tuple(f in official for f in candidates),0,'O')
    return episode,selected,proposal


def support_gaps(frame,support,lo,hi):
    left=bisect_left(support,frame)-1;right=bisect_right(support,frame)
    span=max(hi-lo,1)
    return ((frame-support[left])/span if left>=0 else (frame-lo)/span,
        (support[right]-frame)/span if right<len(support) else (hi-frame)/span)


def reverse_descriptors(row,forward):
    """Build phi(C,S-i+j,j->i), separately for every forward candidate.

    The support mean uses its exact real-arithmetic incremental identity. Its
    float32 discrepancy is validated against the original768-point cheap view
    before these derived views are used in the mini. No192 approximation enters.
    """
    forward=np.asarray(forward)
    if forward.shape!=(len(row['action_pairs']),FEATURE_DIM) or forward.dtype!=np.float32:
        raise ValueError('Reverse mini expects the original float32 407D view')
    episode,selected,proposal=public_geometry(row)
    lo,hi=episode.bounds;k=len(selected.support)
    result=forward.copy();records=[]
    for index,(remove,insert) in enumerate(row['action_pairs']):
        changed=swap(selected,remove,insert,proposal)
        restored=swap(changed,insert,remove,proposal)
        if restored!=selected:raise ValueError('Support round trip changed capacity or padding')
        x=forward[index];r=result[index]
        r[:96]=x[96:192];r[96:192]=x[:96]
        r[192:288]=x[192:288]+(x[96:192]-x[:96])/np.float32(k)
        # Global mean and plan/validity quantities remain unchanged.
        r[384]=x[385];r[385]=x[384];r[386]=-x[386]
        r[387:389]=support_gaps(insert,changed.support,lo,hi)
        r[389:391]=support_gaps(remove,changed.support,lo,hi)
        for first,second in ((391,392),(393,394),(395,396),(403,404),(405,406)):
            r[first]=x[second];r[second]=x[first]
        # These are reverse roles, not fixed labels attached to original i/j.
        r[397]=1.;r[398]=0.
        records.append(dict(remove=insert,insert=remove,
            selected_frame_ids=list(changed.frame_ids),selected_valid=list(changed.valid)))
    if not np.isfinite(result).all():raise ValueError('Nonfinite reverse descriptor')
    return result,records


def paired_predictions(head,positive,negative):
    if head.variant!='plain_m':
        raise ValueError('This mini excludes Graph; reverse Graph needs per-candidate S-prime context')
    forward=head({'descriptor':positive})
    reverse=head({'descriptor':negative})
    return forward,reverse,(forward-reverse)*.5


def reverse_loss(forward,reverse,target,valid,scale,condition):
    if condition=='p1':
        a=torch.nn.functional.smooth_l1_loss(forward/scale,target/scale,reduction='none').mean(-1)
        b=torch.nn.functional.smooth_l1_loss(reverse/scale,-target/scale,reduction='none').mean(-1)
        return ((a+b)*.5)[valid].mean()
    if condition=='p2':
        difference=(forward-reverse)*.5
        return torch.nn.functional.smooth_l1_loss(difference/scale,target/scale,reduction='none').mean(-1)[valid].mean()
    raise ValueError('Only bidirectional supervision p1 and antisymmetric-output supervision p2 are registered')
