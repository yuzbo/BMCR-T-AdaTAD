"""Exact reverse roles, current-support reconstruction and antisymmetric output."""
import copy
import numpy as np
import torch
from h65.raw.contracts import EpisodePublic,PreviewTimeline,proposals,physical_uniform,swap
from h65.raw.value import descriptors
from h65.rfv.reverse import reverse_descriptors,paired_predictions,reverse_loss
from h65.rfv.value import TemporalProbe


def fixture(valid_count=768):
    torch.manual_seed(9)
    real=tuple(4*i for i in range(valid_count))
    ids=real+(real[-1],)*(768-valid_count)
    episode=EpisodePublic('fixture',0,'fit','unused',25.,3072,3072/25.,0,4,
        ids,tuple(i<valid_count for i in range(768)))
    timeline=PreviewTimeline(real,tuple(f/25 for f in real),(True,)*valid_count,real)
    public=dict(hidden=torch.randn(1,valid_count,96),action_logits=torch.randn(1,valid_count),
        transition_logits=torch.randn(1,valid_count))
    proposal=proposals(episode,timeline,'O');selection=physical_uniform(episode,proposal)
    unused=sorted(set(proposal.frame_ids)-set(selection.support))
    pairs=[]
    for insert in [unused[0],unused[len(unused)//2],unused[-1]]:
        remove=min(selection.support,key=lambda f:(abs(f-insert),f));pairs.append((remove,insert))
    row=dict(episode=episode.record(),selected_frame_ids=list(selection.frame_ids),
        selected_valid=list(selection.valid),candidate_frame_ids=list(proposal.frame_ids),action_pairs=pairs)
    x=descriptors(episode,timeline,public,proposal,selection,pairs).numpy()
    return row,x,(episode,timeline,public,proposal,selection)


def test_reverse_cache_matches_direct_current_support_and_roundtrip():
    for count in [768,395]:
        row,x,(episode,timeline,public,proposal,selection)=fixture(count)
        reverse,records=reverse_descriptors(row,x)
        for index,((remove,insert),record) in enumerate(zip(row['action_pairs'],records)):
            changed=swap(selection,remove,insert,proposal)
            direct=descriptors(episode,timeline,public,proposal,changed,[(insert,remove)]).numpy()
            np.testing.assert_allclose(reverse[index:index+1],direct,atol=1e-6,rtol=1e-6)
            assert reverse[index,397:399].tolist()==[1.,0.]
            after=copy.deepcopy(row);after.update(record,action_pairs=[(insert,remove)])
            back,_=reverse_descriptors(after,reverse[index:index+1])
            np.testing.assert_allclose(back,x[index:index+1],atol=1e-6,rtol=1e-6)


def test_antisymmetric_output_and_null_view_are_exact():
    torch.manual_seed(11);head=TemporalProbe('plain_m')
    torch.nn.init.normal_(head.network[-1].weight,std=.02)
    a=torch.randn(2,8,407);b=torch.randn(2,8,407)
    forward,reverse,value=paired_predictions(head,a,b)
    _,_,opposite=paired_predictions(head,b,a)
    assert torch.equal(value,-opposite)
    assert torch.equal(paired_predictions(head,a,a)[2],torch.zeros_like(value))
    target=torch.randn(2,8,2)*.01;valid=torch.ones(2,8,dtype=torch.bool)
    loss=reverse_loss(forward,reverse,target,valid,torch.ones(2),'p2')
    loss.backward()
    assert torch.isfinite(head.network[0].weight.grad).all()
    assert head.network[-1].bias.grad.abs().max()<1e-7


def test_p1_and_p2_losses_keep_original_components_and_padding():
    a=torch.tensor([[[.01,.02],[.02,-.01],[200.,200.]]],requires_grad=True)
    b=torch.tensor([[[-.02,-.01],[-.01,.01],[-200.,-200.]]],requires_grad=True)
    target=torch.tensor([[[.005,.012],[.009,-.003],[900.,900.]]])
    valid=torch.tensor([[True,True,False]]);scale=torch.tensor([.01,.02])
    p1=reverse_loss(a,b,target,valid,scale,'p1')
    direct=.5*(torch.nn.functional.smooth_l1_loss(a[:,:2]/scale,target[:,:2]/scale)+
        torch.nn.functional.smooth_l1_loss(b[:,:2]/scale,-target[:,:2]/scale))
    torch.testing.assert_close(p1,direct,rtol=1e-6,atol=1e-8)
    p2=reverse_loss(a,b,target,valid,scale,'p2')
    torch.testing.assert_close(p2,torch.nn.functional.smooth_l1_loss((a[:,:2]-b[:,:2])/(2*scale),target[:,:2]/scale),rtol=1e-6,atol=1e-8)
