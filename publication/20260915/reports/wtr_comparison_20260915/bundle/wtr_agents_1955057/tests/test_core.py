import pytest
import torch
from wtr.core import (extrapolate,action_distribution,js_distill,exact_capacity,
    gain_per_cost,align_records,tensor_digest,digest,taylor_residual_gain,OperatorValueHead)
from wtr.operators import same_input_residual_loss,pair_gain_loss
from wtr.forecast import cluster_ci,rank_quality


def test_beta_one_is_current():
    a=torch.tensor([[1.,2.]],requires_grad=True);c=torch.tensor([[3.,4.]],requires_grad=True)
    out=extrapolate(a,c,1.)
    assert torch.equal(out,c) and not out.requires_grad


def test_beta_gain_and_clipping():
    a=torch.zeros(2,2);c=torch.ones(2,2)
    assert torch.allclose(extrapolate(a,c,1.2),c*1.2)
    assert torch.allclose(extrapolate(a,c,1.2,max_step=.1),c*1.1)


@pytest.mark.parametrize('beta',[.5,2.1,float('nan')])
def test_invalid_beta(beta):
    with pytest.raises(ValueError):extrapolate(torch.zeros(1,2),torch.ones(1,2),beta)


def test_invalid_values():
    with pytest.raises(ValueError):extrapolate(torch.tensor([[float('nan'),1]]),torch.zeros(1,2),1.1)


def test_stop_dominates_negative_actions():
    p=action_distribution(torch.tensor([[-2.,-2.],[-1.,-1.]]),torch.ones(2))
    assert p.argmax().item()==2 and torch.allclose(p.sum(),torch.tensor(1.))


def test_all_masked_stop():
    p=action_distribution(torch.randn(3,2),torch.ones(2),torch.zeros(3,dtype=torch.bool))
    assert torch.equal(p,torch.tensor([0.,0.,0.,1.]))


def test_js_gradient_only_student():
    s=torch.randn(4,requires_grad=True);t=torch.randn(4,requires_grad=True)
    loss=js_distill(s.softmax(0),t.softmax(0));loss.backward()
    assert s.grad is not None and t.grad is None and loss.item()>=0


def test_js_zero_and_upper_bound():
    p=torch.tensor([1.,0.]);q=torch.tensor([0.,1.])
    assert js_distill(p,p).item()==0
    assert abs(js_distill(p,q).item()-torch.log(torch.tensor(2.)).item())<1e-6


def test_capacity_exact():
    s=torch.tensor([1.,4.,3.,9.]);v=torch.tensor([1,1,1,0],dtype=torch.bool)
    m=exact_capacity(s,v,2)
    assert m.tolist()==[False,True,True,False]


def test_coverage_quota():
    s=torch.tensor([4.,3.,2.,1.]);v=torch.ones(4,dtype=torch.bool);g=torch.tensor([0,0,1,1])
    m=exact_capacity(s,v,2,g,1)
    assert m.tolist()==[True,False,True,False]


def test_infeasible_coverage():
    with pytest.raises(ValueError):exact_capacity(torch.arange(4.).float(),torch.ones(4,dtype=torch.bool),1,torch.tensor([0,0,1,1]),1)


@pytest.mark.parametrize('cost',[0.,-1.])
def test_equal_cost_swap_rejected(cost):
    with pytest.raises(ValueError):gain_per_cost(torch.ones(2),torch.full((2,),cost))


def test_raw_gain_per_cost():
    assert gain_per_cost(torch.tensor([2.]),torch.tensor([4.])).item()==.5


def row():
    return dict(action_id='a',bank_hash='b',window_hash='w',support_id='s',split='fit',action_type='frame')


def test_exact_alignment():
    a=row();assert len(align_records([a],[dict(a)]))==1
    b=dict(a,window_hash='different')
    with pytest.raises(ValueError):align_records([a],[b])
    with pytest.raises(ValueError):align_records([a],[])


def test_tensor_hash_reproducible():
    x=torch.randn(2,3,4)
    assert tensor_digest(x)==tensor_digest(x.clone())
    assert tensor_digest(x)!=tensor_digest(x+1)
    assert digest({'x':x})==digest({'x':x.clone()})


def test_taylor_signed():
    g=torch.ones(2,3);h=torch.ones(2,3);c=torch.zeros(2,3)
    assert torch.equal(taylor_residual_gain(g,h,c),torch.tensor([-3.,-3.]))


def test_head_shapes_gradients():
    head=OperatorValueHead(10);x=torch.randn(4,10)
    mean,lv=head(x);assert mean.shape==(4,2,2)
    target=torch.ones_like(mean);loss=pair_gain_loss(mean,lv,target,torch.ones(2),torch.ones(4,2,dtype=torch.bool))
    loss.backward();assert head.net[-1].weight.grad.abs().sum()>0


def test_residual_teacher_detached():
    light=torch.randn(3,4,requires_grad=True);heavy=torch.randn(3,4,requires_grad=True)
    loss=same_input_residual_loss(light,heavy,torch.ones(3,dtype=torch.bool));loss.backward()
    assert light.grad is not None and heavy.grad is None


def test_empty_operator_mask_safe():
    x=torch.randn(2,3,requires_grad=True)
    loss=same_input_residual_loss(x,torch.zeros_like(x),torch.zeros(2,dtype=torch.bool))
    loss.backward();assert torch.equal(x.grad,torch.zeros_like(x))


def test_video_cluster_unit():
    r=cluster_ci([('one',1.),('one',1.),('two',0.)],n=100)
    assert r['mean']==.5 and r['videos']==2


def test_constant_rank_not_fabricated():
    assert rank_quality([0,0],[1,2]) is None
    assert rank_quality([1,2],[1,2])==pytest.approx(1.)


def test_measurement_normalizer_restores():
    from wtr.operators import fixed_measurement_normalizer
    head=torch.nn.Module();head.register_buffer('loss_normalizer',torch.tensor(12.))
    original=head.loss_normalizer
    with fixed_measurement_normalizer(head,torch.tensor(20.)):
        assert head.loss_normalizer.item()==20.
        head.loss_normalizer.mul_(.5)
    assert head.loss_normalizer is original and original.item()==12.


def test_measurement_normalizer_restores_exception():
    from wtr.operators import fixed_measurement_normalizer
    head=torch.nn.Module();head.register_buffer('loss_normalizer',torch.tensor(12.))
    with pytest.raises(RuntimeError):
        with fixed_measurement_normalizer(head,torch.tensor(20.)):
            head.loss_normalizer=torch.tensor(999.)
            raise RuntimeError('probe interrupted')
    assert head.loss_normalizer.item()==12.
