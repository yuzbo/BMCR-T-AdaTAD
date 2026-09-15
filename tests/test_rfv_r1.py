"""Numerical and causal contracts needed before the new ranking experiment."""
from types import SimpleNamespace
import torch
from h65.rfv.objectives import value_loss,frozen_rank_scale,centered_log_distribution
from h65.rfv.graph import TemporalEdges,OFFSETS
from h65.rfv.value import TemporalProbe,parameter_count,matched_plain_width
from h65.rfv.dataset import normalization
from tools.rfv_fit import train_one
from test_rfv_fit import row


def test_r0_is_the_original_component_huber():
    prediction=torch.tensor([[[.002,.01],[-.003,.02],[3.,4.]]],requires_grad=True)
    target=torch.tensor([[[.001,.003],[-.005,.015],[900.,900.]]])
    valid=torch.tensor([[True,True,False]]);scale=torch.tensor([.004,.01])
    old=torch.nn.functional.smooth_l1_loss(prediction/scale,target/scale,reduction='none').mean(-1)[valid].mean()
    assert torch.equal(value_loss(prediction,target,valid,scale,'r0')['total'],old)


def test_js_is_state_shift_invariant_and_ignores_padding():
    scores=torch.tensor([[.5,-.2,1.1],[.3,.2,90.]],dtype=torch.float64)
    valid=torch.tensor([[True,True,True],[True,True,False]])
    before=centered_log_distribution(scores,valid)
    after=centered_log_distribution(scores+torch.tensor([[23.],[-11.]],dtype=torch.float64),valid)
    assert torch.allclose(before[valid],after[valid],atol=1e-12,rtol=0)
    prediction=torch.stack((scores*.002,scores*.003),-1).requires_grad_()
    target=torch.stack((scores.flip(1)*.003,scores.flip(1)*.002),-1)
    scale=torch.tensor([.01,.01],dtype=torch.float64)
    loss=value_loss(prediction,target,valid,scale,'r1',.005)
    changed=target.clone();changed[~valid]=1e9
    second=value_loss(prediction,changed,valid,scale,'r1',.005)
    assert torch.equal(loss['total'],second['total'])
    loss['total'].backward()
    assert torch.isfinite(prediction.grad).all() and not prediction.grad[~valid].any()
    assert 0<=float(loss['rank'])<=.693148


def test_common_gain_scale_preserves_the_true_total_loss_order():
    target=torch.tensor([[.01,-.005],[.001,.005],[.03,-.02]])
    scale=frozen_rank_scale(target)
    assert torch.equal(target.sum(-1).argsort(),(target.sum(-1)/scale).argsort())
    component=target.square().mean(0).sqrt()
    weights=component/scale
    assert torch.allclose((target/component*weights).sum(-1),target.sum(-1)/scale)


def test_r1_real_updates_keep_fit_only_scaling_and_true_ema():
    torch.set_num_threads(2)
    rows=[row(0,'fit'),row(1,'fit'),row(2,'calibration'),row(3,'holdout')]
    model,payload=train_one('plain_m',42,rows,normalization(rows),
        SimpleNamespace(device='cpu',steps=3,objective='r1',temperature=1.))
    assert payload['objective']['name']=='r1' and payload['objective']['huber_weight']==.1
    assert payload['ema']['ema']['optimizer_updates']==3
    assert model.network[-1].weight.detach().abs().sum()>0
    fit_target=torch.cat([torch.from_numpy(r['arrays']['target']) for r in rows[:2]])
    assert payload['objective']['rank_scale']==float(frozen_rank_scale(fit_target))


def test_graph_addresses_stay_in_the_shared_bounded_temporal_pool():
    torch.manual_seed(42)
    x=torch.randn(1,32,64);valid=torch.ones(1,32,dtype=torch.bool)
    geometry=torch.zeros(1,32,5);geometry[...,0]=torch.linspace(0,1,32)
    geometry[...,3]=torch.rand(1,32);geometry[...,4]=1
    for dynamic in [False,True]:
        router=TemporalEdges(64,dynamic)
        (indices,weights),cost=router(x,valid,geometry)
        for center in range(32):
            expected={center+offset for offset in OFFSETS if 0<=center+offset<32}
            retained=set(indices[0,center][weights[0,center]>0].tolist())
            assert center in retained and retained<=expected
        assert cost['candidate_degree']==9 and cost['referral_paths']==0
        assert torch.allclose(weights.sum(-1),torch.ones(1,32))
        assert weights.requires_grad
    static=TemporalProbe('static_graph');dynamic=TemporalProbe('dynamic_graph')
    plain=TemporalProbe('plain_l',hidden=matched_plain_width())
    assert parameter_count(static)==parameter_count(dynamic)
    assert abs(parameter_count(plain)-parameter_count(static))/parameter_count(static)<.01
