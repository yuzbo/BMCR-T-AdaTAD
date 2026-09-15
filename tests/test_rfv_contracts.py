"""Regression tests for the supported RFV temporal and forecast contracts."""
import numpy as np
import pytest
import torch
from h65.transport import Selection
from h65.raw.contracts import selection_from_support
from h65.raw.value import FEATURE_DIM,TemporalValueHead
from h65.paper.temporal_value import to_standard
from h65.rfv.metrics import ranking_metrics,video_aggregate,forecast
from h65.rfv.value import TemporalProbe,ProbeEMA,from_snapshot,parameter_count,matched_plain_width


@pytest.mark.parametrize('valid_candidates',[1,2,15,383,384,385,767,768])
def test_temporal_duplicate_suffix_cannot_capture_a_valid_observation(valid_candidates):
    n=valid_candidates;count=min(n,384)
    frames=tuple(range(n))+(n-1,)*(768-n)
    candidate=torch.arange(768)[None]<n
    support=tuple(range(count))
    if n>count:support=tuple(range(count-1))+(n-1,)
    raw=selection_from_support(support)
    ids=torch.tensor([list(range(count))+[n-1]*(384-count)])
    valid=torch.arange(384)[None]<count
    rates=candidate.float()
    old=Selection(ids,ids.float(),valid,rates,rates)
    mapped=to_standard(raw,old,frames,candidate)
    assert candidate.gather(1,mapped.indices)[mapped.valid].all()
    assert mapped.indices[0,count-1].item()==support[-1]
    assert torch.equal(mapped.valid,old.valid)


def test_temporal_zero_init_preserves_stop_and_output_is_trainable():
    assert FEATURE_DIM==407
    head=TemporalValueHead();x=torch.randn(16,FEATURE_DIM)
    assert torch.equal(head(x),torch.zeros(16,2))
    loss=(head(x)-.1).square().mean();loss.backward()
    assert head.network[-1].weight.grad.abs().sum()>0


def state():
    nodes=192;actions=8
    times=torch.linspace(0,1,nodes)[None]
    return dict(descriptor=torch.randn(1,actions,FEATURE_DIM),cheap=torch.randn(1,nodes,96),
        times=times,node_valid=torch.ones(1,nodes,dtype=torch.bool),actionness=torch.rand(1,nodes),
        transition=torch.rand(1,nodes),support_distance=torch.zeros(1,nodes),support_occupancy=torch.ones(1,nodes),
        remove_times=torch.linspace(0,.8,actions)[None],insert_times=torch.linspace(.1,1,actions)[None],
        support_times=torch.linspace(0,1,384)[None],support_valid=torch.ones(1,384,dtype=torch.bool))


def test_graph_controls_share_information_and_match_capacity():
    batch=state();width=matched_plain_width()
    models=[TemporalProbe('plain_m'),TemporalProbe('plain_l',hidden=width),
        TemporalProbe('static_graph'),TemporalProbe('dynamic_graph')]
    assert parameter_count(models[2])==parameter_count(models[3])
    assert abs(parameter_count(models[1])-parameter_count(models[3]))/parameter_count(models[3])<.01
    for model in models:
        output=model(batch)
        assert output.shape==(1,8,2) and torch.isfinite(output).all()
        assert torch.equal(output,torch.zeros_like(output))
        output.sum().backward()
        assert model.network[-1].weight.grad.abs().sum()>0


def test_true_ema_is_cumulative_and_snapshot_owns_normalization():
    model=TemporalProbe();ema=ProbeEMA(model,decay=.5)
    key='network.4.bias'
    for value in (2.,4.,10.):
        with torch.no_grad():model.network[-1].bias.fill_(value)
        ema.update(model)
    assert torch.allclose(ema.state[key],torch.full((2,),6.25))
    saved=ema.snapshot(model)
    assert saved['ema']['optimizer_updates']==3
    restored=from_snapshot(saved)
    before=restored.input_mean.clone()
    model.input_mean.add_(3)
    assert torch.equal(restored.input_mean,before)


def test_forecast_beta_one_is_current_post_and_regret_includes_stop():
    anchor=np.array([-2.,1.]);post=np.array([1.,-.2])
    assert np.array_equal(forecast(anchor,post,1.),post)
    assert np.allclose(forecast(anchor,post,1.1),post+.1*(post-anchor))
    assert ranking_metrics(np.array([-1.,-2.]),np.array([-3.,-4.]))['regret']==0.
    assert ranking_metrics(np.array([1.,-2.]),np.array([-3.,-4.]))['regret']==3.


def test_video_aggregation_does_not_overweight_many_windows():
    rows=[dict(video_id='long',regret=1.,ndcg=.5) for _ in range(10)]
    rows.append(dict(video_id='short',regret=3.,ndcg=1.))
    result=video_aggregate(rows)
    assert result['mean']['regret']==2.
    assert result['mean']['ndcg']==.75
