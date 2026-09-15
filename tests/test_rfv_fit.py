"""Small optimizer/normalization integration tests; no scientific gate claims."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from h65.rfv.dataset import collate_states,normalization
from tools.rfv_fit import train_one


def row(seed,partition,actions=8):
    rng=np.random.default_rng(seed)
    descriptor=rng.normal(size=(actions,407)).astype('float32')
    arrays=dict(descriptor=descriptor,cheap=rng.normal(size=(192,96)).astype('float32'),
        times=np.linspace(0,1,192,dtype='float32'),node_valid=np.ones(192,dtype=bool),
        actionness=rng.random(192).astype('float32'),transition=rng.normal(size=192).astype('float32'),
        support_distance=np.zeros(192,dtype='float32'),support_occupancy=np.ones(192,dtype='float32'),
        remove_times=np.linspace(0,.8,actions,dtype='float32'),insert_times=np.linspace(.1,1,actions,dtype='float32'),
        support_times=np.linspace(0,1,384,dtype='float32'),support_valid=np.ones(384,dtype=bool),
        target=np.stack((descriptor[:,0]*.02,descriptor[:,1]*.01),-1))
    return dict(video_id=f'fixture_{seed}',state_key=f'fixture_{seed}',partition=partition,
        action_pairs=[[i,i+1] for i in range(actions)],arrays=arrays,no_op_error=0.,replay_max_error=0.)


def test_fit_normalization_excludes_calibration_and_holdout():
    rows=[row(0,'fit'),row(1,'fit'),row(2,'calibration'),row(3,'holdout')]
    original=normalization(rows)
    for value in rows[2:]:
        value['arrays']['descriptor']*=10000
        value['arrays']['target']*=10000
        value['arrays']['cheap']*=10000
    after=normalization(rows)
    assert all(torch.equal(a,b) for a,b in zip(original,after))


def test_variable_candidate_padding_is_not_a_training_action():
    batch=collate_states([row(1,'fit',2),row(2,'fit',8)],'cpu')
    assert batch['descriptor'].shape==(2,8,407)
    assert batch['action_valid'].sum().item()==10
    assert batch['action_valid'][0].tolist()==[True,True,False,False,False,False,False,False]


@pytest.mark.parametrize('variant',['plain_m','plain_l','static_graph','dynamic_graph'])
def test_real_optimizer_path_and_ema_snapshot(variant):
    torch.set_num_threads(2)
    rows=[row(0,'fit'),row(1,'fit'),row(2,'calibration'),row(3,'holdout')]
    model,payload=train_one(variant,42,rows,normalization(rows),SimpleNamespace(device='cpu',steps=2))
    assert payload['steps']==2 and payload['ema']['ema']['optimizer_updates']==2
    assert payload['calibration']['video_count']==1
    assert torch.isfinite(model.network[-1].weight).all()
    assert model.network[-1].weight.detach().abs().sum()>0
