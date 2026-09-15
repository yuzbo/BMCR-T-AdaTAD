import pytest
import torch
from h65.rfv.gradients import derivatives,add_vectors,gradient_summary


def test_detached_value_changes_shared_clip_but_not_detector_gradient():
    detector=torch.tensor([1.],requires_grad=True);value=torch.tensor([2.],requires_grad=True)
    parameters=[detector,value];names=['readout.weight','encoder.engine.value_router.weight']
    task=3*detector.sum();auxiliary=4*value.sum()
    task_grad=derivatives(task,parameters);value_grad=derivatives(auxiliary,parameters)
    assert value_grad[0].item()==0 and task_grad[1].item()==0
    plain=gradient_summary(names,task_grad)
    mixed=gradient_summary(names,add_vectors(task_grad,value_grad))
    assert plain['global_norm']==3 and mixed['global_norm']==5
    assert mixed['clip_coefficient']/plain['clip_coefficient']==pytest.approx((3+1e-6)/(5+1e-6))
    assert detector.grad is None and value.grad is None
    assert detector.item()==1 and value.item()==2
