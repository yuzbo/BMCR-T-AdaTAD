"""Standard-library check of the assigned-device guard; never allocates a GPU."""
import os
from unittest.mock import patch
from h65.rfv.hardware import initialize

resources=dict(execution_backend='autodl_owner',assigned_physical_gpu=1,gpu_type='4080',handoff_receipt='owner.json')
with patch('torch.cuda.device_count',return_value=1), patch('torch.cuda.get_device_name',return_value='NVIDIA GeForce RTX 4080 SUPER'):
    with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
        try:initialize(resources)
        except RuntimeError:pass
        else:raise AssertionError('Wrong physical device was accepted')
    with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1'}):
        result=initialize(resources)
        assert result['assigned_physical_gpu']=='1'
        assert result['allocation_kind']=='owner_assigned_direct_process'
        assert 'slurm_job_id' not in result
print('PASS: assigned AutoDL device and honest non-Slurm execution identity')
