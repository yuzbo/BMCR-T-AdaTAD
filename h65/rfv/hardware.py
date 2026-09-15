"""Explicit Slurm or owner-assigned AutoDL execution identity."""
import os
import socket
import torch


def initialize(resources):
    if resources.get('execution_backend')!='autodl_owner':
        from h65.full.runtime import initialize_gpu
        return initialize_gpu(resources.get('gpu_type','4090'))
    assigned=str(resources['assigned_physical_gpu'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=assigned or torch.cuda.device_count()!=1:
        raise RuntimeError('AutoDL RFV must see only the GPU handed over by the Atlas owner')
    name=torch.cuda.get_device_name(0)
    if resources.get('gpu_type','4080') not in name:
        raise RuntimeError('AutoDL GPU does not match the registered device')
    torch.set_num_threads(4)
    return dict(gpu=name,allocation_id=f'autodl_{socket.gethostname()}_{os.getpid()}',
        allocation_kind='owner_assigned_direct_process',assigned_physical_gpu=assigned,
        handoff_receipt=resources['handoff_receipt'],torch=torch.__version__,cuda=torch.version.cuda,
        source_revision=os.environ.get('H65_SOURCE_REVISION','unrecorded'))
