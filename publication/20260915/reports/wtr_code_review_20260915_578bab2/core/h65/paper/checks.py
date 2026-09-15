"""Diagnostic evidence for the real preflight inference contract."""
import torch
from contextlib import ExitStack

@torch.no_grad()
def no_gt_probe(model,data):
    # The production CUDA kernels can differ across identical FP32 forwards.
    # Use deterministic convolution and math attention for this semantic test;
    # keep the original tolerance and also require the repeated-input control.
    from .interventions import evaluation_state
    with ExitStack() as stack:
        stack.enter_context(evaluation_state(model))
        stack.enter_context(torch.backends.cudnn.flags(benchmark=False,deterministic=True))
        if data['inputs'].is_cuda:
            stack.enter_context(torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False))
        return _no_gt_probe(model,data)

def _no_gt_probe(model,data):
    before=torch.cuda.get_rng_state(data['inputs'].device) if data['inputs'].is_cuda else torch.get_rng_state()
    first,a=model.forward_native(data)
    after=torch.cuda.get_rng_state(data['inputs'].device) if data['inputs'].is_cuda else torch.get_rng_state()
    repeat,r=model.forward_native(data)
    changed=dict(data);changed['gt_segments']=[x*0 for x in data['gt_segments']];changed['gt_labels']=[x*0+999 for x in data['gt_labels']]
    other,b=model.forward_native(changed)
    finite=[bool(torch.isfinite(x).all()) for x in (first,repeat,other)]
    def difference(x,y):return float((x-y).abs().max()) if x.shape==y.shape and bool(torch.isfinite(x).all()&torch.isfinite(y).all()) else None
    equal=torch.equal(a['selection'].indices,b['selection'].indices)
    repeat_equal=torch.equal(a['selection'].indices,r['selection'].indices)
    repeat_close=first.shape==repeat.shape and torch.allclose(first,repeat,atol=1e-6,rtol=1e-5)
    plans=[x['plan']['id'] for x in (a,r,b)]
    passed=all(finite) and equal and repeat_equal and repeat_close and len(set(plans))==1 and first.shape==other.shape and torch.allclose(first,other,atol=1e-6,rtol=1e-5)
    record=dict(passed=bool(passed),finite=finite,repeat_max_abs=difference(first,repeat),changed_gt_max_abs=difference(first,other),
                selection_equal=equal,repeat_selection_equal=repeat_equal,repeat_close=bool(repeat_close),
                plans=plans,rng_changed=not torch.equal(before,after),dtype=str(first.dtype),
                probe_backend='deterministic_cudnn_math_sdpa',atol=1e-6,rtol=1e-5,
                module_dropouts=[dict(name=n,p=m.p,training=m.training) for n,m in model.named_modules() if isinstance(m,torch.nn.Dropout) and m.p])
    if not passed:
        record['nonfinite_parameters']=[n for n,p in model.named_parameters() if not bool(torch.isfinite(p).all())]
        record['nonfinite_buffers']=[n for n,p in model.named_buffers() if p.is_floating_point() and not bool(torch.isfinite(p).all())]
    return bool(passed),record
