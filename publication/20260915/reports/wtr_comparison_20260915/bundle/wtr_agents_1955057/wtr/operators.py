"""Proposed training primitives for agents integrating operator-aware routing.

Not automatically attached to PaperModel. Does not pretend a per-token MLP can
reproduce a context-dependent attention operator without contextual input.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn


def same_input_residual_loss(light_output: Tensor, heavy_target: Tensor,
                             valid: Tensor, eps: float = 1e-6) -> Tensor:
    """Heavy/cheap outputs evaluated on exactly the same pre-operator state.

    This differs from the repo's initial-asset full-D/S trajectory-state matching.
    Sampling and extra heavy teacher forwards must be accounted for by the caller.
    """
    if light_output.shape != heavy_target.shape or valid.shape != light_output.shape[:-1]:
        raise ValueError('Same-input operator target shape mismatch.')
    if not bool(valid.any()):
        return light_output.sum()*0.
    target=heavy_target.detach().float();prediction=light_output.float()
    scale=target[valid].square().mean().clamp_min(eps)
    return (prediction[valid]-target[valid]).square().mean()/scale


def freeze_target(module: nn.Module, inputs: Tensor) -> Tensor:
    """Caller must pass a deterministic module or manage its eval state."""
    with torch.no_grad():
        target=module(inputs.detach())
    return target.detach()


def pair_gain_loss(mean_raw: Tensor, logvar_normalized: Tensor,
                   actual_raw: Tensor, fixed_scale: Tensor, mask: Tensor) -> Tensor:
    """Explicit supervised value loss; hard top-k itself need not supply task gradients."""
    if mean_raw.shape != actual_raw.shape or mean_raw.shape != logvar_normalized.shape:
        raise ValueError('Value target shape mismatch.')
    if mask.shape != mean_raw.shape[:-1]:raise ValueError('Mask shape mismatch.')
    if not bool(mask.any()):return mean_raw.sum()*0.
    delta=(mean_raw.float()-actual_raw.detach().float())/fixed_scale
    lv=logvar_normalized.float().clamp(-8,8)
    value=.5*(delta.square()*(-lv).exp()+lv).mean(-1)
    return value[mask].mean()


from contextlib import contextmanager


@contextmanager
def fixed_measurement_normalizer(head, value):
    """Use a bank-fixed loss normalizer, restoring even in-place changes/exceptions.

    A moving loss normalizer can otherwise masquerade as cross-checkpoint value drift.
    This affects measurement only, never the saved model or its inference outputs.
    """
    if value is None:
        yield
        return
    before = getattr(head, 'loss_normalizer', None)
    if not isinstance(before, Tensor):
        raise ValueError('Expected a tensor loss_normalizer in the audited point head.')
    snapshot = before.detach().clone()
    replacement = torch.as_tensor(value, device=before.device, dtype=before.dtype).clone()
    if replacement.shape != before.shape or not bool(torch.isfinite(replacement).all()) or not bool((replacement > 0).all()):
        raise ValueError('Invalid fixed measurement normalizer.')
    head.loss_normalizer = replacement
    try:
        yield
    finally:
        with torch.no_grad():
            before.copy_(snapshot)
        head.loss_normalizer = before
