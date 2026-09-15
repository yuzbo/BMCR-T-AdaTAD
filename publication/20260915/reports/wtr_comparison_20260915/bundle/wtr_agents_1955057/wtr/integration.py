"""Repository bridge, inspected against 1955057. CUDA/OpenTAD integration not run locally."""
from __future__ import annotations
from argparse import Namespace
from pathlib import Path
import contextlib
import json
import subprocess
import sys
import torch
from . import BASE_SHA
from .core import digest, tensor_digest, file_digest, jsonable
from .operators import fixed_measurement_normalizer


def setup_repository(repo: str, allow_descendant: bool = False) -> Path:
    root = Path(repo).resolve()
    if not (root / 'h65/paper/model.py').is_file():
        raise FileNotFoundError('Not a BMCR-T-AdaTAD checkout: ' + str(root))
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if head != BASE_SHA:
        if not allow_descendant:
            raise RuntimeError(f'Expected {BASE_SHA}, found {head}. Use an isolated fixed worktree.')
        subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', BASE_SHA, head], check=True)
    dirty = subprocess.check_output(['git', '-C', str(root), 'diff', '--name-only'], text=True)
    staged = subprocess.check_output(['git', '-C', str(root), 'diff', '--cached', '--name-only'], text=True)
    if dirty.strip() or staged.strip():
        raise RuntimeError('Tracked source changes are uncommitted. Commit the experiment revision first.')
    sys.path[:0] = [str(root), str(root / 'upstream')]
    return root


def load(args):
    root = setup_repository(args.repo, getattr(args, 'allow_descendant', False))
    from tools.paper_eval import load_model
    model, cfg, resources, metadata = load_model(Namespace(
        config=str(Path(args.config).resolve()), checkpoint=str(Path(args.checkpoint).resolve()),
        resources=str(Path(args.resources).resolve()), state=args.state, need_teacher=False,
        budget_fraction=None, selector=None, disable_frame=False, disable_plan_context=False))
    if model.config['dataset'] != 'thumos' or model.readout.family != 'point':
        raise ValueError('This pilot is restricted to THUMOS point heads; do not silently generalize it.')
    if model.config.get('recipe') == 'graph_tad_v1':
        raise ValueError('Use the non-Graph H65/V2 pilot to avoid plan-0/Graph topology confounding.')
    if model.config.get('static_compression'):
        raise ValueError('Static-compression replay needs a separate retained-layer contract.')
    metadata['probe_checkout'] = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    metadata['checkpoint_sha256'] = file_digest(args.checkpoint)
    return model, cfg, resources, metadata


def training_data(model_cfg, cfg, epoch: int, seed: int):
    from opentad.datasets.builder import build_dataset
    from h65.paper.training import EpochDataset
    dataset = build_dataset(model_cfg.dataset.train)
    wrapped = EpochDataset(dataset, seed); wrapped.epoch = epoch
    return wrapped


def sample_at(dataset, index: int):
    from opentad.datasets.builder import collate
    return collate([dataset[index]])


def window_signature(cpu: dict) -> tuple[str, dict]:
    meta = cpu['metas'][0]
    info = {k: jsonable(meta.get(k)) for k in (
        'video_name', 'frame_inds', 'window_start_frame', 'snippet_stride', 'fps',
        'resize_length', 'window_size', 'offset_frames', 'total_frames')}
    info['inputs_sha256'] = tensor_digest(cpu['inputs'])
    info['masks_sha256'] = tensor_digest(cpu['masks'])
    info['gt_segments'] = jsonable(cpu.get('gt_segments'))
    info['gt_labels'] = jsonable(cpu.get('gt_labels'))
    info['gt_boundary_validity'] = jsonable(cpu.get('gt_boundary_validity'))
    return digest(info), info


def pack_selection(selection) -> dict:
    return {k: getattr(selection, k).detach().cpu() for k in (
        'indices', 'continuous', 'valid', 'density', 'rates')}


def unpack_selection(value: dict, device):
    from h65.transport import Selection
    return Selection(**{k: v.to(device) for k,v in value.items()})


@torch.no_grad()
def execute(model, data, plan, selection, preview=None, measured=False):
    from h65.frame.measure import matrix_counter
    from h65.paper.profile import execution_flops
    from h65.paper.interventions import evaluation_state
    cm = matrix_counter() if measured else contextlib.nullcontext(None)
    with evaluation_state(model), torch.autocast('cuda', dtype=torch.bfloat16), cm as counter:
        if counter is not None:
            counter.phase = ['student_and_task_head']
        native, detail = model.forward_native(data, force_plan=plan, selection=selection,
                                              preview=preview, apply_refiner=False)
        head = model.readout.detector.rpn_head
        with fixed_measurement_normalizer(head, getattr(model, '_wtr_measurement_normalizer', None)):
            losses = model.readout.loss(native, data)
            components = model.readout.components(losses).detach().float()
    if counter is not None:
        unresolved = counter.unresolved_matrix_ops()
        if unresolved:
            raise RuntimeError('Uncounted matrix ops: ' + str(unresolved))
        flops = 2 * sum(counter.macs.values())
        scope = 'matrix_counter_forward_plus_task_loss_ops; not exact all-operator FLOPs'
    else:
        flops = execution_flops(model, detail)
        scope = 'calibrated_inference_forward_proxy; not backward or end-to-end latency'
    if not torch.isfinite(components).all():
        raise RuntimeError('Non-finite task losses.')
    return components, detail, float(flops), scope
