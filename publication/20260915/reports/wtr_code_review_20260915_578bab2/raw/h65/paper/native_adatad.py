"""Unmodified upstream AdaTAD with uniform RGB subsampling on the same windows."""
import copy
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import opentad.datasets  # registers the upstream Compose transforms
from opentad.datasets.builder import PIPELINES, build_dataset, collate
from opentad.models.builder import build_detector
from mmengine.config import Config

from .runtime import ROOT, META_KEYS, json_write

RECIPE = 'native_adatad_uniform_v1'


@PIPELINES.register_module()
class NativeUniformSubsample:
    """Sample *after* the official crop/window, before RGB decoding.

    Keeping the original dataset window grid avoids changing tail-window starts.
    The shortened detector axis has stride r times the original physical stride.
    """
    def __init__(self, ratio):
        if ratio not in (1, 2):
            raise ValueError('Registered native baselines use K768 or K384')
        self.ratio = ratio

    def __call__(self, results):
        r = self.ratio
        results['source_valid_candidates'] = int(results['masks'].sum())
        results['source_candidate_count'] = len(results['frame_inds'])
        results['source_snippet_stride'] = results['snippet_stride']
        if r == 1:
            return results
        results['frame_inds'] = results['frame_inds'][::r].copy()
        results['masks'] = results['masks'][::r].clone()
        results['clip_len'] //= r
        results['snippet_stride'] *= r
        if 'window_size' in results:
            results['window_size'] //= r
        if 'gt_segments' in results:
            results['gt_segments'] = results['gt_segments'] / r
        return results


def read_native_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    if cfg['recipe'] != RECIPE or cfg['dataset'] != 'thumos' or cfg['seed'] != 42:
        raise ValueError('Wrong native AdaTAD experiment contract')
    if cfg['frames'] not in (384, 768) or cfg['backbone'] not in ('s', 'b'):
        raise ValueError('Unregistered native geometry')
    return cfg


def build_native_config(cfg, resources):
    path = ROOT / f'upstream/configs/adatad/thumos/e2e_thumos_videomae_{cfg["backbone"]}_768x1_160_adapter.py'
    native = Config.fromfile(str(path))
    k = cfg['frames']; r = 768 // k; ds = resources['datasets']['thumos']
    native.model.backbone.custom.pretrain = None
    native.model.backbone.backbone.total_frames = k
    # Keep upstream block checkpointing, all 12 adapters and all dense blocks.
    for transform in native.model.backbone.custom.pre_processing_pipeline:
        if transform.type == 'Rearrange': transform.t1 = k // 16
    for transform in native.model.backbone.custom.post_processing_pipeline:
        if transform.type == 'Rearrange': transform.t1 = k // 16
        if transform.type == 'Interpolate': transform.size = k
    native.model.projection.max_seq_len = k
    for split in ('train', 'val', 'test'):
        spec = native.dataset[split]
        spec.ann_file = ds['annotations']; spec.class_map = ds['class_map']
        spec.data_path = ds['train_videos'] if split == 'train' else ds['test_videos']
        # Do NOT change feature_stride=4 or dataset window_size=768 here.
        # The original crop/window and its GT filtering run before subsampling.
        pipeline = []
        for transform in spec.pipeline:
            pipeline.append(transform)
            if transform.type == 'LoadFrames':
                pipeline.append(dict(type='NativeUniformSubsample', ratio=r))
            if transform.type == 'Collect':
                transform.meta_keys = [*META_KEYS, 'source_valid_candidates',
                                       'source_candidate_count', 'source_snippet_stride']
        spec.pipeline = pipeline
    native.evaluation.ground_truth_filename = ds['annotations']
    return native


class NativeAdaTAD(nn.Module):
    """Checkpoint/EMA bookkeeping only; computation is the upstream detector."""
    def __init__(self, native_cfg, cfg, resources):
        super().__init__()
        self.config = copy.deepcopy(cfg)
        self.detector = build_detector(native_cfg.model)
        # Upstream constructor starts this buffer as int; trained checkpoints are
        # floating point. Preserve the checkpoint value rather than truncate it.
        self.detector.rpn_head.loss_normalizer = self.detector.rpn_head.loss_normalizer.float()
        path = resources['teachers']['thumos:' + cfg['backbone']]
        payload = torch.load(path, map_location='cpu')
        state = {k.removeprefix('module.'): v for k, v in payload['state_dict_ema'].items()}
        self.detector.load_state_dict(state, strict=True)
        self.provenance = dict(checkpoint=str(path), state='state_dict_ema',
                               checkpoint_epoch=payload.get('epoch'), strict_keys=len(state),
                               detector_class=type(self.detector).__name__)
        for name, parameter in self.detector.backbone.named_parameters():
            parameter.requires_grad_('adapter' in name)
        self._learned_keys = {name for name, p in self.named_parameters() if p.requires_grad}
        self._learned_keys.update(name for name, _ in self.named_buffers())

    def learned_state(self):
        return {k: v for k, v in self.state_dict().items() if k in self._learned_keys}

    def load_learned(self, state):
        if set(state) != self._learned_keys:
            raise ValueError('Native trainable-state keys changed')
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(k in self._learned_keys for k in result.missing_keys):
            raise ValueError(result)

    def task_loss(self, data):
        return self.detector.forward_train(inputs=data['inputs'], masks=data['masks'],
                    metas=data['metas'], gt_segments=data['gt_segments'], gt_labels=data['gt_labels'])

    def predictions(self, data):
        return self.detector.forward_test(inputs=data['inputs'], masks=data['masks'], metas=data['metas'])

    def optimizer(self, cfg):
        groups = self.detector.get_optim_groups(dict(lr=cfg['head_lr'], weight_decay=cfg['weight_decay']))
        groups.append(dict(params=[p for p in self.detector.backbone.parameters() if p.requires_grad],
                           lr=cfg['adapter_lr'], weight_decay=cfg['weight_decay']))
        return torch.optim.AdamW(groups)


@contextmanager
def inference_mode(model):
    training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            yield
    finally:
        model.train(training)


def _profile_hooks(model, counter):
    handles = []
    def add(module, name):
        def enter(m, args): counter.phase.append(name)
        def leave(m, args, value): counter.phase.pop()
        handles.extend((module.register_forward_pre_hook(enter), module.register_forward_hook(leave)))
    add(model.detector.backbone, 'backbone')
    for block in model.detector.backbone.model.backbone.blocks:
        add(block.mlp, 'ffn')
        if block.use_adapter: add(block.adapter, 'tia')
    for name in ('projection', 'neck', 'rpn_head'):
        module = getattr(model.detector, name, None)
        if module is not None: add(module, name)
    return handles


@torch.no_grad()
def measure_native(model, data, repeats=20):
    from h65.frame.measure import matrix_counter
    torch.cuda.synchronize(); resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with inference_mode(model):
        counter = matrix_counter(); hooks = _profile_hooks(model, counter)
        try:
            with counter: model.predictions(data)
        finally:
            for hook in hooks: hook.remove()
        if counter.unresolved_matrix_ops():
            raise RuntimeError('Uncounted native matrix operations: ' + str(counter.unresolved_matrix_ops()))
        latencies = []
        if repeats:
            for _ in range(3): model.predictions(data)
            torch.cuda.synchronize()
            for _ in range(repeats):
                start = time.perf_counter(); model.predictions(data); torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)
    return dict(matrix_conv_flops=2 * sum(counter.macs.values()), macs_by_component=dict(counter.macs),
        operations=dict(counter.operations), fused_attention=counter.fused_attention,
        raw_model_latency_ms=latencies, latency_mean_ms=float(np.mean(latencies)) if latencies else None,
        latency_median_ms=float(np.median(latencies)) if latencies else None,
        latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else None,
        peak_gib=torch.cuda.max_memory_allocated()/2**30, resident_gib=resident/2**30,
        incremental_peak_gib=(torch.cuda.max_memory_allocated()-resident)/2**30,
        rgb_frames=model.config['frames'], detector_length=model.config['frames'],
        scope='Original AdaTAD: normalization, dense backbone, all TIA adapters, native interpolation, projection/neck/head; RGB resident; decode and NMS excluded',
        matrix_scope='2 MAC: convolution, linear, QK/AV and matrix products; non-matrix arithmetic excluded',
        memory_scope='Includes resident model and, for inline evaluation, optimizer/EMA states')


@torch.no_grad()
def evaluate_native(model, native_cfg, resources, out, metadata):
    from h65.full.runtime import to_gpu
    from opentad.evaluations import build_evaluator
    from .evaluation import merge_windows
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(native_cfg.dataset.test)
    expected = set(resources['datasets']['thumos']['test_ids'])
    if {x[0] for x in dataset.data_list} != expected or len(dataset) != 792:
        raise RuntimeError('Native comparison requires the same 211 videos and 792 windows')
    post = copy.deepcopy(native_cfg.post_processing); post.sliding_window = True
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True)
    result = {}; profiles = {}; times = []; costs = []; full_costs = []; begin = time.perf_counter()
    ledger = out / f'window_execution_{metadata["slurm_job_id"]}.jsonl'
    with inference_mode(model), ledger.open('w') as stream:
        for index, cpu in enumerate(loader):
            data = to_gpu(cpu); meta = data['metas'][0]
            valid = meta['source_valid_candidates']; kind = 'full' if valid == 768 else 'partial' if valid > 384 else 'short'
            # Fixed dense execution at fixed K: profile each padding class once.
            if kind not in profiles:
                profiles[kind] = dict(window_index=index, **measure_native(model, data))
            torch.cuda.synchronize(); start = time.perf_counter()
            predictions = model.predictions(data); torch.cuda.synchronize()
            elapsed = (time.perf_counter()-start)*1000; times.append(elapsed)
            window = model.detector.post_processing(predictions, data['metas'], post, dataset.class_map)
            for name, rows in window.items(): result.setdefault(name, []).extend(rows)
            cost = profiles[kind]['matrix_conv_flops']/1e9; costs.append(cost)
            if kind == 'full': full_costs.append(cost)
            frames = np.asarray(meta['frame_inds'])[:int(data['masks'].sum())]
            stream.write(json.dumps(dict(index=index, video_name=meta['video_name'], kind=kind,
                window_start_frame=meta['window_start_frame'], snippet_stride=meta['snippet_stride'],
                source_valid_candidates=valid, valid_sampled_frames=int(data['masks'].sum()),
                source_first=int(frames[0]), source_last=int(frames[-1]),
                source_gap_max=int(np.diff(frames).max()) if len(frames)>1 else 0,
                gflops=cost, model_ms=elapsed))+'\n')
            if index % 100 == 0:
                json_write(out/'progress.json', dict(windows=index+1, total_windows=len(dataset)))
                print(f'Native {model.config["id"]}: {index+1}/{len(dataset)}', flush=True)
    if set(result) != expected: raise RuntimeError('Missing test videos')
    result = merge_windows(result, post)
    predictions = dict(results=result); json_write(out/'result_detection.json', predictions)
    metrics = build_evaluator(dict(prediction_filename=predictions, **native_cfg.evaluation)).evaluate()
    json_write(out/'profile.json', profiles)
    primary = profiles['full']
    record = dict(metadata, metrics=metrics, test_videos=len(expected), test_windows=len(dataset),
        gflops=float(np.mean(full_costs)), full_window_mean_gflops=float(np.mean(full_costs)),
        dataset_mean_gflops=float(np.mean(costs)), dataset_total_gflops=sum(costs),
        representative_gflops=primary['matrix_conv_flops']/1e9, latency_ms=primary['latency_mean_ms'],
        dataset_model_mean_ms=float(np.mean(times)), dataset_model_ms_quantiles=np.percentile(times,[10,50,90,95]).tolist(),
        e2e_seconds=time.perf_counter()-begin, e2e_scope='Complete evaluation including per-class profiling and AP; not isolated production latency',
        compute_scope='Actual operator profile; fixed dense computation at the same K for every window',
        window_execution_file=ledger.name, profile='profile.json', status='complete')
    np.savez_compressed(out/'window_distribution.npz', gflops=np.asarray(costs), model_ms=np.asarray(times))
    json_write(out/'metrics.json', record); json_write(out/'completed.json', record)
    print(json.dumps(record['metrics']), flush=True)
    return record
