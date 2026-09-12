"""Complete211-video test and matched-runtime compute/latency measurement."""
import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'upstream')]
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils._python_dispatch import TorchDispatchMode
from opentad.datasets.builder import build_dataset, collate
from opentad.models.builder import build_detector
from opentad.cores.test_engine import gather_ddp_results
from opentad.evaluations import build_evaluator
from h65.full.model import FormalH65
from h65.full.runtime import config, OFFICIAL, RUNS, initialize_gpu, to_gpu, json_write


class OfficialRuntime(torch.nn.Module):
    def __init__(self, cfg, backbone):
        super().__init__()
        cfg.model.backbone.custom.pretrain = None
        cfg.model.backbone.backbone.with_cp = False
        self.core = build_detector(cfg.model)
        payload = torch.load(OFFICIAL[backbone], map_location='cpu')
        key = 'state_dict_ema' if 'state_dict_ema' in payload else 'state_dict'
        state = {name.removeprefix('module.'): value for name, value in payload[key].items()}
        self.core.load_state_dict(state, strict=True)
        self.initialization = dict(checkpoint=str(OFFICIAL[backbone]), state_key=key, epoch=payload.get('epoch'), strict=True)

    def predictions(self, inputs, masks, metas=None):
        features = self.core.backbone(inputs)
        with torch.autocast('cuda', enabled=False):
            x, mask = self.core.pad_data(features.float(), masks)
            x, mask = self.core.projection(x, mask)
            if self.core.with_neck:
                x, mask = self.core.neck(x, mask)
            predictions = self.core.rpn_head.forward_test(x, mask)
        return predictions, None

    def forward(self, inputs, masks, metas, post_cfg, ext_cls, **kwargs):
        predictions, _ = self.predictions(inputs, masks, metas)
        return self.core.post_processing(predictions, metas, post_cfg, ext_cls)


class ArithmeticCounter(TorchDispatchMode):
    """Actual executed matrix/conv MACs; other operator calls explicitly listed."""
    counted = {'aten.mm.default', 'aten.bmm.default', 'aten.addmm.default',
               'aten.convolution.default', 'aten._convolution.default',
               'aten._scaled_dot_product_flash_attention.default'}

    def __init__(self):
        super().__init__()
        self.macs = collections.Counter()
        self.operations = collections.Counter()
        self.phase = []
        self.fused_attention = []

    @staticmethod
    def attention_macs(q, k, v):
        # Actual VideoMAE q/k/v layout is [batch, heads, tokens, channels].
        return q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2] * (q.shape[3] + v.shape[3])

    def unresolved_matrix_ops(self):
        return [op for op in self.operations if op not in self.counted and
                ('attention' in op or op.startswith(('aten.einsum.', 'aten.matmul.', 'aten.linear.')))]

    def leave_scope(self, module, inputs, output):
        # Returning the popped string would replace the module's actual output
        # under PyTorch forward-hook semantics. Scope bookkeeping returns None.
        self.phase.pop()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        op = str(func)
        self.operations[op] += 1
        count = 0
        if op == 'aten.mm.default':
            count = args[0].shape[0] * args[0].shape[1] * args[1].shape[1]
        elif op == 'aten.bmm.default':
            count = args[0].shape[0] * args[0].shape[1] * args[0].shape[2] * args[1].shape[2]
        elif op == 'aten.addmm.default':
            count = args[1].shape[0] * args[1].shape[1] * args[2].shape[1]
        elif op in ('aten.convolution.default', 'aten._convolution.default'):
            if args[6]:
                raise RuntimeError('transposed convolution needs a separate MAC convention')
            count = result.numel() * args[1][0].numel()
        elif op == 'aten._scaled_dot_product_flash_attention.default':
            q, k, v = args[:3]
            count = self.attention_macs(q, k, v)
            self.fused_attention.append(dict(q=list(q.shape), k=list(k.shape), v=list(v.shape), macs=count))
        self.macs[self.phase[-1] if self.phase else 'routing_and_other'] += count
        return result


def profile(model, sample):
    one = {key: value[:1] if torch.is_tensor(value) or isinstance(value, list) else value for key, value in sample.items()}
    counter, hooks, execution = ArithmeticCounter(), [], []
    core = model.core if isinstance(model, OfficialRuntime) else model.detector
    backbone = model.core.backbone if isinstance(model, OfficialRuntime) else model.backbone
    modules = [(backbone, 'backbone_adapter'), (core.projection, 'projection')]
    if core.with_neck:
        modules.append((core.neck, 'neck'))
    if hasattr(model, 'scout'):
        modules.append((model.scout, 'scout'))
    modules.extend((module, 'head') for module in core.rpn_head.modules())
    for module, name in modules:
        hooks.append(module.register_forward_pre_hook(lambda m, a, name=name: counter.phase.append(name)))
        hooks.append(module.register_forward_hook(counter.leave_scope))
    hooks.append(backbone.model.backbone.patch_embed.register_forward_pre_hook(lambda m, a: execution.append(list(a[0].shape))))
    try:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16), counter:
            model.predictions(one['inputs'], one['masks'], one['metas'])
    finally:
        for hook in hooks:
            hook.remove()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(5):
            model.predictions(one['inputs'], one['masks'], one['metas'])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        latencies = []
        for _ in range(20):
            begin = time.perf_counter()
            model.predictions(one['inputs'], one['masks'], one['metas'])
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - begin)
    unknown_matrix_ops = counter.unresolved_matrix_ops()
    return dict(macs_by_component=dict(counter.macs), total_macs=sum(counter.macs.values()),
                matrix_conv_flops=2*sum(counter.macs.values()), flops_convention='2MAC matrix/conv; other ops excluded and listed',
                uncounted_operator_calls={op: n for op, n in counter.operations.items() if op not in counter.counted},
                unresolved_matrix_ops=unknown_matrix_ops, heavy_inputs=execution,
                fused_attention=counter.fused_attention,
                fused_attention_convention='QK and AV products counted; fused softmax/scaling and any masking/dropout arithmetic excluded',
                latency_seconds=latencies, latency_mean_seconds=sum(latencies)/len(latencies),
                windows_per_second=len(latencies)/sum(latencies), peak_gib=torch.cuda.max_memory_allocated()/2**30,
                scope='batch1 input already on GPU: scout/router/backbone/adapter/detector/preNMS inverse; excludes decoding and NMS',
                precision='BF16 backbone/scout, FP32 detector', valid_input_candidates=int(one['masks'].sum()),
                physical_input_candidates=one['masks'].shape[-1], input_shape=list(one['inputs'].shape),
                input_metadata=one['metas'])


def profile_case(sample):
    valid = int(sample['masks'].sum())
    if valid == sample['masks'].shape[-1]:
        return 'full'
    return 'short' if valid <= 384 else 'partial'


def profile_windows(model, samples):
    """Same first full/partial/short windows for each method, chosen without scores."""
    if 'full' not in samples:
        raise RuntimeError('matched-runtime comparison needs a full768-candidate window')
    cases = {}
    for case in ('full', 'partial', 'short'):
        if case not in samples:
            continue
        index, cpu_sample = samples[case]
        sample = to_gpu(cpu_sample)
        cases[case] = dict(**profile(model, sample), window_index=index)
        del sample
        if cases[case]['unresolved_matrix_ops']:
            raise RuntimeError(f'unresolved matrix operators: {cases[case]["unresolved_matrix_ops"]}')
    return dict(**cases['full'], cases=cases, primary_case='full',
                sample_selection='First full, partial(384<valid<768), short(valid<=384) windows in fixed test order; primary comparison uses full window, not test-set mean latency')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['s', 'b'], required=True)
    parser.add_argument('--variant', choices=['official', 'h65', 'bmcr'], required=True)
    parser.add_argument('--profile-only', action='store_true', help='Repair measurement using saved full-test metrics and the same checkpoint')
    parser.add_argument('--milestone', type=int, choices=list(range(25,81,5)), default=60,
                        help='Total course epoch; joint-stage EMA candidates are tested every5 epochs')
    parser.add_argument('--metrics-only', action='store_true', help='Full211-video accuracy without profiling this candidate')
    parser.add_argument('--total-epochs', type=int, choices=[60,80], default=60)
    args = parser.parse_args()
    if args.milestone > args.total_epochs or (args.total_epochs == 80 and args.variant != 'bmcr'):
        parser.error('milestone must belong to the requested course;80 is the corrected BMCR course')
    hardware = initialize_gpu()
    cfg = config(args.backbone)
    out = RUNS / (f'{args.backbone}_official_test' if args.variant == 'official' else
                  f'{args.backbone}_{args.variant}_test_epoch_{args.milestone:02}')
    out.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(cfg.dataset.test)
    db = json.loads(Path(cfg.dataset.test.ann_file).read_text())['database']
    expected = {name for name, info in db.items() if info['subset'] == 'validation'}
    actual = {row[0] for row in dataset.data_list}
    if len(expected) != 211 or actual != expected:
        raise RuntimeError('full test dataset does not match all211 benchmark video IDs')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)
    if len(loader) != 792:
        raise RuntimeError('fidelity evaluation requires all792 windows of the fixed211-video protocol')
    if args.variant == 'official':
        model = OfficialRuntime(cfg, args.backbone)
        initialization = model.initialization
    else:
        model = FormalH65(cfg.model, variant=args.variant)
        joint_epoch = args.milestone - 20
        checkpoint = RUNS / f'{args.backbone}_{args.variant}/epoch_{joint_epoch:02}.pth'
        payload = torch.load(checkpoint, map_location='cpu')
        if payload['metadata'].get('fidelity_revision') != 'lr_identity_crop_validity_v1':
            raise RuntimeError('fidelity test requires the newly corrected training recipe')
        if payload['completed_epochs'] != joint_epoch or payload['successful_updates'] != joint_epoch*100:
            raise RuntimeError('candidate must match the requested complete joint epoch/update count')
        if payload['metadata']['phase'] != 'joint' or not payload['metadata']['full_training']:
            raise RuntimeError('candidate must come from formal joint training, not preflight')
        if args.total_epochs == 80:
            from h65.full.course import BMCR80_RECIPE
            if payload['metadata'].get('recipe') != BMCR80_RECIPE or payload['metadata'].get('course_total_epochs') != 80:
                raise RuntimeError('BMCR80 evaluation requires the new80-epoch trajectory')
        model.load_state_dict(payload['state_dict_ema'], strict=True)
        initialization = dict(checkpoint=str(checkpoint), state_key='state_dict_ema', total_epochs=args.milestone,
                              fidelity_revision=payload['metadata']['fidelity_revision'])
        if args.total_epochs == 80:
            initialization.update(recipe=BMCR80_RECIPE,course_total_epochs=80,warm_origin=payload['metadata']['warm_origin'])
        del payload
    model.cuda().eval()
    profile_samples = {}
    if args.profile_only:
        saved_metrics = json.loads((out/'metrics.json').read_text())
        if saved_metrics['initialization'] != initialization or saved_metrics['test_videos'] != 211:
            raise RuntimeError('profile-only must preserve the completed full-test checkpoint')
        for index, sample in enumerate(loader):
            profile_samples.setdefault(profile_case(sample), (index, sample))
            if len(profile_samples) == 3:
                break
        del sample
        began = time.perf_counter()
        measured = profile_windows(model, profile_samples)
        elapsed = time.perf_counter()-began
        json_write(out/'profile.json', dict(**hardware, **measured))
        json_write(out/'completed.json', dict(**saved_metrics, profiling_slurm_job_id=hardware['slurm_job_id'],
                   profile_elapsed_seconds=elapsed, profiling_repaired_without_repeating_inference=True))
        print('Completed corrected profiles using saved211-video metrics', flush=True)
        return
    cfg.post_processing.sliding_window = True
    result, began = {}, time.perf_counter()
    for index, cpu_sample in enumerate(loader):
        profile_samples.setdefault(profile_case(cpu_sample), (index, cpu_sample))
        sample = to_gpu(cpu_sample)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            window = model(**sample, post_cfg=cfg.post_processing, ext_cls=dataset.class_map)
        for name, proposals in window.items():
            result.setdefault(name, []).extend(proposals)
        if index % 100 == 0:
            json_write(out/'progress.json', dict(windows=index+1, total_windows=len(loader), videos=len(result)))
            print(f'test windows {index+1}/{len(loader)}, videos {len(result)}', flush=True)
    del sample, cpu_sample
    if set(result) != expected:
        raise RuntimeError('inference did not return all211 video keys')
    rendezvous = out / f'dist_{os.environ["SLURM_JOB_ID"]}'
    dist.init_process_group('gloo', init_method=rendezvous.as_uri(), rank=0, world_size=1)
    try:
        result = gather_ddp_results(1, result, cfg.post_processing)
    finally:
        dist.destroy_process_group()
    prediction = dict(results=result)
    json_write(out/'result_detection.json', prediction)
    evaluator = build_evaluator(dict(prediction_filename=prediction, **cfg.evaluation))
    metrics = evaluator.evaluate()
    selection = ('provided official checkpoint; no selection performed in this experiment'
                 if args.variant == 'official' else f'EMA candidate in user-requested full-test peak selection; total epochs25..{args.total_epochs} every5')
    json_write(out/'metrics.json', dict(**hardware, variant=args.variant, backbone=args.backbone,
               initialization=initialization, metrics=metrics, test_videos=len(result), test_windows=len(loader),
               elapsed_seconds=time.perf_counter()-began, checkpoint_selection=selection))
    print(metrics, flush=True)
    if args.metrics_only:
        json_write(out/'completed.json', dict(**hardware, variant=args.variant, backbone=args.backbone,
                   initialization=initialization, metrics=metrics, test_videos=len(result), test_windows=len(loader),
                   elapsed_seconds=time.perf_counter()-began, checkpoint_selection=selection,
                   accuracy_complete=True, profile_pending_until_selected=True))
        return
    measured = profile_windows(model, profile_samples)
    json_write(out/'profile.json', dict(**hardware, **measured))
    json_write(out/'completed.json', dict(**hardware, variant=args.variant, backbone=args.backbone,
               initialization=initialization, metrics=metrics, test_videos=len(result), test_windows=len(loader),
               elapsed_seconds=time.perf_counter()-began, checkpoint_selection=selection))


if __name__ == '__main__':
    main()
