"""Collect actual completed S/B test/profiling artifacts; never fill missing scores."""
import argparse
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get('H65_RUNS_DIR', ROOT/'runs')).expanduser().resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['s', 'b'], help='Report one completed backbone while the other is still testing')
    args = parser.parse_args()
    rows = []
    for backbone in ((args.backbone,) if args.backbone else ('s', 'b')):
        baseline = json.loads((RUNS/f'{backbone}_official_test/profile.json').read_text())
        for variant in ('official', 'h65', 'bmcr'):
            root = RUNS/f'{backbone}_{variant}_test'
            result = json.loads((root/'completed.json').read_text())
            profile = json.loads((root/'profile.json').read_text())
            if result['test_videos'] != 211 or profile['unresolved_matrix_ops']:
                raise RuntimeError(f'{backbone}/{variant}: incomplete test or unresolved MAC operators')
            if profile.get('primary_case') != 'full' or baseline.get('primary_case') != 'full':
                raise RuntimeError(f'{backbone}/{variant}: primary profiles must use a full candidate window')
            if profile['window_index'] != baseline['window_index']:
                raise RuntimeError(f'{backbone}/{variant}: profiles used different input windows')
            rows.append(dict(backbone=backbone.upper(), method=variant, metrics=result['metrics'],
                test_videos=result['test_videos'], initialization=result['initialization'],
                gmacs=profile['total_macs']/1e9, matrix_conv_gflops=profile['matrix_conv_flops']/1e9,
                mac_reduction=1-profile['total_macs']/baseline['total_macs'],
                latency_ms=1000*profile['latency_mean_seconds'],
                latency_median_ms=1000*statistics.median(profile['latency_seconds']),
                latency_range_ms=[1000*min(profile['latency_seconds']), 1000*max(profile['latency_seconds'])],
                latency_reduction=1-profile['latency_mean_seconds']/baseline['latency_mean_seconds'],
                median_latency_reduction=1-statistics.median(profile['latency_seconds'])/statistics.median(baseline['latency_seconds']),
                peak_gib=profile['peak_gib'], component_macs=profile['macs_by_component']))
    prefix = args.backbone.upper() + '_' if args.backbone else ''
    out = RUNS.parent/(prefix+'COMPARISON.json' if prefix else 'comparison.json')
    out.write_text(json.dumps(dict(results=rows, precision='BF16 backbone/scout, FP32 detector',
        compute='Matrix/conv FLOPs=2MAC; other operators listed in individual profile.json files',
        course='New models20 uniform warm epochs +40 joint epochs; official existing checkpoint, no retraining',
        interpretation='Full-test method comparison. Different official/new training schedules are not a single-variable sampler ablation.'), indent=2)+'\n')
    labels = dict(official='官方AdaTAD', h65='H65-C', bmcr='BMCR-T')
    thresholds = ['mAP@0.3', 'mAP@0.4', 'mAP@0.5', 'mAP@0.6', 'mAP@0.7', 'average_mAP']
    lines = [f'# H65 / BMCR-T：VideoMAE-{args.backbone.upper() if args.backbone else "S/B"} 完整实验', '',
        '训练集200视频，测试集211视频。新模型从K400识别预训练开始，20轮均匀采样预热后分别联合训练40轮，取末轮EMA。官方AdaTAD使用现有TAD权重，仅测试。', '',
        '| 骨干 | 方法 | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | 平均mAP |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        metrics = ' | '.join(f"{100*row['metrics'][key]:.2f}%" for key in thresholds)
        lines.append(f"| {row['backbone']} | {labels[row['method']]} | {metrics} |")
    lines += ['', '| 骨干 | 方法 | GMAC | MAC削减 | 平均推理ms | 中位推理ms | 平均延迟削减 | 显存GiB |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['backbone']} | {labels[row['method']]} | {row['gmacs']:.2f} | {row['mac_reduction']:.2%} | {row['latency_ms']:.2f} | {row['latency_median_ms']:.2f} | {row['latency_reduction']:.2%} | {row['peak_gib']:.2f} |")
    lines += ['', 'MAC来自实际执行的矩阵与卷积运算，包括融合注意力中的QK与AV乘法；对应FLOPs按2MAC计算，其他算子单列于各profile.json。表内计算和耗时比较使用测试顺序中第一个完整768候选窗口，各方法窗口一致，不是测试集平均时延；短窗口和部分填充窗口另存于profile.json的cases。耗时为相同精度、batch1、GPU已有输入、5次预热后20次测量，含scout/路由/主干/Adapter/检测器和坐标回映，不含视频解码及NMS。', '',
        '20次计时同时报告均值、中位数及原始范围，不删除长耗时样本。各项来自同型号RTX4090的独立作业，节点可能不同；这些结果描述本次运行条件，未测量各组件的时间占比。', '',
        '训练开销应另看各训练阶段的train.jsonl、completed.json与Slurm时长，包括梯度重算、ASFormer辅助回放及EMA反事实教师。共享预热只实际执行一次，两个方法的概念训练周期均为60轮。官方与新方法的训练日程不同，这组结果不能解释为仅改变采样器的单变量消融。', '',
        '训练集交换审计、匹配覆盖率及视频隔离诊断见runs/s_audit与runs/b_audit。未使用测试集选轮次或调整超参数。']
    (RUNS.parent/(prefix+'RESULTS.md')).write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
