"""Finalize the completed H65 fidelity experiment from its collected evidence."""
import copy
import json
import math
import os
import statistics
from pathlib import Path
from fidelity_select import EPOCHS, select_best
from fidelity_progress import main as publish_accuracy

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'fidelity_20260911'
RUNS = EXP / 'runs'
PUBLIC = EXP / 'evaluations'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def geometry(profile):
    return [{k: v for k, v in meta.items() if k != 'data_path'} for meta in profile['input_metadata']]


def verify_profile(profile, reference):
    assert profile['primary_case'] == 'full'
    assert set(profile['cases']) == {'full', 'partial', 'short'}
    cases = {}
    for name, case in profile['cases'].items():
        baseline = reference['cases'][name]
        assert not case['unresolved_matrix_ops']
        assert case['total_macs'] == sum(case['macs_by_component'].values())
        assert case['matrix_conv_flops'] == 2*case['total_macs']
        assert case['input_shape'] == baseline['input_shape']
        assert case['precision'] == baseline['precision']
        assert geometry(case) == geometry(baseline)
        assert case['window_index'] == baseline['window_index']
        assert case['valid_input_candidates'] == baseline['valid_input_candidates']
        assert len(case['heavy_inputs']) == len(baseline['heavy_inputs']) == 1
        assert case['heavy_inputs'][0][0] == 24 and baseline['heavy_inputs'][0][0] == 48
        assert case['heavy_inputs'][0][1:] == baseline['heavy_inputs'][0][1:]
        assert len(case['latency_seconds']) == 20
        assert all(math.isfinite(x) and x > 0 for x in case['latency_seconds'])
        assert abs(statistics.mean(case['latency_seconds'])-case['latency_mean_seconds']) < 1e-12
        assert len(case['fused_attention']) == 12
        for call in case['fused_attention']:
            q, k, v = call['q'], call['k'], call['v']
            assert call['macs'] == q[0]*q[1]*q[2]*k[2]*(q[3]+v[3])
        cases[name] = dict(window_index=case['window_index'], valid_candidates=case['valid_input_candidates'],
                           fused_calls=12, fused_macs=sum(x['macs'] for x in case['fused_attention']),
                           unresolved_matrix_ops=[], matched_reference_geometry=True)
    full = profile['cases']['full']
    assert full['valid_input_candidates'] == 768 and full['window_index'] == 2
    assert profile['total_macs'] == full['total_macs']
    assert profile['latency_seconds'] == full['latency_seconds']
    return cases


def public_profile(profile):
    result = copy.deepcopy(profile)
    for value in [result, *result['cases'].values()]:
        for meta in value['input_metadata']:
            meta['data_path'] = '<resource-root>/thumos14/videos/test'
    result['publication_note'] = 'Only resource data_path normalized; measurements and operator traces unchanged.'
    return result


def main():
    publish_accuracy()
    training = read(EXP/'TRAINING_COMPLETION.json')
    deployment = read(EXP/'deployment.json')
    assert training['full60_epoch_courses_complete'] and training['total_formal_updates'] == 12000
    assert deployment['comparison'] == 'COMPLETED'
    assert all(v['status'] == 'COMPLETED' for v in deployment['stages'].values())
    legacy = read(ROOT/'phase2_20260910/comparison.json')['results']
    rows, selections, validation = [], {}, {}
    for b in ('s', 'b'):
        selection = select_best(RUNS, b)
        assert selection['all_candidates_evaluated'] and selection['evaluated'] == 8
        official = next(x for x in legacy if x['backbone'] == b.upper() and x['method'] == 'official')
        old = next(x for x in legacy if x['backbone'] == b.upper() and x['method'] == 'h65')
        reference = read(ROOT/f'phase2_20260910/runs/{b}_official_test/profile.json')
        assert abs(reference['total_macs']/1e9-official['gmacs']) < 1e-9
        for candidate in selection['candidates']:
            epoch = candidate['total_epochs']
            verification = read(RUNS/f'{b}_h65_test_epoch_{epoch}/verification.json')
            assert all(verification['checks'].values())
            assert verification['test_videos'] == 211
        selections[b.upper()] = dict(selected_total_epoch=selection['best']['total_epochs'],
            checkpoint=f"runs/{b}_h65/epoch_{selection['best']['total_epochs']-20:02}.pth",
            state_key='state_dict_ema', terminal_checkpoint=f'runs/{b}_h65/epoch_40.pth',
            candidates=list(EPOCHS), criterion='highest full211-test average_mAP; earliest tie',
            test_based_selection=True)
        for label, candidate in (('test_peak', selection['best']), ('epoch60_terminal', selection['terminal'])):
            epoch = candidate['total_epochs']
            folder = RUNS/f'{b}_h65_test_epoch_{epoch}'
            profile = read(folder/'profile.json')
            validation[f'{b}_{epoch}'] = verify_profile(profile, reference)
            write(PUBLIC/f'{b}_h65_test_epoch_{epoch}'/'profile.json', public_profile(profile))
            average = candidate['average_mAP']
            ratio = profile['total_macs']/1e9/official['gmacs']
            mean_ms = 1000*profile['latency_mean_seconds']
            median_ms = 1000*statistics.median(profile['latency_seconds'])
            rows.append(dict(backbone=b.upper(), selection=label, total_epoch=epoch, metrics=candidate['metrics'],
                checkpoint=f'runs/{b}_h65/epoch_{epoch-20:02}.pth',
                delta_vs_legacy_terminal_pp=100*(average-old['metrics']['average_mAP']),
                gap_to_official_pp=100*(average-official['metrics']['average_mAP']),
                map_retention_of_official=average/official['metrics']['average_mAP'],
                gmacs=profile['total_macs']/1e9, matrix_conv_gflops=profile['matrix_conv_flops']/1e9,
                flops_fraction_of_official=ratio, flops_reduction=1-ratio,
                latency_mean_ms=mean_ms, latency_median_ms=median_ms,
                latency_min_ms=1000*min(profile['latency_seconds']), latency_max_ms=1000*max(profile['latency_seconds']),
                latency_mean_change_vs_official=mean_ms/official['latency_ms']-1,
                latency_median_change_vs_official=median_ms/official['latency_median_ms']-1,
                peak_gib=profile['peak_gib'], memory_reduction=1-profile['peak_gib']/official['peak_gib'],
                profile_job_id=profile['slurm_job_id'],
                profile_path=f'evaluations/{b}_h65_test_epoch_{epoch}/profile.json'))
    curves = read(EXP/'INTERMEDIATE_RESULTS.json')['results']
    by_epoch = {(r['backbone'], r['total_epoch']): r['metrics']['average_mAP'] for r in curves}
    history = dict(s_original_90_epoch_terminal_percent=65.3857244379457,
                   s_original_90_epoch_observed_peak_percent_rounded=65.65,
                   s_original_total60_percent_rounded=64.40,
                   s_corrected_gap_to_historical_terminal_pp=100*by_epoch['S', 60]-65.3857244379457,
                   s_corrected_gap_to_historical_total60_pp=100*by_epoch['S', 60]-64.40,
                   s_last5_epoch_change_pp=100*(by_epoch['S', 60]-by_epoch['S', 55]),
                   b_last5_epoch_change_pp=100*(by_epoch['B', 60]-by_epoch['B', 55]),
                   b_peak_selection_gain_over_terminal_pp=100*(by_epoch['B', 55]-by_epoch['B', 60]))
    result = dict(seed=3407, train_videos=200, test_videos=211, test_windows=792, evaluated_candidates=16,
                  training_epochs_per_backbone=60, training_updates_total=12000,
                  scientific_code='1c1552a', deployment_code='990169a', selected_models=selections,
                  results=rows, legacy_reference=legacy, historical_context=history,
                  compute_convention='matrix/conv FLOPs=2MAC, including fused QK/AV; other arithmetic excluded',
                  latency_scope='20 synchronized batch1 forwards after5 warmups; fixed full768 window on RTX4090; input on GPU; decoding and NMS excluded',
                  latency_samples='All20 retained; B55 includes253.6554ms sample. Different4090 jobs/nodes; not whole-test-set end-to-end latency.',
                  selection_scope='User-requested full-test peak across8 candidates per backbone; not independent unseen-test performance.')
    write(EXP/'FINAL_COMPARISON.json', result)
    write(EXP/'SELECTED_CHECKPOINTS.json', selections)
    write(EXP/'FINAL_VALIDATION.json', dict(training_complete=True, formal_updates=12000,
        all16_accuracy_results_verified=True, all24_controller_stages_completed=True,
        selected_and_terminal_profiles=validation, distinct_profiles=len(validation),
        no_retraining_performed_during_finalization=True))
    make_figures(result, curves)
    make_report(result, curves)
    print(json.dumps(dict(selected_models=selections, results=rows, historical_context=history), indent=2))


def make_figures(result, curves):
    os.environ['MPLCONFIGDIR'] = str(ROOT/'cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = EXP/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, b in zip(axes, ('S', 'B')):
        values = [r for r in curves if r['backbone'] == b]
        official = next(r for r in result['legacy_reference'] if r['backbone'] == b and r['method'] == 'official')
        old = next(r for r in result['legacy_reference'] if r['backbone'] == b and r['method'] == 'h65')
        chosen = next(r for r in result['results'] if r['backbone'] == b and r['selection'] == 'test_peak')
        ax.plot([r['total_epoch'] for r in values], [100*r['metrics']['average_mAP'] for r in values], 'o-', color='#2563eb', label='Corrected H65')
        ax.axhline(100*official['metrics']['average_mAP'], color='#202938', ls='--', label='Official provided checkpoint')
        ax.axhline(100*old['metrics']['average_mAP'], color='#94a3b8', ls=':', label='Legacy H65, epoch60')
        if b == 'S': ax.axhline(65.3857244379457, color='#15803d', ls='-.', lw=1, label='Historical H65-S, epoch90')
        ax.scatter([chosen['total_epoch']], [100*chosen['metrics']['average_mAP']], color='#f59e0b', marker='*', s=180, zorder=5)
        ax.annotate(f"{100*chosen['metrics']['average_mAP']:.2f}% @ {chosen['total_epoch']}", (chosen['total_epoch'],100*chosen['metrics']['average_mAP']),xytext=(-72,-20),textcoords='offset points',fontsize=10)
        ax.set(title=f'VideoMAE-{b}', xlabel='Total training epoch', xlim=(23,63), ylim=(51,74), xticks=list(EPOCHS))
        ax.grid(alpha=.2); ax.legend(loc='lower right', fontsize=8)
    axes[0].set_ylabel('Average mAP (%) at tIoU 0.3-0.7')
    fig.suptitle('Full211-video EMA test curve | seed3407 | 20 warm + 40 joint epochs')
    fig.tight_layout(); fig.savefig(figures/'accuracy_curve.png',dpi=180); fig.savefig(figures/'accuracy_curve.svg'); plt.close(fig)
    selected = [r for r in result['results'] if r['selection'] == 'test_peak']
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    labels = [f"{r['backbone']} @ {r['total_epoch']}" for r in selected]
    for ax, key, title in ((axes[0],'flops_fraction_of_official','Matrix/conv FLOPs'),(axes[2],'memory_reduction','Peak allocated VRAM')):
        values=[100*(1-r[key] if key=='memory_reduction' else r[key]) for r in selected]
        bars=ax.bar(labels,values,color='#2563eb',width=.5); ax.bar_label(bars,fmt='%.2f%%',padding=3)
        ax.axhline(100,color='#64748b',ls='--'); ax.set(title=title,ylim=(0,115),ylabel='% of corresponding official'); ax.grid(axis='y',alpha=.2)
    x=[0,1]
    for offset,key,label,color in ((-.18,'latency_mean_change_vs_official','Mean','#f59e0b'),(.18,'latency_median_change_vs_official','Median','#2563eb')):
        bars=axes[1].bar([v+offset for v in x],[100*(1+r[key]) for r in selected],width=.34,label=label,color=color)
        axes[1].bar_label(bars,fmt='%.1f%%',padding=3,fontsize=8)
    axes[1].axhline(100,color='#64748b',ls='--'); axes[1].set(title='Fixed-window latency',xticks=x,xticklabels=labels,ylim=(0,195)); axes[1].legend(fontsize=8); axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Selected H65 checkpoints | input on GPU | decoding and NMS excluded')
    fig.tight_layout(); fig.savefig(figures/'resource_comparison.png',dpi=180); fig.savefig(figures/'resource_comparison.svg'); plt.close(fig)


def make_report(data, curves):
    selected = [r for r in data['results'] if r['selection'] == 'test_peak']
    terminal = {r['backbone']: r for r in data['results'] if r['selection'] == 'epoch60_terminal'}
    h = data['historical_context']
    lines = ['# H65保真修正完整单种子实验：最终报告','',
        '两项保真修正后，H65-S测试峰值为63.4094%（第60轮），H65-B为67.1328%（第55轮）。相同第60轮终点相对旧H65-C提高0.5501/0.2898个百分点。计算量仍约为官方一半，但指定窗口实测没有推理加速；历史H65-S的65.3857%尚未恢复。','',
        '全部200训练视频、211测试视频/792窗口；固定seed3407。S/B均从识别预训练完成20轮预热＋40轮联合，共12000次正式更新。官方只复用此前已完整评测的提供检查点，不重新训练。本轮不加入独立均匀基线、内部160/40划分、原时间检测或粗细融合结构。','',
        '按用户要求，分别在总25/30/35/40/45/50/55/60轮完整测试EMA，按平均mAP峰值选模、同分取早轮。全部16次评测已完成；这是测试集选模，不能称为独立未见测试性能。第60轮另行保留。','',
        '| 骨干 | 选模 | 平均mAP | @0.5 | 相对官方(pp) | FLOPs/官方 | FLOPs降幅 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in selected:
        lines.append(f"| {r['backbone']} | 第{r['total_epoch']}轮 | {100*r['metrics']['average_mAP']:.4f}% | {100*r['metrics']['mAP@0.5']:.4f}% | {r['gap_to_official_pp']:+.4f} | {100*r['flops_fraction_of_official']:.4f}% | {100*r['flops_reduction']:.4f}% |")
    lines += ['', '官方参考平均mAP为S69.0126%、B71.1280%；选中H65分别保留其约91.88%/94.38%的平均mAP。这是观测比例，未新增90%等验收门槛。', '',
        '| 骨干 | 旧H65-C第60轮 | 修正后第60轮 | 相同终点变化(pp) | 修正后测试峰值 | 峰值额外收益(pp) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in selected:
        old=next(x for x in data['legacy_reference'] if x['backbone']==r['backbone'] and x['method']=='h65'); end=terminal[r['backbone']]
        lines.append(f"| {r['backbone']} | {100*old['metrics']['average_mAP']:.4f}% | {100*end['metrics']['average_mAP']:.4f}% | {end['delta_vs_legacy_terminal_pp']:+.4f} | {100*r['metrics']['average_mAP']:.4f}% | {100*(r['metrics']['average_mAP']-end['metrics']['average_mAP']):+.4f} |")
    lines += ['', 'B的峰值相对旧终点共提高0.5200个百分点，其中0.2898来自相同终点的观测变化，另0.2302来自本轮测试峰值选择；不能把后者都归因于模型修正。两项修正一起应用且仅一个种子，未分离各自贡献或证明统计稳定性。旧BMCR-T未按新配方重训，不能据此建立与本轮H65的配方匹配机制对照。','',
        '![完整准确率曲线](figures/accuracy_curve.png)','',
        '60轮课程的判断：S第55至60轮仍提高0.3261个百分点，但B在第55轮达到峰值后，第60轮下降0.2302个百分点。当前数据不支持统一断言“S/B都因60轮不足而掉分”。S仍有末段上升，单独比较更长课程有价值，但不能预先保证收益。','',
        f"历史S为30预热＋60联合，固定终点65.385724%；其历史总60轮日志值约64.40%，晚期已取得曲线峰值约65.65%。本轮S终点距历史90轮终点仍{abs(h['s_corrected_gap_to_historical_terminal_pp']):.4f}个百分点，距历史总60轮日志值约{abs(h['s_corrected_gap_to_historical_total60_pp']):.4f}个百分点。历史总60轮处在另一套30+60课程中，不能把它视为本轮20+40的等价时点。",'',
        '本轮确实修复了错误LR身份分组与裁剪伪端点监督，但没有恢复全部历史分数。因此这两处缺口不能解释完整的历史差距。历史65.39对应42dba3f的global-TIA192，与当前相同；rank检测/K384也为共同机制。不要重新归因于已排除的“局部TIA换成全局TIA”、TCN改ASFormer或无关诊断分支。历史数字沿用已取得的原始回执/日志，本轮没有重新评测历史90轮模型。','',
        '若下一轮专门验证课程压缩，应在训练前锁定20+40与30+60对照，固定模型、目标、输入、数据和种子，显式登记LR与课程时间轴的映射；优先比较固定终点，峰值候选数量与取样规则也预先固定。本轮没有临时加训至90轮。原时间检测/粗细融合以及BMCR增量应在这个已明确的修正基线上单独开展，避免将课程变化与新结构一起归因。','',
        '| 骨干/检查点 | GMAC | 矩阵/卷积GFLOPs | 平均/中位时延ms | 时延范围ms | 峰值显存GiB |',
        '|---|---:|---:|---:|---:|---:|']
    for b in ('S','B'):
        r=next(x for x in data['legacy_reference'] if x['backbone']==b and x['method']=='official')
        lines.append(f"| {b}官方参考 | {r['gmacs']:.4f} | {r['matrix_conv_gflops']:.4f} | {r['latency_ms']:.2f}/{r['latency_median_ms']:.2f} | {r['latency_range_ms'][0]:.2f}–{r['latency_range_ms'][1]:.2f} | {r['peak_gib']:.4f} |")
        for r in data['results']:
            if r['backbone']!=b or (b=='S' and r['selection']=='epoch60_terminal'):continue
            label='峰值/终点' if b=='S' else '峰值' if r['selection']=='test_peak' else '终点'
            lines.append(f"| {b}第{r['total_epoch']}轮({label}) | {r['gmacs']:.4f} | {r['matrix_conv_gflops']:.4f} | {r['latency_mean_ms']:.2f}/{r['latency_median_ms']:.2f} | {r['latency_min_ms']:.2f}–{r['latency_max_ms']:.2f} | {r['peak_gib']:.4f} |")
    lines += ['', '![计算量与运行成本](figures/resource_comparison.png)','',
        '计算口径为实际执行的矩阵/卷积FLOPs=2MAC，包含12层融合注意力QK和AV，逐元素运算、softmax等排除并在profile中列出。完整窗口的重型输入由官方48个16帧块减少到24个，是真实骨干执行减少；没有做空间或深度跳算。修正前后的H65矩阵MAC相同。','',
        '时延为RTX4090独立作业中的同一完整窗口：video_test_0000007、起点0、768候选、batch1，输入已在GPU；BF16骨干/scout、FP32检测器，5次预热后保留20次同步计时。已核对完整/部分/短窗三种案例与官方参考的几何、输入形状、精度一致，且无标记为未解决的矩阵算子。官方与新模型使用不同作业/节点，时间对比有相应限制；这里不包含解码和NMS，也不是全测试集平均端到端时延。','',
        'B第55轮有一条253.66ms长样本，全部20条保留，故平均129.39ms高于中位122.47ms。即使看中位数，S/B选中模型仍慢于对应官方参考。旧H65-S的92.84ms均值也含长样本，其86.67ms中位数与本轮86.89ms接近；不能把均值下降解释为此次修正实现了算法加速。当前未做分模块计时，不能定量把额外时延归到某一模块。','',
        '所有16次完整预测、每阈值指标和验证记录见[中间测试总表](INTERMEDIATE_RESULTS.md)。选中模型与终点路径见[模型选择清单](SELECTED_CHECKPOINTS.json)，未舍入比较见[FINAL_COMPARISON.json](FINAL_COMPARISON.json)，验证结果见[FINAL_VALIDATION.json](FINAL_VALIDATION.json)，12000次训练更新与保留检查点见[训练完成记录](TRAINING_COMPLETION.md)。大型权重保留在独立远端实验目录；清单给出相对路径和state_dict_ema键。','',
        '本轮训练、全部测试、峰值选择、峰值/终点测量及汇总均完成。远端控制器完成24个阶段后正常退出，无需重启或补训练。']
    (EXP/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
