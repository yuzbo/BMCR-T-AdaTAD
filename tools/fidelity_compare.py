"""Keep test-selected peaks and fixed terminal results visibly separate."""
import json
from pathlib import Path
from fidelity_select import RUNS, save_selection

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'fidelity_20260911'
legacy=json.loads((ROOT/'phase2_20260910/comparison.json').read_text())['results']
rows=[]
for b in ('s','b'):
    selection=save_selection(RUNS,b)
    if not selection['all_candidates_evaluated']:
        raise RuntimeError('all predefined complete tests must finish before the final comparison')
    old=next(r for r in legacy if r['backbone']==b.upper() and r['method']=='h65')
    official=next(r for r in legacy if r['backbone']==b.upper() and r['method']=='official')
    for label,candidate in [('test_peak',selection['best']),('epoch60_terminal',selection['terminal'])]:
        profile=json.loads((Path(candidate['folder'])/'profile.json').read_text())
        rows.append(dict(backbone=b.upper(),selection=label,total_epochs=candidate['total_epochs'],
            metrics=candidate['metrics'],checkpoint=candidate['checkpoint'],
            delta_vs_legacy_terminal_pp=100*(candidate['average_mAP']-old['metrics']['average_mAP']),
            gmacs=profile['total_macs']/1e9,flops_fraction_of_official=profile['total_macs']/1e9/official['gmacs'],
            latency_mean_ms=profile['latency_mean_seconds']*1000,peak_gib=profile['peak_gib']))
payload=dict(results=rows,selection_policy='User-requested full-test peak across total epochs25..60 every5; terminal separately retained',
    course_judgment='A remaining gap alone does not isolate duration; historical30+60 is not the same20+40 course.',
    historical_s_terminal_percent=65.3857244379457,historical_s_total60_curve_percent_rounded=64.40,
    historical_s_observed_late_curve_peak_percent_rounded=65.65)
(out/'comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
lines=['# H65保真修正：测试峰值与60轮终点','',
    '全部200训练视频，每骨干20+40轮；联合阶段每5轮完整测试211视频。峰值是测试集选模结果，不能当成固定终点的同口径独立测试。','',
    '| 骨干 | 选模规则 | 总轮次 | 平均mAP | 相对旧终点变化(pp) | GMAC | FLOPs/官方 |',
    '|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['backbone']} | {r['selection']} | {r['total_epochs']} | {100*r['metrics']['average_mAP']:.4f}% | {r['delta_vs_legacy_terminal_pp']:+.4f} | {r['gmacs']:.4f} | {r['flops_fraction_of_official']:.4%} |")
lines+=['','历史S固定90轮终点65.385724%，历史总60轮曲线64.40%，已取得晚期曲线的峰值65.65%（后两者是日志舍入值）。课程和选模规则不同，不能把差值单独归因LR、边界监督或训练时长。',
        '','主计算/计时来自同一指定完整窗口；完整候选曲线及峰值指针见runs/*_best_test_checkpoint.json。']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(payload,indent=2))
