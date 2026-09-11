"""Publish verified intermediate accuracy, retaining the full preset test curve."""
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from fidelity_select import EPOCHS, select_best

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'fidelity_20260911'
RUNS = Path(os.environ.get('H65_RUNS_DIR', EXP / 'runs')).resolve()
PUBLIC = EXP / 'evaluations'


def main():
    for backbone in ('s', 'b'):
        for epoch in EPOCHS:
            name = f'{backbone}_h65_test_epoch_{epoch:02}'
            source, destination = RUNS / name, PUBLIC / name
            files = ('metrics.json', 'verification.json', 'result_detection.json.gz')
            if not all((source / name).exists() for name in files):
                continue
            verification = json.loads((source / 'verification.json').read_text())
            if not all(verification['checks'].values()):
                raise ValueError(f'{source}: result verification failed')
            record = json.loads((source / 'metrics.json').read_text())
            record['initialization']['checkpoint'] = f'runs/{backbone}_h65/epoch_{epoch-20:02}.pth'
            record['publication_note'] = 'Only checkpoint path made relative; metrics and predictions unchanged.'
            destination.mkdir(parents=True, exist_ok=True)
            (destination / 'metrics.json').write_text(json.dumps(record, indent=2)+'\n')
            shutil.copy2(source / 'verification.json', destination / 'verification.json')
            if not (destination / 'result_detection.json.gz').exists():
                shutil.copy2(source / 'result_detection.json.gz', destination / 'result_detection.json.gz')

    rows, best = [], {}
    for backbone in ('s', 'b'):
        selection = select_best(PUBLIC, backbone)
        for candidate in selection['candidates']:
            rows.append(dict(backbone=backbone.upper(), total_epoch=candidate['total_epochs'],
                             metrics=candidate['metrics'], checkpoint=candidate['checkpoint']))
        winner = selection['best']
        best[backbone.upper()] = (dict(total_epoch=winner['total_epochs'], average_mAP=winner['average_mAP'],
                                      evaluated=selection['evaluated'], expected=len(EPOCHS),
                                      final_selection=selection['all_candidates_evaluated']) if winner else None)
    report = dict(updated_at=datetime.now().astimezone().isoformat(timespec='seconds'), seed=3407,
                  expected_total_epochs=list(EPOCHS), evaluated=len(rows), expected=2*len(EPOCHS),
                  full60_epoch_training_complete=True, test_based_selection=True, results=rows, best_so_far=best)
    (EXP / 'INTERMEDIATE_RESULTS.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    lines = ['# H65完整中间测试结果', '', f"已测{len(rows)}/16个预设候选。S/B的60轮训练均已完成；每个表列结果覆盖全部211视频、792窗口，使用对应轮次EMA。", '',
             '最终模型按总epoch25/30/35/40/45/50/55/60的平均mAP峰值选择，同分取较早轮次；第60轮另行报告。尚未测完全部候选时，当前最高值仅为已测候选中的最高值。', '',
             '| 骨干 | 总轮次 | 平均mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 | 记录 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        metric = row['metrics']
        scores = [metric['average_mAP']] + [metric[f'mAP@{t:.1f}'] for t in (.3, .4, .5, .6, .7)]
        path = f"evaluations/{row['backbone'].lower()}_h65_test_epoch_{row['total_epoch']:02}"
        lines.append(f"| {row['backbone']} | {row['total_epoch']} | " + ' | '.join(f'{100*x:.4f}%' for x in scores) +
                     f' | [指标]({path}/metrics.json) · [预测]({path}/result_detection.json.gz) |')
    lines += ['', '本表是测试集选模过程，不能称为独立未见测试性能。中间点不能直接当作第60轮终点；计算量与时延在峰值和终点的独立测量完成后报告。']
    (EXP / 'INTERMEDIATE_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(evaluated=report['evaluated'], best_so_far=best), indent=2))


if __name__ == '__main__':
    main()
