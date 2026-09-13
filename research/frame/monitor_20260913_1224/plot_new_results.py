"""Reproduce the dated result figure; use points for small GT-loss differences."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent
records = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))['records']

def score(run, epoch, state='ema'):
    row = next(r for r in records if r.get('run_id') == run and
               r.get('epoch') == epoch and r.get('checkpoint_state') == state)
    return 100 * row['metrics']['average_mAP']

fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
ax = axes[0]
for bb in ('s', 'b'):
    points = sorted((r['epoch'], 100*r['metrics']['average_mAP']) for r in records
                    if r.get('run_id') == 'BMCR80_'+bb and r.get('checkpoint_state') == 'ema')
    ax.plot(*zip(*points), 'o-', label='BMCR-'+bb.upper(), markersize=4)
    x, y = points[-1]
    ax.annotate(f'{y:.2f}%', (x, y), xytext=(5, 0), textcoords='offset points', fontsize=9)
ax.axvspan(65, 80, color='.85', alpha=.3)
ax.text(71.5, 55, 'tests\npending', ha='center', color='.4', fontsize=9)
ax.set(xlim=(23, 82), ylim=(52, 69), xlabel='Total epoch (80-epoch course)',
       ylabel='Full-test average mAP (%)', title='BMCR: S reaches epoch 65')
ax.legend(fontsize=8); ax.grid(alpha=.15)

ax = axes[1]
values = [score('R04_feature_only_s', 5), score('R03_cross_s', 5)]
ax.scatter([0, 1], values, s=65)
for i, value in enumerate(values):
    ax.annotate(f'{value:.4f}%', (i, value), xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9)
ref = score('R01_interpolate_s', 0)
ax.axhline(ref, color='C1', ls='--', label=f'R01 interpolation {ref:.3f}%')
ax.set(xticks=[0, 1], xticklabels=['Feature KD only', 'Feature KD + direct GT'],
       xlim=(-.45, 1.45), ylim=(63.65, 64.35), ylabel='Full-test average mAP (%)',
       title='Cross-S EMA5: matched GT ablation')
ax.legend(loc='lower left', fontsize=8); ax.grid(axis='y', alpha=.15)

ax = axes[2]
xs = [5, 10, 15, 20]; ys = [score('R03_cross_s', e) for e in xs]
online = score('R03_cross_s', 20, 'learned')
ax.plot(xs, ys, 'o-', label='EMA')
ax.scatter([20], [online], marker='x', s=65, color='C1', label='Online weights', zorder=5)
ax.annotate(f'EMA {ys[-1]:.4f}%', (20, ys[-1]), xytext=(-90, 16), textcoords='offset points', fontsize=9)
ax.annotate(f'Online {online:.4f}%', (20, online), xytext=(-105, -30), textcoords='offset points', fontsize=9)
ax.set(xlim=(3, 23), ylim=(63.95, 64.8), xticks=xs, xlabel='New-module epoch',
       ylabel='Full-test average mAP (%)', title='Cross-S: terminal EMA vs online')
ax.legend(loc='lower right', fontsize=8); ax.grid(alpha=.15)
fig.suptitle('THUMOS14 full 211-video / 792-window results | 2026-09-13 12:28 CST', fontsize=12)
for ext in ('png', 'svg', 'pdf'):
    fig.savefig(out / ('new_results.'+ext), dpi=300, bbox_inches='tight')
plt.close(fig)
