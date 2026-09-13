#!/usr/bin/env python3
"""Validate planned experiments and claim/figure references; no GPU or network."""
from pathlib import Path
import csv, json
ROOT=Path(__file__).resolve().parents[1]
p=json.loads((ROOT/'plans/experiments.json').read_text())
assert p['base_commit']=='239d098cd899936c35989fae6243c70259a85adb'
assert p['data']['train_videos']==200 and p['data']['test_videos']==211 and p['data']['test_windows']==792
exp={e['id']:e for e in p['experiments']}
assert len(exp)==len(p['experiments'])
for e in exp.values():
    assert e['state']=='planned' and e['metrics'] is None
    assert e['temporal_selection_unit']=='candidate_frame'
    assert not e['gpu_execution_authorized_in_this_package']
    assert set(e['depends_on']).issubset(exp)
visiting,done=set(),set()
def visit(key):
    if key in visiting: raise ValueError('Dependency cycle: '+key)
    if key in done:return
    visiting.add(key)
    for dep in exp[key]['depends_on']:visit(dep)
    visiting.remove(key);done.add(key)
for key in exp:visit(key)
with (ROOT/'plans/figures.csv').open(encoding='utf-8-sig') as f:figs=list(csv.DictReader(f))
for row in figs:
    assert set(row['experiments'].split(',')).issubset(exp), row['id']
with (ROOT/'plans/claims.csv').open(encoding='utf-8-sig') as f:claims=list(csv.DictReader(f))
figids={row['id'] for row in figs}
for row in claims:
    assert set(row['evidence_experiments'].split(',')).issubset(exp)
    assert set(row['figures'].split(',')).issubset(figids)
result=dict(success=True,experiments=len(exp),main_figure_groups=sum(x['placement']=='main' for x in figs),
            supplementary_figure_groups=sum(x['placement']=='supp' for x in figs),hypotheses=len(claims),
            new_experimental_scores_filled=0,remote_jobs_launched=0)
(ROOT/'validation/plan_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
