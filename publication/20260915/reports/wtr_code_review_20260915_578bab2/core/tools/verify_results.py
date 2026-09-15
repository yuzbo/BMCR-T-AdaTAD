"""Validate the published evidence with Python's standard library; no GPU or dataset needed."""
import gzip
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'phase2_20260910'


def main():
    rows = json.loads((RESULTS / 'comparison.json').read_text(encoding='utf-8'))['results']
    assert {(r['backbone'], r['method']) for r in rows} == {(b, m) for b in ('S', 'B') for m in ('official', 'h65', 'bmcr')}
    video_ids, checked = None, []
    for row in rows:
        run = RESULTS / 'runs' / f"{row['backbone'].lower()}_{row['method']}_test"
        metrics = json.loads((run / 'metrics.json').read_text())
        profile = json.loads((run / 'profile.json').read_text())
        assert metrics['test_videos'] == 211 and metrics['test_windows'] == 792
        assert metrics['metrics'] == row['metrics']
        mean = sum(row['metrics'][f'mAP@{t}'] for t in ('0.3', '0.4', '0.5', '0.6', '0.7')) / 5
        assert math.isclose(mean, row['metrics']['average_mAP'], abs_tol=1e-12)
        assert profile['primary_case'] == 'full' and profile['window_index'] == 2
        assert not profile['unresolved_matrix_ops']
        assert math.isclose(profile['total_macs'] / 1e9, row['gmacs'], abs_tol=1e-9)
        assert len(profile['latency_seconds']) == 20
        with gzip.open(run / 'result_detection.json.gz', 'rt', encoding='utf-8') as stream:
            predictions = json.load(stream)['results']
        assert len(predictions) == 211 and all(predictions.values())
        if video_ids is None:
            video_ids = set(predictions)
        assert set(predictions) == video_ids
        checked.append(dict(backbone=row['backbone'], method=row['method'], videos=len(predictions),
                            predictions=sum(map(len, predictions.values())), average_mAP=mean))
    training = []
    for name in ('s_warm', 'b_warm', 's_h65', 's_bmcr', 'b_h65', 'b_bmcr'):
        expected = 2000 if name.endswith('warm') else 4000
        with (RESULTS / 'runs' / name / 'train.jsonl').open() as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        assert [r['successful_updates'] for r in records] == list(range(1, expected + 1))
        assert all(math.isfinite(r['losses']['cost']) and math.isfinite(r['grad_norm']) for r in records)
        training.append(dict(stage=name, successful_updates=expected))
    print(json.dumps(dict(passed=True, full_tests=checked, training=training,
        scope='Published metric/profile/prediction/log consistency only; does not independently recompute mAP from annotations or rerun models.'), indent=2))


if __name__ == '__main__':
    main()
