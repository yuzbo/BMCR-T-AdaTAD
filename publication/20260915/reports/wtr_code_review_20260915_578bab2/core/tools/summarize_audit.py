"""Clarify effective sample counts and assignment sensitivity of completed audits."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sign_counts(values):
    return dict(positive=sum(x > 1e-8 for x in values), negative=sum(x < -1e-8 for x in values),
                near_zero=sum(abs(x) <= 1e-8 for x in values), zero_tolerance=1e-8)


def sign_disagreement(first, second):
    valid = [(a,b) for a,b in zip(first, second) if abs(a)>1e-8 and abs(b)>1e-8]
    flips = sum(a*b < 0 for a,b in valid)
    return dict(nonzero_pairs=len(valid), opposite_signs=flips,
                fraction=flips/len(valid) if valid else None, zero_tolerance=1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['s','b'], required=True)
    args = parser.parse_args()
    root = Path(os.environ.get('H65_RUNS_DIR', ROOT/'runs')).expanduser().resolve()/f'{args.backbone}_audit'
    rows = json.loads((root/'observations.json').read_text())
    completed = json.loads((root/'completed.json').read_text())
    fixed_cls = [r['target'][0] for r in rows]
    rematched_cls = [r['cls_rematched_delta'] for r in rows]
    rematched_loc = [r['target'][1] for r in rows]
    fixed_loc = [r['loc_fixed_delta'] for r in rows]
    review = dict(windows_requested=completed['windows'], windows_with_feasible_swaps=len({r['window_index'] for r in rows}),
        videos_with_feasible_swaps=len({r['video'] for r in rows}), measured_swaps=len(rows),
        fit_observed_videos=len({r['video'] for r in rows if r['partition']=='fit'}),
        holdout_observed_videos=len({r['video'] for r in rows if r['partition']=='holdout'}),
        fit_swaps=completed['fit_swaps'], holdout_swaps=completed['holdout_swaps'],
        classification_signs=sign_counts(fixed_cls), localization_signs=sign_counts(rematched_loc),
        classification_assignment_sign_disagreement=sign_disagreement(fixed_cls, rematched_cls),
        localization_matching_sign_disagreement=sign_disagreement(fixed_loc, rematched_loc),
        gt_instances_counted_per_swap=sum(r['gt_count'] for r in rows),
        matched_gt_counted_per_swap=sum(r['matched_before'] for r in rows),
        count_note='GT counts repeat instances across swaps/windows and are not independent unique GT instances.',
        attribution_holdout_spearman=completed['attribution_spearman'],
        diagnostic_ridge_holdout_spearman=completed['diagnostic_ridge_spearman'],
        inference_limit='Training-video diagnostic only; no test-set or final model-performance result.')
    (root/'review.json').write_text(json.dumps(review, indent=2)+'\n')
    print(json.dumps(review, indent=2))


if __name__ == '__main__':
    main()
