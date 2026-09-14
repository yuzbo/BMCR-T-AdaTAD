"""CPU-only public geometry. No decoded RGB or annotation reaches a selector."""
from dataclasses import asdict, dataclass
from bisect import bisect_left
import math


@dataclass(frozen=True)
class EpisodePublic:
    video_id: str
    window_index: int
    split: str
    video_path: str
    fps: float
    total_frames: int
    duration: float
    window_start_frame: int
    snippet_stride: int
    official_frame_ids: tuple
    official_valid: tuple
    transform: str = 'official_resize_short160_center160_rgb255'
    time_source: str = 'benchmark_frame_over_duration_fps'

    def __post_init__(self):
        ids, valid = self.official_frame_ids, self.official_valid
        n = sum(valid)
        if len(ids) != 768 or len(valid) != 768 or n < 1:
            raise ValueError('Raw-v1 preserves the THUMOS 768-position window')
        if tuple(valid) != tuple(i < n for i in range(768)):
            raise ValueError('Official validity must be a nonempty prefix')
        if self.fps <= 0 or min(ids) < 0 or max(ids) >= self.total_frames:
            raise ValueError('Invalid benchmark time/frame geometry')
        if tuple(sorted(ids)) != tuple(ids):
            raise ValueError('Official frames must be ordered')

    @property
    def bounds(self):
        return self.official_frame_ids[0], self.official_frame_ids[sum(self.official_valid)-1]

    @property
    def key(self):
        return f'{self.video_id}__{self.window_index:05d}'

    def record(self):
        return asdict(self)


@dataclass(frozen=True)
class PreviewTimeline:
    frame_ids: tuple
    times_seconds: tuple
    valid: tuple
    target_frames: tuple


@dataclass(frozen=True)
class RawProposal:
    frame_ids: tuple
    official: tuple
    preview_observed: tuple
    proposal_count_before_dedup: int
    domain: str  # bookkeeping only; never encoded in the Value descriptor


@dataclass(frozen=True)
class RawSelection:
    frame_ids: tuple
    valid: tuple

    def __post_init__(self):
        n = sum(self.valid)
        if len(self.frame_ids) != len(self.valid) or len(self.frame_ids) % 16 or n < 1:
            raise ValueError('Selection must occupy complete 16-observation packs')
        real = self.frame_ids[:n]
        if tuple(self.valid) != tuple(i < n for i in range(len(self.valid))):
            raise ValueError('Selection padding is a suffix')
        if tuple(sorted(set(real))) != tuple(real):
            raise ValueError('Real observations must be distinct and ordered')
        if any(x != real[-1] for x in self.frame_ids[n:]):
            raise ValueError('Padding must explicitly repeat the last frame')

    @property
    def support(self):
        return self.frame_ids[:sum(self.valid)]


@dataclass(frozen=True)
class TubeletMetadata:
    contributor_pairs: tuple
    contributor_valid: tuple
    centers_seconds: tuple
    spans_seconds: tuple
    pack_ids: tuple
    native_ids: tuple


def nearest_id(sorted_ids, target):
    p = bisect_left(sorted_ids, target)
    choices = sorted_ids[max(0, p-1):min(len(sorted_ids), p+1)]
    return min(choices, key=lambda x: (abs(x-target), x))


def nearest_raw(target, lower, upper):
    # A half-frame tie always chooses the earlier frame.
    return max(lower, min(upper, math.ceil(target-.5)))


def preview_timeline(episode, count=192):
    lo, hi = episode.bounds
    targets = tuple(lo+(i+.5)*(hi-lo)/count for i in range(count))
    ids = tuple(nearest_raw(t, lo, hi) for t in targets)
    return PreviewTimeline(ids, tuple(i/episode.fps for i in ids), (True,)*count, targets)


def proposals(episode, preview, domain, offsets=(-.375, -.125, .125, .375)):
    if domain not in ('O', 'R+'):
        raise ValueError('Only Official and Raw-Expand are registered in this pilot')
    official = set(episode.official_frame_ids[:sum(episode.official_valid)])
    lo, hi = episode.bounds
    width = (hi-lo)/len(preview.frame_ids)
    generated = [nearest_raw(t+d*width, lo, hi) for t in preview.target_frames for d in offsets]
    ids = tuple(sorted(official | set(generated))) if domain == 'R+' else tuple(sorted(official))
    observed = set(preview.frame_ids)
    return RawProposal(ids, tuple(i in official for i in ids), tuple(i in observed for i in ids),
                       len(generated) if domain == 'R+' else 0, domain)


def selection_from_support(support, budget=384):
    support = tuple(sorted(support))
    if not support or len(support) > budget:
        raise ValueError('Invalid real observation count')
    return RawSelection(support+(support[-1],)*(budget-len(support)),
                        tuple(i < len(support) for i in range(budget)))


def physical_uniform(episode, proposal, budget=384):
    # Shared valid capacity prevents short-window padding from changing the O/R budget.
    count = min(budget, len(set(episode.official_frame_ids[:sum(episode.official_valid)])))
    lo, hi = episode.bounds
    available = list(proposal.frame_ids)
    selected = []
    for i in range(count):
        frame = nearest_id(available, lo+(i+.5)*(hi-lo)/count)
        selected.append(frame)
        available.remove(frame)
    return selection_from_support(selected, budget)


def swap(selection, remove, insert, proposal):
    support = set(selection.support)
    if remove not in support or insert in support or insert not in proposal.frame_ids:
        raise ValueError('A swap must remove a member and insert a distinct legal frame')
    return selection_from_support((support-{remove}) | {insert}, len(selection.frame_ids))


def tubelets(selection, fps):
    pairs = tuple(zip(selection.frame_ids[::2], selection.frame_ids[1::2]))
    flags = tuple(zip(selection.valid[::2], selection.valid[1::2]))
    centers = tuple(sum(i*v for i,v in zip(pair, flag))/max(sum(flag), 1)/fps
                    for pair,flag in zip(pairs,flags))
    spans = tuple((pair[1]-pair[0])/fps if all(flag) else 0. for pair,flag in zip(pairs,flags))
    return TubeletMetadata(pairs, flags, centers, spans, tuple(i//8 for i in range(len(pairs))),
                           tuple(range(len(pairs))))


def repartition(before, after, fps):
    old, new = tubelets(before, fps), tubelets(after, fps)
    changed = [i for i,(a,b,av,bv) in enumerate(zip(old.contributor_pairs,new.contributor_pairs,
                 old.contributor_valid,new.contributor_valid)) if a != b or av != bv]
    return dict(n_changed_pairs=len(changed), n_changed_packs=len({i//8 for i in changed}),
                changed_pair_ids=changed, delta_pair_span_seconds=[b-a for a,b in zip(old.spans_seconds,new.spans_seconds)],
                before_pair_spans_seconds=old.spans_seconds, after_pair_spans_seconds=new.spans_seconds)


def geometric_swaps(episode, proposal, selection, limit=16, round_index=0):
    """One local exchange per equal physical cell; no GT, RGB or domain-specific rule."""
    unused = sorted(set(proposal.frame_ids)-set(selection.support))
    if not unused:
        return []
    lo, hi = episode.bounds
    pairs = []
    for i in range(limit):
        # Alternate within-cell targets across the four registered rounds.
        phase = (.5, .25, .75, .375)[round_index % 4]
        target = lo+(i+phase)*(hi-lo)/limit
        insert = nearest_id(unused, target)
        remove = nearest_id(selection.support, insert)
        if (remove, insert) not in pairs:
            pairs.append((remove, insert))
        unused.remove(insert)
        if not unused:
            break
    return pairs


def project_ground_truth(episode, annotations, class_map):
    """Supervision-only path, matching Atlas truncation in benchmark frame units."""
    lo, hi = episode.bounds
    segments, labels, boundaries, original = [], [], [], []
    for annotation in annotations:
        if annotation['label'] not in class_map:
            continue
        seconds = annotation['segment']
        a, b = (int(t*episode.fps) for t in seconds)
        if b <= lo or a >= hi or min(b, hi) <= max(a, lo):
            continue
        segments.append(((max(a,lo)-episode.window_start_frame)/episode.snippet_stride,
                         (min(b,hi)-episode.window_start_frame)/episode.snippet_stride))
        labels.append(class_map.index(annotation['label']))
        boundaries.append((a >= lo, b <= hi))
        original.append(dict(segment=seconds, label=labels[-1], boundary_valid=boundaries[-1]))
    return dict(gt_segments=segments, gt_labels=labels, gt_boundary_validity=boundaries, original=original)
