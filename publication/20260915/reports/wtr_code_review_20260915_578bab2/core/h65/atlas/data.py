"""Deterministic official sliding windows with local GT and physical-time metadata."""
import copy
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch


class CharacterizationData:
    def __init__(self, native_cfg, resources, split, video_ids, one_window=False):
        from opentad.datasets.builder import build_dataset
        cfg = copy.deepcopy(native_cfg.dataset.test)
        ds = resources['datasets']['thumos']
        cfg.subset_name = 'training' if split == 'development' else 'validation'
        cfg.data_path = ds['train_videos'] if split == 'development' else ds['test_videos']
        cfg.test_mode = True  # Retain background/partial windows like official test.
        self.database = json.loads(Path(ds['annotations']).read_text())['database']
        expected = set(video_ids)
        cfg.block_list=[name for name in self.database if name not in expected]
        self.dataset = build_dataset(cfg)
        self.dataset.data_list = [x for x in self.dataset.data_list if x[0] in expected]
        if {x[0] for x in self.dataset.data_list} != expected:
            raise RuntimeError('Characterization cohort differs from its locked video IDs')
        self.class_map = self.dataset.class_map
        self.split = split
        self.video_ordinals={name:i for i,name in enumerate(sorted(ds['train_ids'] if split=='development' else ds['test_ids']))}
        self.by_video = defaultdict(list)
        for i, item in enumerate(self.dataset.data_list):
            self.by_video[item[0]].append(i)
        self.indices = []
        for name in sorted(expected):
            choices = self.by_video[name]
            # Deterministic middle official window, selected without scores/GT.
            self.indices.extend([choices[len(choices)//2]] if one_window else choices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, ordinal):
        from opentad.datasets.builder import collate
        index = self.indices[ordinal]
        item = self.dataset[index]
        data = collate([item])
        data['inputs'] = data['inputs'].float()
        meta = data['metas'][0]
        name = meta['video_name']
        valid = int(data['masks'].sum())
        fps = float(meta['fps'])
        start = float(meta['window_start_frame'])
        stride = float(meta['snippet_stride'])
        end = start + max(0, valid-1)*stride
        original, segments, labels = [], [], []
        for annotation in self.database[name]['annotations']:
            if annotation['label'] not in self.class_map:
                continue
            seconds = np.asarray(annotation['segment'], dtype=float)
            frames = np.asarray([int(value*fps) for value in seconds],dtype=float)
            label = self.class_map.index(annotation['label'])
            if frames[1] <= start or frames[0] >= end:
                continue
            truncated = np.clip(frames, start, end)
            if truncated[1] <= truncated[0]:
                continue
            segments.append(((truncated-start)/stride).tolist())
            labels.append(label)
            original.append(dict(segment=seconds.tolist(), label=label,
                                 fully_contained=bool(seconds[0]*fps>=start and seconds[1]*fps<=end)))
        data['gt_segments'] = [torch.tensor(segments, dtype=torch.float32).reshape(-1, 2)]
        data['gt_labels'] = [torch.tensor(labels, dtype=torch.long)]
        meta['characterization_gt'] = original
        meta['characterization_split'] = self.split
        meta['characterization_window_index'] = index
        meta['characterization_video_ordinal'] = self.video_ordinals[name]
        return data

    def ground_truth(self, path):
        from h65.paper.runtime import json_write
        database = {name: copy.deepcopy(self.database[name]) for name in sorted(self.by_video)}
        for value in database.values():
            value['annotations'] = [a for a in value['annotations'] if a['label'] in self.class_map]
        json_write(path, dict(database=database))


def window_metadata(data):
    meta = data['metas'][0]
    return dict(video_id=meta['video_name'], split=meta['characterization_split'],
        video_ordinal=meta['characterization_video_ordinal'],
        window_index=meta['characterization_window_index'], fps=float(meta['fps']),
        snippet_stride=float(meta['snippet_stride']), start_frame=float(meta['window_start_frame']),
        valid_candidates=int(data['masks'].sum()), input_candidates=int(data['masks'].shape[-1]),
        frame_indices=np.asarray(meta['frame_inds']).tolist(),gt=meta['characterization_gt'])


def position_metadata(meta, candidate):
    seconds = float(np.interp(candidate,np.arange(len(meta['frame_indices'])),meta['frame_indices']))/meta['fps']
    inside = [a for a in meta['gt'] if a['segment'][0]<=seconds<=a['segment'][1]]
    if not meta['gt']:
        return dict(time_seconds=seconds, region='background', boundary_distance=None,
                    action_duration=None, fully_contained=None)
    action = min(inside, key=lambda a: a['segment'][1]-a['segment'][0]) if inside else min(
        meta['gt'],key=lambda a:min(abs(seconds-a['segment'][0]),abs(seconds-a['segment'][1]))/(a['segment'][1]-a['segment'][0]))
    start, end = action['segment']
    duration = end-start
    distance = min(seconds-start, end-seconds)/duration if inside else None
    region = 'start' if abs(seconds-start)/duration<=.1 else 'end' if abs(end-seconds)/duration<=.1 else 'interior' if inside else 'background'
    if len(inside)>1:region='overlap'
    return dict(time_seconds=seconds, region=region, boundary_distance=distance,
                action_duration=duration if region!='background' else None,fully_contained=action['fully_contained'],
                action_start=start,action_end=end,within_action=bool(inside),overlapping_actions=len(inside))


def time_actions(valid, span=32, layers=None):
    result = []
    for start in range(0, valid, span):
        stop = min(start+span, 768)
        result.append(dict(id=f'T_{start:03}_{stop:03}', axis='T', native_range=[start//2, (stop+1)//2],
                           candidate_center=(start+min(start+span,valid)-1)/2,
                           **({} if layers is None else dict(layers=layers))))
    return result


def spatial_actions(tile=1, layers=None):
    return [dict(id=f'S_{y:02}_{x:02}', axis='S', patch_box=[y,min(y+tile,10),x,min(x+tile,10)],
                 **({} if layers is None else dict(layers=layers)))
            for y in range(0,10,tile) for x in range(0,10,tile)]


def depth_actions(interior=True):
    return [dict(id=f'D_{layer:02}', axis='D', layer=layer)
            for layer in range(1 if interior else 0, 11 if interior else 12)]
