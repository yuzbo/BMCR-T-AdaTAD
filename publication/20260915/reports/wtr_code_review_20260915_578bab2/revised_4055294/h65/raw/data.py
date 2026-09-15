"""Official windows without RGB materialization; bounded reads happen after selection."""
from pathlib import Path
from dataclasses import asdict
import copy
import time
import numpy as np
import torch
from .contracts import EpisodePublic, project_ground_truth


class EpisodeData:
    def __init__(self, cfg, resources, ids, split='development'):
        from mmengine.dataset import Compose
        from h65.atlas.data import CharacterizationData
        self.source = CharacterizationData(cfg, resources, split, ids)
        transforms = cfg.dataset.test.pipeline
        stop = next(i for i,t in enumerate(transforms) if t.type == 'mmaction.DecordDecode')
        self.header_pipeline = Compose(copy.deepcopy(transforms[:stop]))

    def __len__(self):
        return len(self.source)

    def episode(self, ordinal):
        dataset = self.source.dataset
        index = self.source.indices[ordinal]
        name, info, _, centers = dataset.data_list[index]
        # Same public dictionary as ThumosSlidingDataset, before DecordDecode.
        header = self.header_pipeline(dict(video_name=name, data_path=dataset.data_path,
            window_size=dataset.window_size, feature_start_idx=int(centers[0]/dataset.snippet_stride),
            feature_end_idx=int(centers[-1]/dataset.snippet_stride), sample_stride=dataset.sample_stride,
            fps=info['frame']/info['duration'], snippet_stride=dataset.snippet_stride,
            window_start_frame=int(centers[0]), duration=info['duration'], offset_frames=dataset.offset_frames))
        result = EpisodePublic(name, index, self.source.split, header['filename'], float(header['fps']),
            int(header['total_frames']), float(info['duration']), int(centers[0]), int(dataset.snippet_stride),
            tuple(int(i) for i in header['frame_inds']), tuple(bool(v) for v in header['masks']))
        del header
        return result

    def supervision(self, episode):
        return project_ground_truth(episode, self.source.database[episode.video_id]['annotations'], self.source.class_map)

    def target_data(self, episode, device):
        supervision = self.supervision(episode)
        return dict(masks=torch.tensor([episode.official_valid], dtype=torch.bool, device=device),
            metas=[detector_meta(episode)],
            gt_segments=[torch.tensor(supervision['gt_segments'], device=device, dtype=torch.float32).reshape(-1,2)],
            gt_labels=[torch.tensor(supervision['gt_labels'], device=device, dtype=torch.long)])


def detector_meta(episode):
    return dict(video_name=episode.video_id, data_path=str(Path(episode.video_path).parent),
        fps=episode.fps, duration=episode.duration, snippet_stride=episode.snippet_stride,
        window_start_frame=episode.window_start_frame, window_size=768, offset_frames=0,
        frame_inds=np.asarray(episode.official_frame_ids), total_frames=episode.total_frames)


class RGBReader:
    """Decord CPU decode + the original deterministic spatial transform.

    Preview returns only 64px tensors. Heavy cache contains only acquired frames.
    Codec-internal GOP work is not observable through Decord and is not invented.
    """
    def __init__(self, episode):
        import decord
        from opentad.datasets.builder import PIPELINES
        self.episode = episode
        started = time.perf_counter()
        self.reader = decord.VideoReader(episode.video_path, num_threads=2)
        if len(self.reader) != episode.total_frames:
            raise ValueError('Video header differs from the frozen episode')
        self.resize = PIPELINES.build(dict(type='mmaction.Resize', scale=(-1,160)))
        self.crop = PIPELINES.build(dict(type='mmaction.CenterCrop', crop_size=160))
        self.cache = {}
        self.events = []
        self.header_ms = 1000*(time.perf_counter()-started)

    def read(self, ids, kind='heavy'):
        ids = tuple(int(i) for i in ids)
        unique = sorted(set(ids))
        if min(ids) < 0 or max(ids) >= self.episode.total_frames:
            raise ValueError('Frame acquisition outside the video')
        if kind not in ('preview','heavy','gate_official'):
            raise ValueError(kind)
        start = time.perf_counter()
        missing = [i for i in unique if kind != 'heavy' or i not in self.cache]
        fresh = {}
        decoded_pixels = 0
        if missing:
            raw = self.reader.get_batch(missing).asnumpy()
            decoded_pixels = int(np.prod(raw.shape[:-1]))
            for frame_id, frame in zip(missing, raw):
                item = dict(imgs=[frame], img_shape=frame.shape[:2], original_shape=frame.shape[:2])
                img = self.crop(self.resize(item))['imgs'][0]
                fresh[frame_id] = torch.from_numpy(img.copy()).permute(2,0,1).float()
            del raw
        if kind == 'heavy':
            self.cache.update(fresh)
            source = self.cache
        else:
            source = fresh
        rgb = torch.stack([source[i] for i in ids])
        if kind == 'preview':
            rgb = torch.nn.functional.interpolate(rgb, (64,64), mode='bilinear', align_corners=False)
        self.events.append(dict(kind=kind, requested_slots=len(ids), distinct_requested=len(unique),
            returned_decode_frames=len(missing), returned_decode_pixels=decoded_pixels,
            cache_hit_frames=len(unique)-len(missing), output_pixels=int(rgb.shape[0]*rgb.shape[-2]*rgb.shape[-1]),
            decode_transform_ms=1000*(time.perf_counter()-start),
            codec_internal_decoded_frames=None))
        return rgb.permute(1,0,2,3)[None,None].contiguous()

    def accounting(self):
        return dict(header_ms=self.header_ms, events=self.events, heavy_cache_frames=len(self.cache),
            acquisition_unit='individual frame', codec='Decord CPU full-resolution decode; GOP work unmeasured',
            scope='algorithm pilot; 64px preview tensor does not imply 64px codec decoding')
