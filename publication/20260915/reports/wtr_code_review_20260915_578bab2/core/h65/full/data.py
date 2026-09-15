"""Preserve upstream crops while retaining the historical H65 endpoint-validity label."""
import numpy as np
from opentad.datasets.builder import PIPELINES
from opentad.datasets.transforms.end_to_end import LoadFrames


@PIPELINES.register_module()
class LoadFramesWithBoundaryValidity(LoadFrames):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.method != 'random_trunc':
            raise ValueError('boundary-validity extension is for the training random_trunc pipeline')

    def random_trunc(self, feats, trunc_len, gt_segments, gt_labels, offset=0, max_num_trials=200):
        # Delegate all random choices, crop acceptance, GT clipping and label
        # ordering to the unmodified upstream implementation.
        cropped, segments, labels = super().random_trunc(
            feats, trunc_len, gt_segments, gt_labels, offset, max_num_trials)
        if len(cropped) == len(feats):
            validity = np.ones((len(segments), 2), dtype=np.bool_)
        else:
            # LoadFrames supplies the strictly increasing frame-index array.
            # Its surviving first value identifies the crop's original index.
            start = int(np.searchsorted(feats, cropped[0]))
            end = start + len(cropped)
            window = np.repeat(np.array([[start, end]], dtype=np.float32), len(gt_segments), axis=0)
            left = np.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = np.minimum(window[:, 1] + offset, gt_segments[:, 1])
            overlap = np.clip(right-left, 0, None)
            keep = overlap / np.abs(gt_segments[:, 1]-gt_segments[:, 0]) >= self.trunc_thresh
            validity = np.stack((np.isclose(left[keep], gt_segments[keep, 0], rtol=0., atol=1e-6),
                                 np.isclose(right[keep], gt_segments[keep, 1], rtol=0., atol=1e-6)), axis=1)
            if len(validity) != len(segments):
                raise RuntimeError('boundary validity must preserve upstream GT ordering/count')
        self.boundary_validity = validity.astype(np.bool_)
        return cropped, segments, labels

    def __call__(self, results):
        result = super().__call__(results)
        result['gt_boundary_validity'] = self.boundary_validity
        return result
