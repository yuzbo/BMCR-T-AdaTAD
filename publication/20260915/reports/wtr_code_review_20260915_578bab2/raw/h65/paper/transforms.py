import numpy as np
from opentad.datasets.builder import PIPELINES
from opentad.datasets.transforms.end_to_end import LoadFrames


@PIPELINES.register_module()
class PaperResizeFrames(LoadFrames):
    def __call__(self, results):
        value = super().__call__(results)
        if 'gt_segments' in value:
            value['gt_boundary_validity'] = np.ones((len(value['gt_segments']),2), dtype=np.bool_)
        return value
