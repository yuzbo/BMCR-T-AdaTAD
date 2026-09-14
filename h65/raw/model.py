"""Frozen V2-S bridge: arbitrary physical frames -> 192 anchors -> 384 Cross -> 768 head."""
from types import SimpleNamespace
import copy
import time
import torch
from .contracts import tubelets
from .data import detector_meta


def identity_selection(selection, device):
    from h65.transport import Selection
    k = len(selection.frame_ids)
    ids = torch.arange(k, device=device)[None]
    valid = torch.tensor([selection.valid], dtype=torch.bool, device=device)
    rates = valid.float()
    return Selection(ids, ids.float(), valid, rates/rates.sum(), rates)


def geometry(episode, selection, native, trace):
    from h65.frame.contracts import AnchorBatch, QueryBatch
    device = native.device
    tm = tubelets(selection, episode.fps)
    pairs = torch.tensor([tm.contributor_pairs], device=device)
    flags = torch.tensor([tm.contributor_valid], device=device)
    times = pairs.float()  # Existing trained recovery metadata uses raw-frame units.
    count = flags.sum(-1)
    centers = (times*flags).sum(-1)/count.clamp_min(1)
    spans = torch.where(count == 2, times[...,1]-times[...,0], 0.)
    a = len(tm.pack_ids)
    anchors = AnchorBatch(native, pairs, times, flags, centers, spans, count > 0,
        torch.tensor([tm.pack_ids],device=device), torch.arange(a,device=device)[None]%8,
        trace.get('last_heavy_depth',native.new_full((1,a),12)),
        trace.get('spatial_quality',native.new_ones((1,a))))
    original = torch.tensor([episode.official_frame_ids],device=device,dtype=torch.float32)
    masks = torch.tensor([episode.official_valid],device=device)
    valid_pairs = masks.reshape(1,384,2)
    query_centers = (original.reshape(1,384,2)*valid_pairs).sum(-1)/valid_pairs.sum(-1).clamp_min(1)
    support = set(selection.support)
    member = torch.tensor([[bool(v) and i in support for i,v in zip(episode.official_frame_ids,episode.official_valid)]],device=device)
    queries = QueryBatch(original,query_centers,valid_pairs.any(-1),masks,member)
    return anchors, queries


def interpolate_preview(values, frame_ids, target_frames):
    """Physical interpolation; repeated preview frames share their mean state."""
    times = torch.as_tensor(frame_ids,device=values.device,dtype=torch.float32)
    unique, inverse, counts = torch.unique_consecutive(times,return_inverse=True,return_counts=True)
    hidden = values.new_zeros((len(unique), values.shape[-1])).index_add(0,inverse,values)/counts[:,None]
    target = torch.as_tensor(target_frames,device=values.device,dtype=torch.float32)
    if len(unique) == 1:
        return hidden.expand(len(target),-1)
    right = torch.searchsorted(unique.contiguous(),target.contiguous()).clamp(1,len(unique)-1)
    left = right-1
    f = ((target-unique[left])/(unique[right]-unique[left])).clamp(0,1)
    return hidden[left]*(1-f[:,None])+hidden[right]*f[:,None]


class FrozenRawModel:
    def __init__(self, resources):
        from h65.atlas.recovery import build_probe
        self.model, self.cfg, self.provenance = build_probe(SimpleNamespace(backbone='s'), resources)
        self.device = next(self.model.parameters()).device
        self.plan = dict(id='K384_D100_S100',frames=384,depth=1.,space=1.,use_light=True,full_kv=True)
        self.profile = None

    def preview(self, reader, timeline):
        from h65.atlas.reference import deterministic_fp32
        from h65.frame.measure import matrix_counter
        rgb = reader.read(timeline.frame_ids, 'preview').to(self.device)
        mask = torch.tensor([timeline.valid],device=self.device)
        with deterministic_fp32(), matrix_counter() as counter:
            result = self.model.encoder.preview(rgb,mask)
        if counter.unresolved_matrix_ops():
            raise RuntimeError(str(counter.unresolved_matrix_ops()))
        return result, 2*sum(counter.macs.values())/1e9

    def execute(self, episode, selection, reader, timeline, preview, target_data=None, context_override=None, rgb_override=None):
        from h65.atlas.reference import deterministic_fp32
        from h65.frame.measure import matrix_counter
        rgb = reader.read(selection.frame_ids).to(self.device) if rgb_override is None else rgb_override.to(self.device)
        identity = identity_selection(selection,self.device)
        start = time.perf_counter()
        torch.cuda.synchronize()
        with deterministic_fp32(), matrix_counter() as counter:
            native, levels, trace = self.model.encoder.encode(rgb,identity,self.plan,capture=True)
            anchors, queries = geometry(episode,selection,native,trace)
            context = (interpolate_preview(preview['hidden'][0],timeline.frame_ids,queries.centers[0])[None]
                       if context_override is None else context_override)
            recovered = self.model.decoder(anchors,queries,context,levels).float()
            if native.shape[1] != 192 or recovered.shape[-1] != 384:
                raise RuntimeError('Native/Cross axes changed')
            prediction = self.model.readout.predictions(recovered,queries.candidate_mask,[detector_meta(episode)])
        torch.cuda.synchronize()
        if counter.unresolved_matrix_ops():
            raise RuntimeError(str(counter.unresolved_matrix_ops()))
        result = dict(prediction=prediction,recovered=recovered,trace=trace,
            gflops=2*sum(counter.macs.values())/1e9, model_ms=1000*(time.perf_counter()-start))
        if target_data is not None:
            with deterministic_fp32():
                normalizer = self.model.readout.detector.rpn_head.loss_normalizer.detach().clone()
                try:
                    result['loss'] = self.model.readout.components(self.model.readout.loss(recovered,target_data)).cpu()
                finally:
                    self.model.readout.detector.rpn_head.loss_normalizer = normalizer
                if not torch.isfinite(result['loss']).all():
                    raise RuntimeError('Nonfinite task gain label')
        return result

    def postprocess(self, result, episode, class_map):
        post = copy.deepcopy(self.cfg.post_processing)
        post.sliding_window = True
        return self.model.readout.post_processing(result['prediction'],[detector_meta(episode)],post,class_map)
