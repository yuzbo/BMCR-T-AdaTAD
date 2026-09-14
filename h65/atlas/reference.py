"""Frozen official EMA reference; FP32 task scoring and operator-measured inference."""
import copy
from contextlib import contextmanager
import time
import numpy as np
import torch
from .compute import DenseInterventions, execution_shape_key


@contextmanager
def deterministic_fp32():
    old = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
           torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.no_grad(), torch.autocast('cuda', enabled=False), torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False):
            yield
    finally:
        (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
         torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) = old


class FrozenReference:
    def __init__(self, backbone, resources, device='cuda'):
        from h65.paper.native_adatad import build_native_config, NativeAdaTAD
        cfg = dict(recipe='native_adatad_uniform_v1', id=f'characterization_{backbone}',
                   dataset='thumos', backbone=backbone, frames=768, seed=42)
        self.cfg = build_native_config(cfg, resources)
        self.model = NativeAdaTAD(self.cfg, cfg, resources).to(device).float().eval().requires_grad_(False)
        self.detector = self.model.detector
        self.vit = self.detector.backbone.model.backbone
        self.interventions = DenseInterventions(self.vit)
        self.costs = {}
        self.profile_records = []
        self.executions = 0
        self.scoring_gflops = 0.
        self.proxy_gflops = 0.

    def prediction_head(self, features, data):
        detector = self.detector
        x, valid = detector.pad_data(features, data['masks'])
        if detector.with_projection:
            x, valid = detector.projection(x, valid)
        if detector.with_neck:
            x, valid = detector.neck(x, valid)
        return detector.rpn_head.forward_test(x, valid)

    def loss_head(self, features, data):
        detector = self.detector
        before = detector.rpn_head.loss_normalizer.detach().clone()
        x, valid = detector.pad_data(features, data['masks'])
        if detector.with_projection:
            x, valid = detector.projection(x, valid)
        if detector.with_neck:
            x, valid = detector.neck(x, valid)
        loss = detector.rpn_head.forward_train(x, valid, gt_segments=data['gt_segments'], gt_labels=data['gt_labels'])
        detector.rpn_head.loss_normalizer = before
        values = np.asarray([float(loss['cls_loss']), float(loss['reg_loss'])], dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite paired characterization loss')
        return values

    def execute(self, data, attention=None, ffn=None, force_profile=False, capture_proxies=False):
        from h65.frame.measure import matrix_counter
        from h65.paper.native_adatad import _profile_hooks
        support = int(data['masks'].sum())
        key = (support, 'dense') if attention is None else (support, execution_shape_key(attention, ffn))
        profile = force_profile or key not in self.costs
        torch.cuda.synchronize()
        start = time.perf_counter()
        captured = {}
        capture = self.vit.blocks[0].attn.register_forward_pre_hook(
            lambda module,args: captured.update(x=args[0].detach())) if capture_proxies else None
        with deterministic_fp32(), self.interventions.apply(attention, ffn):
            if profile:
                counter = matrix_counter()
                handles = _profile_hooks(self.model, counter)
                try:
                    with counter:
                        features = self.detector.backbone(data['inputs'])
                        prediction = self.prediction_head(features, data)
                finally:
                    for handle in handles:
                        handle.remove()
                if counter.unresolved_matrix_ops():
                    raise RuntimeError(str(counter.unresolved_matrix_ops()))
                cost = 2*sum(counter.macs.values())/1e9
                if key in self.costs and abs(self.costs[key]-cost)>1e-8:
                    raise RuntimeError('Equal execution shapes produced different operator counts')
                self.costs[key] = cost
                self.profile_records.append(dict(gflops=cost, macs=dict(counter.macs),
                    operations=dict(counter.operations), attention_shapes=counter.fused_attention,
                    source='actual operator dispatch; complete backbone/TIA/projection/neck/head'))
            else:
                features = self.detector.backbone(data['inputs'])
                prediction = self.prediction_head(features, data)
                cost = self.costs[key]
            torch.cuda.synchronize()
            model_ms = 1000*(time.perf_counter()-start)
            # The extra head scoring pass is diagnostic overhead, not charged to
            # the inference-only curve. Its loss normalizer is always restored.
            losses = self.loss_head(features.float(), data)
            attention_value = None
            if capture is not None:
                capture.remove()
                with matrix_counter() as proxy_counter:
                    attention_value = self.attention_proxy(captured['x'])
                if proxy_counter.unresolved_matrix_ops():
                    raise RuntimeError(str(proxy_counter.unresolved_matrix_ops()))
                self.proxy_gflops += 2*sum(proxy_counter.macs.values())/1e9
        self.executions += 1
        self.scoring_gflops += cost
        proposals, scores = prediction
        return dict(losses=losses, loss=float(losses.sum()), gflops=cost, model_ms=model_ms,
                    proposals=proposals[0].detach(), scores=scores[0].detach(),
                    features=features.detach(), prediction=prediction, attention=attention_value)

    def attention_proxy(self, x):
        """Incoming first-block attention, measured only as a diagnostic proxy."""
        attn = self.vit.blocks[0].attn
        b,n,c = x.shape
        values = []
        for chunk in x.split(4):
            bias = torch.cat((attn.q_bias,torch.zeros_like(attn.v_bias),attn.v_bias))
            qkv = torch.nn.functional.linear(chunk,attn.qkv.weight,bias)
            qkv = qkv.reshape(len(chunk),n,3,attn.num_heads,-1).permute(2,0,3,1,4)
            q,k = qkv[0],qkv[1]
            incoming = q.new_zeros((len(chunk),n))
            for query in q.split(128,dim=2):
                probabilities = ((query*(q.shape[-1]**-.5))@k.transpose(-2,-1)).softmax(-1)
                incoming += probabilities.mean(1).sum(1)/n
            values.append(incoming)
        temporal = torch.cat(values).reshape(1,384,10,10).mean((-1,-2))
        return torch.nn.functional.interpolate(temporal[:,None],size=768,mode='linear',align_corners=False)[0,0]

    def postprocess(self, result, data, class_map):
        post = copy.deepcopy(self.cfg.post_processing)
        post.sliding_window = True
        return self.detector.post_processing(result['prediction'], data['metas'], post, class_map)

    def accounting(self):
        return dict(forward_executions=self.executions, inference_equivalent_gflops=self.scoring_gflops,
                    extra_loss_head_forward_count=self.executions,
                    diagnostic_attention_proxy_gflops=self.proxy_gflops,
                    precision='FP32, TF32 disabled, deterministic cuDNN and math SDPA',
                    inference_scope='RGB resident; complete matrix/conv operators; decode/NMS excluded',
                    diagnostic_scope='Oracle/value acquisition and loss scoring are not deployment inference',
                    profiles=self.profile_records, reference=self.model.provenance)

    def shape_cost(self, dense_gflops, attention, ffn):
        """Exact matrix-shape ledger for searching budgets, not a keep ratio.

        Every reported allocation is subsequently operator-profiled. All TIA
        and head operations stay in the measured dense constant.
        """
        c = self.vit.embed_dims
        n = 8*attention.shape[-2]*attention.shape[-1]
        packs = attention.numel()//(len(attention)*n)
        full = packs*(12*n*c*c+2*n*n*c)
        total = 0.
        for am, fm in zip(attention, ffn):
            q = am.reshape(packs,n).sum(-1).double()
            active = int((q>0).sum())
            selected = float(q.sum())
            attn = (2*active*n+2*selected)*c*c+2*n*selected*c
            mlp = 8*int(fm.sum())*c*c
            total += attn+mlp-full
        return dense_gflops+2*total/1e9


def prediction_jsd(reference, changed):
    p = reference['scores'].float().clamp(1e-7, 1-1e-7)
    q = changed['scores'].float().clamp(1e-7, 1-1e-7)
    if p.shape != q.shape:
        raise RuntimeError('Input-preserving interventions changed the raw prediction support')
    m = (p+q)/2
    def kl(a,b):
        return a*(a/b).log()+(1-a)*((1-a)/(1-b)).log()
    return float((.5*kl(p,m)+.5*kl(q,m)).mean())


def proxies(result, center):
    midpoint = result['proposals'].float().mean(-1)
    nearest = (midpoint-center).abs().argsort()[:16]
    scores = result['scores'][nearest].float().clamp(1e-7,1-1e-7)
    probability = scores.max(-1).values
    entropy = -(scores*scores.log()+(1-scores)*(1-scores).log()).mean()
    index = int(np.clip(round(center),0,result['features'].shape[-1]-1))
    out = dict(actionness=float(probability.mean()), confidence=float(probability.max()),
               entropy=float(entropy), feature_norm=float(result['features'][0,:,index].float().norm()))
    if result.get('attention') is not None:
        out['attention_score'] = float(result['attention'][index])
    return out
