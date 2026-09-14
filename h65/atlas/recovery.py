"""Same frozen V2 support and heavy tensors for four recovery comparisons."""
import copy
from pathlib import Path
import numpy as np
import torch
from .reference import deterministic_fp32


def build_probe(reference,resources):
    from h65.paper.model import PaperModel
    from h65.paper.runtime import build_config
    b=reference.backbone
    payload=torch.load(resources['atlas_light'][b],map_location='cpu')
    initial=Path(resources['atlas_light'][b]).with_name(f'v2_{b}_initial_source.pth')
    if not initial.exists():
        teacher=torch.load(resources['teachers'][f'thumos:{b}'],map_location='cpu')['state_dict_ema']
        state={k.removeprefix('module.'):v for k,v in teacher.items()}
        state.update(payload['backbone_overrides']);state.update(payload['scout_initial'])
        torch.save(dict(state_dict_ema=state,source=payload['original_encoder_checkpoint']),initial)
    res=copy.deepcopy(resources)
    res['encoders']={f'thumos:{b}':dict(checkpoint=str(initial),kind='task',
        variant=payload['metadata']['encoder']['variant'],scout_checkpoint=str(initial))}
    config=copy.deepcopy(payload['metadata']['config'])
    # Every decoder tensor is restored from V2 EMA; an R03 file is not needed again.
    res['recovery_initialization']={}
    cfg=build_config(config,res)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        model=PaperModel(cfg,config,res,with_teacher=False)
    model.load_learned(payload['ema'])
    model=model.cuda().float().eval().requires_grad_(False)
    return model,cfg,dict(original_checkpoint=payload['original_checkpoint'],state='ema',
        source_revision=payload['metadata']['source_revision'],reconstruction=payload['reconstruction'],
        initial_backbone_overrides=len(payload['backbone_overrides']),learned_tensors=len(payload['ema']))


def endpoint(prediction,data):
    boxes,scores=prediction;boxes=boxes[0];scores=scores[0]
    errors=[];targets=len(data['gt_labels'][0])
    for target,label in zip(data['gt_segments'][0],data['gt_labels'][0]):
        intersection=(torch.minimum(boxes[:,1],target[1])-torch.maximum(boxes[:,0],target[0])).clamp_min(0)
        union=(boxes[:,1]-boxes[:,0]).clamp_min(0)+(target[1]-target[0])-intersection
        iou=intersection/union.clamp_min(1e-8)
        index=(iou*scores[:,int(label)]).argmax()
        if float(iou[index])>=.1 and float(scores[index,int(label)])>=.05:
            errors.append((boxes[index]-target).abs())
    scale=data['metas'][0]['snippet_stride']/data['metas'][0]['fps']
    return dict(gt=targets,matched=len(errors),sum_abs_endpoint_error_seconds=(torch.stack(errors).sum(0)*scale).cpu().tolist() if errors else [0.,0.],
        matching='Same-GT class; score*IoU association; IoU>=.1, score>=.05; missing matches reported')


def recovery_window(reference,data,class_map,meta,resources):
    from h65.frame.measure import matrix_counter
    from h65.frame.geometry import make_anchors,make_queries
    from h65.paper.geometry import interpolate_anchors,scout_context
    if not hasattr(reference,'paper_probe'):
        reference.paper_probe=build_probe(reference,resources)
    model,cfg,provenance=reference.paper_probe
    with deterministic_fp32():
        with matrix_counter() as shared_counter:
            preview=model.encoder.preview(data['inputs'],data['masks'])
            selection=model.encoder.select(preview,data['masks'],384,'anchor',data['metas'])
            plan=model.plan(1)
            native,levels,trace=model.encoder.encode(data['inputs'],selection,plan,capture=True)
            anchors=make_anchors(native,selection,data['masks'],data['metas'],trace)
            queries=make_queries(data['masks'],data['metas'],selection)
            context=scout_context(preview,data['masks'],False)
        if shared_counter.unresolved_matrix_ops():raise RuntimeError(str(shared_counter.unresolved_matrix_ops()))
        shared=2*sum(shared_counter.macs.values())/1e9
        full_selection=model.encoder.select(preview,data['masks'],768,'uniform',data['metas'])
        with matrix_counter() as target_counter:
            target_native,_,target_trace=model.encoder.encode(data['inputs'],full_selection,model.plan(0),capture=False)
            target_anchors=make_anchors(target_native,full_selection,data['masks'],data['metas'],target_trace)
            target=interpolate_anchors(target_anchors,queries).transpose(1,2)
        valid=queries.valid[0]
        gaps=(queries.centers[0,:,None]-anchors.centers[0,anchors.valid[0]][None]).abs().amin(1)
        rows=[]
        original_multidepth=model.decoder.multidepth
        for variant in ('packed_naive','physical_interpolation','cross_without_multidepth','cross_multidepth'):
            with matrix_counter() as counter:
                if variant=='packed_naive':
                    source=anchors.features[0,anchors.valid[0]].transpose(0,1)[None]
                    recovered=source.new_zeros((1,source.shape[1],len(valid)))
                    recovered[:,:,valid]=torch.nn.functional.interpolate(source,size=int(valid.sum()),mode='linear',align_corners=False)
                elif variant=='physical_interpolation':
                    recovered=interpolate_anchors(anchors,queries).transpose(1,2)
                else:
                    model.decoder.multidepth=variant=='cross_multidepth'
                    recovered=model.decoder(anchors,queries,context,levels)
                prediction=model.readout.predictions(recovered.float(),data['masks'],data['metas'])
            model.decoder.multidepth=original_multidepth
            if counter.unresolved_matrix_ops():raise RuntimeError(str(counter.unresolved_matrix_ops()))
            norm=model.readout.detector.rpn_head.loss_normalizer.detach().clone()
            loss=model.readout.components(model.readout.loss(recovered.float(),data))
            model.readout.detector.rpn_head.loss_normalizer=norm
            post=copy.deepcopy(cfg.post_processing);post.sliding_window=True
            pred=model.readout.post_processing(prediction,data['metas'],post,class_map)
            nmse=(recovered-target).square().sum(1)/target.square().sum(1).clamp_min(1e-8)
            cosine=torch.nn.functional.cosine_similarity(recovered,target,dim=1)
            cost=shared+2*sum(counter.macs.values())/1e9
            rows.append(dict(method=variant,gflops=cost,predictions=pred,loss_cls_reg=loss.cpu().tolist(),
                nmse=nmse[0,valid].cpu().tolist(),cosine=cosine[0,valid].cpu().tolist(),
                endpoint=endpoint(prediction,data)))
    return dict(meta=meta,checkpoint=provenance,variants=rows,
        selected_candidates=selection.indices.cpu().tolist(),selected_valid=selection.valid.cpu().tolist(),
        query_gap_seconds=gaps[valid].cpu().tolist(),shared_gflops=shared,
        dense_target_gflops=2*sum(target_counter.macs.values())/1e9,
        recovery_claim='Same-checkpoint component intervention; Cross-only is not separately retrained',
        fidelity_target='Same frozen V2 encoder with full RGB/full D/S, physically interpolated onto original queries',
        graph_enabled=False)
